"""
Revisit-entry backtest — IS/OOS.

User's proposed order logic:
  After confirmation bar closes, check current price (modeled as next-bar open):
  - If price moved >= 5 ticks AWAY from confirm close → place conditional order
    at confirm close. Fills only if price revisits confirm close.
    * LONG, price above close: LIMIT buy at close (wait for pullback)
    * LONG, price below close: STOP-LIMIT buy at close (wait for reclaim)
    * SHORT, price below close: LIMIT sell at close (wait for bounce)
    * SHORT, price above close: STOP-LIMIT sell at close (wait for rejection)
  - If price within 5 ticks of close → market entry (small latency slippage)

Fills at confirm close = zero entry slippage on the conditional path.
But some trades are skipped if price never revisits before SL/TP hits.

This script reports:
  - how many trades are taken vs skipped
  - PnL / PF / DD / Sharpe with the conditional model
  - comparison vs pure-market baseline

Usage:
  python scripts/vwap_reaction_strat_backtest/revisit_entry_isoos.py
"""
import pickle, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from adx_filter_common import (
    ET, DATA_DIR, build_adx_lookup, get_adx_at_time, passes_adx_filter,
    LUNCH_START, LUNCH_END,
)
from trending_vwap_atr_grid import (
    load_5min_bars as load_5min_bars_trending,
    precompute_day_state, detect_signals as detect_trend_signals,
)

VWAP_CACHE_DIR = DATA_DIR / "vwap_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
VWAP_PRICE_CACHE_DIR = DATA_DIR / "vwap_cache"
TIMEBARS_DIR = DATA_DIR / "timebars_5min"

POINT_VALUE = 20.0
TICK_SIZE = 0.25

# ------------ tunables ------------
REVISIT_THRESHOLD_TICKS = 5   # if |next_open - confirm_close| >= this → conditional order
MARKET_FALLBACK_SLIP_TICKS = 2  # within threshold → market entry; model this slip
SL_SLIP_TICKS = 2              # stop-loss exit slip (unchanged)
TP_SLIP_TICKS = 0              # limit target fills exact
# ----------------------------------

ENTRY_CUTOFF = "16:00"
FORCE_CLOSE  = "16:58"
BASE_SL, BASE_TP = 0.50, 1.90
TREND_SL, TREND_TP = 1.00, 1.00
TREND_STD, TREND_LB, TREND_ADX_MIN = 1, 14, 30.0

START_DATE = "2025-03-13"
END_DATE   = "2026-04-13"
IS_END     = "2025-09-25"


def to_et(ct):
    return ct.tz_convert(ET) if hasattr(ct, "tz_convert") else pd.Timestamp(ct, tz="UTC").tz_convert(ET)


def prepare_base(signals, adx_lookup):
    out = []
    for s in signals:
        ct_et = to_et(s["confirm_time"])
        hm = ct_et.strftime("%H:%M")
        if "16:00" <= hm < "19:10" or (LUNCH_START <= hm < LUNCH_END):
            continue
        atr = s["atr"]
        if atr is None or atr <= 0: continue
        adx = get_adx_at_time(ct_et, adx_lookup)
        if not passes_adx_filter(adx): continue
        ep, d = s["entry_price"], s["direction"]
        if d == "long":
            sl, tp = ep - atr*BASE_SL, ep + atr*BASE_TP
        else:
            sl, tp = ep + atr*BASE_SL, ep - atr*BASE_TP
        out.append({"source":"base","bar_index":s["bar_index"],"confirm_time":s["confirm_time"],
                    "entry_price":ep,"direction":d,"sl":sl,"tp":tp})
    return out


def prepare_trend(signals, adx_lookup):
    out = []
    for s in signals:
        ct_et = to_et(s["confirm_time"])
        hm = ct_et.strftime("%H:%M")
        if "16:00" <= hm < "19:10" or (LUNCH_START <= hm < LUNCH_END):
            continue
        atr = s["atr"]
        if atr is None or atr <= 0: continue
        adx = get_adx_at_time(ct_et, adx_lookup)
        if np.isnan(adx) or adx < TREND_ADX_MIN: continue
        ep, d = s["entry_price"], s["direction"]
        if d == "long":
            sl, tp = ep - atr*TREND_SL, ep + atr*TREND_TP
        else:
            sl, tp = ep + atr*TREND_SL, ep - atr*TREND_TP
        out.append({"source":"trend","bar_index":s["bar_index"],"confirm_time":s["confirm_time"],
                    "entry_price":ep,"direction":d,"sl":sl,"tp":tp})
    return out


def simulate_revisit(bars, sigs):
    """Conditional-entry simulation with revisit logic. Returns (trades, stats)."""
    sigs = sorted(sigs, key=lambda x: x["confirm_time"])
    trades = []
    stats = {"n_sig": 0, "n_market": 0, "n_conditional_filled": 0,
             "n_missed_sl": 0, "n_missed_tp": 0, "n_missed_eod": 0}
    last_exit = None
    threshold = REVISIT_THRESHOLD_TICKS * TICK_SIZE

    for sig in sigs:
        stats["n_sig"] += 1
        if last_exit is not None and sig["confirm_time"] <= last_exit:
            continue

        d = sig["direction"]
        confirm_close = sig["entry_price"]
        sl = sig["sl"]; tp = sig["tp"]
        cbi = sig["bar_index"] + 1   # confirm bar index
        next_idx = cbi + 1
        if next_idx >= len(bars):
            continue

        next_bar = bars[next_idx]
        observed = next_bar.open
        delta = observed - confirm_close  # + if price above confirm close

        # Decide path
        if abs(delta) < threshold:
            # Market path
            slip = MARKET_FALLBACK_SLIP_TICKS * TICK_SIZE
            entry_price = observed + slip if d == "long" else observed - slip
            entry_bar_idx = next_idx
            stats["n_market"] += 1
        else:
            # Conditional path — wait for revisit at confirm_close
            filled_idx = None
            missed_reason = None
            for j in range(next_idx, len(bars)):
                b = bars[j]
                if not b.closed: continue
                bet = to_et(b.close_time)
                if bet.strftime("%H:%M") >= FORCE_CLOSE:
                    missed_reason = "eod"; break
                # Check revisit first (same-bar priority: if bar contains confirm_close, fill)
                if b.low <= confirm_close <= b.high:
                    filled_idx = j
                    break
                # Check SL / TP before fill
                if d == "long":
                    if b.low <= sl: missed_reason = "sl"; break
                    if b.high >= tp: missed_reason = "tp"; break
                else:
                    if b.high >= sl: missed_reason = "sl"; break
                    if b.low <= tp: missed_reason = "tp"; break
            if filled_idx is None:
                if missed_reason == "sl": stats["n_missed_sl"] += 1
                elif missed_reason == "tp": stats["n_missed_tp"] += 1
                else: stats["n_missed_eod"] += 1
                continue
            entry_price = confirm_close  # zero slip
            entry_bar_idx = filled_idx
            stats["n_conditional_filled"] += 1

        # Simulate exit from entry_bar_idx onward
        ex_price = ex_time = ex_reason = None
        # The entry bar itself can also trigger SL/TP if it prints those levels
        # (conservative: scan from entry_bar_idx + 1 to stay consistent with prior sims)
        for j in range(entry_bar_idx + 1, len(bars)):
            b = bars[j]
            if not b.closed: continue
            bet = to_et(b.close_time)
            if bet.strftime("%H:%M") >= FORCE_CLOSE:
                ex_price, ex_time, ex_reason = b.close, b.close_time, "eod"; break
            if d == "long":
                if b.low <= sl:
                    ex_price = sl - SL_SLIP_TICKS * TICK_SIZE
                    ex_time, ex_reason = b.close_time, "sl"; break
                if b.high >= tp:
                    ex_price = tp - TP_SLIP_TICKS * TICK_SIZE
                    ex_time, ex_reason = b.close_time, "tp"; break
            else:
                if b.high >= sl:
                    ex_price = sl + SL_SLIP_TICKS * TICK_SIZE
                    ex_time, ex_reason = b.close_time, "sl"; break
                if b.low <= tp:
                    ex_price = tp + TP_SLIP_TICKS * TICK_SIZE
                    ex_time, ex_reason = b.close_time, "tp"; break
        if ex_price is None:
            ex_price, ex_time, ex_reason = bars[-1].close, bars[-1].close_time, "eod"

        pnl_pts = (ex_price - entry_price) if d == "long" else (entry_price - ex_price)
        date_str = str(to_et(sig["confirm_time"]).date())
        trades.append({"source":sig["source"],"date":date_str,
                       "pnl":pnl_pts*POINT_VALUE,"reason":ex_reason})
        last_exit = ex_time

    return trades, stats


def metrics(trades, label):
    if not trades:
        print(f"  {label}: no trades"); return
    df = pd.DataFrame(trades)
    n = len(df); pnl = df["pnl"].sum()
    wins = df[df["pnl"] > 0]; losses = df[df["pnl"] < 0]
    wr = len(wins)/n*100
    gp = wins["pnl"].sum() if len(wins) else 0
    gl = abs(losses["pnl"].sum()) if len(losses) else 1
    pf = gp/gl if gl>0 else float("inf")
    cum = df["pnl"].cumsum()
    dd = float((cum - cum.cummax()).min())
    daily = df.groupby("date")["pnl"].sum()
    sharpe = (daily.mean() / daily.std()) * np.sqrt(252) if len(daily) > 1 and daily.std() > 0 else 0.0
    nb = (df["source"]=="base").sum(); nt = (df["source"]=="trend").sum()
    print(f"  {label}: {n} tr ({nb}b/{nt}t)  PnL ${pnl:+,.0f}  WR {wr:.1f}%  PF {pf:.2f}  MaxDD ${dd:+,.0f}  Sharpe {sharpe:.2f}")


def main():
    print("=" * 100)
    print(f"  REVISIT-ENTRY STRESS — threshold={REVISIT_THRESHOLD_TICKS}t  "
          f"market_fallback_slip={MARKET_FALLBACK_SLIP_TICKS}t  SL_slip={SL_SLIP_TICKS}t")
    print(f"  IS: {START_DATE}-{IS_END}  |  OOS: then-{END_DATE}")
    print("=" * 100)
    adx_lookup = build_adx_lookup()

    is_end_d  = datetime.strptime(IS_END, "%Y-%m-%d").date()
    start_d   = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end_d     = datetime.strptime(END_DATE, "%Y-%m-%d").date()

    is_trades, oos_trades = [], []
    is_stats = {"n_sig":0,"n_market":0,"n_conditional_filled":0,"n_missed_sl":0,"n_missed_tp":0,"n_missed_eod":0}
    oos_stats = dict(is_stats)

    for cache_file in sorted(VWAP_CACHE_DIR.glob("*.pkl")):
        date_str = cache_file.stem
        try:
            fd = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if fd < start_d or fd > end_d:
            continue
        signal_file = SIGNAL_CACHE_DIR / f"{date_str}.pkl"
        if not signal_file.exists(): continue
        with open(signal_file, "rb") as f:
            bars = pickle.load(f).get("bars") or []
        if not bars: continue
        with open(cache_file, "rb") as f:
            base_raw = pickle.load(f).get("signals") or []
        trend_raw = []
        vwap_price_file = VWAP_PRICE_CACHE_DIR / f"{date_str}.pkl"
        tb_file = TIMEBARS_DIR / f"timebars_5min_{date_str.replace('-','_')}.pkl"
        if vwap_price_file.exists() and tb_file.exists():
            with open(vwap_price_file, "rb") as f:
                vwap_df = pickle.load(f)
            bars_5min = load_5min_bars_trending(date_str)
            if bars_5min is not None:
                ds = precompute_day_state(bars_5min, vwap_df, TREND_LB, TREND_STD)
                trend_raw = detect_trend_signals(bars, ds)

        sigs = prepare_base(base_raw, adx_lookup) + prepare_trend(trend_raw, adx_lookup)
        day_trades, day_stats = simulate_revisit(bars, sigs)

        bucket_tr = is_trades if fd <= is_end_d else oos_trades
        bucket_st = is_stats if fd <= is_end_d else oos_stats
        bucket_tr.extend(day_trades)
        for k in day_stats: bucket_st[k] += day_stats[k]

    metrics(is_trades, "IS ")
    metrics(oos_trades, "OOS")
    metrics(is_trades + oos_trades, "ALL")

    total_stats = {k: is_stats[k] + oos_stats[k] for k in is_stats}
    print()
    print("  Signal disposition (combined IS+OOS):")
    print(f"    Total signals:           {total_stats['n_sig']}")
    print(f"    Market entries:          {total_stats['n_market']}")
    print(f"    Conditional filled:      {total_stats['n_conditional_filled']}")
    print(f"    Missed — SL pre-fill:    {total_stats['n_missed_sl']}")
    print(f"    Missed — TP pre-fill:    {total_stats['n_missed_tp']}")
    print(f"    Missed — EOD no revisit: {total_stats['n_missed_eod']}")
    n_taken = total_stats['n_market'] + total_stats['n_conditional_filled']
    n_missed = total_stats['n_missed_sl'] + total_stats['n_missed_tp'] + total_stats['n_missed_eod']
    n_locked_out = total_stats['n_sig'] - n_taken - n_missed  # position-lock skips
    if total_stats['n_sig']:
        print(f"    Miss rate:               {100*n_missed/total_stats['n_sig']:.1f}%  "
              f"(of {total_stats['n_sig']} signals, {n_missed} missed, {n_taken} taken, {n_locked_out} locked out)")


if __name__ == "__main__":
    main()
