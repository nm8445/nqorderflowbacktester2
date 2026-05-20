"""
Slippage stress test — IS/OOS with asymmetric fill assumptions.

Models the realistic live-trading slippage profile you're seeing:
  - Entry slippage: UP TO `ENTRY_SLIP_TICKS` against you (market orders, latency)
  - Stop-loss slippage: `SL_SLIP_TICKS` against you (stop-market slip)
  - Target slippage: 0 (limit orders fill exactly or don't fill)

Runs through all cached days with the asymmetric band zone and Option B latch.
Reports PF, DD, Sharpe, WR IS vs OOS so you can see how much edge survives
worst-case fills.

Usage:
  python scripts/vwap_reaction_strat_backtest/slippage_stress_isoos.py
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
TICK_SIZE   = 0.25

# Slippage model (user-tunable)
ENTRY_SLIP_TICKS = 6   # worst case market-order entry slip
SL_SLIP_TICKS    = 2   # stop-market exit slip (capped)
TP_SLIP_TICKS    = 0   # limit-target fills exactly

ENTRY_CUTOFF = "16:00"
FORCE_CLOSE  = "16:58"
BASE_SL, BASE_TP = 0.50, 1.90
TREND_SL, TREND_TP = 1.00, 1.00
TREND_STD, TREND_LB, TREND_ADX_MIN = 1, 14, 30.0

START_DATE = "2025-03-13"
END_DATE   = "2026-04-13"
IS_END     = "2025-09-25"
OOS_START  = "2025-09-26"


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
        if atr is None or atr <= 0:
            continue
        adx = get_adx_at_time(ct_et, adx_lookup)
        if not passes_adx_filter(adx):
            continue
        ep, d = s["entry_price"], s["direction"]
        # Apply entry slippage (against us)
        slip = ENTRY_SLIP_TICKS * TICK_SIZE
        ep_slipped = ep + slip if d == "long" else ep - slip
        if d == "long":
            sl, tp = ep - atr * BASE_SL, ep + atr * BASE_TP
        else:
            sl, tp = ep + atr * BASE_SL, ep - atr * BASE_TP
        out.append({"source":"base","bar_index":s["bar_index"],"confirm_time":s["confirm_time"],
                    "entry_price":ep_slipped,"direction":d,"sl":sl,"tp":tp})
    return out


def prepare_trend(signals, adx_lookup):
    out = []
    for s in signals:
        ct_et = to_et(s["confirm_time"])
        hm = ct_et.strftime("%H:%M")
        if "16:00" <= hm < "19:10" or (LUNCH_START <= hm < LUNCH_END):
            continue
        atr = s["atr"]
        if atr is None or atr <= 0:
            continue
        adx = get_adx_at_time(ct_et, adx_lookup)
        if np.isnan(adx) or adx < TREND_ADX_MIN:
            continue
        ep, d = s["entry_price"], s["direction"]
        slip = ENTRY_SLIP_TICKS * TICK_SIZE
        ep_slipped = ep + slip if d == "long" else ep - slip
        if d == "long":
            sl, tp = ep - atr * TREND_SL, ep + atr * TREND_TP
        else:
            sl, tp = ep + atr * TREND_SL, ep - atr * TREND_TP
        out.append({"source":"trend","bar_index":s["bar_index"],"confirm_time":s["confirm_time"],
                    "entry_price":ep_slipped,"direction":d,"sl":sl,"tp":tp})
    return out


def simulate(bars, sigs):
    sigs = sorted(sigs, key=lambda x: x["confirm_time"])
    trades, last_exit = [], None
    for sig in sigs:
        if last_exit is not None and sig["confirm_time"] <= last_exit:
            continue
        d, ep, sl, tp = sig["direction"], sig["entry_price"], sig["sl"], sig["tp"]
        cbi = sig["bar_index"] + 1
        ex_price = ex_time = ex_reason = None
        for j in range(cbi + 1, len(bars)):
            b = bars[j]
            if not b.closed:
                continue
            bet = to_et(b.close_time)
            if bet.strftime("%H:%M") >= FORCE_CLOSE:
                ex_price, ex_time, ex_reason = b.close, b.close_time, "eod"; break
            if d == "long":
                if b.low <= sl:
                    # Stop slipped against us by SL_SLIP_TICKS
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
        pnl_pts = (ex_price - ep) if d == "long" else (ep - ex_price)
        date_str = str(to_et(sig["confirm_time"]).date())
        trades.append({"source":sig["source"],"date":date_str,
                       "pnl":pnl_pts*POINT_VALUE,"reason":ex_reason})
        last_exit = ex_time
    return trades


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
    print("=" * 95)
    print(f"  SLIPPAGE STRESS — entry={ENTRY_SLIP_TICKS}t  SL={SL_SLIP_TICKS}t  TP={TP_SLIP_TICKS}t")
    print(f"  IS: {START_DATE}-{IS_END}  |  OOS: {OOS_START}-{END_DATE}")
    print("=" * 95)
    adx_lookup = build_adx_lookup()

    is_end_d  = datetime.strptime(IS_END, "%Y-%m-%d").date()
    start_d   = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end_d     = datetime.strptime(END_DATE, "%Y-%m-%d").date()

    is_trades, oos_trades = [], []
    for cache_file in sorted(VWAP_CACHE_DIR.glob("*.pkl")):
        date_str = cache_file.stem
        try:
            fd = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if fd < start_d or fd > end_d:
            continue
        signal_file = SIGNAL_CACHE_DIR / f"{date_str}.pkl"
        if not signal_file.exists():
            continue
        with open(signal_file, "rb") as f:
            bars = pickle.load(f).get("bars") or []
        if not bars:
            continue
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
        day_trades = simulate(bars, sigs)
        bucket = is_trades if fd <= is_end_d else oos_trades
        bucket.extend(day_trades)

    metrics(is_trades, "IS ")
    metrics(oos_trades, "OOS")
    metrics(is_trades + oos_trades, "ALL")

    # Also: per-trade slippage cost summary
    all_trades = is_trades + oos_trades
    if all_trades:
        per_trade_drag = (ENTRY_SLIP_TICKS * TICK_SIZE) * POINT_VALUE
        # SL exits also pay SL slip on top of entry
        n_sl = sum(1 for t in all_trades if t["reason"] == "sl")
        sl_drag_total = n_sl * (SL_SLIP_TICKS * TICK_SIZE) * POINT_VALUE
        entry_drag_total = len(all_trades) * per_trade_drag
        print()
        print(f"  Slippage cost breakdown:")
        print(f"    Entry drag: {len(all_trades)} tr × {ENTRY_SLIP_TICKS}t = ${entry_drag_total:,.0f}")
        print(f"    SL drag:    {n_sl} sl × {SL_SLIP_TICKS}t = ${sl_drag_total:,.0f}")
        print(f"    Total cost imposed by slippage: ${entry_drag_total + sl_drag_total:,.0f}")


if __name__ == "__main__":
    main()
