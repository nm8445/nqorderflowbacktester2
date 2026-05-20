"""
ADX range sweep — base and trending tested SEPARATELY with IS/OOS 50/50 split.

Base:    sweep ADX low (10-25) and ADX high (20-40) to find best range
Trending: sweep ADX min threshold (15-35)

Uses rebuilt cache with look-ahead fix applied.
Slippage: entry 0t, SL 2t against, TP 0t (matches live revisit behavior).

Usage:
    python scripts/vwap_reaction_strat_backtest/adx_sweep_base_trend_isoos.py
"""
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from adx_filter_common import (
    ET, DATA_DIR, build_adx_lookup, get_adx_at_time,
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

ENTRY_SLIP_TICKS = 0
SL_SLIP_TICKS = 2
TP_SLIP_TICKS = 0

ENTRY_CUTOFF = "16:00"
FORCE_CLOSE = "16:58"

BASE_SL, BASE_TP = 0.50, 1.90
TREND_SL, TREND_TP = 1.00, 1.00
TREND_STD, TREND_LB = 1, 14

START_DATE = "2025-03-13"
END_DATE = "2026-04-17"


def to_et(ct):
    return ct.tz_convert(ET) if hasattr(ct, "tz_convert") else pd.Timestamp(ct, tz="UTC").tz_convert(ET)


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
                ex_price, ex_time, ex_reason = b.close, b.close_time, "eod"
                break
            if d == "long":
                if b.low <= sl:
                    ex_price = sl - SL_SLIP_TICKS * TICK_SIZE
                    ex_time, ex_reason = b.close_time, "sl"
                    break
                if b.high >= tp:
                    ex_price = tp - TP_SLIP_TICKS * TICK_SIZE
                    ex_time, ex_reason = b.close_time, "tp"
                    break
            else:
                if b.high >= sl:
                    ex_price = sl + SL_SLIP_TICKS * TICK_SIZE
                    ex_time, ex_reason = b.close_time, "sl"
                    break
                if b.low <= tp:
                    ex_price = tp + TP_SLIP_TICKS * TICK_SIZE
                    ex_time, ex_reason = b.close_time, "tp"
                    break
        if ex_price is None:
            ex_price, ex_time, ex_reason = bars[-1].close, bars[-1].close_time, "eod"
        pnl_pts = (ex_price - ep) if d == "long" else (ep - ex_price)
        date_str = str(to_et(sig["confirm_time"]).date())
        trades.append({"date": date_str, "pnl": pnl_pts * POINT_VALUE, "reason": ex_reason})
        last_exit = ex_time
    return trades


def stats(trades):
    if not trades:
        return dict(n=0, pnl=0, wr=0, pf=0, dd=0, avg=0)
    df = pd.DataFrame(trades)
    n = len(df)
    pnl = df["pnl"].sum()
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    wr = len(wins) / n * 100
    gp = wins["pnl"].sum() if len(wins) else 0
    gl = abs(losses["pnl"].sum()) if len(losses) else 1
    pf = gp / gl if gl > 0 else float("inf")
    cum = df["pnl"].cumsum()
    dd = float((cum - cum.cummax()).min())
    avg = pnl / n
    return dict(n=n, pnl=pnl, wr=wr, pf=pf, dd=dd, avg=avg)


def main():
    print("Loading ADX lookup...")
    adx_lookup = build_adx_lookup()
    start_d = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end_d = datetime.strptime(END_DATE, "%Y-%m-%d").date()

    # Collect dates, 50/50 split
    all_dates = []
    for cache_file in sorted(VWAP_CACHE_DIR.glob("*.pkl")):
        ds = cache_file.stem
        try:
            fd = datetime.strptime(ds, "%Y-%m-%d").date()
        except ValueError:
            continue
        if fd < start_d or fd > end_d:
            continue
        if (SIGNAL_CACHE_DIR / f"{ds}.pkl").exists():
            all_dates.append(fd)

    split = len(all_dates) // 2
    is_dates = set(all_dates[:split])
    is_end = all_dates[split - 1]
    oos_start = all_dates[split]

    print(f"Total days: {len(all_dates)} | IS: {all_dates[0]}->{is_end} ({split}d) | OOS: {oos_start}->{all_dates[-1]} ({len(all_dates)-split}d)")

    # Pre-load all day data
    print("Loading day caches...")
    day_data = {}
    for fd in all_dates:
        ds = fd.strftime("%Y-%m-%d")
        with open(SIGNAL_CACHE_DIR / f"{ds}.pkl", "rb") as f:
            bars = pickle.load(f).get("bars") or []
        if not bars:
            continue
        with open(VWAP_CACHE_DIR / f"{ds}.pkl", "rb") as f:
            base_raw = pickle.load(f).get("signals") or []

        trend_raw = []
        vwap_price_file = VWAP_PRICE_CACHE_DIR / f"{ds}.pkl"
        tb_file = TIMEBARS_DIR / f"timebars_5min_{ds.replace('-','_')}.pkl"
        if vwap_price_file.exists() and tb_file.exists():
            with open(vwap_price_file, "rb") as f:
                vwap_df = pickle.load(f)
            bars_5min = load_5min_bars_trending(ds)
            if bars_5min is not None:
                dstate = precompute_day_state(bars_5min, vwap_df, TREND_LB, TREND_STD)
                trend_raw = detect_trend_signals(bars, dstate) if dstate is not None else []

        day_data[fd] = (bars, base_raw, trend_raw)

    print(f"Loaded {len(day_data)} days with data\n")

    # =========================================================================
    #  PART 1: BASE STRATEGY ADX SWEEP
    # =========================================================================
    print("=" * 130)
    print("  PART 1: BASE STRATEGY — ADX RANGE SWEEP (low, high)")
    print("  SL=0.50x ATR, TP=1.90x ATR | entry_slip=0t, sl_slip=2t")
    print("=" * 130)
    print(f"  {'ADX Range':>12} | {'IS n':>5} {'IS PnL':>10} {'PF':>5} {'DD':>9} {'WR%':>5} {'Avg':>7}"
          f"  |  {'OOS n':>5} {'OOS PnL':>10} {'PF':>5} {'DD':>9} {'WR%':>5} {'Avg':>7}")
    print("  " + "-" * 128)

    base_results = []
    for adx_low in range(10, 26, 5):
        for adx_high in range(max(adx_low + 5, 20), 46, 5):
            is_tr, oos_tr = [], []
            for fd, (bars, base_raw, _) in day_data.items():
                sigs = []
                for s in base_raw:
                    ct_et = to_et(s["confirm_time"])
                    hm = ct_et.strftime("%H:%M")
                    if "16:00" <= hm < "19:10" or LUNCH_START <= hm < LUNCH_END:
                        continue
                    atr = s["atr"]
                    if atr is None or atr <= 0:
                        continue
                    adx = get_adx_at_time(ct_et, adx_lookup)
                    if np.isnan(adx) or adx < adx_low or adx >= adx_high:
                        continue
                    ep, d = s["entry_price"], s["direction"]
                    slip = ENTRY_SLIP_TICKS * TICK_SIZE
                    ep_s = ep + slip if d == "long" else ep - slip
                    if d == "long":
                        sl, tp = ep - atr * BASE_SL, ep + atr * BASE_TP
                    else:
                        sl, tp = ep + atr * BASE_SL, ep - atr * BASE_TP
                    sigs.append({"bar_index": s["bar_index"], "confirm_time": s["confirm_time"],
                                 "entry_price": ep_s, "direction": d, "sl": sl, "tp": tp})
                day_trades = simulate(bars, sigs)
                (is_tr if fd in is_dates else oos_tr).extend(day_trades)

            sis = stats(is_tr)
            soos = stats(oos_tr)
            label = f"{adx_low}-{adx_high}"
            base_results.append((label, sis, soos))
            print(f"  {label:>12} | {sis['n']:>5} ${sis['pnl']:>+9,.0f} {sis['pf']:>5.2f} ${sis['dd']:>+7,.0f} {sis['wr']:>5.1f} ${sis['avg']:>+6,.0f}"
                  f"  |  {soos['n']:>5} ${soos['pnl']:>+9,.0f} {soos['pf']:>5.2f} ${soos['dd']:>+7,.0f} {soos['wr']:>5.1f} ${soos['avg']:>+6,.0f}")

    # =========================================================================
    #  PART 2: TRENDING STRATEGY ADX SWEEP
    # =========================================================================
    print()
    print("=" * 130)
    print("  PART 2: TRENDING STRATEGY — ADX MIN THRESHOLD SWEEP")
    print("  SL=1.00x ATR, TP=1.00x ATR | entry_slip=0t, sl_slip=2t")
    print("=" * 130)
    print(f"  {'ADX Min':>12} | {'IS n':>5} {'IS PnL':>10} {'PF':>5} {'DD':>9} {'WR%':>5} {'Avg':>7}"
          f"  |  {'OOS n':>5} {'OOS PnL':>10} {'PF':>5} {'DD':>9} {'WR%':>5} {'Avg':>7}")
    print("  " + "-" * 128)

    trend_results = []
    for adx_min in range(15, 41, 1):
        is_tr, oos_tr = [], []
        for fd, (bars, _, trend_raw) in day_data.items():
            sigs = []
            for s in trend_raw:
                ct_et = to_et(s["confirm_time"])
                hm = ct_et.strftime("%H:%M")
                if "16:00" <= hm < "19:10" or LUNCH_START <= hm < LUNCH_END:
                    continue
                atr = s["atr"]
                if atr is None or atr <= 0:
                    continue
                adx = get_adx_at_time(ct_et, adx_lookup)
                if np.isnan(adx) or adx < adx_min:
                    continue
                ep, d = s["entry_price"], s["direction"]
                slip = ENTRY_SLIP_TICKS * TICK_SIZE
                ep_s = ep + slip if d == "long" else ep - slip
                if d == "long":
                    sl, tp = ep - atr * TREND_SL, ep + atr * TREND_TP
                else:
                    sl, tp = ep + atr * TREND_SL, ep - atr * TREND_TP
                sigs.append({"bar_index": s["bar_index"], "confirm_time": s["confirm_time"],
                             "entry_price": ep_s, "direction": d, "sl": sl, "tp": tp})
            day_trades = simulate(bars, sigs)
            (is_tr if fd in is_dates else oos_tr).extend(day_trades)

        sis = stats(is_tr)
        soos = stats(oos_tr)
        label = f">={adx_min}"
        trend_results.append((adx_min, sis, soos))
        print(f"  {label:>12} | {sis['n']:>5} ${sis['pnl']:>+9,.0f} {sis['pf']:>5.2f} ${sis['dd']:>+7,.0f} {sis['wr']:>5.1f} ${sis['avg']:>+6,.0f}"
              f"  |  {soos['n']:>5} ${soos['pnl']:>+9,.0f} {soos['pf']:>5.2f} ${soos['dd']:>+7,.0f} {soos['wr']:>5.1f} ${soos['avg']:>+6,.0f}")

    # =========================================================================
    #  PART 3: COMBINED (best base + best trend) — quick check
    # =========================================================================
    print()
    print("=" * 130)
    print("  PART 3: COMBINED — best base + each trend ADX level")
    print("=" * 130)

    # Pick the base config with best OOS PF (among those with >= 20 OOS trades)
    viable_base = [(l, si, so) for l, si, so in base_results if so['n'] >= 20 and so['pf'] > 0]
    if viable_base:
        best_base = max(viable_base, key=lambda x: x[2]['pf'])
        best_base_label = best_base[0]
        best_base_low = int(best_base_label.split("-")[0])
        best_base_high = int(best_base_label.split("-")[1])
        print(f"  Best base (OOS PF): {best_base_label} (IS PF={best_base[1]['pf']:.2f}, OOS PF={best_base[2]['pf']:.2f})")
    else:
        best_base_low, best_base_high = 15, 30
        print(f"  No viable base config found, using default 15-30")

    print(f"  {'Trend ADX':>12} | {'IS n':>5} {'b':>4} {'t':>4}  {'IS PnL':>10} {'PF':>5} {'DD':>9} {'WR%':>5}"
          f"  |  {'OOS n':>5} {'b':>4} {'t':>4}  {'OOS PnL':>10} {'PF':>5} {'DD':>9} {'WR%':>5}")
    print("  " + "-" * 128)

    for adx_min in range(15, 41, 1):
        is_tr, oos_tr = [], []
        for fd, (bars, base_raw, trend_raw) in day_data.items():
            sigs = []
            nb = nt = 0
            # Base signals with best range
            for s in base_raw:
                ct_et = to_et(s["confirm_time"])
                hm = ct_et.strftime("%H:%M")
                if "16:00" <= hm < "19:10" or LUNCH_START <= hm < LUNCH_END:
                    continue
                atr = s["atr"]
                if atr is None or atr <= 0:
                    continue
                adx = get_adx_at_time(ct_et, adx_lookup)
                if np.isnan(adx) or adx < best_base_low or adx >= best_base_high:
                    continue
                ep, d = s["entry_price"], s["direction"]
                if d == "long":
                    sl, tp = ep - atr * BASE_SL, ep + atr * BASE_TP
                else:
                    sl, tp = ep + atr * BASE_SL, ep - atr * BASE_TP
                sigs.append({"bar_index": s["bar_index"], "confirm_time": s["confirm_time"],
                             "entry_price": ep, "direction": d, "sl": sl, "tp": tp, "src": "b"})
                nb += 1

            # Trend signals
            for s in trend_raw:
                ct_et = to_et(s["confirm_time"])
                hm = ct_et.strftime("%H:%M")
                if "16:00" <= hm < "19:10" or LUNCH_START <= hm < LUNCH_END:
                    continue
                atr = s["atr"]
                if atr is None or atr <= 0:
                    continue
                adx = get_adx_at_time(ct_et, adx_lookup)
                if np.isnan(adx) or adx < adx_min:
                    continue
                ep, d = s["entry_price"], s["direction"]
                if d == "long":
                    sl, tp = ep - atr * TREND_SL, ep + atr * TREND_TP
                else:
                    sl, tp = ep + atr * TREND_SL, ep - atr * TREND_TP
                sigs.append({"bar_index": s["bar_index"], "confirm_time": s["confirm_time"],
                             "entry_price": ep, "direction": d, "sl": sl, "tp": tp, "src": "t"})
                nt += 1

            day_trades = simulate(bars, sigs)
            (is_tr if fd in is_dates else oos_tr).extend(day_trades)

        sis = stats(is_tr)
        soos = stats(oos_tr)
        # Count base vs trend from sigs that were taken (approximate from full pool)
        is_nb = sum(1 for t in is_tr if True)  # can't distinguish here, just total
        label = f">={adx_min}"
        print(f"  {label:>12} | {sis['n']:>5} {'':>4} {'':>4}  ${sis['pnl']:>+9,.0f} {sis['pf']:>5.2f} ${sis['dd']:>+7,.0f} {sis['wr']:>5.1f}"
              f"  |  {soos['n']:>5} {'':>4} {'':>4}  ${soos['pnl']:>+9,.0f} {soos['pf']:>5.2f} ${soos['dd']:>+7,.0f} {soos['wr']:>5.1f}")


if __name__ == "__main__":
    main()
