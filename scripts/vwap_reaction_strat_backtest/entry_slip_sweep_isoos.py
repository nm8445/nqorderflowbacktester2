"""
Entry slippage sweep — IS/OOS 50/50 split.

Sweeps entry slippage from 1 to 8 ticks with exit slip fixed at:
  - SL: 2 ticks against (stop-market)
  - TP: 0 ticks (limit fills exact)

ADX >= 25 for trending (lowered from 30 per optimization).

Usage:
  python scripts/vwap_reaction_strat_backtest/entry_slip_sweep_isoos.py
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

SL_SLIP_TICKS = 2
TP_SLIP_TICKS = 0

ENTRY_CUTOFF = "16:00"
FORCE_CLOSE  = "16:58"
BASE_SL, BASE_TP = 0.50, 1.90
TREND_SL, TREND_TP = 1.00, 1.00
TREND_STD, TREND_LB = 1, 14
TREND_ADX_MIN = 25.0

START_DATE = "2025-03-13"
END_DATE   = "2026-04-13"


def to_et(ct):
    return ct.tz_convert(ET) if hasattr(ct, "tz_convert") else pd.Timestamp(ct, tz="UTC").tz_convert(ET)


def prepare_base(signals, adx_lookup, entry_slip_ticks):
    slip = entry_slip_ticks * TICK_SIZE
    out = []
    for s in signals:
        ct_et = to_et(s["confirm_time"])
        hm = ct_et.strftime("%H:%M")
        if "16:00" <= hm < "19:10" or (LUNCH_START <= hm < LUNCH_END): continue
        atr = s["atr"]
        if atr is None or atr <= 0: continue
        adx = get_adx_at_time(ct_et, adx_lookup)
        if not passes_adx_filter(adx): continue
        ep, d = s["entry_price"], s["direction"]
        ep_slipped = ep + slip if d == "long" else ep - slip
        if d == "long":
            sl, tp = ep - atr*BASE_SL, ep + atr*BASE_TP
        else:
            sl, tp = ep + atr*BASE_SL, ep - atr*BASE_TP
        out.append({"source":"base","bar_index":s["bar_index"],"confirm_time":s["confirm_time"],
                    "entry_price":ep_slipped,"direction":d,"sl":sl,"tp":tp})
    return out


def prepare_trend(signals, adx_lookup, entry_slip_ticks):
    slip = entry_slip_ticks * TICK_SIZE
    out = []
    for s in signals:
        ct_et = to_et(s["confirm_time"])
        hm = ct_et.strftime("%H:%M")
        if "16:00" <= hm < "19:10" or (LUNCH_START <= hm < LUNCH_END): continue
        atr = s["atr"]
        if atr is None or atr <= 0: continue
        adx = get_adx_at_time(ct_et, adx_lookup)
        if np.isnan(adx) or adx < TREND_ADX_MIN: continue
        ep, d = s["entry_price"], s["direction"]
        ep_slipped = ep + slip if d == "long" else ep - slip
        if d == "long":
            sl, tp = ep - atr*TREND_SL, ep + atr*TREND_TP
        else:
            sl, tp = ep + atr*TREND_SL, ep - atr*TREND_TP
        out.append({"source":"trend","bar_index":s["bar_index"],"confirm_time":s["confirm_time"],
                    "entry_price":ep_slipped,"direction":d,"sl":sl,"tp":tp})
    return out


def simulate(bars, sigs):
    sigs = sorted(sigs, key=lambda x: x["confirm_time"])
    trades, last_exit = [], None
    for sig in sigs:
        if last_exit is not None and sig["confirm_time"] <= last_exit: continue
        d, ep, sl, tp = sig["direction"], sig["entry_price"], sig["sl"], sig["tp"]
        cbi = sig["bar_index"] + 1
        ex_price = ex_time = ex_reason = None
        for j in range(cbi + 1, len(bars)):
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
        pnl_pts = (ex_price - ep) if d == "long" else (ep - ex_price)
        date_str = str(to_et(sig["confirm_time"]).date())
        trades.append({"source":sig["source"],"date":date_str,
                       "pnl":pnl_pts*POINT_VALUE,"reason":ex_reason})
        last_exit = ex_time
    return trades


def stats(trades):
    if not trades:
        return dict(n=0, pnl=0, wr=0, pf=0, dd=0, sh=0)
    df = pd.DataFrame(trades)
    n = len(df); pnl = df["pnl"].sum()
    w = df[df["pnl"] > 0]; l = df[df["pnl"] < 0]
    wr = 100*len(w)/n
    gp = w["pnl"].sum() if len(w) else 0
    gl = abs(l["pnl"].sum()) if len(l) else 1
    pf = gp/gl if gl > 0 else float("inf")
    cum = df["pnl"].cumsum()
    dd = float((cum - cum.cummax()).min())
    daily = df.groupby("date")["pnl"].sum()
    sh = (daily.mean()/daily.std())*np.sqrt(252) if len(daily) > 1 and daily.std() > 0 else 0
    return dict(n=n, pnl=pnl, wr=wr, pf=pf, dd=dd, sh=sh)


def main():
    adx_lookup = build_adx_lookup()
    start_d = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end_d   = datetime.strptime(END_DATE,   "%Y-%m-%d").date()

    # Collect dates, 50/50 split
    all_dates = []
    for cf in sorted(VWAP_CACHE_DIR.glob("*.pkl")):
        ds = cf.stem
        try: fd = datetime.strptime(ds, "%Y-%m-%d").date()
        except ValueError: continue
        if fd < start_d or fd > end_d: continue
        if (SIGNAL_CACHE_DIR / f"{ds}.pkl").exists():
            all_dates.append(fd)
    split = len(all_dates) // 2
    is_dates = set(all_dates[:split])

    print("=" * 120)
    print(f"  ENTRY SLIPPAGE SWEEP — SL_slip={SL_SLIP_TICKS}t  TP_slip={TP_SLIP_TICKS}t  ADX>={TREND_ADX_MIN:.0f}  50/50 IS/OOS")
    print(f"  IS:  {all_dates[0]} -> {all_dates[split-1]}  ({split} days)")
    print(f"  OOS: {all_dates[split]} -> {all_dates[-1]}  ({len(all_dates)-split} days)")
    print("=" * 120)

    # Pre-load day data
    day_cache = {}
    for fd in all_dates:
        ds = fd.strftime("%Y-%m-%d")
        with open(SIGNAL_CACHE_DIR / f"{ds}.pkl", "rb") as f:
            bars = pickle.load(f).get("bars") or []
        if not bars: continue
        with open(VWAP_CACHE_DIR / f"{ds}.pkl", "rb") as f:
            base_raw = pickle.load(f).get("signals") or []
        trend_raw = []
        vpf = VWAP_PRICE_CACHE_DIR / f"{ds}.pkl"
        tbf = TIMEBARS_DIR / f"timebars_5min_{ds.replace('-','_')}.pkl"
        if vpf.exists() and tbf.exists():
            with open(vpf, "rb") as f: vwap_df = pickle.load(f)
            b5 = load_5min_bars_trending(ds)
            if b5 is not None:
                dstate = precompute_day_state(b5, vwap_df, TREND_LB, TREND_STD)
                trend_raw = detect_trend_signals(bars, dstate)
        day_cache[fd] = (bars, base_raw, trend_raw)

    # Header
    print(f"  {'Entry':>6} | {'IS n':>5}  {'IS PnL':>10} {'PF':>5} {'DD':>9} {'WR%':>5} {'Sh':>5}"
          f"  |  {'OOS n':>5}  {'OOS PnL':>10} {'PF':>5} {'DD':>9} {'WR%':>5} {'Sh':>5}"
          f"  |  {'ALL PnL':>10} {'PF':>5} {'Drag $':>8}")
    print("  " + "-" * 118)

    for entry_slip in range(1, 9):
        is_tr, oos_tr = [], []
        for fd, (bars, base_raw, trend_raw) in day_cache.items():
            sigs = prepare_base(base_raw, adx_lookup, entry_slip) + \
                   prepare_trend(trend_raw, adx_lookup, entry_slip)
            day_trades = simulate(bars, sigs)
            (is_tr if fd in is_dates else oos_tr).extend(day_trades)

        si = stats(is_tr); so = stats(oos_tr)
        sa = stats(is_tr + oos_tr)
        drag = sa["n"] * entry_slip * TICK_SIZE * POINT_VALUE
        print(f"  {entry_slip:>4}t  | {si['n']:>5}  ${si['pnl']:>+9,.0f} {si['pf']:>5.2f} ${si['dd']:>+7,.0f} {si['wr']:>5.1f} {si['sh']:>5.2f}"
              f"  |  {so['n']:>5}  ${so['pnl']:>+9,.0f} {so['pf']:>5.2f} ${so['dd']:>+7,.0f} {so['wr']:>5.1f} {so['sh']:>5.2f}"
              f"  |  ${sa['pnl']:>+9,.0f} {sa['pf']:>5.2f} ${drag:>7,.0f}")


if __name__ == "__main__":
    main()
