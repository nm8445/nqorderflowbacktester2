"""
Trend ADX threshold sweep — IS/OOS 50/50 split.

Holds base strategy fixed (ADX 15-30 via passes_adx_filter). Varies only the
trending-band ADX minimum from 20 to 30 to find where the trending source
actually starts adding edge.

Slippage model matches live revisit behavior:
  - Entry:  0 ticks (revisit fills at confirm close)
  - SL:     2 ticks against
  - TP:     0 ticks

Usage:
  python scripts/vwap_reaction_strat_backtest/trend_adx_sweep_isoos.py
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

# Live-matched slippage (revisit zero entry slip)
ENTRY_SLIP_TICKS = 0
SL_SLIP_TICKS    = 2
TP_SLIP_TICKS    = 0

ENTRY_CUTOFF = "16:00"
FORCE_CLOSE  = "16:58"
BASE_SL, BASE_TP = 0.50, 1.90
TREND_SL, TREND_TP = 1.00, 1.00
TREND_STD, TREND_LB = 1, 14

START_DATE = "2025-03-13"
END_DATE   = "2026-04-13"

# ADX sweep range
ADX_MIN_LOW = 20
ADX_MIN_HIGH = 30


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
        slip = ENTRY_SLIP_TICKS * TICK_SIZE
        ep_slipped = ep + slip if d == "long" else ep - slip
        if d == "long":
            sl, tp = ep - atr*BASE_SL, ep + atr*BASE_TP
        else:
            sl, tp = ep + atr*BASE_SL, ep - atr*BASE_TP
        out.append({"source":"base","bar_index":s["bar_index"],"confirm_time":s["confirm_time"],
                    "entry_price":ep_slipped,"direction":d,"sl":sl,"tp":tp})
    return out


def prepare_trend(signals, adx_lookup, adx_min):
    out = []
    for s in signals:
        ct_et = to_et(s["confirm_time"])
        hm = ct_et.strftime("%H:%M")
        if "16:00" <= hm < "19:10" or (LUNCH_START <= hm < LUNCH_END):
            continue
        atr = s["atr"]
        if atr is None or atr <= 0: continue
        adx = get_adx_at_time(ct_et, adx_lookup)
        if np.isnan(adx) or adx < adx_min: continue
        ep, d = s["entry_price"], s["direction"]
        slip = ENTRY_SLIP_TICKS * TICK_SIZE
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
        if last_exit is not None and sig["confirm_time"] <= last_exit:
            continue
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
        return dict(n=0, nb=0, nt=0, pnl=0, wr=0, pf=0, dd=0, sh=0)
    df = pd.DataFrame(trades)
    n = len(df); pnl = df["pnl"].sum()
    wins = df[df["pnl"] > 0]; losses = df[df["pnl"] < 0]
    wr = len(wins)/n*100
    gp = wins["pnl"].sum() if len(wins) else 0
    gl = abs(losses["pnl"].sum()) if len(losses) else 1
    pf = gp/gl if gl > 0 else float("inf")
    cum = df["pnl"].cumsum()
    dd = float((cum - cum.cummax()).min())
    daily = df.groupby("date")["pnl"].sum()
    sh = (daily.mean() / daily.std()) * np.sqrt(252) if len(daily) > 1 and daily.std() > 0 else 0.0
    return dict(n=n, nb=(df["source"]=="base").sum(), nt=(df["source"]=="trend").sum(),
                pnl=pnl, wr=wr, pf=pf, dd=dd, sh=sh)


def main():
    adx_lookup = build_adx_lookup()
    start_d = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end_d   = datetime.strptime(END_DATE,   "%Y-%m-%d").date()

    # Collect all eligible dates first, split 50/50
    all_dates = []
    for cache_file in sorted(VWAP_CACHE_DIR.glob("*.pkl")):
        ds = cache_file.stem
        try:
            fd = datetime.strptime(ds, "%Y-%m-%d").date()
        except ValueError: continue
        if fd < start_d or fd > end_d: continue
        if (SIGNAL_CACHE_DIR / f"{ds}.pkl").exists():
            all_dates.append(fd)
    split = len(all_dates) // 2
    is_dates = set(all_dates[:split])
    is_end = all_dates[split-1]
    oos_start = all_dates[split]

    print("=" * 115)
    print(f"  TREND ADX SWEEP — IS/OOS 50/50 split  |  entry_slip={ENTRY_SLIP_TICKS}t  sl_slip={SL_SLIP_TICKS}t")
    print(f"  IS:  {all_dates[0]} ->{is_end}  ({split} days)")
    print(f"  OOS: {oos_start} ->{all_dates[-1]}  ({len(all_dates) - split} days)")
    print("=" * 115)

    # Pre-load everything per-day, cache base+trend-raw so ADX filter change is cheap
    day_cache = {}
    for fd in all_dates:
        ds = fd.strftime("%Y-%m-%d")
        with open(SIGNAL_CACHE_DIR / f"{ds}.pkl", "rb") as f:
            bars = pickle.load(f).get("bars") or []
        if not bars: continue
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
                trend_raw = detect_trend_signals(bars, dstate)
        day_cache[fd] = (bars, base_raw, trend_raw)

    # Header
    print(f"  {'ADX':>4} | {'IS n':>5} {'b':>4} {'t':>4}  {'IS PnL':>10} {'PF':>5} {'DD':>9} {'WR%':>5} {'Sh':>5}"
          f"  |  {'OOS n':>5} {'b':>4} {'t':>4}  {'OOS PnL':>10} {'PF':>5} {'DD':>9} {'WR%':>5} {'Sh':>5}")
    print("  " + "-" * 113)

    for adx_min in range(ADX_MIN_LOW, ADX_MIN_HIGH + 1):
        is_tr, oos_tr = [], []
        base_prepped_cache = {}
        for fd, (bars, base_raw, trend_raw) in day_cache.items():
            # Base prep is independent of trend ADX — memoize
            if fd not in base_prepped_cache:
                base_prepped_cache[fd] = prepare_base(base_raw, adx_lookup)
            sigs = base_prepped_cache[fd] + prepare_trend(trend_raw, adx_lookup, float(adx_min))
            day_trades = simulate(bars, sigs)
            (is_tr if fd in is_dates else oos_tr).extend(day_trades)

        sis = stats(is_tr); soos = stats(oos_tr)
        print(f"  {adx_min:>4} | {sis['n']:>5} {sis['nb']:>4} {sis['nt']:>4}  "
              f"${sis['pnl']:>+9,.0f} {sis['pf']:>5.2f} ${sis['dd']:>+7,.0f} {sis['wr']:>5.1f} {sis['sh']:>5.2f}"
              f"  |  {soos['n']:>5} {soos['nb']:>4} {soos['nt']:>4}  "
              f"${soos['pnl']:>+9,.0f} {soos['pf']:>5.2f} ${soos['dd']:>+7,.0f} {soos['wr']:>5.1f} {soos['sh']:>5.2f}")


if __name__ == "__main__":
    main()
