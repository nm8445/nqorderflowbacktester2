"""
Base strategy SL/TP grid with level_delta >= 50 absorption filter.
IS/OOS 50/50 chronological split. No ADX filter.
SL and TP from 0.5 to 4.0 ATR in 0.5 increments.
Shows top 20 robust results with positive R:R (TP > SL).

Usage:
    python scripts/vwap_reaction_strat_backtest/base_lvldelta50_sltp_grid.py
"""
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from adx_filter_common import ET, DATA_DIR, LUNCH_START, LUNCH_END

SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
VWAP_CACHE_DIR = DATA_DIR / "vwap_cache"
TIMEBARS_DIR = DATA_DIR / "timebars_5min"

POINT_VALUE = 20.0
TICK_SIZE = 0.25
VWAP_ZONE_POINTS = 3.0
ATR_PERIOD = 14
LEVEL_DELTA_MIN = 50

ENTRY_SLIP_TICKS = 0
SL_SLIP_TICKS = 2
TP_SLIP_TICKS = 0
FORCE_CLOSE = "16:58"

START_DATE = "2025-03-13"
END_DATE = "2026-04-17"


def to_et(ct):
    return ct.tz_convert(ET) if hasattr(ct, "tz_convert") else pd.Timestamp(ct, tz="UTC").tz_convert(ET)


def load_5min_bars_fixed(date_str):
    bar_file = TIMEBARS_DIR / f"timebars_5min_{date_str.replace('-','_')}.pkl"
    if not bar_file.exists():
        return None
    with open(bar_file, "rb") as f:
        bars = pickle.load(f)
    if not bars:
        return None
    rows = [{'timestamp': b['open_time'], 'high': b['high'], 'low': b['low'], 'close': b['close']} for b in bars]
    df = pd.DataFrame(rows).set_index('timestamp').sort_index()
    df.index = df.index + pd.Timedelta(minutes=5)
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['prev_close']),
                                     abs(df['low'] - df['prev_close'])))
    df['atr'] = df['tr'].rolling(ATR_PERIOD, min_periods=1).mean()
    return df


def get_vwap_at_time(vwap_df, bar_time):
    if vwap_df is None or vwap_df.empty:
        return None
    prior = vwap_df[vwap_df.index <= bar_time]
    if prior.empty:
        return None
    return float(prior['vwap'].iloc[-1])


def get_atr_at_time(bars_5min, signal_time):
    if bars_5min is None or bars_5min.empty:
        return None
    prior = bars_5min[bars_5min.index <= signal_time]
    if prior.empty:
        return None
    return float(prior['atr'].iloc[-1])


def get_5min_bias_zone(bars_5min, vwap_df, as_of_time):
    if bars_5min is None or bars_5min.empty or vwap_df is None or vwap_df.empty:
        return None
    prior_bars = bars_5min[bars_5min.index <= as_of_time]
    if prior_bars.empty:
        return None
    for idx in range(len(prior_bars) - 1, max(len(prior_bars) - 50, -1), -1):
        bar_close = float(prior_bars['close'].iloc[idx])
        bar_time = prior_bars.index[idx]
        vwap_at_bar = get_vwap_at_time(vwap_df, bar_time)
        if vwap_at_bar is None:
            continue
        if bar_close > vwap_at_bar + VWAP_ZONE_POINTS:
            return "long"
        elif bar_close < vwap_at_bar - VWAP_ZONE_POINTS:
            return "short"
    return None


def detect_signals(bars, vwap_df, bars_5min):
    """Detect base VWAP reaction signals with level_delta >= LEVEL_DELTA_MIN."""
    signals = []
    for i in range(len(bars) - 1):
        sb = bars[i]
        cb = bars[i + 1]
        if not sb.closed or not cb.closed:
            continue

        bt = sb.close_time
        bt_et = to_et(bt)
        hm = bt_et.strftime("%H:%M")
        if "16:00" <= hm < "19:10":
            continue

        closed_bearish = sb.close < sb.open
        closed_bullish = sb.close > sb.open
        if not closed_bearish and not closed_bullish:
            continue

        has_bear_abs = False
        has_bull_abs = False

        for price, lv in sb.levels.items():
            delta = lv.delta
            if closed_bearish and delta >= LEVEL_DELTA_MIN:
                has_bear_abs = True
                break
            elif closed_bullish and delta <= -LEVEL_DELTA_MIN:
                has_bull_abs = True
                break

        if not has_bear_abs and not has_bull_abs:
            continue

        vwap_value = get_vwap_at_time(vwap_df, bt)
        if vwap_value is None:
            continue

        zh = vwap_value + VWAP_ZONE_POINTS
        zl = vwap_value - VWAP_ZONE_POINTS
        if not (sb.high >= zl and sb.low <= zh):
            continue

        confirm_vwap = get_vwap_at_time(vwap_df, cb.close_time)
        if confirm_vwap is None:
            confirm_vwap = vwap_value
        czh = confirm_vwap + VWAP_ZONE_POINTS
        czl = confirm_vwap - VWAP_ZONE_POINTS
        if not (cb.high >= czl and cb.low <= czh):
            continue

        bias = get_5min_bias_zone(bars_5min, vwap_df, bt)
        if bias is None:
            continue

        lookback = min(10, i)
        approached_above = any(bars[j].close > zh for j in range(i - lookback, i) if bars[j].closed)
        approached_below = any(bars[j].close < zl for j in range(i - lookback, i) if bars[j].closed)

        confirm_bearish = cb.close < cb.open
        confirm_bullish = cb.close > cb.open

        atr = get_atr_at_time(bars_5min, bt)
        if atr is None or atr <= 0:
            continue

        if bias == "long" and approached_above and has_bull_abs and confirm_bullish:
            signals.append({"bar_index": i, "direction": "long",
                            "confirm_time": cb.close_time, "entry_price": cb.close, "atr": atr})
        if bias == "short" and approached_below and has_bear_abs and confirm_bearish:
            signals.append({"bar_index": i, "direction": "short",
                            "confirm_time": cb.close_time, "entry_price": cb.close, "atr": atr})

    return signals


def simulate(bars, sigs, sl_mult, tp_mult):
    sigs = sorted(sigs, key=lambda x: x["confirm_time"])
    trades, last_exit = [], None
    for sig in sigs:
        ct_et = to_et(sig["confirm_time"])
        hm = ct_et.strftime("%H:%M")
        if LUNCH_START <= hm < LUNCH_END:
            continue
        if last_exit is not None and sig["confirm_time"] <= last_exit:
            continue
        d, ep, atr = sig["direction"], sig["entry_price"], sig["atr"]
        if d == "long":
            sl = ep - atr * sl_mult
            tp = ep + atr * tp_mult
        else:
            sl = ep + atr * sl_mult
            tp = ep - atr * tp_mult

        cbi = sig["bar_index"] + 1
        ex_price = ex_time = None
        for j in range(cbi + 1, len(bars)):
            b = bars[j]
            if not b.closed:
                continue
            bet = to_et(b.close_time)
            if bet.strftime("%H:%M") >= FORCE_CLOSE:
                ex_price, ex_time = b.close, b.close_time
                break
            if d == "long":
                if b.low <= sl:
                    ex_price, ex_time = sl - SL_SLIP_TICKS * TICK_SIZE, b.close_time
                    break
                if b.high >= tp:
                    ex_price, ex_time = tp - TP_SLIP_TICKS * TICK_SIZE, b.close_time
                    break
            else:
                if b.high >= sl:
                    ex_price, ex_time = sl + SL_SLIP_TICKS * TICK_SIZE, b.close_time
                    break
                if b.low <= tp:
                    ex_price, ex_time = tp + TP_SLIP_TICKS * TICK_SIZE, b.close_time
                    break
        if ex_price is None:
            ex_price, ex_time = bars[-1].close, bars[-1].close_time
        pnl = ((ex_price - ep) if d == "long" else (ep - ex_price)) * POINT_VALUE
        trades.append({"date": str(to_et(sig["confirm_time"]).date()), "pnl": pnl})
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
    start_d = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end_d = datetime.strptime(END_DATE, "%Y-%m-%d").date()

    all_dates = []
    for sig_file in sorted(SIGNAL_CACHE_DIR.glob("*.pkl")):
        ds = sig_file.stem
        try:
            fd = datetime.strptime(ds, "%Y-%m-%d").date()
        except ValueError:
            continue
        if fd < start_d or fd > end_d:
            continue
        vwap_file = VWAP_CACHE_DIR / f"{ds}.pkl"
        tb_file = TIMEBARS_DIR / f"timebars_5min_{ds.replace('-','_')}.pkl"
        if vwap_file.exists() and tb_file.exists():
            all_dates.append(fd)

    split = len(all_dates) // 2
    is_dates = set(all_dates[:split])
    print(f"Total days: {len(all_dates)} | IS: {all_dates[0]}->{all_dates[split-1]} ({split}d) | OOS: {all_dates[split]}->{all_dates[-1]} ({len(all_dates)-split}d)")

    # Pre-load data and detect signals once (signal detection doesn't depend on SL/TP)
    print("Loading data and detecting signals (level_delta >= 50)...")
    day_signals = {}
    for fd in all_dates:
        ds = fd.strftime("%Y-%m-%d")
        with open(SIGNAL_CACHE_DIR / f"{ds}.pkl", "rb") as f:
            bars = pickle.load(f).get("bars") or []
        if not bars:
            continue
        with open(VWAP_CACHE_DIR / f"{ds}.pkl", "rb") as f:
            vwap_df = pickle.load(f)
        bars_5min = load_5min_bars_fixed(ds)
        if bars_5min is None:
            continue
        sigs = detect_signals(bars, vwap_df, bars_5min)
        day_signals[fd] = (bars, sigs)

    total_sigs = sum(len(s) for _, s in day_signals.values())
    print(f"Loaded {len(day_signals)} days, {total_sigs} raw signals\n")

    # SL/TP grid
    sl_values = [x * 0.5 for x in range(1, 9)]   # 0.5 to 4.0
    tp_values = [x * 0.5 for x in range(1, 9)]   # 0.5 to 4.0

    results = []
    for sl_m in sl_values:
        for tp_m in tp_values:
            if tp_m <= sl_m:  # positive R:R only
                continue
            is_tr, oos_tr = [], []
            for fd, (bars, sigs) in day_signals.items():
                day_trades = simulate(bars, sigs, sl_m, tp_m)
                (is_tr if fd in is_dates else oos_tr).extend(day_trades)
            si = stats(is_tr)
            so = stats(oos_tr)
            results.append({
                "sl": sl_m, "tp": tp_m, "rr": tp_m / sl_m,
                "is": si, "oos": so,
            })

    # Rank by robustness: both IS and OOS profitable, sort by min(IS PF, OOS PF)
    # Filter: both IS and OOS must be profitable (PF > 1) and have trades
    robust = [r for r in results
              if r["is"]["pf"] > 1.0 and r["oos"]["pf"] > 1.0
              and r["is"]["n"] >= 5 and r["oos"]["n"] >= 5]
    robust.sort(key=lambda r: min(r["is"]["pf"], r["oos"]["pf"]), reverse=True)

    # Full grid
    print("=" * 150)
    print(f"  FULL SL/TP GRID: BASE + LEVEL_DELTA >= 50  (positive R:R only)")
    print("=" * 150)
    print(f"  {'SL':>4} {'TP':>4} {'R:R':>5}"
          f" | {'IS n':>5} {'IS PnL':>9} {'IS PF':>6} {'IS DD':>8} {'IS WR%':>6} {'IS Avg':>7}"
          f" | {'OOS n':>5} {'OOS PnL':>9} {'OOS PF':>6} {'OOS DD':>8} {'OOS WR%':>6} {'OOS Avg':>7}")
    print("  " + "-" * 145)

    for r in sorted(results, key=lambda x: (x["sl"], x["tp"])):
        si, so = r["is"], r["oos"]
        flag = " *" if r in robust else "  "
        print(f"  {r['sl']:>4.1f} {r['tp']:>4.1f} {r['rr']:>5.1f}"
              f" | {si['n']:>5} ${si['pnl']:>+8,.0f} {si['pf']:>6.2f} ${si['dd']:>+7,.0f} {si['wr']:>6.1f} ${si['avg']:>+6,.0f}"
              f" | {so['n']:>5} ${so['pnl']:>+8,.0f} {so['pf']:>6.2f} ${so['dd']:>+7,.0f} {so['wr']:>6.1f} ${so['avg']:>+6,.0f}"
              f"{flag}")

    # Top 20 robust
    print()
    print("=" * 150)
    print(f"  TOP 20 ROBUST CONFIGS (both IS & OOS PF > 1.0, positive R:R, min 5 trades each)")
    print(f"  Ranked by min(IS PF, OOS PF)")
    print("=" * 150)
    print(f"  {'#':>3} {'SL':>4} {'TP':>4} {'R:R':>5}"
          f" | {'IS n':>5} {'IS PnL':>9} {'IS PF':>6} {'IS DD':>8} {'IS WR%':>6}"
          f" | {'OOS n':>5} {'OOS PnL':>9} {'OOS PF':>6} {'OOS DD':>8} {'OOS WR%':>6}"
          f" | {'min PF':>6}")
    print("  " + "-" * 130)

    for rank, r in enumerate(robust[:20], 1):
        si, so = r["is"], r["oos"]
        mpf = min(si["pf"], so["pf"])
        print(f"  {rank:>3} {r['sl']:>4.1f} {r['tp']:>4.1f} {r['rr']:>5.1f}"
              f" | {si['n']:>5} ${si['pnl']:>+8,.0f} {si['pf']:>6.2f} ${si['dd']:>+7,.0f} {si['wr']:>6.1f}"
              f" | {so['n']:>5} ${so['pnl']:>+8,.0f} {so['pf']:>6.2f} ${so['dd']:>+7,.0f} {so['wr']:>6.1f}"
              f" | {mpf:>6.2f}")

    if not robust:
        print("  No configs found with both IS and OOS PF > 1.0")

    # Also show total stats for reference
    print()
    print(f"  Total configs tested: {len(results)} | Robust (both PF>1): {len(robust)}")


if __name__ == "__main__":
    main()
