"""
Grid optimization: ATR SL (0.1-3.0) x ATR TP (0.1-3.0) in 0.1 increments.
Split: first half in-sample, second half out-of-sample.
With and without martingale (double after loss, reset after win, cap 8x).
Config: Longs only + contra CVD (10-bar).
"""

import pickle
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
POC_REACTION_CACHE_DIR = DATA_DIR / "poc_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
POINT_VALUE = 20.0
MAX_MARTINGALE = 8  # cap at 8x contracts

ENTRY_START = "09:00"
ENTRY_END = "11:00"


def compute_recent_cvd(bars, up_to_index, lookback=10):
    start_idx = max(0, up_to_index - lookback)
    cvd = 0.0
    for i in range(start_idx, min(up_to_index, len(bars))):
        bar = bars[i]
        if not bar.closed:
            continue
        for price, lv in bar.levels.items():
            cvd += lv.delta
    return cvd


def precompute_signals():
    """Load all qualifying signals with bar sequences for fast replay."""
    start = datetime.strptime("2025-03-13", "%Y-%m-%d").date()
    end = datetime.strptime("2026-04-08", "%Y-%m-%d").date()

    all_signals = []

    for cache_file in sorted(POC_REACTION_CACHE_DIR.glob("*.pkl")):
        date_str = cache_file.stem
        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if file_date < start or file_date > end:
            continue

        with open(cache_file, "rb") as f:
            poc_data = pickle.load(f)

        signals = poc_data["signals"]
        if not signals:
            continue

        signal_cache_file = SIGNAL_CACHE_DIR / f"{date_str}.pkl"
        if not signal_cache_file.exists():
            continue

        with open(signal_cache_file, "rb") as f:
            signal_data = pickle.load(f)

        bars = signal_data["bars"]

        filtered_signals = []
        for s in signals:
            # Longs only
            if s["direction"] != "long":
                continue

            confirm_time = s["confirm_time"]
            if hasattr(confirm_time, "tz_convert"):
                confirm_et = confirm_time.tz_convert(ET)
            else:
                confirm_et = pd.Timestamp(confirm_time, tz="UTC").tz_convert(ET)
            hour_min = confirm_et.strftime("%H:%M")
            if ENTRY_START <= hour_min < ENTRY_END:
                filtered_signals.append(s)

        if not filtered_signals:
            continue

        in_trade = False
        for signal in filtered_signals:
            if in_trade:
                continue

            entry_price = signal["entry_price"]
            confirm_bar_idx = signal["bar_index"] + 1
            atr = signal["atr"]

            if atr is None or atr <= 0:
                continue

            # Contra CVD filter
            cvd = compute_recent_cvd(bars, confirm_bar_idx + 1)
            if cvd >= 0:
                continue

            # Precompute bar sequence after entry (highs and lows)
            bar_seq = []
            for j in range(confirm_bar_idx + 1, len(bars)):
                bar = bars[j]
                if not bar.closed:
                    continue
                bar_seq.append((bar.high, bar.low, bar.close))

            if not bar_seq:
                continue

            in_trade = True  # one trade at a time per day

            all_signals.append({
                "date": file_date,
                "entry_price": entry_price,
                "atr": atr,
                "signal_bar_low": signal["signal_bar_low"],
                "bar_seq": bar_seq,
                "last_close": bar_seq[-1][2],
            })
            in_trade = False

    return all_signals


def simulate_trade(sig, sl_mult, tp_mult):
    """Simulate a single trade, return pnl in points."""
    entry = sig["entry_price"]
    atr = sig["atr"]

    sl_price = entry - atr * sl_mult
    tp_price = entry + atr * tp_mult

    if tp_price <= entry or sl_price >= entry:
        return None

    for high, low, close in sig["bar_seq"]:
        # Check SL first (adverse for longs)
        if low <= sl_price:
            return sl_price - entry
        # Check TP
        if high >= tp_price:
            return tp_price - entry

    # EOD
    return sig["last_close"] - entry


def run_grid(signals, sl_range, tp_range):
    """Run full grid and return results dict keyed by (sl_mult, tp_mult)."""

    # Split IS/OOS by date
    dates = sorted(set(s["date"] for s in signals))
    mid = len(dates) // 2
    is_dates = set(dates[:mid])
    oos_dates = set(dates[mid:])

    is_signals = [s for s in signals if s["date"] in is_dates]
    oos_signals = [s for s in signals if s["date"] in oos_dates]

    print(f"Total signals: {len(signals)}")
    print(f"IS dates: {min(is_dates)} to {max(is_dates)} ({len(is_signals)} trades)")
    print(f"OOS dates: {min(oos_dates)} to {max(oos_dates)} ({len(oos_signals)} trades)")
    print(f"Grid: {len(sl_range)} x {len(tp_range)} = {len(sl_range)*len(tp_range)} combos")
    print()

    results = []

    for sl_mult in sl_range:
        for tp_mult in tp_range:
            for label, sigs in [("IS", is_signals), ("OOS", oos_signals)]:
                pnls = []
                for sig in sigs:
                    pnl = simulate_trade(sig, sl_mult, tp_mult)
                    if pnl is not None:
                        pnls.append(pnl)

                if not pnls:
                    continue

                n = len(pnls)
                wins = sum(1 for p in pnls if p > 0)
                wr = wins / n * 100

                gross_profit = sum(p for p in pnls if p > 0)
                gross_loss = abs(sum(p for p in pnls if p < 0))
                pf = gross_profit / gross_loss if gross_loss > 0 else 999

                total_pnl = sum(pnls)
                avg_win = np.mean([p for p in pnls if p > 0]) if wins > 0 else 0
                avg_loss = np.mean([p for p in pnls if p < 0]) if (n - wins) > 0 else 0
                exp = total_pnl / n

                # Max drawdown
                cum = np.cumsum(pnls)
                running_max = np.maximum.accumulate(cum)
                dd = cum - running_max
                max_dd = float(dd.min()) if len(dd) > 0 else 0

                # Martingale
                marty_pnl = 0.0
                contracts = 1
                marty_cum = []
                for p in pnls:
                    marty_pnl += p * contracts * POINT_VALUE
                    marty_cum.append(marty_pnl)
                    if p > 0:
                        contracts = 1
                    else:
                        contracts = min(contracts * 2, MAX_MARTINGALE)

                marty_total = marty_pnl
                marty_arr = np.array(marty_cum)
                marty_running_max = np.maximum.accumulate(marty_arr)
                marty_dd = marty_arr - marty_running_max
                marty_max_dd = float(marty_dd.min()) if len(marty_dd) > 0 else 0

                results.append({
                    "sl": sl_mult,
                    "tp": tp_mult,
                    "period": label,
                    "trades": n,
                    "wr": wr,
                    "pf": pf,
                    "exp_pts": exp,
                    "total_pts": total_pnl,
                    "total_dollars": total_pnl * POINT_VALUE,
                    "max_dd_pts": max_dd,
                    "max_dd_dollars": max_dd * POINT_VALUE,
                    "avg_win": avg_win,
                    "avg_loss": avg_loss,
                    "marty_total": marty_total,
                    "marty_max_dd": marty_max_dd,
                })

    return pd.DataFrame(results)


def find_robust(df):
    """Find combos that are profitable in BOTH IS and OOS."""
    is_df = df[df["period"] == "IS"].set_index(["sl", "tp"])
    oos_df = df[df["period"] == "OOS"].set_index(["sl", "tp"])

    common = is_df.index.intersection(oos_df.index)

    robust = []
    for key in common:
        is_row = is_df.loc[key]
        oos_row = oos_df.loc[key]

        # Both profitable, both PF > 1
        if is_row["total_pts"] > 0 and oos_row["total_pts"] > 0 and is_row["pf"] > 1 and oos_row["pf"] > 1:
            combined_pnl = is_row["total_dollars"] + oos_row["total_dollars"]
            combined_marty = is_row["marty_total"] + oos_row["marty_total"]
            worse_dd = min(is_row["max_dd_dollars"], oos_row["max_dd_dollars"])
            worse_marty_dd = min(is_row["marty_max_dd"], oos_row["marty_max_dd"])

            robust.append({
                "sl": key[0],
                "tp": key[1],
                "is_trades": is_row["trades"],
                "is_wr": is_row["wr"],
                "is_pf": is_row["pf"],
                "is_pnl": is_row["total_dollars"],
                "is_dd": is_row["max_dd_dollars"],
                "is_exp": is_row["exp_pts"],
                "oos_trades": oos_row["trades"],
                "oos_wr": oos_row["wr"],
                "oos_pf": oos_row["pf"],
                "oos_pnl": oos_row["total_dollars"],
                "oos_dd": oos_row["max_dd_dollars"],
                "oos_exp": oos_row["exp_pts"],
                "combined_pnl": combined_pnl,
                "combined_marty": combined_marty,
                "worse_dd": worse_dd,
                "worse_marty_dd": worse_marty_dd,
                "is_marty": is_row["marty_total"],
                "oos_marty": oos_row["marty_total"],
                "is_marty_dd": is_row["marty_max_dd"],
                "oos_marty_dd": oos_row["marty_max_dd"],
            })

    return pd.DataFrame(robust)


if __name__ == "__main__":
    print("Precomputing signals...")
    signals = precompute_signals()
    print(f"Found {len(signals)} qualifying signals")
    print()

    sl_range = np.round(np.arange(0.1, 3.05, 0.1), 1)
    tp_range = np.round(np.arange(0.1, 3.05, 0.1), 1)

    print("Running grid optimization...")
    df = run_grid(signals, sl_range, tp_range)

    robust = find_robust(df)

    if robust.empty:
        print("NO ROBUST COMBOS FOUND (profitable in both IS and OOS)")
    else:
        robust = robust.sort_values("combined_pnl", ascending=False)

        print("=" * 120)
        print(f"ROBUST COMBOS: Profitable in both IS and OOS ({len(robust)} found)")
        print("=" * 120)
        print()

        # Top 20 by combined P&L (no martingale)
        print("--- TOP 20 BY COMBINED P&L (no martingale) ---")
        print(f"{'SL':>4s} {'TP':>4s} | {'IS tr':>5s} {'IS WR':>6s} {'IS PF':>6s} {'IS P&L':>9s} {'IS DD':>9s} | {'OOS tr':>6s} {'OOS WR':>6s} {'OOS PF':>6s} {'OOS P&L':>9s} {'OOS DD':>9s} | {'TOTAL':>9s}")
        print("-" * 120)
        for _, r in robust.head(20).iterrows():
            print(f"{r['sl']:4.1f} {r['tp']:4.1f} | "
                  f"{r['is_trades']:5.0f} {r['is_wr']:5.1f}% {r['is_pf']:6.2f} ${r['is_pnl']:+8,.0f} ${r['is_dd']:+8,.0f} | "
                  f"{r['oos_trades']:5.0f} {r['oos_wr']:5.1f}% {r['oos_pf']:6.2f} ${r['oos_pnl']:+8,.0f} ${r['oos_dd']:+8,.0f} | "
                  f"${r['combined_pnl']:+8,.0f}")
        print()

        # Top 20 by combined P&L (martingale)
        robust_marty = robust.sort_values("combined_marty", ascending=False)
        print("--- TOP 20 BY COMBINED P&L (martingale, 2x after loss, cap 8x) ---")
        print(f"{'SL':>4s} {'TP':>4s} | {'IS $':>9s} {'IS DD':>9s} | {'OOS $':>9s} {'OOS DD':>9s} | {'TOTAL':>9s}")
        print("-" * 90)
        for _, r in robust_marty.head(20).iterrows():
            print(f"{r['sl']:4.1f} {r['tp']:4.1f} | "
                  f"${r['is_marty']:+8,.0f} ${r['is_marty_dd']:+8,.0f} | "
                  f"${r['oos_marty']:+8,.0f} ${r['oos_marty_dd']:+8,.0f} | "
                  f"${r['combined_marty']:+8,.0f}")
        print()

        # Stability: combos where OOS PF >= 80% of IS PF
        stable = robust[robust["oos_pf"] >= robust["is_pf"] * 0.8].sort_values("combined_pnl", ascending=False)
        print(f"--- MOST STABLE (OOS PF >= 80% of IS PF): {len(stable)} combos ---")
        print(f"{'SL':>4s} {'TP':>4s} | {'IS PF':>6s} {'OOS PF':>6s} | {'IS P&L':>9s} {'OOS P&L':>9s} | {'TOTAL':>9s} | {'IS mty':>9s} {'OOS mty':>9s} {'mty TOT':>9s}")
        print("-" * 120)
        for _, r in stable.head(20).iterrows():
            print(f"{r['sl']:4.1f} {r['tp']:4.1f} | "
                  f"{r['is_pf']:6.2f} {r['oos_pf']:6.2f} | "
                  f"${r['is_pnl']:+8,.0f} ${r['oos_pnl']:+8,.0f} | "
                  f"${r['combined_pnl']:+8,.0f} | "
                  f"${r['is_marty']:+8,.0f} ${r['oos_marty']:+8,.0f} ${r['combined_marty']:+8,.0f}")
        print()

        print(f"Total robust combos: {len(robust)}")
        print(f"Stable combos (OOS PF >= 80% IS PF): {len(stable)}")
