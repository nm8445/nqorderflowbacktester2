"""
Asymmetric ATR SL/TP grid for the overnight drift strategy.

Entry: long at 19:00 ET (single 20-min bar close).
Exit:
  - SL_pts = sl_mult * ATR(atr_len)_at_entry
  - TP_pts = tp_mult * ATR(atr_len)_at_entry
  - Force-close at 08:00 ET if neither hit.
  - Gap-through fills at bar open. If both could hit on same bar, SL wins.
  - ATR locked at the entry bar (not rolling).
  - No BE, no martingale, qty=1.

Sweep sl_mult x tp_mult. Report:
  - All-period PF, win%, gross_$, max DD
  - Per-fold (pre + 4 walk-forward folds) breakdown for top picks
  - Also dumps per-trade outcomes for the best config so the prop-firm
    sim can read them back.
"""

from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overnight_drift_strategy import build_full_20min_series, rma_atr  # noqa: E402

NQ_POINT = 20.0
PARQUET = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
PICKLES = "D:/trading_pythonbacktest_data/timebars_5min"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
TZ = "America/New_York"

ENTRY_T = time(19, 0)
FORCE_T = time(8, 0)
ATR_LEN = 14

FOLDS = [
    ("Pre-fold", "2020-12-01", "2022-11-30"),
    ("Fold 1  ", "2022-12-01", "2023-11-30"),
    ("Fold 2  ", "2023-12-01", "2024-11-30"),
    ("Fold 3  ", "2024-12-01", "2025-11-30"),
    ("Fold 4  ", "2025-12-01", "2026-05-31"),
]


def run_asym(bars: pd.DataFrame, sl_mult: float, tp_mult: float, atr_len: int = ATR_LEN) -> pd.DataFrame:
    atr = rma_atr(bars["high"], bars["low"], bars["close"], atr_len).values
    o = bars["open"].values
    h = bars["high"].values
    l = bars["low"].values
    c = bars["close"].values
    idx = bars.index
    rows = []
    in_pos = False
    entry = np.nan
    sl_lvl = np.nan
    tp_lvl = np.nan
    et = None
    atr_entry = np.nan
    sl_pts = np.nan
    tp_pts = np.nan
    for i in range(len(bars)):
        ts = idx[i]
        t_local = ts.time()
        if not in_pos and t_local == ENTRY_T and not np.isnan(atr[i]):
            in_pos = True
            entry = c[i]
            atr_entry = atr[i]
            sl_pts = sl_mult * atr_entry
            tp_pts = tp_mult * atr_entry
            sl_lvl = entry - sl_pts
            tp_lvl = entry + tp_pts
            et = ts
            continue
        if in_pos:
            exit_p = np.nan
            reason = ""
            if o[i] <= sl_lvl:
                exit_p, reason = o[i], "SL"
            elif o[i] >= tp_lvl:
                exit_p, reason = o[i], "TP"
            else:
                hit_sl = l[i] <= sl_lvl
                hit_tp = h[i] >= tp_lvl
                if hit_sl and hit_tp:
                    exit_p, reason = sl_lvl, "SL"
                elif hit_sl:
                    exit_p, reason = sl_lvl, "SL"
                elif hit_tp:
                    exit_p, reason = tp_lvl, "TP"
                elif t_local == FORCE_T:
                    exit_p, reason = c[i], "Force"
            if not np.isnan(exit_p):
                pnl_pts = exit_p - entry
                rows.append(
                    {
                        "entry_time": et,
                        "exit_time": ts,
                        "entry": entry,
                        "exit": exit_p,
                        "atr_entry": atr_entry,
                        "sl_pts": sl_pts,
                        "tp_pts": tp_pts,
                        "reason": reason,
                        "pnl_pts": pnl_pts,
                        "pnl_$_per_NQ_contract": pnl_pts * NQ_POINT,
                    }
                )
                in_pos = False
    return pd.DataFrame(rows)


def stats(df: pd.DataFrame) -> dict:
    pnl = df["pnl_$_per_NQ_contract"].values if "pnl_$_per_NQ_contract" in df.columns else df["pnl_$"].values
    if len(pnl) == 0:
        return {"trades": 0, "win%": np.nan, "PF": np.nan, "gross_$": 0.0, "avg_$": np.nan}
    wins = (pnl > 0).sum()
    gw = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    return {
        "trades": int(len(pnl)),
        "win%": float(wins / len(pnl) * 100),
        "PF": float(gw / gl) if gl > 0 else float("inf"),
        "gross_$": float(pnl.sum()),
        "avg_$": float(pnl.mean()),
    }


def maxdd(df: pd.DataFrame) -> float:
    s = df.sort_values("entry_time").copy()
    s["c"] = s["pnl_$_per_NQ_contract"].cumsum()
    s["p"] = s["c"].cummax()
    return float((s["c"] - s["p"]).min()) if len(s) else 0.0


def main() -> None:
    print("Loading bars...", flush=True)
    bars = build_full_20min_series(PARQUET, PICKLES)
    print(f"  bars: {len(bars):,}  range: {bars.index.min()} -> {bars.index.max()}\n", flush=True)

    SL = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    TP = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

    rows = []
    best_df = None
    best_score = -1e18
    best_key = None
    for sl in SL:
        for tp in TP:
            df = run_asym(bars, sl, tp)
            df["entry_time"] = pd.to_datetime(df["entry_time"]).dt.tz_convert(TZ)
            s = stats(df)
            dd = maxdd(df)
            rows.append(
                {
                    "sl_mult": sl,
                    "tp_mult": tp,
                    "RR": round(tp / sl, 2),
                    **s,
                    "MaxDD_$": dd,
                    "TP_hits": int((df["reason"] == "TP").sum()),
                    "SL_hits": int((df["reason"] == "SL").sum()),
                    "Force_hits": int((df["reason"] == "Force").sum()),
                }
            )
            # Score by PF * (gross > 0)
            score = s["PF"] if s["gross_$"] > 0 else 0
            if score > best_score:
                best_score = score
                best_df = df
                best_key = (sl, tp)

    grid = pd.DataFrame(rows)

    # PF pivot
    print("=== PF (rows=sl_mult, cols=tp_mult) ===")
    pf_piv = grid.pivot(index="sl_mult", columns="tp_mult", values="PF").round(2)
    print(pf_piv.to_string())

    print("\n=== Win % ===")
    wr_piv = grid.pivot(index="sl_mult", columns="tp_mult", values="win%").round(1)
    print(wr_piv.to_string())

    print("\n=== Gross $ ===")
    gr_piv = grid.pivot(index="sl_mult", columns="tp_mult", values="gross_$").round(0)
    print(gr_piv.to_string())

    print("\n=== Max DD $ ===")
    dd_piv = grid.pivot(index="sl_mult", columns="tp_mult", values="MaxDD_$").round(0)
    print(dd_piv.to_string())

    # Top 10 by PF (eligible PF > 1.0)
    eligible = grid[grid["PF"] > 1.0].copy()
    top = eligible.sort_values("PF", ascending=False).head(15)
    print("\n=== Top 15 configs by PF ===")
    show = top[["sl_mult", "tp_mult", "RR", "trades", "win%", "PF",
                "TP_hits", "SL_hits", "Force_hits", "gross_$", "MaxDD_$"]].round(2)
    with pd.option_context("display.width", 200):
        print(show.to_string(index=False))

    # Per-fold for the top 5
    print("\n=== Per-fold breakdown for top 5 ===")
    top5 = top.head(5)[["sl_mult", "tp_mult"]].values.tolist()
    fold_rows = []
    for sl, tp in top5:
        df = run_asym(bars, sl, tp)
        df["entry_time"] = pd.to_datetime(df["entry_time"]).dt.tz_convert(TZ)
        for lbl, lo, hi in FOLDS:
            m = (df["entry_time"] >= pd.Timestamp(lo, tz=TZ)) & (df["entry_time"] <= pd.Timestamp(hi + " 23:59:59", tz=TZ))
            sub = df[m]
            s = stats(sub)
            fold_rows.append({"sl": sl, "tp": tp, "fold": lbl, **s})
    fold_df = pd.DataFrame(fold_rows)
    piv = fold_df.pivot_table(index=["sl", "tp"], columns="fold", values="PF").round(2)
    print(piv.to_string())
    print()
    piv_g = fold_df.pivot_table(index=["sl", "tp"], columns="fold", values="gross_$").round(0)
    print(piv_g.to_string())

    # Save grid
    grid.to_csv(OUT / "asym_atr_grid.csv", index=False)

    # Save best trade log (for prop firm sim downstream)
    sl_b, tp_b = best_key
    print(f"\nBest config: sl_mult={sl_b} tp_mult={tp_b}  PF={best_score:.2f}")
    best_df["entry_time"] = pd.to_datetime(best_df["entry_time"]).dt.tz_convert(TZ)
    best_df["exit_time"] = pd.to_datetime(best_df["exit_time"]).dt.tz_convert(TZ)
    best_df.to_csv(OUT / "asym_atr_best_trades.csv", index=False)
    print(f"Saved trade log -> {OUT / 'asym_atr_best_trades.csv'}")
    print(f"Saved grid     -> {OUT / 'asym_atr_grid.csv'}")


if __name__ == "__main__":
    main()
