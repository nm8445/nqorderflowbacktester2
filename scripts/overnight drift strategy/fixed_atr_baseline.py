"""
Baseline: same entry (19:00 ET long), force-close 08:00 ET, but with a fixed
1:1 ATR-multiple SL/TP locked at entry. Compare win rate against the
yellow/green/red structure.

Both runs use:
- No BE
- No martingale (qty=1 always)
- Same 20-min bar series
- ATR(14) computed off the same series (yellow's ATR)

For each entry, the SL and TP levels are computed as
    sl = entry_close - mult * ATR_at_entry
    tp = entry_close + mult * ATR_at_entry
and held fixed for the trade. If both could fire on the same bar, the SL is
booked (worst-case convention used by most platforms when stop and target
straddle the open).
"""

from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overnight_drift_strategy import (  # noqa: E402
    NQ_POINT_VALUE,
    StrategyParams,
    build_full_20min_series,
    rma_atr,
    run_backtest,
    trades_to_df,
)

PARQUET_PATH = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
PICKLE_FOLDER = "D:/trading_pythonbacktest_data/timebars_5min"


def run_fixed_atr(
    bars: pd.DataFrame,
    mult: float,
    atr_len: int = 14,
    entry_t: time = time(19, 0),
    force_t: time = time(8, 0),
) -> pd.DataFrame:
    atr = rma_atr(bars["high"], bars["low"], bars["close"], atr_len).values
    o = bars["open"].values
    h = bars["high"].values
    l = bars["low"].values
    c = bars["close"].values
    idx = bars.index

    in_pos = False
    entry_price = np.nan
    sl_lvl = np.nan
    tp_lvl = np.nan
    entry_ts = None
    entry_idx = -1
    rows = []

    for i in range(len(bars)):
        ts = idx[i]
        local_t = ts.time()

        if not in_pos and local_t == entry_t and not np.isnan(atr[i]):
            in_pos = True
            entry_price = c[i]
            entry_ts = ts
            entry_idx = i
            sl_lvl = entry_price - mult * atr[i]
            tp_lvl = entry_price + mult * atr[i]
            continue

        if in_pos:
            exited = False
            exit_price = np.nan
            reason = ""

            # Gap-through handling: if the bar opens beyond a level, fill at open.
            if o[i] <= sl_lvl:
                exit_price = o[i]
                reason = "SL"
                exited = True
            elif o[i] >= tp_lvl:
                exit_price = o[i]
                reason = "TP"
                exited = True
            else:
                hit_sl = l[i] <= sl_lvl
                hit_tp = h[i] >= tp_lvl
                if hit_sl and hit_tp:
                    exit_price = sl_lvl  # worst-case
                    reason = "SL"
                    exited = True
                elif hit_sl:
                    exit_price = sl_lvl
                    reason = "SL"
                    exited = True
                elif hit_tp:
                    exit_price = tp_lvl
                    reason = "TP"
                    exited = True
                elif local_t == force_t:
                    exit_price = c[i]
                    reason = "Force"
                    exited = True

            if exited:
                rows.append(
                    {
                        "entry_time": entry_ts,
                        "exit_time": ts,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "reason": reason,
                        "bars_held": i - entry_idx,
                        "pnl_points": exit_price - entry_price,
                        "pnl_dollars": (exit_price - entry_price) * NQ_POINT_VALUE,
                    }
                )
                in_pos = False
                entry_price = np.nan
                sl_lvl = np.nan
                tp_lvl = np.nan
                entry_ts = None
                entry_idx = -1

    return pd.DataFrame(rows)


def stats(df: pd.DataFrame) -> dict:
    wins = (df["pnl_dollars"] > 0).sum()
    losses = (df["pnl_dollars"] < 0).sum()
    gw = df.loc[df["pnl_dollars"] > 0, "pnl_dollars"].sum()
    gl = abs(df.loc[df["pnl_dollars"] < 0, "pnl_dollars"].sum())
    return {
        "trades": len(df),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate_%": (wins / len(df) * 100) if len(df) else float("nan"),
        "gross_$": df["pnl_dollars"].sum(),
        "avg_$": df["pnl_dollars"].mean(),
        "median_$": df["pnl_dollars"].median(),
        "PF": gw / gl if gl else float("inf"),
        "avg_pts": df["pnl_points"].mean(),
    }


def main() -> None:
    bars = build_full_20min_series(PARQUET_PATH, PICKLE_FOLDER)
    print(f"bars: {len(bars):,}  range: {bars.index.min()} -> {bars.index.max()}\n")

    # ---------- baseline at each multiplier ----------
    rows = []
    for m in [1.0, 1.5, 2.0, 2.5, 3.0]:
        df = run_fixed_atr(bars, m)
        s = stats(df)
        s["mult"] = m
        s["TP_hits"] = (df["reason"] == "TP").sum()
        s["SL_hits"] = (df["reason"] == "SL").sum()
        s["Force_hits"] = (df["reason"] == "Force").sum()
        rows.append(s)
    base = pd.DataFrame(rows).set_index("mult")[
        [
            "trades",
            "wins",
            "losses",
            "win_rate_%",
            "TP_hits",
            "SL_hits",
            "Force_hits",
            "gross_$",
            "avg_$",
            "median_$",
            "PF",
            "avg_pts",
        ]
    ]
    print("=== Fixed ATR 1:1 RR baseline (no BE, no martingale, qty=1) ===")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(base.round(2).to_string())

    # ---------- yellow/green strategy with same settings (no BE, no marti) ----------
    p = StrategyParams(use_be=False, use_martingale=False, base_qty=1, loss_qty=1)
    yg = trades_to_df(run_backtest(bars, p))
    s = stats(yg)
    s["TP_hits"] = (yg["reason"] == "TP Green").sum()
    s["SL_hits"] = (yg["reason"] == "SL Yellow").sum()
    s["Force_hits"] = (yg["reason"] == "Force Close").sum()
    s["BE_hits"] = (yg["reason"] == "BE Stop").sum()
    print("\n=== Yellow/Green/Red strategy (no BE, no martingale, qty=1) ===")
    print(pd.Series(s).round(2).to_string())

    # ---------- yearly win rate per variant ----------
    print("\n\n=== Yearly win-rate comparison ===")
    bars_year_wr = {}
    for m in [1.0, 1.5, 2.0, 2.5, 3.0]:
        df = run_fixed_atr(bars, m)
        df["entry_time"] = pd.to_datetime(df["entry_time"]).dt.tz_convert("America/New_York")
        df["year"] = df["entry_time"].dt.year
        bars_year_wr[f"ATR x{m}"] = df.groupby("year").apply(
            lambda g: (g["pnl_dollars"] > 0).mean() * 100, include_groups=False
        )
    yg["entry_time"] = pd.to_datetime(yg["entry_time"]).dt.tz_convert("America/New_York")
    yg["year"] = yg["entry_time"].dt.year
    bars_year_wr["Yellow/Green"] = yg.groupby("year").apply(
        lambda g: (g["pnl_dollars"] > 0).mean() * 100, include_groups=False
    )
    cmp_wr = pd.DataFrame(bars_year_wr).round(1)
    print(cmp_wr.to_string())

    print("\n=== Yearly gross $ comparison ===")
    rows = {}
    for m in [1.0, 1.5, 2.0, 2.5, 3.0]:
        df = run_fixed_atr(bars, m)
        df["entry_time"] = pd.to_datetime(df["entry_time"]).dt.tz_convert("America/New_York")
        df["year"] = df["entry_time"].dt.year
        rows[f"ATR x{m}"] = df.groupby("year")["pnl_dollars"].sum()
    rows["Yellow/Green"] = yg.groupby("year")["pnl_dollars"].sum()
    cmp_pnl = pd.DataFrame(rows).round(0).fillna(0).astype(int)
    print(cmp_pnl.to_string())


if __name__ == "__main__":
    main()
