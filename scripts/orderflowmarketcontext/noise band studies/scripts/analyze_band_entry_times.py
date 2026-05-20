"""Distribution and PnL of band-break entries by time-of-day.

Reads overnight_band_per_day.parquet and produces:
  1. A per-trade CSV sorted by entry time
  2. Summary tables: 30-min entry buckets with hit rate, mean PnL,
     and per-direction breakdown
  3. Hourly granularity comparison
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PARQUET   = Path(__file__).parent / "overnight_band_per_day.parquet"
OUT_CSV   = Path(__file__).parent / "band_entries_by_time.csv"
OUT_TXT   = Path(__file__).parent / "band_entries_by_time_summary.txt"

LB = "14d"  # focus per user request

def main():
    df = pd.read_parquet(PARQUET)
    df = df[df[f"break_class_simple_{LB}"].isin(["above_only","below_only"])].copy()
    df["direction"] = np.where(df[f"break_class_simple_{LB}"] == "above_only", "long", "short")
    df["entry_time"] = df[f"first_break_time_{LB}"]
    df["entry_price"] = df[f"entry_price_{LB}"]
    df["pnl"] = df[f"ret_from_entry_{LB}"]
    df.loc[df["direction"] == "short", "pnl"] *= -1  # short PnL = -ret

    # Useful columns only
    keep = ["date","direction","entry_time","entry_price","close_1700","pnl",
            "qqq_regime_open","ndx_regime_open",
            f"upper_band_{LB}", f"lower_band_{LB}"]
    sub = df[keep].copy()
    sub = sub.rename(columns={f"upper_band_{LB}": "upper_band",
                              f"lower_band_{LB}": "lower_band"})
    sub = sub.sort_values(["entry_time", "date"]).reset_index(drop=True)
    sub.to_csv(OUT_CSV, index=False)
    print(f"wrote per-trade CSV: {OUT_CSV}  ({len(sub)} trades)")

    lines = []
    lines.append(f"Band-break entries (lookback={LB}) by time-of-day")
    lines.append(f"Total trades: {len(sub)}  |  Long: {(sub['direction']=='long').sum()}  "
                 f"|  Short: {(sub['direction']=='short').sum()}")
    lines.append(f"Date range: {sub['date'].min()} -> {sub['date'].max()}")
    lines.append("")

    # Convert entry_time to minute-of-day for bucketing
    sub["t_min"] = sub["entry_time"].apply(
        lambda s: int(s.split(":")[0]) * 60 + int(s.split(":")[1]))

    def bucket_label(start_min: int, end_min: int) -> str:
        sh, sm = divmod(start_min, 60); eh, em = divmod(end_min, 60)
        return f"{sh:02d}:{sm:02d}-{eh:02d}:{em:02d}"

    def stats_block(label: str, df_block: pd.DataFrame) -> str:
        n = len(df_block)
        if n == 0: return f"  {label:<14}  n=0"
        p_profit = (df_block["pnl"] > 0).mean()
        m = df_block["pnl"].mean()
        std = df_block["pnl"].std()
        return (f"  {label:<14}  n={n:>4}  P(profit)={p_profit:.1%}  "
                f"mean={m:+7.2f} pts  std={std:7.2f}")

    # 30-min buckets from 9:30 to 17:00
    starts = list(range(9*60+30, 17*60, 30))  # 570, 600, 630, ... 990
    lines.append("=" * 90)
    lines.append("30-MIN ENTRY BUCKETS (LONG and SHORT combined)")
    lines.append("=" * 90)
    for s in starts:
        e = s + 30
        block = sub[(sub["t_min"] >= s) & (sub["t_min"] < e)]
        lines.append(stats_block(bucket_label(s, e), block))

    for direction in ["long", "short"]:
        lines.append("")
        lines.append("=" * 90)
        lines.append(f"30-MIN ENTRY BUCKETS — {direction.upper()} ONLY")
        lines.append("=" * 90)
        df_d = sub[sub["direction"] == direction]
        for s in starts:
            e = s + 30
            block = df_d[(df_d["t_min"] >= s) & (df_d["t_min"] < e)]
            lines.append(stats_block(bucket_label(s, e), block))

    # Hourly granularity for both directions side-by-side
    lines.append("")
    lines.append("=" * 90)
    lines.append("HOURLY BUCKETS — LONG vs SHORT side-by-side")
    lines.append("=" * 90)
    lines.append(f"  {'Hour':<14}  {'LONG':<40}  {'SHORT':<40}")
    hourly_starts = list(range(9*60+30, 17*60, 60))  # ditto but 60-min steps
    for s in hourly_starts:
        e = s + 60
        long_block = sub[(sub["direction"]=="long") & (sub["t_min"]>=s) & (sub["t_min"]<e)]
        short_block = sub[(sub["direction"]=="short") & (sub["t_min"]>=s) & (sub["t_min"]<e)]
        def short_stats(b):
            n = len(b)
            if n == 0: return "n=0"
            return f"n={n:>3}  P={ (b['pnl']>0).mean():.0%}  m={b['pnl'].mean():+6.1f}"
        lines.append(f"  {bucket_label(s, e):<14}  "
                     f"{short_stats(long_block):<40}  {short_stats(short_block):<40}")

    # Top entry times by frequency
    lines.append("")
    lines.append("=" * 90)
    lines.append("TOP 15 EXACT ENTRY TIMES BY FREQUENCY")
    lines.append("=" * 90)
    grp = sub.groupby(["entry_time", "direction"]).agg(
        n=("pnl", "size"),
        p_profit=("pnl", lambda x: (x > 0).mean()),
        mean_pnl=("pnl", "mean"),
    ).reset_index().sort_values("n", ascending=False)
    for _, r in grp.head(20).iterrows():
        lines.append(f"  {r['entry_time']}  {r['direction']:<6}  n={int(r['n']):>4}  "
                     f"P(profit)={r['p_profit']:.1%}  mean={r['mean_pnl']:+7.2f}")

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote summary: {OUT_TXT}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
