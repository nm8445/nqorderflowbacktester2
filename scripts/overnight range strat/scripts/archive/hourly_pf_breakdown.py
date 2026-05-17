"""Hour-of-day PF breakdown for the 977-trade config:
  Locked: B2 X=0.75 N=15 D=70 strict BAND_K=0.25 TP=SL=1.0 chained Mode 1
  Conf:   conf_N=5, conf_D=75  (HALF-of-candle)

Buckets entry_time (ET) into hour ranges and reports per-bucket stats for
IS, OOS, and Combined. Identifies which hours weaken the strategy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from range_break_entry_summary import (
    apply_filters, mode1_chained_dedupe, trade_pnls_vectorized,
)

PARQUET_DIR  = Path(__file__).parent / "parquets"
TRADELOG_DIR = Path(__file__).parent.parent / "tradelogs"
OUT_TXT      = TRADELOG_DIR / "robust_configs" / "hourly_pf_977_trades.txt"

VARIANT, X, N, D, STRICT, BAND_K = "B2", 0.75, 15, 70, True, 0.25
TP_M, SL_M = 1.0, 1.0
CONF_N, CONF_D = 5, 75

# Hour buckets — RTH 9:30-16:00 ET. First bucket is 30-min, rest are 1-hr.
BUCKETS = [
    ("09:30-10:00",  9*60+30, 10*60),
    ("10:00-11:00", 10*60,    11*60),
    ("11:00-12:00", 11*60,    12*60),
    ("12:00-13:00", 12*60,    13*60),
    ("13:00-14:00", 13*60,    14*60),
    ("14:00-15:00", 14*60,    15*60),
    ("15:00-16:00", 15*60,    16*60),
    ("16:00+",      16*60,    24*60),  # any post-16:00 entries (rare)
]


def filter_trades(df: pd.DataFrame) -> pd.DataFrame:
    f = apply_filters(df, VARIANT, X, N, D, STRICT, BAND_K)
    col = f"conf_delta_half_w{CONF_N}"
    cf = f[((f["direction"]=="LONG")  & (f[col].notna()) & (f[col] >=  CONF_D)) |
           ((f["direction"]=="SHORT") & (f[col].notna()) & (f[col] <= -CONF_D))]
    ded = mode1_chained_dedupe(cf, TP_M, SL_M)
    if ded.empty:
        return ded
    ded = ded.copy()
    ded["pnl"]  = trade_pnls_vectorized(ded, TP_M, SL_M)
    ded["date"] = pd.to_datetime(ded["date"]).dt.date
    et = pd.to_datetime(ded["entry_time"])
    if et.dt.tz is None:
        et = et.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    else:
        et = et.dt.tz_convert("America/New_York")
    ded["entry_minute"] = et.dt.hour * 60 + et.dt.minute
    return ded


def stats(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {"n": 0, "total": 0.0, "wr": float("nan"), "pf": float("nan"),
                "sharpe": float("nan"), "n_long": 0, "n_short": 0,
                "long_total": 0.0, "short_total": 0.0, "max_dd": 0.0}
    pnl = sub["pnl"].values
    long_mask  = (sub["direction"] == "LONG").values
    short_mask = (sub["direction"] == "SHORT").values
    pos = pnl[pnl > 0].sum(); neg = -pnl[pnl < 0].sum()
    pf = pos / neg if neg > 0 else (np.inf if pos > 0 else 0.0)
    daily = pd.Series(pnl, index=sub["date"].values).groupby(level=0).sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0
    eq = np.cumsum(pnl); peak = np.maximum.accumulate(eq); max_dd = (eq - peak).min()
    return {
        "n": len(sub), "total": pnl.sum(), "wr": (pnl > 0).mean(),
        "pf": pf, "sharpe": sharpe,
        "n_long": int(long_mask.sum()), "n_short": int(short_mask.sum()),
        "long_total": pnl[long_mask].sum(), "short_total": pnl[short_mask].sum(),
        "max_dd": max_dd,
    }


def fmt_row(label, s):
    pf_str = f"{s['pf']:>5.2f}" if np.isfinite(s['pf']) else "  ∞ "
    return (f"  {label:<14}  n={s['n']:>4}  L={s['n_long']:>3}/S={s['n_short']:>3}  "
            f"total={s['total']:>+8.1f}  L_t={s['long_total']:>+7.1f}  S_t={s['short_total']:>+7.1f}  "
            f"WR={s['wr']:>5.1%}  PF={pf_str}  Sh={s['sharpe']:>+5.2f}  MDD={s['max_dd']:>+7.0f}")


def hourly_breakdown(deduped: pd.DataFrame, label: str, lines: list):
    lines.append("")
    lines.append("=" * 200)
    lines.append(f"{label} HOURLY BREAKDOWN")
    lines.append("=" * 200)
    overall = stats(deduped)
    lines.append(fmt_row("OVERALL", overall))
    lines.append("")
    lines.append("By hour bucket (entry_time ET):")
    for name, lo, hi in BUCKETS:
        sub = deduped[(deduped["entry_minute"] >= lo) & (deduped["entry_minute"] < hi)]
        s = stats(sub)
        lines.append(fmt_row(name, s))


def main():
    print("loading + filtering trades for 977-trade config...")
    is_df  = pd.read_parquet(PARQUET_DIR / "entry_signal_trades.parquet")
    oos_df = pd.read_parquet(PARQUET_DIR / "entry_signal_trades_oos.parquet")
    is_ded  = filter_trades(is_df)
    oos_ded = filter_trades(oos_df)
    combined = pd.concat([is_ded, oos_ded], ignore_index=True)
    print(f"  IS:       {len(is_ded):,} trades")
    print(f"  OOS:      {len(oos_ded):,} trades")
    print(f"  Combined: {len(combined):,} trades")

    lines = []
    lines.append("=" * 200)
    lines.append("HOURLY ENTRY-TIME PF BREAKDOWN")
    lines.append("=" * 200)
    lines.append("")
    lines.append("Config: B2 X=0.75 N=15 D=70 strict BAND_K=0.25 TP=SL=1.0 chained Mode 1")
    lines.append(f"        + conf_N={CONF_N}, conf_D={CONF_D} (HALF-of-candle)")
    lines.append("")
    lines.append("Trade volume by period:")
    lines.append(f"  IS  (2020-12 -> 2024-12) : {len(is_ded):,} trades")
    lines.append(f"  OOS (2025-01 -> 2026-05) : {len(oos_ded):,} trades")
    lines.append(f"  Combined                  : {len(combined):,} trades")

    hourly_breakdown(combined, "COMBINED IS+OOS", lines)
    hourly_breakdown(is_ded,   "IN-SAMPLE", lines)
    hourly_breakdown(oos_ded,  "OUT-OF-SAMPLE", lines)

    # Identify weak hours per period
    lines.append("")
    lines.append("=" * 200)
    lines.append("WEAK HOURS — buckets where PF dips below the period overall PF")
    lines.append("=" * 200)
    for name_period, ded in [("IS", is_ded), ("OOS", oos_ded), ("COMBINED", combined)]:
        ov = stats(ded)
        lines.append(f"\n{name_period} overall PF: {ov['pf']:.2f}")
        for name, lo, hi in BUCKETS:
            sub = ded[(ded["entry_minute"] >= lo) & (ded["entry_minute"] < hi)]
            if len(sub) < 5:
                continue
            s = stats(sub)
            tag = ""
            if s['pf'] < 1.0:
                tag = "  <- LOSING (PF<1.0)"
            elif s['pf'] < ov['pf']:
                tag = f"  <- BELOW OVERALL (PF {s['pf']:.2f} < {ov['pf']:.2f})"
            if tag:
                lines.append(f"  {name:<14}  n={s['n']:>4}  PF={s['pf']:.2f}  Sharpe={s['sharpe']:+.2f}  total={s['total']:+.1f}{tag}")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}")
    print()
    print("\n".join(lines[lines.index("=" * 200, 5)+1:]))   # print all output


if __name__ == "__main__":
    sys.exit(main())
