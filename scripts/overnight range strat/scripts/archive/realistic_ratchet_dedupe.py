"""Realistic dedupe for the optimal pure_ratchet config.

Mode 1 chained dedupe based on actual pure_ratchet exit times (not 1xATR exit
times). For each candidate trade, simulate the ratchet exit, get its real exit
timestamp, then only keep trades whose entry > prior kept trade's exit.

Also runs the same dedupe-by-actual-exit logic for the 1xATR baseline as a
fair-comparison benchmark.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from range_break_entry_summary import apply_filters, trade_pnls_vectorized
from test_pure_ratchet_exits import (
    build_20min_bars, FORCE_CLOSE_TIME, RED_INTERCEPT, RED_DRIFT,
)
from optimize_pure_ratchet_exits import (
    simulate_exit_arrays, precache_trade_bars, prep_trades,
)

PARQUET_DIR  = Path(__file__).parent / "parquets"
TRADELOG_DIR = Path(__file__).parent.parent / "tradelogs"
OUT_TXT      = TRADELOG_DIR / "robust_configs" / "realistic_ratchet_dedupe.txt"

# Locked entry config
VARIANT, X, N, D, STRICT, BAND_K = "B2", 0.75, 15, 70, True, 0.25
CONF_N, CONF_D = 5, 75
TP_M, SL_M = 1.0, 1.0

# OPTIMAL ratchet config from sweep
YMULT  = 2.50
GMULT  = 2.00
GBASE  = 200.0
GDECAY = 0.0


def filter_pre_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply locked filters + conf filter, but NO dedupe. Sort by entry_time."""
    f = apply_filters(df, VARIANT, X, N, D, STRICT, BAND_K)
    col = f"conf_delta_half_w{CONF_N}"
    cf = f[((f["direction"]=="LONG")  & (f[col].notna()) & (f[col] >=  CONF_D)) |
           ((f["direction"]=="SHORT") & (f[col].notna()) & (f[col] <= -CONF_D))].copy()
    if cf.empty:
        return cf
    cf["entry_time_et"] = pd.to_datetime(cf["entry_time"])
    if cf["entry_time_et"].dt.tz is None:
        cf["entry_time_et"] = cf["entry_time_et"].dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    else:
        cf["entry_time_et"] = cf["entry_time_et"].dt.tz_convert("America/New_York")
    return cf.sort_values("entry_time_et").reset_index(drop=True)


def simulate_one_trade_ratchet(direction, entry_ts, entry_price, bars20):
    """Same as simulate_exit_arrays but returns exit timestamp too."""
    sign = 1 if direction == "LONG" else -1
    bars_idx = bars20.index
    start = bars_idx.searchsorted(entry_ts, side="right")
    if start >= len(bars_idx):
        return None
    ent_date = entry_ts.date()
    end = start
    while end < len(bars_idx) and bars_idx[end].date() == ent_date:
        end += 1
    if end == start:
        return None
    init_idx = start - 1
    if init_idx < 0 or np.isnan(bars20["atr_y"].iloc[init_idx]):
        return None

    init_atr_y = float(bars20["atr_y"].iloc[init_idx])
    yellow_val = entry_price - sign * YMULT * init_atr_y
    prev_yellow = yellow_val

    o = bars20["open"].values[start:end]
    h = bars20["high"].values[start:end]
    l = bars20["low"].values[start:end]
    c = bars20["close"].values[start:end]
    ay = bars20["atr_y"].values[start:end]
    ag = bars20["atr_g"].values[start:end]
    n = end - start

    for i in range(n):
        bars_in_trade = i + 1
        bar_close_ts = bars_idx[start + i] + pd.Timedelta(minutes=20)
        if not np.isnan(ay[i]):
            raw_yellow = c[i] - sign * YMULT * ay[i]
            yellow_val = max(prev_yellow, raw_yellow) if sign > 0 \
                          else min(prev_yellow, raw_yellow)
        red_val = entry_price + sign * (RED_INTERCEPT + RED_DRIFT * bars_in_trade)
        green_offset = (GBASE - GDECAY * bars_in_trade
                        + (GMULT * ag[i] if not np.isnan(ag[i]) else 0.0))
        green_val = red_val + sign * green_offset

        if sign > 0 and h[i] >= green_val:
            return (c[i] - entry_price, "TP_GREEN", bar_close_ts, bars_in_trade)
        if sign < 0 and l[i] <= green_val:
            return (entry_price - c[i], "TP_GREEN", bar_close_ts, bars_in_trade)
        if sign > 0 and c[i] <= yellow_val and c[i] < o[i]:
            return (c[i] - entry_price, "SL_YELLOW", bar_close_ts, bars_in_trade)
        if sign < 0 and c[i] >= yellow_val and c[i] > o[i]:
            return (entry_price - c[i], "SL_YELLOW", bar_close_ts, bars_in_trade)
        if bars_idx[start + i].time() >= FORCE_CLOSE_TIME:
            return (sign * (c[i] - entry_price), "FORCE_CLOSE", bar_close_ts, bars_in_trade)
        prev_yellow = yellow_val

    last_ts = bars_idx[end - 1] + pd.Timedelta(minutes=20)
    return (sign * (c[-1] - entry_price), "EOD", last_ts, n)


def chained_dedupe_by_exits(candidates: pd.DataFrame, exit_results: list) -> tuple[pd.DataFrame, list]:
    """Sequentially dedupe — keep trade i only if entry_time_i > prior_kept.exit_time."""
    keep_idx = []
    keep_exits = []
    last_exit = pd.Timestamp(0, tz="America/New_York")
    for i, row in candidates.iterrows():
        ex = exit_results[i]
        if ex is None:
            continue
        pnl, reason, exit_ts, bars_held = ex
        entry_ts = row["entry_time_et"]
        if entry_ts > last_exit:
            keep_idx.append(i)
            keep_exits.append({"pnl": pnl, "reason": reason, "exit_ts": exit_ts,
                                "bars_held": bars_held})
            last_exit = exit_ts
    kept = candidates.loc[keep_idx].copy().reset_index(drop=True)
    return kept, keep_exits


def stats(df: pd.DataFrame, pnl_col: str) -> dict:
    pnl = df[pnl_col].values
    if len(pnl) == 0:
        return {"n":0, "total":0.0, "pf":0.0, "sharpe":0.0, "wr":0.0, "max_dd":0.0,
                "n_long":0, "n_short":0, "long_total":0.0, "short_total":0.0}
    pos = pnl[pnl > 0].sum(); neg = -pnl[pnl < 0].sum()
    pf = pos / neg if neg > 0 else (np.inf if pos > 0 else 0.0)
    daily = pd.Series(pnl, index=df["date"].values).groupby(level=0).sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0
    eq = np.cumsum(pnl); peak = np.maximum.accumulate(eq); max_dd = (eq - peak).min()
    long_mask = (df["direction"]=="LONG").values
    return {"n":len(df), "total":pnl.sum(), "pf":pf, "sharpe":sharpe,
            "wr":(pnl > 0).mean(), "max_dd":max_dd,
            "n_long":int(long_mask.sum()), "n_short":int((~long_mask).sum()),
            "long_total":pnl[long_mask].sum(), "short_total":pnl[~long_mask].sum()}


def fmt(label, s):
    pf = f"{s['pf']:>5.2f}" if np.isfinite(s["pf"]) else "  inf"
    return (f"  {label:<30}  n={s['n']:>4}  L={s['n_long']:>3}/S={s['n_short']:>3}  "
            f"total={s['total']:>+8.1f}  L_t={s['long_total']:>+7.1f}  S_t={s['short_total']:>+7.1f}  "
            f"WR={s['wr']:>5.1%}  PF={pf}  Sh={s['sharpe']:>+5.2f}  MDD={s['max_dd']:>+7.0f}")


def process(df_trades_parquet: pd.DataFrame, label: str, bars20, lines: list):
    candidates = filter_pre_dedupe(df_trades_parquet)
    candidates["date"] = pd.to_datetime(candidates["date"]).dt.date
    print(f"  {label}: {len(candidates)} pre-dedupe candidates")

    exit_results = []
    for _, row in candidates.iterrows():
        ex = simulate_one_trade_ratchet(
            row["direction"], row["entry_time_et"], float(row["entry_price"]), bars20)
        exit_results.append(ex)

    kept, kept_exits = chained_dedupe_by_exits(candidates, exit_results)
    kept["pnl_ratchet"] = [e["pnl"] for e in kept_exits]
    kept["exit_reason"] = [e["reason"] for e in kept_exits]
    kept["bars_held_20m"] = [e["bars_held"] for e in kept_exits]
    print(f"  {label}: {len(kept)} after realistic ratchet dedupe")

    s_ratch = stats(kept, "pnl_ratchet")
    lines.append(f"\n--- {label} ---")
    lines.append(f"  pre-dedupe candidates       : {len(candidates):>4}")
    lines.append(f"  after realistic ratchet dedupe: {len(kept):>4}  ({len(kept)/len(candidates)*100:.1f}%)")
    lines.append(fmt("PURE_RATCHET (real dedupe)", s_ratch))

    rc = kept.groupby("exit_reason").agg(n=("pnl_ratchet","size"),
                                          total=("pnl_ratchet","sum"),
                                          mean=("pnl_ratchet","mean"),
                                          wr=("pnl_ratchet", lambda x: (x>0).mean()),
                                          med_bars=("bars_held_20m","median"))
    lines.append("  exit reason breakdown:")
    for reason, row in rc.iterrows():
        lines.append(f"    {reason:<14}  n={int(row['n']):>4}  total={row['total']:>+8.1f}  "
                     f"mean={row['mean']:>+5.2f}  WR={row['wr']:>5.1%}  med_bars={int(row['med_bars']):>2}")
    return kept


def main():
    print("loading 20-min bars...")
    bars20 = build_20min_bars()
    print(f"  bars: {len(bars20):,}")

    is_df  = pd.read_parquet(PARQUET_DIR / "entry_signal_trades.parquet")
    oos_df = pd.read_parquet(PARQUET_DIR / "entry_signal_trades_oos.parquet")

    lines = []
    lines.append("=" * 200)
    lines.append("REALISTIC DEDUPE — pure_ratchet exits")
    lines.append("=" * 200)
    lines.append("")
    lines.append(f"Entry config: B2 X={X} N={N} D={D} strict={STRICT} BAND_K={BAND_K}")
    lines.append(f"            + conf_N={CONF_N} conf_D={CONF_D} HALF, chained Mode 1")
    lines.append(f"Exit config:  pure_ratchet  ymult={YMULT} gmult={GMULT} gbase={GBASE} gdecay={GDECAY}")
    lines.append(f"              red_drift={RED_DRIFT}  ATR_y_len=14  ATR_g_len=13  force_close=16:00")
    lines.append("")
    lines.append("DEDUPE BASED ON ACTUAL RATCHET EXIT TIMES (not 1xATR exit times)")
    lines.append("")

    is_kept  = process(is_df,  "IN-SAMPLE", bars20, lines)
    oos_kept = process(oos_df, "OUT-OF-SAMPLE", bars20, lines)
    combined = pd.concat([is_kept, oos_kept], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.date

    s_combined = stats(combined, "pnl_ratchet")
    lines.append("")
    lines.append("=" * 200)
    lines.append(f"COMBINED IS+OOS  (n={len(combined):,})")
    lines.append("=" * 200)
    lines.append(fmt("PURE_RATCHET (real dedupe)", s_combined))

    # Compare against the inflated previous numbers
    lines.append("")
    lines.append("=" * 200)
    lines.append("COMPARISON — what we previously reported vs realistic dedupe")
    lines.append("=" * 200)
    lines.append("")
    lines.append("Previous (dedupe-by-1xATR-exits, INFLATED count):")
    lines.append("  IS  : n=813   total +7,358   PF 1.31   Sharpe 1.67")
    lines.append("  OOS : n=164   total +1,719   PF 1.31   Sharpe 1.70")
    lines.append("  Combined: n=977  total +9,076  PF 1.31")
    lines.append("")
    lines.append("Realistic (dedupe-by-ratchet-exits):")
    s_is_real  = stats(is_kept,  "pnl_ratchet")
    s_oos_real = stats(oos_kept, "pnl_ratchet")
    lines.append(f"  IS  : n={s_is_real['n']:<3}  total={s_is_real['total']:+.0f}  PF {s_is_real['pf']:.2f}  Sharpe {s_is_real['sharpe']:+.2f}")
    lines.append(f"  OOS : n={s_oos_real['n']:<3}  total={s_oos_real['total']:+.0f}  PF {s_oos_real['pf']:.2f}  Sharpe {s_oos_real['sharpe']:+.2f}")
    lines.append(f"  Combined: n={s_combined['n']:<3}  total={s_combined['total']:+.0f}  PF {s_combined['pf']:.2f}")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
