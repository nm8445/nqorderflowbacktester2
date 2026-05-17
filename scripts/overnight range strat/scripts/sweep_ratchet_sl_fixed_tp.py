"""Ratchet SL + Fixed TP sweep with realistic dedupe.

  Yellow (SL): pure_ratchet at ymult * ATR_14_20min  (trails, never moves against)
  Green  (TP): FIXED at entry + tp_mult * ATR_13_20min (no decay, no base offset)
  Force close: 16:00 ET

Sweep:
  ymult    in {1.5, 2.0, 2.5}     (yellow ratchet stop tightness)
  tp_mult  in {1.0, 1.25, ..., 2.5} (fixed TP distance, ATR multiple)

Filters / sorts for low MDD (prop-firm friendly).
"""
from __future__ import annotations

import datetime as dt
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from range_break_entry_summary import apply_filters
from test_pure_ratchet_exits import build_20min_bars, FORCE_CLOSE_TIME

PARQUET_DIR  = Path(__file__).parent / "parquets"
TRADELOG_DIR = Path(__file__).parent.parent / "tradelogs"
OUT_TXT      = TRADELOG_DIR / "robust_configs" / "ratchet_sl_fixed_tp_sweep.txt"

VARIANT, X, N, D, STRICT, BAND_K = "B2", 0.75, 15, 70, True, 0.25
CONF_N, CONF_D = 5, 75

YMULT_GRID   = [1.5, 2.0, 2.5]
TPMULT_GRID  = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]

# Prop-firm friendly DD targets (NQ pts; $20/pt)
PROP_DD_TARGETS = {
    "$5K  / 250 pts ": 250,
    "$7.5K / 375 pts": 375,
    "$10K / 500 pts ": 500,
    "$15K / 750 pts ": 750,
}


def filter_pre_dedupe(df):
    f = apply_filters(df, VARIANT, X, N, D, STRICT, BAND_K)
    col = f"conf_delta_half_w{CONF_N}"
    cf = f[((f["direction"]=="LONG")  & (f[col].notna()) & (f[col] >=  CONF_D)) |
           ((f["direction"]=="SHORT") & (f[col].notna()) & (f[col] <= -CONF_D))].copy()
    if cf.empty: return cf
    cf["entry_time_et"] = pd.to_datetime(cf["entry_time"])
    if cf["entry_time_et"].dt.tz is None:
        cf["entry_time_et"] = cf["entry_time_et"].dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    else:
        cf["entry_time_et"] = cf["entry_time_et"].dt.tz_convert("America/New_York")
    return cf.sort_values("entry_time_et").reset_index(drop=True)


def simulate_exit(direction, entry_ts, entry_price, bars20, ymult, tp_mult):
    sign = 1 if direction == "LONG" else -1
    bars_idx = bars20.index
    start = bars_idx.searchsorted(entry_ts, side="right")
    if start >= len(bars_idx): return None
    ent_date = entry_ts.date()
    end = start
    while end < len(bars_idx) and bars_idx[end].date() == ent_date:
        end += 1
    if end == start: return None
    init_idx = start - 1
    if init_idx < 0 or np.isnan(bars20["atr_y"].iloc[init_idx]): return None
    init_atr_y = float(bars20["atr_y"].iloc[init_idx])
    yellow_val = entry_price - sign * ymult * init_atr_y
    prev_yellow = yellow_val

    o = bars20["open"].values[start:end]
    h = bars20["high"].values[start:end]
    l = bars20["low"].values[start:end]
    c = bars20["close"].values[start:end]
    ay = bars20["atr_y"].values[start:end]
    ag = bars20["atr_g"].values[start:end]
    n = end - start

    for i in range(n):
        bar_close_ts = bars_idx[start + i] + pd.Timedelta(minutes=20)

        # ratchet yellow
        if not np.isnan(ay[i]):
            raw_yellow = c[i] - sign * ymult * ay[i]
            yellow_val = max(prev_yellow, raw_yellow) if sign > 0 \
                          else min(prev_yellow, raw_yellow)

        # fixed TP based on initial entry-bar ATR (or rolling — use entry-bar for stability)
        green_val = entry_price + sign * tp_mult * init_atr_y

        # 1) TP fixed
        if sign > 0 and h[i] >= green_val:
            return (sign * (green_val - entry_price), "TP_FIXED",
                    bar_close_ts, i + 1)
        if sign < 0 and l[i] <= green_val:
            return (sign * (green_val - entry_price), "TP_FIXED",
                    bar_close_ts, i + 1)
        # 2) SL ratchet (bearish-close-below for LONG; bullish-close-above for SHORT)
        if sign > 0 and c[i] <= yellow_val and c[i] < o[i]:
            return (c[i] - entry_price, "SL_YELLOW", bar_close_ts, i + 1)
        if sign < 0 and c[i] >= yellow_val and c[i] > o[i]:
            return (entry_price - c[i], "SL_YELLOW", bar_close_ts, i + 1)
        # 3) Force close at 16:00
        if bars_idx[start + i].time() >= FORCE_CLOSE_TIME:
            return (sign * (c[i] - entry_price), "FORCE_CLOSE",
                    bar_close_ts, i + 1)

        prev_yellow = yellow_val

    return (sign * (c[-1] - entry_price), "EOD",
            bars_idx[end - 1] + pd.Timedelta(minutes=20), n)


def chained_dedupe(candidates, exits):
    keep_idx = []; out = []
    last_exit = pd.Timestamp(0, tz="America/New_York")
    for i, row in candidates.iterrows():
        ex = exits[i]
        if ex is None: continue
        pnl, reason, exit_ts, bars_held = ex
        if row["entry_time_et"] > last_exit:
            keep_idx.append(i)
            out.append({"pnl": pnl, "reason": reason, "exit_ts": exit_ts, "bars_held": bars_held})
            last_exit = exit_ts
    return candidates.loc[keep_idx].copy().reset_index(drop=True), out


def stats_curve(df, exits):
    n = len(df)
    if n == 0:
        return {"n":0,"total":0,"pf":0,"sharpe":0,"wr":0,"max_dd":0,
                "n_long":0,"n_short":0,"long_total":0,"short_total":0,
                "tp_n":0,"sl_n":0,"fc_n":0}
    pnl = np.array([e["pnl"] for e in exits])
    pos = pnl[pnl > 0].sum(); neg = -pnl[pnl < 0].sum()
    pf = pos/neg if neg > 0 else (np.inf if pos > 0 else 0)
    daily = pd.Series(pnl, index=pd.to_datetime(df["date"]).dt.date.values).groupby(level=0).sum()
    sharpe = daily.mean()/daily.std()*np.sqrt(252) if daily.std() > 0 else 0
    eq = np.cumsum(pnl); peak = np.maximum.accumulate(eq); max_dd = (eq - peak).min()
    long_mask = (df["direction"]=="LONG").values
    reasons = [e["reason"] for e in exits]
    return {"n":n, "total":pnl.sum(), "pf":pf, "sharpe":sharpe,
            "wr":(pnl>0).mean(), "max_dd":max_dd,
            "n_long":int(long_mask.sum()), "n_short":int((~long_mask).sum()),
            "long_total":pnl[long_mask].sum(), "short_total":pnl[~long_mask].sum(),
            "tp_n": reasons.count("TP_FIXED"),
            "sl_n": reasons.count("SL_YELLOW"),
            "fc_n": reasons.count("FORCE_CLOSE")}


def evaluate(trades_parquet, bars20, ymult, tp_mult):
    cands = filter_pre_dedupe(trades_parquet)
    if cands.empty: return None
    exits = [simulate_exit(r["direction"], r["entry_time_et"], float(r["entry_price"]),
                            bars20, ymult, tp_mult) for _, r in cands.iterrows()]
    kept, kept_exits = chained_dedupe(cands, exits)
    return stats_curve(kept, kept_exits)


def main():
    print("loading 20-min bars + filtering trades...")
    bars20 = build_20min_bars()
    is_df  = pd.read_parquet(PARQUET_DIR / "entry_signal_trades.parquet")
    oos_df = pd.read_parquet(PARQUET_DIR / "entry_signal_trades_oos.parquet")

    rows = []
    print(f"sweeping {len(YMULT_GRID)*len(TPMULT_GRID)} combos...")
    for ym, tpm in product(YMULT_GRID, TPMULT_GRID):
        is_s  = evaluate(is_df,  bars20, ym, tpm)
        oos_s = evaluate(oos_df, bars20, ym, tpm)
        if is_s is None or oos_s is None: continue
        rows.append({
            "ymult":ym, "tp_mult":tpm,
            "is_n":is_s["n"], "is_total":is_s["total"], "is_pf":is_s["pf"],
            "is_sharpe":is_s["sharpe"], "is_wr":is_s["wr"], "is_mdd":is_s["max_dd"],
            "is_tp":is_s["tp_n"], "is_sl":is_s["sl_n"], "is_fc":is_s["fc_n"],
            "oos_n":oos_s["n"], "oos_total":oos_s["total"], "oos_pf":oos_s["pf"],
            "oos_sharpe":oos_s["sharpe"], "oos_wr":oos_s["wr"], "oos_mdd":oos_s["max_dd"],
            "oos_tp":oos_s["tp_n"], "oos_sl":oos_s["sl_n"], "oos_fc":oos_s["fc_n"],
            "is_long_total":is_s["long_total"], "is_short_total":is_s["short_total"],
            "oos_long_total":oos_s["long_total"], "oos_short_total":oos_s["short_total"],
        })

    df = pd.DataFrame(rows)
    df["worst_mdd"]  = df[["is_mdd","oos_mdd"]].abs().max(axis=1)   # max DD across periods
    df["min_pf"]     = df[["is_pf","oos_pf"]].min(axis=1).clip(upper=99)
    df["min_sharpe"] = df[["is_sharpe","oos_sharpe"]].min(axis=1)
    df["combined_total"] = df["is_total"] + df["oos_total"]

    lines = []
    lines.append("=" * 220)
    lines.append("RATCHET SL + FIXED TP SWEEP — realistic dedupe")
    lines.append("=" * 220)
    lines.append("")
    lines.append(f"Entry: B2 X={X} N={N} D={D} strict={STRICT} BAND_K={BAND_K} + conf_N={CONF_N} D={CONF_D} HALF")
    lines.append(f"Yellow SL: pure_ratchet at ymult * ATR_14_20min (sweep ymult={YMULT_GRID})")
    lines.append(f"Green TP:  FIXED at entry + tp_mult * ATR_at_entry (sweep tp_mult={TPMULT_GRID})")
    lines.append(f"Force close: 16:00 ET")
    lines.append("")

    # Full sweep table
    lines.append("=" * 220)
    lines.append("FULL SWEEP — all combos sorted by min Sharpe descending")
    lines.append("=" * 220)
    cols = ["ymult","tp_mult","is_n","is_total","is_pf","is_sharpe","is_wr","is_mdd","is_tp","is_sl","is_fc",
            "oos_n","oos_total","oos_pf","oos_sharpe","oos_wr","oos_mdd","oos_tp","oos_sl","oos_fc",
            "min_pf","min_sharpe","worst_mdd","combined_total"]
    lines.append(df.sort_values("min_sharpe", ascending=False)[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Lowest worst-DD configs (prop-firm friendly)
    lines.append("")
    lines.append("=" * 220)
    lines.append("LOWEST WORST-CASE DRAWDOWN (prop-firm compatible)")
    lines.append("=" * 220)
    cols_dd = ["ymult","tp_mult","is_n","is_total","is_pf","is_sharpe","is_mdd",
               "oos_n","oos_total","oos_pf","oos_sharpe","oos_mdd","worst_mdd","combined_total"]
    lines.append(df.sort_values("worst_mdd")[cols_dd].head(15).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Bucket by prop-firm DD targets
    lines.append("")
    lines.append("=" * 220)
    lines.append("CONFIGS BY PROP-FIRM DD CAP (must hold both IS and OOS DD under the cap)")
    lines.append("=" * 220)
    for label, cap_pts in PROP_DD_TARGETS.items():
        sub = df[df["worst_mdd"] <= cap_pts]
        sub = sub[(sub["is_total"] > 0) & (sub["oos_total"] > 0)]
        sub = sub.sort_values("min_sharpe", ascending=False)
        lines.append(f"\n--- DD cap: {label}  (worst_mdd <= {cap_pts} pts AND positive both IS+OOS) ---")
        if sub.empty:
            lines.append("  NONE — no config keeps DD under this cap with positive both periods")
            continue
        lines.append(f"  {len(sub)} configs:")
        cols_show = ["ymult","tp_mult","is_n","is_total","is_pf","is_sharpe","is_mdd",
                     "oos_n","oos_total","oos_pf","oos_sharpe","oos_mdd","worst_mdd","min_pf","min_sharpe"]
        lines.append(sub[cols_show].head(10).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
