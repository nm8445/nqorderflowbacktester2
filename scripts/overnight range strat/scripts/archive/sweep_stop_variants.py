"""Sweep stop variants with realistic dedupe (chained Mode 1 by actual exit times).

Three regimes tested, each with the optimal green-band setup:
  A) FIXED stop:      stop set at entry - X*ATR (signed), never moves
  B) PURE_RATCHET:    stop trails (current logic) at the same X grid
  C) NO STOP:         only green TP + force close at 16:00

Green-band held constant at: gmult=2.0  gbase=200  gdecay=0  (the optimal from prior sweep)
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from range_break_entry_summary import apply_filters
from test_pure_ratchet_exits import build_20min_bars, FORCE_CLOSE_TIME, RED_INTERCEPT, RED_DRIFT

PARQUET_DIR  = Path(__file__).parent / "parquets"
TRADELOG_DIR = Path(__file__).parent.parent / "tradelogs"
OUT_TXT      = TRADELOG_DIR / "robust_configs" / "stop_variants_sweep.txt"

VARIANT, X, N, D, STRICT, BAND_K = "B2", 0.75, 15, 70, True, 0.25
CONF_N, CONF_D = 5, 75
GMULT, GBASE, GDECAY = 2.00, 200.0, 0.0

YMULT_GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]


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


def simulate_exit(direction, entry_ts, entry_price, bars20, ymult, mode):
    """mode in {'fixed', 'ratchet', 'no_sl'}"""
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

    fixed_yellow = entry_price - sign * ymult * init_atr_y
    yellow_val = fixed_yellow
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

        if mode == "ratchet":
            if not np.isnan(ay[i]):
                raw_yellow = c[i] - sign * ymult * ay[i]
                yellow_val = max(prev_yellow, raw_yellow) if sign > 0 \
                              else min(prev_yellow, raw_yellow)
        elif mode == "fixed":
            yellow_val = fixed_yellow

        red_val = entry_price + sign * (RED_INTERCEPT + RED_DRIFT * bars_in_trade)
        green_offset = (GBASE - GDECAY * bars_in_trade
                        + (GMULT * ag[i] if not np.isnan(ag[i]) else 0.0))
        green_val = red_val + sign * green_offset

        # 1) TP green
        if sign > 0 and h[i] >= green_val:
            return (c[i] - entry_price, "TP_GREEN", bar_close_ts, bars_in_trade)
        if sign < 0 and l[i] <= green_val:
            return (entry_price - c[i], "TP_GREEN", bar_close_ts, bars_in_trade)
        # 2) SL yellow (skip in no_sl mode)
        if mode != "no_sl":
            if sign > 0 and c[i] <= yellow_val and c[i] < o[i]:
                return (c[i] - entry_price, "SL_YELLOW", bar_close_ts, bars_in_trade)
            if sign < 0 and c[i] >= yellow_val and c[i] > o[i]:
                return (entry_price - c[i], "SL_YELLOW", bar_close_ts, bars_in_trade)
        # 3) Force close
        if bars_idx[start + i].time() >= FORCE_CLOSE_TIME:
            return (sign * (c[i] - entry_price), "FORCE_CLOSE", bar_close_ts, bars_in_trade)

        if mode == "ratchet":
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


def stats(df, exits):
    n = len(df)
    if n == 0:
        return {"n":0,"total":0,"pf":0,"sharpe":0,"wr":0,"max_dd":0}
    pnl = np.array([e["pnl"] for e in exits])
    pos = pnl[pnl > 0].sum(); neg = -pnl[pnl < 0].sum()
    pf = pos/neg if neg > 0 else (np.inf if pos > 0 else 0)
    daily = pd.Series(pnl, index=pd.to_datetime(df["date"]).dt.date.values).groupby(level=0).sum()
    sharpe = daily.mean()/daily.std()*np.sqrt(252) if daily.std() > 0 else 0
    eq = np.cumsum(pnl); peak = np.maximum.accumulate(eq); max_dd = (eq - peak).min()
    return {"n":n, "total":pnl.sum(), "pf":pf, "sharpe":sharpe,
            "wr":(pnl>0).mean(), "max_dd":max_dd}


def evaluate(trades_parquet, bars20, ymult, mode):
    cands = filter_pre_dedupe(trades_parquet)
    if cands.empty: return None
    exits = [simulate_exit(r["direction"], r["entry_time_et"], float(r["entry_price"]),
                            bars20, ymult, mode) for _, r in cands.iterrows()]
    kept, kept_exits = chained_dedupe(cands, exits)
    return stats(kept, kept_exits)


def fmt_row(label, s):
    pf = f"{s['pf']:>5.2f}" if np.isfinite(s["pf"]) else "  inf"
    return (f"  {label:<22}  n={s['n']:>4}  total={s['total']:>+8.1f}  "
            f"WR={s['wr']:>5.1%}  PF={pf}  Sh={s['sharpe']:>+5.2f}  MDD={s['max_dd']:>+7.0f}")


def main():
    print("loading 20-min bars...")
    bars20 = build_20min_bars()
    is_df  = pd.read_parquet(PARQUET_DIR / "entry_signal_trades.parquet")
    oos_df = pd.read_parquet(PARQUET_DIR / "entry_signal_trades_oos.parquet")

    lines = []
    lines.append("=" * 200)
    lines.append("STOP VARIANTS SWEEP — realistic dedupe by actual exit times")
    lines.append("=" * 200)
    lines.append("")
    lines.append(f"Entry: B2 X={X} N={N} D={D} strict={STRICT} BAND_K={BAND_K} + conf_N={CONF_N} D={CONF_D} HALF")
    lines.append(f"Green band held constant: gmult={GMULT}  gbase={GBASE}  gdecay={GDECAY}")
    lines.append(f"Force close: 16:00 ET    Yellow ATR length: 14 (20-min bars)")
    lines.append("")

    # --- Variant A: FIXED stop ---
    lines.append("=" * 200)
    lines.append("A) FIXED STOP  (yellow set once at entry-X*ATR, never moves)")
    lines.append("=" * 200)
    lines.append(f"  {'ymult':<6}  {'IS_n':>5}  {'IS_total':>10}  {'IS_PF':>6}  {'IS_Sh':>6}  {'IS_MDD':>8}    {'OOS_n':>5}  {'OOS_total':>10}  {'OOS_PF':>6}  {'OOS_Sh':>6}  {'OOS_MDD':>8}")
    lines.append("  " + "-" * 130)
    for ym in YMULT_GRID:
        is_s  = evaluate(is_df,  bars20, ym, "fixed")
        oos_s = evaluate(oos_df, bars20, ym, "fixed")
        if is_s is None or oos_s is None: continue
        is_pf  = f"{is_s['pf']:>6.2f}"  if np.isfinite(is_s['pf']) else "   inf"
        oos_pf = f"{oos_s['pf']:>6.2f}" if np.isfinite(oos_s['pf']) else "   inf"
        lines.append(f"  {ym:<6.2f}  {is_s['n']:>5}  {is_s['total']:>+10.1f}  {is_pf}  {is_s['sharpe']:>+6.2f}  {is_s['max_dd']:>+8.0f}    "
                     f"{oos_s['n']:>5}  {oos_s['total']:>+10.1f}  {oos_pf}  {oos_s['sharpe']:>+6.2f}  {oos_s['max_dd']:>+8.0f}")

    # --- Variant B: PURE_RATCHET ---
    lines.append("")
    lines.append("=" * 200)
    lines.append("B) PURE_RATCHET STOP  (yellow trails up; never moves against position)")
    lines.append("=" * 200)
    lines.append(f"  {'ymult':<6}  {'IS_n':>5}  {'IS_total':>10}  {'IS_PF':>6}  {'IS_Sh':>6}  {'IS_MDD':>8}    {'OOS_n':>5}  {'OOS_total':>10}  {'OOS_PF':>6}  {'OOS_Sh':>6}  {'OOS_MDD':>8}")
    lines.append("  " + "-" * 130)
    for ym in YMULT_GRID:
        is_s  = evaluate(is_df,  bars20, ym, "ratchet")
        oos_s = evaluate(oos_df, bars20, ym, "ratchet")
        if is_s is None or oos_s is None: continue
        is_pf  = f"{is_s['pf']:>6.2f}"  if np.isfinite(is_s['pf']) else "   inf"
        oos_pf = f"{oos_s['pf']:>6.2f}" if np.isfinite(oos_s['pf']) else "   inf"
        lines.append(f"  {ym:<6.2f}  {is_s['n']:>5}  {is_s['total']:>+10.1f}  {is_pf}  {is_s['sharpe']:>+6.2f}  {is_s['max_dd']:>+8.0f}    "
                     f"{oos_s['n']:>5}  {oos_s['total']:>+10.1f}  {oos_pf}  {oos_s['sharpe']:>+6.2f}  {oos_s['max_dd']:>+8.0f}")

    # --- Variant C: NO STOP ---
    lines.append("")
    lines.append("=" * 200)
    lines.append("C) NO STOP  (only green TP + force close at 16:00)")
    lines.append("=" * 200)
    is_s  = evaluate(is_df,  bars20, 0.0, "no_sl")
    oos_s = evaluate(oos_df, bars20, 0.0, "no_sl")
    is_pf  = f"{is_s['pf']:>6.2f}"  if np.isfinite(is_s['pf']) else "   inf"
    oos_pf = f"{oos_s['pf']:>6.2f}" if np.isfinite(oos_s['pf']) else "   inf"
    lines.append(f"  IS:  n={is_s['n']:>4}  total={is_s['total']:>+9.1f}  PF={is_pf}  Sharpe={is_s['sharpe']:>+5.2f}  WR={is_s['wr']:.1%}  MDD={is_s['max_dd']:+.0f}")
    lines.append(f"  OOS: n={oos_s['n']:>4}  total={oos_s['total']:>+9.1f}  PF={oos_pf}  Sharpe={oos_s['sharpe']:>+5.2f}  WR={oos_s['wr']:.1%}  MDD={oos_s['max_dd']:+.0f}")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
