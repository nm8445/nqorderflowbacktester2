"""For the realistic-dedupe pure_ratchet config, report:
  - Yellow-stop loss/win profile (are most yellow stops losses?)
  - Max DD of the realistic equity curve (IS, OOS, combined)
  - Per hour-of-entry PF breakdown (optimal entry hours)
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
OUT_TXT      = TRADELOG_DIR / "robust_configs" / "ratchet_hourly_and_dd.txt"

VARIANT, X, N, D, STRICT, BAND_K = "B2", 0.75, 15, 70, True, 0.25
CONF_N, CONF_D = 5, 75
YMULT, GMULT, GBASE, GDECAY = 2.50, 2.00, 200.0, 0.0

BUCKETS = [
    ("09:30-10:00",  9*60+30, 10*60),
    ("10:00-11:00", 10*60,    11*60),
    ("11:00-12:00", 11*60,    12*60),
    ("12:00-13:00", 12*60,    13*60),
    ("13:00-14:00", 13*60,    14*60),
    ("14:00-15:00", 14*60,    15*60),
    ("15:00-16:00", 15*60,    16*60),
    ("16:00+",      16*60,    24*60),
]


def filter_pre_dedupe(df: pd.DataFrame) -> pd.DataFrame:
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
    cf["entry_minute"] = cf["entry_time_et"].dt.hour * 60 + cf["entry_time_et"].dt.minute
    return cf.sort_values("entry_time_et").reset_index(drop=True)


def simulate_ratchet(direction, entry_ts, entry_price, bars20):
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
    kept = candidates.loc[keep_idx].copy().reset_index(drop=True)
    return kept, out


def stats(df, pnl_col):
    if df.empty:
        return {"n":0,"total":0,"pf":0,"sharpe":0,"wr":0,"max_dd":0,
                "n_long":0,"n_short":0,"long_total":0,"short_total":0}
    pnl = df[pnl_col].values
    pos = pnl[pnl > 0].sum(); neg = -pnl[pnl < 0].sum()
    pf = pos/neg if neg > 0 else (np.inf if pos > 0 else 0)
    daily = pd.Series(pnl, index=df["date"].values).groupby(level=0).sum()
    sharpe = daily.mean()/daily.std()*np.sqrt(252) if daily.std() > 0 else 0
    eq = np.cumsum(pnl); peak = np.maximum.accumulate(eq); max_dd = (eq - peak).min()
    long_mask = (df["direction"]=="LONG").values
    return {"n":len(df), "total":pnl.sum(), "pf":pf, "sharpe":sharpe,
            "wr":(pnl>0).mean(), "max_dd":max_dd,
            "n_long":int(long_mask.sum()), "n_short":int((~long_mask).sum()),
            "long_total":pnl[long_mask].sum(), "short_total":pnl[~long_mask].sum()}


def fmt(label, s):
    pf = f"{s['pf']:>5.2f}" if np.isfinite(s["pf"]) else "  inf"
    return (f"  {label:<24}  n={s['n']:>4}  total={s['total']:>+8.1f}  "
            f"WR={s['wr']:>5.1%}  PF={pf}  Sh={s['sharpe']:>+5.2f}  MDD={s['max_dd']:>+7.0f}")


def get_kept(trades_parquet, bars20):
    cands = filter_pre_dedupe(trades_parquet)
    cands["date"] = pd.to_datetime(cands["date"]).dt.date
    exits = [simulate_ratchet(r["direction"], r["entry_time_et"], float(r["entry_price"]), bars20)
             for _, r in cands.iterrows()]
    kept, out = chained_dedupe(cands, exits)
    kept["pnl_ratchet"] = [e["pnl"] for e in out]
    kept["exit_reason"] = [e["reason"] for e in out]
    kept["bars_held"]   = [e["bars_held"] for e in out]
    return kept


def main():
    print("loading 20-min bars + filtering trades...")
    bars20 = build_20min_bars()
    is_df  = pd.read_parquet(PARQUET_DIR / "entry_signal_trades.parquet")
    oos_df = pd.read_parquet(PARQUET_DIR / "entry_signal_trades_oos.parquet")
    is_kept  = get_kept(is_df,  bars20)
    oos_kept = get_kept(oos_df, bars20)
    combined = pd.concat([is_kept, oos_kept], ignore_index=True).reset_index(drop=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.date

    lines = []
    lines.append("=" * 200)
    lines.append("PURE_RATCHET (REAL DEDUPE) — yellow-stop profile, max DD, hourly PF")
    lines.append("=" * 200)
    lines.append("")
    lines.append(f"Entry: B2 X={X} N={N} D={D} strict={STRICT} BAND_K={BAND_K} + conf_N={CONF_N} D={CONF_D} HALF")
    lines.append(f"Exit:  ymult={YMULT} gmult={GMULT} gbase={GBASE} gdecay={GDECAY} (20-min bars)")
    lines.append(f"Trade counts: IS={len(is_kept)}  OOS={len(oos_kept)}  Combined={len(combined)}")
    lines.append("")

    # ---- 1) Yellow-stop profile ----
    lines.append("=" * 200)
    lines.append("YELLOW-STOP PROFILE (are most yellow stops losses?)")
    lines.append("=" * 200)
    for label, ded in [("IS", is_kept), ("OOS", oos_kept), ("COMBINED", combined)]:
        sl = ded[ded["exit_reason"] == "SL_YELLOW"]
        if sl.empty:
            lines.append(f"\n{label}: no SL_YELLOW exits"); continue
        n_loss = (sl["pnl_ratchet"] < 0).sum()
        n_win  = (sl["pnl_ratchet"] > 0).sum()
        n_be   = (sl["pnl_ratchet"] == 0).sum()
        lines.append(f"\n{label}  total SL_YELLOW: {len(sl)}")
        lines.append(f"  losers  (pnl<0):     n={n_loss:>4}  total={sl[sl['pnl_ratchet']<0]['pnl_ratchet'].sum():+.1f}  mean={sl[sl['pnl_ratchet']<0]['pnl_ratchet'].mean():+.1f}")
        lines.append(f"  winners (pnl>0):     n={n_win:>4}  total={sl[sl['pnl_ratchet']>0]['pnl_ratchet'].sum():+.1f}  mean={(sl[sl['pnl_ratchet']>0]['pnl_ratchet'].mean() if n_win else 0):+.1f}")
        lines.append(f"  break-even (pnl=0):  n={n_be:>4}")
        lines.append(f"  -> yellow stops are losses {n_loss/len(sl)*100:.1f}% of the time, profitable {n_win/len(sl)*100:.1f}%")

    # ---- 2) Max DD ----
    lines.append("")
    lines.append("=" * 200)
    lines.append("MAX DRAWDOWN")
    lines.append("=" * 200)
    s_is  = stats(is_kept,  "pnl_ratchet")
    s_oos = stats(oos_kept, "pnl_ratchet")
    s_com = stats(combined, "pnl_ratchet")
    lines.append(fmt("IS",       s_is))
    lines.append(fmt("OOS",      s_oos))
    lines.append(fmt("COMBINED", s_com))

    # ---- 3) Hourly entry-bucket PF ----
    lines.append("")
    lines.append("=" * 200)
    lines.append("HOURLY ENTRY BUCKET PF — entry_time ET")
    lines.append("=" * 200)
    for label, ded in [("IS", is_kept), ("OOS", oos_kept), ("COMBINED", combined)]:
        lines.append(f"\n--- {label} ---  (overall PF: {stats(ded,'pnl_ratchet')['pf']:.2f})")
        lines.append(f"  {'bucket':<14}  {'n':>4}  {'L':>3}/{'S':>3}  {'total':>9}  {'WR':>5}  {'PF':>5}  {'Sharpe':>7}  {'L_t':>7}  {'S_t':>7}")
        lines.append(f"  {'-'*100}")
        for name, lo, hi in BUCKETS:
            sub = ded[(ded["entry_minute"] >= lo) & (ded["entry_minute"] < hi)]
            if sub.empty:
                lines.append(f"  {name:<14}  EMPTY"); continue
            s = stats(sub, "pnl_ratchet")
            pf_str = f"{s['pf']:>5.2f}" if np.isfinite(s['pf']) else "  inf"
            lines.append(f"  {name:<14}  {s['n']:>4}  {s['n_long']:>3}/{s['n_short']:>3}  "
                         f"{s['total']:>+9.1f}  {s['wr']:>5.1%}  {pf_str}  "
                         f"{s['sharpe']:>+7.2f}  {s['long_total']:>+7.1f}  {s['short_total']:>+7.1f}")

    # ---- 4) Buckets that are losing money ----
    lines.append("")
    lines.append("=" * 200)
    lines.append("WEAK / LOSING HOUR BUCKETS")
    lines.append("=" * 200)
    for label, ded in [("IS", is_kept), ("OOS", oos_kept), ("COMBINED", combined)]:
        ov = stats(ded, "pnl_ratchet")
        lines.append(f"\n{label} overall PF: {ov['pf']:.2f}")
        for name, lo, hi in BUCKETS:
            sub = ded[(ded["entry_minute"] >= lo) & (ded["entry_minute"] < hi)]
            if len(sub) < 5: continue
            s = stats(sub, "pnl_ratchet")
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
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
