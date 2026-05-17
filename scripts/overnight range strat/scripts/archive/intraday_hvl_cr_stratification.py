"""Intraday HVL / CR stratification for the locked B2 config.

Tests three hypotheses:
  H1: Both longs and shorts work better in NEG gamma regimes (intraday gamma_sign).
  H2: Longs taken ABOVE HVL have lower hit rate than longs BELOW HVL.
  H3: Longs above HVL only work when ALSO above CR.

And tests the proposed alternative:
  - When entry is above HVL intraday, take SHORT instead of LONG (mean-reversion),
    unless price has cleared CR (then keep LONG).

Trade -> intraday snapshot lookup:
  For each trade, find the most recent intraday snapshot at-or-before entry_time
  (e.g., entry at 10:05 -> use 09:50 snapshot; entry at 10:35 -> use 10:20).

Locked config: B2 X=1.25 N=5 D=70 STRICT=True BAND_K=0.25 TP=SL=1.0 chained Mode 1.
"""

from __future__ import annotations

import datetime as dt
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
TRADELOG_DIR.mkdir(exist_ok=True)
TRADES   = PARQUET_DIR / "entry_signal_trades.parquet"
LEVELS   = Path("D:/trading_pythonbacktest_data/qqq_intraday_levels.parquet")
EOD_MQ   = Path("D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet")
OUT_TXT  = TRADELOG_DIR / "intraday_hvl_cr_stratification.txt"

VARIANT, X, N, D, STRICT, BAND_K = "B2", 1.25, 5, 70, True, 0.25
TP_M, SL_M = 1.0, 1.0

# Intraday MenthorQ snapshots sorted ascending
SNAP_TIMES = ["08:00", "09:00", "09:30", "09:50", "10:20", "10:50",
              "11:20", "11:50", "12:20", "12:50", "13:20", "13:50",
              "14:20", "14:50", "15:20", "15:50"]
SNAP_MINS = [int(s[:2]) * 60 + int(s[3:]) for s in SNAP_TIMES]


def _snap_for_entry(entry_time: pd.Timestamp) -> str:
    """Return the most recent intraday snapshot label at-or-before entry_time."""
    if pd.isna(entry_time):
        return SNAP_TIMES[-1]
    if entry_time.tz is not None:
        et = entry_time.tz_convert("America/New_York")
    else:
        et = entry_time
    em = et.hour * 60 + et.minute
    if em < SNAP_MINS[0]:
        return SNAP_TIMES[0]
    pick = SNAP_TIMES[0]
    for lab, m in zip(SNAP_TIMES, SNAP_MINS):
        if m <= em:
            pick = lab
        else:
            break
    return pick


def _stats(label: str, sub: pd.DataFrame, lines: list):
    if sub.empty:
        lines.append(f"  {label:<28}  EMPTY")
        return
    pnl = sub["pnl"].values
    long_mask  = (sub["direction"] == "LONG").values
    short_mask = (sub["direction"] == "SHORT").values
    wins = pnl > 0
    pos_pnl = pnl[pnl > 0].sum(); neg_pnl = -pnl[pnl < 0].sum()
    pf = pos_pnl / neg_pnl if neg_pnl > 0 else (np.inf if pos_pnl > 0 else 0.0)
    daily = pd.Series(pnl, index=sub["date"].values).groupby(level=0).sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0
    eq = np.cumsum(pnl); peak = np.maximum.accumulate(eq); max_dd = (eq - peak).min()
    wr_l = wins[long_mask].mean()  if long_mask.any()  else float("nan")
    wr_s = wins[short_mask].mean() if short_mask.any() else float("nan")
    lines.append(f"  {label:<28}  n={len(sub):>4}"
                 f"  L={int(long_mask.sum()):>4}/S={int(short_mask.sum()):>4}"
                 f"  total={pnl.sum():>+8.1f}  mean={pnl.mean():>+5.2f}"
                 f"  WR={wins.mean():>5.1%}  WR_L={wr_l:>5.1%}  WR_S={wr_s:>5.1%}"
                 f"  PF={min(pf,999.0):>5.2f}  Sharpe={sharpe:>+5.2f}  MDD={max_dd:>+8.0f}")


def main():
    print(f"loading trades + applying locked config "
          f"(B2 X={X} N={N} D={D} strict={STRICT} BAND_K={BAND_K})...")
    df = pd.read_parquet(TRADES)
    filtered = apply_filters(df, VARIANT, X, N, D, STRICT, BAND_K)
    deduped  = mode1_chained_dedupe(filtered, TP_M, SL_M)
    deduped["pnl"] = trade_pnls_vectorized(deduped, TP_M, SL_M)
    deduped["date"] = pd.to_datetime(deduped["date"]).dt.date
    print(f"  {len(deduped)} trades (locked config, chained Mode 1)")

    print(f"loading intraday levels...")
    lv = pd.read_parquet(LEVELS)
    lv["date"] = pd.to_datetime(lv["date"]).dt.date
    lv_idx = lv.set_index(["date", "snapshot_label"])
    print(f"  {len(lv):,} rows over {lv['date'].nunique()} dates")

    # Compute snapshot label per trade
    deduped["entry_time_et"] = pd.to_datetime(deduped["entry_time"])
    if deduped["entry_time_et"].dt.tz is None:
        deduped["entry_time_et"] = deduped["entry_time_et"].dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    else:
        deduped["entry_time_et"] = deduped["entry_time_et"].dt.tz_convert("America/New_York")
    deduped["snap"] = deduped["entry_time_et"].apply(_snap_for_entry)

    # Look up HVL / CR / PS / gamma_sign / spot from the intraday rollup
    cols_map = ["hvl_nq", "hvl_extended_nq", "cr_nq", "ps_nq",
                "gamma_sign", "hvl_source", "spot_nq"]
    lookups = {c: [] for c in cols_map}
    for _, t in deduped.iterrows():
        key = (t["date"], t["snap"])
        if key in lv_idx.index:
            row = lv_idx.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for c in cols_map:
                lookups[c].append(row.get(c, np.nan))
        else:
            for c in cols_map:
                lookups[c].append(np.nan)
    for c in cols_map:
        deduped[c] = lookups[c]

    n_with_hvl = deduped["hvl_nq"].notna().sum()
    n_with_cr  = deduped["cr_nq"].notna().sum()
    n_with_g   = deduped["gamma_sign"].notna().sum()
    print(f"  trades with intraday hvl_nq  : {n_with_hvl}/{len(deduped)} ({n_with_hvl/len(deduped)*100:.1f}%)")
    print(f"  trades with intraday cr_nq   : {n_with_cr}/{len(deduped)} ({n_with_cr/len(deduped)*100:.1f}%)")
    print(f"  trades with intraday gamma   : {n_with_g}/{len(deduped)} ({n_with_g/len(deduped)*100:.1f}%)")

    # Also load prior-day EOD multi-DTE gamma_sign (the regime classifier from previous study)
    print(f"loading EOD multi-DTE gamma_sign from menthorq_levels_nq.parquet...")
    eod = pd.read_parquet(EOD_MQ)
    eod["date"] = pd.to_datetime(eod["date"]).dt.date
    eod = eod.set_index("date")
    eod_dates = sorted(eod.index.tolist())
    def _prior_mq(d):
        prev = None
        for md in eod_dates:
            if md < d: prev = md
            else: break
        return prev
    eod_gamma_lookup = {}
    for d in deduped["date"].unique():
        p = _prior_mq(d)
        if p is None: continue
        eod_gamma_lookup[d] = eod.loc[p, "qqq_gamma_sign"] if "qqq_gamma_sign" in eod.columns else np.nan
    deduped["eod_gamma_sign"] = deduped["date"].map(eod_gamma_lookup)
    n_with_eod_g = deduped["eod_gamma_sign"].notna().sum()
    print(f"  trades with EOD multi-DTE gamma: {n_with_eod_g}/{len(deduped)} ({n_with_eod_g/len(deduped)*100:.1f}%)")

    # Position relative to levels (NaN-safe boolean: True = above, False = below or NaN-pivot)
    deduped["above_hvl"] = (deduped["entry_price"] > deduped["hvl_nq"]).fillna(False).astype(bool)
    deduped["above_cr"]  = (deduped["entry_price"] > deduped["cr_nq"] ).fillna(False).astype(bool)
    # below = strictly below; NaN level => excluded from below-filter via has_hvl
    deduped["below_hvl"] = (deduped["entry_price"] < deduped["hvl_nq"]).fillna(False).astype(bool)
    deduped["below_cr"]  = (deduped["entry_price"] < deduped["cr_nq"] ).fillna(False).astype(bool)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    lines = []
    lines.append("=" * 200)
    lines.append("INTRADAY HVL / CR STRATIFICATION — LOCKED CONFIG")
    lines.append(f"Config: B2 X={X} N={N} D={D} strict={STRICT} BAND_K={BAND_K} TP={TP_M} SL={SL_M} (chained Mode 1)")
    lines.append(f"Trades: {len(deduped)}  date range: {deduped['date'].min()} -> {deduped['date'].max()}")
    lines.append(f"Lookup: intraday snapshot at-or-before entry_time")
    lines.append("=" * 200)

    # ----- BASELINE
    lines.append("")
    lines.append("BASELINE — all trades:")
    _stats("ALL TRADES", deduped, lines)
    _stats("  -> LONG",  deduped[deduped["direction"]=="LONG"], lines)
    _stats("  -> SHORT", deduped[deduped["direction"]=="SHORT"], lines)

    # ----- H1a: Gamma regime — INTRADAY 0-1 DTE chain
    lines.append("")
    lines.append("=" * 200)
    lines.append("H1a: Gamma regime — INTRADAY 0-1 DTE chain (snapshot at-or-before entry)")
    lines.append("=" * 200)
    pos = deduped[deduped["gamma_sign"] == 1]
    neg = deduped[deduped["gamma_sign"] == -1]
    _stats("POS gamma (intraday, sign=+1)",  pos, lines)
    _stats("  -> LONG",            pos[pos["direction"]=="LONG"], lines)
    _stats("  -> SHORT",           pos[pos["direction"]=="SHORT"], lines)
    _stats("NEG gamma (intraday, sign=-1)",  neg, lines)
    _stats("  -> LONG",            neg[neg["direction"]=="LONG"], lines)
    _stats("  -> SHORT",           neg[neg["direction"]=="SHORT"], lines)

    # ----- H1b: Gamma regime — PRIOR-DAY EOD MULTI-DTE chain (matches previous study)
    lines.append("")
    lines.append("=" * 200)
    lines.append("H1b: Gamma regime — PRIOR-DAY EOD MULTI-DTE (0-45 DTE) chain")
    lines.append("    (this is what hvl_stratification.txt used; broader dealer positioning)")
    lines.append("=" * 200)
    pos_e = deduped[deduped["eod_gamma_sign"] == 1]
    neg_e = deduped[deduped["eod_gamma_sign"] == -1]
    _stats("POS gamma (EOD MDTE, sign=+1)",  pos_e, lines)
    _stats("  -> LONG",                       pos_e[pos_e["direction"]=="LONG"], lines)
    _stats("  -> SHORT",                      pos_e[pos_e["direction"]=="SHORT"], lines)
    _stats("NEG gamma (EOD MDTE, sign=-1)",  neg_e, lines)
    _stats("  -> LONG",                       neg_e[neg_e["direction"]=="LONG"], lines)
    _stats("  -> SHORT",                      neg_e[neg_e["direction"]=="SHORT"], lines)

    # ----- Agreement check between intraday and EOD gamma_signs
    lines.append("")
    both_known = deduped["gamma_sign"].notna() & deduped["eod_gamma_sign"].notna()
    if both_known.any():
        d2 = deduped[both_known]
        agree = (d2["gamma_sign"] == d2["eod_gamma_sign"]).sum()
        total = len(d2)
        lines.append(f"  Intraday vs EOD MDTE gamma agreement: {agree}/{total} ({agree/total*100:.1f}%)")
        crosstab = pd.crosstab(d2["eod_gamma_sign"], d2["gamma_sign"], margins=True)
        lines.append("  Crosstab (rows=EOD MDTE, cols=intraday):")
        for line in crosstab.to_string().split("\n"):
            lines.append("    " + line)

    # ----- H2: Position vs intraday HVL
    lines.append("")
    lines.append("=" * 200)
    lines.append("H2: Position vs intraday HVL at entry snapshot (gated +/-5%, carry-fwd)")
    lines.append("=" * 200)
    has_hvl = deduped["hvl_nq"].notna()
    above = deduped[has_hvl & deduped["above_hvl"]]
    below = deduped[has_hvl & deduped["below_hvl"]]
    _stats("ABOVE HVL", above, lines)
    _stats("  -> LONG",  above[above["direction"]=="LONG"], lines)
    _stats("  -> SHORT", above[above["direction"]=="SHORT"], lines)
    _stats("BELOW HVL", below, lines)
    _stats("  -> LONG",  below[below["direction"]=="LONG"], lines)
    _stats("  -> SHORT", below[below["direction"]=="SHORT"], lines)

    # ----- H3: Position vs HVL AND CR (compound)
    lines.append("")
    lines.append("=" * 200)
    lines.append("H3: Position vs HVL + CR  (longs above HVL: do they need CR break?)")
    lines.append("=" * 200)
    has_both = deduped["hvl_nq"].notna() & deduped["cr_nq"].notna()
    h_above_cr_above = deduped[has_both & deduped["above_hvl"] & deduped["above_cr"]]
    h_above_cr_below = deduped[has_both & deduped["above_hvl"] & deduped["below_cr"]]
    h_below_cr_above = deduped[has_both & deduped["below_hvl"] & deduped["above_cr"]]
    h_below_cr_below = deduped[has_both & deduped["below_hvl"] & deduped["below_cr"]]
    _stats("ABOVE HVL + ABOVE CR (cleared)", h_above_cr_above, lines)
    _stats("  -> LONG",  h_above_cr_above[h_above_cr_above["direction"]=="LONG"], lines)
    _stats("  -> SHORT", h_above_cr_above[h_above_cr_above["direction"]=="SHORT"], lines)
    _stats("ABOVE HVL + BELOW CR (capped)", h_above_cr_below, lines)
    _stats("  -> LONG",  h_above_cr_below[h_above_cr_below["direction"]=="LONG"], lines)
    _stats("  -> SHORT", h_above_cr_below[h_above_cr_below["direction"]=="SHORT"], lines)
    _stats("BELOW HVL + ABOVE CR (rare)", h_below_cr_above, lines)
    _stats("  -> LONG",  h_below_cr_above[h_below_cr_above["direction"]=="LONG"], lines)
    _stats("  -> SHORT", h_below_cr_above[h_below_cr_above["direction"]=="SHORT"], lines)
    _stats("BELOW HVL + BELOW CR", h_below_cr_below, lines)
    _stats("  -> LONG",  h_below_cr_below[h_below_cr_below["direction"]=="LONG"], lines)
    _stats("  -> SHORT", h_below_cr_below[h_below_cr_below["direction"]=="SHORT"], lines)

    # ----- ALTERNATIVE STRATEGY TEST
    lines.append("")
    lines.append("=" * 200)
    lines.append("ALT STRATEGY TEST: Above HVL + below CR -> SHORT (mean-reversion)")
    lines.append("                   Above HVL + above CR -> LONG  (breakout, CR cleared)")
    lines.append("                   Below HVL            -> existing direction")
    lines.append("Direct comparison of LONGs vs SHORTs in the 'above HVL + below CR' bucket")
    lines.append("=" * 200)
    bucket = h_above_cr_below
    if not bucket.empty:
        _stats("Bucket (ABOVE HVL + BELOW CR)", bucket, lines)
        l = bucket[bucket["direction"]=="LONG"]
        s = bucket[bucket["direction"]=="SHORT"]
        _stats("  Existing LONGs in bucket",  l, lines)
        _stats("  Existing SHORTs in bucket", s, lines)
        if not l.empty:
            inverted_pnl = -l["pnl"].values
            inv_total = inverted_pnl.sum()
            inv_wr = (inverted_pnl > 0).mean()
            inv_pos = inverted_pnl[inverted_pnl>0].sum()
            inv_neg = -inverted_pnl[inverted_pnl<0].sum()
            inv_pf = inv_pos / inv_neg if inv_neg > 0 else float("inf")
            inv_daily = pd.Series(inverted_pnl, index=l["date"].values).groupby(level=0).sum()
            inv_sharpe = inv_daily.mean()/inv_daily.std()*np.sqrt(252) if inv_daily.std()>0 else 0.0
            lines.append(f"  HYPOTHETICAL: LONGs flipped to SHORTS (same entry/exit logic, sign-flipped PnL):")
            lines.append(f"    n={len(l)}  total_inv={inv_total:+.1f}  WR={inv_wr:.1%}  PF={min(inv_pf,999):.2f}  Sharpe={inv_sharpe:+.2f}")
    else:
        lines.append("  No trades in 'above HVL + below CR' bucket")

    # ----- Compound: gamma regime x HVL position
    lines.append("")
    lines.append("=" * 200)
    lines.append("COMPOUND: gamma regime  x  HVL position  (best/worst combinations)")
    lines.append("=" * 200)
    for gname, gdf in [("POS gamma", pos), ("NEG gamma", neg)]:
        for hvname, hvfilter in [
            ("ABOVE HVL", lambda d: d[d["above_hvl"] == True]),
            ("BELOW HVL", lambda d: d[d["below_hvl"] == True]),
        ]:
            sub = hvfilter(gdf[gdf["hvl_nq"].notna()])
            label = f"{gname} + {hvname}"
            _stats(label, sub, lines)
            _stats(f"  {gname} {hvname} -> LONG",  sub[sub["direction"]=="LONG"],  lines)
            _stats(f"  {gname} {hvname} -> SHORT", sub[sub["direction"]=="SHORT"], lines)

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}")
    print()
    print("\n".join(lines[-100:]))


if __name__ == "__main__":
    sys.exit(main())
