"""Lock-in writeup for the chosen config:

   ENTRY:  B2 X=0.75 N=15 D=70 strict=True BAND_K=0.25
           + conf_N=5 conf_D=75 HALF
   EXIT:   pure_ratchet SL at ymult=2.5 * ATR_14_20min
           + V2 MFE-anchored guard (K=0.8, lock=0.45)
           fixed TP at entry +/- 2.0 * ATR_at_entry
           force close 16:00 ET
   DEDUPE: chained Mode 1 by actual exit times
   SIZING: martingale FC-only  (loss via FORCE_CLOSE -> 2x next trade -> back to 1x)
   INSTR:  MNQ ($2/pt), 1-min entry / 20-min exit management

Produces: tradelogs/robust_configs/locked_v2_k08_lock045_mart_fc.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from sweep_ratchet_sl_fixed_tp import filter_pre_dedupe
from test_pure_ratchet_exits import build_20min_bars, FORCE_CLOSE_TIME

PARQUET_DIR = Path(__file__).parent / "parquets"
EOD_MQ      = Path("D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet")
OUT_TXT     = Path(__file__).parent.parent / "tradelogs" / "robust_configs" / "locked_v2_k08_lock045_mart_fc.txt"

VARIANT, X, N, D, STRICT, BAND_K = "B2", 0.75, 15, 70, True, 0.25
CONF_N, CONF_D = 5, 75
YMULT, TPMULT  = 2.50, 2.00
MFE_K, MFE_LOCK = 0.8, 0.45


def simulate_exit_v2(direction, entry_ts, entry_price, bars20):
    sign = 1 if direction == "LONG" else -1
    bars_idx = bars20.index
    start = bars_idx.searchsorted(entry_ts, side="right")
    if start >= len(bars_idx): return None
    ent_date = entry_ts.date()
    end = start
    while end < len(bars_idx) and bars_idx[end].date() == ent_date: end += 1
    if end == start: return None
    init_idx = start - 1
    if init_idx < 0 or np.isnan(bars20["atr_y"].iloc[init_idx]): return None
    init_atr_y = float(bars20["atr_y"].iloc[init_idx])
    yellow_val = entry_price - sign * YMULT * init_atr_y
    prev_yellow = yellow_val
    o = bars20["open"].values[start:end]; h = bars20["high"].values[start:end]
    l = bars20["low"].values[start:end];  c = bars20["close"].values[start:end]
    ay = bars20["atr_y"].values[start:end]; ts_arr = bars_idx[start:end]
    n = end - start
    green_val = entry_price + sign * TPMULT * init_atr_y
    tp_dist = abs(green_val - entry_price)
    mfe_so_far = 0.0
    peak_mfe_pts = 0.0
    for i in range(n):
        bar_close_ts = ts_arr[i] + pd.Timedelta(minutes=20)
        cur_mfe = (h[i] - entry_price) if sign > 0 else (entry_price - l[i])
        if cur_mfe > mfe_so_far: mfe_so_far = cur_mfe
        peak_mfe_pts = mfe_so_far
        if not np.isnan(ay[i]):
            raw_yellow = c[i] - sign * YMULT * ay[i]
            yellow_val = max(prev_yellow, raw_yellow) if sign > 0 else min(prev_yellow, raw_yellow)
        if mfe_so_far >= MFE_K * tp_dist:
            mfe_stop = entry_price + sign * MFE_LOCK * mfe_so_far
            stop_level = max(yellow_val, mfe_stop) if sign > 0 else min(yellow_val, mfe_stop)
        else:
            stop_level = yellow_val
        if sign > 0 and h[i] >= green_val:
            return (sign * (green_val - entry_price), "TP_FIXED", bar_close_ts, i + 1, peak_mfe_pts, init_atr_y)
        if sign < 0 and l[i] <= green_val:
            return (sign * (green_val - entry_price), "TP_FIXED", bar_close_ts, i + 1, peak_mfe_pts, init_atr_y)
        if sign > 0 and c[i] <= stop_level and c[i] < o[i]:
            return (c[i] - entry_price, "SL_TRAIL", bar_close_ts, i + 1, peak_mfe_pts, init_atr_y)
        if sign < 0 and c[i] >= stop_level and c[i] > o[i]:
            return (entry_price - c[i], "SL_TRAIL", bar_close_ts, i + 1, peak_mfe_pts, init_atr_y)
        if ts_arr[i].time() >= FORCE_CLOSE_TIME:
            return (sign * (c[i] - entry_price), "FORCE_CLOSE", bar_close_ts, i + 1, peak_mfe_pts, init_atr_y)
        prev_yellow = yellow_val
    return (sign * (c[-1] - entry_price), "EOD", ts_arr[-1] + pd.Timedelta(minutes=20),
            n, peak_mfe_pts, init_atr_y)


def run(cands, bars20, period_label):
    rows = []
    last_exit = pd.Timestamp(0, tz="America/New_York")
    for i, row in cands.iterrows():
        ex = simulate_exit_v2(row["direction"], row["entry_time_et"],
                              float(row["entry_price"]), bars20)
        if ex is None: continue
        pnl, reason, exit_ts, bars_held, peak_mfe, init_atr = ex
        if row["entry_time_et"] > last_exit:
            rows.append({
                "entry_ts": row["entry_time_et"], "exit_ts": exit_ts,
                "direction": row["direction"], "bars_held": bars_held,
                "reason": reason, "pnl": pnl, "period": period_label,
                "entry_hour": row["entry_time_et"].hour,
                "peak_mfe": peak_mfe, "init_atr": init_atr,
                "tp_dist": TPMULT * init_atr,
                "mfe_pct": peak_mfe / (TPMULT * init_atr) if init_atr > 0 else 0,
                "date": row["entry_time_et"].date(),
            })
            last_exit = exit_ts
    return pd.DataFrame(rows)


def apply_mart_fc(df):
    sizes = []
    cur = 1
    for pnl, reason in zip(df["pnl"].values, df["reason"].values):
        sizes.append(cur)
        if cur == 2:
            cur = 1
        else:
            if pnl < 0 and reason == "FORCE_CLOSE":
                cur = 2
            else:
                cur = 1
    return np.array(sizes)


def stats_block(df, pnl_col):
    pnl = df[pnl_col].values
    n = len(pnl)
    if n == 0:
        return {"n": 0}
    wins = pnl[pnl > 0]; losses = pnl[pnl < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    avg_win = wins.mean() if len(wins) else 0
    avg_loss = losses.mean() if len(losses) else 0
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    daily = pd.Series(pnl, index=df["date"].values).groupby(level=0).sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0
    eq = np.cumsum(pnl); peak = np.maximum.accumulate(eq); mdd = (eq - peak).min()
    wr = (pnl > 0).mean() * 100
    return {"n": n, "total": pnl.sum(), "pf": pf, "wr": wr,
            "sharpe": sharpe, "mdd": mdd, "payoff": payoff,
            "avg_win": avg_win, "avg_loss": avg_loss,
            "mnq_$": pnl.sum() * 2, "mnq_mdd_$": mdd * 2,
            "tp_n": int((df["reason"] == "TP_FIXED").sum()),
            "sl_n": int((df["reason"] == "SL_TRAIL").sum()),
            "fc_n": int((df["reason"] == "FORCE_CLOSE").sum())}


def hour_bucket(df, pnl_col):
    out = []
    for h in sorted(df["entry_hour"].unique()):
        sub = df[df["entry_hour"] == h]
        s = stats_block(sub, pnl_col)
        out.append((h, s))
    return out


def load_gamma_lookup():
    if not EOD_MQ.exists():
        return None, None
    eod = pd.read_parquet(EOD_MQ)
    eod["date"] = pd.to_datetime(eod["date"]).dt.date
    eod = eod.set_index("date").sort_index()
    eod_dates = sorted(eod.index.tolist())
    def prior_mq(d):
        prev = None
        for md in eod_dates:
            if md < d: prev = md
            else: break
        return prev
    return eod, prior_mq


def attach_gamma(df):
    eod, prior_mq = load_gamma_lookup()
    if eod is None:
        df["gamma_sign"] = np.nan
        return df
    col = None
    for c in ("qqq_gamma_sign", "gamma_sign"):
        if c in eod.columns:
            col = c; break
    if col is None:
        df["gamma_sign"] = np.nan
        return df
    g_lookup = {}
    for d in df["date"].unique():
        p = prior_mq(d)
        g_lookup[d] = eod.loc[p, col] if p in eod.index else np.nan
    df["gamma_sign"] = df["date"].map(g_lookup)
    return df


def main():
    print("loading 20-min bars + entry candidates...")
    bars20 = build_20min_bars()
    is_df  = pd.read_parquet(PARQUET_DIR / "entry_signal_trades.parquet")
    oos_df = pd.read_parquet(PARQUET_DIR / "entry_signal_trades_oos.parquet")
    is_cands  = filter_pre_dedupe(is_df)
    oos_cands = filter_pre_dedupe(oos_df)

    print("simulating exits...")
    is_t  = run(is_cands,  bars20, "IS")
    oos_t = run(oos_cands, bars20, "OOS")
    comb  = pd.concat([is_t, oos_t], ignore_index=True).sort_values("entry_ts").reset_index(drop=True)

    # Apply martingale FC-only chronologically across IS+OOS
    sizes = apply_mart_fc(comb)
    comb["size"] = sizes
    comb["scaled_pnl"] = comb["pnl"] * sizes

    # Stats (no-mart) and (mart)
    is_no  = stats_block(comb[comb["period"]=="IS"],  "pnl")
    oos_no = stats_block(comb[comb["period"]=="OOS"], "pnl")
    all_no = stats_block(comb, "pnl")

    is_m   = stats_block(comb[comb["period"]=="IS"],  "scaled_pnl")
    oos_m  = stats_block(comb[comb["period"]=="OOS"], "scaled_pnl")
    all_m  = stats_block(comb, "scaled_pnl")

    # Attach gamma
    comb = attach_gamma(comb)

    # ----- Build the txt file -----
    L = []
    L.append("=" * 200)
    L.append("LOCKED CONFIG — V2 K=0.8 lock=0.45 + martingale FC-only  (chosen 2026-05-10)")
    L.append("=" * 200)
    L.append("")
    L.append("ENTRY (5-min bar signals):")
    L.append(f"  variant         B2")
    L.append(f"  pinbar X        {X}      (signal candle wick prominence floor)")
    L.append(f"  window N        {N}      (signal candle absorption window in ticks)")
    L.append(f"  delta D         {D}      (signal candle absorption threshold)")
    L.append(f"  strict          {STRICT}    (require SHORT close < OLO / LONG close > OHO)")
    L.append(f"  band_K          {BAND_K}    (level-proximity band: clip(band_K * ATR, 5, 20))")
    L.append(f"  conf_N          {CONF_N}      (confirmation candle window)")
    L.append(f"  conf_D          {CONF_D}     (confirmation delta threshold, HALF mode)")
    L.append("")
    L.append("EXIT (20-min bar management):")
    L.append("  STOP TYPE:      pure_ratchet  +  V2 MFE-anchored guard")
    L.append(f"                  yellow_val      = close - sign * {YMULT} * ATR_14_20min  (ratchet — never moves against)")
    L.append(f"                  MFE-guard arms  when peak_MFE >= {MFE_K} * (TP_distance)")
    L.append(f"                  mfe_stop        = entry +/- {MFE_LOCK} * peak_MFE")
    L.append(f"                  effective_stop  = MORE FAVORABLE of (yellow_val, mfe_stop)")
    L.append("                  trigger:        close beyond stop AND adverse-direction bar")
    L.append(f"  TP TYPE:        FIXED at entry +/- {TPMULT} * ATR_at_entry  (no decay)")
    L.append("  FORCE CLOSE:    16:00 ET  (RTH end)")
    L.append("")
    L.append("SIZING (martingale FC-only):")
    L.append("  default size = 1 contract")
    L.append("  loss with reason == FORCE_CLOSE -> next trade = 2 contracts")
    L.append("  after 2-contract trade (any outcome) -> reset to 1")
    L.append("  loss with reason == SL_TRAIL  -> NO CHANGE (stay at 1)")
    L.append("  wins -> stay at 1 until next FC-loss")
    L.append("  Maximum size ever = 2 contracts.")
    L.append("")
    L.append("DEDUPE: chained Mode 1 by actual exit times (no overlapping trades)")
    L.append("INSTRUMENT: MNQ ($2/pt, $0.50 tick).  NQ basis: $20/pt.")
    L.append("DATA RANGE: IS = 2020-12-01 → 2024-12-31  |  OOS = 2025-01 → 2025-11")
    L.append("")
    L.append("=" * 200)
    L.append("RANK-STYLE SUMMARY (matches top_25_robust_configs.txt header)")
    L.append("=" * 200)
    L.append("")
    L.append("With martingale FC-only applied (scaled PnL):")
    hdr = (f"{'rank':>4}  {'conf_N':>6}  {'conf_D':>6}  {'tot_n':>5}  "
           f"{'is_n':>5}  {'is_pf':>5}  {'is_sharpe':>9}  {'is_wr':>6}  {'is_total':>9}  "
           f"{'oos_n':>5}  {'oos_pf':>5}  {'oos_sharpe':>10}  {'oos_wr':>6}  {'oos_total':>9}  "
           f"{'min_pf':>6}  {'min_sharpe':>10}  {'is_mdd':>7}  {'oos_mdd':>7}  {'worst_mdd':>9}")
    L.append(hdr)
    line_mart = (f"{1:>4}  {CONF_N:>6}  {CONF_D:>6}  {all_m['n']:>5}  "
                 f"{is_m['n']:>5}  {is_m['pf']:>5.2f}  {is_m['sharpe']:>9.2f}  {is_m['wr']:>5.1f}%  {is_m['total']:>+9.1f}  "
                 f"{oos_m['n']:>5}  {oos_m['pf']:>5.2f}  {oos_m['sharpe']:>10.2f}  {oos_m['wr']:>5.1f}%  {oos_m['total']:>+9.1f}  "
                 f"{min(is_m['pf'], oos_m['pf']):>6.2f}  {min(is_m['sharpe'], oos_m['sharpe']):>10.2f}  "
                 f"{is_m['mdd']:>+7.1f}  {oos_m['mdd']:>+7.1f}  {max(abs(is_m['mdd']), abs(oos_m['mdd'])):>9.1f}")
    L.append(line_mart)
    L.append("")
    L.append("Without martingale (raw 1-contract baseline):")
    L.append(hdr)
    line_no = (f"{0:>4}  {CONF_N:>6}  {CONF_D:>6}  {all_no['n']:>5}  "
               f"{is_no['n']:>5}  {is_no['pf']:>5.2f}  {is_no['sharpe']:>9.2f}  {is_no['wr']:>5.1f}%  {is_no['total']:>+9.1f}  "
               f"{oos_no['n']:>5}  {oos_no['pf']:>5.2f}  {oos_no['sharpe']:>10.2f}  {oos_no['wr']:>5.1f}%  {oos_no['total']:>+9.1f}  "
               f"{min(is_no['pf'], oos_no['pf']):>6.2f}  {min(is_no['sharpe'], oos_no['sharpe']):>10.2f}  "
               f"{is_no['mdd']:>+7.1f}  {oos_no['mdd']:>+7.1f}  {max(abs(is_no['mdd']), abs(oos_no['mdd'])):>9.1f}")
    L.append(line_no)
    L.append("")
    L.append("(All totals/MDDs in NQ pts.  $ value: NQ = pts × $20  |  MNQ = pts × $2)")

    L.append("")
    L.append("=" * 200)
    L.append("HEADLINE METRICS")
    L.append("=" * 200)
    L.append(f"  Combined IS+OOS (with FC-only mart):")
    L.append(f"    trades:        {all_m['n']}    total:        {all_m['total']:+.1f} pts   = ${all_m['mnq_$']:+,.0f} MNQ   (${all_m['total']*20:+,.0f} NQ)")
    L.append(f"    PF:            {all_m['pf']:.3f}     win-rate:     {all_m['wr']:.2f}%     payoff (avg_w/|avg_l|): {all_m['payoff']:.3f}")
    L.append(f"    avg win:       {all_m['avg_win']:+.2f} pts     avg loss:     {all_m['avg_loss']:+.2f} pts")
    L.append(f"    max DD:        {all_m['mdd']:+.1f} pts ({all_m['mnq_mdd_$']:+,.0f} MNQ$)  ({all_m['mdd']*20:+,.0f} NQ$)")
    L.append(f"    exit-reason mix: TP={all_m['tp_n']}  SL_TRAIL={all_m['sl_n']}  FORCE_CLOSE={all_m['fc_n']}")
    L.append("")
    L.append(f"  Marginal contribution of martingale (FC-only):")
    L.append(f"    no-mart total: {all_no['total']:+.1f} pts   ${all_no['mnq_$']:+,.0f} MNQ")
    L.append(f"    mart total:    {all_m['total']:+.1f} pts   ${all_m['mnq_$']:+,.0f} MNQ")
    L.append(f"    delta:         {all_m['total']-all_no['total']:+.1f} pts   ${all_m['mnq_$']-all_no['mnq_$']:+,.0f} MNQ  (+{(all_m['total']-all_no['total'])/all_no['total']*100:.1f}%)")
    L.append(f"    no-mart MDD:   {all_no['mdd']:+.1f} pts   ${all_no['mnq_mdd_$']:+,.0f} MNQ")
    L.append(f"    mart MDD:      {all_m['mdd']:+.1f} pts   ${all_m['mnq_mdd_$']:+,.0f} MNQ")
    L.append(f"    DD penalty:    +${abs(all_m['mnq_mdd_$']) - abs(all_no['mnq_mdd_$']):,.0f} MNQ ({abs(all_m['mdd']) - abs(all_no['mdd']):+.1f} pts)")

    # MFE / excursion stats
    L.append("")
    L.append("=" * 200)
    L.append("MFE / TP-PROXIMITY DIAGNOSTICS")
    L.append("=" * 200)
    L.append("")
    L.append("  Trades by peak MFE as fraction of TP distance (peak_mfe / (2 * ATR_at_entry)):")
    bucket_edges = [0.0, 0.25, 0.5, 0.75, 0.8, 1.0, 99]
    bucket_labels = ["0-25%", "25-50%", "50-75%", "75-80%", ">=80% (guard armed)", ">=100% (TP hit)"]
    L.append(f"  {'mfe_bucket':<22} {'n':>4} {'tp':>4} {'sl':>4} {'fc':>4}  {'avg_pnl':>8}  {'wr':>6}  {'total':>9}")
    for lo, hi, lab in zip(bucket_edges[:-1], bucket_edges[1:], bucket_labels):
        if lab.startswith(">=80"):
            sub = comb[(comb["mfe_pct"] >= 0.8) & (comb["mfe_pct"] < 1.0)]
        elif lab.startswith(">=100"):
            sub = comb[comb["mfe_pct"] >= 1.0]
        else:
            sub = comb[(comb["mfe_pct"] >= lo) & (comb["mfe_pct"] < hi)]
        if sub.empty:
            L.append(f"  {lab:<22} {0:>4}")
            continue
        s = stats_block(sub, "scaled_pnl")
        L.append(f"  {lab:<22} {s['n']:>4} {s['tp_n']:>4} {s['sl_n']:>4} {s['fc_n']:>4}  "
                 f"{s['total']/max(s['n'],1):>+8.2f}  {s['wr']:>5.1f}%  {s['total']:>+9.1f}")
    L.append("")
    L.append(f"  Peak MFE (pts) — combined: median={comb['peak_mfe'].median():.1f}  mean={comb['peak_mfe'].mean():.1f}  "
             f"p25={comb['peak_mfe'].quantile(0.25):.1f}  p75={comb['peak_mfe'].quantile(0.75):.1f}")
    L.append(f"  TP distance (pts) — combined: median={comb['tp_dist'].median():.1f}  mean={comb['tp_dist'].mean():.1f}")

    # Entry by hour
    L.append("")
    L.append("=" * 200)
    L.append("ENTRY-HOUR BREAKDOWN (entry_hour ET)  — both raw and martingale-applied")
    L.append("=" * 200)
    L.append(f"  {'hour':<5} {'n':>4} {'tp':>4} {'sl':>4} {'fc':>4}   "
             f"{'no-mart_total':>13} {'no-mart_pf':>10} {'no-mart_wr':>10}   "
             f"{'mart_total':>10} {'mart_pf':>7} {'mart_wr':>7}   {'mart_mdd':>9}")
    for h, _ in hour_bucket(comb, "pnl"):
        sub = comb[comb["entry_hour"] == h]
        s_no = stats_block(sub, "pnl")
        s_m  = stats_block(sub, "scaled_pnl")
        L.append(f"  {h:<5} {s_no['n']:>4} {s_no['tp_n']:>4} {s_no['sl_n']:>4} {s_no['fc_n']:>4}   "
                 f"{s_no['total']:>+13.1f} {s_no['pf']:>10.2f} {s_no['wr']:>9.1f}%   "
                 f"{s_m['total']:>+10.1f} {s_m['pf']:>7.2f} {s_m['wr']:>6.1f}%   {s_m['mdd']:>+9.1f}")
    L.append("")
    L.append("  Filtering hint: hours with negative no-mart total or PF < 1 are candidates to drop.")

    # Per-period detailed breakdown
    L.append("")
    L.append("=" * 200)
    L.append("PER-PERIOD DETAIL")
    L.append("=" * 200)
    for label, no, m in [("IS", is_no, is_m), ("OOS", oos_no, oos_m)]:
        L.append("")
        L.append(f"  {label}:")
        L.append(f"    no-mart  n={no['n']}  total={no['total']:+.1f} pts (${no['mnq_$']:+,.0f} MNQ)  pf={no['pf']:.3f}  "
                 f"sharpe={no['sharpe']:+.2f}  wr={no['wr']:.1f}%  mdd={no['mdd']:+.1f} (${no['mnq_mdd_$']:+,.0f})")
        L.append(f"    mart     n={m['n']}  total={m['total']:+.1f} pts (${m['mnq_$']:+,.0f} MNQ)  pf={m['pf']:.3f}  "
                 f"sharpe={m['sharpe']:+.2f}  wr={m['wr']:.1f}%  mdd={m['mdd']:+.1f} (${m['mnq_mdd_$']:+,.0f})")

    # Gamma regime
    L.append("")
    L.append("=" * 200)
    L.append("GAMMA REGIME FILTER  (prior-day EOD multi-DTE qqq_gamma_sign)")
    L.append("=" * 200)
    n_with_gamma = comb["gamma_sign"].notna().sum()
    L.append(f"  trades with gamma label: {n_with_gamma} / {len(comb)} ({n_with_gamma/len(comb)*100:.1f}%)")
    L.append("")
    L.append(f"  {'subset':<35} {'n':>4} {'L':>4} {'S':>4}  {'no-mart_total':>13} {'no-mart_pf':>10} "
             f"{'no-mart_wr':>10}   {'mart_total':>10} {'mart_pf':>7} {'mart_wr':>7}   {'mart_mdd':>9}")
    for label, mask in [
        ("ALL labelled",            comb["gamma_sign"].notna()),
        ("POS gamma  (sign = +1)",  comb["gamma_sign"] == 1),
        ("NEG gamma  (sign = -1)",  comb["gamma_sign"] == -1),
        ("POS gamma + LONG only",   (comb["gamma_sign"] == 1) & (comb["direction"] == "LONG")),
        ("POS gamma + SHORT only",  (comb["gamma_sign"] == 1) & (comb["direction"] == "SHORT")),
        ("NEG gamma + LONG only",   (comb["gamma_sign"] == -1) & (comb["direction"] == "LONG")),
        ("NEG gamma + SHORT only",  (comb["gamma_sign"] == -1) & (comb["direction"] == "SHORT")),
        ("Drop POS+SHORT only",     ~((comb["gamma_sign"] == 1) & (comb["direction"] == "SHORT"))),
        ("Drop NEG+SHORT only",     ~((comb["gamma_sign"] == -1) & (comb["direction"] == "SHORT"))),
    ]:
        sub = comb[mask]
        if sub.empty:
            L.append(f"  {label:<35}  n=0")
            continue
        s_no = stats_block(sub, "pnl")
        s_m  = stats_block(sub, "scaled_pnl")
        n_l = int((sub["direction"] == "LONG").sum())
        n_s = int((sub["direction"] == "SHORT").sum())
        L.append(f"  {label:<35} {s_no['n']:>4} {n_l:>4} {n_s:>4}  "
                 f"{s_no['total']:>+13.1f} {s_no['pf']:>10.2f} {s_no['wr']:>9.1f}%   "
                 f"{s_m['total']:>+10.1f} {s_m['pf']:>7.2f} {s_m['wr']:>6.1f}%   {s_m['mdd']:>+9.1f}")

    # Gamma regime split per period
    L.append("")
    L.append("  Gamma regime — IS / OOS split:")
    for period_label in ("IS", "OOS"):
        L.append("")
        L.append(f"    {period_label}:")
        for label, sign_val in [("POS", 1), ("NEG", -1)]:
            sub = comb[(comb["period"] == period_label) & (comb["gamma_sign"] == sign_val)]
            if sub.empty: continue
            s = stats_block(sub, "scaled_pnl")
            L.append(f"      {label}-gamma  n={s['n']:>3}  total={s['total']:>+8.1f} pts (${s['mnq_$']:>+7,.0f} MNQ)  "
                     f"pf={s['pf']:.2f}  wr={s['wr']:.1f}%  mdd={s['mdd']:+.1f}")

    L.append("")
    L.append("=" * 200)
    L.append("END.")
    L.append("=" * 200)

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}")
    print()
    print("\n".join(L))


if __name__ == "__main__":
    main()
