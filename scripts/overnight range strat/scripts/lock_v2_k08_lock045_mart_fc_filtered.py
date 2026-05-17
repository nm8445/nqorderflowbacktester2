"""Locked config + entry filters:
   - Drop POS-gamma SHORT entries (keep all LONGs in any regime + NEG SHORTs)
   - Restrict entry window to hours 9-14 ET (drop 15:xx and 16:xx entries)

Otherwise identical to lock_v2_k08_lock045_mart_fc.py.

Produces: tradelogs/robust_configs/locked_v2_k08_lock045_mart_fc_filtered.txt
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
OUT_TXT     = Path(__file__).parent.parent / "tradelogs" / "robust_configs" / "locked_v2_k08_lock045_mart_fc_filtered.txt"

VARIANT, X, N, D, STRICT, BAND_K = "B2", 0.75, 15, 70, True, 0.25
CONF_N, CONF_D = 5, 75
YMULT, TPMULT  = 2.50, 2.00
MFE_K, MFE_LOCK = 0.8, 0.45

# ---- ENTRY FILTERS (new) ----
ALLOWED_HOURS = {9, 10, 11, 12, 13, 14}    # drop 15, 16
DROP_POS_SHORT = True                      # drop SHORT entries when prior-day gamma_sign == +1


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


def attach_gamma_to_candidates(cands):
    """Add prior-day gamma_sign to candidates dataframe."""
    eod, prior_mq = load_gamma_lookup()
    if eod is None:
        cands = cands.copy()
        cands["gamma_sign"] = np.nan
        return cands
    col = None
    for c in ("qqq_gamma_sign", "gamma_sign"):
        if c in eod.columns:
            col = c; break
    if col is None:
        cands = cands.copy()
        cands["gamma_sign"] = np.nan
        return cands
    cands = cands.copy()
    cands["entry_date"] = pd.to_datetime(cands["entry_time_et"]).dt.date
    g_lookup = {}
    for d in cands["entry_date"].unique():
        p = prior_mq(d)
        g_lookup[d] = eod.loc[p, col] if p in eod.index else np.nan
    cands["gamma_sign"] = cands["entry_date"].map(g_lookup)
    return cands


def filter_candidates(cands):
    """Apply hour and POS+SHORT filters. Returns (kept_df, drop_counts)."""
    n0 = len(cands)
    cands["entry_hour"] = pd.to_datetime(cands["entry_time_et"]).dt.hour

    # 1) hour filter
    hour_mask = cands["entry_hour"].isin(ALLOWED_HOURS)
    n_drop_hour = int((~hour_mask).sum())

    # 2) POS+SHORT filter
    if DROP_POS_SHORT:
        pos_short_mask = (cands["gamma_sign"] == 1) & (cands["direction"] == "SHORT")
        n_drop_pos_short = int(pos_short_mask.sum())
    else:
        pos_short_mask = pd.Series(False, index=cands.index)
        n_drop_pos_short = 0

    keep = hour_mask & (~pos_short_mask)
    out = cands[keep].copy().reset_index(drop=True)
    return out, {
        "n_in": n0,
        "n_dropped_hour": n_drop_hour,
        "n_dropped_pos_short": n_drop_pos_short,
        "n_out": len(out),
    }


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
                "gamma_sign": row.get("gamma_sign", np.nan),
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
        return {"n": 0, "total": 0, "pf": 0, "wr": 0, "sharpe": 0,
                "mdd": 0, "payoff": 0, "avg_win": 0, "avg_loss": 0,
                "mnq_$": 0, "mnq_mdd_$": 0, "tp_n": 0, "sl_n": 0, "fc_n": 0}
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


def main():
    print("loading 20-min bars + entry candidates...")
    bars20 = build_20min_bars()
    is_df  = pd.read_parquet(PARQUET_DIR / "entry_signal_trades.parquet")
    oos_df = pd.read_parquet(PARQUET_DIR / "entry_signal_trades_oos.parquet")
    is_cands  = filter_pre_dedupe(is_df)
    oos_cands = filter_pre_dedupe(oos_df)

    # attach gamma + apply filters BEFORE exit simulation/dedupe
    is_cands  = attach_gamma_to_candidates(is_cands)
    oos_cands = attach_gamma_to_candidates(oos_cands)
    is_kept,  is_drop  = filter_candidates(is_cands)
    oos_kept, oos_drop = filter_candidates(oos_cands)

    print(f"  IS  candidates  {is_drop['n_in']} -> kept {is_drop['n_out']}  "
          f"(dropped: hour {is_drop['n_dropped_hour']}, POS+SHORT {is_drop['n_dropped_pos_short']})")
    print(f"  OOS candidates  {oos_drop['n_in']} -> kept {oos_drop['n_out']}  "
          f"(dropped: hour {oos_drop['n_dropped_hour']}, POS+SHORT {oos_drop['n_dropped_pos_short']})")

    print("simulating exits...")
    is_t  = run(is_kept,  bars20, "IS")
    oos_t = run(oos_kept, bars20, "OOS")
    comb  = pd.concat([is_t, oos_t], ignore_index=True).sort_values("entry_ts").reset_index(drop=True)

    sizes = apply_mart_fc(comb)
    comb["size"] = sizes
    comb["scaled_pnl"] = comb["pnl"] * sizes

    # Dump trade-level CSV for downstream combined-portfolio analysis
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_TXT.parent / "locked_v2_k08_lock045_mart_fc_filtered_trades.csv"
    comb.to_csv(csv_path, index=False)
    print(f"  wrote trade-level CSV -> {csv_path}")

    is_no  = stats_block(comb[comb["period"] == "IS"],  "pnl")
    oos_no = stats_block(comb[comb["period"] == "OOS"], "pnl")
    all_no = stats_block(comb, "pnl")
    is_m   = stats_block(comb[comb["period"] == "IS"],  "scaled_pnl")
    oos_m  = stats_block(comb[comb["period"] == "OOS"], "scaled_pnl")
    all_m  = stats_block(comb, "scaled_pnl")

    L = []
    L.append("=" * 200)
    L.append("LOCKED CONFIG (FILTERED) — V2 K=0.8 lock=0.45 + martingale FC-only")
    L.append("                          + drop POS-gamma SHORTs + restrict entries to 09:00-14:59 ET")
    L.append("=" * 200)
    L.append("")
    L.append("ENTRY (5-min bar signals):")
    L.append(f"  variant         B2")
    L.append(f"  pinbar X        {X}")
    L.append(f"  window N        {N}")
    L.append(f"  delta D         {D}")
    L.append(f"  strict          {STRICT}")
    L.append(f"  band_K          {BAND_K}")
    L.append(f"  conf_N          {CONF_N}")
    L.append(f"  conf_D          {CONF_D}     (HALF mode)")
    L.append("")
    L.append("ENTRY FILTERS (new):")
    L.append(f"  hours allowed   {sorted(ALLOWED_HOURS)} ET   (drops 15:xx and 16:xx)")
    L.append(f"  drop POS-gamma SHORT entries: {DROP_POS_SHORT}   (gamma_sign comes from prior-day EOD multi-DTE qqq_gamma_sign)")
    L.append("")
    L.append("EXIT (20-min bar management):")
    L.append("  STOP TYPE:      pure_ratchet  +  V2 MFE-anchored guard")
    L.append(f"                  yellow_val      = close - sign * {YMULT} * ATR_14_20min  (ratchet)")
    L.append(f"                  MFE-guard arms  when peak_MFE >= {MFE_K} * (TP_distance)")
    L.append(f"                  mfe_stop        = entry +/- {MFE_LOCK} * peak_MFE")
    L.append(f"                  effective_stop  = MORE FAVORABLE of (yellow_val, mfe_stop)")
    L.append("                  trigger:        close beyond stop AND adverse-direction bar")
    L.append(f"  TP TYPE:        FIXED at entry +/- {TPMULT} * ATR_at_entry")
    L.append("  FORCE CLOSE:    16:00 ET")
    L.append("")
    L.append("SIZING (martingale FC-only):")
    L.append("  default size = 1 contract; FC loss -> 2 contracts next; reset to 1 after the 2x trade.")
    L.append("")
    L.append("DEDUPE: chained Mode 1 by actual exit times")
    L.append("INSTRUMENT: MNQ ($2/pt).  NQ basis: $20/pt.")
    L.append("DATA RANGE: IS = 2020-12-01 -> 2024-12-31  |  OOS = 2025-01 -> 2025-11")
    L.append("")
    L.append("CANDIDATE FILTERING:")
    L.append(f"  IS  pre-filter candidates  : {is_drop['n_in']}")
    L.append(f"      dropped by hour filter : {is_drop['n_dropped_hour']}")
    L.append(f"      dropped POS+SHORT      : {is_drop['n_dropped_pos_short']}")
    L.append(f"      kept                   : {is_drop['n_out']}")
    L.append(f"  OOS pre-filter candidates  : {oos_drop['n_in']}")
    L.append(f"      dropped by hour filter : {oos_drop['n_dropped_hour']}")
    L.append(f"      dropped POS+SHORT      : {oos_drop['n_dropped_pos_short']}")
    L.append(f"      kept                   : {oos_drop['n_out']}")
    L.append(f"  (Note: chained dedupe is applied after candidate filtering, so the final trade count")
    L.append(f"   below differs from the kept-candidate count by the dedupe overlap.)")

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
    L.append("(All totals/MDDs in NQ pts.  $ value: NQ = pts * $20  |  MNQ = pts * $2)")

    L.append("")
    L.append("=" * 200)
    L.append("HEADLINE METRICS (combined IS+OOS)")
    L.append("=" * 200)
    L.append(f"  With FC-only mart:")
    L.append(f"    trades:        {all_m['n']}    total:        {all_m['total']:+.1f} pts   = ${all_m['mnq_$']:+,.0f} MNQ   (${all_m['total']*20:+,.0f} NQ)")
    L.append(f"    PF:            {all_m['pf']:.3f}     win-rate:     {all_m['wr']:.2f}%     payoff: {all_m['payoff']:.3f}")
    L.append(f"    avg win:       {all_m['avg_win']:+.2f} pts     avg loss:     {all_m['avg_loss']:+.2f} pts")
    L.append(f"    max DD:        {all_m['mdd']:+.1f} pts   ${all_m['mnq_mdd_$']:+,.0f} MNQ")
    L.append(f"    exit mix:      TP={all_m['tp_n']}  SL_TRAIL={all_m['sl_n']}  FORCE_CLOSE={all_m['fc_n']}")
    L.append(f"  No-mart (1 contract):")
    L.append(f"    trades:        {all_no['n']}    total:        {all_no['total']:+.1f} pts   = ${all_no['mnq_$']:+,.0f} MNQ")
    L.append(f"    PF:            {all_no['pf']:.3f}     wr:           {all_no['wr']:.2f}%")
    L.append(f"    max DD:        {all_no['mdd']:+.1f} pts   ${all_no['mnq_mdd_$']:+,.0f} MNQ")

    L.append("")
    L.append("=" * 200)
    L.append("VS UNFILTERED LOCKED CONFIG (locked_v2_k08_lock045_mart_fc.txt)")
    L.append("=" * 200)
    L.append("  Unfiltered (baseline locked):")
    L.append("    mart total:    +8781.6 pts   $+17,563 MNQ")
    L.append("    mart PF:       1.336")
    L.append("    mart WR:       58.89%")
    L.append("    mart MDD:      -1109.8 pts   $-2,220 MNQ")
    L.append("    no-mart total: +6622.0 pts   $+13,244 MNQ   PF 1.30   MDD -657.7 pts")
    L.append("")
    L.append("  Filtered (this run):")
    L.append(f"    mart total:    {all_m['total']:+.1f} pts   ${all_m['mnq_$']:+,.0f} MNQ")
    L.append(f"    mart PF:       {all_m['pf']:.3f}")
    L.append(f"    mart WR:       {all_m['wr']:.2f}%")
    L.append(f"    mart MDD:      {all_m['mdd']:+.1f} pts   ${all_m['mnq_mdd_$']:+,.0f} MNQ")
    L.append(f"    no-mart total: {all_no['total']:+.1f} pts   ${all_no['mnq_$']:+,.0f} MNQ   PF {all_no['pf']:.2f}   MDD {all_no['mdd']:+.1f} pts")
    L.append("")
    L.append(f"  delta vs unfiltered (mart):  total  {all_m['total']-8781.6:+.1f} pts (${all_m['mnq_$']-17563:+,.0f} MNQ)")
    L.append(f"                                 PF     {all_m['pf']-1.336:+.3f}")
    L.append(f"                                 MDD    {all_m['mdd']-(-1109.8):+.1f} pts (${all_m['mnq_mdd_$']-(-2220):+,.0f} MNQ)")

    # Per-period breakdown
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

    # Entry hour breakdown (post-filter, should only be 9-14)
    L.append("")
    L.append("=" * 200)
    L.append("ENTRY-HOUR BREAKDOWN (post-filter)")
    L.append("=" * 200)
    L.append(f"  {'hour':<5} {'n':>4} {'tp':>4} {'sl':>4} {'fc':>4}   "
             f"{'no-mart_total':>13} {'no-mart_pf':>10} {'no-mart_wr':>10}   "
             f"{'mart_total':>10} {'mart_pf':>7} {'mart_wr':>7}   {'mart_mdd':>9}")
    for h in sorted(comb["entry_hour"].unique()):
        sub = comb[comb["entry_hour"] == h]
        s_no = stats_block(sub, "pnl")
        s_m  = stats_block(sub, "scaled_pnl")
        L.append(f"  {h:<5} {s_no['n']:>4} {s_no['tp_n']:>4} {s_no['sl_n']:>4} {s_no['fc_n']:>4}   "
                 f"{s_no['total']:>+13.1f} {s_no['pf']:>10.2f} {s_no['wr']:>9.1f}%   "
                 f"{s_m['total']:>+10.1f} {s_m['pf']:>7.2f} {s_m['wr']:>6.1f}%   {s_m['mdd']:>+9.1f}")

    # Direction x gamma breakdown post-filter
    L.append("")
    L.append("=" * 200)
    L.append("DIRECTION x GAMMA REGIME (post-filter)")
    L.append("=" * 200)
    L.append(f"  {'subset':<35} {'n':>4} {'L':>4} {'S':>4}  {'no-mart_total':>13} {'no-mart_pf':>10} {'no-mart_wr':>10}   "
             f"{'mart_total':>10} {'mart_pf':>7} {'mart_wr':>7}   {'mart_mdd':>9}")
    for label, mask in [
        ("ALL labelled",            comb["gamma_sign"].notna()),
        ("POS gamma  (sign = +1)",  comb["gamma_sign"] == 1),
        ("NEG gamma  (sign = -1)",  comb["gamma_sign"] == -1),
        ("POS gamma + LONG",        (comb["gamma_sign"] == 1) & (comb["direction"] == "LONG")),
        ("POS gamma + SHORT",       (comb["gamma_sign"] == 1) & (comb["direction"] == "SHORT")),
        ("NEG gamma + LONG",        (comb["gamma_sign"] == -1) & (comb["direction"] == "LONG")),
        ("NEG gamma + SHORT",       (comb["gamma_sign"] == -1) & (comb["direction"] == "SHORT")),
    ]:
        sub = comb[mask]
        if sub.empty:
            L.append(f"  {label:<35}  n=0  (filtered out)")
            continue
        s_no = stats_block(sub, "pnl")
        s_m  = stats_block(sub, "scaled_pnl")
        n_l = int((sub["direction"] == "LONG").sum())
        n_s = int((sub["direction"] == "SHORT").sum())
        L.append(f"  {label:<35} {s_no['n']:>4} {n_l:>4} {n_s:>4}  "
                 f"{s_no['total']:>+13.1f} {s_no['pf']:>10.2f} {s_no['wr']:>9.1f}%   "
                 f"{s_m['total']:>+10.1f} {s_m['pf']:>7.2f} {s_m['wr']:>6.1f}%   {s_m['mdd']:>+9.1f}")

    # MFE diagnostics
    L.append("")
    L.append("=" * 200)
    L.append("MFE / TP-PROXIMITY DIAGNOSTICS (post-filter)")
    L.append("=" * 200)
    L.append("")
    L.append(f"  {'mfe_bucket':<22} {'n':>4} {'tp':>4} {'sl':>4} {'fc':>4}  {'avg_pnl':>8}  {'wr':>6}  {'total':>9}")
    bucket_edges = [(0.0, 0.25, "0-25%"), (0.25, 0.5, "25-50%"), (0.5, 0.75, "50-75%"),
                    (0.75, 0.8, "75-80%"), (0.8, 1.0, ">=80% (guard armed)"),
                    (1.0, 99.0, ">=100% (TP hit)")]
    for lo, hi, lab in bucket_edges:
        sub = comb[(comb["mfe_pct"] >= lo) & (comb["mfe_pct"] < hi)]
        if sub.empty:
            L.append(f"  {lab:<22} {0:>4}")
            continue
        s = stats_block(sub, "scaled_pnl")
        L.append(f"  {lab:<22} {s['n']:>4} {s['tp_n']:>4} {s['sl_n']:>4} {s['fc_n']:>4}  "
                 f"{s['total']/max(s['n'],1):>+8.2f}  {s['wr']:>5.1f}%  {s['total']:>+9.1f}")

    L.append("")
    L.append("=" * 200)
    L.append("END.")
    L.append("=" * 200)

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
