"""Audit the locked filtered config for ATR lookahead bias.

Compare two ATR sources:
  LEAKY: init_atr_y = bars20[start-1].atr_y  (bar containing entry_ts — has future data)
  FIXED: init_atr_y = bars20[start-1] only IF that bar's close is <= entry_ts,
          else step back further until that condition is met.

Strategy: V2 K=0.8 lock=0.45 + martingale FC-only + filters
          (drop POS-gamma SHORT entries, hours 9-14 ET).

Runs the simulation twice with identical entries; only the ATR source differs.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from sweep_ratchet_sl_fixed_tp import filter_pre_dedupe
from test_pure_ratchet_exits import build_20min_bars, FORCE_CLOSE_TIME

PARQUET_DIR  = Path(__file__).parent / "parquets"
EOD_MQ       = Path("D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet")
TRADELOG_DIR = Path(__file__).parent.parent / "tradelogs"
OUT_TXT      = TRADELOG_DIR / "robust_configs" / "locked_config_atr_leak_audit.txt"

VARIANT, X, N, D, STRICT, BAND_K = "B2", 0.75, 15, 70, True, 0.25
CONF_N, CONF_D = 5, 75
YMULT, TPMULT   = 2.50, 2.00
MFE_K, MFE_LOCK = 0.8, 0.45
ALLOWED_HOURS   = {9, 10, 11, 12, 13, 14}
DROP_POS_SHORT  = True
BAR_MINUTES     = 20


def simulate_exit(direction, entry_ts, entry_price, bars20, *, atr_mode: str):
    """atr_mode in {'leaky', 'fixed'}.
       'leaky'  -> bar at start-1 (the bar CONTAINING entry_ts; has 15 min of future)
       'fixed'  -> step back until bar's close <= entry_ts (no future data)
    """
    sign = 1 if direction == "LONG" else -1
    bars_idx = bars20.index
    start = bars_idx.searchsorted(entry_ts, side="right")
    if start >= len(bars_idx): return None
    ent_date = entry_ts.date()
    end = start
    while end < len(bars_idx) and bars_idx[end].date() == ent_date: end += 1
    if end == start: return None

    # Pick the init bar for ATR
    if atr_mode == "leaky":
        init_idx = start - 1
    else:  # fixed
        init_idx = start - 1
        bar_minutes = pd.Timedelta(minutes=BAR_MINUTES)
        while init_idx >= 0 and (bars_idx[init_idx] + bar_minutes) > entry_ts:
            init_idx -= 1
    if init_idx < 0 or np.isnan(bars20["atr_y"].iloc[init_idx]):
        return None
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
    for i in range(n):
        bar_close_ts = ts_arr[i] + pd.Timedelta(minutes=BAR_MINUTES)
        cur_mfe = (h[i] - entry_price) if sign > 0 else (entry_price - l[i])
        if cur_mfe > mfe_so_far: mfe_so_far = cur_mfe
        if not np.isnan(ay[i]):
            raw_yellow = c[i] - sign * YMULT * ay[i]
            yellow_val = max(prev_yellow, raw_yellow) if sign > 0 else min(prev_yellow, raw_yellow)
        if mfe_so_far >= MFE_K * tp_dist:
            mfe_stop = entry_price + sign * MFE_LOCK * mfe_so_far
            stop_level = max(yellow_val, mfe_stop) if sign > 0 else min(yellow_val, mfe_stop)
        else:
            stop_level = yellow_val
        if sign > 0 and h[i] >= green_val:
            return (sign * (green_val - entry_price), "TP_FIXED", bar_close_ts)
        if sign < 0 and l[i] <= green_val:
            return (sign * (green_val - entry_price), "TP_FIXED", bar_close_ts)
        if sign > 0 and c[i] <= stop_level and c[i] < o[i]:
            return (c[i] - entry_price, "SL_TRAIL", bar_close_ts)
        if sign < 0 and c[i] >= stop_level and c[i] > o[i]:
            return (entry_price - c[i], "SL_TRAIL", bar_close_ts)
        if ts_arr[i].time() >= FORCE_CLOSE_TIME:
            return (sign * (c[i] - entry_price), "FORCE_CLOSE", bar_close_ts)
        prev_yellow = yellow_val
    return (sign * (c[-1] - entry_price), "EOD", ts_arr[-1] + pd.Timedelta(minutes=BAR_MINUTES))


def attach_gamma(cands):
    if not EOD_MQ.exists():
        cands = cands.copy(); cands["gamma_sign"] = np.nan; return cands
    eod = pd.read_parquet(EOD_MQ)
    eod["date"] = pd.to_datetime(eod["date"]).dt.date
    eod = eod.set_index("date").sort_index()
    dates = sorted(eod.index.tolist())
    def prior_mq(d):
        prev = None
        for md in dates:
            if md < d: prev = md
            else: break
        return prev
    col = "qqq_gamma_sign" if "qqq_gamma_sign" in eod.columns else None
    cands = cands.copy()
    cands["entry_date"] = pd.to_datetime(cands["entry_time_et"]).dt.date
    if col is None:
        cands["gamma_sign"] = np.nan
    else:
        g = {d: (eod.loc[prior_mq(d), col] if prior_mq(d) in eod.index else np.nan)
             for d in cands["entry_date"].unique()}
        cands["gamma_sign"] = cands["entry_date"].map(g)
    return cands


def filter_candidates(cands):
    cands = cands.copy()
    cands["entry_hour"] = pd.to_datetime(cands["entry_time_et"]).dt.hour
    keep = cands["entry_hour"].isin(ALLOWED_HOURS)
    if DROP_POS_SHORT:
        keep &= ~((cands["gamma_sign"] == 1) & (cands["direction"] == "SHORT"))
    return cands[keep].reset_index(drop=True)


def run_chained(cands, bars20, atr_mode):
    rows = []
    last_exit = pd.Timestamp(0, tz="America/New_York")
    for _, t in cands.iterrows():
        ent_t = pd.Timestamp(t["entry_time_et"])
        if ent_t <= last_exit: continue
        ex = simulate_exit(t["direction"], ent_t, float(t["entry_price"]), bars20, atr_mode=atr_mode)
        if ex is None: continue
        pnl, reason, exit_ts = ex
        rows.append({"entry_ts": ent_t, "exit_ts": exit_ts,
                     "direction": t["direction"], "reason": reason, "pnl": pnl})
        last_exit = pd.Timestamp(exit_ts)
    return pd.DataFrame(rows)


def apply_mart_fc(df):
    sizes = []; cur = 1
    for pnl, reason in zip(df["pnl"].values, df["reason"].values):
        sizes.append(cur)
        if cur == 2: cur = 1
        else: cur = 2 if (pnl < 0 and reason == "FORCE_CLOSE") else 1
    return np.array(sizes)


def stats(pnls):
    arr = np.asarray(pnls)
    n = len(arr)
    if n == 0: return {"n":0,"wr":0,"pf":0,"total":0,"mdd":0,"sharpe":0}
    wins = arr[arr > 0]; losses = arr[arr < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    wr = (arr > 0).mean() * 100
    eq = np.cumsum(arr); peak = np.maximum.accumulate(eq); mdd = (eq - peak).min()
    return {"n":n,"wr":wr,"pf":pf,"total":arr.sum(),"mdd":mdd}


def main():
    print("loading 20-min bars + entry candidates...")
    bars20 = build_20min_bars()
    is_df  = pd.read_parquet(PARQUET_DIR / "entry_signal_trades.parquet")
    oos_df = pd.read_parquet(PARQUET_DIR / "entry_signal_trades_oos.parquet")
    is_cands  = filter_candidates(attach_gamma(filter_pre_dedupe(is_df)))
    oos_cands = filter_candidates(attach_gamma(filter_pre_dedupe(oos_df)))

    L = []
    L.append("=" * 200)
    L.append("LOCKED FILTERED CONFIG — ATR LOOKAHEAD AUDIT  (LEAKY vs FIXED)")
    L.append("=" * 200)
    L.append("V2 K=0.8 lock=0.45 + mart-FC + hours 9-14 + drop POS+SHORT")
    L.append("LEAKY = init_atr_y uses bar CONTAINING entry_ts (has 15 min of future data baked in)")
    L.append("FIXED = step back until bar's close <= entry_ts (no future leak)")
    L.append("")

    for atr_mode in ("leaky", "fixed"):
        print(f"running {atr_mode}...")
        is_t  = run_chained(is_cands,  bars20, atr_mode)
        oos_t = run_chained(oos_cands, bars20, atr_mode)
        all_t = pd.concat([is_t, oos_t], ignore_index=True).sort_values("entry_ts").reset_index(drop=True)
        sizes = apply_mart_fc(all_t)
        all_t["size"] = sizes
        all_t["scaled"] = all_t["pnl"] * sizes

        # Stats no-mart and mart, per period
        is_pnl  = all_t.loc[all_t["entry_ts"].dt.date.isin(is_t["entry_ts"].dt.date.unique()), "pnl"].values
        oos_pnl = all_t.loc[all_t["entry_ts"].dt.date.isin(oos_t["entry_ts"].dt.date.unique()), "pnl"].values
        is_scaled  = all_t.loc[all_t["entry_ts"].dt.date.isin(is_t["entry_ts"].dt.date.unique()), "scaled"].values
        oos_scaled = all_t.loc[all_t["entry_ts"].dt.date.isin(oos_t["entry_ts"].dt.date.unique()), "scaled"].values

        s_is_nm  = stats(is_pnl);   s_oos_nm = stats(oos_pnl);   s_all_nm = stats(all_t["pnl"].values)
        s_is_m   = stats(is_scaled);s_oos_m  = stats(oos_scaled);s_all_m  = stats(all_t["scaled"].values)

        L.append("")
        L.append(f"-- ATR mode: {atr_mode.upper()} --")
        L.append("  NO-MART (raw 1-contract):")
        for lab, s in [("IS", s_is_nm), ("OOS", s_oos_nm), ("ALL", s_all_nm)]:
            L.append(f"    {lab:<4} n={s['n']:>3} WR={s['wr']:>5.1f}% PF={s['pf']:>5.2f} "
                     f"total={s['total']:>+8.1f} pts (${s['total']*20:>+8,.0f} NQ / ${s['total']*2:>+7,.0f} MNQ) "
                     f"MDD={s['mdd']:>+7.1f}")
        L.append("  WITH MART-FC-ONLY (scaled PnL):")
        for lab, s in [("IS", s_is_m), ("OOS", s_oos_m), ("ALL", s_all_m)]:
            L.append(f"    {lab:<4} n={s['n']:>3} WR={s['wr']:>5.1f}% PF={s['pf']:>5.2f} "
                     f"total={s['total']:>+8.1f} pts (${s['total']*20:>+8,.0f} NQ / ${s['total']*2:>+7,.0f} MNQ) "
                     f"MDD={s['mdd']:>+7.1f}")

    # Re-run both modes for delta table
    L.append("")
    L.append("=" * 200)
    L.append("SUMMARY DELTA (LEAKY - FIXED)")
    L.append("=" * 200)
    is_l  = run_chained(is_cands,  bars20, "leaky")
    oos_l = run_chained(oos_cands, bars20, "leaky")
    all_l = pd.concat([is_l, oos_l], ignore_index=True).sort_values("entry_ts").reset_index(drop=True)
    sizes_l = apply_mart_fc(all_l); all_l["scaled"] = all_l["pnl"] * sizes_l

    is_f  = run_chained(is_cands,  bars20, "fixed")
    oos_f = run_chained(oos_cands, bars20, "fixed")
    all_f = pd.concat([is_f, oos_f], ignore_index=True).sort_values("entry_ts").reset_index(drop=True)
    sizes_f = apply_mart_fc(all_f); all_f["scaled"] = all_f["pnl"] * sizes_f

    def stats_period(df, kind):
        if kind == "IS":
            sub = df[df["entry_ts"].dt.date < dt.date(2025,1,1)]
        elif kind == "OOS":
            sub = df[df["entry_ts"].dt.date >= dt.date(2025,1,1)]
        else:
            sub = df
        return stats(sub["pnl"].values), stats(sub["scaled"].values)

    for kind in ("IS", "OOS", "ALL"):
        nm_l, m_l = stats_period(all_l, kind)
        nm_f, m_f = stats_period(all_f, kind)
        L.append(f"\n  {kind}:")
        L.append(f"    no-mart  LEAKY n={nm_l['n']:>3} PF={nm_l['pf']:.2f} total={nm_l['total']:>+8.1f} pts   "
                 f"FIXED n={nm_f['n']:>3} PF={nm_f['pf']:.2f} total={nm_f['total']:>+8.1f} pts   "
                 f"ΔPF={nm_f['pf']-nm_l['pf']:+.3f}  Δtotal={nm_f['total']-nm_l['total']:+.1f} pts")
        L.append(f"    mart     LEAKY n={m_l['n']:>3} PF={m_l['pf']:.2f} total={m_l['total']:>+8.1f} pts   "
                 f"FIXED n={m_f['n']:>3} PF={m_f['pf']:.2f} total={m_f['total']:>+8.1f} pts   "
                 f"ΔPF={m_f['pf']-m_l['pf']:+.3f}  Δtotal={m_f['total']-m_l['total']:+.1f} pts")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
