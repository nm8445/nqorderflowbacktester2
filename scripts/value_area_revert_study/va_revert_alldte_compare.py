"""Re-run the VA-revert backtest using the QQQ ALL-EXPIRY (0-45 DTE) GEX levels
as the reaction-level pool, then compare to the existing 0-1 DTE results.

Only the QQQ g1-g10 source changes:
  0-1 DTE (existing): qqq_g{1..10}_nq         (the cached va_revert_signals.parquet uses these)
  0-45 DTE (new):     qqq_g{1..10}_alldte_nq

CR, PS, HVL, HVL_0DTE, HVL_extended, and all NDX levels stay the same in both
versions to keep the comparison clean (only the QQQ gex source flips).

Outputs:
  parquets/va_revert_signals_alldte.parquet
  tradelogs/robust_configs/va_revert_alldte_vs_0dte.txt
"""
from __future__ import annotations

import datetime as dt
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_SCRIPTS = Path(__file__).parent.parent / "overnight range strat" / "scripts"
sys.path.insert(0, str(PROJECT_SCRIPTS))
from range_break_entry_signal_study import (
    compute_windowed_absorption, load_range_per_day, load_5min_features,
)

THIS_DIR     = Path(__file__).parent
SIGS_OLD     = THIS_DIR / "va_revert_signals.parquet"            # 0-1 DTE (existing)
SIGS_ALL     = THIS_DIR / "va_revert_signals_alldte.parquet"     # 0-45 DTE (new)
VA_PARQUET   = THIS_DIR / "prev_day_rth_va.parquet"
MQ_PATH      = Path("D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet")
M1_BARS      = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
OUT_TXT      = Path(__file__).parent.parent / "overnight range strat" / "tradelogs" / "robust_configs" / "va_revert_alldte_vs_0dte.txt"
ET           = "America/New_York"

RTH_START   = dt.time(9, 30)
RTH_END     = dt.time(16, 0)
FORCE_CLOSE = dt.time(16, 0)
TICK        = 0.25
ATR_PERIOD  = 14
OVERSHOOT_LIMIT_PCT = 75.0
MIN_VA_WIDTH = 20.0
MIN_RTH_BARS = 30
N_GRID       = [5, 10, 15, 20]
SIG_D_GRID   = [0, 50, 100, 150]
CONF_D_GRID  = [0, 50, 100, 150]
SL_MULT_GRID = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
MIN_TRADES   = 30


def load_mq_with_filter(version: str) -> pd.DataFrame:
    """Return mq DataFrame keeping only the columns relevant for the chosen
    QQQ gex source ('0dte' = qqq_g{1..10}_nq; 'alldte' = qqq_g{1..10}_alldte_nq)."""
    df = pd.read_parquet(MQ_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    keep = []
    for c in df.columns:
        if not c.endswith("_nq"): continue
        if not (c.startswith("qqq_") or c.startswith("ndx_")): continue
        # QQQ g1..g10 source switch
        if c.startswith("qqq_g") and "_alldte" not in c:
            if version == "alldte":
                continue       # drop 0-1 DTE QQQ g levels for alldte version
        if c.startswith("qqq_g") and "_alldte" in c:
            if version == "0dte":
                continue       # drop alldte QQQ g levels for 0-1 DTE version
        keep.append(c)
    return df.set_index("date")[keep]


def levels_for_date(mq: pd.DataFrame, d: dt.date) -> np.ndarray:
    if d not in mq.index: return np.array([])
    row = mq.loc[d]
    return row.dropna().values.astype(float)


def load_premarket_open():
    df = pd.read_parquet(M1_BARS, columns=["close"])
    idx = pd.to_datetime(df.index, utc=True).tz_convert(ET)
    df = df.set_index(idx).sort_index()
    df["date"] = df.index.date
    df["t"]    = df.index.time
    pre = df[df["t"] < RTH_START].copy()
    last = pre.groupby("date").tail(1)
    return last[["date", "close"]].rename(columns={"close": "open_price"}).set_index("date")


def build_qualifying_days(va, pre):
    """Same logic as the original backtest."""
    dates = va.dropna(subset=["vah","val"]).sort_values("date")["date"].tolist()
    prev_lookup = {dates[i]: dates[i-1] for i in range(1, len(dates))}
    va_idx = va.set_index("date")
    rows = []
    for today in sorted(set(va["date"]).intersection(pre.index)):
        prev_d = prev_lookup.get(today)
        if prev_d is None or prev_d not in va_idx.index: continue
        vah_p = va_idx.loc[prev_d, "vah"]; val_p = va_idx.loc[prev_d, "val"]
        if not (np.isfinite(vah_p) and np.isfinite(val_p)): continue
        width = vah_p - val_p
        if width < MIN_VA_WIDTH: continue
        op = float(pre.loc[today, "open_price"])
        if not np.isfinite(op): continue
        if op > vah_p:
            direction = "SHORT"; distance = op - vah_p
        elif op < val_p:
            direction = "LONG"; distance = val_p - op
        else:
            continue
        pct = distance / width * 100
        if pct >= OVERSHOOT_LIMIT_PCT: continue
        rows.append({"date": today, "direction": direction,
                     "open_price": op, "vah_prev": float(vah_p),
                     "val_prev": float(val_p), "width_prev": float(width),
                     "distance_pct": pct,
                     "target": float(vah_p) if direction == "SHORT" else float(val_p)})
    return pd.DataFrame(rows)


def adaptive_atr(bars_day, atr_period=ATR_PERIOD):
    b = bars_day.copy()
    b["prev_close"] = b["close"].shift(1)
    b["tr"] = np.maximum.reduce([
        (b["high"] - b["low"]).values,
        (b["high"] - b["prev_close"]).abs().values,
        (b["low"]  - b["prev_close"]).abs().values,
    ])
    b["atr14"] = b["tr"].ewm(alpha=1/atr_period, adjust=False, min_periods=1).mean()
    return b


def build_signals(qual_df, bars_by_day, levels_by_day, ohi_olo_lookup, mq_levels_by_day):
    rows = []
    for _, row in qual_df.iterrows():
        d = row["date"]; direction = row["direction"]
        open_price = row["open_price"]; target = row["target"]
        bars = bars_by_day.get(d)
        if bars is None or len(bars) < MIN_RTH_BARS: continue
        if direction == "SHORT":
            ohi_today = ohi_olo_lookup.get(d, {}).get("ohi", np.nan)
            gex_today = mq_levels_by_day.get(d, np.array([]))
            pool = []
            if np.isfinite(ohi_today) and ohi_today > open_price: pool.append(ohi_today)
            pool.extend([lv for lv in gex_today if lv > open_price])
            pool = sorted(set(pool))
        else:
            olo_today = ohi_olo_lookup.get(d, {}).get("olo", np.nan)
            gex_today = mq_levels_by_day.get(d, np.array([]))
            pool = []
            if np.isfinite(olo_today) and olo_today < open_price: pool.append(olo_today)
            pool.extend([lv for lv in gex_today if lv < open_price])
            pool = sorted(set(pool), reverse=True)
        if not pool: continue
        pool_arr = np.array(pool, dtype=float)

        bars = adaptive_atr(bars).reset_index(drop=True)
        bars["body_low"]  = bars[["open","close"]].min(axis=1)
        bars["body_high"] = bars[["open","close"]].max(axis=1)
        levels_day = levels_by_day.get(d, pd.DataFrame())
        levels_by_bar = ({t: g for t, g in levels_day.groupby("bar_open_time")}
                          if not levels_day.empty else {})

        for idx in range(len(bars) - 2):
            bar = bars.iloc[idx]
            if direction == "SHORT":
                touched = pool_arr[(pool_arr <= float(bar["high"])) & (pool_arr >= float(bar["low"]))]
                if len(touched) == 0: continue
                lvl = float(touched[np.argmax(np.abs(touched - float(bar["high"])))])
            else:
                touched = pool_arr[(pool_arr >= float(bar["low"])) & (pool_arr <= float(bar["high"]))]
                if len(touched) == 0: continue
                lvl = float(touched[np.argmax(np.abs(touched - float(bar["low"])))])
            conf_bar  = bars.iloc[idx + 1]
            entry_bar = bars.iloc[idx + 2]
            if direction == "SHORT" and not (float(conf_bar["close"]) < float(conf_bar["open"])): continue
            if direction == "LONG"  and not (float(conf_bar["close"]) > float(conf_bar["open"])): continue
            atr_entry = float(conf_bar.get("atr14", np.nan))
            if not np.isfinite(atr_entry) or atr_entry <= 0: continue

            scan_dir = "SHORT" if direction == "SHORT" else "LONG"
            sig_lvls = levels_by_bar.get(bar["bar_open_time"], pd.DataFrame())
            cf_lvls  = levels_by_bar.get(conf_bar["bar_open_time"], pd.DataFrame())
            sig_win = compute_windowed_absorption(
                sig_lvls, float(bar["body_low"]), float(bar["body_high"]),
                scan_dir, N_GRID) if not sig_lvls.empty else {N:(None,None,0,0,0) for N in N_GRID}
            cf_win = compute_windowed_absorption(
                cf_lvls, float(conf_bar["body_low"]), float(conf_bar["body_high"]),
                scan_dir, N_GRID) if not cf_lvls.empty else {N:(None,None,0,0,0) for N in N_GRID}

            rec = {
                "date": d, "direction": direction,
                "signal_time": pd.Timestamp(bar["bar_open_time"]),
                "conf_time":   pd.Timestamp(conf_bar["bar_open_time"]),
                "entry_time":  pd.Timestamp(entry_bar["bar_open_time"]),
                "entry_idx":   int(idx + 2),
                "entry_price": float(entry_bar["open"]),
                "atr_at_entry": atr_entry,
                "level": lvl, "target": target,
                "vah_prev": row["vah_prev"], "val_prev": row["val_prev"],
                "distance_pct": row["distance_pct"],
            }
            for N in N_GRID:
                rec[f"sig_abs_w{N}"]  = abs(sig_win.get(N,(None,None,0,0,0))[3])
                rec[f"conf_abs_w{N}"] = abs(cf_win.get(N,(None,None,0,0,0))[3])
            rows.append(rec)
    return pd.DataFrame(rows)


def simulate_exit(direction, entry_price, entry_idx, bars_day, sl_pts, tp_price):
    sign = 1 if direction == "LONG" else -1
    sl_price = entry_price - sign * sl_pts
    n = len(bars_day)
    for k in range(entry_idx + 1, n):
        bar = bars_day.iloc[k]
        bt = pd.Timestamp(bar["bar_open_time"])
        if bt.time() >= FORCE_CLOSE:
            close_px = float(bars_day.iloc[k-1]["close"]) if k > 0 else float(bar["open"])
            return ("held->close", sign * (close_px - entry_price), bt)
        hi, lo = float(bar["high"]), float(bar["low"])
        if sign > 0:
            tp_hit = hi >= tp_price; sl_hit = lo <= sl_price
        else:
            tp_hit = lo <= tp_price; sl_hit = hi >= sl_price
        if tp_hit and sl_hit: return ("SL", sign*(sl_price-entry_price), bt)
        if tp_hit: return ("TP", sign*(tp_price-entry_price), bt)
        if sl_hit: return ("SL", sign*(sl_price-entry_price), bt)
    last = bars_day.iloc[-1]
    return ("held->close", sign*(float(last["close"])-entry_price),
            pd.Timestamp(last["bar_open_time"]))


def run_chained(sigs, bars_by_day, sl_mult):
    rows = []
    last_exit = pd.Timestamp(0, tz=ET)
    for _, t in sigs.sort_values("entry_time").iterrows():
        ent_t = pd.Timestamp(t["entry_time"])
        if ent_t <= last_exit: continue
        d = pd.to_datetime(t["date"]).date()
        bars_day = bars_by_day.get(d)
        if bars_day is None: continue
        entry_idx = int(t["entry_idx"])
        if entry_idx >= len(bars_day): continue
        entry_price = float(t["entry_price"]); atr = float(t["atr_at_entry"])
        sl_pts = sl_mult * atr; tp_price = float(t["target"])
        outcome, pnl, exit_t = simulate_exit(t["direction"], entry_price, entry_idx,
                                              bars_day, sl_pts, tp_price)
        rows.append({"pnl": pnl, "outcome": outcome, "direction": t["direction"]})
        last_exit = pd.Timestamp(exit_t)
    return pd.DataFrame(rows)


def stats(df):
    n = len(df)
    if n == 0: return {"n":0,"wr":0,"pf":0,"total":0,"mdd":0}
    pnl = df["pnl"].values
    wins = pnl[pnl>0]; losses = pnl[pnl<0]
    wr = (pnl>0).mean()*100
    pf = wins.sum()/abs(losses.sum()) if losses.sum() != 0 else float("inf")
    eq = np.cumsum(pnl); peak = np.maximum.accumulate(eq); mdd = (eq-peak).min()
    return {"n":n,"wr":wr,"pf":pf,"total":pnl.sum(),"mdd":mdd}


def sweep(sigs, bars_by_day, label):
    is_sigs  = sigs[sigs["date"] <  dt.date(2025,1,1)].copy()
    oos_sigs = sigs[sigs["date"] >= dt.date(2025,1,1)].copy()
    rows = []
    for N, sig_D, conf_D in product(N_GRID, SIG_D_GRID, CONF_D_GRID):
        sig_c  = f"sig_abs_w{N}"; conf_c = f"conf_abs_w{N}"
        sub_is  = is_sigs[(is_sigs[sig_c].fillna(0)  >= sig_D) &
                           (is_sigs[conf_c].fillna(0) >= conf_D)]
        sub_oos = oos_sigs[(oos_sigs[sig_c].fillna(0)  >= sig_D) &
                            (oos_sigs[conf_c].fillna(0) >= conf_D)]
        if len(sub_is) + len(sub_oos) < MIN_TRADES // 2: continue
        for sl in SL_MULT_GRID:
            it = run_chained(sub_is,  bars_by_day, sl)
            ot = run_chained(sub_oos, bars_by_day, sl)
            at = pd.concat([it, ot], ignore_index=True)
            s_is, s_oos, s_all = stats(it), stats(ot), stats(at)
            rows.append({"src":label,"N":N,"sig_D":sig_D,"conf_D":conf_D,"sl":sl,
                         **{f"is_{k}":v for k,v in s_is.items()},
                         **{f"oos_{k}":v for k,v in s_oos.items()},
                         **{f"all_{k}":v for k,v in s_all.items()}})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["min_pf"] = df[["is_pf","oos_pf"]].min(axis=1)
    return df


def main():
    print("loading shared data ...")
    va = pd.read_parquet(VA_PARQUET)
    va["date"] = pd.to_datetime(va["date"]).dt.date
    pre = load_premarket_open()
    qual = build_qualifying_days(va, pre)
    print(f"  qualifying days: {len(qual)}")
    rng = load_range_per_day()
    ohi_olo = {d: {"ohi": float(r.get("overnight_high", np.nan)),
                   "olo": float(r.get("overnight_low",  np.nan))}
               for d, r in rng.iterrows()}

    print("loading 5-min features (slow) ...")
    bars_all, levels_all = load_5min_features((dt.date(2020,12,1), dt.date(2026,5,7)))
    bars_by_day   = {d: g.sort_values("bar_open_time").reset_index(drop=True)
                     for d, g in bars_all.groupby("session_date")}
    levels_by_day = {d: g for d, g in levels_all.groupby("session_date")}

    # ALLDTE build (0-45 DTE QQQ gex)
    if SIGS_ALL.exists():
        print(f"loading cached alldte signals from {SIGS_ALL} ...")
        sigs_all = pd.read_parquet(SIGS_ALL)
    else:
        print("building alldte signals (using qqq_g{1..10}_alldte_nq) ...")
        mq_all = load_mq_with_filter("alldte")
        mq_by_day_all = {d: levels_for_date(mq_all, d) for d in bars_by_day.keys()}
        sigs_all = build_signals(qual, bars_by_day, levels_by_day, ohi_olo, mq_by_day_all)
        sigs_all.to_parquet(SIGS_ALL)
        print(f"  saved {len(sigs_all)} signals to {SIGS_ALL}")
    sigs_all["date"] = pd.to_datetime(sigs_all["date"]).dt.date
    sigs_all["entry_time"] = pd.to_datetime(sigs_all["entry_time"])

    # 0-1 DTE load (existing cache, but rebuild to be safe since menthorq parquet changed)
    print("rebuilding 0-1 DTE signals (filter to original qqq_g levels) ...")
    mq_0dte = load_mq_with_filter("0dte")
    mq_by_day_0 = {d: levels_for_date(mq_0dte, d) for d in bars_by_day.keys()}
    SIGS_0DTE = THIS_DIR / "va_revert_signals_0dte_v2.parquet"
    if SIGS_0DTE.exists():
        sigs_0 = pd.read_parquet(SIGS_0DTE)
    else:
        sigs_0 = build_signals(qual, bars_by_day, levels_by_day, ohi_olo, mq_by_day_0)
        sigs_0.to_parquet(SIGS_0DTE)
        print(f"  saved {len(sigs_0)} signals to {SIGS_0DTE}")
    sigs_0["date"] = pd.to_datetime(sigs_0["date"]).dt.date
    sigs_0["entry_time"] = pd.to_datetime(sigs_0["entry_time"])

    print(f"\nsignal counts:  0-1 DTE = {len(sigs_0)},  0-45 DTE alldte = {len(sigs_all)}")

    print("\nrunning sweep -- 0-1 DTE ...")
    df_0 = sweep(sigs_0, bars_by_day, "0dte")
    print(f"  {len(df_0)} combos done")
    print("running sweep -- 0-45 DTE alldte ...")
    df_a = sweep(sigs_all, bars_by_day, "alldte")
    print(f"  {len(df_a)} combos done")

    cols = ["src","N","sig_D","conf_D","sl",
            "is_n","is_wr","is_pf","is_total","is_mdd",
            "oos_n","oos_wr","oos_pf","oos_total","oos_mdd",
            "all_n","all_pf","all_total","min_pf"]

    L = []
    L.append("=" * 220)
    L.append("VA-REVERT: QQQ 0-1 DTE  vs  0-45 DTE (alldte) GEX-level pool comparison")
    L.append("=" * 220)
    L.append(f"Only QQQ g1-g10 source differs. CR/PS/HVL and all NDX levels identical in both.")
    L.append(f"Signal counts: 0-1 DTE={len(sigs_0)}  alldte={len(sigs_all)}")
    L.append(f"Sweep grid: N={N_GRID}, sig_D={SIG_D_GRID}, conf_D={CONF_D_GRID}, sl={SL_MULT_GRID}")
    L.append("")

    for label, sub in [("0-1 DTE", df_0), ("0-45 DTE (alldte)", df_a)]:
        L.append("=" * 220)
        L.append(f"{label} -- TOP 15 BY min(IS_PF, OOS_PF)  (n>=200 trades, both periods profitable)")
        L.append("=" * 220)
        if sub.empty:
            L.append("  (no combos)")
            continue
        qual = sub[(sub["is_n"] >= 200) & (sub["is_total"] > 0) & (sub["oos_total"] > 0)]
        L.append(f"  {len(qual)} qualifying configs")
        if not qual.empty:
            L.append(qual.sort_values("min_pf", ascending=False).head(15)[cols].to_string(
                index=False, float_format=lambda x: f"{x:.2f}"))
        L.append("")
        L.append(f"{label} -- TOP 10 BY combined total pts")
        L.append("=" * 220)
        if not qual.empty:
            L.append(qual.sort_values("all_total", ascending=False).head(10)[cols].to_string(
                index=False, float_format=lambda x: f"{x:.2f}"))
        L.append("")

    # Head-to-head: same (N, sig_D, conf_D, sl) config in both — what's the delta?
    if not df_0.empty and not df_a.empty:
        L.append("")
        L.append("=" * 220)
        L.append("HEAD-TO-HEAD: same (N, sig_D, conf_D, sl) -- 0-1 DTE vs alldte side by side")
        L.append("=" * 220)
        m = df_0.merge(df_a, on=["N","sig_D","conf_D","sl"], suffixes=("_0", "_a"))
        m["d_min_pf"]  = m["min_pf_a"] - m["min_pf_0"]
        m["d_total"]   = m["all_total_a"] - m["all_total_0"]
        m["d_is_n"]    = m["is_n_a"] - m["is_n_0"]
        m["d_oos_n"]   = m["oos_n_a"] - m["oos_n_0"]
        L.append(f"  {len(m)} matching combos")
        L.append(f"  Of those, alldte has HIGHER min_pf in {(m['d_min_pf'] > 0).sum()} combos, "
                 f"lower in {(m['d_min_pf'] < 0).sum()}, equal in {(m['d_min_pf'] == 0).sum()}")
        L.append(f"  Median min_pf delta (alldte - 0dte): {m['d_min_pf'].median():+.3f}")
        L.append(f"  Median total$ delta (alldte - 0dte): {m['d_total'].median():+.1f} pts")
        L.append(f"  Median trade-count delta IS:  {m['d_is_n'].median():+.0f}")
        L.append(f"  Median trade-count delta OOS: {m['d_oos_n'].median():+.0f}")
        L.append("")
        L.append("Top 15 (N, sig_D, conf_D, sl) combos by alldte_min_pf - 0dte_min_pf (where alldte wins):")
        win_cols = ["N","sig_D","conf_D","sl",
                     "is_n_0","is_pf_0","oos_pf_0","all_total_0","min_pf_0",
                     "is_n_a","is_pf_a","oos_pf_a","all_total_a","min_pf_a",
                     "d_min_pf","d_total"]
        L.append(m.sort_values("d_min_pf", ascending=False).head(15)[win_cols].to_string(
            index=False, float_format=lambda x: f"{x:.2f}"))
        L.append("")
        L.append("Top 15 where 0-1 DTE wins (alldte loses most):")
        L.append(m.sort_values("d_min_pf", ascending=True).head(15)[win_cols].to_string(
            index=False, float_format=lambda x: f"{x:.2f}"))

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}\n")
    # Truncate console output for brevity
    print("\n".join(L[:80]))


if __name__ == "__main__":
    main()
