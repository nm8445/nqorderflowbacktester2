"""Is the user's OD 1hr candidate overfit to the current regime?

CONFIG UNDER TEST (U) — as supplied (Pine input panel):
  entry 19:00 ET, force-close 14:00 ET
  yellow: atr_len 12, mult 1.05, pure_ratchet + giveback 0.33,
          scale_by_body FALSE, max_giveback 0.30, min_gap_floor 0.82
  green:  atr_len 4, mult 3.0, base 102.4, decay 0.17
  red:    intercept 0.0, drift 0.47
  geometry-only: martingale OFF, BE OFF, qty 1

NOTE ON U's SHAPE: the giveback floor is `yellow >= raw_yellow + min_gap*ATR`
and raw_yellow = close - mult*ATR, so U's EFFECTIVE trail is
(1.05 - 0.82) = 0.23 x ATR below the close — a difference of two fitted numbers.
Test 2 below perturbs both, because a config whose live behaviour depends on a
small difference of large fitted params is the classic fragile shape.

TESTS
  1  headline + IS/OOS + per-year + rolling-12mo PF vs live 20-min and vs the
     recorded rank-1 1hr config
  2  one-at-a-time parameter stability (repo Test-1 style: all-profitable,
     all-PF>1, min/max net relative to U)
  3  regime slices — volatility terciles (by the strategy's own ATR at entry),
     per-year, and pre/post 2024
  4  period-rank: random search, then where does U rank on the EARLY half vs the
     LATE half? plus IS->OOS rank correlation of the whole sample.
     (U ranked top on late data but mid-pack on early data == regime-overfit.)
  5  walk-forward: 24mo train / 6mo test, select on TRAIN only from the random
     sample, stitch tests, compare to U held fixed.

Usage:  python "scripts/overnight drift strategy/overfit_test_1hr_candidate.py" [n_trials]
Output -> scripts/overnight drift strategy/results/overfit_test_1hr_candidate.txt
"""
from __future__ import annotations

import random
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from overnight_drift_strategy import (  # noqa: E402
    StrategyParams, run_backtest, trades_to_df, rma_atr,
)
from sweep_1hr_timeframe import build_series  # noqa: E402

OUT_DIR = HERE / "results"
ET = "America/New_York"
IS_END = pd.Timestamp("2024-01-01", tz=ET)

# ---- config under test ----
U = dict(force_hour=14, y_len=12, y_mult=1.05, gb=0.33, sbb=False, max_gb=0.30,
         min_gap=0.82, g_len=4, g_mult=3.0, g_base=102.4, g_decay=0.17, red_drift=0.47)

# recorded rank-1 1hr config (project_od_1hr_config.md)
R1 = dict(force_hour=14, y_len=7, y_mult=0.99, gb=0.33, sbb=False, max_gb=0.79,
          min_gap=0.36, g_len=7, g_mult=3.40, g_base=102.4, g_decay=0.17, red_drift=0.45)

# live 20-min production geometry (run on 20-min bars)
LIVE20 = dict(force_hour=8, y_len=14, y_mult=1.30, gb=0.0, sbb=True, max_gb=0.75,
              min_gap=0.0, g_len=14, g_mult=1.00, g_base=82.5, g_decay=1.50, red_drift=0.45)

_BARS: pd.DataFrame | None = None


def to_params(d: dict) -> StrategyParams:
    return StrategyParams(
        entry_hour=19, entry_minute=0, forced_hour=int(d["force_hour"]), forced_minute=0,
        yellow_atr_len=int(d["y_len"]), yellow_atr_mult=float(d["y_mult"]),
        yellow_drift=0.0, yellow_mode="pure_ratchet",
        yellow_giveback=float(d["gb"]), scale_by_body=bool(d["sbb"]),
        max_giveback_atr=float(d["max_gb"]), giveback_min_gap_atr=float(d["min_gap"]),
        green_atr_len=int(d["g_len"]), green_atr_mult=float(d["g_mult"]),
        green_base=float(d["g_base"]), green_decay=float(d["g_decay"]),
        red_intercept=0.0, red_drift=float(d["red_drift"]),
        use_be=False, tp_intrabar_fill=False, yellow_suppress_bars=0,
        use_martingale=False, base_qty=1, loss_qty=1,
    )


def backtest(d: dict, bars=None) -> pd.DataFrame:
    b = bars if bars is not None else _BARS
    df = trades_to_df(run_backtest(b, to_params(d)))
    if df.empty:
        return df
    df["entry_time"] = pd.to_datetime(df["entry_time"]).dt.tz_convert(ET)
    df["date"] = df["entry_time"].dt.date
    df["year"] = df["entry_time"].dt.year
    df["period"] = np.where(df["entry_time"] < IS_END, "IS", "OOS")
    return df


def m(df: pd.DataFrame) -> dict:
    if df is None or len(df) == 0:
        return dict(n=0, net=0.0, pf=0.0, wr=0.0, mdd=0.0, sharpe=0.0, worst=0.0)
    p = df["pnl_dollars"].values
    gw = p[p > 0].sum(); gl = abs(p[p < 0].sum())
    eq = np.cumsum(p)
    daily = pd.Series(p, index=pd.to_datetime(df["date"].values)).groupby(level=0).sum()
    return dict(n=len(p), net=p.sum(), pf=(gw / gl) if gl > 0 else float("inf"),
                wr=(p > 0).mean() * 100, mdd=(eq - np.maximum.accumulate(eq)).min(),
                sharpe=(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0,
                worst=p.min())


def line(tag, df, width=30):
    a, i, o = m(df), m(df[df.period == "IS"]) if len(df) else m(df), \
        m(df[df.period == "OOS"]) if len(df) else m(df)
    return (f"  {tag:<{width}} n={a['n']:<5} net=${a['net']:>+10,.0f}  pf={a['pf']:>5.3f}  "
            f"wr={a['wr']:>5.1f}%  sharpe={a['sharpe']:>5.2f}  mdd=${a['mdd']:>+9,.0f}  "
            f"| IS pf={i['pf']:>5.3f} (${i['net']:>+9,.0f})  OOS pf={o['pf']:>5.3f} (${o['net']:>+9,.0f})")


# ---------------- worker plumbing ----------------
def _init(bars):
    global _BARS
    _BARS = bars


def _work(d):
    df = backtest(d)
    a, i, o = m(df), m(df[df.period == "IS"]) if len(df) else m(df), \
        m(df[df.period == "OOS"]) if len(df) else m(df)
    return {**d, "n": a["n"], "net": a["net"], "pf": a["pf"], "sharpe": a["sharpe"],
            "mdd": a["mdd"], "wr": a["wr"],
            "is_n": i["n"], "is_net": i["net"], "is_pf": i["pf"],
            "oos_n": o["n"], "oos_net": o["net"], "oos_pf": o["pf"],
            "min_pf": min(i["pf"], o["pf"])}


def _work_folds(d):
    """Full-history run, then net per 6-month bucket — for the walk-forward."""
    df = backtest(d)
    out = {**d, "buckets": {}}
    if len(df):
        b = df.set_index("entry_time").resample("6MS")["pnl_dollars"]
        out["buckets"] = {str(k.date()): (float(v.sum()), int(v.count()),
                                          float(v[v > 0].sum()),
                                          float(abs(v[v < 0].sum())))
                          for k, v in b}
    return out


def sample_near(rng: random.Random) -> dict:
    """Random draw. 50% in a tight neighbourhood of U, 50% wide — so U's rank is
    measured against both close cousins and the whole space."""
    tight = rng.random() < 0.5
    if tight:
        j = lambda v, f: round(v * rng.uniform(1 - f, 1 + f), 3)
        return dict(force_hour=rng.choice([13, 14, 15]),
                    y_len=rng.choice([10, 12, 14, 16]), y_mult=j(1.05, 0.30),
                    gb=j(0.33, 0.40), sbb=rng.random() < 0.3, max_gb=j(0.30, 0.50),
                    min_gap=round(rng.uniform(0.55, 0.95), 3),
                    g_len=rng.choice([4, 5, 7, 10]), g_mult=j(3.0, 0.30),
                    g_base=j(102.4, 0.30), g_decay=j(0.17, 0.80), red_drift=0.47)
    return dict(force_hour=rng.choice([8, 10, 11, 12, 13, 14, 15, 16]),
                y_len=rng.choice([7, 10, 12, 14, 16, 20, 25]),
                y_mult=round(rng.uniform(0.75, 3.0), 2),
                gb=0.0 if rng.random() < 0.15 else round(rng.uniform(0.1, 0.9), 2),
                sbb=rng.random() < 0.5, max_gb=round(rng.uniform(0.25, 1.5), 2),
                min_gap=round(rng.uniform(0.0, 0.95), 2),
                g_len=rng.choice([4, 5, 7, 10, 14, 20]),
                g_mult=round(rng.uniform(0.5, 4.0), 2),
                g_base=round(rng.uniform(20.0, 200.0), 1),
                g_decay=round(rng.uniform(0.0, 3.0), 2), red_drift=0.47)


# ---------------- test 2: one-at-a-time stability ----------------
PERTURB = {
    "y_mult":   [0.85, 0.95, 1.05, 1.15, 1.25, 1.45],
    "min_gap":  [0.50, 0.65, 0.75, 0.82, 0.88, 0.95],
    "y_len":    [7, 10, 12, 14, 16, 20],
    "gb":       [0.15, 0.25, 0.33, 0.45, 0.60, 0.80],
    "max_gb":   [0.15, 0.22, 0.30, 0.45, 0.70, 1.00],
    "g_len":    [3, 4, 5, 7, 10, 14],
    "g_mult":   [2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
    "g_base":   [70.0, 85.0, 102.4, 120.0, 140.0, 165.0],
    "g_decay":  [0.0, 0.10, 0.17, 0.30, 0.60, 1.20],
    "red_drift": [0.30, 0.40, 0.47, 0.55, 0.70, 0.90],
    "force_hour": [10, 11, 12, 13, 14, 15, 16],
    "sbb":      [False, True],
}
# The effective trail is (y_mult - min_gap)*ATR; sweep it directly too.
EFF_TRAIL = [0.10, 0.15, 0.23, 0.32, 0.45, 0.60]


def main(n_trials=600):
    t0 = _time.time()
    OUT_DIR.mkdir(exist_ok=True)
    print("building 60-min + 20-min bar series...")
    b60 = build_series("60min")
    b20 = build_series("20min")
    print(f"  60m {len(b60):,} bars {b60.index.min()} -> {b60.index.max()}")

    L = ["=" * 190,
         "OD 1-HOUR CANDIDATE — OVERFIT / REGIME-DEPENDENCE TEST",
         "=" * 190, "",
         "  U (under test): entry 19:00, force 14:00 | yellow len12 mult1.05 gb0.33 "
         "sbb=False max_gb0.30 min_gap0.82 | green len4 mult3.0 base102.4 decay0.17 | red_drift 0.47",
         f"  geometry-only (marti OFF, BE OFF, 1 NQ).  IS < 2024-01-01, OOS after. "
         f"Data {b60.index.min().date()} -> {b60.index.max().date()}",
         "",
         "  ** U's effective trail = (y_mult - min_gap) = 0.23 x ATR below close "
         "(giveback floor pins yellow that tight) **", ""]

    # ---------- TEST 1 ----------
    du = backtest(U, b60)
    dr = backtest(R1, b60)
    dl = backtest(LIVE20, b20)
    L += ["=" * 190, "TEST 1 — headline vs references", "=" * 190, "",
          line("U (candidate, 1hr)", du), line("rank-1 1hr (on record)", dr),
          line("LIVE 20-min production", dl), ""]
    L.append(f"  {'year':<6} {'U n':>5} {'U net':>11} {'U pf':>6} | {'R1 net':>11} {'R1 pf':>6} "
             f"| {'live20 net':>11} {'live20 pf':>9}")
    for y in sorted(set(du["year"]) | set(dl["year"])):
        a = m(du[du.year == y]); r = m(dr[dr.year == y]); lv = m(dl[dl.year == y])
        L.append(f"  {y:<6} {a['n']:>5} {a['net']:>+11,.0f} {a['pf']:>6.2f} | "
                 f"{r['net']:>+11,.0f} {r['pf']:>6.2f} | {lv['net']:>+11,.0f} {lv['pf']:>9.2f}")
    L.append("")
    roll = du.set_index("entry_time")["pnl_dollars"].resample("MS").sum().rolling(12).sum()
    rv = roll.dropna()
    L.append(f"  rolling 12-month net $ (U): min ${rv.min():+,.0f} @ {rv.idxmin().date()}   "
             f"max ${rv.max():+,.0f} @ {rv.idxmax().date()}   "
             f"months negative: {int((rv < 0).sum())}/{len(rv)}")
    L.append("  " + "  ".join(f"{k.date()}:{v/1000:+.0f}k" for k, v in rv.items() if k.month in (1, 7)))
    L.append("")

    # ---------- TEST 3 (regime) ----------
    atr = rma_atr(b60["high"], b60["low"], b60["close"], U["y_len"])
    du2 = du.copy()
    du2["atr_at_entry"] = atr.reindex(du2["entry_time"]).values
    ok = du2["atr_at_entry"].notna()
    q = du2.loc[ok, "atr_at_entry"].quantile([1/3, 2/3]).values
    L += ["=" * 190, "TEST 3 — regime slices", "=" * 190, "",
          f"  volatility terciles by 60-min ATR({U['y_len']}) at entry "
          f"(cuts {q[0]:.0f} / {q[1]:.0f} pts)"]
    L.append(f"  {'regime':<22} {'n':>5} {'net':>11} {'pf':>6} {'wr':>7} {'mdd':>10}")
    for name, mask in (("LOW vol", du2.atr_at_entry <= q[0]),
                       ("MID vol", (du2.atr_at_entry > q[0]) & (du2.atr_at_entry <= q[1])),
                       ("HIGH vol", du2.atr_at_entry > q[1])):
        s = m(du2[mask & ok])
        L.append(f"  {name:<22} {s['n']:>5} {s['net']:>+11,.0f} {s['pf']:>6.3f} "
                 f"{s['wr']:>6.1f}% {s['mdd']:>+10,.0f}")
    for nm, sub in (("pre-2024 (IS)", du[du.period == "IS"]), ("2024+ (OOS)", du[du.period == "OOS"])):
        s = m(sub)
        L.append(f"  {nm:<22} {s['n']:>5} {s['net']:>+11,.0f} {s['pf']:>6.3f} "
                 f"{s['wr']:>6.1f}% {s['mdd']:>+10,.0f}")
    L.append("")

    # ---------- TEST 2 (stability) ----------
    jobs = []
    for k, vals in PERTURB.items():
        for v in vals:
            d = dict(U); d[k] = v; jobs.append(("param", k, v, d))
    for e in EFF_TRAIL:
        d = dict(U); d["min_gap"] = round(U["y_mult"] - e, 3)
        jobs.append(("eff_trail", "eff_trail", e, d))
    print(f"test 2: {len(jobs)} perturbations...")
    with ProcessPoolExecutor(max_workers=6, initializer=_init, initargs=(b60,)) as ex:
        res = list(ex.map(_work, [j[3] for j in jobs], chunksize=4))
    base = m(du)
    L += ["=" * 190,
          "TEST 2 — one-at-a-time parameter stability (everything else pinned at U). "
          "rel = net / U's net.",
          "=" * 190, "",
          f"  {'param':<12} {'values -> rel net (pf)':<118} {'min_rel':>8} {'all>0':>6} {'allPF>1':>8}"]
    for k in list(PERTURB) + ["eff_trail"]:
        rows = [(j[2], r) for j, r in zip(jobs, res) if j[1] == k]
        cells = "  ".join(f"{v}:{r['net']/base['net']:.2f}({r['pf']:.2f})" for v, r in rows)
        rels = [r["net"] / base["net"] for _, r in rows]
        L.append(f"  {k:<12} {cells:<118} {min(rels):>8.2f} "
                 f"{str(all(r['net'] > 0 for _, r in rows)):>6} "
                 f"{str(all(r['pf'] > 1 for _, r in rows)):>8}")
    L.append("")

    # ---------- TEST 4 + 5 (random search, period rank, walk-forward) ----------
    rng = random.Random(7)
    draws = [sample_near(rng) for _ in range(n_trials)]
    print(f"test 4/5: {len(draws)} random configs...")
    with ProcessPoolExecutor(max_workers=6, initializer=_init, initargs=(b60,)) as ex:
        rs = list(ex.map(_work, draws, chunksize=8))
    g = pd.DataFrame(rs)
    g = g[g["n"] >= 300]
    ur = _work_wrapper(U, b60)
    g_all = pd.concat([g, pd.DataFrame([ur])], ignore_index=True)
    n_tot = len(g_all)
    rank_is = int((g_all["is_net"] > ur["is_net"]).sum()) + 1
    rank_oos = int((g_all["oos_net"] > ur["oos_net"]).sum()) + 1
    rank_full = int((g_all["net"] > ur["net"]).sum()) + 1
    corr = g_all[["is_net", "oos_net"]].corr(method="spearman").iloc[0, 1]
    corr_pf = g_all[["is_pf", "oos_pf"]].corr(method="spearman").iloc[0, 1]
    L += ["=" * 190,
          f"TEST 4 — period rank: {n_tot} configs with >=300 trades (50% drawn near U, 50% wide)",
          "=" * 190, "",
          f"  U's rank on EARLY half (IS 2020-12..2023-12) net : {rank_is}/{n_tot} "
          f"(top {rank_is/n_tot*100:.0f}%)",
          f"  U's rank on LATE  half (OOS 2024-01..2026-06) net: {rank_oos}/{n_tot} "
          f"(top {rank_oos/n_tot*100:.0f}%)",
          f"  U's rank on FULL period net                      : {rank_full}/{n_tot} "
          f"(top {rank_full/n_tot*100:.0f}%)",
          "",
          f"  Spearman IS->OOS net corr across the sample : {corr:+.3f}",
          f"  Spearman IS->OOS PF  corr across the sample : {corr_pf:+.3f}",
          "    (positive => configs that worked early also work late; a good in-sample rank",
          "     carries information. negative => the search is fitting noise.)", ""]
    top_is = g_all.nlargest(10, "is_net")
    L.append("  top-10 by EARLY-half net, and how they did LATE:")
    L.append(f"  {'y_mult':>7} {'min_gap':>8} {'eff':>6} {'y_len':>6} {'g_len':>6} {'g_mult':>7} "
             f"{'g_base':>7} {'fh':>3} {'is_net':>10} {'oos_net':>10} {'full_pf':>8}")
    for _, r in top_is.iterrows():
        L.append(f"  {r.y_mult:>7.2f} {r.min_gap:>8.2f} {r.y_mult-r.min_gap:>6.2f} {int(r.y_len):>6} "
                 f"{int(r.g_len):>6} {r.g_mult:>7.2f} {r.g_base:>7.1f} {int(r.force_hour):>3} "
                 f"{r.is_net:>+10,.0f} {r.oos_net:>+10,.0f} {r.pf:>8.3f}")
    L.append("")

    # walk-forward on the same sample
    print("test 5: walk-forward...")
    with ProcessPoolExecutor(max_workers=6, initializer=_init, initargs=(b60,)) as ex:
        fb = list(ex.map(_work_folds, draws + [U], chunksize=8))
    buckets = sorted({k for r in fb for k in r["buckets"]})
    starts = [pd.Timestamp(b) for b in buckets]
    L += ["=" * 190,
          "TEST 5 — walk-forward: 24mo train / 6mo test / 6mo step, config picked on TRAIN PF only",
          "=" * 190, "",
          f"  {'train':<22} {'test':<12} {'picked eff-trail':>17} {'test_net':>10} "
          f"{'U_net':>10} {'WF-U':>10}"]
    wf_net = u_net = 0.0
    wf_rows = 0
    for i, ts in enumerate(starts):
        tr = [b for b in buckets if pd.Timestamp(b) < ts and pd.Timestamp(b) >= ts - pd.DateOffset(months=24)]
        if len(tr) < 4:
            continue
        best, bkey = None, None
        for r in fb:
            gw = sum(r["buckets"].get(b, (0, 0, 0, 0))[2] for b in tr)
            gl = sum(r["buckets"].get(b, (0, 0, 0, 0))[3] for b in tr)
            nn = sum(r["buckets"].get(b, (0, 0, 0, 0))[1] for b in tr)
            if nn < 120 or gl <= 0:
                continue
            k = gw / gl
            if bkey is None or k > bkey:
                bkey, best = k, r
        if best is None:
            continue
        tb = buckets[i]
        tn = best["buckets"].get(tb, (0.0, 0, 0, 0))[0]
        un = fb[-1]["buckets"].get(tb, (0.0, 0, 0, 0))[0]
        wf_net += tn; u_net += un; wf_rows += 1
        L.append(f"  {tr[0]}..{tb:<12} {tb:<12} "
                 f"{best['y_mult']-best['min_gap']:>17.2f} {tn:>+10,.0f} {un:>+10,.0f} "
                 f"{tn-un:>+10,.0f}")
    L.append(f"  STITCHED  walk-forward ${wf_net:>+12,.0f}   U held fixed ${u_net:>+12,.0f}   "
             f"diff ${wf_net-u_net:>+11,.0f}  over {wf_rows} test windows")
    L.append("")

    out = OUT_DIR / "overfit_test_1hr_candidate.txt"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out}\ntotal {_time.time()-t0:.0f}s")


def _work_wrapper(d, bars):
    global _BARS
    _BARS = bars
    return _work(d)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 600)
