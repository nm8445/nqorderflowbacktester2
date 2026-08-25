"""Test 5 (direction permutation) split BY WINDOW — does the direction edge survive out-of-sample?

The existing test_5_direction_permutation.py scores the permutation on the FULL sample. That
answers "are the entries directional over 5 years", not "are they STILL directional in the OOS
window". Since the OOS window is what the live deploy is betting on, score it separately.

Method (unchanged from test 5): keep each strategy's entry bars and exit engine fixed, replace the
DIRECTION decision with a coin flip, re-run the exits, repeat N times. Real result should sit in
the top 1% of the permutation distribution. Median permutation != 0 because the exit engine itself
makes money on random directions — that is exactly why this control is needed.

Here every permutation returns PER-TRADE (date, pnl), so the same draws can be scored on IS and
OOS independently. Split at 2024-01-01, matching oos_live_vs_is_significance.py.

LIVE is NOT testable this way: a counterfactual short needs the engine re-run on the same session,
and the forward paper log has no such counterfactual. With n=115 it would be powerless anyway
(see project_oos_live_significance).

Run:  python scripts/overfit_framework/test_5_by_window.py [n_perms]
"""
from __future__ import annotations

import datetime as dt
import multiprocessing as mp
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

ROOT = Path("C:/trading/nqorderflowbacktester")
SPLIT = dt.date(2024, 1, 1)
N_PERMS = 1000

RV_DIR = ROOT / "scripts" / "rough vol orderflow"
OD_DIR = ROOT / "scripts" / "overnight drift strategy"
FB_DIR = ROOT / "scripts" / "fabio_orb"


# ---------------------------------------------------------------- reporting
def score(real_is, real_oos, perm_is, perm_oos, name, unit="$"):
    print("=" * 78)
    print(f"{name}")
    print("=" * 78)
    print(f"{'window':<8}{'real':>13}{'perm med':>13}{'perm 95%':>13}{'beats':>9}{'p':>9}  verdict")
    print("-" * 78)
    out = {}
    for w, real, perms in [("IS", real_is, perm_is), ("OOS", real_oos, perm_oos)]:
        perms = np.asarray(perms, float)
        pct = float((perms < real).mean())
        p = 1.0 - pct
        out[w] = p
        print(f"{w:<8}{real:>13,.0f}{np.median(perms):>13,.0f}"
              f"{np.quantile(perms,0.95):>13,.0f}{pct*100:>8.1f}%{p:>9.4f}"
              f"  {'PASS' if p < 0.01 else ('weak' if p < 0.05 else 'FAIL')}")
    print()
    return out


# ---------------------------------------------------------------- FB
def fb_prep():
    sys.path.insert(0, str(FB_DIR))
    from run_overfit_giveback import GB, run_gb
    from run_giveback_variant import load_days
    days = load_days()
    keys = sorted(days)
    L, S, D = [], [], []
    for d in keys:
        rl = run_gb(days[d], 1, **GB)
        if rl is None:
            continue
        rs = run_gb(days[d], -1, **GB)
        L.append(rl["net_dollars"])
        S.append(rs["net_dollars"] if rs else 0.0)
        D.append(pd.Timestamp(d).date())
    return np.array(L), np.array(S), np.array(D)


def fb_run(n_perms):
    L, S, D = fb_prep()
    m_is = D < SPLIT
    rng = np.random.default_rng(44)
    pis = np.empty(n_perms)
    pos = np.empty(n_perms)
    for k in range(n_perms):
        coin = rng.integers(0, 2, len(L))
        pnl = np.where(coin == 1, L, S)
        pis[k] = pnl[m_is].sum()
        pos[k] = pnl[~m_is].sum()
    return score(L[m_is].sum(), L[~m_is].sum(), pis, pos, "FB (Fabio ORB, giveback) — real = all long")


# ---------------------------------------------------------------- RV
_RV = None


def _rv_init():
    global _RV
    sys.path.insert(0, str(RV_DIR))
    import test_5_direction_permutation as T5
    sig, b = T5.rv_extract_signals()
    _RV = (sig, b, T5)


def _rv_sim(signals, b, directions, T5):
    """Copy of T5.rv_simulate_with_directions that also records the trade's DATE."""
    import core as rv_core
    highs, lows, closes = b["highs"], b["lows"], b["closes"]
    mod, atr, di = b["minutes_of_day"], b["atr"], b["day_idx"]
    sig_map = {s[0]: directions[k] for k, s in enumerate(signals)}
    pos = 0
    ep = sl_p = tp_p = 0.0
    ent_day = 0
    rows = []
    for i in range(len(closes)):
        m = mod[i]
        in_sess = T5.ENTRY_START_MIN <= m < T5.SESSION_END_MIN
        if pos != 0 and m >= T5.SESSION_END_MIN:
            rows.append((ent_day, (closes[i] - ep) * pos)); pos = 0
        if not in_sess:
            continue
        if pos != 0:
            exited, xp = False, 0.0
            if pos == 1:
                if lows[i] <= sl_p: xp, exited = sl_p, True
                elif highs[i] >= tp_p: xp, exited = tp_p, True
            else:
                if highs[i] >= sl_p: xp, exited = sl_p, True
                elif lows[i] <= tp_p: xp, exited = tp_p, True
            if exited:
                rows.append((ent_day, (xp - ep) * pos)); pos = 0
                continue
        if pos == 0 and not (T5.SKIP_START_MIN <= m < T5.SKIP_END_MIN) and i in sig_map:
            d = sig_map[i]
            if d == 0:
                continue
            a = atr[i]
            if a <= 0:
                continue
            pos, ep, ent_day = d, closes[i], di[i]
            sl_p = ep - d * T5.RV_SL_ATR * a
            tp_p = ep + d * T5.RV_TP_ATR * a
    pv = rv_core.POINT_VALUE
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows]) * pv


def _rv_perm(seed):
    sig, b, T5 = _RV
    rng = np.random.default_rng(seed)
    dirs = np.zeros(len(sig), dtype=np.int8)
    for k, (idx, al, ash) in enumerate(sig):
        if al and ash:
            dirs[k] = 1 if rng.random() < 0.5 else -1
        elif al:
            dirs[k] = 1 if rng.random() < 0.5 else 0
        elif ash:
            dirs[k] = -1 if rng.random() < 0.5 else 0
    days, pnl = _rv_sim(sig, b, dirs, T5)
    m = np.array([dt.date.fromordinal(int(x)) < SPLIT for x in days])
    return pnl[m].sum(), pnl[~m].sum()


def rv_run(n_perms, workers):
    _rv_init()
    sig, b, T5 = _RV
    import core as rv_core
    ema = rv_core.compute_ema(b["closes"], T5.RV_EMA_LEN)
    closes = b["closes"]
    real = np.zeros(len(sig), dtype=np.int8)
    for k, (idx, al, ash) in enumerate(sig):
        if closes[idx] > ema[idx] and al: real[k] = 1
        elif closes[idx] < ema[idx] and ash: real[k] = -1
    days, pnl = _rv_sim(sig, b, real, T5)
    m = np.array([dt.date.fromordinal(int(x)) < SPLIT for x in days])
    with mp.Pool(workers, initializer=_rv_init) as pool:
        res = list(pool.imap_unordered(_rv_perm, range(n_perms), chunksize=8))
    return score(pnl[m].sum(), pnl[~m].sum(),
                 [r[0] for r in res], [r[1] for r in res], "RV (Rough Vol orderflow)")


# ---------------------------------------------------------------- OD
_OD = None


def _od_init():
    global _OD
    sys.path.insert(0, str(OD_DIR))
    import overnight_drift_strategy as ods
    bars = ods.build_full_20min_series("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet",
                                       "D:/trading_pythonbacktest_data/timebars_5min")
    atr = ods.rma_atr(bars["high"], bars["low"], bars["close"], 14)
    _OD = (bars, atr.values)


def _od_sim(rng):
    """Live 20-min OD geometry; records each trade's entry DATE."""
    bars, atr = _OD
    idx = bars.index
    o, h, l, c = (bars[x].values for x in ("open", "high", "low", "close"))
    YM, GM, GB_, GD, RI, RD = 1.30, 1.00, 82.5, 1.5, 0.0, 0.45
    pos = 0
    ep = 0.0
    prev_y = np.nan
    nb = 0
    ent = None
    rows = []
    for i in range(len(idx)):
        t = idx[i].time()
        a = atr[i]
        if pos != 0:
            nb += 1
            s = pos
            raw = c[i] - s * YM * a if not np.isnan(a) else np.nan
            y = raw if np.isnan(prev_y) else (max(prev_y, raw) if s > 0 else min(prev_y, raw))
            red = ep + s * (RI + RD * nb)
            grn = red + s * (GB_ - GD * nb) + s * (GM * a if not np.isnan(a) else 0.0)
            ex = False
            if not np.isnan(grn) and ((s == 1 and h[i] >= grn) or (s == -1 and l[i] <= grn)):
                ex = True
            if not ex and not np.isnan(y) and ((s == 1 and c[i] <= y and c[i] < o[i]) or
                                               (s == -1 and c[i] >= y and c[i] > o[i])):
                ex = True
            if not ex and t == dt.time(8, 0):
                ex = True
            if ex:
                rows.append((ent, (c[i] - ep) * s * 20.0))
                pos, prev_y, nb = 0, np.nan, 0
                continue
            prev_y = y
        if pos == 0 and t == dt.time(19, 0):
            pos = 1 if rng.random() < 0.5 else -1
            ep, prev_y, nb, ent = c[i], np.nan, 0, idx[i].date()
    d = np.array([r[0] for r in rows])
    p = np.array([r[1] for r in rows])
    return d, p


class _AllLong:
    def random(self):
        return 0.0


def _od_perm(seed):
    d, p = _od_sim(np.random.default_rng(seed))
    m = d < SPLIT
    return p[m].sum(), p[~m].sum()


def od_run(n_perms, workers):
    _od_init()
    d, p = _od_sim(_AllLong())
    m = d < SPLIT
    with mp.Pool(workers, initializer=_od_init) as pool:
        res = list(pool.imap_unordered(_od_perm, range(n_perms), chunksize=4))
    return score(p[m].sum(), p[~m].sum(),
                 [r[0] for r in res], [r[1] for r in res], "OD (Overnight Drift, live 20-min) — real = all long")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_PERMS
    w = max(1, min(6, (mp.cpu_count() or 2) - 1))
    print(f"Direction permutation BY WINDOW — split {SPLIT}, {n} perms, {w} workers\n")
    t0 = time.time()
    fb_run(n)
    rv_run(n, w)
    od_run(n, w)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    mp.freeze_support()
    main()
