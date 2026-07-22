"""Overfit suite for the FB GIVEBACK stop (k=1.5/gb0.3/min_gap0.3, drift_floor, scale_body).

Mirrors run_overfit_tests.py (the locked-config suite) but on the giveback exit, and — critically —
Test 1 sweeps the GIVEBACK param neighborhood (k x giveback x min_gap) to answer: is the sweep-picked
config a broad plateau or a lucky spike? The giveback exit is written via a signed transform so ONE
function handles long (sgn=+1) and short (sgn=-1); the short mirror is only needed for Test 5.

  1 Parameter Stability   — 3D neighborhood around (k, giveback, min_gap)   <- the key overfit test
  2 Walk-Forward          — non-overlapping 6-month windows
  3 MC Order Shuffle      — 10k trade-order permutations (DD robustness)
  4 Bootstrap CI          — 10k resamples (net/PF confidence, P(losing))
  5 Direction Permutation — random long/short per entry (edge-is-real, tests the ENTRY)

Run:  python scripts/fabio_orb/run_overfit_giveback.py
"""
from __future__ import annotations
import warnings; warnings.simplefilter("ignore")
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_giveback_variant import load_days, find_entry   # noqa: E402

ET = "America/New_York"
TRADE_END_HHMM = 1400
DPP, SLIP_PTS, COMM = 80.0, 0.25, 5.0        # $/pt (20) ... wait: DPP set below
DPP = 5.0 / 0.25                              # = 20 $/pt
COST = 2 * SLIP_PTS * DPP + COMM              # = 15 $/round-trip, 1 contract
# LOCKED giveback config
GB = dict(k=1.5, mode="drift_floor", drift=0.0, gb=0.3, scale_body=True, max_gb=0.5, min_gap=0.3)


def run_gb(day, sgn, k, mode, drift, gb, scale_body, max_gb, min_gap):
    """Giveback exit in SIGNED space (u = sgn*price -> always long-like). sgn=+1 long, -1 short."""
    ent = find_entry(day)
    if ent is None: return None
    i, ep, orb_low, _tp_long, t0 = ent
    risk = ep - orb_low
    u_ep = sgn * ep
    u_floor = sgn * (orb_low if sgn == 1 else ep + risk)          # stop floor (below u_ep)
    u_tp = sgn * (ep + 4 * risk if sgn == 1 else ep - 4 * risk)   # 4R target
    op, hi, lo, cl = day["open"], day["high"], day["low"], day["close"]
    hhmm, atr, etime = day["hhmm"], day["atr"], day["close_et"]
    u_yellow = u_floor; prev_uy = np.nan; gb_on = gb > 0

    def out(u_xp, t1, reason):
        return {"net_dollars": (u_xp - u_ep) * DPP - COST, "entry_time": pd.Timestamp(t0),
                "exit_time": pd.Timestamp(t1), "reason": reason}

    for j in range(i + 1, len(hhmm)):
        a = atr[j] if atr[j] > 0 else 1e-6
        u_cl, u_op = sgn * cl[j], sgn * op[j]
        u_raw = u_cl - k * a
        prev_bear = (sgn * cl[j - 1]) < (sgn * op[j - 1])
        if gb_on and prev_bear and not np.isnan(prev_uy):
            gap = max(0.0, prev_uy - u_raw); frac = gb
            if scale_body:
                frac *= min(1.0, ((sgn * op[j - 1]) - (sgn * cl[j - 1])) / a)
            cand = prev_uy - min(gap * frac, max_gb * a)
        elif mode == "pure_ratchet":
            cand = max(prev_uy, u_raw) if not np.isnan(prev_uy) else u_raw
        else:
            base = (prev_uy + drift) if not np.isnan(prev_uy) else u_raw
            cand = max(base, u_raw)
        if gb_on:
            cand = max(cand, u_raw + min_gap * a)
        u_yellow = max(cand, u_floor)
        u_fav = max(sgn * hi[j], sgn * lo[j])       # favorable extreme (signed)
        if u_fav >= u_tp: return out(u_tp, etime[j], "TP")
        if u_cl <= u_yellow and u_cl < u_op: return out(u_cl, etime[j], "YELLOW")
        if hhmm[j] >= TRADE_END_HHMM: return out(u_cl, etime[j], "EOD")
        prev_uy = u_yellow
    return out(sgn * cl[-1], etime[-1], "EOD_LAST")


def trades_long(days, keys, **p):
    return [r for d in keys if (r := run_gb(days[d], 1, **p)) is not None]


def perf(pnls):
    n = len(pnls)
    if n == 0: return dict(n=0, net=0.0, PF=float("nan"), win=float("nan"), dd=0.0)
    wd = pnls[pnls > 0].sum(); ld = -pnls[pnls < 0].sum()
    eq = np.cumsum(pnls)
    return dict(n=n, net=float(pnls.sum()), PF=(wd / ld if ld > 0 else float("inf")),
                win=(pnls > 0).mean(), dd=float((eq - np.maximum.accumulate(eq)).min()))


def test1_param_stability(days, keys):
    print("\n" + "=" * 74)
    print("TEST 1  Parameter Stability — neighborhood around k=1.5, gb=0.3, min_gap=0.3")
    print("=" * 74)
    cells = []
    for k in [1.0, 1.5, 2.0]:
        for gbv in [0.2, 0.3, 0.4]:
            for mg in [0.2, 0.3, 0.4]:
                p = dict(GB); p.update(k=k, gb=gbv, min_gap=mg)
                s = perf(np.array([t["net_dollars"] for t in trades_long(days, keys, **p)]))
                cells.append({"k": k, "gb": gbv, "min_gap": mg, "n": s["n"],
                              "win%": round(100 * s["win"], 1), "PF": round(s["PF"], 3),
                              "net": round(s["net"]), "maxdd": round(s["dd"]),
                              "tag": "*LOCK*" if (k == 1.5 and gbv == 0.3 and mg == 0.3) else ""})
    df = pd.DataFrame(cells)
    print(df.to_string(index=False))
    prof = (df.net > 0).sum()
    lock = df[df.tag == "*LOCK*"].iloc[0]; oth = df[df.tag == ""]
    print(f"\n  Profitable cells: {prof}/{len(df)}")
    print(f"  Locked PF {lock['PF']} net ${lock['net']:,.0f} | neighbors PF median {oth['PF'].median():.3f} "
          f"p25 {oth['PF'].quantile(.25):.3f} | net median ${oth['net'].median():,.0f} p25 ${oth['net'].quantile(.25):,.0f}")
    spike = lock["net"] > oth["net"].quantile(0.90)
    passed = prof >= len(df) - 1 and not spike
    print(f"  Locked is a SPIKE (net > p90 of neighbors)? {'YES -> overfit flag' if spike else 'no (on a plateau)'}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}  (>= {len(df)-1} profitable AND locked not a spike)")
    return passed


def test2_walk_forward(trades):
    print("\n" + "=" * 74); print("TEST 2  Walk-Forward (6-month non-overlapping windows)"); print("=" * 74)
    df = pd.DataFrame(trades); df["entry_time"] = pd.to_datetime(df.entry_time, utc=True)
    df["date"] = df.entry_time.dt.tz_convert(ET).dt.date
    cur, end = pd.Timestamp(df.date.min()), pd.Timestamp(df.date.max()); rows = []
    while cur < end:
        we = cur + pd.DateOffset(months=6)
        s = df[(df.date >= cur.date()) & (df.date < we.date())]
        if len(s):
            m = perf(s.net_dollars.to_numpy())
            rows.append({"start": cur.date(), "n": m["n"], "win%": round(100 * m["win"], 1),
                         "PF": round(m["PF"], 3), "net": round(m["net"]), "maxdd": round(m["dd"])})
        cur = we
    w = pd.DataFrame(rows); print(w.to_string(index=False))
    prof = (w.net > 0).sum(); passed = prof >= 0.75 * len(w) and w.net.min() > -10000
    print(f"\n  Profitable windows: {prof}/{len(w)}   worst ${w.net.min():,.0f}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}  (>=75% profitable + no window < -$10k)")
    return passed


def test3_mc_shuffle(pnls, n=10000, seed=42):
    print("\n" + "=" * 74); print(f"TEST 3  MC Order Shuffle ({n:,} perms)"); print("=" * 74)
    real = (np.cumsum(pnls) - np.maximum.accumulate(np.cumsum(pnls))).min()
    rng = np.random.default_rng(seed); dds = np.empty(n)
    for k in range(n):
        eq = np.cumsum(rng.permutation(pnls)); dds[k] = (eq - np.maximum.accumulate(eq)).min()
    pct = 100 * (dds < real).mean()
    print(f"  Real MaxDD ${real:,.0f} | shuffled p5 ${np.percentile(dds,5):,.0f} med ${np.percentile(dds,50):,.0f} p95 ${np.percentile(dds,95):,.0f}")
    print(f"  Real DD better than {pct:.1f}% of shuffles")
    passed = pct >= 50
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}  (real DD better than >=50%)")
    return passed


def test4_bootstrap(pnls, n=10000, seed=43):
    print("\n" + "=" * 74); print(f"TEST 4  Bootstrap CI ({n:,} resamples)"); print("=" * 74)
    rng = np.random.default_rng(seed); N = len(pnls); sums = np.empty(n); pfs = np.empty(n)
    for k in range(n):
        s = rng.choice(pnls, N, replace=True); sums[k] = s.sum()
        ld = -s[s < 0].sum(); pfs[k] = (s[s > 0].sum() / ld if ld > 0 else np.inf)
    plose = 100 * (sums < 0).mean(); sc = np.percentile(sums, [2.5, 50, 97.5]); pc = np.percentile(pfs, [2.5, 50, 97.5])
    print(f"  Net 95% CI [${sc[0]:,.0f}, ${sc[2]:,.0f}] med ${sc[1]:,.0f} | PF 95% CI [{pc[0]:.3f}, {pc[2]:.3f}] | P(losing) {plose:.2f}%")
    passed = plose < 1.0 and sc[0] > 0 and pc[0] >= 1.0
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}  (P(loss)<1%, net CI lo>0, PF CI lo>=1.0)")
    return passed


def test5_direction(days, keys, real_net, n=1000, seed=44):
    print("\n" + "=" * 74); print(f"TEST 5  Direction Permutation ({n:,} random long/short)"); print("=" * 74)
    longs = np.array([t["net_dollars"] for t in trades_long(days, keys, **GB)])
    shorts = np.array([(r["net_dollars"] if (r := run_gb(days[d], -1, **GB)) else 0.0) for d in keys
                       if run_gb(days[d], 1, **GB) is not None])
    rng = np.random.default_rng(seed); sums = np.empty(n)
    for k in range(n):
        coin = rng.integers(0, 2, len(longs)); sums[k] = np.where(coin == 1, longs, shorts).sum()
    pct = 100 * (sums < real_net).mean()
    print(f"  Real all-long ${real_net:,.0f} | all-short ${shorts.sum():,.0f} | perm med ${np.median(sums):,.0f}")
    print(f"  Real beats {pct:.1f}% of random direction assignments (p={1-pct/100:.4f})")
    passed = pct >= 99.0
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}  (real in top 1%, p<0.01)")
    return passed


def main():
    print("Loading 5-min bars...", flush=True)
    days = load_days(); keys = sorted(days.keys())
    tr = trades_long(days, keys, **GB); pnls = np.array([t["net_dollars"] for t in tr])
    s = perf(pnls)
    print(f"  {len(keys)} days | GIVEBACK k1.5/gb0.3: n={s['n']} win {100*s['win']:.1f}% net ${s['net']:,.0f} "
          f"PF {s['PF']:.3f} MaxDD ${s['dd']:,.0f}")
    R = {}
    R["1_param_stability"] = test1_param_stability(days, keys)
    R["2_walk_forward"] = test2_walk_forward(tr)
    R["3_mc_shuffle"] = test3_mc_shuffle(pnls)
    R["4_bootstrap"] = test4_bootstrap(pnls)
    R["5_direction_perm"] = test5_direction(days, keys, s["net"])
    print("\n" + "=" * 74); print("SUMMARY (giveback k=1.5/gb0.3)"); print("=" * 74)
    for k, v in R.items(): print(f"  {k:22s} {'PASS' if v else 'FAIL'}")
    print(f"\n  {sum(R.values())}/5 passed")


if __name__ == "__main__":
    main()
