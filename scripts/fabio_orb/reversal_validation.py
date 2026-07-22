"""Validate the Fabio bullish-reversal LONG variant (N=4 green closes + delta>=400, SL=ORB_range, TP=4R)
against the overfit framework (run_overfit_tests) + a combined-with-UP test.

  Test 1  Parameter stability  (reversal neighborhood: N x delta x TP)
  Test 2-5  Walk-forward / MC shuffle / Bootstrap / Direction-permutation  (imported, generic)
  + Combined: UP-only vs REV-only vs BOTH(first-of-either)  — cannibalization & risk-adjusted.

Run: python scripts/fabio_orb/reversal_validation.py
"""
from __future__ import annotations
import warnings; warnings.simplefilter("ignore")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np, pandas as pd
from run_overfit_tests import (perf, test_2_walk_forward, test_3_mc_shuffle, test_4_bootstrap,
                               test_5_direction_permutation, VOL_PARQUET, ORB_START_HHMM,
                               ORB_END_HHMM, TRADE_END_HHMM, SKIP_BUCKET, DPP, SLIP_PTS, COMM)

REV_N, REV_DTHR, REV_TP = 4, 400.0, 4.0
COST = 2 * SLIP_PTS * DPP + COMM


def load_days():
    df = pd.read_parquet(VOL_PARQUET)
    agg = df.groupby("bar_open_time", as_index=False).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"),
    ).dropna(subset=["open", "high", "low", "close"])
    agg["close_et"] = agg["bar_open_time"] + pd.Timedelta(minutes=5)
    agg["hhmm"] = agg["close_et"].dt.hour * 100 + agg["close_et"].dt.minute
    agg = agg[(agg["hhmm"] > ORB_START_HHMM) & (agg["hhmm"] <= TRADE_END_HHMM)].copy()
    agg["delta"] = agg["buy_vol"] - agg["sell_vol"]
    agg["session_date"] = agg["close_et"].dt.normalize()
    days = {}
    for sd, sub in agg.groupby("session_date"):
        sub = sub.sort_values("close_et").reset_index(drop=True)
        in_orb = sub[(sub["hhmm"] > ORB_START_HHMM) & (sub["hhmm"] <= ORB_END_HHMM)]
        post = sub[(sub["hhmm"] > ORB_END_HHMM) & (sub["hhmm"] <= TRADE_END_HHMM)]
        if in_orb.empty or post.empty: continue
        days[pd.Timestamp(sd).tz_localize(None)] = {
            "orb_high": float(in_orb["high"].max()), "orb_low": float(in_orb["low"].min()),
            "hhmm": post["hhmm"].to_numpy(), "open": post["open"].to_numpy(np.float64),
            "high": post["high"].to_numpy(np.float64), "low": post["low"].to_numpy(np.float64),
            "close": post["close"].to_numpy(np.float64), "delta": post["delta"].to_numpy(np.float64),
            "close_et": post["close_et"].to_numpy(),
        }
    return days


def find_rev(day, n_bull, dthr):
    ol = day["orb_low"]; rng = day["orb_high"] - ol
    hhmm = day["hhmm"]; opn = day["open"]; close = day["close"]; low = day["low"]; delta = day["delta"]
    for i in range(n_bull - 1, len(hhmm)):
        if hhmm[i] > TRADE_END_HHMM: break
        if hhmm[i] == SKIP_BUCKET: continue
        if np.min(low[:i + 1]) >= ol: continue
        if not all(close[i - k] > opn[i - k] for k in range(n_bull)): continue
        if delta[i] < dthr: continue
        return (i, float(close[i]), float(close[i] - rng))     # idx, entry, sl
    return None


def find_up(day):
    oh = day["orb_high"]; ol = day["orb_low"]; hhmm = day["hhmm"]; close = day["close"]; delta = day["delta"]
    for i in range(3, len(hhmm)):
        if hhmm[i] > TRADE_END_HHMM: break
        if hhmm[i] == SKIP_BUCKET: continue
        if not all(close[i - k] > oh for k in range(4)): continue
        if delta[i] < 300.0: continue
        ep = close[i]
        if ol >= ep: continue
        return (i, float(ep), float(ol))
    return None


def sim_net(day, i, ep, sl, direction, tp_mult):
    hhmm = day["hhmm"]; high = day["high"]; low = day["low"]; close = day["close"]; n = len(hhmm)
    risk = ep - sl
    if direction == 1:
        s, t = sl, ep + tp_mult * risk
        xp = close[-1]
        for j in range(i + 1, n):
            if hhmm[j] >= TRADE_END_HHMM: xp = close[j]; break
            if low[j] <= s: xp = s; break
            if high[j] >= t: xp = t; break
    else:
        s, t = ep + risk, ep - tp_mult * risk
        xp = close[-1]
        for j in range(i + 1, n):
            if hhmm[j] >= TRADE_END_HHMM: xp = close[j]; break
            if high[j] >= s: xp = s; break
            if low[j] <= t: xp = t; break
    return (xp - ep) * direction * DPP - COST


def build_rev(days, n_bull=REV_N, dthr=REV_DTHR, tp_mult=REV_TP):
    trades, perm = [], []
    for d in sorted(days):
        pick = find_rev(days[d], n_bull, dthr)
        if pick is None: continue
        i, ep, sl = pick
        ln = sim_net(days[d], i, ep, sl, 1, tp_mult); sn = sim_net(days[d], i, ep, sl, -1, tp_mult)
        trades.append({"entry_time": pd.Timestamp(days[d]["close_et"][i]), "net_dollars": ln,
                       "idx": i, "which": "rev"})
        perm.append((ln, sn))
    return trades, perm


def test_1_rev_stability(days):
    print("\n" + "=" * 78)
    print("TEST 1: Reversal parameter stability  (N x delta x TP around N=4, d=400, TP=4)")
    print("=" * 78)
    rows = []
    for N in (3, 4, 5):
        for D in (350, 400, 450):
            for TP in (3.5, 4.0, 4.5):
                tr, _ = build_rev(days, N, float(D), TP)
                s = perf(np.array([t["net_dollars"] for t in tr]))
                rows.append({"N": N, "delta": D, "TP": TP, "n": s["n"],
                             "WR%": round(100 * s["win_rate"], 1), "PF": round(s["PF"], 3),
                             "net": round(s["net"]), "maxdd": round(s["maxdd"])})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    npr = (df["net"] > 0).sum()
    print(f"\n  Profitable cells: {npr}/{len(df)}   (locked N=4/d400/TP4 in the middle)")
    return npr >= len(df) - 1


def combined(days):
    print("\n" + "=" * 78)
    print("COMBINED: UP-only vs REV-only vs BOTH (one trade/day, first of either)")
    print("=" * 78)
    def book(which):
        nets, times, comp = [], [], {"up": 0, "rev": 0}
        for d in sorted(days):
            up = find_up(days[d]); rv = find_rev(days[d], REV_N, REV_DTHR)
            if which == "up": pick, tag = (up, "up")
            elif which == "rev": pick, tag = (rv, "rev")
            else:
                if up and rv: pick, tag = ((up, "up") if up[0] <= rv[0] else (rv, "rev"))
                elif up: pick, tag = up, "up"
                else: pick, tag = rv, "rev"
            if pick is None: continue
            i, ep, sl = pick; comp[tag] += 1
            nets.append(sim_net(days[d], i, ep, sl, 1, REV_TP)); times.append(days[d]["close_et"][i])
        return np.array(nets), comp
    print(f"  {'book':>5} {'tr':>5} {'WR%':>5} {'net$':>10} {'PF':>6} {'MaxDD$':>10} {'ret/DD':>7} {'comp'}")
    for which in ("up", "rev", "both"):
        nets, comp = book(which)
        s = perf(nets); rdd = s["net"] / abs(s["maxdd"]) if s["maxdd"] else float("inf")
        c = "" if which != "both" else f"up={comp['up']} rev={comp['rev']}"
        print(f"  {which:>5} {s['n']:>5} {100*s['win_rate']:>4.0f}% {s['net']:>10,.0f} "
              f"{s['PF']:>6.2f} {s['maxdd']:>10,.0f} {rdd:>7.1f}  {c}")


def main():
    print("Loading bars...", flush=True)
    days = load_days()
    print(f"  {len(days)} days")
    tr, perm = build_rev(days)
    pnls = np.array([t["net_dollars"] for t in tr])
    s = perf(pnls)
    print(f"\nREVERSAL locked (N=4 green, delta>=400, TP=4R): {s['n']} tr  "
          f"WR {100*s['win_rate']:.1f}%  net ${s['net']:,.0f}  PF {s['PF']:.3f}  MaxDD ${s['maxdd']:,.0f}")
    res = {}
    res["1_param_stability"] = test_1_rev_stability(days)
    res["2_walk_forward"] = test_2_walk_forward(tr)
    res["3_mc_shuffle"] = test_3_mc_shuffle(tr)
    res["4_bootstrap"] = test_4_bootstrap(tr)
    res["5_direction_perm"] = test_5_direction_permutation(perm, s["net"])
    combined(days)
    print("\n" + "=" * 78 + "\nSUMMARY")
    for k, v in res.items():
        print(f"  {k:24} {'PASS' if v else 'FAIL'}")
    print(f"  {sum(res.values())}/5 overfit tests passed")


if __name__ == "__main__":
    main()
