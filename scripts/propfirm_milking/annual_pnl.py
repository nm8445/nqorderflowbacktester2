"""Annual P&L projection for the prop-milking cycle (RV + FB(N=3) + MEC, ~$900 reach risk).

Scales the eval batch to a target net/cycle, then simulates many YEARS as a renewal of
cycles (including losing cycles) to produce the distribution of annual profit.

Two cadence assumptions:
  - aggressive: redeploy when you recoup your spend (~6 weeks) -> overlapping, needs ~2x capital
  - sequential: redeploy when a batch fully resolves (~9 weeks) -> one batch at a time

Run:  python -m scripts.propfirm_milking.annual_pnl
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

import scripts.propfirm_milking.mc_portfolio as M
import scripts.propfirm_milking.fb_entry_sweep as FB
from scripts.propfirm_milking.common import eval_bracket
from scripts.propfirm_milking.entries import build_entry_packs, packs_from_fills

TD_YEAR = 252
ACCOUNT_COST = 100.0
REACH_SCALE = 0.9          # ~$900 reach-phase risk
N_CYCLE_SIMS = 2500
N_YEARS = 20000


def build_daypacks():
    rv = eval_bracket(build_entry_packs("RV", verbose=False), 10, 1.5, 2.0)
    mec = eval_bracket(build_entry_packs("MEC", verbose=False), 10, 2.0, 2.75)
    bars5 = FB.build_5min_delta_bars()
    fills, signs = FB.generate_fb_entries(bars5, 3, 300)              # FB N=3
    fb = eval_bracket(packs_from_fills("FB", fills, signs, verbose=False), 14, 2.5, 3.25)
    allt = pd.concat([rv[["date", "fill_time", "win", "pnl_$"]],
                      fb[["date", "fill_time", "win", "pnl_$"]],
                      mec[["date", "fill_time", "win", "pnl_$"]]]).sort_values(["date", "fill_time"])
    return [list(zip(g["win"].astype(bool), g["pnl_$"])) for _, g in allt.groupby("date", sort=True)]


def funded_timed(rng, chal, funded, reach_scale=REACH_SCALE):
    """Returns list of (internal_day, payout_amount) for one funded account."""
    bal = 50000.0; hwm = 50000.0; floor = 48000.0; wd = 0; ev = []; used = 0; reached = False
    nday = len(chal); nfund = len(funded)
    for d in range(504):
        used += 1
        dp = M._play_day_single(chal[rng.integers(0, nday)], rng) * reach_scale
        bal += dp
        if bal < floor:
            return ev
        if dp >= 150: wd += 1
        if bal > hwm: hwm = bal; floor = max(48000.0, hwm - 2000.0)
        if bal - 50000.0 >= 3000.0:
            reached = True; break
    if not reached:
        return ev
    for d in range(used, 504):
        dp = funded[rng.integers(0, nfund)]; bal += dp
        if bal < floor: break
        if dp >= 150: wd += 1
        if bal > hwm: hwm = bal; floor = max(48000.0, hwm - 2000.0)
        if wd >= 5:
            prof = bal - 50000.0
            if prof > 0: ev.append((d, 0.5 * prof)); bal -= 0.5 * prof
            wd = 0
    return ev


def cycle_dist(chal, funded, n_accts, pair, nsims=N_CYCLE_SIMS):
    """Per-cycle joint draws of (net, recoup_day, span_day)."""
    M._CHAL_DAYS = chal; M._FUNDED_DAILY = funded
    M.N_ACCTS = n_accts; M.PAIR = pair
    cost = n_accts * ACCOUNT_COST
    rng = np.random.default_rng(123)
    nets = np.empty(nsims); recoup = np.empty(nsims); span = np.empty(nsims)
    for s in range(nsims):
        npass, pass_days = M.run_challenge(rng)
        ev = []
        for pday in pass_days:
            for (idy, amt) in funded_timed(rng, chal, funded):
                ev.append((int(pday) + int(idy), amt))
        gross = sum(a for _, a in ev)
        nets[s] = gross - cost
        if ev:
            ev.sort()
            cum = 0.0; rc = None
            for g, a in ev:
                cum += a
                if rc is None and cum >= cost:
                    rc = g
            recoup[s] = rc if rc is not None else 504
            span[s] = ev[-1][0]
        else:
            recoup[s] = 504; span[s] = 60
    return nets, recoup, span


def annual(nets, gaps, n_years=N_YEARS):
    """Renewal: deploy batches separated by `gaps` (trading days) until the year is full."""
    rng = np.random.default_rng(7)
    n = len(nets)
    out = np.empty(n_years)
    n_cycles = np.empty(n_years)
    for y in range(n_years):
        t = 0.0; tot = 0.0; k = 0
        while t < TD_YEAR:
            i = rng.integers(0, n)
            tot += nets[i]
            t += gaps[i]
            k += 1
        out[y] = tot; n_cycles[y] = k
    return out, n_cycles


def report(label, annual_profit, n_cycles, spend):
    p = np.percentile(annual_profit, [10, 25, 50, 75, 90])
    print(f"\n  [{label}]  (~{n_cycles.mean():.1f} cycles/yr, ${spend:,.0f}/batch)")
    print(f"    median annual profit : ${p[2]:,.0f}")
    print(f"    p10 / p90            : ${p[0]:,.0f} / ${p[4]:,.0f}")
    print(f"    p25 / p75            : ${p[1]:,.0f} / ${p[3]:,.0f}")
    print(f"    mean                 : ${annual_profit.mean():,.0f}")
    print(f"    P(losing year)       : {(annual_profit < 0).mean():.1%}")


def main():
    print("Building day-packs (RV + FB[N=3] + MEC)...")
    chal = build_daypacks()
    funded = M.build_funded_daily()

    for n_accts, pair, tag in [(30, 2, "30 evals (baseline)"), (85, 6, "85 evals (~$10k/cycle target)")]:
        spend = n_accts * ACCOUNT_COST
        nets, recoup, span = cycle_dist(chal, funded, n_accts, pair)
        print(f"\n=== {tag}: spend ${spend:,.0f} ===")
        print(f"  per-cycle net: mean ${nets.mean():,.0f}  median ${np.median(nets):,.0f}  "
              f"P(net>0) {(nets>0).mean():.0%}")
        print(f"  recoup (6wk?): median {np.median(recoup):.0f} td  | full span median {np.median(span):.0f} td")
        ap_a, nc_a = annual(nets, recoup)
        ap_s, nc_s = annual(nets, span)
        report("aggressive: redeploy at recoup (~overlap, ~2x capital)", ap_a, nc_a, spend)
        report("sequential: one batch at a time", ap_s, nc_s, spend)


if __name__ == "__main__":
    main()
