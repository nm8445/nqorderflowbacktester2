"""Optimal funded-gamble RR: sweep TP:SL (risk fixed at $1500) and find which maximizes P(reach +$3k)
before the trailing $2k floor. WR at each RR comes from the REAL combined trades (live stops, re-bracketed
TP = RR x SL, force-closes included) -- not assumed. OD/B2/RV use ATR stops; FB uses entry->ORB-low.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scripts.propfirm_milking.common import eval_bracket, first_touch, load_1min_bars
from scripts.propfirm_milking.entries import build_entry_packs
from scripts.propfirm_milking.farm_config_wr import orb_low_for_date, LIVE_STOP

RR_GRID = [0.75, 1.0, 1.2, 1.33, 1.5, 1.75, 2.0, 2.5]
RISK = 1500.0


def build_all_packs(df1m):
    packs = {}
    for s in ["OD", "B2", "RV"]:
        _, alen, _ = LIVE_STOP[s]
        packs[s] = build_entry_packs(s, atr_lens=[alen], verbose=False)
    packs["FB"] = build_entry_packs("FB", atr_lens=[10], verbose=False)
    # precompute FB ORB-low sl distances once
    fb_sl = []
    for p in packs["FB"]:
        orb = orb_low_for_date(df1m, p.date)
        fb_sl.append(p.sign * (p.entry_price - orb) if np.isfinite(orb) else np.nan)
    packs["_fb_sl"] = np.array(fb_sl)
    return packs


def combined_R(packs, rr):
    """Per-trade R (pnl/risk) for the combined book at TP = rr x SL. SL=-1, TP=+rr, FC=partial."""
    Rs = []
    for s in ["OD", "B2", "RV"]:
        _, alen, mult = LIVE_STOP[s]
        tr = eval_bracket(packs[s], atr_len=alen, sl_mult=mult, tp_mult=rr * mult)
        Rs.append((tr["pnl_pts"] / tr["sl_dist_pts"]).to_numpy())
    # FB: ORB-low stop, TP = rr x stop
    fb = []
    for p, sl in zip(packs["FB"], packs["_fb_sl"]):
        if not np.isfinite(sl) or sl <= 0:
            continue
        exit_price, reason, _ = first_touch(p, sl, rr * sl)
        fb.append(p.sign * (exit_price - p.entry_price) / sl)
    Rs.append(np.array(fb))
    return np.concatenate(Rs)


def reach_3k(pool, rr, rng, n=60000, max_tr=300):
    """P(reach +3000 before the trailing $2k floor) gambling risk=$1500, win=+rr*1500. Returns
    (p_reach, median trades to reach)."""
    pool_d = pool * RISK
    reached = 0; trades_list = []
    N = pool_d.size
    for _ in range(n):
        profit = 0.0; peak = 0.0; t = 0
        while t < max_tr:
            profit += pool_d[rng.integers(N)]; t += 1
            peak = max(peak, profit)
            floor = min(0.0, peak - 2000.0)        # trailing, locks at 0 once peak>=2000
            if profit <= floor + 1e-9:
                break
            if profit >= 3000.0:
                reached += 1; trades_list.append(t); break
    return reached / n, (np.median(trades_list) if trades_list else float("nan"))


def main():
    df1m = load_1min_bars()
    packs = build_all_packs(df1m)
    print(f"funded-gamble RR sweep (risk ${RISK:.0f}, reach +$3k before trailing $2k floor):\n")
    print(f"  {'RR':>5} {'win$':>6} {'WR':>6} {'meanR':>6} {'P(reach 3k)':>12} {'med trades':>11}")
    best = (None, -1)
    for rr in RR_GRID:
        pool = combined_R(packs, rr)
        rng = np.random.default_rng(7)
        p, mt = reach_3k(pool, rr, rng)
        wr = (pool > 0).mean() * 100
        print(f"  {rr:>5.2f} {rr*RISK:>6.0f} {wr:>5.1f}% {pool.mean():>6.2f} {p*100:>11.1f}% {mt:>11.0f}")
        if p > best[1]:
            best = (rr, p)
    print(f"\n  OPTIMAL RR = {best[0]} (P reach 3k = {best[1]*100:.1f}%)  [current farm uses 1.0]")


if __name__ == "__main__":
    main()
