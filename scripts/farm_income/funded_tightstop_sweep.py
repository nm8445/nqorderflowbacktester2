"""Does TIGHTENING the 1:1 bracket (smaller stop) cut force-closes and lift E[$/funded]? Sweep a SCALE
on each strat's native stop (OD 1.30xATR, B2 2.50xATR, RV 2.0xATR, FB entry->ORB-low), keep 1:1, include
$4/MNQ RT cost (commission+slippage) -- tighter stops cost more (more contracts). Measure FC%, WR, net
meanR, and E[$/funded]. scale 1.0 = current farm config.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scripts.propfirm_milking.common import eval_bracket, first_touch, load_1min_bars
from scripts.propfirm_milking.entries import build_entry_packs
from scripts.propfirm_milking.farm_config_wr import orb_low_for_date, LIVE_STOP
from funded_rr_sweep import build_all_packs
from farm_income_mc import load_daily_1mnq
from funded_target_risk_sweep import funded_life

SCALE_GRID = [0.5, 0.75, 1.0, 1.25, 1.5]
COST_PER_MNQ = 4.0      # $2 commission + 2 ticks/side slippage (common.py)
RISK = 1500.0


def combined_net(packs, scale):
    """Per-trade NET R (after $4/MNQ cost) at 1:1 with stops scaled by `scale`. Also returns FC frac."""
    Rs, fc = [], []
    for s in ["OD", "B2", "RV"]:
        _, alen, mult = LIVE_STOP[s]
        m = scale * mult
        tr = eval_bracket(packs[s], atr_len=alen, sl_mult=m, tp_mult=m)
        sl_pts = tr["sl_dist_pts"].to_numpy()
        gross = (tr["pnl_pts"] / tr["sl_dist_pts"]).to_numpy()
        # cost_$ = mnq*COST_PER_MNQ where mnq = 1500/(sl_pts*2);  cost_R = cost_$/1500
        cost_R = (1500.0 / (sl_pts * 2.0) * COST_PER_MNQ) / 1500.0
        Rs.append(gross - cost_R)
        fc.append((tr["reason"] == "FC").to_numpy())
    # FB (ORB-low scaled)
    fbR, fbfc = [], []
    for p, sl0 in zip(packs["FB"], packs["_fb_sl"]):
        if not np.isfinite(sl0) or sl0 <= 0:
            continue
        sl = scale * sl0
        ep, reason, _ = first_touch(p, sl, sl)
        g = p.sign * (ep - p.entry_price) / sl
        cost_R = (1500.0 / (sl * 2.0) * COST_PER_MNQ) / 1500.0
        fbR.append(g - cost_R); fbfc.append(reason == "FC")
    Rs.append(np.array(fbR)); fc.append(np.array(fbfc))
    return np.concatenate(Rs), np.concatenate(fc)


def main():
    df1m = load_1min_bars()
    packs = build_all_packs(df1m)
    daily = load_daily_1mnq()
    print("Tighten-stop sweep (1:1, $4/MNQ cost). scale 1.0 = current farm config.\n")
    print(f"  {'scale':>5} {'FC%':>5} {'WR%':>5} {'netMeanR':>9} {'cost$/tr':>9} {'E[$/funded]':>12}")
    base = None
    for sc in SCALE_GRID:
        pool, fc = combined_net(packs, sc)
        rng = np.random.default_rng(5)
        vals = np.array([funded_life(pool * RISK, 3000.0, daily, rng) for _ in range(40000)])
        ef = vals.mean()
        if sc == 1.0:
            base = ef
        # mean cost: back out from net vs gross would need gross; approximate via 2/sl proxy not stored -> skip exact $
        print(f"  {sc:>5.2f} {fc.mean()*100:>4.0f}% {(pool>0).mean()*100:>4.0f}% {pool.mean():>9.3f} "
              f"{'':>9} {ef:>12.0f}")
    print(f"\n  (baseline scale 1.0 E[$/funded] above; compare the others to it)")


if __name__ == "__main__":
    main()
