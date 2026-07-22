"""Structure A vs B for the funded phase, with adaptive milk sizing (size to cushion). 30 accounts/cycle.
A = re-gamble to +$3k each payout (cushion reset -> milk ~3 MNQ every round, big payouts, but re-gamble
    can blow it). B = milk-down (no re-gamble; cushion shrinks -> size steps 3->2->1 MNQ, smaller payouts,
    safer). Milk samples the real 1-MNQ 4-way daily (net AND intraday low) x size. Gamble = 1:1 $1500.
FULLY MAE-AWARE: both the buffer-build gamble (build_outcomes pool) AND the milk days (intraday low)
blow on a floating dip through the floor. Payout cap varies by firm -> swept (2 and 3) below.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scripts.futurespropmc.challenge_gamble_pass import build_outcomes
from farm_income_mc import load_daily_1mnq_mae

SPLIT = 0.9; WIN = 150.0
EVAL_PASS = 0.34; EVAL_FEE = 100.0; ACCOUNTS = 30; EVAL_DAYS = 5  # MAE-aware (real firm floating rule; old 0.43 was realized-only/optimistic)


def milk_size(profit):
    return 3 if profit >= 2500 else (2 if profit >= 1200 else 1)


def gamble(pool, target, rng, locked, start=0.0):
    """MAE-AWARE buffer-build (real firm floating rule). pool = array of (realized_g_$, worst_floating_mae_$)
    at $1500 1:1 -- the SAME pool as the eval, so reach-target here == the eval's MAE-aware ~34% (this used
    to be realized-only, which inflated the payout rate to a bogus 43%). A trade whose floating dip
    (profit + mae) breaches the floor blows the account mid-trade, even if it would have settled positive."""
    p = start; peak = max(start, 0.0); wins = 0; tr = 0; n = len(pool)
    for _ in range(300):
        tr += 1
        g, mae = pool[rng.integers(n)]
        floor = 0.0 if locked else min(0.0, peak - 2000.0)
        if p + mae <= floor + 1e-9: return None, wins, tr    # floating dip -> blown mid-trade
        p += g
        if g > 0: wins += 1
        peak = max(peak, p)
        floor = 0.0 if locked else min(0.0, peak - 2000.0)
        if p <= floor + 1e-9: return None, wins, tr
        if p >= target: return p, wins, tr
    return None, wins, tr


def milk(p, need_wins, d1, rng):
    """Milk to need_wins winning days; size adapts to cushion each day. MAE-AWARE: an intraday dip
    (profit + size*day_low) through the locked $0 floor blows the day even if it closes green.
    d1 = array of (net_$, intraday_low_$) at 1 MNQ. Returns (profit|None, days)."""
    days = 0; w = 0; n = len(d1)
    while w < need_wins:
        days += 1
        net, low = d1[rng.integers(n)]; s = milk_size(p)
        if p + s * low <= 1e-9: return None, days      # intraday floating dip -> blown (floor 0)
        pnl = s * net; p += pnl
        if p <= 1e-9: return None, days                # (redundant w/ dip check, but keeps net-loss blows)
        if pnl >= WIN: w += 1
    return p, days


def funded(structure, pool, d1, rng, cap):
    p, gw, tr = gamble(pool, 3000.0, rng, locked=False)
    days = tr
    if p is None: return 0.0, days
    cash = 0.0; nwd = 0
    while nwd < cap:
        if nwd > 0:
            if structure == "A":
                p2, gw, tr2 = gamble(pool, 3000.0, rng, locked=True, start=p); days += tr2
                if p2 is None: return cash, days
                p = p2
            else:
                gw = 0
        need = max(1, 5 - gw)
        p2, md = milk(p, need, d1, rng); days += md
        if p2 is None: return cash, days
        p = p2
        w = p / 2.0; cash += w * SPLIT; p -= w; nwd += 1
    return cash, days


def main():
    df = build_outcomes(1500., 1500.)               # MAE-aware 4-way $1500 1:1 pool (== eval 34%)
    pool = df[["g", "mae"]].values
    d1 = load_daily_1mnq_mae()                        # (net_$, intraday_low_$) per firm-day at 1 MNQ
    print(f"30 accounts/cycle, eval pass {EVAL_PASS:.0%}, FULLY MAE-aware (gamble + milk intraday lows), "
          f"adaptive milk (3/2/1 MNQ by cushion)\n")
    print(f"  {'cap':>3} {'struct':>7} {'E[$/funded]':>11} {'P(>=1 pay)':>10} {'fund days':>9} {'cyc days':>8} "
          f"{'cyc/yr':>6} {'net/cyc':>9} {'ANNUAL':>10}")
    for cap in (2, 3):                                # payout cap varies by firm
        for s in ("A", "B"):
            rng = np.random.default_rng(5)
            res = [funded(s, pool, d1, rng, cap) for _ in range(80000)]
            cash = np.array([r[0] for r in res]); days = np.array([r[1] for r in res])
            ef = cash.mean(); p_any = (cash > 0).mean(); fdays = np.median(days[cash > 0])
            cyc_days = EVAL_DAYS + np.percentile(days, 80)
            cyc_yr = 251 / cyc_days
            funded_per = ACCOUNTS * EVAL_PASS
            net_cyc = funded_per * ef - ACCOUNTS * EVAL_FEE
            annual = net_cyc * cyc_yr
            print(f"  {cap:>3} {s:>7} {ef:>11,.0f} {p_any*100:>9.0f}% {fdays:>9.0f} {cyc_days:>8.0f} {cyc_yr:>6.1f} "
                  f"{net_cyc:>9,.0f} {annual:>10,.0f}")
    print("\n  (funded days = trades-as-days for gamble + real milk days; throughput-adjusted would be longer)")


if __name__ == "__main__":
    main()
