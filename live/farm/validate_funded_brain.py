"""Regression: drive the FundedFarm BRAIN (funded_state_machine.py) through real gamble + milk
outcomes and confirm it reproduces account_lifecycle.py's economics (the playbook numbers).

Same data as the playbook: gamble_pool() (risk-normalised per-strat outcomes) + milk_packs()
(per-day 1-MNQ P&L with intraday float). For each simulated funded account we exercise the brain's
ACTUAL methods — sync_accounts / route_signal / on_position_closed / end_of_day / record_withdrawal —
exactly as the live executor will, and read off reach-buffer %, payouts, take-home and first-payout day.

The one deliberate model difference vs account_lifecycle: the brain re-gambles sized to the *remaining
buffer* (the user's rule), whereas account_lifecycle re-gambles a flat $2k. They agree on the first
gamble (buffer = $2k) and on the blow condition (a stop-out, mae >= 1R); they diverge only on the
small-loss-then-recover paths, so the aggregates should land in the same ballpark.

Run: python live/farm/validate_funded_brain.py
"""
from __future__ import annotations
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "futurespropmc"))
from farm_firing_order import gamble_pool, milk_packs            # noqa: E402  (same data as the playbook)
from funded_state_machine import FundedFarm, Phase, START, SIGNALS, SPLIT  # noqa: E402

GCOST, MILK_COST = 30.0, 2.0


def simulate_account(G, M, rng, max_gamble=60, max_milk=504):
    """One funded account through the brain. Returns the same outcome dict shape as lifecycle()."""
    farm = FundedFarm(quiet=True)
    aid, d = "A", date(2026, 1, 1)
    farm.sync_accounts({aid: {"cash": START, "equity": START}})
    a = farm.accounts[aid]
    ng, nm = len(G), len(M)
    gd = 0

    # ---- gamble to buffer ----
    while a.phase is Phase.GAMBLING and gd < max_gamble:
        gd += 1; d += timedelta(days=1)
        buf = a.buffer
        strat = next((s for s in SIGNALS if s != a.last_lost_signal), SIGNALS[0])
        farm.route_signal(strat, d)            # marks last_gamble_date, sizes to buffer
        pr, ma = G[rng.integers(0, ng)]
        farm.sync_accounts({aid: {"cash": a.cash, "equity": a.cash - ma * buf}})  # intraday float -> blow check
        if a.phase is Phase.RETIRED:
            return dict(out="blow_gamble", take=farm.total_take_home)
        close = a.cash + pr * buf - GCOST
        farm.sync_accounts({aid: {"cash": close, "equity": close}})
        farm.on_position_closed(aid, strat, pr * buf - GCOST)
    if a.phase is not Phase.MILKING:
        return dict(out="no_buffer", take=farm.total_take_home)
    buffer = a.cash - START

    # ---- milk 1 MNQ ----
    md, first, prev_pays = 0, None, 0
    while a.phase is Phase.MILKING and md < max_milk:
        md += 1; d += timedelta(days=1)
        bal, min_eq = a.cash, a.cash
        for pn, fl in M[rng.integers(0, nm)]:
            min_eq = min(min_eq, bal - fl)
            bal += pn - MILK_COST
        farm.sync_accounts({aid: {"cash": a.cash, "equity": min_eq}})   # intraday float -> blow check
        if a.phase is Phase.RETIRED:
            return dict(out="blow_milk", buffer=buffer, pays=a.payouts_done,
                        take=farm.total_take_home, first=first)
        farm.sync_accounts({aid: {"cash": bal, "equity": bal}})
        farm.end_of_day(d)
        if a.payouts_done > prev_pays:                                  # a payout fired this EOD
            w = farm.pending_payouts[-1][1]
            farm.record_withdrawal(aid, w)                             # cash -= w, floor held
            if first is None:
                first = (gd + md, SPLIT * w)
            prev_pays = a.payouts_done
    out = "capped" if a.payouts_done >= 4 else "survive"
    return dict(out=out, buffer=buffer, pays=a.payouts_done, take=farm.total_take_home, first=first)


def main():
    G, M = gamble_pool(), milk_packs()
    rng = np.random.default_rng(7)
    N = 60000
    res = [simulate_account(G, M, rng) for _ in range(N)]

    der = [r for r in res if r["out"] not in ("blow_gamble", "no_buffer")]
    paid = [r for r in der if r["pays"] >= 1]
    firsts = [r["first"] for r in paid if r["first"]]
    fd = np.array([f[0] for f in firsts]); fa = np.array([f[1] for f in firsts])
    take_der = np.array([r["take"] for r in der])
    take_all = np.array([r["take"] for r in res])
    pays = np.array([r["pays"] for r in der])

    print("FUNDED BRAIN regression: driving funded_state_machine.py through real outcomes\n")
    print(f"  {'metric':<42}{'BRAIN':>12}{'playbook ref':>16}")
    print(f"  {'-'*70}")
    print(f"  {'reach buffer & de-risk':<42}{len(der)/len(res)*100:>11.0f}%{'~61%':>16}")
    print(f"  {'of de-risked, >=1 payout':<42}{len(paid)/max(1,len(der))*100:>11.0f}%{'~96%':>16}")
    print(f"  {'first payout: median day (from funding)':<42}{int(np.median(fd)):>12}{'~17':>16}")
    print(f"  {'first payout: median take-home':<42}{'$'+format(np.median(fa),',.0f'):>12}{'~$953':>16}")
    print(f"  {'payouts / de-risked acct (mean)':<42}{pays.mean():>12.2f}{'~2.94':>16}")
    print(f"  {'take-home / de-risked acct (mean)':<42}{'$'+format(take_der.mean(),',.0f'):>12}{'~$2,601':>16}")
    print(f"  {'take-home / funded acct (mean, incl blows)':<42}{'$'+format(take_all.mean(),',.0f'):>12}{'~$1,590':>16}")


if __name__ == "__main__":
    main()
