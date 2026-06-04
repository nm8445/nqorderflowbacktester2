"""Drive the EVAL brain (eval_passer.py) through a POOL of sim accounts on REAL signal outcomes,
printing the state of every account each day. This is the "watch the rotation play out" harness.

Uses the same per-day first-touch outcomes as the rotation Monte-Carlo (eval_rotation_sim.py).
Each signal has ONE outcome that is COPIED to every account taking it (capped per account at the
remaining daily cap), exactly as route_signal()/on_position_closed() will be driven live.

Tweak N_ACCOUNTS / COPIES / DAY_CAP below. Run: python live/farm/sim_eval_passer.py
"""
from __future__ import annotations
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "futurespropmc"))
from eval_rotation_sim import per_day_outcomes                     # noqa: E402  (real first-touch $)
from eval_passer import EvalFarm, EState, START, DAY_CAP_50         # noqa: E402

N_ACCOUNTS = 20
COPIES = 1 if N_ACCOUNTS <= 20 else 2
DAY_CAP = DAY_CAP_50
RISK = DAY_CAP                 # 1:1 bracket sized to the cap
MAX_DAYS = 40
SEED = 3


def apply_signal(farm, strat, out, d):
    """Route one signal and copy its single outcome onto every taker (sync + close per account).
    Wins are capped at the remaining daily cap; losses at the remaining buffer (the prop floor stops
    you out there — you blow AT -$2k, you can't overshoot it)."""
    for r in farm.route_signal(strat, d):
        a = farm.accounts[r.account_id]
        cap = min(out, farm.day_cap - a.day_profit) if out > 0 else -min(-out, a.buffer)
        new_cash = a.cash + cap
        farm.sync_accounts({r.account_id: {"cash": new_cash, "equity": new_cash}})
        farm.on_position_closed(r.account_id, strat, cap)


def main():
    print(f"Loading real first-touch outcomes (1:1 ${RISK:,.0f} bracket)...")
    days = per_day_outcomes(RISK, DAY_CAP)
    rng = np.random.default_rng(SEED)

    farm = EvalFarm(copies=COPIES, day_cap=DAY_CAP, quiet=True)     # quiet: we print our own table
    farm.sync_accounts({f"E{i:02d}": {"cash": START, "equity": START} for i in range(1, N_ACCOUNTS + 1)})
    print(f"\n{N_ACCOUNTS} eval accounts, copies/signal={COPIES}, daily cap ${DAY_CAP:,.0f}, "
          f"target +$3,000 ({len(days)} historical days to sample)\n")

    d = date(2026, 6, 4)
    for day in range(MAX_DAYS):
        live = [a for a in farm.accounts.values() if a.state in (EState.FRESH, EState.ACTIVE, EState.DONE)]
        if not live:
            break
        d += timedelta(days=1)
        for out in days[rng.integers(0, len(days))]:               # a sampled day's signals
            apply_signal(farm, "SIG", out, d)
        c = farm.counts()
        print(f"=== day {day + 1} ({d}) ===  "
              f"ACTIVE {c.get('ACTIVE',0)} | DONE {c.get('DONE',0)} | FRESH {c.get('FRESH',0)} | "
              f"PASSED {c.get('PASSED',0)} | BLOWN {c.get('BLOWN',0)}")
        print(farm.state_table())
        farm.end_of_day(d)

    c = farm.counts()
    np_, nb = c.get("PASSED", 0), c.get("BLOWN", 0)
    print(f"\nFINAL after {day + 1} days: PASSED {np_}/{N_ACCOUNTS} ({100*np_/N_ACCOUNTS:.0f}%), "
          f"BLOWN {nb}, still trading {N_ACCOUNTS - np_ - nb}")
    print(f"passed -> {farm.passed_ids}")


if __name__ == "__main__":
    main()
