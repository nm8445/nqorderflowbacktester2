"""Eval-passer farm — the BACKEND BRAIN for CHALLENGE accounts (lead + copy rotation).

Sibling to funded_state_machine.py. Pure Python, no NT8/MT5 deps, sim-testable. The funded brain
spends its de-correlation budget on the gamble; the eval brain does the opposite — it COPIES each
signal across accounts for throughput, because evals are cheap and the only cost of correlation is
lumpier (not lower) pass counts (see scripts/futurespropmc/eval_rotation_sim.py).

STATES (per account)
  FRESH   flat, never started   -> waits to be promoted as a lead
  ACTIVE  trading today, has buffer -> copies every signal until it resolves
  DONE    hit the daily cap (+$1,500 / +$1,200) today -> sits out the rest of the day
  BLOWN   equity <= trailing floor (-$2k)
  PASSED  total >= +$3,000 -> handed off to the funded farm

ROTATION (per signal)
  • promote up to `copies` FRESH accounts -> ACTIVE (the leads); copies=1 (<=20 evals) or 2 (>20)
  • copy the signal onto EVERY ACTIVE account that still has buffer
  • an ACTIVE account keeps copying until DONE (daily cap) or BLOWN; at end-of-day DONE/ACTIVE resume
    ACTIVE for tomorrow with the daily tally reset. Pass = +$3k total without a day exceeding the cap.

EQUITY FIELDS (from NT8, validated live later): cash=CashValue (realized, EOD/pass), equity=NetLiq
(floor/blow). Trailing-then-lock floor = min($50k, peak_equity − $2k).

Run `python live/farm/eval_passer.py` for a quick hand-crafted demo of the transitions; see
sim_eval_passer.py to run a pool of accounts on real outcomes with per-account state prints.
"""
from __future__ import annotations
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from enum import Enum

START = 50_000.0
DD = 2_000.0
LOCK_BUFFER = 100.0      # floor locks at start_bal + this once up (DD + buffer) at EOD
TARGET = 3_000.0
DAY_CAP_50 = 1_500.0     # 50%-consistency firm: max profit bankable per day
DAY_CAP_40 = 1_200.0     # 40%-consistency firm


class EState(Enum):
    FRESH = "FRESH"
    ACTIVE = "ACTIVE"
    DONE = "DONE"
    BLOWN = "BLOWN"
    PASSED = "PASSED"


@dataclass
class EvalAccount:
    id: str
    state: EState = EState.FRESH
    start_bal: float = START
    dd: float = DD                   # per-account trailing distance (sim)
    dd_remaining: float | None = None  # LIVE remaining DD from NT8 (TrailingMaxDrawdown); None -> compute
    peak_balance: float = START      # high-water of EOD realized BALANCE -> trailing floor basis (sim)
    cash: float = START
    equity: float = START
    day_profit: float = 0.0          # realized P&L banked TODAY (executor caps it at the daily cap)
    last_seq: int = 0                # round-robin marker: bigger = more recently traded (de-corr mode)

    @property
    def total(self) -> float:
        return self.cash - self.start_bal

    @property
    def floor(self) -> float:
        # LIVE: kill level = equity - NT8's remaining DD. SIM: computed EOD-trailing floor.
        if self.dd_remaining is not None:
            return self.equity - self.dd_remaining
        return min(self.start_bal + LOCK_BUFFER, self.peak_balance - self.dd)

    @property
    def buffer(self) -> float:
        if self.dd_remaining is not None:
            return max(0.0, self.dd_remaining)
        return max(0.0, self.equity - self.floor)

    def to_dict(self) -> dict:
        return {"id": self.id, "state": self.state.value, "start_bal": self.start_bal, "dd": self.dd,
                "peak_balance": self.peak_balance, "cash": self.cash, "equity": self.equity,
                "day_profit": self.day_profit, "last_seq": self.last_seq}

    @classmethod
    def from_dict(cls, d: dict) -> "EvalAccount":
        a = cls(id=d["id"], start_bal=d["start_bal"], dd=d.get("dd", DD),
                peak_balance=d.get("peak_balance", d["start_bal"]),
                cash=d["cash"], equity=d["equity"], day_profit=d.get("day_profit", 0.0))
        a.state = EState(d["state"])
        a.last_seq = d.get("last_seq", 0)
        return a


@dataclass
class Route:
    account_id: str
    strat: str
    intent: str          # "EVAL"
    size_hint: str       # "risk~$<day_cap>"  (1:1 bracket sized to the daily cap)


class EvalFarm:
    def __init__(self, copies: int = 1, day_cap: float = DAY_CAP_50,
                 eval_pattern: str | None = None, quiet: bool = False):
        self.accounts: dict[str, EvalAccount] = {}
        # Round-robin: each signal -> the next `copies` least-recently-traded accounts.
        #   copies=1 -> fully de-correlated (low variance, the default).
        #   copies=2 -> two accounts per signal (your copy-2-above-20; same rotation, just 2 at once).
        self.copies = copies
        self.day_cap = day_cap
        self.eval_re = re.compile(eval_pattern) if eval_pattern else None
        self.quiet = quiet
        self.passed_ids: list[str] = []       # -> hand these to the funded farm
        self.blown_ids: list[str] = []
        self._tick = 0                        # monotonic counter for round-robin ordering
        self.log: list[str] = []

    # -- ingest /accounts snapshot --------------------------------------------------------------
    def sync_accounts(self, snapshot: dict[str, dict]) -> list[str]:
        """Add newly-appeared eval accounts as FRESH; update equity/peak/floor; flag BLOWN (equity
        breached the floor) and PASSED (total >= +$3k). Returns new ids."""
        added: list[str] = []
        for aid, m in snapshot.items():
            if self.eval_re and not self.eval_re.match(aid):
                continue
            a = self.accounts.get(aid)
            if a is None:
                # start_bal from the registry (plan baseline: $2k sims, $0 MFF) or first-seen cash.
                a = EvalAccount(id=aid, start_bal=m.get("start_bal", m["cash"]), peak_balance=m["cash"],
                                cash=m["cash"], equity=m["equity"], dd_remaining=m.get("dd"))
                self.accounts[aid] = a
                added.append(aid)
                self._say(f"+ new eval {aid} -> FRESH (bal ${m['cash']:,.0f})")
            elif a.state in (EState.BLOWN, EState.PASSED):
                continue
            else:
                a.cash, a.equity = m["cash"], m["equity"]
                if "dd" in m:
                    a.dd_remaining = m["dd"]  # live remaining DD (NT8 TrailingMaxDrawdown)
            if "realized_today" in m:
                a.day_profit = m["realized_today"]   # live today's P&L (resets at prop EOD -> restart/day-proof)
                if a.state is EState.FRESH and abs(a.day_profit) > 1e-9:
                    a.state = EState.ACTIVE          # it has already traded today
            if a.equity <= a.floor + 1e-9:
                self._blow(a)
            elif a.total >= TARGET:
                self._pass(a)
            elif a.state is EState.ACTIVE and a.day_profit >= self.day_cap - 1e-9:
                a.state = EState.DONE
        return added

    def _eligible(self) -> list:
        # in-play, has buffer, not at today's cap. Strat does NOT restrict eligibility — an account
        # can take OD again whether it won or lost OD before; the only question is whose turn it is.
        return [a for a in self.accounts.values()
                if a.state in (EState.FRESH, EState.ACTIVE) and a.buffer > 0]

    # -- route one signal: round-robin to the next `copies` accounts ----------------------------
    def route_signal(self, strat: str, today: date) -> list[Route]:
        """The signal goes to the next `copies` least-recently-traded accounts. copies=1 = fully
        de-correlated; copies=2 = the same round-robin, two accounts at a time. No account takes two
        of the same rotation; an account waits its turn before trading again."""
        takers = sorted(self._eligible(), key=lambda x: (x.last_seq, x.id))[: self.copies]
        routes = []
        for a in takers:
            self._tick += 1
            a.last_seq = self._tick
            if a.state is EState.FRESH:
                a.state = EState.ACTIVE
            routes.append(Route(a.id, strat, "EVAL", f"risk~${self.day_cap:,.0f}"))
        if routes:
            self._say(f"route {strat} -> {[r.account_id for r in routes]}")
        return routes

    # -- a copied position closed --------------------------------------------------------------
    def on_position_closed(self, account_id: str, strat: str, realized_pnl: float) -> None:
        """Add the realized result to today's tally; flag DONE at the daily cap. BLOWN/PASSED are
        decided in sync_accounts off equity/total (call sync with the post-close snapshot first)."""
        a = self.accounts.get(account_id)
        if a is None or a.state is not EState.ACTIVE:
            return
        a.day_profit += realized_pnl
        if a.day_profit >= self.day_cap - 1e-9:
            a.state = EState.DONE
            self._say(f"{a.id}: daily cap (+${a.day_profit:,.0f}) -> DONE for the day")

    # -- end of day: DONE/ACTIVE resume tomorrow -----------------------------------------------
    def end_of_day(self, today: date) -> None:
        for a in self.accounts.values():
            if a.state in (EState.BLOWN, EState.PASSED):
                continue
            if a.cash > a.peak_balance:          # EOD trailing ratchet (realized closing balance)
                a.peak_balance = a.cash
            if a.state in (EState.DONE, EState.ACTIVE):
                a.state = EState.ACTIVE
                a.day_profit = 0.0

    # -- transitions / helpers -----------------------------------------------------------------
    def _blow(self, a: EvalAccount) -> None:
        a.state = EState.BLOWN
        self.blown_ids.append(a.id)
        self._say(f"{a.id}: BLOWN (total ${a.total:,.0f})")

    def _pass(self, a: EvalAccount) -> None:
        a.state = EState.PASSED
        self.passed_ids.append(a.id)
        self._say(f"{a.id}: PASSED (total ${a.total:,.0f}) -> handoff to funded farm")

    def _say(self, msg: str) -> None:
        self.log.append(msg)
        if not self.quiet:
            print(f"  {msg}")

    def counts(self) -> Counter:
        return Counter(a.state.value for a in self.accounts.values())

    def next_signal_takers(self) -> tuple[list[str], list[str]]:
        """The next `copies` accounts in the round-robin (the Ready bar)."""
        takers = sorted(self._eligible(), key=lambda x: (x.last_seq, x.id))[: self.copies]
        return [a.id for a in takers], []

    def save_state(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"copies": self.copies, "day_cap": self.day_cap, "tick": self._tick,
                       "accounts": [a.to_dict() for a in self.accounts.values()],
                       "passed_ids": self.passed_ids, "blown_ids": self.blown_ids}, f, indent=2)

    def load_state(self, path: str) -> None:
        """Reload rotation bookkeeping across a restart/sleep. Live numbers (cash/equity/dd/day P&L)
        are then refreshed from NT8 on the next sync_accounts, so only the state machine is persisted."""
        if not os.path.exists(path):
            return
        with open(path) as f:
            d = json.load(f)
        self.accounts = {x["id"]: EvalAccount.from_dict(x) for x in d.get("accounts", [])}
        self.passed_ids = d.get("passed_ids", [])
        self.blown_ids = d.get("blown_ids", [])
        self._tick = d.get("tick", 0)

    def state_table(self) -> str:
        order = {EState.ACTIVE: 0, EState.DONE: 1, EState.FRESH: 2, EState.PASSED: 3, EState.BLOWN: 4}
        rows = []
        for a in sorted(self.accounts.values(), key=lambda x: (order[x.state], x.id)):
            rows.append(f"    {a.id:>6}  {a.state.value:<7}  total ${a.total:>+8,.0f}  "
                        f"today ${a.day_profit:>+7,.0f}  buffer ${a.buffer:>6,.0f}")
        return "\n".join(rows)


# ============================ quick hand-crafted transition demo =============================
def _demo() -> None:
    from datetime import timedelta
    print("EVAL FARM demo: lead+copy rotation, 3 accounts, daily cap $1,500\n")
    farm = EvalFarm(copies=1)
    d = date(2026, 6, 4)
    farm.sync_accounts({f"E{i}": {"cash": START, "equity": START} for i in (1, 2, 3)})

    def signal(strat, outcomes):
        """outcomes: {acct_id: pnl}; routes the signal then applies each close."""
        routes = farm.route_signal(strat, d)
        for r in routes:
            pnl = outcomes.get(r.account_id, 0.0)
            a = farm.accounts[r.account_id]
            cap = min(pnl, farm.day_cap - a.day_profit) if pnl > 0 else pnl
            new_cash = a.cash + cap
            farm.sync_accounts({r.account_id: {"cash": new_cash, "equity": new_cash}})
            farm.on_position_closed(r.account_id, strat, cap)

    print(f"[{d}] OD fires -> E1 leads; E1 wins the cap")
    signal("OD", {"E1": +1500})
    print(f"[{d}] RV fires -> E2 leads; E1 is DONE so only E2 takes it; E2 loses $1,500")
    signal("RV", {"E2": -1500})
    print(f"[{d}] B2 fires -> E3 leads; E2 (active, $500 buffer) copies; both lose")
    signal("B2", {"E2": -500, "E3": -800})
    print(f"   E2 blows (buffer gone); E3 survives\n   state:\n{farm.state_table()}")
    farm.end_of_day(d)

    d += timedelta(days=1)
    print(f"\n[{d}] new day — E1 resumes (needs +$1,500 more to pass), E3 continues")
    signal("OD", {"E1": +1500, "E3": +900})
    print(f"   E1 hits +$3,000 total -> PASSED\n   state:\n{farm.state_table()}")
    print(f"\n   counts: {dict(farm.counts())}   passed -> {farm.passed_ids}")


if __name__ == "__main__":
    _demo()
