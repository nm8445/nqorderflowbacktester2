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
    MILKING = "MILKING"   # close to passing: out of the $-bracket rotation, milks 1 MNQ 4-strat to target
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
    # --- per-account PLAN (e.g. 5%ers-50% vs Tradeify-40%). `plan` is the source of truth (persisted);
    #     the app resolves it into the 4 numbers below, which the brain logic reads. ---
    plan: str = "default"
    day_cap: float = DAY_CAP_50      # max daily profit / per-trade risk
    stop_new_at: float = DAY_CAP_50  # day_profit at/above -> DONE (no new trades today)
    target: float = TARGET           # pass threshold
    pass_buffer: float = 0.0         # force-close the final trade at target+buffer
    consistency: float = 0.5         # firm rule: biggest day must be <= this fraction of total
    peak_day_profit: float = 0.0     # biggest finalized day's profit -> drives the consistency target
    milk_at: float = 2800.0          # total >= this (but < true_target) -> eval-milking phase
    manual_phase: bool = False       # dashboard lock: hold MILKING/ACTIVE, suppress auto milk_at/DONE
    #                                  transitions (blow/pass still terminal). AUTO releases it.
    risk_mode: str = "auto"          # per-account risk tier: 'auto' ($750 while buffer < $2k, else day_cap)
    #                                  | '750' (force tight) | '1500' (force full). Dashboard-settable.
    manual_blown: bool = False       # dashboard 'mark blown' (firm force-closed) — STICKY: sync skips it
    #                                  so it can't be revived by the recreated-same-name guard.
    manual_passed: bool = False      # dashboard 'mark passed' (firm passed it, farm missed it) — STICKY too.

    @property
    def total(self) -> float:
        return self.cash - self.start_bal

    @property
    def true_target(self) -> float:
        """Consistency-adjusted pass target. Passing needs total >= target AND biggest_day <=
        consistency*total, i.e. total >= max(target, biggest_day / consistency)."""
        c = self.consistency if self.consistency > 0 else 1.0
        return max(self.target, self.peak_day_profit / c)

    @property
    def floor(self) -> float:
        # TRUE EOD-trailing floor from the peak EOD BALANCE (frozen intraday), locked at start+$100 once
        # banked. We do NOT use NT8's dd_remaining: it trails INTRADAY netliq (running high incl.
        # unrealized), reading too tight vs the firm's EOD-realized DD. peak_balance ratchets on EOD cash.
        return min(self.start_bal + LOCK_BUFFER, self.peak_balance - self.dd)

    @property
    def buffer(self) -> float:
        """Room to the EOD-trailing floor = equity - floor. Floor frozen intraday, so today's realized
        gains lift it 1:1 (matches the firm). Drives tier-sizing + auto-blow + blow detection."""
        return max(0.0, self.equity - self.floor)

    def to_dict(self) -> dict:
        return {"id": self.id, "state": self.state.value, "start_bal": self.start_bal, "dd": self.dd,
                "peak_balance": self.peak_balance, "cash": self.cash, "equity": self.equity,
                "day_profit": self.day_profit, "last_seq": self.last_seq, "plan": self.plan,
                "day_cap": self.day_cap, "stop_new_at": self.stop_new_at,
                "target": self.target, "pass_buffer": self.pass_buffer,
                "consistency": self.consistency, "peak_day_profit": self.peak_day_profit,
                "milk_at": self.milk_at, "manual_phase": self.manual_phase,
                "risk_mode": self.risk_mode, "manual_blown": self.manual_blown,
                "manual_passed": self.manual_passed}

    @classmethod
    def from_dict(cls, d: dict) -> "EvalAccount":
        a = cls(id=d["id"], start_bal=d["start_bal"], dd=d.get("dd", DD),
                peak_balance=d.get("peak_balance", d["start_bal"]),
                cash=d["cash"], equity=d["equity"], day_profit=d.get("day_profit", 0.0))
        a.state = EState(d["state"])
        a.last_seq = d.get("last_seq", 0)
        a.plan = d.get("plan", "default")
        a.day_cap = d.get("day_cap", DAY_CAP_50)
        a.stop_new_at = d.get("stop_new_at", DAY_CAP_50)
        a.target = d.get("target", TARGET)
        a.pass_buffer = d.get("pass_buffer", 0.0)
        a.consistency = d.get("consistency", 0.5)
        a.peak_day_profit = d.get("peak_day_profit", 0.0)
        a.milk_at = d.get("milk_at", 2800.0)
        a.manual_phase = d.get("manual_phase", False)
        a.risk_mode = d.get("risk_mode", "auto")
        a.manual_blown = d.get("manual_blown", False)
        a.manual_passed = d.get("manual_passed", False)
        return a


@dataclass
class Route:
    account_id: str
    strat: str
    intent: str          # "EVAL"
    size_hint: str       # "risk~$<day_cap>"  (1:1 bracket sized to the daily cap)


class EvalFarm:
    def __init__(self, copies: int = 1, day_cap: float = DAY_CAP_50, target: float = TARGET,
                 stop_new_at: float | None = None, pass_buffer: float = 0.0,
                 eval_pattern: str | None = None, quiet: bool = False):
        self.accounts: dict[str, EvalAccount] = {}
        self.target = target                  # pass threshold (real 50k eval = $3,000)
        # Round-robin: each signal -> the next `copies` least-recently-traded accounts.
        #   copies=1 -> fully de-correlated (low variance, the default).
        #   copies=2 -> two accounts per signal (your copy-2-above-20; same rotation, just 2 at once).
        self.copies = copies
        self.day_cap = day_cap                # max profit banked per day (also = per-trade risk, 1:1)
        # day_profit at/above this -> DONE (no NEW trades today). Below day_cap so an account that's
        # already near the cap doesn't take a pointless trade it would only get force-closed out of.
        self.stop_new_at = stop_new_at if stop_new_at is not None else day_cap
        # Force-close the FINAL trade at target+pass_buffer (a small slippage/commission cushion) so the
        # realized pass total lands just over the line instead of overshooting and raising the bar.
        self.pass_buffer = pass_buffer
        self.eval_re = re.compile(eval_pattern) if eval_pattern else None
        self.quiet = quiet
        self.passed_ids: list[str] = []       # -> hand these to the funded farm
        self.blown_ids: list[str] = []
        self.removed_ids: list[str] = []      # user-removed (dead/passed accts NT8 still lists) -> never re-add
        self._tick = 0                        # monotonic counter for round-robin ordering
        self.log: list[str] = []

    # -- ingest /accounts snapshot --------------------------------------------------------------
    RESET_TOL = 1.0   # $ tolerance for "balance is back at baseline" reset detection

    def _looks_reset(self, a: "EvalAccount", m: dict) -> bool:
        """A known account reappears at its baseline balance in a state that's impossible there.
        Signals it was deleted + recreated under the same name (e.g. a OneDrive desync wiped NT8's
        accounts and they were rebuilt fresh). DONE/MILKING require banked profit and BLOWN/PASSED
        are terminal off-baseline outcomes, so none of them can legitimately sit at exactly start_bal
        with nothing booked today. ACTIVE/FRESH at baseline are normal and left alone."""
        at_baseline = abs(m.get("cash", a.cash) - a.start_bal) <= self.RESET_TOL
        fresh_today = abs(m.get("realized_today", 0.0)) <= self.RESET_TOL
        return at_baseline and (
            a.state in (EState.BLOWN, EState.PASSED)
            or (a.state in (EState.DONE, EState.MILKING) and fresh_today))

    def _reset_account(self, a: "EvalAccount", m: dict) -> None:
        """Wipe a reappeared account's carried-over state machine back to a clean FRESH baseline."""
        prev = a.state.value
        a.state = EState.FRESH
        a.cash = a.equity = m.get("cash", a.start_bal)
        a.peak_balance = a.start_bal
        a.day_profit = 0.0
        a.peak_day_profit = 0.0
        a.last_seq = 0
        if "dd" in m:
            a.dd_remaining = m["dd"]
        if a.id in self.blown_ids:
            self.blown_ids.remove(a.id)
        if a.id in self.passed_ids:
            self.passed_ids.remove(a.id)
        self._say(f"{a.id}: reappeared at baseline ${a.start_bal:,.0f} while {prev} -> RESET to FRESH "
                  f"(account looks recreated)")

    def sync_accounts(self, snapshot: dict[str, dict]) -> list[str]:
        """Add newly-appeared eval accounts as FRESH; update equity/peak/floor; flag BLOWN (equity
        breached the floor) and PASSED (total >= +$3k). Returns new ids."""
        added: list[str] = []
        for aid, m in snapshot.items():
            if self.eval_re and not self.eval_re.match(aid):
                continue
            if aid in self.removed_ids:
                continue                              # user-removed: never re-add (NT8 still lists dead accts)
            a = self.accounts.get(aid)
            if a is not None and (a.manual_blown or a.manual_passed):
                continue                              # STICKY manual blow/pass: never revert or re-sync it
            if a is None:
                # start_bal from the registry (plan baseline: $2k sims, $0 MFF) or first-seen cash.
                a = EvalAccount(id=aid, start_bal=m.get("start_bal", m["cash"]), peak_balance=m["cash"],
                                cash=m["cash"], equity=m["equity"], dd_remaining=m.get("dd"))
                self.accounts[aid] = a
                added.append(aid)
                self._say(f"+ new eval {aid} -> FRESH (bal ${m['cash']:,.0f})")
            elif self._looks_reset(a, m):
                # Recreated-same-name guard (runs even for terminal BLOWN/PASSED so they recover).
                self._reset_account(a, m)
                added.append(aid)
            elif a.state in (EState.BLOWN, EState.PASSED):
                continue
            else:
                a.cash, a.equity = m["cash"], m["equity"]
                if "dd" in m:
                    a.dd_remaining = m["dd"]  # live remaining DD (NT8 TrailingMaxDrawdown)
            if "realized_today" in m:
                a.day_profit = m["realized_today"]   # live today's P&L (resets at prop EOD -> restart/day-proof)
                if a.day_profit > a.peak_day_profit:
                    a.peak_day_profit = a.day_profit  # biggest day -> consistency target
                # Discretionary day transitions are suppressed while manually locked (below).
                if not a.manual_phase:
                    if a.state is EState.FRESH and abs(a.day_profit) > 1e-9:
                        a.state = EState.ACTIVE          # it has already traded today
                    # New trading day for this account: it's DONE (capped) but its daily P&L has reset to ~0,
                    # which means the firm rolled its day. Return it to the rotation NOW. Without this an
                    # account that caps once stays DONE forever — sync has no other DONE->ACTIVE path, and the
                    # timed end_of_day roll only fires once/day (and not at all on the old running build).
                    # Robust to the firm's reset clock: recovery happens the poll after realized_today clears.
                    if a.state is EState.DONE and abs(a.day_profit) <= 1e-9:
                        a.state = EState.ACTIVE
            if a.equity <= a.floor + 1e-9:
                self._blow(a)                         # BLOW is terminal even under a manual lock
            elif a.total >= a.true_target:
                self._pass(a)                         # PASS is terminal even under a manual lock
            elif a.manual_phase:
                pass                                  # locked: hold the forced state (no auto milk/DONE)
            elif a.total >= a.milk_at and a.state in (EState.ACTIVE, EState.DONE, EState.MILKING):
                if a.state is not EState.MILKING:
                    self._say(f"{a.id}: total ${a.total:,.0f} -> MILKING to ${a.true_target:,.0f} "
                              f"(biggest day ${a.peak_day_profit:,.0f})")
                a.state = EState.MILKING              # close to passing -> 1-MNQ 4-strat to true_target
            elif a.state is EState.ACTIVE and a.day_profit >= a.stop_new_at - 1e-9:
                a.state = EState.DONE
        return added

    def cap_close_room(self, a: EvalAccount) -> float:
        """$ of additional (unrealized) gain at which an OPEN trade should be force-closed: the smaller
        of today's room to the daily cap and the total room to (target + pass_buffer). Closing there
        lands the day at ~day_cap and cuts the FINAL trade short at the pass line so the account passes
        just over +$3k without overshooting (which would raise the consistency bar). <=0 => already
        at/over a line."""
        day_room = a.day_cap - a.day_profit
        pass_room = (a.true_target + a.pass_buffer) - a.total   # consistency-adjusted pass line
        return min(day_room, pass_room)

    def _eligible(self) -> list:
        # in-play, has buffer, not at today's cap. Strat does NOT restrict eligibility — an account
        # can take OD again whether it won or lost OD before; the only question is whose turn it is.
        return [a for a in self.accounts.values()
                if a.state in (EState.FRESH, EState.ACTIVE) and a.buffer > 0]

    def eligible_ids(self) -> list[str]:
        """In-play account ids. FarmState needs these to pick the lane, because ONE lane is chosen
        per signal across the eval farm AND the funded farm together."""
        return [a.id for a in self._eligible()]

    def _takers(self, lanes=None, active_lane: int | None = None) -> list:
        """Who takes this signal.

        LANES OFF (default): the next `copies` least-recently-traded accounts — copies=1 is fully
        de-correlated, copies=2 routes two at a time. Unchanged legacy behaviour.

        LANES ON: EVERY eligible account in the active lane takes it (that is the point of a lane —
        one bet held across several firms), PLUS the legacy round-robin over accounts that have NOT
        been assigned a lane, so lanes can be switched on for only part of the farm.
        """
        elig = self._eligible()
        if lanes is None or not lanes.enabled:
            return sorted(elig, key=lambda x: (x.last_seq, x.id))[: self.copies]
        takers = [a for a in elig if active_lane is not None and lanes.lane_of(a.id) == active_lane]
        loose = [a for a in elig if lanes.lane_of(a.id) is None]
        return takers + sorted(loose, key=lambda x: (x.last_seq, x.id))[: self.copies]

    # -- route one signal: the active lane (or the legacy round-robin) --------------------------
    def route_signal(self, strat: str, today: date, lanes=None,
                     active_lane: int | None = None) -> list[Route]:
        """Route one signal. `lanes`/`active_lane` come from FarmState, which picks the lane ONCE per
        signal so the eval farm and the funded farm serve the SAME lane. No account takes two of the
        same rotation; an account waits its turn before trading again."""
        takers = self._takers(lanes, active_lane)
        routes = []
        for a in takers:
            self._tick += 1
            a.last_seq = self._tick
            if a.state is EState.FRESH:
                a.state = EState.ACTIVE
            routes.append(Route(a.id, strat, "EVAL", f"risk~${a.day_cap:,.0f}"))
        if routes:
            self._say(f"route {strat} -> {[r.account_id for r in routes]}")
        return routes

    def set_state(self, account_id: str, target: str) -> dict:
        """Manually move an eval account MILKING <-> ACTIVE from the dashboard, or AUTO to release the
        lock. MILKING/ACTIVE set the state AND lock it (manual_phase=True) so sync_accounts won't
        auto-milk (total>=milk_at), auto-DONE, or auto-revert; AUTO clears the lock so the automatic rule
        resumes. BLOWN/PASSED are terminal — can't be manually set. A manually-locked MILKER milks
        unconditionally (the consistency ceiling only gates AUTO milkers)."""
        a = self.accounts.get(account_id)
        if a is None:
            return {"ok": False, "error": f"unknown account {account_id}"}
        if a.state in (EState.BLOWN, EState.PASSED):
            return {"ok": False, "error": f"account {a.state.value.lower()}"}
        t = str(target).upper()
        if t == "AUTO":
            a.manual_phase = False
            self._say(f"{a.id}: manual lock released -> AUTO (state {a.state.value})")
            return {"ok": True, "account": account_id, "state": a.state.value, "manual": False}
        if t not in ("MILKING", "ACTIVE"):
            return {"ok": False, "error": f"bad state {target}"}
        a.state = EState(t)
        a.manual_phase = True
        self._say(f"{a.id}: manually set -> {t} (locked)")
        return {"ok": True, "account": account_id, "state": a.state.value, "manual": True}

    def set_risk_mode(self, account_id: str, mode: str) -> dict:
        """Per-account risk tier: 'auto' ($750 while buffer < $2k, else day_cap), '750', or '1500'."""
        a = self.accounts.get(account_id)
        if a is None:
            return {"ok": False, "error": f"unknown account {account_id}"}
        if mode not in ("auto", "750", "1500"):
            return {"ok": False, "error": f"bad risk mode {mode}"}
        a.risk_mode = mode
        self._say(f"{a.id}: risk mode -> {mode}")
        return {"ok": True, "account": account_id, "risk_mode": mode}

    def mark_blown(self, account_id: str) -> dict:
        """Manually mark an account BLOWN (the firm force-closed it before our DD read caught up).
        STICKY: sets manual_blown so sync_accounts skips it and can never revive it."""
        a = self.accounts.get(account_id)
        if a is None:
            return {"ok": False, "error": f"unknown account {account_id}"}
        a.manual_blown = True
        if a.state is not EState.BLOWN:
            a.state = EState.BLOWN
            if a.id not in self.blown_ids:
                self.blown_ids.append(a.id)
        self._say(f"{a.id}: MANUALLY MARKED BLOWN (firm force-close)")
        return {"ok": True, "account": account_id, "state": "BLOWN"}

    def mark_passed(self, account_id: str) -> dict:
        """Manually mark an account PASSED (the firm passed it but the farm missed the +$3k). STICKY:
        sets manual_passed so sync won't revert it. Takes it out of the signal rotation."""
        a = self.accounts.get(account_id)
        if a is None:
            return {"ok": False, "error": f"unknown account {account_id}"}
        a.manual_passed = True
        if a.state is not EState.PASSED:
            a.state = EState.PASSED
            if a.id not in self.passed_ids:
                self.passed_ids.append(a.id)
        self._say(f"{a.id}: MANUALLY MARKED PASSED")
        return {"ok": True, "account": account_id, "state": "PASSED"}

    def revive(self, account_id: str) -> dict:
        """Undo a MANUAL blow/pass (or an auto-blow): clear the sticky flag and return the account to
        ACTIVE. Safety valve for a false blow (bad NT8 read) or a mis-click. Won't touch a GENUINE
        pass/blow (state terminal but no manual flag) — those self-correct on sync, so leave them."""
        a = self.accounts.get(account_id)
        if a is None:
            return {"ok": False, "error": f"unknown account {account_id}"}
        if not (a.state is EState.BLOWN or a.manual_blown or a.manual_passed):
            return {"ok": False, "error": f"{account_id} is {a.state.value}, not manually blown/passed"}
        a.manual_blown = False
        a.manual_passed = False
        if a.id in self.blown_ids:
            self.blown_ids.remove(a.id)
        if a.id in self.passed_ids:
            self.passed_ids.remove(a.id)
        a.state = EState.ACTIVE
        self._say(f"{a.id}: REVIVED -> ACTIVE")
        return {"ok": True, "account": account_id, "state": "ACTIVE"}

    def remove(self, account_id: str) -> dict:
        """Remove an account from the farm entirely (e.g. a passed eval whose NT8 account is dead but
        still lingers in Account.All). Blocklisted in removed_ids so sync_accounts won't re-add it."""
        if account_id not in self.accounts:
            return {"ok": False, "error": f"{account_id} not an eval account"}
        self.accounts.pop(account_id, None)
        if account_id not in self.removed_ids:
            self.removed_ids.append(account_id)
        self._say(f"{account_id}: REMOVED from farm (blocklisted)")
        return {"ok": True, "account": account_id, "removed": True}

    # -- a copied position closed --------------------------------------------------------------
    def on_position_closed(self, account_id: str, strat: str, realized_pnl: float) -> None:
        """Add the realized result to today's tally; flag DONE at the daily cap. BLOWN/PASSED are
        decided in sync_accounts off equity/total (call sync with the post-close snapshot first)."""
        a = self.accounts.get(account_id)
        if a is None or a.state is not EState.ACTIVE:
            return
        a.day_profit += realized_pnl
        if a.day_profit > a.peak_day_profit:
            a.peak_day_profit = a.day_profit
        if a.day_profit >= a.stop_new_at - 1e-9:
            a.state = EState.DONE
            self._say(f"{a.id}: daily cap (+${a.day_profit:,.0f}) -> DONE for the day")

    # -- end of day: DONE/ACTIVE resume tomorrow -----------------------------------------------
    def end_of_day(self, today: date) -> None:
        for a in self.accounts.values():
            if a.state in (EState.BLOWN, EState.PASSED):
                continue
            if a.cash > a.peak_balance:          # EOD trailing ratchet (realized closing balance)
                a.peak_balance = a.cash
            if a.day_profit > a.peak_day_profit:  # finalize the biggest-day tracker
                a.peak_day_profit = a.day_profit
            if a.state in (EState.DONE, EState.ACTIVE):
                a.state = EState.ACTIVE
                a.day_profit = 0.0
            elif a.state is EState.MILKING:       # stay MILKING; just reset the daily tally
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

    def next_signal_takers(self, lanes=None, active_lane: int | None = None) -> tuple[list[str], list[str]]:
        """Who would take the next signal (the dashboard's Ready bar). Mirrors _takers exactly so the
        bar can't drift from what routing actually does."""
        return [a.id for a in self._takers(lanes, active_lane)], []

    def save_state(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"copies": self.copies, "day_cap": self.day_cap, "tick": self._tick,
                       "accounts": [a.to_dict() for a in self.accounts.values()],
                       "passed_ids": self.passed_ids, "blown_ids": self.blown_ids,
                       "removed_ids": self.removed_ids}, f, indent=2)

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
        self.removed_ids = d.get("removed_ids", [])
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
