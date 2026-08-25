"""Funded-account farm — the BACKEND BRAIN (pure Python, no NT8/MT5 deps, sim-testable).

This is the router we build first. It owns the lifecycle of every FUNDED account and decides,
for each signal that fires, which accounts take it and at what size. The NT8 test addon (built
later) is only its hands: it feeds `/accounts` snapshots in and executes the Routes that come out.

LIFECYCLE
  GAMBLING ─ account in profit (equity>$50k) ─▶ MILKING ── 5 winning days ─▶ PAYOUT ─(4th)─▶ RETIRED
     │  └ closes still underwater: re-gamble (sized to remaining buffer; a LOSS rotates signal)  ▲
     └ equity ≤ floor ───────────────────────────────────────────────────────────────────────┘ (blown)

  De-risk is gated on the ACCOUNT being net positive, not on a single green trade: an account down to
  $500 of buffer that wins $500 is still underwater -> it keeps gambling (each re-gamble sized to its
  current buffer, so it shrinks as the cushion erodes) until it climbs back above $50k or blows.

DE-CORRELATION (the whole point of the gamble phase)
  • across accounts: no two gamblers take the same signal on the same day
  • within an account: a survivor's next gamble != the signal it just lost on
  Milkers are deliberately CORRELATED (everyone copies the 4-way at 1 MNQ) — the de-correlation
  budget is spent entirely on the gamble, so accounts reach milk staggered, with different buffers.

EQUITY FIELDS (from NT8 Account API, validated live later)
  • cash    = CashValue              (realized balance)      -> EOD winning-days
  • equity  = NetLiquidation         (cash + unrealized)     -> floor / blow / peak
  Trailing-then-lock floor = min($50k, peak_equity − $2k): trails up, locks at $50k once +$2k banked.

Run `python live/farm/funded_state_machine.py` for a self-contained sim of the transitions.
"""
from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

# ---- rule constants (50k futures funded) ----------------------------------------------------
START = 50_000.0
DD = 2_000.0                 # trailing drawdown distance (per-account default; from the registry live)
LOCK_BUFFER = 100.0          # floor locks at start_bal + this once up (DD + buffer) at EOD -> you keep it
SIGNALS = ("OD", "RV", "B2", "FB")   # firing order within a day (OD enters 19:00 prev eve)
WIN_DAY_MIN = 150.0          # a "winning day" = realized cash gain >= this
WIN_DAYS_FOR_PAYOUT = 5
MAX_PAYOUTS = 4
PAYOUT_FRAC = 0.50           # withdraw 50% of profit ...
PAYOUT_CAP = 2_000.0         # ... capped at $2k
SPLIT = 0.80                 # trader keeps 80% of each withdrawal (firm takes the rest)


class Phase(Enum):
    GAMBLING = "GAMBLING"
    MILKING = "MILKING"
    RETIRED = "RETIRED"


@dataclass
class FundedAccount:
    id: str
    phase: Phase = Phase.GAMBLING
    start_bal: float = START
    dd: float = DD                       # per-account trailing distance (sim)
    dd_remaining: float | None = None    # LIVE remaining DD from NT8 (TrailingMaxDrawdown); None -> compute
    derisk_at: float | None = None       # de-risk GAMBLING->MILKING once profit >= this ($); None -> use
    #                                      dd (= bank the drawdown -> floor locks at start). NOT "any profit".
    peak_balance: float = START          # high-water of EOD realized BALANCE -> trailing floor basis (sim)
    cash: float = START                 # realized balance (CashValue)
    equity: float = START               # NetLiquidation (realized + floating)
    day_start_cash: float = START        # cash at the open of today (winning-day calc)
    winning_days: int = 0
    payouts_done: int = 0
    last_gamble_date: date | None = None  # informational: last day this account gambled
    last_lost_signal: str | None = None   # survivor rotates away from this (legacy; unused by round-robin)
    last_seq: int = 0                     # eval-style round-robin recency marker (bigger = more recent)
    scaling_profit: float = 0.0           # realized profit AS OF the last session close -> LucidFlex scaling tier
    retired_reason: str = ""
    # Manual phase lock: when True, the auto GAMBLING->MILKING de-risk is suppressed so you can keep an
    # account gambling for a bigger buffer (or force it to milk) regardless of profit. Cleared via the
    # dashboard "Auto" button. Blows still happen regardless — you can't un-blow a breached account.
    manual_phase: bool = False
    risk_mode: str = "auto"          # per-account gamble risk tier: 'auto' ($750 while buffer < $2k,
    #                                  else $1500) | '750' | '1500'. Dashboard-settable.
    manual_blown: bool = False       # dashboard 'mark blown' (firm force-closed) — STICKY, retired

    def to_dict(self) -> dict:
        return {"id": self.id, "phase": self.phase.value, "start_bal": self.start_bal, "dd": self.dd,
                "peak_balance": self.peak_balance, "cash": self.cash, "equity": self.equity,
                "day_start_cash": self.day_start_cash, "winning_days": self.winning_days,
                "payouts_done": self.payouts_done,
                "last_gamble_date": self.last_gamble_date.isoformat() if self.last_gamble_date else None,
                "last_lost_signal": self.last_lost_signal, "retired_reason": self.retired_reason,
                "manual_phase": self.manual_phase, "derisk_at": self.derisk_at, "last_seq": self.last_seq,
                "scaling_profit": self.scaling_profit,
                "risk_mode": self.risk_mode, "manual_blown": self.manual_blown}

    @classmethod
    def from_dict(cls, d: dict) -> "FundedAccount":
        a = cls(id=d["id"], start_bal=d.get("start_bal", START), dd=d.get("dd", DD),
                peak_balance=d.get("peak_balance", START), cash=d.get("cash", START),
                equity=d.get("equity", START), day_start_cash=d.get("day_start_cash", START))
        a.phase = Phase(d.get("phase", "GAMBLING"))
        a.winning_days = d.get("winning_days", 0)
        a.payouts_done = d.get("payouts_done", 0)
        lg = d.get("last_gamble_date")
        a.last_gamble_date = date.fromisoformat(lg) if lg else None
        a.last_lost_signal = d.get("last_lost_signal")
        a.retired_reason = d.get("retired_reason", "")
        a.manual_phase = d.get("manual_phase", False)
        a.derisk_at = d.get("derisk_at")
        a.last_seq = d.get("last_seq", 0)
        a.scaling_profit = d.get("scaling_profit", 0.0)
        a.risk_mode = d.get("risk_mode", "auto")
        a.manual_blown = d.get("manual_blown", False)
        return a

    @property
    def floor(self) -> float:
        # TRUE EOD-trailing floor from the peak EOD BALANCE (frozen intraday), locked at start+$100 once
        # banked. We deliberately do NOT use NT8's dd_remaining: it trails INTRADAY netliq (the running
        # high incl. unrealized), so it reads far too tight vs Lucid's EOD-realized DD (saw $1624 vs a
        # true $2376 after a +$376 realized day). peak_balance ratchets on EOD cash (end_of_day).
        return min(self.start_bal + LOCK_BUFFER, self.peak_balance - self.dd)

    @property
    def buffer(self) -> float:
        """Room to the EOD-trailing floor = current equity - floor. What a gamble is sized against and
        what the two-tier risk / auto-blow read. Frozen floor intraday, so today's realized gains lift it
        1:1 (a +$376 day on a fresh $2k-DD account -> ~$2,376), matching the firm's actual rule."""
        return max(0.0, self.equity - self.floor)


@dataclass
class Route:
    """One execution instruction the executor should place."""
    account_id: str
    strat: str
    intent: str          # "GAMBLE" | "MILK"
    size_hint: str       # GAMBLE -> "risk~$<buffer>"; MILK -> "1MNQ"


class FundedFarm:
    def __init__(self, funded_pattern: str | None = None, quiet: bool = False):
        # funded_pattern: regex that identifies a FUNDED account name (vs an eval). None = accept all.
        # quiet: suppress per-event prints (for Monte-Carlo runs).
        self.accounts: dict[str, FundedAccount] = {}
        self.quiet = quiet
        self.funded_re = re.compile(funded_pattern) if funded_pattern else None
        self.pending_payouts: list[tuple[str, float]] = []   # (account_id, gross amount) to withdraw at firm
        self.total_take_home = 0.0                           # cumulative trader's share (after the split)
        self.gamble_risk: float | None = None                # fixed $ risk per funded gamble; None -> full buffer
        self.derisk_at: float | None = None                  # farm-wide de-risk profit ($): None -> account DD
        #                                                      (lock the floor); 0 -> first profit; >0 -> that buffer
        self.milk_qty: int = 1                               # MNQ per signal once an account is MILKING
        self.profit_lock: float | None = None                # $ account profit -> force-close an open GAMBLE
        # Manual one-shot override: force WHICH gambling account takes the NEXT decorrelated signal
        # (dashboard "gamble next" pick). Cleared automatically once that account gambles.
        self.forced_gambler_id: str | None = None
        self._tick = 0                                       # monotonic counter for round-robin ordering
        self.busy_ids: set[str] = set()                      # GAMBLERS holding an open position (one trade/acct);
        #                                                      milkers ignore this (they run the full 4-way concurrently)
        self.removed_ids: list[str] = []                     # user-removed accounts -> never re-add on sync
        self.log: list[str] = []

    # -- ingest /accounts snapshot --------------------------------------------------------------
    def sync_accounts(self, snapshot: dict[str, dict]) -> list[str]:
        """snapshot = {account_id: {"cash": float, "equity": float}}.
        Adds newly-appeared funded accounts (the eval->funded handoff) as GAMBLING; updates equity,
        peak and the trailing floor; retires anything that has breached the floor. Returns new ids."""
        added: list[str] = []
        for aid, m in snapshot.items():
            if self.funded_re and not self.funded_re.match(aid):
                continue
            if aid in self.removed_ids:
                continue                              # user-removed: never re-add
            a = self.accounts.get(aid)
            if a is not None and a.manual_blown:
                continue                              # STICKY manual blow: never re-sync/revive it
            if a is None:
                a = FundedAccount(id=aid, start_bal=m["cash"], peak_balance=m["cash"],
                                  cash=m["cash"], equity=m["equity"], day_start_cash=m["cash"],
                                  dd_remaining=m.get("dd"))
                self.accounts[aid] = a
                added.append(aid)
                self._say(f"+ new funded account {aid} -> GAMBLING (bal ${m['cash']:,.0f})")
                continue
            a.cash, a.equity = m["cash"], m["equity"]
            if "dd" in m:
                a.dd_remaining = m["dd"]      # live remaining DD (NT8 TrailingMaxDrawdown)
            # blow when equity hits the floor (live: remaining DD <= 0). Sim floor frozen intraday.
            if a.phase is not Phase.RETIRED and a.equity <= a.floor + 1e-9:
                self._retire(a, "blown (equity <= floor)")
        return added

    # -- de-risk threshold (gamble -> milk) ----------------------------------------------------
    def _derisk_target(self, a: FundedAccount) -> float:
        """$ profit at which a GAMBLING account de-risks to MILKING. Precedence: farm-wide setting
        (dashboard) -> per-account override -> the account's DD (= lock the floor). 0 -> 'first profit'."""
        if self.derisk_at is not None:
            return self.derisk_at
        if a.derisk_at is not None:
            return a.derisk_at
        return a.dd

    def _should_derisk(self, a: FundedAccount) -> bool:
        tgt = self._derisk_target(a)
        banked = a.cash - a.start_bal
        return banked > 1e-9 if tgt <= 0 else banked >= tgt   # tgt 0 = de-risk on any profit

    # -- a position closed (from the tick monitor) ---------------------------------------------
    def on_position_closed(self, account_id: str, strat: str, realized_pnl: float) -> None:
        """Call AFTER sync_accounts has applied the post-close snapshot (so a.cash is current).
        De-risk only once the account has BANKED ITS BUFFER (derisk_at, default = dd = the floor-lock
        point) — NOT at any profit. Below that, a winner keeps gambling to grow the buffer."""
        a = self.accounts.get(account_id)
        if a is None or a.phase is not Phase.GAMBLING:
            return  # milk closes need no per-trade action; winning-days are tallied EOD
        tgt = self._derisk_target(a)
        if self._should_derisk(a) and not a.manual_phase:       # banked the buffer -> de-risk (unless locked)
            a.phase = Phase.MILKING
            a.last_lost_signal = None
            self._say(f"{a.id}: account +${a.cash - a.start_bal:,.0f} (>= ${tgt:,.0f}) -> DE-RISK -> MILKING")
        elif realized_pnl > 0:                        # won but buffer not yet reached (or manually locked)
            a.last_lost_signal = None                 # last gamble wasn't a loss; no signal to avoid
            why = "manually locked" if a.manual_phase and self._should_derisk(a) else "buffer not reached"
            self._say(f"{a.id}: gamble {strat} +${realized_pnl:,.0f}, {why} "
                      f"(${a.cash - a.start_bal:,.0f}/${tgt:,.0f}) -> keep gambling")
        else:                                         # lost but survived -> re-gamble, rotate signal
            a.last_lost_signal = strat
            self._say(f"{a.id}: gamble {strat} ${realized_pnl:,.0f} (buffer ${a.buffer:,.0f}) "
                      f"-> re-gamble (avoid {strat})")

    # -- route one signal ----------------------------------------------------------------------
    def route_signal(self, strat: str, today: date, lanes=None,
                     active_lane: int | None = None) -> list[Route]:
        """Called each time a signal fires. EVERY milker copies EVERY signal at 1 MNQ — milking runs the
        full 4-way combined concurrently (one open trade per active strat, all milkers copy the same set),
        so milkers are NOT gated by busy_ids (that one-trade-at-a-time gate is for GAMBLERS only, which
        take a single de-correlated signal).

        LANES OFF: at most ONE eligible gambler takes each signal (legacy round-robin).
        LANES ON: every eligible gambler in the ACTIVE LANE takes it, plus the legacy single pick among
        gamblers with no lane assigned. `active_lane` is chosen by FarmState so evals and fundeds in the
        same lane trade together. MILKERS ARE NEVER LANE-GATED — milking is correlated on purpose."""
        routes = [Route(a.id, strat, "MILK", "1MNQ")
                  for a in self.accounts.values()
                  if a.phase is Phase.MILKING]
        for g in self._gamblers_for(strat, today, lanes, active_lane):
            self._tick += 1
            g.last_seq = self._tick          # advance the round-robin marker
            g.last_gamble_date = today
            routes.append(Route(g.id, strat, "GAMBLE", f"risk~${g.buffer:,.0f}"))
            self._say(f"route {strat}: GAMBLE->{g.id} (risk ${g.buffer:,.0f}) + "
                      f"MILK->{sum(1 for r in routes if r.intent=='MILK')} accts")
        return routes

    def eligible_gambler_ids(self) -> list[str]:
        """Gambling accounts that could take a signal now — FarmState needs these for the lane pick."""
        return [a.id for a in self._gamble_candidates()]

    def _gamble_candidates(self) -> list[FundedAccount]:
        return [a for a in self.accounts.values()
                if a.phase is Phase.GAMBLING and a.buffer > 0 and a.id not in self.busy_ids]

    def _gamblers_for(self, strat: str, today: date, lanes=None,
                      active_lane: int | None = None) -> list[FundedAccount]:
        """The gambler(s) taking this signal. Lanes off -> the legacy single pick. Lanes on -> the whole
        active lane + the legacy single pick among unassigned gamblers."""
        if lanes is None or not lanes.enabled:
            g = self._next_gambler(strat, today)
            return [g] if g is not None else []
        cands = self._gamble_candidates()
        picked = [a for a in cands if active_lane is not None and lanes.lane_of(a.id) == active_lane]
        loose = [a for a in cands if lanes.lane_of(a.id) is None]
        if loose:
            g = self._next_gambler(strat, today, pool=loose)
            if g is not None and g not in picked:
                picked.append(g)
        return picked

    def _next_gambler(self, strat: str, today: date,
                      pool: list[FundedAccount] | None = None) -> FundedAccount | None:
        """Eval-style round-robin: the next least-recently-gambled GAMBLING account with buffer takes
        the signal. NO 1/day limit — an account keeps gambling (accumulating toward its de-risk buffer)
        until it de-risks to MILKING or blows. A 'gamble next' forced pick wins if it's still eligible.
        `pool` narrows the candidates (lanes mode passes only the unassigned gamblers)."""
        cands = self._gamble_candidates() if pool is None else list(pool)
        if not cands:
            return None
        if self.forced_gambler_id:
            forced = next((a for a in cands if a.id == self.forced_gambler_id), None)
            if forced is not None:
                self.forced_gambler_id = None
                self._say(f"{forced.id}: manual gamble-next pick takes {strat}")
                return forced
        cands.sort(key=lambda a: (a.last_seq, a.id))   # least-recently-gambled first, then by id
        return cands[0]

    # -- manual dashboard controls -------------------------------------------------------------
    def set_phase(self, account_id: str, phase: str) -> dict:
        """Force a funded account's phase from the dashboard. phase in {GAMBLING, MILKING, AUTO}:
        GAMBLING/MILKING set the phase AND lock it (auto de-risk won't override); AUTO releases the
        lock so the brain resumes auto de-risk on the next close."""
        a = self.accounts.get(account_id)
        if a is None:
            return {"ok": False, "error": f"unknown account {account_id}"}
        if a.phase is Phase.RETIRED:
            return {"ok": False, "error": "account retired"}
        if phase == "AUTO":
            a.manual_phase = False
            self._say(f"{a.id}: manual lock released -> AUTO (phase {a.phase.value})")
            return {"ok": True, "account": account_id, "phase": a.phase.value, "manual": False}
        try:
            a.phase = Phase(phase)
        except ValueError:
            return {"ok": False, "error": f"bad phase {phase}"}
        a.manual_phase = True
        self._say(f"{a.id}: manually set -> {a.phase.value} (locked)")
        return {"ok": True, "account": account_id, "phase": a.phase.value, "manual": True}

    def set_forced_gambler(self, account_id: str) -> dict:
        """Tag a gambling account to take the NEXT decorrelated signal (one-shot)."""
        a = self.accounts.get(account_id)
        if a is None:
            return {"ok": False, "error": f"unknown account {account_id}"}
        if a.phase is not Phase.GAMBLING:
            return {"ok": False, "error": f"{account_id} is {a.phase.value}, not GAMBLING"}
        self.forced_gambler_id = account_id
        a.last_gamble_date = None       # override the 1-gamble/day limit so a forced pick is eligible NOW
        a.last_lost_signal = None       # ...and not blocked by the just-lost-signal rotation either
        self._say(f"{account_id}: tagged to gamble the next signal (daily limit cleared)")
        return {"ok": True, "account": account_id}

    def set_risk_mode(self, account_id: str, mode: str) -> dict:
        """Per-account gamble risk tier: 'auto' ($750 while buffer < $2k, else $1500), '750', '1500'."""
        a = self.accounts.get(account_id)
        if a is None:
            return {"ok": False, "error": f"unknown account {account_id}"}
        if mode not in ("auto", "750", "1500"):
            return {"ok": False, "error": f"bad risk mode {mode}"}
        a.risk_mode = mode
        self._say(f"{a.id}: risk mode -> {mode}")
        return {"ok": True, "account": account_id, "risk_mode": mode}

    def mark_blown(self, account_id: str) -> dict:
        """Manually mark a funded account BLOWN (firm force-closed it). STICKY: manual_blown + RETIRED
        so sync won't re-sync or revive it."""
        a = self.accounts.get(account_id)
        if a is None:
            return {"ok": False, "error": f"unknown account {account_id}"}
        a.manual_blown = True
        self._retire(a, "manually marked blown (firm force-close)")
        return {"ok": True, "account": account_id, "phase": "RETIRED"}

    def revive(self, account_id: str) -> dict:
        """Undo a (manual/auto) blow: clear manual_blown and return the account to GAMBLING. Only
        acts on accounts that were BLOWN (manual_blown) — NOT ones retired for other reasons (payouts
        cashed). Safety valve for a false blow; the auto-rule re-fires if it's genuinely dead."""
        a = self.accounts.get(account_id)
        if a is None:
            return {"ok": False, "error": f"unknown account {account_id}"}
        if not a.manual_blown:
            return {"ok": False, "error": f"{account_id} not blown (retired: {a.retired_reason or 'n/a'})"}
        a.manual_blown = False
        a.phase = Phase.GAMBLING
        a.retired_reason = ""
        self._say(f"{a.id}: REVIVED -> GAMBLING")
        return {"ok": True, "account": account_id, "phase": "GAMBLING"}

    def remove(self, account_id: str) -> dict:
        """Remove a funded account from the farm entirely. Blocklisted so sync won't re-add it."""
        if account_id not in self.accounts:
            return {"ok": False, "error": f"{account_id} not a funded account"}
        self.accounts.pop(account_id, None)
        if account_id not in self.removed_ids:
            self.removed_ids.append(account_id)
        self._say(f"{account_id}: REMOVED from farm (blocklisted)")
        return {"ok": True, "account": account_id, "removed": True}

    # -- end of day: tally winning-days, fire payouts ------------------------------------------
    def end_of_day(self, today: date) -> None:
        for a in self.accounts.values():
            if a.phase is Phase.RETIRED:
                continue
            if a.cash > a.peak_balance:          # EOD trailing ratchet (realized closing balance)
                a.peak_balance = a.cash
            if a.phase is Phase.MILKING and (a.cash - a.day_start_cash) >= WIN_DAY_MIN:
                a.winning_days += 1
                self._say(f"{a.id}: winning day {a.winning_days}/{WIN_DAYS_FOR_PAYOUT} "
                          f"(+${a.cash - a.day_start_cash:,.0f})")
                # need 5 winning days AND net profit (losing days can sit you sub-start at 5 wins)
                if a.winning_days >= WIN_DAYS_FOR_PAYOUT and a.cash > a.start_bal:
                    self._trigger_payout(a)
            a.day_start_cash = a.cash
            a.scaling_profit = max(0.0, a.cash - a.start_bal)   # LucidFlex scaling tier basis (EOD-updated)

    def _trigger_payout(self, a: FundedAccount) -> None:
        profit = a.cash - a.start_bal
        amt = min(PAYOUT_FRAC * profit, PAYOUT_CAP)   # withdrawn from the account
        take = SPLIT * amt                            # trader's share
        self.pending_payouts.append((a.id, amt))
        self.total_take_home += take
        a.winning_days = 0
        a.payouts_done += 1
        self._say(f"{a.id}: PAYOUT #{a.payouts_done} withdraw ${amt:,.0f} (take-home ${take:,.0f})")
        if a.payouts_done >= MAX_PAYOUTS:
            self._retire(a, f"{MAX_PAYOUTS} payouts cashed")

    def record_withdrawal(self, account_id: str, amount: float) -> None:
        """Call after the user actually withdraws at the firm. Cash drops but the trailing floor
        stays at HWM (peak_equity unchanged), so the withdrawal isn't mistaken for a loss/blow."""
        a = self.accounts.get(account_id)
        if a is None:
            return
        a.cash -= amount
        a.day_start_cash -= amount   # so the withdrawal doesn't read as a losing day
        self._say(f"{a.id}: withdrew ${amount:,.0f} (floor held at HWM)")

    def _retire(self, a: FundedAccount, reason: str) -> None:
        a.phase = Phase.RETIRED
        a.retired_reason = reason
        self._say(f"{a.id}: RETIRED ({reason})")

    # -- helpers -------------------------------------------------------------------------------
    def _say(self, msg: str) -> None:
        self.log.append(msg)
        if not self.quiet:
            print(f"  {msg}")

    def snapshot_phases(self) -> dict[str, str]:
        return {a.id: a.phase.value for a in self.accounts.values()}

    def save_state(self, path: str) -> None:
        """Persist the state machine (phase, manual lock, payout progress, rotation bookkeeping).
        Live numbers (cash/equity/dd) are refreshed from NT8 on the next sync, so they're informational
        here only."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"accounts": [a.to_dict() for a in self.accounts.values()],
                       "pending_payouts": self.pending_payouts,
                       "total_take_home": self.total_take_home,
                       "forced_gambler_id": self.forced_gambler_id,
                       "gamble_risk": self.gamble_risk, "derisk_at": self.derisk_at,
                       "milk_qty": self.milk_qty, "profit_lock": self.profit_lock,
                       "tick": self._tick, "removed_ids": self.removed_ids}, f, indent=2)

    def load_state(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path) as f:
            d = json.load(f)
        self.accounts = {x["id"]: FundedAccount.from_dict(x) for x in d.get("accounts", [])}
        self.pending_payouts = [tuple(p) for p in d.get("pending_payouts", [])]
        self.total_take_home = d.get("total_take_home", 0.0)
        self.forced_gambler_id = d.get("forced_gambler_id")
        self.gamble_risk = d.get("gamble_risk")
        self.derisk_at = d.get("derisk_at")
        self.milk_qty = d.get("milk_qty", 1)
        self.profit_lock = d.get("profit_lock")
        self._tick = d.get("tick", 0)
        self.removed_ids = d.get("removed_ids", [])


# ============================ self-contained sim of the transitions ==========================
def _demo() -> None:
    from datetime import timedelta
    print("FUNDED FARM demo: F1 clean win->milk->payout ; F2 underwater grind->milk\n")
    farm = FundedFarm(funded_pattern=None)
    d = date(2026, 6, 2)

    print(f"[{d}] two funded accounts appear")
    farm.sync_accounts({"F1": {"cash": START, "equity": START}, "F2": {"cash": START, "equity": START}})

    print(f"\n[{d}] OD -> F1 gamble, RV -> F2 gamble")
    farm.route_signal("OD", d); farm.route_signal("RV", d)
    print("  ... F1 OD +$3,000 (in profit) ; F2 RV -$1,500 (down to $48.5k, only $500 buffer)")
    farm.sync_accounts({"F1": {"cash": 53_000, "equity": 53_000}, "F2": {"cash": 48_500, "equity": 48_500}})
    farm.on_position_closed("F1", "OD", +3_000)      # net positive -> MILKING
    farm.on_position_closed("F2", "RV", -1_500)      # underwater, avoid RV next
    farm.end_of_day(d)

    d += timedelta(days=1)
    print(f"\n[{d}] OD -> F2 re-gambles (avoiding RV); size auto-shrinks to its $500 buffer")
    farm.route_signal("OD", d)                        # F1 milks OD too
    print("  ... F2 OD +$500 -> $49k, STILL underwater -> keeps gambling (no de-risk)")
    farm.sync_accounts({"F1": {"cash": 53_200, "equity": 53_200}, "F2": {"cash": 49_000, "equity": 49_000}})
    farm.on_position_closed("F2", "OD", +500)
    farm.end_of_day(d)

    d += timedelta(days=1)
    print(f"\n[{d}] B2 -> F2 gambles again (risk~$1,000 now, its grown buffer)")
    farm.route_signal("B2", d)
    print("  ... F2 B2 +$1,400 -> $50.4k -> IN PROFIT -> de-risk -> MILKING")
    farm.sync_accounts({"F1": {"cash": 53_400, "equity": 53_400}, "F2": {"cash": 50_400, "equity": 50_400}})
    farm.on_position_closed("F2", "B2", +1_400)
    farm.end_of_day(d)

    print(f"\n[{d}+] F1 keeps banking winning days -> 5th triggers first payout")
    base = 53_400
    for k in range(3):
        d += timedelta(days=1)
        farm.sync_accounts({"F1": {"cash": base + 200 * (k + 1), "equity": base + 200 * (k + 1)},
                            "F2": {"cash": 50_400, "equity": 50_400}})
        farm.end_of_day(d)

    print("\nfinal phases:", farm.snapshot_phases())
    print("pending withdrawals:", farm.pending_payouts)


if __name__ == "__main__":
    _demo()
