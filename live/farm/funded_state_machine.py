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
    peak_balance: float = START          # high-water of EOD realized BALANCE -> trailing floor basis (sim)
    cash: float = START                 # realized balance (CashValue)
    equity: float = START               # NetLiquidation (realized + floating)
    day_start_cash: float = START        # cash at the open of today (winning-day calc)
    winning_days: int = 0
    payouts_done: int = 0
    last_gamble_date: date | None = None  # enforces 1 gamble/account/day
    last_lost_signal: str | None = None   # survivor rotates away from this
    retired_reason: str = ""

    @property
    def floor(self) -> float:
        # LIVE: NT8 reports the remaining DD, so kill level = equity - remaining.
        # SIM (no feed): EOD-trailing floor (frozen intraday, locks at start + $100).
        if self.dd_remaining is not None:
            return self.equity - self.dd_remaining
        return min(self.start_bal + LOCK_BUFFER, self.peak_balance - self.dd)

    @property
    def buffer(self) -> float:
        """Room to the floor (= NT8's live remaining DD) — what a gamble is sized against."""
        if self.dd_remaining is not None:
            return max(0.0, self.dd_remaining)
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
            a = self.accounts.get(aid)
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

    # -- a position closed (from the tick monitor) ---------------------------------------------
    def on_position_closed(self, account_id: str, strat: str, realized_pnl: float) -> None:
        """Call AFTER sync_accounts has applied the post-close snapshot (so a.cash is current).
        De-risk only when the ACCOUNT is net in profit; otherwise keep gambling."""
        a = self.accounts.get(account_id)
        if a is None or a.phase is not Phase.GAMBLING:
            return  # milk closes need no per-trade action; winning-days are tallied EOD
        if a.cash > a.start_bal:                      # account net positive -> bank the buffer
            a.phase = Phase.MILKING
            a.last_lost_signal = None
            self._say(f"{a.id}: account +${a.cash - a.start_bal:,.0f} -> DE-RISK -> MILKING")
        elif realized_pnl > 0:                        # won, but still underwater -> keep gambling
            a.last_lost_signal = None                 # last gamble wasn't a loss; no signal to avoid
            self._say(f"{a.id}: gamble {strat} +${realized_pnl:,.0f}, still underwater "
                      f"(${a.cash - a.start_bal:,.0f}) -> keep gambling")
        else:                                         # lost but survived -> re-gamble, rotate signal
            a.last_lost_signal = strat
            self._say(f"{a.id}: gamble {strat} ${realized_pnl:,.0f} (buffer ${a.buffer:,.0f}) "
                      f"-> re-gamble (avoid {strat})")

    # -- route one signal ----------------------------------------------------------------------
    def route_signal(self, strat: str, today: date) -> list[Route]:
        """Called each time a signal fires. Every milker copies it at 1 MNQ; at most ONE eligible
        gambler takes it (de-correlated)."""
        routes = [Route(a.id, strat, "MILK", "1MNQ")
                  for a in self.accounts.values() if a.phase is Phase.MILKING]
        g = self._next_gambler(strat, today)
        if g is not None:
            g.last_gamble_date = today
            routes.append(Route(g.id, strat, "GAMBLE", f"risk~${g.buffer:,.0f}"))
            self._say(f"route {strat}: GAMBLE->{g.id} (risk ${g.buffer:,.0f}) + "
                      f"MILK->{sum(1 for r in routes if r.intent=='MILK')} accts")
        return routes

    def _next_gambler(self, strat: str, today: date) -> FundedAccount | None:
        cands = [a for a in self.accounts.values()
                 if a.phase is Phase.GAMBLING
                 and a.last_gamble_date != today        # 1 gamble/day
                 and a.last_lost_signal != strat]       # don't repeat the signal you just lost on
        if not cands:
            return None
        # fairness: whoever has waited longest (never-gambled first), then by id
        cands.sort(key=lambda a: (a.last_gamble_date or date.min, a.id))
        return cands[0]

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
