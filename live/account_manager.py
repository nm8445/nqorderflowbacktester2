"""
Account rotation & status tracking for multi-account funded trading.

Firm: Lucid Trading (challenge phase has 50% consistency rule)

Concept:
  - N accounts trade in round-robin (trade 1 -> acct[0], trade 2 -> acct[1], ...)
  - Each account tracks its own equity, HWM, trailing DD floor, daily PnL,
    best-day PnL (for consistency), and blown/passed status
  - Blown or passed accounts are skipped in rotation
  - If all accounts are blown/passed, trading stops

Lucid challenge rules (defaults - confirm with user):
  - Starting equity: $50,000
  - Trailing DD: $2,000 below HWM, trails up to $50,000 then locks
  - Profit target: $3,000 above starting equity
  - Consistency rule (challenge): best day's profit <= 50% of total cumulative
    profit at the moment target is hit. If violated, account must keep trading
    (more smaller days) until ratio complies.

PnL calc:
  pnl = (exit_price - entry_price) * direction * contracts * point_value

Persistence:
  State (per-account equity/HWM/DD/best_day/etc + rotation pointer) is written
  to a JSON file after every trade exit and at daily reset. On startup, if the
  file exists and the account list matches, state is restored. Restart in the
  middle of a challenge = rotation pointer preserved (Sim101 took last ->
  Sim102 next), and all challenge tracking (best_day, HWM) survives.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Optional
import json
import logging

import pandas as pd

log = logging.getLogger(__name__)

ET = "America/New_York"

# Lucid challenge defaults (same $ structure as Apex $50k sim)
STARTING_EQUITY = 50_000.0
TRAILING_DD = 2_000.0
PROFIT_TARGET = 3_000.0  # pass when equity - starting >= this
DD_LOCK_AT = STARTING_EQUITY  # (unused in challenge — DD never locks)
CONSISTENCY_RATIO = 0.50  # best day <= 50% of total profit (challenge only)


@dataclass
class Account:
    name: str
    equity: float = STARTING_EQUITY
    hwm: float = STARTING_EQUITY          # high water mark
    dd_floor: float = STARTING_EQUITY - TRAILING_DD  # if equity hits this, account is blown
    dd_locked: bool = False               # True once HWM reaches DD_LOCK_AT
    daily_pnl: float = 0.0
    best_day_pnl: float = 0.0             # for consistency rule (highest single-day profit)
    last_reset_date: Optional[date] = None  # ET session date of last daily reset
    blown: bool = False
    passed: bool = False                  # target hit AND consistency satisfied
    target_hit: bool = False              # equity >= start + target (may still need consistency)
    trades_taken: int = 0
    phase: str = "challenge"              # "challenge" or "funded"

    @property
    def available(self) -> bool:
        return not self.blown and not self.passed

    @property
    def total_profit(self) -> float:
        return self.equity - STARTING_EQUITY

    @property
    def consistency_ok(self) -> bool:
        """
        Challenge consistency: best single day <= 50% of total profit.
        Only meaningful when total_profit > 0 and in challenge phase.
        Funded phase ignores this rule here.
        """
        if self.phase != "challenge":
            return True
        if self.total_profit <= 0:
            return True
        return self.best_day_pnl <= CONSISTENCY_RATIO * self.total_profit

    def reset_daily(self, session_date: date):
        """Reset daily PnL at new session. Equity/HWM/DD/best_day persist."""
        if self.last_reset_date != session_date:
            # Before resetting, roll current day's PnL into best_day tracker
            if self.daily_pnl > self.best_day_pnl:
                self.best_day_pnl = self.daily_pnl
            self.daily_pnl = 0.0
            self.last_reset_date = session_date

    def apply_trade(self, pnl: float) -> tuple[bool, bool]:
        """
        Apply a trade's PnL to this account. Returns (newly_blown, newly_passed).
        """
        if not self.available:
            return (False, False)

        self.equity += pnl
        self.daily_pnl += pnl
        self.trades_taken += 1

        # Track best day incrementally (current day could become best after this trade)
        if self.daily_pnl > self.best_day_pnl:
            self.best_day_pnl = self.daily_pnl

        newly_blown = False
        newly_passed = False

        # Update HWM + trailing DD floor (trails forever, never locks)
        if self.equity > self.hwm:
            self.hwm = self.equity
            new_floor = self.hwm - TRAILING_DD
            self.dd_floor = max(self.dd_floor, new_floor)

        # Check blown (trailing DD breach)
        if self.equity <= self.dd_floor:
            self.blown = True
            newly_blown = True
            return (newly_blown, newly_passed)

        # Check target hit
        if self.total_profit >= PROFIT_TARGET:
            if not self.target_hit:
                self.target_hit = True
            # Only flag as "passed" (stop trading) if consistency is satisfied
            if self.consistency_ok:
                self.passed = True
                newly_passed = True

        return (newly_blown, newly_passed)


class AccountManager:
    """
    Manages a list of accounts with round-robin rotation.

    Usage:
      mgr = AccountManager(["Sim101", "Sim102"])
      acct = mgr.next_account()       # pick account for upcoming trade
      ...trade executes & exits...
      mgr.on_trade_exit(acct, pnl=+120.0)  # update equity/status
    """

    def __init__(self, account_names: list[str], state_path: Optional[Path] = None):
        if not account_names:
            raise ValueError("AccountManager requires at least one account name")
        self.accounts: list[Account] = [Account(name=n) for n in account_names]
        self._rotation_idx: int = 0
        self.state_path: Optional[Path] = Path(state_path) if state_path else None
        if self.state_path is not None:
            self._load_state()

    @property
    def any_available(self) -> bool:
        return any(a.available for a in self.accounts)

    def get(self, name: str) -> Optional[Account]:
        for a in self.accounts:
            if a.name == name:
                return a
        return None

    def reset_daily_for_all(self, ts: pd.Timestamp):
        """Call at session reset (6pm ET) or at the start of each day."""
        ts_et = ts.tz_convert(ET) if ts.tz is not None else ts.tz_localize("UTC").tz_convert(ET)
        # Session date: if before 6pm ET, session_date = today; else session_date = next day
        if ts_et.hour < 18:
            session_date = ts_et.date()
        else:
            from datetime import timedelta
            session_date = (ts_et + timedelta(days=1)).date()
        changed = False
        for a in self.accounts:
            if a.last_reset_date != session_date:
                changed = True
            a.reset_daily(session_date)
        if changed:
            self._save_state()

    def next_account(self) -> Optional[Account]:
        """
        Pick the next available account in rotation.
        Returns None if all accounts are blown/passed.
        Advances rotation index AFTER picking.
        """
        n = len(self.accounts)
        for i in range(n):
            idx = (self._rotation_idx + i) % n
            acct = self.accounts[idx]
            if acct.available:
                # Advance rotation past this account for next call
                self._rotation_idx = (idx + 1) % n
                self._save_state()
                return acct
        return None

    def on_trade_exit(self, account: Account, pnl: float):
        """Apply trade PnL to the given account. Log status changes."""
        newly_blown, newly_passed = account.apply_trade(pnl)
        log.info(
            f"[acct] {account.name} trade #{account.trades_taken} pnl=${pnl:+.0f} "
            f"| equity=${account.equity:,.0f} hwm=${account.hwm:,.0f} "
            f"dd_floor=${account.dd_floor:,.0f} daily=${account.daily_pnl:+.0f}"
        )
        if newly_blown:
            log.warning(f"[acct] {account.name} BLOWN (equity ${account.equity:,.0f} <= DD floor ${account.dd_floor:,.0f})")
        if account.target_hit and not account.passed and not account.blown:
            ratio = account.best_day_pnl / account.total_profit if account.total_profit > 0 else 0
            log.info(
                f"[acct] {account.name} TARGET HIT but consistency PENDING "
                f"(best day ${account.best_day_pnl:,.0f} = {ratio*100:.1f}% of total ${account.total_profit:,.0f}; "
                f"need <= {CONSISTENCY_RATIO*100:.0f}%). Keep trading."
            )
        if newly_passed:
            log.info(f"[acct] {account.name} PASSED (equity ${account.equity:,.0f} >= target ${STARTING_EQUITY + PROFIT_TARGET:,.0f}, consistency OK)")

        if not self.any_available:
            log.warning("[acct] All accounts unavailable (blown/passed). Trading paused.")

        self._save_state()

    def status_summary(self) -> str:
        parts = []
        for a in self.accounts:
            tag = "BLOWN" if a.blown else ("PASSED" if a.passed else "OK")
            parts.append(f"{a.name}=${a.equity:,.0f}({tag})")
        return " | ".join(parts)

    # --- Persistence -------------------------------------------------------

    def _save_state(self):
        if self.state_path is None:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "rotation_idx": self._rotation_idx,
                "accounts": [
                    {
                        **asdict(a),
                        "last_reset_date": a.last_reset_date.isoformat() if a.last_reset_date else None,
                    }
                    for a in self.accounts
                ],
                "saved_at": datetime.now().isoformat(),
            }
            tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            tmp.replace(self.state_path)
        except Exception as e:
            log.warning(f"[acct] failed to save state: {e}")

    def _load_state(self):
        if self.state_path is None or not self.state_path.exists():
            log.info(f"[acct] no saved state at {self.state_path} (fresh start)")
            return
        try:
            with open(self.state_path) as f:
                payload = json.load(f)
            saved_names = [a["name"] for a in payload.get("accounts", [])]
            current_names = [a.name for a in self.accounts]
            if saved_names != current_names:
                log.warning(
                    f"[acct] saved state account list {saved_names} != current {current_names}; "
                    f"ignoring saved state (fresh start)"
                )
                return
            for a, saved in zip(self.accounts, payload["accounts"]):
                lrd = saved.get("last_reset_date")
                a.equity = saved.get("equity", a.equity)
                a.hwm = saved.get("hwm", a.hwm)
                a.dd_floor = saved.get("dd_floor", a.dd_floor)
                a.dd_locked = saved.get("dd_locked", a.dd_locked)
                a.daily_pnl = saved.get("daily_pnl", a.daily_pnl)
                a.best_day_pnl = saved.get("best_day_pnl", a.best_day_pnl)
                a.last_reset_date = date.fromisoformat(lrd) if lrd else None
                a.blown = saved.get("blown", a.blown)
                a.passed = saved.get("passed", a.passed)
                a.target_hit = saved.get("target_hit", a.target_hit)
                a.trades_taken = saved.get("trades_taken", a.trades_taken)
                a.phase = saved.get("phase", a.phase)
            self._rotation_idx = payload.get("rotation_idx", 0) % len(self.accounts)
            log.info(
                f"[acct] restored state from {self.state_path} | "
                f"rotation_idx={self._rotation_idx} | " + self.status_summary()
            )
        except Exception as e:
            log.warning(f"[acct] failed to load state ({e}); starting fresh")
