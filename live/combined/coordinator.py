"""Phase 5: Strategy coordinator — enforces the no-hedge rule across all 4 strategies.

The coordinator sits between the strategy engines and the downstream consumers
(paper logger, NT8 executor). It listens to every engine's signals and decides
whether to forward them downstream.

No-hedge rule (matches scripts/rough vol orderflow/combined_4way.py:80):
  - Same direction overlap (both LONG, both SHORT)  → ALL signals allowed
  - Opposite direction overlap                       → SECOND signal BLOCKED

Example flow:
    Fabio fires LONG @ 09:35  → no other position open      → APPROVED
    RV    fires LONG @ 09:40  → Fabio LONG already open     → APPROVED (same dir)
    B2    fires SHORT @ 10:35 → Fabio + RV LONG open        → BLOCKED (no hedge)
    Fabio exits SL @ 10:50    → coordinator removes Fabio from open dirs
    RV    exits TP @ 11:15    → coordinator removes RV → no positions left
    B2    next signal SHORT   → APPROVED (no opposing direction now)

When a signal is blocked, the coordinator also resets the engine's position
field to None — otherwise the engine thinks it's in a trade and the next bar's
exit logic would fire a spurious EXIT signal.

Usage:
    coord = Coordinator()
    coord.register("RV", rv_engine, paper_cb, nt8_cb)
    coord.register("B2", b2_engine, paper_cb, nt8_cb)
    coord.register("OD", od_engine, paper_cb, nt8_cb)
    coord.register("FB", fb_engine, paper_cb, nt8_cb)
    # Coordinator subscribes to each engine internally.
    # Approved signals flow to paper_cb + nt8_cb. Blocked signals are dropped.

State persistence (light, not required for no-hedge):
  - `save_state()` writes current open_dir + per-engine position summary to JSON
  - `load_state()` restores on startup if file < 5 min old (skip stale state)
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from live.combined.config import STATE_DIR

COORD_STATE_PATH = STATE_DIR / "coordinator_state.json"
STATE_FRESH_SEC = 300   # only restore state if file < 5 min old


@dataclass
class Coordinator:
    """No-hedge enforcement + downstream signal routing for the 4-strat stack."""

    # strat_name -> "LONG" / "SHORT" of the currently-open position (if any)
    _open_dir: dict[str, str] = field(default_factory=dict)
    # strat_name -> engine instance (for rollback on block)
    _engines: dict = field(default_factory=dict)
    # strat_name -> list of downstream callbacks (paper logger, NT8 executor)
    _downstream: dict[str, list[Callable]] = field(default_factory=dict)

    n_approved: int = 0
    n_blocked: int = 0
    blocked_log: list[dict] = field(default_factory=list)

    def register(self, strat_name: str, engine, *downstream_callbacks: Callable) -> None:
        """Register one strategy. Coordinator subscribes to the engine and
        routes approved signals to all downstream callbacks."""
        self._engines[strat_name] = engine
        self._downstream[strat_name] = list(downstream_callbacks)

        def _handler(sig, _strat=strat_name):
            self._handle(_strat, sig)
        engine.subscribe(_handler)

    def _handle(self, strat_name: str, sig) -> None:
        """Inspect signal; decide approve vs block; forward or rollback."""
        if sig.event == "ENTRY":
            # Direction enum-or-string handling
            direction = sig.direction.name if hasattr(sig.direction, "name") else str(sig.direction)

            # Check every OTHER strategy's currently-open direction
            for other_strat, other_dir in self._open_dir.items():
                if other_strat == strat_name:
                    continue
                if other_dir != direction:
                    # No-hedge violation — BLOCK
                    self.n_blocked += 1
                    ts_str = (sig.timestamp.strftime("%Y-%m-%d %H:%M ET")
                              if hasattr(sig.timestamp, "strftime") else str(sig.timestamp))
                    print(f"  [COORD-BLOCKED] {strat_name} {direction} @ {sig.price:.2f}  "
                          f"({ts_str}) — opposing {other_strat}={other_dir}")
                    self.blocked_log.append({
                        "ts": ts_str, "strat": strat_name, "direction": direction,
                        "price": float(sig.price), "blocked_by": other_strat,
                        "blocked_by_dir": other_dir,
                    })
                    # Rollback engine state — engine emitted ENTRY but signal
                    # was blocked. Clear position so next bar's exit logic
                    # doesn't fire on the phantom position.
                    engine = self._engines.get(strat_name)
                    if engine is not None and hasattr(engine, "position"):
                        engine.position = None
                    return

            # No conflict → approve
            self._open_dir[strat_name] = direction
            self.n_approved += 1

        elif sig.event == "EXIT":
            # Clear tracking — strat is now flat
            self._open_dir.pop(strat_name, None)

        # Forward to downstream callbacks (paper logger, NT8 executor)
        for cb in self._downstream.get(strat_name, []):
            try:
                cb(sig)
            except Exception as e:
                print(f"  [coord] downstream callback error in {strat_name}: {e}")

    # ============== state persistence ==============

    def save_state(self) -> None:
        """Persist coordinator state to disk (lightweight — open directions
        per strat + counters). Engines manage their own position state via
        their own checkpointing if needed."""
        state = {
            "saved_at": dt.datetime.utcnow().isoformat() + "Z",
            "open_dir": dict(self._open_dir),
            "n_approved": self.n_approved,
            "n_blocked": self.n_blocked,
        }
        COORD_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = COORD_STATE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(COORD_STATE_PATH)

    def load_state(self) -> bool:
        """Restore coordinator state if file exists and is fresh.
        Returns True if loaded, False if skipped (missing or stale)."""
        if not COORD_STATE_PATH.exists():
            return False
        try:
            state = json.loads(COORD_STATE_PATH.read_text())
            saved = dt.datetime.fromisoformat(state["saved_at"].rstrip("Z"))
            age = (dt.datetime.utcnow() - saved).total_seconds()
            if age > STATE_FRESH_SEC:
                print(f"  [coord] state file too stale ({age:.0f}s old) — starting fresh")
                return False
            self._open_dir = state.get("open_dir", {})
            self.n_approved = int(state.get("n_approved", 0))
            self.n_blocked = int(state.get("n_blocked", 0))
            print(f"  [coord] restored state from {saved.isoformat()}Z  "
                  f"open_dir={self._open_dir}")
            return True
        except Exception as e:
            print(f"  [coord] failed to load state: {e}")
            return False

    def summary(self) -> dict:
        return {
            "approved": self.n_approved,
            "blocked": self.n_blocked,
            "currently_open": dict(self._open_dir),
            "blocked_log": list(self.blocked_log),
        }
