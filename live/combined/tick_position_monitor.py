"""Tick-level SL/TP monitor for MT5 protective closes.

PROBLEM this solves:
  RV/B2/Fabio have intrabar SL/TP levels. The Python engines only DETECT a
  hit on bar close (20-min for RV/B2, 5-min for Fabio). NT8 closes instantly
  because it has resting OCO bracket orders at the broker. But MT5 was set to
  disaster-SL-only (no resting TP), so MT5 relies on the engine's emitted
  close signal — which lags by up to a full bar (20 min for RV/B2).

  Result: when price touches RV's TP intrabar, NT8 fills immediately but the
  MT5 position stays open until the 20-min bar closes. During that gap the MT5
  position is unhedged/unmanaged.

SOLUTION:
  Watch every price tick. For each open position, read its CURRENT intended
  SL/TP from the engine's position object. The moment price crosses the level,
  immediately send an MT5 close for that strategy — matching NT8's intrabar
  fill timing.

SCOPE:
  - MT5 only. NT8 already has resting brackets; OD has no intrabar levels.
  - Does NOT change strategy logic. The Python engine still emits its own
    bar-close EXIT (keeps paper + NT8 + coordinator in sync). This monitor is
    a parallel MT5-protective layer. When the engine's bar-close EXIT later
    routes to MT5, the position is already closed → MT5 logs a graceful no-op.
  - Runs on the consumer thread (same thread that mutates engine.position),
    so reads are race-free.

WIRING (in run_phase1.py):
    monitor = TickPositionMonitor(engines={"RV":rv,"B2":b2,"FB":fb}, mt5x=mt5x)
    orig = multi.on_tick
    def wrapped(ts, price, size, side):
        orig(ts, price, size, side)
        monitor.on_tick(price)
    feed = DatabentoLiveFeed(on_tick=wrapped)
"""
from __future__ import annotations
from typing import Optional


def _direction_name(pos, default: str = "LONG") -> str:
    d = getattr(pos, "direction", None)
    if d is None:
        return default          # Fabio has no direction field — always LONG
    return d.name if hasattr(d, "name") else str(d)


def _levels_for(strat: str, pos):
    """Return (direction, stop, target) in NQ-price terms for the given strategy.
    Returns (None, None, None) if the strategy has no intrabar levels (OD)."""
    if strat == "RV":
        return _direction_name(pos), pos.stop_price, pos.target_price
    if strat == "B2":
        return _direction_name(pos), pos.yellow_val, pos.green_val
    if strat == "FB":
        return "LONG", pos.sl_price, pos.tp_price
    return None, None, None     # OD or unknown — no tick-level monitoring


class TickPositionMonitor:
    """Monitors open positions tick-by-tick; fires MT5 closes on SL/TP crossing.

    engines: dict of {strat_name: engine}. Only RV/B2/FB are monitored.
    mt5x: MT5Executor (or None — monitor becomes a no-op if absent).
    """

    # Strategies with intrabar SL/TP that MT5 needs help closing.
    MONITORED = ("RV", "B2", "FB")

    def __init__(self, engines: dict, mt5x=None):
        self.engines = engines
        self.mt5x = mt5x
        # Track which strategies we've already tick-closed this position-cycle,
        # so a single SL/TP touch doesn't spam multiple MT5 closes.
        self._tick_closed: set[str] = set()
        self.n_tick_closes = 0

    def on_tick(self, price: float) -> None:
        if self.mt5x is None:
            return
        for strat in self.MONITORED:
            engine = self.engines.get(strat)
            if engine is None:
                continue
            pos = getattr(engine, "position", None)

            # Position gone (engine closed it normally) → reset our flag so the
            # NEXT position on this strat can be monitored again.
            if pos is None:
                self._tick_closed.discard(strat)
                continue

            # Already tick-closed this position — wait for engine to go flat.
            if strat in self._tick_closed:
                continue

            direction, stop, target = _levels_for(strat, pos)
            if direction is None:
                continue

            hit_reason = None
            if direction == "LONG":
                if stop is not None and price <= stop:
                    hit_reason = "tick_SL"
                elif target is not None and price >= target:
                    hit_reason = "tick_TP"
            else:  # SHORT
                if stop is not None and price >= stop:
                    hit_reason = "tick_SL"
                elif target is not None and price <= target:
                    hit_reason = "tick_TP"

            if hit_reason is not None:
                self._fire_mt5_close(strat, hit_reason, price, stop, target, direction)

    def _fire_mt5_close(self, strat: str, reason: str, price: float,
                        stop: float, target: float, direction: str) -> None:
        lvl = stop if reason == "tick_SL" else target
        print(f"  [tick_monitor] {strat} {direction} {reason} @ tick={price:.2f} "
              f"(level={lvl:.2f}) — sending immediate MT5 close")
        try:
            self.mt5x._enqueue(("CLOSE", strat, reason))
            self._tick_closed.add(strat)
            self.n_tick_closes += 1
        except Exception as e:
            print(f"  [tick_monitor] {strat} MT5 close enqueue FAILED: {e}")
