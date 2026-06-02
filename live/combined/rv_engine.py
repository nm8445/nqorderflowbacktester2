"""RV live signal engine.

State machine that consumes 20-min bars + per-bar volume profile (level data)
and emits entry/exit signals per the locked config.

Position lifecycle:
  FLAT → (entry signal at bar close) → LONG/SHORT (with SL/TP/force-close armed)
  LONG/SHORT → (SL hit / TP hit / force-close at 14:45 ET) → FLAT

Entry conditions (ALL must hold on signal bar):
  1. Session window: 09:00-13:00 OR 14:00-14:45 ET (close time)
  2. z_vol > 2.00
  3. ATR > 0
  4. EMA filter: close > ema → LONG candidate, close < ema → SHORT candidate
  5. Orderflow window absorption: 8-tick contiguous window, |delta sum| >= 150
     in opposing half of the bar

Exit conditions (priority order):
  1. Stop: opposite-side touch in subsequent bar (LONG: low<=SL; SHORT: high>=SL)
  2. Target: same-side touch (LONG: high>=TP; SHORT: low<=TP)
  3. Force-close: 14:45 ET or later, exit at bar close

Conservative: if SL and TP both inside same bar's range, assume SL fills first.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from enum import Enum
from typing import Callable, Optional
import pandas as pd

from live.combined.bar_builder import Bar
from live.combined.rv_features import RVFeatures, RVConfig
from live.combined.rv_orderflow import windowed_absorption_check, WINDOW_N_TICKS, WINDOW_D


class Direction(Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1


class ExitReason(Enum):
    NONE = "none"
    STOP = "stop"
    TARGET = "target"
    FORCE_CLOSE = "force_close"


@dataclass
class Position:
    direction: Direction
    entry_price: float
    entry_time: pd.Timestamp
    stop_price: float
    target_price: float
    qty: int = 1


@dataclass
class Signal:
    """Emitted by engine on entry or exit event."""
    event: str            # "ENTRY", "EXIT"
    direction: Direction  # entry direction (or position's direction at exit)
    price: float          # fill price (bar close for entry / force-close exit; SL/TP level for stop/target)
    timestamp: pd.Timestamp
    reason: str = ""      # for exits: stop/target/force_close


# Session gate per locked config
SESSION_MORNING_START = time(9, 0)
SESSION_MORNING_END = time(13, 0)
SESSION_AFTERNOON_START = time(14, 0)
SESSION_AFTERNOON_END = time(14, 45)
FORCE_CLOSE_TIME = time(14, 45)
HIGH_Z = 2.00
ATR_SL_MULT = 2.0
ATR_TP_MULT = 2.0
# Max 20-min ATR(14) at entry (points). Since SL/TP = 2xATR, a blown-out ATR makes the $ bracket
# explode. Set to 150 to skip ONLY the genuine crash-vol outliers: over 5yr that's just 2 trades
# (2025-04-07 tariff crash ATR 280 / float $1,622, + 2025-04-04 ATR 168), net -$445 to remove, and it
# cuts RV's worst float $1,622 -> $738/MNQ. NOTE: the prior ATR_MAX=70 was based on a BUGGY ATR calc
# (entry times in ET compared vs UTC bar stamps undercounted ATR ~1.6x) -> it actually dropped 52
# net-PROFITABLE trades. Corrected (UTC-consistent) ATR shows tightening below 150 removes profit for
# zero extra float benefit. RV-only. See scripts/montecarlo/farm_firing_order.py + FUNDEDNEXT_ONESIDED.md.
ATR_MAX = 150.0


def in_session(t: time) -> bool:
    """True if t is within RV entry session window."""
    return ((SESSION_MORNING_START <= t < SESSION_MORNING_END) or
            (SESSION_AFTERNOON_START <= t < SESSION_AFTERNOON_END))


class RVEngine:
    """20-min RV signal engine. Stateful — feed bars in order."""

    def __init__(self,
                 cfg: Optional[RVConfig] = None,
                 high_z: float = HIGH_Z,
                 sl_mult: float = ATR_SL_MULT,
                 tp_mult: float = ATR_TP_MULT,
                 atr_max: float = ATR_MAX,
                 window_n_ticks: int = WINDOW_N_TICKS,
                 window_d: float = WINDOW_D):
        self.features = RVFeatures(cfg or RVConfig())
        self.high_z = high_z
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult
        self.atr_max = atr_max
        self.window_n_ticks = window_n_ticks
        self.window_d = window_d

        self.position: Optional[Position] = None
        self.subscribers: list[Callable[[Signal], None]] = []

        self.n_bars_seen = 0
        self.n_signals_emitted = 0
        self.n_entries = 0
        self.n_exits = 0

        # Last-seen bar's level volumes (set by external caller before on_bar)
        self._pending_level_volumes: Optional[dict[float, tuple[int, int]]] = None

    def subscribe(self, cb: Callable[[Signal], None]) -> None:
        self.subscribers.append(cb)

    def emit(self, sig: Signal) -> None:
        self.n_signals_emitted += 1
        if sig.event == "ENTRY":
            self.n_entries += 1
        elif sig.event == "EXIT":
            self.n_exits += 1
        for cb in self.subscribers:
            try:
                cb(sig)
            except Exception as e:
                import traceback
                print(f"[rv_engine] subscriber error: {e}")
                traceback.print_exc()

    def set_bar_levels(self, level_volumes: dict[float, tuple[int, int]]) -> None:
        """Optional override: caller can pre-set level_volumes for the next bar.
        For live use, prefer subscribing on_bar to the bar builder — it'll read
        level_volumes directly from bar.level_volumes (populated automatically
        as ticks come in)."""
        self._pending_level_volumes = level_volumes

    def on_bar(self, bar: Bar) -> None:
        """Process a closed 20-min bar. Will check exits first (against open position),
        then evaluate entry conditions if flat.

        Reads per-level orderflow from bar.level_volumes (auto-populated by
        BarBuilder during tick aggregation). Caller may override via
        set_bar_levels() before this call (used by the replay harness which
        pulls level data from the volumetric parquet)."""
        self.n_bars_seen += 1

        # Update rolling features FIRST (so z_vol/ema/atr reflect this bar)
        self.features.update(bar.open, bar.high, bar.low, bar.close)

        # If caller didn't override, read level data from the bar itself
        if self._pending_level_volumes is None and bar.level_volumes:
            self._pending_level_volumes = bar.level_volumes_tuples()

        # === Check exits FIRST on existing position ===
        if self.position is not None:
            self._check_exit(bar)

        # If exit fired and we're now flat, fall through to entry check (same bar can re-enter)
        # === Check entry conditions ===
        if self.position is None and self.features.ready:
            self._check_entry(bar)

        # Clear pending levels (caller must re-set for next bar)
        self._pending_level_volumes = None

    def _check_exit(self, bar: Bar) -> None:
        """Evaluate exit conditions against the current bar.
        Force-close has highest priority (15:00 closures get cleaned up cleanly)."""
        pos = self.position
        # Force-close window
        close_t = bar.close_time.time() if bar.close_time.tzinfo else bar.close_time.time()
        # bar.close_time is end of bar period; we check on time >= 14:45 ET
        if close_t >= FORCE_CLOSE_TIME:
            sig = Signal(event="EXIT", direction=pos.direction, price=bar.close,
                         timestamp=bar.close_time, reason="force_close")
            self.position = None
            self.emit(sig)
            return

        # Stop/target check
        if pos.direction == Direction.LONG:
            # SL first (conservative), then TP
            if bar.low <= pos.stop_price:
                sig = Signal(event="EXIT", direction=pos.direction, price=pos.stop_price,
                             timestamp=bar.close_time, reason="stop")
                self.position = None
                self.emit(sig)
                return
            if bar.high >= pos.target_price:
                sig = Signal(event="EXIT", direction=pos.direction, price=pos.target_price,
                             timestamp=bar.close_time, reason="target")
                self.position = None
                self.emit(sig)
                return
        else:  # SHORT
            if bar.high >= pos.stop_price:
                sig = Signal(event="EXIT", direction=pos.direction, price=pos.stop_price,
                             timestamp=bar.close_time, reason="stop")
                self.position = None
                self.emit(sig)
                return
            if bar.low <= pos.target_price:
                sig = Signal(event="EXIT", direction=pos.direction, price=pos.target_price,
                             timestamp=bar.close_time, reason="target")
                self.position = None
                self.emit(sig)
                return

    def _check_entry(self, bar: Bar) -> None:
        f = self.features

        # Session gate (use bar CLOSE time per backtest)
        close_t = bar.close_time.time()
        if not in_session(close_t):
            return

        # Volatility filter
        if f.z_vol is None or f.z_vol <= self.high_z:
            return
        if f.atr is None or f.atr <= 0:
            return
        # Max-ATR filter: skip crash-vol entries whose 2xATR bracket would blow up the $ risk.
        if f.atr > self.atr_max:
            return
        if f.ema is None or f.close is None:
            return

        # Direction by EMA
        if bar.close > f.ema:
            candidate = Direction.LONG
        elif bar.close < f.ema:
            candidate = Direction.SHORT
        else:
            return

        # Orderflow filter
        if self._pending_level_volumes is None:
            # No level data — skip per operational checks
            return
        long_ok, short_ok = windowed_absorption_check(
            bar.high, bar.low, self._pending_level_volumes,
            window_n_ticks=self.window_n_ticks, window_d=self.window_d,
        )

        if candidate == Direction.LONG and not long_ok:
            return
        if candidate == Direction.SHORT and not short_ok:
            return

        # Place the entry
        entry_price = bar.close
        sign = 1 if candidate == Direction.LONG else -1
        sl = entry_price - sign * self.sl_mult * f.atr
        tp = entry_price + sign * self.tp_mult * f.atr

        self.position = Position(
            direction=candidate,
            entry_price=entry_price,
            entry_time=bar.close_time,
            stop_price=sl,
            target_price=tp,
            qty=1,
        )
        sig = Signal(event="ENTRY", direction=candidate, price=entry_price,
                     timestamp=bar.close_time, reason="")
        self.emit(sig)
