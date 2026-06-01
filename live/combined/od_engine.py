"""Overnight Drift (OD) live signal engine.

Port of the locked overnight-drift strategy. Long NQ at 19:00 ET, force-close
at 08:00 ET, exits via yellow stop / green target / time-stop.

Locked config (ORIGINAL LIVE — reverted from upgrade 2026-05-25):
  - Entry: 19:00 ET long
  - Force-close: 08:00 ET
  - Yellow ATR len 14, mult 1.30, mode pure_ratchet (only moves up)
  - Yellow ACTIVE from bar 1 (no suppress)
  - Green: red_val + 82.5 - 1.50*bars_in_trade + 1.00*ATR(14)
  - Red:   entry_close + 0 + 0.45*bars_in_trade
  - BE rule: OFF
  - Martingale: ON (s1-L2)  base=1, loss_qty=2
      state 0 -> base
      state 1 (after a base-size loss) -> loss_qty (2)
      state 2 (after the recovery trade) -> base again regardless

REVERT history (2026-05-25):
- Initially upgraded to yellow_suppress=30 + wider green (160/2.0 + 1.4/1.5).
- Live OOS in 5/7-22 produced -$19,370 single-trade disaster on 5/14
  (yellow_suppress let a 2c martingale bleed 484 pts overnight before stopping).
- Wider-green improvements DEPENDED on yellow_suppress; without suppress,
  no green widening beat original on both IS+OOS.
- Reverted to original config for tail-risk protection. Net ~$209K (was theoretical
  $343K), worst trade -$10,870 (was -$19,370). Better prop-firm fit.
See live/od_green_sweep_top_configs.md for full sweep analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from enum import Enum
from typing import Callable, Optional

import numpy as np
import pandas as pd

from live.combined.bar_builder import Bar
from live.combined.config import ET_TZ

# ============ Locked params (REVERTED to original live 2026-05-25) ============
ENTRY_TIME           = time(19, 0)
FORCE_CLOSE_TIME     = time(8, 0)
ATR_LEN              = 14
YELLOW_MULT          = 1.30    # original live
GREEN_MULT           = 1.00    # original live
GREEN_BASE           = 82.5    # original live
GREEN_DECAY          = 1.50    # original live
YELLOW_SUPPRESS_BARS = 0       # 0 = no suppress (yellow active from bar 1)
RED_INTERCEPT        = 0.0
RED_DRIFT            = 0.45
USE_BE               = False
USE_MARTINGALE       = True
BASE_QTY             = 1
LOSS_QTY             = 2

# Toggle to disable mart entirely (per Phase 4 task description "marti OFF" — leave default ON)
ENABLE_MART = False   # marti OFF for prop accounts (floating-DD safety; +7.5pt pass rate)


class Direction(Enum):
    FLAT = 0
    LONG = 1


@dataclass
class ODPosition:
    entry_time: pd.Timestamp
    entry_price: float
    entry_bar_idx: int
    qty: int
    yellow_val: float
    prev_yellow: float = float("nan")
    prev_close: float = float("nan")


@dataclass
class ODSignal:
    event: str            # "ENTRY", "EXIT"
    direction: Direction
    price: float
    timestamp: pd.Timestamp
    reason: str = ""
    qty: int = 1


@dataclass
class _ATR:
    """Wilder/Pine RMA ATR — seed = SMA over first `length` true-range values."""
    length: int = ATR_LEN
    _tr_buf: list = field(default_factory=list)
    _atr: float = float("nan")
    _prev_close: float = float("nan")

    def update(self, high: float, low: float, close: float) -> float:
        if not np.isnan(self._prev_close):
            tr = max(high - low,
                     abs(high - self._prev_close),
                     abs(low  - self._prev_close))
        else:
            tr = high - low
        self._prev_close = close
        if np.isnan(self._atr):
            self._tr_buf.append(tr)
            if len(self._tr_buf) == self.length:
                self._atr = sum(self._tr_buf) / self.length
        else:
            alpha = 1.0 / self.length
            self._atr = (1 - alpha) * self._atr + alpha * tr
        return self._atr


class ODEngine:
    """Long-only at 19:00 ET, force-close 08:00 ET. Consumes 20-min bars."""

    def __init__(self):
        self.position: Optional[ODPosition] = None
        self.subscribers: list[Callable[[ODSignal], None]] = []
        self._atr_y = _ATR(length=ATR_LEN)
        self._atr_g = _ATR(length=ATR_LEN)   # locked: same length
        self._bar_idx = 0
        self._marti_state = 0    # 0=base, 1=loss_recovery, 2=cooldown
        self._next_qty = BASE_QTY

        self.n_bars_seen = 0
        self.n_entries = 0
        self.n_exits = 0

    def subscribe(self, cb: Callable[[ODSignal], None]) -> None:
        self.subscribers.append(cb)

    def emit(self, sig: ODSignal) -> None:
        if sig.event == "ENTRY": self.n_entries += 1
        elif sig.event == "EXIT": self.n_exits += 1
        for cb in self.subscribers:
            try:
                cb(sig)
            except Exception as e:
                print(f"[od_engine] subscriber error: {e}")

    def _bar_time_et(self, bar: Bar) -> pd.Timestamp:
        ts = bar.open_time
        if ts.tzinfo is None:
            ts = ts.tz_localize(ET_TZ)
        else:
            ts = ts.tz_convert(ET_TZ)
        return ts

    def on_20min_bar(self, bar: Bar) -> None:
        """Process one closed 20-min bar (open_time is bar OPEN, ET-aware)."""
        self.n_bars_seen += 1
        self._bar_idx += 1

        # Update ATRs every bar
        atr_y = self._atr_y.update(bar.high, bar.low, bar.close)
        atr_g = self._atr_g.update(bar.high, bar.low, bar.close)

        ts_et = self._bar_time_et(bar)
        bar_t = ts_et.time()

        # === EXIT logic (if in position) — runs FIRST so a position can exit then re-enter ===
        if self.position is not None:
            self._check_exit(bar, ts_et, atr_y, atr_g)

        # === ENTRY at 19:00 ET ===
        if self.position is None and bar_t == ENTRY_TIME and not np.isnan(atr_y):
            qty = self._compute_next_qty()
            self.position = ODPosition(
                entry_time=ts_et,
                entry_price=float(bar.close),
                entry_bar_idx=self._bar_idx,
                qty=qty,
                yellow_val=float(bar.close) - YELLOW_MULT * atr_y,
            )
            self.emit(ODSignal(event="ENTRY", direction=Direction.LONG,
                                price=float(bar.close), timestamp=ts_et,
                                reason="", qty=qty))

    def _compute_next_qty(self) -> int:
        if not (USE_MARTINGALE and ENABLE_MART):
            return BASE_QTY
        if self._marti_state == 0:
            return BASE_QTY
        elif self._marti_state == 1:
            return LOSS_QTY
        else:   # state 2 = cooldown
            return BASE_QTY

    def _update_marti_after_exit(self, exit_price: float) -> None:
        if not (USE_MARTINGALE and ENABLE_MART):
            return
        last_was_loss = (exit_price - self.position.entry_price) < 0
        if self._marti_state == 0:
            self._marti_state = 1 if last_was_loss else 0
        elif self._marti_state == 1:
            self._marti_state = 2   # forced cooldown next
        else:   # state 2
            self._marti_state = 1 if last_was_loss else 0

    def _check_exit(self, bar: Bar, ts_et: pd.Timestamp,
                     atr_y: float, atr_g: float) -> None:
        pos = self.position
        bars_in_trade = self._bar_idx - pos.entry_bar_idx

        # Yellow (pure_ratchet — only moves up)
        if not np.isnan(atr_y):
            raw_yellow = bar.close - YELLOW_MULT * atr_y
            if np.isnan(pos.prev_yellow):
                pos.yellow_val = raw_yellow
            else:
                pos.yellow_val = max(pos.prev_yellow, raw_yellow)

        # Red & Green
        red_val   = pos.entry_price + RED_INTERCEPT + RED_DRIFT * bars_in_trade
        green_val = (red_val + GREEN_BASE - GREEN_DECAY * bars_in_trade
                     + (GREEN_MULT * atr_g if not np.isnan(atr_g) else 0.0))

        bar_t = ts_et.time()
        exited = False
        # 1. TP Green — intrabar touch (Pine `process_orders_on_close=true` → fills at bar close)
        if not exited and bar.high >= green_val:
            self._do_exit(bar.close, ts_et, "TP_GREEN")
            exited = True
        # 2. SL Yellow — close at/below yellow AND bearish bar.
        # SUPPRESSED for first YELLOW_SUPPRESS_BARS bars (lets new trends breathe).
        if (not exited and not np.isnan(pos.yellow_val)
                and bar.close <= pos.yellow_val and bar.close < bar.open
                and bars_in_trade >= YELLOW_SUPPRESS_BARS):
            self._do_exit(bar.close, ts_et, "SL_YELLOW")
            exited = True
        # 3. Force close at 08:00 ET
        if not exited and bar_t == FORCE_CLOSE_TIME:
            self._do_exit(bar.close, ts_et, "FORCE_CLOSE")
            exited = True

        if not exited:
            # Roll state for next bar
            pos.prev_yellow = pos.yellow_val
            pos.prev_close  = bar.close

    def _do_exit(self, price: float, ts: pd.Timestamp, reason: str) -> None:
        pos = self.position
        self._update_marti_after_exit(price)
        self.emit(ODSignal(event="EXIT", direction=Direction.LONG, price=price,
                            timestamp=ts, reason=reason, qty=pos.qty))
        self.position = None
