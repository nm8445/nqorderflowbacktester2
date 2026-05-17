"""Fabio ORB (Opening Range Breakout) live signal engine.

Implements the locked Fabio ORB v1 strategy per
`scripts/fabio_orb/LOCKED_CONFIG_fabio_orb.md`:

  - Long-only.
  - Opening range: bars whose close_time is in (08:30, 09:00] ET (6 bars).
  - Trade window: bar close_time in (09:00, 14:00] ET.
  - Entry: 4 consecutive 5-min closes above ORB_High AND entry bar's
    buy_vol - sell_vol >= 300 AND ORB_Low < close (sanity).
  - Skip entries whose bar closes at exactly 09:30 ET (structurally bad).
  - SL: static ORB_Low (never moved).
  - TP: entry + 4.0 * (entry - ORB_Low) (rarely hit ~0.3% of trades).
  - EOD: exit at close of first bar whose close is >= 14:00 ET.
  - If SL and TP both touched intrabar, SL fills first (conservative).

Backtest baseline (5.4 years 2020-12 → 2026-05, 1-contract NQ):
  709 trades, 53.7% WR, $157,965 net, PF 1.347, MaxDD -$20,240
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date, time
from enum import Enum
from typing import Callable, Optional

import pandas as pd

from live.combined.bar_builder import Bar
from live.combined.config import ET_TZ

# Locked params (LOCKED_CONFIG_fabio_orb.md)
ORB_START_HHMM = 830   # bars whose close > 08:30 included
ORB_END_HHMM   = 900   # ... and close <= 09:00
TRADE_END_HHMM = 1400  # entries allowed for closes <= 14:00 ET
SKIP_BUCKET_HHMM = 930
N_CONFIRM = 4
DELTA_THRESHOLD = 300   # buy_vol - sell_vol >= 300 contracts
TP_RR_RATIO = 4.0
EOD_TIME = time(14, 0)


class Direction(Enum):
    FLAT = 0
    LONG = 1


@dataclass
class FabioPosition:
    entry_price: float
    entry_time: pd.Timestamp
    sl_price: float            # ORB_Low — static
    tp_price: float            # entry + 4 * (entry - ORB_Low)
    qty: int = 1


@dataclass
class FabioSignal:
    event: str               # "ENTRY", "EXIT"
    direction: Direction
    price: float
    timestamp: pd.Timestamp
    reason: str = ""         # "SL", "TP", "EOD"
    qty: int = 1


class FabioORBEngine:
    """Stateful Fabio ORB engine. Long-only.

    Caller must:
      1. Feed 5-min bars chronologically via `on_5min_bar(bar)`.
      2. Bars must have buy_vol/sell_vol populated from a real trade-tape feed
         (Databento MBP-1 aggressor-classified). L2-depth aggregation will
         not produce correct delta.
    """

    def __init__(self):
        self.position: Optional[FabioPosition] = None
        self.subscribers: list[Callable[[FabioSignal], None]] = []

        # Per-day state — reset at first bar of a new session day
        self._current_date: Optional[_date] = None
        self._orb_high: Optional[float] = None
        self._orb_low: Optional[float] = None
        self._post_orb_bars: list[Bar] = []   # bars in (09:00, 14:00] today

        self.n_bars_seen = 0
        self.n_entries = 0
        self.n_exits = 0
        self.n_blocked_skip_bucket = 0
        self.n_blocked_no_confirm = 0
        self.n_blocked_delta = 0

    def subscribe(self, cb: Callable[[FabioSignal], None]) -> None:
        self.subscribers.append(cb)

    def emit(self, sig: FabioSignal) -> None:
        if sig.event == "ENTRY": self.n_entries += 1
        elif sig.event == "EXIT": self.n_exits += 1
        for cb in self.subscribers:
            try:
                cb(sig)
            except Exception as e:
                print(f"[fabio_orb_engine] subscriber error: {e}")

    def _hhmm(self, ts: pd.Timestamp) -> int:
        if ts.tzinfo is None:
            ts = ts.tz_localize(ET_TZ)
        else:
            ts = ts.tz_convert(ET_TZ)
        return ts.hour * 100 + ts.minute

    def _reset_day(self, d: _date) -> None:
        self._current_date = d
        self._orb_high = None
        self._orb_low = None
        self._post_orb_bars = []

    def on_5min_bar(self, bar: Bar) -> None:
        """Process a closed 5-min bar."""
        self.n_bars_seen += 1
        close_ts = bar.close_time
        if close_ts.tzinfo is None:
            close_ts = close_ts.tz_localize(ET_TZ)
        else:
            close_ts = close_ts.tz_convert(ET_TZ)
        bar_date = close_ts.date()
        hhmm = close_ts.hour * 100 + close_ts.minute

        # New session day → reset
        if bar_date != self._current_date:
            self._reset_day(bar_date)

        # === EXITS first ===
        if self.position is not None:
            # SL — intrabar low touched ORB_Low (or below)
            if bar.low <= self.position.sl_price:
                # SL fills at sl_price (or worse — assume fills at sl_price)
                self._do_exit(self.position.sl_price, close_ts, "SL")
            elif bar.high >= self.position.tp_price:
                # TP — intrabar high touched
                self._do_exit(self.position.tp_price, close_ts, "TP")
            elif close_ts.time() >= EOD_TIME:
                # EOD — first bar that closes at or after 14:00 ET
                self._do_exit(bar.close, close_ts, "EOD")

        # === ORB construction (08:35 → 09:00 closes) ===
        if ORB_START_HHMM < hhmm <= ORB_END_HHMM:
            if self._orb_high is None:
                self._orb_high = bar.high
                self._orb_low = bar.low
            else:
                self._orb_high = max(self._orb_high, bar.high)
                self._orb_low = min(self._orb_low, bar.low)

        # === Post-ORB tracking ===
        if ORB_END_HHMM < hhmm <= TRADE_END_HHMM:
            self._post_orb_bars.append(bar)

            # === Entry evaluation ===
            if self.position is None and self._orb_high is not None:
                self._try_entry(bar, hhmm, close_ts)

    def _try_entry(self, bar: Bar, hhmm: int, close_ts: pd.Timestamp) -> None:
        # 1. Skip 09:30 bucket
        if hhmm == SKIP_BUCKET_HHMM:
            self.n_blocked_skip_bucket += 1
            return

        # 2. N=4 consecutive closes above ORB_High
        if len(self._post_orb_bars) < N_CONFIRM:
            self.n_blocked_no_confirm += 1
            return
        confirm = self._post_orb_bars[-N_CONFIRM:]
        if not all(b.close > self._orb_high for b in confirm):
            return

        # 3. Delta filter on entry bar only
        delta = (bar.buy_vol or 0) - (bar.sell_vol or 0)
        if delta < DELTA_THRESHOLD:
            self.n_blocked_delta += 1
            return

        # 4. Sanity: ORB_Low < close
        if self._orb_low >= bar.close:
            return

        # ENTER LONG at bar.close
        entry_price = float(bar.close)
        sl_price = float(self._orb_low)
        risk = entry_price - sl_price
        tp_price = entry_price + TP_RR_RATIO * risk

        self.position = FabioPosition(
            entry_price=entry_price,
            entry_time=close_ts,
            sl_price=sl_price,
            tp_price=tp_price,
            qty=1,
        )
        self.emit(FabioSignal(event="ENTRY", direction=Direction.LONG,
                               price=entry_price, timestamp=close_ts,
                               reason="", qty=1))

    def _do_exit(self, price: float, ts: pd.Timestamp, reason: str) -> None:
        pos = self.position
        self.emit(FabioSignal(event="EXIT", direction=Direction.LONG, price=price,
                               timestamp=ts, reason=reason, qty=pos.qty))
        self.position = None
