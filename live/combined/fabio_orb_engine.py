"""Fabio ORB (Opening Range Breakout) live signal engine.

Implements the locked Fabio ORB v1 strategy per
`scripts/fabio_orb/LOCKED_CONFIG_fabio_orb.md`:

  - Long-only.
  - Opening range: bars whose close_time is in (08:30, 09:00] ET (6 bars).
  - Trade window: bar close_time in (09:00, 14:00] ET.
  - Entry: 4 consecutive 5-min closes above ORB_High AND entry bar's
    buy_vol - sell_vol >= 300 AND ORB_Low < close (sanity).
  - Skip entries whose bar closes at exactly 09:30 ET (structurally bad).
  - SL: GIVEBACK TRAILING YELLOW (2026-07-09) — starts at ORB_Low (floor), ratchets up toward
    close - 1.5*ATR(14,5min), gives back after a bearish candle. Exit = bearish 5-min CLOSE <= yellow
    (engine-driven, reason SL_YELLOW; NOT a resting stop). Deploy-cleared: PF 1.40, MaxDD -24% vs static,
    combined 1-MNQ pass rate 67->73%. See project_fb_tail_risk_and_filters + scripts/fabio_orb/.
  - TP: entry + 4.0 * (entry - ORB_Low) — kept as a resting-limit backstop (never hit under the trail).
  - EOD: exit at close of first bar whose close is >= 14:00 ET.
  - Exit order per bar: TP (intrabar high) -> yellow (bearish close) -> EOD. Executors: NT8 rests the TP
    only + CLOSE_TAG on exit (cancels the resting TP); MT5 disaster-SL only + close on exit.

Backtest baseline (5.4 years 2020-12 → 2026-05, 1-contract NQ):
  709 trades, 53.7% WR, $157,965 net, PF 1.347, MaxDD -$20,240
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
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

# --- Giveback trailing-yellow stop (replaces the static ORB_Low stop; deploy-cleared config) ---
# The SL is a trailing "yellow" starting at ORB_Low (its hard FLOOR), ratcheting up toward
# close - k*ATR, and giving back a body-scaled fraction after a bearish 5-min candle. Exit is a
# BEARISH 5-min CLOSE <= yellow (engine-driven, like OD/B2 — NOT a resting stop). TP stays at 4R.
YELLOW_K          = 1.5     # raw_yellow = close - k*ATR
YELLOW_MODE       = "drift_floor"   # up-ratchet base (drift_floor / pure_ratchet)
YELLOW_DRIFT      = 0.0     # pts/bar (drift_floor only)
YELLOW_GIVEBACK   = 0.3     # fraction of (prev_yellow - raw_yellow) given back after a bearish candle
YELLOW_SCALE_BODY = True    # scale the retreat by min(1, prev bearish body / ATR)
YELLOW_MAX_GB     = 0.5     # cap on single-bar retreat, in ATR multiples
YELLOW_MIN_GAP    = 0.3     # floor: yellow never below raw_yellow + this*ATR (keeps stop hittable)
ATR_LEN           = 14      # 5-min ATR length (matches backtest; rolling, min 3 bars)


class Direction(Enum):
    FLAT = 0
    LONG = 1


@dataclass
class FabioPosition:
    entry_price: float
    entry_time: pd.Timestamp
    sl_price: float            # ORB_Low — the trailing-yellow FLOOR (also the gamble-sizing stop)
    tp_price: float            # entry + 4 * (entry - ORB_Low)
    qty: int = 1
    yellow: float = 0.0        # current trailing stop (init = ORB_Low); managed per 5-min bar
    prev_yellow: float = field(default=float("nan"))   # for ratchet/giveback


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
        # One trade per day max (matches backtest's run_day which breaks after
        # the first valid entry and returns after the exit).
        self._traded_today: bool = False
        # trailing-yellow support: per-session 5-min ATR + prev bar (TR + bearish/giveback)
        self._day_trs: list[float] = []
        self._atr_prev_close: Optional[float] = None
        self._prev_bar: Optional[Bar] = None

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
        self._traded_today = False
        self._day_trs = []
        self._atr_prev_close = None
        self._prev_bar = None

    def _current_atr(self) -> Optional[float]:
        """Rolling 5-min ATR over the session (matches backtest: rolling(ATR_LEN), min 3 TRs)."""
        if len(self._day_trs) < 3:
            return None
        w = self._day_trs[-ATR_LEN:]
        return sum(w) / len(w)

    def _update_yellow(self, bar: Bar, atr: float) -> None:
        """Ratchet the trailing yellow up / give it back on a post-bearish bar; floor at ORB_Low.
        Bar-close logic (all inputs known at this bar's close) — no lookahead. Matches run_giveback."""
        p = self.position
        raw_yellow = bar.close - YELLOW_K * atr
        prev = self._prev_bar
        prev_bearish = prev is not None and prev.close < prev.open
        gb_on = YELLOW_GIVEBACK > 0.0
        if gb_on and prev_bearish and not math.isnan(p.prev_yellow):
            gap = max(0.0, p.prev_yellow - raw_yellow)
            frac = YELLOW_GIVEBACK
            if YELLOW_SCALE_BODY:
                frac *= min(1.0, (prev.open - prev.close) / atr)
            cand = p.prev_yellow - min(gap * frac, YELLOW_MAX_GB * atr)
        elif YELLOW_MODE == "pure_ratchet":
            cand = max(p.prev_yellow, raw_yellow) if not math.isnan(p.prev_yellow) else raw_yellow
        else:  # drift_floor
            base = (p.prev_yellow + YELLOW_DRIFT) if not math.isnan(p.prev_yellow) else raw_yellow
            cand = max(base, raw_yellow)
        if gb_on:
            cand = max(cand, raw_yellow + YELLOW_MIN_GAP * atr)
        p.yellow = max(cand, p.sl_price)   # hard floor: never looser than ORB_Low

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

        # New session day: a still-open position means the prior session ended with NO 14:00 bar
        # (half-day / early close / data gap). Force-close it at the prior session's last bar
        # (matches the backtest's EOD_LAST) — FB must NEVER carry overnight.
        if bar_date != self._current_date:
            if self.position is not None and self._prev_bar is not None:
                self._do_exit(self._prev_bar.close, self._prev_bar.close_time, "EOD")
            self._reset_day(bar_date)

        # Only the RTH session window (08:35–14:00 ET) drives ATR / ORB / entry / stop mgmt,
        # matching the backtest's bar filter. Ignore out-of-session bars.
        if not (ORB_START_HHMM < hhmm <= TRADE_END_HHMM):
            return

        # --- rolling 5-min ATR (in-session; TR needs a prior in-session close) ---
        if self._atr_prev_close is not None:
            tr = max(bar.high - bar.low,
                     abs(bar.high - self._atr_prev_close), abs(bar.low - self._atr_prev_close))
            self._day_trs.append(tr)
        atr = self._current_atr()

        # === EXITS: trailing yellow (bearish CLOSE <= yellow), TP (intrabar high), EOD (14:00) ===
        #     — engine-driven exits (no resting stop); reasons match OD: SL_YELLOW / TP / EOD.
        if self.position is not None:
            if atr is not None and atr > 0:
                self._update_yellow(bar, atr)
            p = self.position
            if bar.high >= p.tp_price:
                self._do_exit(p.tp_price, close_ts, "TP")
            elif bar.close <= p.yellow and bar.close < bar.open:
                self._do_exit(bar.close, close_ts, "SL_YELLOW")
            elif close_ts.time() >= EOD_TIME:
                self._do_exit(bar.close, close_ts, "EOD")
            else:
                p.prev_yellow = p.yellow    # roll only if still open

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
            # One trade per day max — once `_traded_today` flips True, no more
            # entries until next session.
            if (self.position is None
                    and self._orb_high is not None
                    and not self._traded_today):
                self._try_entry(bar, hhmm, close_ts)

        # --- roll prev-bar state for the next in-session bar ---
        self._atr_prev_close = bar.close
        self._prev_bar = bar

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
            sl_price=sl_price,          # ORB_Low = the yellow's floor + the gamble-sizing stop
            tp_price=tp_price,
            qty=1,
            yellow=sl_price,            # trailing stop starts at ORB_Low
            prev_yellow=float("nan"),
        )
        # Lock out further entries for the rest of today's session
        self._traded_today = True
        self.emit(FabioSignal(event="ENTRY", direction=Direction.LONG,
                               price=entry_price, timestamp=close_ts,
                               reason="", qty=1))

    def _do_exit(self, price: float, ts: pd.Timestamp, reason: str) -> None:
        pos = self.position
        self.emit(FabioSignal(event="EXIT", direction=Direction.LONG, price=price,
                               timestamp=ts, reason=reason, qty=pos.qty))
        self.position = None
