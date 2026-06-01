"""B2 (Overnight Range Gamma) live signal engine.

Implements the locked B2 strategy:
  - Entry signals computed on 5-min bars from pre-computed `entry_signal_trades.parquet`
    (matches the backtest's pre-computed feature pipeline).
  - Exit logic on 20-min bars: yellow ratchet stop + fixed TP + MFE lock + force-close.
  - Filters: hours 9-14 ET only, drop POS-gamma SHORTs.

Locked params (from `live/overnight range gamma strat/best live config/config.md`):
  VARIANT="B2"   X=0.75   N=15   D=70   STRICT=True   BAND_K=0.25
  CONF_N=5       CONF_D=75
  YMULT=2.50     TPMULT=2.00
  MFE_K=0.8      MFE_LOCK=0.45
  FORCE_CLOSE=16:00 ET
  ALLOWED_HOURS=9..14 ET
  DROP_POS_SHORT=True   (skip SHORT entries when prior-day qqq_gamma_sign == +1)
  Martingale: OFF (always 1 contract)

IMPORTANT — current scope:
  This engine assumes pre-computed entry signals are AVAILABLE (replay or daily backfill).
  Live tick-to-signal computation is Phase 3b (port range_break_entry_signal_study.py).
  For now, replay validates the EXIT logic against backtest.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from enum import Enum
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import pandas as pd

from live.combined.bar_builder import Bar
from live.combined.config import ET_TZ

# Locked params
VARIANT = "B2"
X = 0.75
N_DELTA = 15
D_DELTA = 70
STRICT_SHORT = True
BAND_K = 0.25
CONF_N = 5
CONF_D = 75
YMULT = 2.50
TPMULT = 2.00
MFE_K = 0.8
MFE_LOCK = 0.45
FORCE_CLOSE_TIME = time(16, 0)
ALLOWED_HOURS = {9, 10, 11, 12, 13, 14}
DROP_POS_SHORT = True

# FC-only martingale: after a force-close LOSS, next trade size doubles to 2.
# After a size-2 trade (regardless of outcome), size resets to 1.
# Matches apply_mart_fc() in lock_v2_k08_lock045_mart_fc_filtered.py:177.
ENABLE_FC_MART = False   # marti OFF for prop accounts (floating-DD safety)


class Direction(Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass
class B2Position:
    direction: Direction
    entry_price: float
    entry_time: pd.Timestamp
    init_atr_y: float          # 20-min ATR at entry (for fixed green TP)
    yellow_val: float          # current trailing stop level (updates each bar w/ current ATR)
    green_val: float           # fixed TP (computed once at entry)
    mfe_so_far: float = 0.0
    qty: int = 1


@dataclass
class B2Signal:
    event: str            # "ENTRY", "EXIT"
    direction: Direction
    price: float
    timestamp: pd.Timestamp
    reason: str = ""
    qty: int = 1          # NQ contracts (FC-only mart: 1 or 2)


class B2Engine:
    """Stateful B2 engine.

    External caller must:
      1. Set the next entry candidate via `set_pending_entry(...)` BEFORE the bar it should
         fire on. Entries come from the pre-computed signal pipeline (replay) or a live
         signal builder (Phase 3b).
      2. Provide prior-day gamma_sign for the POS-SHORT filter via `set_gamma_sign(date, sign)`.
      3. Feed 20-min bars chronologically via `on_20min_bar(bar)`.
    """

    def __init__(self):
        self.position: Optional[B2Position] = None
        self.subscribers: list[Callable[[B2Signal], None]] = []

        # Pending entry queue: list of dicts with entry_time, direction, entry_price, atr_y
        self._pending_entries: list[dict] = []

        # Gamma sign by date (prior-day mapping for POS-SHORT filter)
        self._gamma_sign: dict = {}  # date -> int (-1, 0, +1)

        # Current bar's ATR (passed in via on_20min_bar)
        self._current_atr_y: float | None = None
        # Last known 20-min ATR — exposed for B2SignalEngine to read at conf
        # bar firing time (so yellow/green use the correct 20-min ATR, not
        # the 5-min ATR the signal engine computes locally).
        self._last_atr_20m: float | None = None
        # Last bar seen (for EOD exit pricing when day changes without force-close)
        self._last_bar: Optional[Bar] = None

        # FC-only martingale state: size for the NEXT trade to fire.
        # Starts at 1; bumps to 2 after a force-close loss; resets to 1 after a size-2 trade.
        self._next_size: int = 1

        self.n_bars_seen = 0
        self.n_entries = 0
        self.n_exits = 0
        self.n_blocked_pos_short = 0
        self.n_blocked_hour = 0
        self.n_blocked_busy = 0

    def subscribe(self, cb: Callable[[B2Signal], None]) -> None:
        self.subscribers.append(cb)

    def emit(self, sig: B2Signal) -> None:
        if sig.event == "ENTRY":
            self.n_entries += 1
        elif sig.event == "EXIT":
            self.n_exits += 1
        for cb in self.subscribers:
            try:
                cb(sig)
            except Exception as e:
                print(f"[b2_engine] subscriber error: {e}")

    def set_pending_entry(self, entry_time: pd.Timestamp, direction: str,
                          entry_price: float, atr_y: float) -> None:
        """Queue an entry to fire at its bar. Caller provides pre-validated signal."""
        self._pending_entries.append({
            "entry_time": entry_time,
            "direction": direction,
            "entry_price": entry_price,
            "atr_y": atr_y,
        })

    def set_gamma_sign(self, date, sign: int) -> None:
        self._gamma_sign[date] = sign

    def fire_entry_from_signal(self, direction: str, entry_price: float,
                                atr_y: float, ts: pd.Timestamp) -> bool:
        """Fire entry IMMEDIATELY (called by B2SignalEngine on conf bar close).

        Bypasses the pending-queue 20-min delay. Used when the live signal
        engine wants entry to fire at the 5-min boundary (within ~1 sec of
        signal confirmation) instead of waiting up to 20 min for the next
        20-min bar.

        Applies the SAME filters as _try_fire_pending: ALLOWED_HOURS check,
        DROP_POS_SHORT, ATR sanity.

        atr_y: should be 20-min ATR for parity with backtest's
        `init_atr_y = bars20["atr_y"].iloc[init_idx]`.

        Returns True if entry fired, False if blocked.
        """
        if self.position is not None:
            return False
        if ts.tzinfo is None:
            ts = ts.tz_localize(ET_TZ)

        # Filter: hours 9-14 ET only
        if ts.hour not in ALLOWED_HOURS:
            self.n_blocked_hour += 1
            print(f"  [B2-BLOCKED] {ts.strftime('%Y-%m-%d %H:%M ET')} {direction} @ "
                  f"{entry_price:.2f} blocked by hour filter (hour {ts.hour})")
            return False

        # Filter: POS-gamma SHORT block
        if DROP_POS_SHORT and direction == "SHORT":
            gs = self._gamma_sign.get(ts.date())
            if gs == 1:
                self.n_blocked_pos_short += 1
                print(f"  [B2-BLOCKED] {ts.strftime('%Y-%m-%d %H:%M ET')} SHORT @ "
                      f"{entry_price:.2f} blocked by POS gamma")
                return False

        if atr_y is None or atr_y <= 0 or atr_y != atr_y:   # nan check
            return False

        sign = 1 if direction == "LONG" else -1
        qty = self._next_size if ENABLE_FC_MART else 1
        yellow = entry_price - sign * YMULT * atr_y
        green = entry_price + sign * TPMULT * atr_y

        self.position = B2Position(
            direction=Direction.LONG if direction == "LONG" else Direction.SHORT,
            entry_price=entry_price,
            entry_time=ts,
            init_atr_y=atr_y,
            yellow_val=yellow,
            green_val=green,
            mfe_so_far=0.0,
            qty=qty,
        )
        self.emit(B2Signal(event="ENTRY", direction=self.position.direction,
                           price=entry_price, timestamp=ts, reason="", qty=qty))
        return True

    def on_20min_bar(self, bar: Bar, current_atr_y: float | None = None) -> None:
        """Process a 20-min bar close. Exit checks first, then candidate entries.

        current_atr_y: current bar's ATR(14) value. Required for yellow ratchet computation.
        If None, falls back to position.init_atr_y (degraded parity)."""
        self.n_bars_seen += 1
        self._current_atr_y = current_atr_y
        # Stash latest 20-min ATR for B2SignalEngine.fire_entry_from_signal()
        if current_atr_y is not None and current_atr_y == current_atr_y:   # not NaN
            self._last_atr_20m = current_atr_y

        # === EXITS ===
        if self.position is not None:
            self._check_exit(bar)

        # === ENTRIES ===
        if self.position is None:
            self._try_fire_pending(bar)

        # Track last bar for EOD safety pricing
        self._last_bar = bar

    def _try_fire_pending(self, bar: Bar) -> None:
        """Fire pending entry if any qualifies.

        Chained-dedupe semantics (matches backtest): a candidate fires on the
        bar containing its entry_time. If position is open at that bar, the
        candidate is DROPPED (not deferred) — same as the backtest's "take a
        trade only when no position is open" rule. Future-dated candidates are
        kept; past-dated candidates that couldn't fire are dropped."""
        bar_close_ts = bar.close_time
        to_fire = None
        kept = []
        for cand in self._pending_entries:
            ets = cand["entry_time"]
            if ets.tzinfo is None:
                ets = ets.tz_localize(ET_TZ)
            if ets > bar_close_ts:
                # Entry time still in the future — keep in queue
                kept.append(cand)
            elif self.position is None and to_fire is None:
                # Entry time has arrived AND we're flat — fire it
                to_fire = cand
            # else: entry time passed but we can't fire (position open, or
            #       already firing another candidate this bar) — DROP
        self._pending_entries = kept
        if to_fire is None:
            return

        # Filter: hours 9-14 ET only (use entry timestamp)
        ets = to_fire["entry_time"]
        if ets.tzinfo is None:
            ets = ets.tz_localize(ET_TZ)
        if ets.hour not in ALLOWED_HOURS:
            self.n_blocked_hour += 1
            print(f"  [B2-BLOCKED] {ets.strftime('%Y-%m-%d %H:%M ET')} {to_fire['direction']} @ "
                  f"{to_fire['entry_price']:.2f} blocked by hour filter "
                  f"(hour {ets.hour} not in {{9..14}})")
            return

        # Filter: POS-gamma SHORT block
        direction = to_fire["direction"]
        if DROP_POS_SHORT and direction == "SHORT":
            gs = self._gamma_sign.get(ets.date())
            if gs == 1:
                self.n_blocked_pos_short += 1
                print(f"  [B2-BLOCKED] {ets.strftime('%Y-%m-%d %H:%M ET')} SHORT @ "
                      f"{to_fire['entry_price']:.2f} blocked by POS gamma "
                      f"(qqq_gamma_sign=+1 on {ets.date()})")
                return

        # Place entry
        sign = 1 if direction == "LONG" else -1
        entry_price = float(to_fire["entry_price"])
        atr_y = float(to_fire["atr_y"])
        if atr_y <= 0 or np.isnan(atr_y):
            return

        # Parity guard: drop entry if THIS bar's date != entry date
        # (matches backtest's `end == start` early-return for missing-data days)
        if bar.open_time.date() != ets.date():
            return

        yellow = entry_price - sign * YMULT * atr_y
        green = entry_price + sign * TPMULT * atr_y

        # FC-only mart: take pending size, freeze on the position
        qty = self._next_size if ENABLE_FC_MART else 1
        self.position = B2Position(
            direction=Direction.LONG if direction == "LONG" else Direction.SHORT,
            entry_price=entry_price,
            entry_time=ets,
            init_atr_y=atr_y,
            yellow_val=yellow,
            green_val=green,
            mfe_so_far=0.0,
            qty=qty,
        )
        self.emit(B2Signal(event="ENTRY", direction=self.position.direction,
                           price=entry_price, timestamp=ets, reason="", qty=qty))

    def _check_exit(self, bar: Bar) -> None:
        """EXIT logic — verbatim port of simulate_exit_v2 from
        scripts/overnight range strat/scripts/lock_v2_k08_lock045_mart_fc_filtered.py:35.

        Loop body order (per backtest line 57-80):
          1. Compute cur_mfe, update mfe_so_far
          2. Update yellow_val from current ATR
          3. Compute stop_level (yellow OR MFE-locked)
          4. Check TP green (intrabar high/low touch)
          5. Check SL trail (close past stop + reversal candle)
          6. Check force-close (bar OPEN time >= 16:00)
        Day-change safety: if bar's date != entry date, EOD-exit at PRIOR bar close.
        (Backtest's `bars_idx[end].date() == ent_date` stop achieves the same.)
        """
        pos = self.position
        sign = 1 if pos.direction == Direction.LONG else -1

        # === Day-change safety: positions NEVER held overnight ===
        # If we see a bar from a new date, force-close didn't fire (data gap or
        # session break). Match backtest behavior: EOD-exit at LAST KNOWN BAR's
        # close (not next-day open which would include overnight gap).
        bar_date = bar.open_time.date() if bar.open_time.tzinfo else bar.open_time.date()
        entry_date = pos.entry_time.date() if pos.entry_time.tzinfo else pos.entry_time.date()
        if bar_date != entry_date:
            last = self._last_bar
            if last is not None:
                self._do_exit(last.close, last.close_time, "EOD")
            else:
                # Should never happen — fallback to bar open
                self._do_exit(bar.open, bar.close_time, "EOD")
            return

        # === Backtest-exact loop body ===
        # 1. MFE update
        cur_mfe = (bar.high - pos.entry_price) if sign > 0 else (pos.entry_price - bar.low)
        if cur_mfe > pos.mfe_so_far:
            pos.mfe_so_far = cur_mfe

        # 2. Yellow ratchet — uses CURRENT bar's ATR
        atr_for_yellow = self._current_atr_y if self._current_atr_y is not None else pos.init_atr_y
        if not (atr_for_yellow != atr_for_yellow):  # not NaN
            raw_yellow = bar.close - sign * YMULT * atr_for_yellow
            if sign > 0:
                pos.yellow_val = max(pos.yellow_val, raw_yellow)
            else:
                pos.yellow_val = min(pos.yellow_val, raw_yellow)

        # 3. Stop level (yellow vs MFE-lock)
        tp_distance = abs(pos.green_val - pos.entry_price)
        if pos.mfe_so_far >= MFE_K * tp_distance:
            mfe_stop = pos.entry_price + sign * MFE_LOCK * pos.mfe_so_far
            stop_level = max(pos.yellow_val, mfe_stop) if sign > 0 else min(pos.yellow_val, mfe_stop)
        else:
            stop_level = pos.yellow_val

        # 4. TP green (intrabar — exits at green_val price, not bar close)
        if sign > 0 and bar.high >= pos.green_val:
            self._do_exit(pos.green_val, bar.close_time, "TP_FIXED")
            return
        if sign < 0 and bar.low <= pos.green_val:
            self._do_exit(pos.green_val, bar.close_time, "TP_FIXED")
            return

        # 5. SL trail (close at/past stop + reversal candle)
        if sign > 0 and bar.close <= stop_level and bar.close < bar.open:
            self._do_exit(bar.close, bar.close_time, "SL_TRAIL")
            return
        if sign < 0 and bar.close >= stop_level and bar.close > bar.open:
            self._do_exit(bar.close, bar.close_time, "SL_TRAIL")
            return

        # 6. Force-close — backtest uses ts_arr[i].time() which is bar OPEN time
        bar_open_time = bar.open_time.time()
        if bar_open_time >= FORCE_CLOSE_TIME:
            self._do_exit(bar.close, bar.close_time, "FORCE_CLOSE")
            return

    def _do_exit(self, price: float, ts: pd.Timestamp, reason: str) -> None:
        pos = self.position
        sign = 1 if pos.direction == Direction.LONG else -1
        trade_pnl_pts = sign * (price - pos.entry_price)

        # FC-only mart state update — apply_mart_fc() in lock_v2 backtest:
        #   if just-exited size was 2 -> next = 1 (always reset)
        #   else (size was 1): if pnl<0 AND reason=FORCE_CLOSE -> next = 2, else 1
        if ENABLE_FC_MART:
            if pos.qty == 2:
                self._next_size = 1
            else:
                if trade_pnl_pts < 0 and reason == "FORCE_CLOSE":
                    self._next_size = 2
                else:
                    self._next_size = 1

        self.emit(B2Signal(event="EXIT", direction=pos.direction, price=price,
                           timestamp=ts, reason=reason, qty=pos.qty))
        self.position = None
