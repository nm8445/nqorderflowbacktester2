"""
Live signal engine — Combined VWAP Strategy.

Two complementary entry modes, shared position lock (one trade at a time):

1. BASE (VWAP Reaction Continuation) — ADX 15-30
   - Absorption at VWAP zone (+/- 3 pts) with matching 5-min bias
   - SL 0.50x ATR, TP 1.90x ATR

2. TRENDING (Std Dev Band Reaction) — ADX 30+
   - When price trends above VWAP for 14 consecutive 5-min closes,
     look for longs at upper STD1 band (pullback to dynamic support)
   - When trending below VWAP, look for shorts at lower STD1 band
   - SL 1.00x ATR, TP 1.00x ATR

Common:
- Session VWAP anchored at 6pm ET, computed incrementally from ticks
- 5-min bias: last 5-min close vs VWAP determines long/short bias
- ADX 14-period on 5-min bars, Wilder smoothing
- Entry window: 7:10 PM ET to 4:00 PM ET (lunch break: no entries 12:00-12:59 PM ET)
- Force close at 4:58 PM ET
- One trade at a time, no breakeven, no trailing stop
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import logging
import math
from datetime import timedelta

import numpy as np
import pandas as pd

from nqbt.analysis.range_bars import RangeBar
from live.bar_builder import TimeBar

log = logging.getLogger(__name__)

TICK_SIZE = 0.25
ET = "America/New_York"

VWAP_ZONE_POINTS = 3.0   # +/- 3 points around VWAP (base strat)
# Asymmetric band zone: "above" = side away from VWAP, "below" = side toward VWAP.
#   Upper band (longs):  zone = [band - BAND_ZONE_BELOW, band + BAND_ZONE_ABOVE]
#   Lower band (shorts): zone = [band - BAND_ZONE_ABOVE, band + BAND_ZONE_BELOW]
BAND_ZONE_ABOVE = 3.0    # outside (past the band, away from VWAP)
BAND_ZONE_BELOW = 4.0    # inside  (toward VWAP) — extended to catch near-misses
APPROACH_LOOKBACK = 10    # bars to check for approach direction
MIN_DELTA = 30            # minimum absorption delta

# ADX filter ranges
ADX_PERIOD = 14
ADX_LOW = 15.0            # base strat lower bound
ADX_BASE_MAX = 30.0       # base strat upper bound
ADX_TREND_MIN = 25.0      # trending strat lower bound

# Trending band config
TRENDING_LOOKBACK = 14    # consecutive 5-min closes on one side of VWAP
TRENDING_STD_BAND = 1     # use STD1 bands


class Signal(Enum):
    NONE = "none"
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT_STOP = "exit_stop"
    EXIT_TARGET = "exit_target"


@dataclass
class Position:
    direction: int  # 1=long, -1=short
    entry_price: float
    stop_price: float
    target_price: float
    entry_time: pd.Timestamp
    source: str = "base"  # "base" or "trend"


@dataclass
class EngineOutput:
    signal: Signal
    position: Optional[Position]
    bar_close: float = 0.0


class LiveSignalEngine:
    """
    Stateful signal engine for combined VWAP + trending band trading.
    """

    def __init__(self, atr_period: int = 14,
                 stop_mult: float = 0.50, target_mult: float = 1.90,
                 trend_stop_mult: float = 1.00, trend_target_mult: float = 1.00):
        self.atr_period = atr_period
        # Base VWAP strat SL/TP
        self.stop_mult = stop_mult
        self.target_mult = target_mult
        # Trending band strat SL/TP
        self.trend_stop_mult = trend_stop_mult
        self.trend_target_mult = trend_target_mult

        self._bars: list[RangeBar] = []
        self._time_bars: list[TimeBar] = []
        # VWAP snapshot at each time bar's close — parallel to self._time_bars.
        # Used so trend detection compares each bar's close vs its CONCURRENT VWAP
        # (matching backtest's merge_asof behavior), not the current VWAP. Prevents
        # rising VWAP from falsely terminating a valid uptrend.
        self._bar_vwaps: list[Optional[float]] = []
        self._position: Optional[Position] = None

        # VWAP state — incremental computation
        self._cum_pv: float = 0.0      # cumulative (price * volume)
        self._cum_vol: float = 0.0     # cumulative volume
        self._cum_pv2: float = 0.0     # cumulative (price^2 * volume) for std dev
        self._session_start: Optional[pd.Timestamp] = None

        # 5-min bias tracking (for base strat)
        self._current_bias: Optional[str] = None  # "long" or "short"

        # Trending state tracking
        self._trend_direction: Optional[str] = None  # "up", "down", or None
        self._band_bias_paused: bool = False  # True if 5-min close broke below/above band zone
        # Approach latches: flip True when a 5-min bar closes AT/BEYOND the active band
        # during the current trend run. Reset when trend ends/flips. Option B gate —
        # replaces the old range-bar approach check so range-bar noise below the band
        # doesn't falsely disqualify a valid setup.
        self._approach_from_above: bool = False  # 5-min close >= upper band (for trend UP longs)
        self._approach_from_below: bool = False  # 5-min close <= lower band (for trend DOWN shorts)

        # ADX state — EWM recursion matching pandas ewm(alpha=1/N, adjust=False)
        self._adx_atr: float = 0.0
        self._adx_plus_dm: float = 0.0
        self._adx_minus_dm: float = 0.0
        self._adx_value: float = 0.0
        self._adx_tr_count: int = 0
        self._adx_ready: bool = False
        self._adx_prev_bar: Optional[TimeBar] = None

    @property
    def in_position(self) -> bool:
        return self._position is not None

    @property
    def adx(self) -> Optional[float]:
        if not self._adx_ready:
            return None
        return self._adx_value

    @property
    def vwap(self) -> Optional[float]:
        if self._cum_vol <= 0:
            return None
        return self._cum_pv / self._cum_vol

    @property
    def vwap_std(self) -> Optional[float]:
        """Current session VWAP standard deviation."""
        if self._cum_vol <= 0:
            return None
        vwap = self._cum_pv / self._cum_vol
        variance = (self._cum_pv2 / self._cum_vol) - vwap ** 2
        if variance < 0:
            variance = 0.0
        return math.sqrt(variance)

    @property
    def std1_upper(self) -> Optional[float]:
        v, s = self.vwap, self.vwap_std
        if v is None or s is None:
            return None
        return v + TRENDING_STD_BAND * s

    @property
    def std1_lower(self) -> Optional[float]:
        v, s = self.vwap, self.vwap_std
        if v is None or s is None:
            return None
        return v - TRENDING_STD_BAND * s

    def on_tick(self, price: float, size: int = 0, ts: Optional[pd.Timestamp] = None) -> EngineOutput:
        """
        Process every tick for:
        1. Stop/target exit checks when in position
        2. VWAP + std dev incremental update
        3. Session reset detection
        """
        if ts is not None:
            self._check_session_reset(ts)

        if size > 0:
            self._cum_pv += price * size
            self._cum_vol += size
            self._cum_pv2 += price * price * size

        if self._position is None:
            return EngineOutput(Signal.NONE, None)

        pos = self._position

        hit_stop = (pos.direction == 1 and price <= pos.stop_price) or \
                   (pos.direction == -1 and price >= pos.stop_price)
        hit_target = (pos.direction == 1 and price >= pos.target_price) or \
                     (pos.direction == -1 and price <= pos.target_price)

        if hit_stop:
            self._position = None
            return EngineOutput(Signal.EXIT_STOP, None, bar_close=pos.stop_price)
        elif hit_target:
            self._position = None
            return EngineOutput(Signal.EXIT_TARGET, None, bar_close=pos.target_price)

        return EngineOutput(Signal.NONE, self._position)

    def _check_session_reset(self, ts: pd.Timestamp):
        """Reset VWAP at 6pm ET (new session start)."""
        ts_et = ts.tz_convert(ET)
        if ts_et.hour < 18:
            session_date = ts_et.date()
            prev_day = session_date - timedelta(days=1)
        else:
            prev_day = ts_et.date()

        session_start_et = pd.Timestamp(f"{prev_day} 18:00:00", tz=ET).tz_convert("UTC")

        if self._session_start is None or session_start_et > self._session_start:
            self._session_start = session_start_et
            self._cum_pv = 0.0
            self._cum_vol = 0.0
            self._cum_pv2 = 0.0
            self._current_bias = None
            self._trend_direction = None
            self._band_bias_paused = False
            self._approach_from_above = False
            self._approach_from_below = False
            self._bars = []
            self._time_bars = []
            self._bar_vwaps = []
            log.info(f"[engine] New session started at {session_start_et.tz_convert(ET)} - VWAP reset (ADX carries over)")

    def on_time_bar_close(self, time_bar: TimeBar):
        """Process completed 5-minute time bar for ATR, ADX, bias, and trending state."""
        self._time_bars.append(time_bar)
        # Snapshot VWAP AT the moment this bar closes. Trend detection later compares
        # each historical bar's close against its own snapshot (not current VWAP).
        self._bar_vwaps.append(self.vwap)

        vwap = self.vwap
        if vwap is not None:
            # Update 5-min bias (for base strat) — only flip when close is
            # outside the VWAP zone (+/- 3 pts). Inside zone = hold previous bias.
            zone_high = vwap + VWAP_ZONE_POINTS
            zone_low = vwap - VWAP_ZONE_POINTS
            if time_bar.close > zone_high:
                old_bias = self._current_bias
                self._current_bias = "long"
                if old_bias != "long":
                    log.info(f"[engine] Bias flipped to LONG (5m close {time_bar.close:.2f} > zone high {zone_high:.2f})")
            elif time_bar.close < zone_low:
                old_bias = self._current_bias
                self._current_bias = "short"
                if old_bias != "short":
                    log.info(f"[engine] Bias flipped to SHORT (5m close {time_bar.close:.2f} < zone low {zone_low:.2f})")

            # Update trending state: N consecutive 5-min closes on one side of VWAP
            self._update_trending_state(vwap)

        # Update ADX from this 5-min bar
        self._update_adx(time_bar)

        atr = self._calculate_atr()
        adx_str = f" | ADX: {self._adx_value:.1f}" if self._adx_ready else ""
        trend_str = f" | Trend: {self._trend_direction or 'none'}" if self._trend_direction else ""
        band_str = ""
        if self._trend_direction and self.std1_upper is not None:
            band_str = f" | STD1: {self.std1_lower:.2f}/{self.std1_upper:.2f}"
        if atr is not None:
            if vwap:
                log.info(f"[engine] Time bar #{len(self._time_bars)} closed | ATR: {atr:.2f}{adx_str} | Bias: {self._current_bias} | VWAP: {vwap:.2f}{trend_str}{band_str}")
            else:
                log.info(f"[engine] Time bar #{len(self._time_bars)} closed | ATR: {atr:.2f}{adx_str}")
        else:
            log.info(f"[engine] Time bar #{len(self._time_bars)} closed - warming up ATR ({len(self._time_bars)}/{self.atr_period + 1} needed)")

    def _update_trending_state(self, vwap: float):
        """
        Check if last N 5-min closes are all above or all below VWAP.
        Each bar is compared against its CONCURRENT VWAP (snapshot at that bar's
        close), not the current VWAP. Matches backtest's merge_asof behavior:
        a rising VWAP should not retroactively "swallow" an older bar that was
        legitimately above VWAP at the time it closed.
        Also update band bias (pause/resume based on 5-min close vs band zone).
        """
        if len(self._time_bars) < TRENDING_LOOKBACK:
            self._trend_direction = None
            return

        recent = self._time_bars[-TRENDING_LOOKBACK:]
        recent_vwaps = self._bar_vwaps[-TRENDING_LOOKBACK:]
        # Require all snapshots present; if any is None (shouldn't happen after warmup), bail
        if any(v is None for v in recent_vwaps):
            self._trend_direction = None
            return
        all_above = all(tb.close > v for tb, v in zip(recent, recent_vwaps))
        all_below = all(tb.close < v for tb, v in zip(recent, recent_vwaps))

        old_trend = self._trend_direction
        if all_above:
            self._trend_direction = "up"
        elif all_below:
            self._trend_direction = "down"
        else:
            self._trend_direction = None

        if self._trend_direction != old_trend:
            if self._trend_direction:
                log.info(f"[engine] Trend detected: {self._trend_direction.upper()} ({TRENDING_LOOKBACK} consecutive 5m closes {'above' if self._trend_direction == 'up' else 'below'} VWAP)")
            else:
                log.info(f"[engine] Trend ended (was {old_trend})")
            # Trend changed: reset approach latches. Each new trend run must re-establish
            # that 5-min price actually tested the band.
            self._approach_from_above = False
            self._approach_from_below = False

        # Latch approach flags: once a 5-min bar closes at/beyond the active band
        # during this trend run, the "came from above/below" condition is satisfied
        # for the remainder of this trend run.
        last_close = self._time_bars[-1].close
        if self._trend_direction == "up":
            upper = self.std1_upper
            if upper is not None and last_close >= upper and not self._approach_from_above:
                self._approach_from_above = True
                log.info(f"[engine] Approach from above LATCHED: 5m close {last_close:.2f} >= upper band {upper:.2f}")
        elif self._trend_direction == "down":
            lower = self.std1_lower
            if lower is not None and last_close <= lower and not self._approach_from_below:
                self._approach_from_below = True
                log.info(f"[engine] Approach from below LATCHED: 5m close {last_close:.2f} <= lower band {lower:.2f}")

        # Update band bias: pause only when 5m close exits the asymmetric band zone
        # on the inside (VWAP side). Upper-band longs pause when close < band - 4.
        # Lower-band shorts pause when close > band + 4.
        if self._trend_direction == "up":
            band = self.std1_upper
            if band is not None:
                zone_low = band - BAND_ZONE_BELOW
                last_close = self._time_bars[-1].close
                if last_close < zone_low:
                    if not self._band_bias_paused:
                        self._band_bias_paused = True
                        log.info(f"[engine] Band bias PAUSED: 5m close {last_close:.2f} below band zone low {zone_low:.2f}")
                elif last_close > band:
                    if self._band_bias_paused:
                        self._band_bias_paused = False
                        log.info(f"[engine] Band bias RESUMED: 5m close {last_close:.2f} above band {band:.2f}")
        elif self._trend_direction == "down":
            band = self.std1_lower
            if band is not None:
                zone_high = band + BAND_ZONE_BELOW
                last_close = self._time_bars[-1].close
                if last_close > zone_high:
                    if not self._band_bias_paused:
                        self._band_bias_paused = True
                        log.info(f"[engine] Band bias PAUSED: 5m close {last_close:.2f} above band zone high {zone_high:.2f}")
                elif last_close < band:
                    if self._band_bias_paused:
                        self._band_bias_paused = False
                        log.info(f"[engine] Band bias RESUMED: 5m close {last_close:.2f} below band {band:.2f}")
        else:
            self._band_bias_paused = False

    def on_bar_close(self, bar: RangeBar) -> EngineOutput:
        """Process completed range bar, check for entry signals."""
        self._bars.append(bar)

        atr = self._calculate_atr()
        if atr is None:
            return EngineOutput(Signal.NONE, None)

        # Force close at 4:58 PM ET
        if self.in_position and bar.close_time is not None:
            bar_time_et = bar.close_time.tz_convert(ET)
            if bar_time_et.strftime("%H:%M") >= "16:58":
                log.info(f"[engine] FORCE CLOSE @ 4:58 PM ET")
                self._position = None
                return EngineOutput(Signal.EXIT_STOP, None, bar_close=bar.close)

        # No new entries while in position
        if self.in_position:
            return EngineOutput(Signal.NONE, self._position)

        # Check base VWAP signal first (ADX 15-30), then trending band signal (ADX 30+)
        signal = self._check_base_entry_signal(bar, atr)
        if signal:
            return signal

        signal = self._check_trending_entry_signal(bar, atr)
        if signal:
            return signal

        return EngineOutput(Signal.NONE, None, bar_close=bar.close)

    def _calculate_atr(self) -> Optional[float]:
        """Calculate ATR from completed 5-minute time bars."""
        if len(self._time_bars) < self.atr_period + 1:
            return None

        time_bars = self._time_bars[-(self.atr_period + 1):]
        true_ranges = []

        for i in range(1, len(time_bars)):
            high = time_bars[i].high
            low = time_bars[i].low
            prev_close = time_bars[i - 1].close

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        return sum(true_ranges) / len(true_ranges)

    def _update_adx(self, time_bar: TimeBar):
        """
        Incrementally update ADX matching backtest exactly.

        Backtest uses pandas ewm(alpha=1/period, min_periods=period, adjust=False):
        - Recursion starts from first value: y[0] = x[0], y[t] = (1-a)*y[t-1] + a*x[t]
        - Output only valid after min_periods (14) values seen
        - Applied twice: once for TR/DM smoothing, once for DX->ADX smoothing

        ADX carries over across sessions (not reset at 6pm), matching backtest behavior
        where the ADX lookup table spans all dates.
        """
        prev_bar = self._adx_prev_bar
        self._adx_prev_bar = time_bar

        if prev_bar is None:
            return

        alpha = 1.0 / ADX_PERIOD
        self._adx_tr_count += 1

        # True Range
        tr = max(
            time_bar.high - time_bar.low,
            abs(time_bar.high - prev_bar.close),
            abs(time_bar.low - prev_bar.close),
        )

        # Directional Movement
        up_move = time_bar.high - prev_bar.high
        down_move = prev_bar.low - time_bar.low
        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0

        # EWM recursion for TR, +DM, -DM (matches pandas adjust=False)
        if self._adx_tr_count == 1:
            self._adx_atr = tr
            self._adx_plus_dm = plus_dm
            self._adx_minus_dm = minus_dm
        else:
            self._adx_atr = self._adx_atr * (1 - alpha) + tr * alpha
            self._adx_plus_dm = self._adx_plus_dm * (1 - alpha) + plus_dm * alpha
            self._adx_minus_dm = self._adx_minus_dm * (1 - alpha) + minus_dm * alpha

        # DI/DX only valid after min_periods (14) TR values
        if self._adx_tr_count < ADX_PERIOD:
            return

        # Compute DX
        plus_di = 100.0 * self._adx_plus_dm / self._adx_atr if self._adx_atr > 0 else 0.0
        minus_di = 100.0 * self._adx_minus_dm / self._adx_atr if self._adx_atr > 0 else 0.0
        di_sum = plus_di + minus_di
        dx = 100.0 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0.0

        # Count DX values seen
        dx_count = self._adx_tr_count - ADX_PERIOD + 1

        # EWM recursion for ADX (matches pandas adjust=False on DX series)
        if dx_count == 1:
            self._adx_value = dx
        else:
            self._adx_value = self._adx_value * (1 - alpha) + dx * alpha

        # ADX valid after min_periods (14) DX values
        if dx_count >= ADX_PERIOD and not self._adx_ready:
            self._adx_ready = True
            log.info(f"[engine] ADX ready: {self._adx_value:.1f} (after {self._adx_tr_count} TR values)")
        elif self._adx_ready:
            log.debug(f"[engine] ADX updated: {self._adx_value:.1f}")

    def _passes_adx_filter_base(self) -> bool:
        """Check if current ADX is in the 15-30 range (base strat)."""
        if not self._adx_ready:
            return False
        return ADX_LOW <= self._adx_value < ADX_BASE_MAX

    def _passes_adx_filter_trending(self) -> bool:
        """Check if current ADX is 25+ (trending strat)."""
        if not self._adx_ready:
            return False
        return self._adx_value >= ADX_TREND_MIN

    def _has_absorption(self, bar: RangeBar) -> dict:
        """Check for absorption on a bar (matches backtest exactly)."""
        closed_bearish = bar.close < bar.open
        closed_bullish = bar.close > bar.open

        bearish_levels = []
        bullish_levels = []

        for price, lv in bar.levels.items():
            delta = lv.delta
            if closed_bearish and delta >= MIN_DELTA:
                bearish_levels.append((price, delta))
            elif closed_bullish and delta <= -MIN_DELTA:
                bullish_levels.append((price, delta))

        return {
            "bearish": closed_bearish and len(bearish_levels) > 0,
            "bullish": closed_bullish and len(bullish_levels) > 0,
        }

    def _check_base_entry_signal(self, confirm_bar: RangeBar, atr: float) -> Optional[EngineOutput]:
        """
        Check for base VWAP reaction continuation entry (ADX 15-30).

        Logic (matches backtest exactly):
        1. Need 2 consecutive bars (signal + confirm)
        2. Both must touch VWAP zone (+/- 3 pts)
        3. Signal bar has absorption
        4. Confirm bar closes same direction
        5. 5-min bias matches direction
        6. Price approached from trend side (pullback for longs, push for shorts)
        7. Entry window: 7:10pm ET to 4:00 PM ET
        8. ADX must be in 15-30 range
        """
        if len(self._bars) < 2:
            return None

        vwap = self.vwap
        if vwap is None:
            return None

        signal_bar = self._bars[-2]
        if not signal_bar.closed or not confirm_bar.closed:
            return None

        absorption = self._has_absorption(signal_bar)
        if not absorption["bearish"] and not absorption["bullish"]:
            return None

        vwap_zone_high = vwap + VWAP_ZONE_POINTS
        vwap_zone_low = vwap - VWAP_ZONE_POINTS

        if not (signal_bar.high >= vwap_zone_low and signal_bar.low <= vwap_zone_high):
            return None
        if not (confirm_bar.high >= vwap_zone_low and confirm_bar.low <= vwap_zone_high):
            return None

        abs_type = "bullish" if absorption["bullish"] else "bearish"
        bar_time_et = confirm_bar.close_time.tz_convert(ET)
        hour_min = bar_time_et.strftime("%H:%M")
        adx_val = self._adx_value if self._adx_ready else None
        adx_str = f"{adx_val:.1f}" if adx_val is not None else "N/A"

        confirm_bullish = confirm_bar.close > confirm_bar.open
        confirm_bearish = confirm_bar.close < confirm_bar.open
        confirm_dir = "bullish" if confirm_bullish else ("bearish" if confirm_bearish else "doji")

        lookback = min(APPROACH_LOOKBACK, len(self._bars) - 2)
        start_idx = len(self._bars) - 2 - lookback
        approached_from_above = any(
            self._bars[j].close > vwap_zone_high
            for j in range(start_idx, len(self._bars) - 2)
            if self._bars[j].closed
        )
        approached_from_below = any(
            self._bars[j].close < vwap_zone_low
            for j in range(start_idx, len(self._bars) - 2)
            if self._bars[j].closed
        )

        debug_ctx = (
            f"absorption={abs_type} confirm={confirm_dir} bias={self._current_bias} "
            f"ADX={adx_str} approach_above={approached_from_above} approach_below={approached_from_below} "
            f"VWAP={vwap:.2f} price={confirm_bar.close:.2f}"
        )

        # Time filter
        if "16:00" <= hour_min < "19:10":
            log.info(f"[debug] BASE signal @ {hour_min} ET REJECTED: outside entry window | {debug_ctx}")
            return None

        # Lunch break filter — no entries 12:00-12:59 ET
        if "12:00" <= hour_min < "13:00":
            log.info(f"[debug] BASE signal @ {hour_min} ET REJECTED: lunch break | {debug_ctx}")
            return None

        if self._current_bias is None:
            log.info(f"[debug] BASE signal @ {hour_min} ET REJECTED: no bias set | {debug_ctx}")
            return None

        # ADX filter: 15-30
        if not self._passes_adx_filter_base():
            if adx_val is not None:
                reason = f"ADX {adx_val:.1f} outside 15-30"
            else:
                reason = "ADX not ready"
            log.info(f"[debug] BASE signal @ {hour_min} ET REJECTED: {reason} | {debug_ctx}")
            return None

        entry_type = None
        reject_reasons = []

        if absorption["bullish"] and confirm_bullish:
            if self._current_bias != "long":
                reject_reasons.append(f"bias={self._current_bias} (need long)")
            if not approached_from_above:
                reject_reasons.append("no approach from above")
            if self._current_bias == "long" and approached_from_above:
                entry_type = "BUY"

        if absorption["bearish"] and confirm_bearish:
            if self._current_bias != "short":
                reject_reasons.append(f"bias={self._current_bias} (need short)")
            if not approached_from_below:
                reject_reasons.append("no approach from below")
            if self._current_bias == "short" and approached_from_below:
                entry_type = "SELL"

        if entry_type is None:
            reasons = ", ".join(reject_reasons) if reject_reasons else "direction mismatch"
            log.info(f"[debug] BASE signal @ {hour_min} ET REJECTED: {reasons} | {debug_ctx}")
            return None

        entry_price = confirm_bar.close
        if entry_type == "BUY":
            stop_price = entry_price - (atr * self.stop_mult)
            target_price = entry_price + (atr * self.target_mult)
        else:
            stop_price = entry_price + (atr * self.stop_mult)
            target_price = entry_price - (atr * self.target_mult)

        stop_price = round(stop_price * 4) / 4
        target_price = round(target_price * 4) / 4

        self._position = Position(
            direction=1 if entry_type == "BUY" else -1,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            entry_time=confirm_bar.close_time,
            source="base",
        )

        signal = Signal.ENTER_LONG if entry_type == "BUY" else Signal.ENTER_SHORT
        log.info(
            f"[engine] BASE ENTRY: {entry_type} @ {entry_price:.2f} | "
            f"Stop={stop_price:.2f} Target={target_price:.2f} | "
            f"ATR={atr:.2f} VWAP={vwap:.2f} Bias={self._current_bias} ADX={adx_str}"
        )

        return EngineOutput(signal, self._position, bar_close=entry_price)

    def _check_trending_entry_signal(self, confirm_bar: RangeBar, atr: float) -> Optional[EngineOutput]:
        """
        Check for trending band entry (ADX 30+).

        Logic (matches backtest):
        1. Must be in a trending state (14 consecutive 5-min closes above/below VWAP)
        2. ADX must be 30+
        3. Trend UP: look for longs at upper STD1 band (pullback to dynamic support)
           - Signal bar touches band zone (+/- 3 pts), bullish absorption, bullish confirm
           - 5-min close must not be below band zone (band bias not paused)
           - Prior bars must show approach from above the band zone
        4. Trend DOWN: look for shorts at lower STD1 band (push-up to dynamic resistance)
           - Signal bar touches band zone, bearish absorption, bearish confirm
           - 5-min close must not be above band zone (band bias not paused)
           - Prior bars must show approach from below the band zone
        """
        if len(self._bars) < 2:
            return None

        # Must be trending
        if self._trend_direction is None:
            return None

        # ADX must be 30+
        if not self._passes_adx_filter_trending():
            return None

        vwap = self.vwap
        if vwap is None:
            return None

        # Band bias must not be paused
        if self._band_bias_paused:
            return None

        signal_bar = self._bars[-2]
        if not signal_bar.closed or not confirm_bar.closed:
            return None

        absorption = self._has_absorption(signal_bar)
        if not absorption["bearish"] and not absorption["bullish"]:
            return None

        bar_time_et = confirm_bar.close_time.tz_convert(ET)
        hour_min = bar_time_et.strftime("%H:%M")

        adx_val = self._adx_value if self._adx_ready else None
        adx_str = f"{adx_val:.1f}" if adx_val is not None else "N/A"
        abs_type = "bullish" if absorption["bullish"] else "bearish"
        confirm_bullish = confirm_bar.close > confirm_bar.open
        confirm_bearish = confirm_bar.close < confirm_bar.open
        confirm_dir = "bullish" if confirm_bullish else ("bearish" if confirm_bearish else "doji")

        # Build diagnostic context shared across all rejection logs below
        def _trend_ctx(band, zone_low, zone_high):
            return (
                f"trend={self._trend_direction} absorption={abs_type} confirm={confirm_dir} "
                f"ADX={adx_str} band={band:.2f} zone={zone_low:.2f}-{zone_high:.2f} "
                f"signal_bar=[{signal_bar.low:.2f}-{signal_bar.high:.2f} c={signal_bar.close:.2f}] "
                f"confirm_bar=[{confirm_bar.low:.2f}-{confirm_bar.high:.2f} c={confirm_bar.close:.2f}] "
                f"VWAP={vwap:.2f} paused={self._band_bias_paused}"
            )

        # Time filter (same as base)
        if "16:00" <= hour_min < "19:10":
            log.info(f"[debug] TREND signal @ {hour_min} ET REJECTED: outside entry window | trend={self._trend_direction} absorption={abs_type} confirm={confirm_dir} ADX={adx_str}")
            return None

        # Lunch break filter — no entries 12:00-12:59 ET
        if "12:00" <= hour_min < "13:00":
            log.info(f"[debug] TREND signal @ {hour_min} ET REJECTED: lunch break | trend={self._trend_direction} absorption={abs_type} confirm={confirm_dir} ADX={adx_str}")
            return None

        # Band bias check (logged here so we get a reason even if absorption fires during a pause)
        if self._band_bias_paused:
            log.info(f"[debug] TREND signal @ {hour_min} ET REJECTED: band bias PAUSED (prior 5m close violated band zone) | trend={self._trend_direction} absorption={abs_type} confirm={confirm_dir} ADX={adx_str}")
            return None

        entry_type = None

        if self._trend_direction == "up":
            band = self.std1_upper
            if band is None:
                return None

            # Upper band zone: 3 pts above, 4 pts below (asymmetric)
            zone_high = band + BAND_ZONE_ABOVE
            zone_low = band - BAND_ZONE_BELOW
            ctx = _trend_ctx(band, zone_low, zone_high)

            # Signal bar must touch band zone
            if not (signal_bar.high >= zone_low and signal_bar.low <= zone_high):
                log.info(f"[debug] TREND signal @ {hour_min} ET REJECTED: signal bar did not touch upper band zone | {ctx}")
                return None
            # Confirm bar must touch band zone
            if not (confirm_bar.high >= zone_low and confirm_bar.low <= zone_high):
                log.info(f"[debug] TREND signal @ {hour_min} ET REJECTED: confirm bar did not touch upper band zone | {ctx}")
                return None

            # Must be bullish absorption + bullish confirm
            if not absorption["bullish"]:
                log.info(f"[debug] TREND signal @ {hour_min} ET REJECTED: need bullish absorption (got {abs_type}) for UP trend long | {ctx}")
                return None
            if confirm_bar.close <= confirm_bar.open:
                log.info(f"[debug] TREND signal @ {hour_min} ET REJECTED: confirm bar not bullish (dir={confirm_dir}) | {ctx}")
                return None

            # Approach gate: a 5-min bar must have closed at/above the upper band
            # during the current trend run (Option B — 5-min timeframe, not range bars).
            if not self._approach_from_above:
                log.info(f"[debug] TREND signal @ {hour_min} ET REJECTED: no 5m approach from above (no 5m close >= band yet in this trend run) | {ctx}")
                return None

            entry_type = "BUY"
            log.info(
                f"[engine] TREND ENTRY: BUY @ {confirm_bar.close:.2f} | "
                f"Trend=UP band={band:.2f} zone={zone_low:.2f}-{zone_high:.2f} | "
                f"ATR={atr:.2f} VWAP={vwap:.2f} ADX={adx_str}"
            )

        elif self._trend_direction == "down":
            band = self.std1_lower
            if band is None:
                return None

            # Lower band zone: 3 pts below (away from VWAP), 4 pts above (toward VWAP)
            zone_low = band - BAND_ZONE_ABOVE
            zone_high = band + BAND_ZONE_BELOW
            ctx = _trend_ctx(band, zone_low, zone_high)

            # Signal bar must touch band zone
            if not (signal_bar.high >= zone_low and signal_bar.low <= zone_high):
                log.info(f"[debug] TREND signal @ {hour_min} ET REJECTED: signal bar did not touch lower band zone | {ctx}")
                return None
            # Confirm bar must touch band zone
            if not (confirm_bar.high >= zone_low and confirm_bar.low <= zone_high):
                log.info(f"[debug] TREND signal @ {hour_min} ET REJECTED: confirm bar did not touch lower band zone | {ctx}")
                return None

            # Must be bearish absorption + bearish confirm
            if not absorption["bearish"]:
                log.info(f"[debug] TREND signal @ {hour_min} ET REJECTED: need bearish absorption (got {abs_type}) for DOWN trend short | {ctx}")
                return None
            if confirm_bar.close >= confirm_bar.open:
                log.info(f"[debug] TREND signal @ {hour_min} ET REJECTED: confirm bar not bearish (dir={confirm_dir}) | {ctx}")
                return None

            # Approach gate: a 5-min bar must have closed at/below the lower band
            # during the current trend run (Option B — 5-min timeframe, not range bars).
            if not self._approach_from_below:
                log.info(f"[debug] TREND signal @ {hour_min} ET REJECTED: no 5m approach from below (no 5m close <= band yet in this trend run) | {ctx}")
                return None

            entry_type = "SELL"
            log.info(
                f"[engine] TREND ENTRY: SELL @ {confirm_bar.close:.2f} | "
                f"Trend=DOWN band={band:.2f} zone={zone_low:.2f}-{zone_high:.2f} | "
                f"ATR={atr:.2f} VWAP={vwap:.2f} ADX={adx_str}"
            )

        if entry_type is None:
            return None

        entry_price = confirm_bar.close
        if entry_type == "BUY":
            stop_price = entry_price - (atr * self.trend_stop_mult)
            target_price = entry_price + (atr * self.trend_target_mult)
        else:
            stop_price = entry_price + (atr * self.trend_stop_mult)
            target_price = entry_price - (atr * self.trend_target_mult)

        stop_price = round(stop_price * 4) / 4
        target_price = round(target_price * 4) / 4

        self._position = Position(
            direction=1 if entry_type == "BUY" else -1,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            entry_time=confirm_bar.close_time,
            source="trend",
        )

        signal = Signal.ENTER_LONG if entry_type == "BUY" else Signal.ENTER_SHORT
        log.info(
            f"[engine] Stop={stop_price:.2f} Target={target_price:.2f} | "
            f"SL={self.trend_stop_mult}x TP={self.trend_target_mult}x ATR"
        )

        return EngineOutput(signal, self._position, bar_close=entry_price)
