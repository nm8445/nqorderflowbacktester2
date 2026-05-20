"""Phase 3b: B2 live signal computation from 5-min bars.

Port of scripts/overnight range strat/scripts/range_break_entry_signal_study.py
to run in real time on closed 5-min bars.

Pipeline per closed 5-min bar:
  1. Overnight range tracker — from 18:00 ET D-1 → 09:30 ET D, accumulate OHI/OLO.
     Freeze at 09:30 ET when RTH opens.
  2. Bias-state segmentation on RTH closes (9:30-17:00 ET):
       - First 3-consec close outside range fires LONG_BREAK or SHORT_BREAK.
       - SHORT_BREAK + 5 inside closes → flip to SHORT_FLIP_LONG bias.
       - Fresh opposite break (3-consec) overrides (max 1 flip).
  3. Per-bar B2 Mech-B signal detection:
       - Pinbar shape (lower_wick or upper_wick / body) >= 0.75
       - Level proximity (clip(0.25*ATR, 5, 20)) overlapping a near gamma level
       - Approach direction (prior close above/below level)
       - Strict-shorts gate (close < OLO)
       - Windowed orderflow scan: |best_delta_w15| >= 70 in bias direction
  4. B2 confirmation: the NEXT 5-min bar must close in bias direction.
     Compute conf_delta_half_w5 on that confirmation bar (bottom/top half of bar).
     Require |conf_delta_half_w5| >= 75 in bias direction.
  5. Fire entry on the bar AFTER confirmation:
       B2Engine.set_pending_entry(entry_time, direction, entry_price, atr_y_20m)

Live state lifecycle:
  - Session date = the CME date the session ENDS (e.g. Mon's session = Sun 18:00 → Mon 17:00)
  - At first bar of a new session (open_time >= 18:00 ET D-1 in tz-aware terms), reset state.

Gamma levels for the current session come from prior-day's row in
menthorq_levels_nq.parquet (lookahead-free).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from live.combined.bar_builder import Bar
from live.combined.b2_engine import (
    B2Engine, VARIANT, X as PINBAR_X, N_DELTA, D_DELTA,
    STRICT_SHORT, BAND_K, CONF_N, CONF_D,
)
from live.combined.config import ET_TZ

# Constants from the signal study (locked)
ENTRY_N        = 3        # consecutive 5-min closes outside range to break
RECLAIM_FLIP   = 5        # SHORT_BREAK → LONG flip threshold
TICK           = 0.25
ATR_PERIOD     = 14
BAND_MIN       = 5.0
BAND_MAX       = 20.0

# B2 entry: signal bar T → confirmation bar T+1 → ENTRY at T+2's open (matches
# the signal study's entry_idx = idx + 2 logic, plus entry_price = bar's open at T+2)


@dataclass
class _ATR5m:
    length: int = ATR_PERIOD
    _tr_buf: list = field(default_factory=list)
    _atr: float = float("nan")
    _prev_close: float = float("nan")
    def update(self, h, l, c) -> float:
        if not np.isnan(self._prev_close):
            tr = max(h - l, abs(h - self._prev_close), abs(l - self._prev_close))
        else:
            tr = h - l
        self._prev_close = c
        if np.isnan(self._atr):
            self._tr_buf.append(tr)
            if len(self._tr_buf) == self.length:
                self._atr = sum(self._tr_buf) / self.length
        else:
            alpha = 1.0 / self.length
            self._atr = (1 - alpha) * self._atr + alpha * tr
        return self._atr


def _session_date_for_bar(bar_open_et: pd.Timestamp) -> dt.date:
    """CME session date: if open >= 18:00 ET, belongs to next day's session."""
    if bar_open_et.hour >= 18:
        return (bar_open_et + pd.Timedelta(days=1)).date()
    return bar_open_et.date()


@dataclass
class _PendingB2:
    """A signal candidate waiting for confirmation on the next bar."""
    signal_bar_close_et: pd.Timestamp
    direction: str           # "LONG" or "SHORT"
    near_level: float
    strict_short: Optional[bool]
    pinbar_ratio: float
    abs_delta_w15: float
    signal_low: float
    signal_high: float


class B2SignalEngine:
    """Real-time B2 signal detector. Subscribes to 5-min bars; emits
    set_pending_entry() calls into a B2Engine."""

    def __init__(self, b2_engine: B2Engine,
                 gamma_parquet: Path | str = Path("D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet")):
        self.b2 = b2_engine
        self._gamma_parquet_path = Path(gamma_parquet)
        self._gp_mtime: float = 0.0
        self._gp = None     # lazy-loaded via _maybe_reload_parquet()
        self._maybe_reload_parquet()

        # Per-session state
        self._session_date: Optional[dt.date] = None
        self._ohi: Optional[float] = None
        self._olo: Optional[float] = None
        # Bias state machine
        self._bias = "NEUTRAL"
        self._bias_origin = "NEUTRAL"
        self._above = 0
        self._below = 0
        self._inside_run = 0
        self._flipped = False
        # Levels for this session
        self._levels: Optional[np.ndarray] = None
        # ATR
        self._atr = _ATR5m(length=ATR_PERIOD)
        # B2 confirmation queue (single pending at a time per direction)
        self._pending: Optional[_PendingB2] = None
        # Confirmation bar buffer — last RTH bar (used for B2 confirm + entry next)
        self._last_rth_bar: Optional[Bar] = None
        # Bars seen during RTH (for chronological position)
        self._rth_bars_seen = 0

        self.n_bars_seen = 0
        self.n_signals_emitted = 0
        self.n_blocked_proximity = 0
        self.n_blocked_pinbar = 0
        self.n_blocked_delta = 0
        self.n_blocked_confirm = 0

    def _maybe_reload_parquet(self) -> bool:
        """Re-read gamma parquet if its mtime changed since last load.
        Also pushes any new qqq_gamma_sign rows into the B2Engine's dict.
        Returns True if reloaded, False if no change."""
        try:
            mt = self._gamma_parquet_path.stat().st_mtime
        except FileNotFoundError:
            return False
        if mt == self._gp_mtime:
            return False
        self._gp = pd.read_parquet(self._gamma_parquet_path)
        self._gp["date"] = pd.to_datetime(self._gp["date"]).dt.date
        self._gp_mtime = mt
        # Push gamma_sign rows into B2Engine (covers new dates added since last load)
        if "qqq_gamma_sign" in self._gp.columns:
            for d, sign in zip(self._gp["date"], self._gp["qqq_gamma_sign"]):
                if pd.notna(sign):
                    self.b2.set_gamma_sign(d, int(sign))
        print(f"  [b2_signal_engine] reloaded gamma parquet "
              f"({len(self._gp)} rows, latest {self._gp['date'].max()})")
        return True

    def _reset_session(self, session_date: dt.date) -> None:
        """Start a new session. Reset bias, OHI/OLO, ATR, pending.

        Also checks the gamma parquet for updates (hot-reload). This lets
        new gamma rows (added by run_daily.py after engine startup) take
        effect on the next session boundary without requiring an engine
        restart."""
        # Hot-reload check before loading levels for the new session
        self._maybe_reload_parquet()

        self._session_date = session_date
        self._ohi = None
        self._olo = None
        self._bias = "NEUTRAL"; self._bias_origin = "NEUTRAL"
        self._above = 0; self._below = 0
        self._inside_run = 0; self._flipped = False
        self._pending = None
        self._last_rth_bar = None
        self._rth_bars_seen = 0
        # Flag — flips True when we do the morning reload at first RTH bar
        # (lets gamma rows added by 8 AM run_daily.py be picked up before RTH
        # opens, so Wed's session can use Tue's freshly-added gamma).
        self._did_rth_reload = False
        # ATR carries across sessions (Pine convention)
        self._levels = self._load_levels_for_session(session_date)

    def _load_levels_for_session(self, session_date: dt.date) -> Optional[np.ndarray]:
        """Pull prior-day gamma levels (lookahead-free)."""
        # Find latest date strictly < session_date
        earlier = self._gp[self._gp["date"] < session_date]
        if earlier.empty:
            return None
        row = earlier.iloc[-1]
        level_cols = [c for c in self._gp.columns
                       if c.endswith("_nq") and (c.startswith("qqq_") or c.startswith("ndx_"))]
        vals = row[level_cols].dropna().astype(float).values
        return np.unique(vals) if len(vals) > 0 else None

    def _bar_time_et(self, bar: Bar) -> pd.Timestamp:
        ts = bar.open_time
        if ts.tzinfo is None:
            ts = ts.tz_localize(ET_TZ)
        else:
            ts = ts.tz_convert(ET_TZ)
        return ts

    def on_5min_bar(self, bar: Bar) -> None:
        """Process one closed 5-min bar."""
        self.n_bars_seen += 1
        ts = self._bar_time_et(bar)
        bar_t = ts.time()
        session_date = _session_date_for_bar(ts)

        if session_date != self._session_date:
            self._reset_session(session_date)

        # ATR updates EVERY bar (RTH + overnight)
        atr_5m = self._atr.update(bar.high, bar.low, bar.close)

        # --- OVERNIGHT range tracking: 18:00 ET (D-1) → 09:30 ET (D) ---
        in_overnight = (bar_t >= dt.time(18, 0)) or (bar_t < dt.time(9, 30))
        if in_overnight:
            if self._ohi is None:
                self._ohi = bar.high
                self._olo = bar.low
            else:
                self._ohi = max(self._ohi, bar.high)
                self._olo = min(self._olo, bar.low)
            return  # No signal logic outside RTH

        # At 09:30 ET first bar — OHI/OLO frozen above already. Continue.
        # Must be RTH (>=09:30, <17:00 ET) for signal logic.
        if not (dt.time(9, 30) <= bar_t < dt.time(17, 0)):
            return

        # MORNING RELOAD — fires ONCE per session at the first RTH bar.
        # If gamma_refresh added new rows since startup (typical: 8 AM task
        # adds yesterday's row), this picks them up before any RTH signal
        # fires. Matches backtest's "use freshest prior-day gamma" behavior.
        if not self._did_rth_reload:
            self._did_rth_reload = True
            reloaded = self._maybe_reload_parquet()
            if reloaded:
                # Re-pick prior-day levels with the updated parquet
                new_levels = self._load_levels_for_session(self._session_date)
                if new_levels is not None:
                    self._levels = new_levels
                    print(f"  [b2_signal_engine] RTH morning reload: {len(new_levels)} levels "
                          f"loaded for session {self._session_date}")

        if self._ohi is None or self._levels is None:
            return  # no overnight range or no gamma levels — can't trade

        self._rth_bars_seen += 1

        # === Stage 1: BIAS state machine on this bar's close ===
        self._update_bias(bar.close)

        # === Stage 2: B2 CONFIRMATION + entry firing ===
        # The PRIOR rth bar may have been a signal candidate; if so, this
        # bar is the confirmation. Then the NEXT bar will be the entry.
        if self._pending is not None:
            self._evaluate_confirmation(bar, ts, atr_5m)

        # === Stage 3: New signal detection on THIS bar ===
        if self._bias in ("LONG", "SHORT") and self._rth_bars_seen >= 2:
            self._try_new_signal(bar, ts, atr_5m)

        self._last_rth_bar = bar

    def _update_bias(self, close: float) -> None:
        if close > self._ohi:
            self._above += 1; self._below = 0
        elif close < self._olo:
            self._below += 1; self._above = 0
        else:
            self._above = 0; self._below = 0

        # inside_run only counts after a SHORT bias is set
        location_inside = (self._olo <= close <= self._ohi)
        if self._bias_origin.startswith("SHORT") and not self._flipped and location_inside:
            self._inside_run += 1
        elif not location_inside:
            self._inside_run = 0

        # First trigger
        if self._bias == "NEUTRAL":
            if self._above >= ENTRY_N:
                self._bias = "LONG"; self._bias_origin = "LONG_BREAK"
            elif self._below >= ENTRY_N:
                self._bias = "SHORT"; self._bias_origin = "SHORT_BREAK"
        elif not self._flipped:
            # Existing bias, check for flip
            if self._bias == "SHORT" and self._inside_run >= RECLAIM_FLIP:
                self._bias = "LONG"; self._bias_origin = "SHORT_FLIP_LONG"; self._flipped = True
            elif self._bias == "SHORT" and self._above >= ENTRY_N:
                self._bias = "LONG"; self._bias_origin = "LONG_BREAK"; self._flipped = True
            elif self._bias == "LONG" and self._below >= ENTRY_N and self._bias_origin != "LONG_BREAK":
                self._bias = "SHORT"; self._bias_origin = "SHORT_BREAK"; self._flipped = True

    def _try_new_signal(self, bar: Bar, ts: pd.Timestamp, atr_5m: float) -> None:
        """Run the B2 Mech-B signal filter chain on THIS bar.
        If passes, queue as pending — waiting for confirmation on the next bar."""
        if not np.isfinite(atr_5m) or atr_5m <= 0:
            return

        direction = self._bias    # LONG or SHORT
        body = max(abs(bar.close - bar.open), TICK)
        # Pinbar shape (relevant wick / body)
        if direction == "LONG":
            relevant_wick = min(bar.open, bar.close) - bar.low
        else:
            relevant_wick = bar.high - max(bar.open, bar.close)
        pinbar_ratio = relevant_wick / body
        if pinbar_ratio < PINBAR_X:
            self.n_blocked_pinbar += 1
            return

        # Level proximity — adaptive band
        band = float(np.clip(BAND_K * atr_5m, BAND_MIN, BAND_MAX))
        # Find any level whose [lvl-band, lvl+band] intersects [bar.low, bar.high]
        diffs = np.abs(self._levels - (bar.low + bar.high) / 2)
        # check ANY level overlaps the bar with the band
        overlap_mask = (self._levels - band <= bar.high) & (self._levels + band >= bar.low)
        if not overlap_mask.any():
            self.n_blocked_proximity += 1
            return
        # Pick nearest qualifying level
        candidate_lvls = self._levels[overlap_mask]
        near_lvl = float(candidate_lvls[np.argmin(np.abs(candidate_lvls - (bar.low + bar.high) / 2))])

        # Approach direction filter (need prior bar)
        if self._last_rth_bar is None:
            return
        prior_close = self._last_rth_bar.close
        if direction == "LONG" and prior_close <= near_lvl:
            return
        if direction == "SHORT" and prior_close >= near_lvl:
            return

        # Strict-shorts gate
        strict_short = None
        if direction == "SHORT":
            close_below = bar.close < self._olo
            wick_below  = bar.low   < self._olo
            if not (close_below or wick_below):
                return
            strict_short = bool(close_below)

        # Windowed absorption — |best_delta_w15| >= 70 in bias direction
        # Need per-price-level breakdown of this bar — comes from bar.level_volumes
        if not bar.level_volumes:
            self.n_blocked_delta += 1
            return
        # Wick zone: below body_low for LONG, above body_high for SHORT
        body_low  = min(bar.open, bar.close)
        body_high = max(bar.open, bar.close)
        zone = []
        for px, (b, s) in bar.level_volumes.items():
            if direction == "LONG" and px < body_low:
                zone.append((px, b, s))
            elif direction == "SHORT" and px > body_high:
                zone.append((px, b, s))
        if len(zone) < N_DELTA:
            self.n_blocked_delta += 1
            return
        zone.sort()   # by price asc
        buys  = np.array([z[1] for z in zone], dtype=float)
        sells = np.array([z[2] for z in zone], dtype=float)
        cb = np.cumsum(buys); cs = np.cumsum(sells)
        # All sliding windows of size N_DELTA
        win_buy  = np.concatenate([[cb[N_DELTA - 1]], cb[N_DELTA:] - cb[:-N_DELTA]])
        win_sell = np.concatenate([[cs[N_DELTA - 1]], cs[N_DELTA:] - cs[:-N_DELTA]])
        deltas = win_buy - win_sell
        if direction == "LONG":
            best_delta = float(deltas.min())   # most negative (sellers absorbed)
        else:
            best_delta = float(deltas.max())   # most positive (buyers absorbed)
        if abs(best_delta) < D_DELTA:
            self.n_blocked_delta += 1
            return

        # Passed all filters — queue as pending, waiting for confirmation next bar
        self._pending = _PendingB2(
            signal_bar_close_et=ts + pd.Timedelta(minutes=5),
            direction=direction,
            near_level=near_lvl,
            strict_short=strict_short,
            pinbar_ratio=pinbar_ratio,
            abs_delta_w15=best_delta,
            signal_low=bar.low,
            signal_high=bar.high,
        )

    def _evaluate_confirmation(self, conf_bar: Bar, conf_ts: pd.Timestamp,
                                atr_5m: float) -> None:
        """conf_bar is the bar AFTER the signal bar. Check B2 confirmation:
           1. close in bias direction (bullish for LONG, bearish for SHORT)
           2. conf_delta_half_w5 in bias direction with |delta| >= CONF_D
        If both pass, queue entry for the NEXT bar (T+2 in signal study)."""
        pend = self._pending
        # Whether THIS bar (the confirmation candidate) closes in bias direction
        bullish = conf_bar.close > conf_bar.open
        if pend.direction == "LONG" and not bullish:
            self._pending = None
            return
        if pend.direction == "SHORT" and bullish:
            self._pending = None
            return

        # Compute conf_delta_half_w5 — best 5-tick window in BOTTOM HALF (LONG)
        # or TOP HALF (SHORT) of the confirmation bar's price range.
        if not conf_bar.level_volumes:
            self._pending = None; self.n_blocked_confirm += 1
            return
        midpoint = (conf_bar.low + conf_bar.high) / 2
        zone = []
        for px, (b, s) in conf_bar.level_volumes.items():
            if pend.direction == "LONG" and px < midpoint:
                zone.append((px, b, s))
            elif pend.direction == "SHORT" and px > midpoint:
                zone.append((px, b, s))
        if len(zone) < CONF_N:
            self._pending = None; self.n_blocked_confirm += 1
            return
        zone.sort()
        buys  = np.array([z[1] for z in zone], dtype=float)
        sells = np.array([z[2] for z in zone], dtype=float)
        cb = np.cumsum(buys); cs = np.cumsum(sells)
        win_buy  = np.concatenate([[cb[CONF_N - 1]], cb[CONF_N:] - cb[:-CONF_N]])
        win_sell = np.concatenate([[cs[CONF_N - 1]], cs[CONF_N:] - cs[:-CONF_N]])
        deltas = win_buy - win_sell
        if pend.direction == "LONG":
            best_conf_delta = float(deltas.max())   # buyers absorbing (positive)
            if best_conf_delta < CONF_D:
                self._pending = None; self.n_blocked_confirm += 1
                return
        else:
            best_conf_delta = float(deltas.min())   # sellers absorbing (negative)
            if best_conf_delta > -CONF_D:
                self._pending = None; self.n_blocked_confirm += 1
                return

        # Confirmation passed. FIRE ENTRY IMMEDIATELY at conf bar close time.
        # entry_price = conf_bar.close (matches backtest's "next bar open"
        # within ~1 tick, since next bar opens at conf close instant).
        # We use 20-min ATR from B2Engine (not local 5-min ATR) so yellow/green
        # match backtest's `bars20["atr_y"]` calculation.
        entry_time = conf_ts + pd.Timedelta(minutes=5)
        entry_price = conf_bar.close

        atr_y_20m = getattr(self.b2, "_last_atr_20m", None)
        if atr_y_20m is None or atr_y_20m <= 0:
            # No 20-min ATR yet (first 14 20-min bars of session not warmed).
            # Skip entry — matches backtest behavior (it requires init_atr_y
            # to be non-NaN).
            print(f"  [B2-SIG] {entry_time.strftime('%Y-%m-%d %H:%M')} {pend.direction} "
                  f"near={pend.near_level:.2f} -- SKIPPED (no 20-min ATR yet)")
            self._pending = None
            return

        print(f"  [B2-SIG] {entry_time.strftime('%Y-%m-%d %H:%M')} {pend.direction} "
              f"near={pend.near_level:.2f} pinbar={pend.pinbar_ratio:.2f} "
              f"abs_delta_w15={pend.abs_delta_w15:+.0f} conf_delta_half_w5={best_conf_delta:+.0f} "
              f"atr_20m={atr_y_20m:.2f}")

        fired = self.b2.fire_entry_from_signal(
            direction=pend.direction,
            entry_price=entry_price,
            atr_y=atr_y_20m,
            ts=entry_time,
        )
        if fired:
            self.n_signals_emitted += 1
        self._pending = None
