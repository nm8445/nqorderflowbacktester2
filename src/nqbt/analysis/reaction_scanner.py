"""
Reaction scanner: identifies qualifying location events from RTH range bars
and measures their forward outcomes to session close.

A qualifying event is any RTH range bar where:
  - price_location is in ('outside_vah', 'outside_val', 'at_vah', 'at_val'), OR
  - bar close is within VWAP_BAND_PROX_PTS (0.75 pts = 3 ticks) of std2_upper or std2_lower

For each event bar, forward outcomes are measured from bar close_time
to end of RTH session (4:15 PM ET):
  - forward_high, forward_low, forward_final (last bar close in session)
  - mfe_up   = forward_high - bar.close  (max gain if long entry at close)
  - mfe_down = bar.close - forward_low   (max gain if short entry at close)
  - net_move = forward_final - bar.close (signed, + = price went up)

Price location classification (relative to RTH developing profile at bar close time):
  - outside_vah: bar.close > vah + 1 tick (0.25)
  - outside_val: bar.close < val - 1 tick (0.25)
  - at_vah:      |bar.close - vah| <= 4 ticks (1.00)   [not outside]
  - at_val:      |bar.close - val| <= 4 ticks (1.00)   [not outside]
  - inside_va:   everything else

VWAP band proximity: |bar.close - std2_upper| <= 0.75 OR |bar.close - std2_lower| <= 0.75
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nqbt.analysis.range_bars import build_range_bars, RangeBar
from nqbt.analysis.volume_profile import VolumeProfile
from nqbt.analysis.vwap import vwap_at

ET             = "America/New_York"
TICK_SIZE      = 0.25
RTH_OPEN       = "09:30"
RTH_CLOSE      = "16:15"
VWAP_BAND_PROX_PTS = 0.75
AT_LEVEL_TICKS = 4
OUTSIDE_TICKS  = 1
TICK_CACHE_DIR = Path("output/tick_cache")

_EVENT_COLUMNS = [
    "session_date", "event_time", "bar_open_time",
    "bar_open", "bar_high", "bar_low", "bar_close",
    "bar_buy_vol", "bar_sell_vol", "bar_delta", "bar_total_vol",
    "price_location", "vwap_band_proximity",
    "vwap", "std", "std1_upper", "std1_lower", "std2_upper", "std2_lower",
    "vah", "val", "poc",
    "bar_idx",
    "forward_high", "forward_low", "forward_final",
    "mfe_up", "mfe_down", "net_move",
]


def _load_rth_ticks(session_date: str) -> pd.DataFrame:
    """
    Load tick cache for a session, convert index to ET, filter to RTH hours.

    Returns an empty DataFrame if the file is not found.
    """
    path = TICK_CACHE_DIR / f"{session_date}_ticks.parquet"
    if not path.exists():
        return pd.DataFrame()

    ticks = pd.read_parquet(path)
    ticks.index = ticks.index.tz_convert(ET)

    rth_start = pd.Timestamp(f"{session_date} {RTH_OPEN}", tz=ET)
    rth_end   = pd.Timestamp(f"{session_date} {RTH_CLOSE}", tz=ET)

    ticks = ticks[(ticks.index >= rth_start) & (ticks.index <= rth_end)]
    return ticks


def _classify_price_location(price: float, vah: float, val: float) -> str:
    """Classify price relative to the developing value area."""
    if price > vah + OUTSIDE_TICKS * TICK_SIZE:
        return "outside_vah"
    if price < val - OUTSIDE_TICKS * TICK_SIZE:
        return "outside_val"
    if abs(price - vah) <= AT_LEVEL_TICKS * TICK_SIZE:
        return "at_vah"
    if abs(price - val) <= AT_LEVEL_TICKS * TICK_SIZE:
        return "at_val"
    return "inside_va"


def _is_vwap_band_proximity(price: float, std2_upper: float, std2_lower: float) -> bool:
    """Return True if price is within VWAP_BAND_PROX_PTS of either std2 band."""
    return (
        abs(price - std2_upper) <= VWAP_BAND_PROX_PTS
        or abs(price - std2_lower) <= VWAP_BAND_PROX_PTS
    )


def scan_session(session_date: str) -> pd.DataFrame:
    """
    Scan one RTH session for qualifying location events and compute
    forward outcomes.

    Parameters
    ----------
    session_date : str
        YYYY-MM-DD string.

    Returns
    -------
    pd.DataFrame
        One row per qualifying bar.  Empty DataFrame with _EVENT_COLUMNS
        if no qualifying bars found or data is unavailable.
    """
    rth_ticks = _load_rth_ticks(session_date)
    if rth_ticks.empty:
        return pd.DataFrame(columns=_EVENT_COLUMNS)

    # Build range bars on UTC-indexed ticks (convert back for bar building)
    rth_ticks_utc = rth_ticks.copy()
    rth_ticks_utc.index = rth_ticks_utc.index.tz_convert("UTC")

    bars: list[RangeBar] = build_range_bars(rth_ticks_utc, range_ticks=40, ticks_per_level=5)

    if not bars:
        return pd.DataFrame(columns=_EVENT_COLUMNS)

    events: list[dict] = []

    for bar_idx, bar in enumerate(bars):
        # Only process closed bars
        if bar.close_time is None:
            continue

        close_time_utc = bar.close_time  # UTC timestamp

        # Developing profile: all RTH ticks up to and including bar.close_time
        profile_ticks = rth_ticks_utc.loc[:close_time_utc]
        if profile_ticks.empty:
            continue

        vp = VolumeProfile.build(profile_ticks, ticks_per_level=10)
        if vp.total_volume == 0:
            continue

        vwap_data = vwap_at(rth_ticks_utc, as_of=close_time_utc, std_multipliers=(1, 2, 3))
        if not vwap_data:
            continue

        price_loc  = _classify_price_location(bar.close, vp.vah, vp.val)
        band_prox  = _is_vwap_band_proximity(
            bar.close,
            vwap_data.get("std2_upper", float("inf")),
            vwap_data.get("std2_lower", float("-inf")),
        )

        qualifying_location = price_loc in ("outside_vah", "outside_val", "at_vah", "at_val")
        if not qualifying_location and not band_prox:
            continue

        # Forward outcome: bars after this event bar through session end
        forward_bars = [b for b in bars[bar_idx + 1:] if b.close_time is not None]

        if forward_bars:
            forward_highs  = [b.high  for b in forward_bars]
            forward_lows   = [b.low   for b in forward_bars]
            forward_closes = [b.close for b in forward_bars]
            forward_high   = max(forward_highs)
            forward_low    = min(forward_lows)
            forward_final  = forward_closes[-1]
        else:
            forward_high  = bar.close
            forward_low   = bar.close
            forward_final = bar.close

        mfe_up   = forward_high  - bar.close
        mfe_down = bar.close     - forward_low
        net_move = forward_final - bar.close

        events.append({
            "session_date":       session_date,
            "event_time":         close_time_utc,
            "bar_open_time":      bar.open_time,
            "bar_open":           bar.open,
            "bar_high":           bar.high,
            "bar_low":            bar.low,
            "bar_close":          bar.close,
            "bar_buy_vol":        bar.buy_vol,
            "bar_sell_vol":       bar.sell_vol,
            "bar_delta":          bar.delta,
            "bar_total_vol":      bar.total_vol,
            "price_location":     price_loc,
            "vwap_band_proximity": band_prox,
            "vwap":               vwap_data.get("vwap"),
            "std":                vwap_data.get("std"),
            "std1_upper":         vwap_data.get("std1_upper"),
            "std1_lower":         vwap_data.get("std1_lower"),
            "std2_upper":         vwap_data.get("std2_upper"),
            "std2_lower":         vwap_data.get("std2_lower"),
            "vah":                vp.vah,
            "val":                vp.val,
            "poc":                vp.poc,
            "bar_idx":            bar_idx,
            "forward_high":       forward_high,
            "forward_low":        forward_low,
            "forward_final":      forward_final,
            "mfe_up":             mfe_up,
            "mfe_down":           mfe_down,
            "net_move":           net_move,
        })

    if not events:
        return pd.DataFrame(columns=_EVENT_COLUMNS)

    return pd.DataFrame(events, columns=_EVENT_COLUMNS)


def scan_date_range(
    start_date: str,
    end_date: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Scan all business day sessions in [start_date, end_date].

    Parameters
    ----------
    start_date, end_date : str
        YYYY-MM-DD strings (inclusive).
    verbose : bool
        Print per-date progress. Default True.

    Returns
    -------
    pd.DataFrame
        Combined events from all sessions, index reset.
    """
    business_days = pd.date_range(start_date, end_date, freq="B")
    all_frames: list[pd.DataFrame] = []

    for dt in business_days:
        date_str = dt.strftime("%Y-%m-%d")
        try:
            df = scan_session(date_str)
            n  = len(df)
            if verbose:
                print(f"{date_str}: {n} events")
            if n > 0:
                all_frames.append(df)
        except Exception as exc:
            print(f"WARNING: {date_str} failed — {exc}")

    if not all_frames:
        return pd.DataFrame(columns=_EVENT_COLUMNS)

    return pd.concat(all_frames, ignore_index=True)
