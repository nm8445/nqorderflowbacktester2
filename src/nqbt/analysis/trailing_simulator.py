"""
Trailing stop rule simulator.

For each reaction event, simulates 5 families of exit rules against the
forward price sequence and computes:
  - captured_points: points captured by the rule (negative = loss)
  - captured_pct_of_mfe: captured_points / mfe (if mfe > 0, else NaN)
  - exit_reason: 'stop', 'target', 'session_end'

Direction conventions:
  - direction='long':  entry at bar_close, profit when price rises
    mfe = mfe_up, captured_points = exit_price - entry_price
  - direction='short': entry at bar_close, profit when price falls
    mfe = mfe_down, captured_points = entry_price - exit_price

For forward price simulation we use bar-by-bar high/low/close from bars
after the event bar. Each bar is processed in order; the stop/target is
checked against bar high/low (stop first, target second within a bar).
"""

from __future__ import annotations

import math

import pandas as pd

TICK_SIZE = 0.25

RULE_FIXED_STOPS       = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]   # pts
RULE_TRAIL_AMOUNTS     = [2.0, 3.0, 4.0, 6.0, 8.0]                        # pts
RULE_BREAKEVEN_TRIG    = [2.0, 4.0, 6.0, 8.0]                             # pts profit to trigger
RULE_PT_TARGETS        = [4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 30.0]   # pts
RULE_COMBO_TRAIL_AFTER = [4.0, 6.0, 8.0, 12.0]                            # pts profit before trailing


# ---------------------------------------------------------------------------
# Rule family 1 — fixed stop, no target
# ---------------------------------------------------------------------------

def _simulate_fixed_stop(
    forward_bars: list,
    entry: float,
    direction: str,
    stop_pts: float,
) -> tuple[float, str]:
    """
    Fixed stop loss, no take-profit target. Ride to session end or stop.

    Stop is placed at entry - stop_pts (long) or entry + stop_pts (short).
    Within each bar: check stop against bar.low (long) or bar.high (short).
    """
    if direction == "long":
        stop = entry - stop_pts
        for bar in forward_bars:
            if bar.low <= stop:
                return stop, "stop"
        # Session end
        return forward_bars[-1].close if forward_bars else entry, "session_end"
    else:  # short
        stop = entry + stop_pts
        for bar in forward_bars:
            if bar.high >= stop:
                return stop, "stop"
        return forward_bars[-1].close if forward_bars else entry, "session_end"


# ---------------------------------------------------------------------------
# Rule family 2 — bar-by-bar trailing stop
# ---------------------------------------------------------------------------

def _simulate_bar_trail(
    forward_bars: list,
    entry: float,
    direction: str,
    trail_pts: float,
) -> tuple[float, str]:
    """
    Trail stop bar-by-bar: after each completed bar, move the stop to
    bar.low - trail_pts (long) or bar.high + trail_pts (short).

    Initial stop before the first bar: entry - trail_pts (long) or
    entry + trail_pts (short).
    """
    if not forward_bars:
        return entry, "session_end"

    if direction == "long":
        stop = entry - trail_pts
        for bar in forward_bars:
            if bar.low <= stop:
                return stop, "stop"
            # Update trail to bar low - trail
            new_stop = bar.low - trail_pts
            if new_stop > stop:
                stop = new_stop
        return forward_bars[-1].close, "session_end"
    else:  # short
        stop = entry + trail_pts
        for bar in forward_bars:
            if bar.high >= stop:
                return stop, "stop"
            new_stop = bar.high + trail_pts
            if new_stop < stop:
                stop = new_stop
        return forward_bars[-1].close, "session_end"


# ---------------------------------------------------------------------------
# Rule family 3 — breakeven then bar-trail
# ---------------------------------------------------------------------------

def _simulate_breakeven_trail(
    forward_bars: list,
    entry: float,
    direction: str,
    trigger_pts: float,
    trail_pts: float,
) -> tuple[float, str]:
    """
    Phase 1: fixed stop at entry - trail_pts (long) / entry + trail_pts (short).
    When profit >= trigger_pts, switch to bar-by-bar trailing stop.

    Trigger is checked against bar.high (long) or bar.low (short) intrabar.
    Once triggered, stop is immediately set to extreme - trail_pts.
    """
    if not forward_bars:
        return entry, "session_end"

    trailing = False
    stop: float

    if direction == "long":
        stop = entry - trail_pts
        for bar in forward_bars:
            if not trailing:
                # Check if stop was hit before trigger
                if bar.low <= stop:
                    return stop, "stop"
                # Check trigger
                if bar.high - entry >= trigger_pts:
                    trailing = True
                    stop = bar.low - trail_pts
            else:
                if bar.low <= stop:
                    return stop, "stop"
                new_stop = bar.low - trail_pts
                if new_stop > stop:
                    stop = new_stop
        return forward_bars[-1].close, "session_end"
    else:  # short
        stop = entry + trail_pts
        for bar in forward_bars:
            if not trailing:
                if bar.high >= stop:
                    return stop, "stop"
                if entry - bar.low >= trigger_pts:
                    trailing = True
                    stop = bar.high + trail_pts
            else:
                if bar.high >= stop:
                    return stop, "stop"
                new_stop = bar.high + trail_pts
                if new_stop < stop:
                    stop = new_stop
        return forward_bars[-1].close, "session_end"


# ---------------------------------------------------------------------------
# Rule family 4 — profit target only
# ---------------------------------------------------------------------------

def _simulate_profit_target(
    forward_bars: list,
    entry: float,
    direction: str,
    target_pts: float,
) -> tuple[float, str]:
    """
    No stop — exit when price reaches entry + target_pts (long) or
    entry - target_pts (short). If never hit, exit at session end.
    """
    if not forward_bars:
        return entry, "session_end"

    if direction == "long":
        target = entry + target_pts
        for bar in forward_bars:
            if bar.high >= target:
                return target, "target"
        return forward_bars[-1].close, "session_end"
    else:  # short
        target = entry - target_pts
        for bar in forward_bars:
            if bar.low <= target:
                return target, "target"
        return forward_bars[-1].close, "session_end"


# ---------------------------------------------------------------------------
# Rule family 5 — combo: ride then trail
# ---------------------------------------------------------------------------

def _simulate_combo(
    forward_bars: list,
    entry: float,
    direction: str,
    switch_after_pts: float,
    trail_pts: float,
) -> tuple[float, str]:
    """
    Phase 1: no stop, just ride.
    When profit >= switch_after_pts: switch to bar-by-bar trailing stop.

    Trigger checked against bar.high (long) or bar.low (short) intrabar.
    """
    if not forward_bars:
        return entry, "session_end"

    trailing = False
    stop = float("nan")

    if direction == "long":
        for bar in forward_bars:
            if not trailing:
                if bar.high - entry >= switch_after_pts:
                    trailing = True
                    stop = bar.low - trail_pts
            else:
                if bar.low <= stop:
                    return stop, "stop"
                new_stop = bar.low - trail_pts
                if new_stop > stop:
                    stop = new_stop
        return forward_bars[-1].close, "session_end"
    else:  # short
        for bar in forward_bars:
            if not trailing:
                if entry - bar.low >= switch_after_pts:
                    trailing = True
                    stop = bar.high + trail_pts
            else:
                if bar.high >= stop:
                    return stop, "stop"
                new_stop = bar.high + trail_pts
                if new_stop < stop:
                    stop = new_stop
        return forward_bars[-1].close, "session_end"


# ---------------------------------------------------------------------------
# Event simulator
# ---------------------------------------------------------------------------

def simulate_event(
    event: pd.Series,
    forward_bars: list,
    directions: list[str] = None,
) -> list[dict]:
    """
    Run all 5 rule families with all param combinations for each direction.

    Parameters
    ----------
    event : pd.Series
        One row from the reaction events DataFrame.
    forward_bars : list[RangeBar]
        Bars after the event bar (can be empty list).
    directions : list[str]
        ['long', 'short'] by default.

    Returns
    -------
    list[dict]
        One dict per (direction × rule_family × params) combination.
        Keys: session_date, event_time, direction, rule_family, params,
              entry_price, exit_price, captured_points, mfe,
              captured_pct_of_mfe, exit_reason.
    """
    if directions is None:
        directions = ["long", "short"]

    entry = float(event["bar_close"])
    results: list[dict] = []

    for direction in directions:
        mfe = float(event["mfe_up"]) if direction == "long" else float(event["mfe_down"])

        def _row(rule_family: str, params: dict, exit_price: float, exit_reason: str) -> dict:
            if direction == "long":
                captured = exit_price - entry
            else:
                captured = entry - exit_price

            if mfe > 0:
                captured_pct = captured / mfe
            else:
                captured_pct = float("nan")

            return {
                "session_date":       event["session_date"],
                "event_time":         event["event_time"],
                "direction":          direction,
                "rule_family":        rule_family,
                "params":             str(params),
                "entry_price":        entry,
                "exit_price":         round(exit_price, 4),
                "captured_points":    round(captured, 4),
                "mfe":                mfe,
                "captured_pct_of_mfe": captured_pct,
                "exit_reason":        exit_reason,
            }

        # Rule family 1 — fixed stop
        for sp in RULE_FIXED_STOPS:
            ep, reason = _simulate_fixed_stop(forward_bars, entry, direction, sp)
            results.append(_row("fixed_stop", {"stop_pts": sp}, ep, reason))

        # Rule family 2 — bar trail
        for ta in RULE_TRAIL_AMOUNTS:
            ep, reason = _simulate_bar_trail(forward_bars, entry, direction, ta)
            results.append(_row("bar_trail", {"trail_pts": ta}, ep, reason))

        # Rule family 3 — breakeven + trail
        for trig in RULE_BREAKEVEN_TRIG:
            for ta in RULE_TRAIL_AMOUNTS:
                ep, reason = _simulate_breakeven_trail(
                    forward_bars, entry, direction, trig, ta
                )
                results.append(_row(
                    "breakeven_trail",
                    {"trigger_pts": trig, "trail_pts": ta},
                    ep, reason,
                ))

        # Rule family 4 — profit target only
        for tgt in RULE_PT_TARGETS:
            ep, reason = _simulate_profit_target(forward_bars, entry, direction, tgt)
            results.append(_row("profit_target", {"target_pts": tgt}, ep, reason))

        # Rule family 5 — combo
        for switch in RULE_COMBO_TRAIL_AFTER:
            for ta in RULE_TRAIL_AMOUNTS:
                ep, reason = _simulate_combo(
                    forward_bars, entry, direction, switch, ta
                )
                results.append(_row(
                    "combo",
                    {"switch_after_pts": switch, "trail_pts": ta},
                    ep, reason,
                ))

    return results


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def simulate_all_events(
    events: pd.DataFrame,
    bars_by_date: dict,
) -> pd.DataFrame:
    """
    Run trailing simulations for every reaction event.

    Parameters
    ----------
    events : pd.DataFrame
        Output of reaction_scanner.scan_date_range().
    bars_by_date : dict
        {date_str: list[RangeBar]}.

    Returns
    -------
    pd.DataFrame
        All simulation rows concatenated, index reset.
    """
    all_rows: list[dict] = []

    for _, event in events.iterrows():
        session_date = event["session_date"]
        bar_idx      = int(event["bar_idx"])

        bars = bars_by_date.get(session_date, [])
        # Forward bars = bars strictly AFTER the event bar (closed bars only)
        forward_bars = [
            b for b in bars[bar_idx + 1:]
            if b.close_time is not None
        ]

        try:
            rows = simulate_event(event, forward_bars)
            all_rows.extend(rows)
        except Exception as exc:
            print(f"WARNING: simulate_event failed for {session_date} bar {bar_idx}: {exc}")

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows).reset_index(drop=True)
