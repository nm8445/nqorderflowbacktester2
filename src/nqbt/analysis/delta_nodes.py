"""
Delta node infrastructure for Block 4 studies.

A delta node is a 20-tick price zone with significant net signed delta
(buy_vol - sell_vol) built from historical tick data. Nodes that sit
near VAH/VAL/POC are flagged as structural.

Zone construction (greedy forward walk):
  1. Round all tick prices to nearest 0.25.
  2. Compute net delta per 0.25-point level: buy_vol - sell_vol.
  3. Compute Nth-percentile threshold from session abs_delta distribution.
     Filter to levels at or above this threshold (session-adaptive).
  4. Sort remaining levels by price ascending.
  5. Walk forward greedily: take the first unclaimed level, claim a
     cluster_ticks-wide window centred on it (±cluster_ticks/2 ticks),
     sum all delta within that window into one node, assign node.price
     to the level with the highest |delta| in the window, mark all
     levels in the window as claimed, advance to the next unclaimed level.
  6. Mark is_structural = True if any part of [price_low, price_high]
     overlaps within structural_ticks of VAH, VAL, or POC.

Cache: output/delta_node_cache/{date}_nodes.pkl
  Stores nodes built with default cluster_ticks=20, including reaction tracking.
  Non-default cluster_ticks bypasses the cache and recomputes.
  NOTE: Delete the cache directory when DeltaNode fields change (pickle stores
  the old dataclass layout and will fail or return stale data).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date as date_type
from pathlib import Path
import pickle
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TICK_SIZE = 0.25
_DEFAULT_CLUSTER_TICKS = 20
_DELTA_CACHE_DIR = Path("output/delta_node_cache")


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class DeltaNode:
    price: float          # price of the highest-|delta| tick within the zone
    price_low: float      # bottom of the zone (price_low = price - half_width)
    price_high: float     # top of the zone  (price_high = price + half_width)
    delta: float          # net signed delta summed across all ticks in zone
    abs_delta: float      # |delta|
    timestamp: datetime   # time of last trade within the zone
    session_date: date_type
    is_structural: bool   # True if zone overlaps within structural_ticks of VAH/VAL/POC

    # Reaction tracking (updated by compute_node_reactions)
    reaction_count: int = 0
    last_reaction_time: datetime | None = None
    last_reaction_hours_ago: float = float("nan")
    is_proven: bool = False    # reaction_count >= 1
    is_expired: bool = False   # price closed through zone without reacting


# ---------------------------------------------------------------------------
# Core weight helper
# ---------------------------------------------------------------------------

def node_weight(node: DeltaNode, current_time: datetime, halflife_hours: float) -> float:
    """
    Time-decayed relevance weight for a delta node.

    weight = abs_delta * exp(-ln(2) * age_hours / halflife_hours)

    Returns 0.0 if halflife_hours <= 0.
    """
    if halflife_hours <= 0:
        return 0.0
    age_hours = (current_time - node.timestamp).total_seconds() / 3600.0
    return float(node.abs_delta * np.exp(-np.log(2) * age_hours / halflife_hours))


# ---------------------------------------------------------------------------
# Reaction tracking
# ---------------------------------------------------------------------------

def compute_node_reactions(
    nodes: list[DeltaNode],
    bars: list,
    reaction_ticks: int = 12,
    expiry_ticks: int = 12,
    max_age_hours: float = 72.0,
) -> None:
    """
    Scan range bars and update each node's reaction tracking in-place.

    A reaction is counted when:
      1. Price enters the node zone [price_low, price_high]
      2. Price then moves away >= reaction_ticks in the delta direction
         (up for positive-delta nodes, down for negative-delta) before
         returning to the zone.

    is_expired = True when:
      - Price closes expiry_ticks (default 12 = 3.0 pts) BEYOND the adverse
        side of the zone without producing a reaction, OR
      - The node is older than max_age_hours (default 72) before the first
        bar's timestamp (auto-expire stale nodes).

    Nodes already marked is_expired are skipped.

    Parameters
    ----------
    nodes : list[DeltaNode]
        Nodes to update in-place.
    bars : list[RangeBar]
        Range bars following node formation (same or subsequent session).
    reaction_ticks : int
        Minimum post-zone excursion in ticks to count as a reaction.
    expiry_ticks : int
        Close must penetrate this many ticks beyond the adverse zone boundary
        to trigger expiry. Default 12 (= 3.0 pts).
    max_age_hours : float
        Nodes older than this many hours before bars[0].open_time are
        auto-expired. Default 72.
    """
    if not nodes or not bars:
        return

    reaction_pts = reaction_ticks * _TICK_SIZE
    expiry_pts   = expiry_ticks   * _TICK_SIZE

    # Determine reference time from first bar for auto-expire
    try:
        first_bar_ts = pd.Timestamp(bars[0].open_time)
        if first_bar_ts.tzinfo is None:
            first_bar_ts = first_bar_ts.tz_localize("UTC")
        has_ref_time = True
    except Exception:
        has_ref_time = False

    for node in nodes:
        if node.is_expired:
            continue

        # Auto-expire nodes older than max_age_hours before this session
        if has_ref_time and max_age_hours > 0:
            try:
                node_ts = pd.Timestamp(node.timestamp)
                if node_ts.tzinfo is None:
                    node_ts = node_ts.tz_localize("UTC")
                age_hrs = (first_bar_ts - node_ts).total_seconds() / 3600.0
                if age_hrs > max_age_hours:
                    node.is_expired = True
                    continue
            except Exception:
                pass

        bull = node.delta > 0   # positive delta = bullish, expect upward reaction

        state = "idle"          # "idle" | "in_zone" | "post_zone"
        best_post: float | None = None
        last_bar_time = None

        for bar in bars:
            last_bar_time = bar.open_time
            touches = bar.low <= node.price_high and bar.high >= node.price_low

            if state == "idle":
                if touches:
                    state = "in_zone"

            elif state == "in_zone":
                if touches:
                    # Expiry: close completely through the adverse side
                    if bull and bar.close < node.price_low - expiry_pts:
                        node.is_expired = True
                        break
                    elif not bull and bar.close > node.price_high + expiry_pts:
                        node.is_expired = True
                        break
                else:
                    # Price left the zone — start measuring excursion
                    state = "post_zone"
                    best_post = bar.close

            elif state == "post_zone":
                if touches:
                    # Returned to zone before reacting — reset
                    state = "in_zone"
                    best_post = None
                else:
                    if bull:
                        best_post = max(best_post, bar.close)
                        if best_post - node.price_high >= reaction_pts:
                            node.reaction_count += 1
                            node.last_reaction_time = bar.open_time
                            node.is_proven = True
                            state = "idle"
                            best_post = None
                    else:
                        best_post = min(best_post, bar.close)
                        if node.price_low - best_post >= reaction_pts:
                            node.reaction_count += 1
                            node.last_reaction_time = bar.open_time
                            node.is_proven = True
                            state = "idle"
                            best_post = None

        # Update last_reaction_hours_ago relative to last bar scanned
        if node.last_reaction_time is not None and last_bar_time is not None:
            try:
                ref = pd.Timestamp(last_bar_time)
                react = pd.Timestamp(node.last_reaction_time)
                if ref.tzinfo is None:
                    ref = ref.tz_localize("UTC")
                if react.tzinfo is None:
                    react = react.tz_localize("UTC")
                node.last_reaction_hours_ago = (ref - react).total_seconds() / 3600.0
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Zone extraction
# ---------------------------------------------------------------------------

def extract_delta_nodes(
    ticks: pd.DataFrame,
    price_levels: dict,           # {"vah": float, "val": float, "poc": float}
    node_delta_percentile: int = 50,  # Nth percentile of session abs_delta used as threshold
    cluster_ticks: int = 20,      # zone width in NQ ticks (0.25 pts each)
    structural_ticks: int = 4,
) -> list[DeltaNode]:
    """
    Extract significant delta nodes from a tick DataFrame.

    Parameters
    ----------
    ticks : pd.DataFrame
        Must contain columns: price (float $), size (int/float), aggressor
        ("buy" | "sell"), session_date (date).  Index must be a DatetimeIndex
        (UTC) representing trade timestamps.
    node_delta_percentile : int
        Percentile of abs_delta distribution used as the qualifying threshold.
        Default 50 (median): takes the top 50% of price levels by delta magnitude.
        Session-adaptive — automatically scales with session volume.
    price_levels : dict
        {"vah": float, "val": float, "poc": float} for the session.
    cluster_ticks : int
        Full width of the zone in ticks.  Default 20 (= ±10 ticks = ±2.5 pts).
    structural_ticks : int
        A zone is structural if any part of [price_low, price_high] comes
        within this many ticks of VAH, VAL, or POC.

    Returns
    -------
    list[DeltaNode]
        Sorted by abs_delta descending.
    """
    if ticks.empty:
        return []

    half_width = (cluster_ticks / 2) * _TICK_SIZE   # in price points
    tolerance = structural_ticks * _TICK_SIZE

    vah = price_levels.get("vah")
    val = price_levels.get("val")
    poc = price_levels.get("poc")

    df = ticks.copy()
    df["price_level"] = (df["price"] / _TICK_SIZE).round() * _TICK_SIZE

    buy_mask = df["aggressor"] == "buy"
    sell_mask = df["aggressor"] == "sell"

    buy_by_level = df.loc[buy_mask].groupby("price_level")["size"].sum().rename("buy_vol")
    sell_by_level = df.loc[sell_mask].groupby("price_level")["size"].sum().rename("sell_vol")

    levels = pd.DataFrame({"buy_vol": buy_by_level, "sell_vol": sell_by_level}).fillna(0.0)
    levels["delta"] = levels["buy_vol"] - levels["sell_vol"]
    levels["abs_delta"] = levels["delta"].abs()

    # Timestamp = last trade at each level
    last_ts = df.groupby("price_level").apply(lambda g: g.index.max(), include_groups=False)
    levels["timestamp"] = last_ts

    # session_date = first non-null per level
    first_session = df.groupby("price_level")["session_date"].first()
    levels["session_date"] = first_session

    # Compute session-adaptive threshold (Nth percentile of abs_delta distribution)
    delta_threshold = float(np.percentile(levels["abs_delta"].values, node_delta_percentile))
    # Filter to seed-eligible levels (|delta| >= threshold), then sort by price
    seed_mask = levels["abs_delta"] >= delta_threshold
    seed_prices = set(levels.index[seed_mask].tolist())

    # All levels sorted by price — used for zone summation
    all_prices_sorted = sorted(levels.index.tolist())

    if not seed_prices:
        return []

    # Greedy forward walk
    claimed: set[float] = set()
    nodes: list[DeltaNode] = []

    for level_price in all_prices_sorted:
        if level_price not in seed_prices:
            continue
        if level_price in claimed:
            continue

        # Define zone bounds
        zone_low = level_price - half_width
        zone_high = level_price + half_width

        # Collect all unclaimed levels within the zone
        zone_levels = [
            p for p in all_prices_sorted
            if zone_low <= p <= zone_high and p not in claimed
        ]

        if not zone_levels:
            continue

        # Sum delta and find peak-|delta| level within zone
        zone_delta = levels.loc[zone_levels, "delta"].sum()
        zone_abs_delta = abs(zone_delta)
        peak_price = float(levels.loc[zone_levels, "abs_delta"].idxmax())

        # Re-centre zone bounds on the peak-delta level
        zone_low  = peak_price - half_width
        zone_high = peak_price + half_width

        # Latest timestamp across zone levels
        zone_timestamp = levels.loc[zone_levels, "timestamp"].max()
        zone_session = levels.loc[zone_levels, "session_date"].iloc[0]

        # Structural check: does [zone_low, zone_high] come within tolerance of
        # any structural reference?
        is_structural = False
        for ref in (vah, val, poc):
            if ref is None:
                continue
            # Overlap condition: zone range and [ref - tol, ref + tol] intersect
            if zone_low <= ref + tolerance and zone_high >= ref - tolerance:
                is_structural = True
                break

        nodes.append(
            DeltaNode(
                price=float(peak_price),
                price_low=float(zone_low),
                price_high=float(zone_high),
                delta=float(zone_delta),
                abs_delta=float(zone_abs_delta),
                timestamp=zone_timestamp,
                session_date=zone_session,
                is_structural=is_structural,
            )
        )

        # Mark all levels in window as claimed
        for p in zone_levels:
            claimed.add(p)

    nodes.sort(key=lambda n: n.abs_delta, reverse=True)
    return nodes


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(session_date: date_type) -> Path:
    return _DELTA_CACHE_DIR / f"{session_date}_nodes.pkl"


def load_nodes_cached(
    session_date: date_type,
    ticks: pd.DataFrame | None = None,
    node_delta_percentile: int = 50,
    price_levels: dict | None = None,
    cluster_ticks: int = _DEFAULT_CLUSTER_TICKS,
    structural_ticks: int = 4,
    force_rebuild: bool = False,
    rth_bars: list | None = None,
) -> list[DeltaNode]:
    """
    Load delta nodes for a session, building from ticks if cache is absent.

    Cache policy
    ------------
    - The cache stores nodes built with the default cluster_ticks (20).
    - If cluster_ticks != default, the cache is bypassed entirely and nodes
      are recomputed from ticks.  A note is printed indicating this.
    - If cache exists and cluster_ticks is default and force_rebuild is False:
      load from pickle.
    - If ticks is None and cache is absent: return [] with a warning.

    rth_bars : list[RangeBar], optional
        RTH range bars for the session. When provided, compute_node_reactions()
        is called after building nodes to track reactions within this session.
        Has no effect when loading from cache (cache already has reaction data).
    """
    non_default_width = cluster_ticks != _DEFAULT_CLUSTER_TICKS

    if non_default_width:
        print(
            f"Note [{session_date}]: cluster_ticks={cluster_ticks} is non-default "
            f"(default={_DEFAULT_CLUSTER_TICKS}); skipping cache and recomputing."
        )
        if ticks is None:
            print(f"Warning [{session_date}]: non-default cluster width but no ticks provided; returning [].")
            return []
        nodes = extract_delta_nodes(
            ticks=ticks,
            price_levels=price_levels if price_levels is not None else {},
            node_delta_percentile=node_delta_percentile,
            cluster_ticks=cluster_ticks,
            structural_ticks=structural_ticks,
        )
        if rth_bars:
            compute_node_reactions(nodes, rth_bars)
        return nodes

    path = _cache_path(session_date)

    if path.exists() and not force_rebuild:
        with open(path, "rb") as f:
            return pickle.load(f)

    if ticks is None:
        print(f"Warning [{session_date}]: no cache found and no ticks provided; returning [].")
        return []

    nodes = extract_delta_nodes(
        ticks=ticks,
        price_levels=price_levels if price_levels is not None else {},
        node_delta_percentile=node_delta_percentile,
        cluster_ticks=cluster_ticks,
        structural_ticks=structural_ticks,
    )

    if rth_bars:
        compute_node_reactions(nodes, rth_bars)

    _DELTA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(nodes, f)

    return nodes


# ---------------------------------------------------------------------------
# Range loader + RTH filtering
# ---------------------------------------------------------------------------

def load_nodes_for_range(
    start_date: date_type,
    end_date: date_type,
) -> list[DeltaNode]:
    """
    Load all cached delta nodes for sessions in [start_date, end_date].

    Only loads from cache (default cluster_ticks=20); does not build.
    Skips missing dates silently.
    Returns combined list sorted by timestamp ascending.
    """
    all_nodes: list[DeltaNode] = []
    for dt in pd.date_range(start=start_date, end=end_date, freq="D"):
        path = _cache_path(dt.date())
        if not path.exists():
            continue
        with open(path, "rb") as f:
            nodes: list[DeltaNode] = pickle.load(f)
        all_nodes.extend(nodes)

    all_nodes.sort(key=lambda n: n.timestamp)
    return all_nodes


def filter_nodes_prior_hours(
    nodes: list[DeltaNode],
    rth_open: datetime,
    lookback_hours: float,
) -> list[DeltaNode]:
    """
    Return nodes whose timestamp falls within the prior N hours of rth_open.
    rth_open should be tz-aware if node timestamps are tz-aware.
    """
    cutoff = rth_open - pd.Timedelta(hours=lookback_hours)
    return [n for n in nodes if cutoff <= n.timestamp < rth_open]


def filter_nodes_touched(
    nodes: list[DeltaNode],
    rth_bar_highs: pd.Series,
    rth_bar_lows: pd.Series,
) -> list[DeltaNode]:
    """
    Return nodes whose zone [price_low, price_high] overlaps any price level
    visited during the RTH session.

    A node is "touched" if there exists at least one bar where:
        bar_low <= node.price_high  AND  bar_high >= node.price_low

    Parameters
    ----------
    nodes : list[DeltaNode]
    rth_bar_highs : pd.Series
        High prices for each RTH range bar (same units as node prices).
    rth_bar_lows : pd.Series
        Low prices for each RTH range bar.
    """
    if not nodes or rth_bar_highs.empty:
        return []

    bar_highs = rth_bar_highs.values
    bar_lows = rth_bar_lows.values

    touched = []
    for node in nodes:
        overlaps = np.any((bar_lows <= node.price_high) & (bar_highs >= node.price_low))
        if overlaps:
            touched.append(node)

    return touched


# ---------------------------------------------------------------------------
# Proximity lookup
# ---------------------------------------------------------------------------

def nodes_at_price(
    nodes: list[DeltaNode],
    price: float,
    tolerance_ticks: int = 4,
    max_age_hours: float | None = None,
    current_time: datetime | None = None,
) -> list[DeltaNode]:
    """
    Find all nodes whose zone [price_low, price_high] contains price,
    or whose peak price is within tolerance_ticks of the given price.

    Optionally filter by max_age_hours (requires current_time).
    """
    tolerance = tolerance_ticks * _TICK_SIZE
    result = [
        n for n in nodes
        if (n.price_low - tolerance) <= price <= (n.price_high + tolerance)
    ]

    if max_age_hours is not None and current_time is not None:
        result = [
            n for n in result
            if (current_time - n.timestamp).total_seconds() / 3600.0 <= max_age_hours
        ]

    return result
