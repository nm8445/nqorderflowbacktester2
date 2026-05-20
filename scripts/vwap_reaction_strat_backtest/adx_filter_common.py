"""
Shared ADX filter logic for all ADX-filtered scripts.
Pre-builds a minute-resolution ADX lookup table for fast merge_asof tagging.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
TIMEBAR_DIR = DATA_DIR / "timebars_5min"

ADX_PERIOD = 14
ADX_LOW = 15.0
ADX_HIGH = 30.0

LUNCH_START = "12:00"
LUNCH_END = "13:00"

ENTRY_CUTOFF = "16:00"
ENTRY_START = "19:10"


def is_lunch_hour(hour_min: str) -> bool:
    """Return True if hour_min (HH:MM) falls within 12:00-12:59 ET lunch break."""
    return LUNCH_START <= hour_min < LUNCH_END


def in_dead_zone(hour_min: str) -> bool:
    """Return True if HH:MM is in the 16:00-19:10 dead zone (no entries)."""
    return ENTRY_CUTOFF <= hour_min < ENTRY_START


def load_5min_bars(date_str):
    fmt = date_str.replace("-", "_")
    path = TIMEBAR_DIR / f"timebars_5min_{fmt}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        bars = pickle.load(f)
    rows = [{"timestamp": b["open_time"], "open": b["open"], "high": b["high"],
             "low": b["low"], "close": b["close"]} for b in bars]
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    # Index at close time (open + 5min) so lookups only see completed bars
    df.index = df.index + pd.Timedelta(minutes=5)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    return df


def compute_adx(df, period=ADX_PERIOD):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                     (low - close.shift()).abs()], axis=1).max(axis=1)
    up = high - high.shift()
    dn = low.shift() - low
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1/period, min_periods=period, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1/period, min_periods=period, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def build_adx_lookup(dates=None):
    """Build combined ADX Series indexed by 5-min bar timestamps across all dates."""
    if dates is None:
        from run_backtest import VWAP_REACTION_CACHE_DIR
        dates = sorted([f.stem for f in VWAP_REACTION_CACHE_DIR.glob("*.pkl")])

    all_rows = []
    for date_str in dates:
        df_5m = load_5min_bars(date_str)
        if df_5m is None or len(df_5m) <= ADX_PERIOD * 2:
            continue
        adx_s = compute_adx(df_5m)
        all_rows.append(pd.DataFrame({"adx": adx_s}, index=df_5m.index))

    if not all_rows:
        return pd.DataFrame(columns=["adx"])

    lookup = pd.concat(all_rows).sort_index()
    lookup.index.name = "timestamp"
    return lookup


def get_adx_at_time(entry_time, adx_lookup):
    """Get ADX value at or before entry_time using the pre-built lookup."""
    prior = adx_lookup[adx_lookup.index <= entry_time]
    if len(prior) == 0:
        return np.nan
    return prior.iloc[-1]["adx"]


def passes_adx_filter(adx_value):
    """Return True if ADX is in the 15-30 sweet spot."""
    if np.isnan(adx_value):
        return False
    return ADX_LOW <= adx_value < ADX_HIGH
