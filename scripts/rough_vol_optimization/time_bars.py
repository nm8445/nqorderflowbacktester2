"""
15-minute OHLC bar loader.

Loads pre-built parquet from build_15min_ohlc.py.
"""

import pandas as pd

import config_15min as cfg

ET = cfg.TIMEZONE


def build_15min_bars() -> pd.DataFrame:
    """Load 15-minute OHLC bars from parquet cache."""
    if not cfg.BAR_CACHE_FILE.exists():
        raise FileNotFoundError(
            f"No bar cache at {cfg.BAR_CACHE_FILE}. "
            "Run scripts/cache_creation_scripts/build_15min_ohlc.py first."
        )

    print(f"Loading cached 15-min bars from {cfg.BAR_CACHE_FILE}")
    df = pd.read_parquet(cfg.BAR_CACHE_FILE)
    if df.index.tz is None:
        df.index = df.index.tz_localize(ET)
    print(f"  {len(df):,} bars loaded")
    return df


if __name__ == "__main__":
    bars = build_15min_bars()
    print(f"\nLoaded {len(bars):,} bars")
    print(f"Date range: {bars.index[0]} to {bars.index[-1]}")
    print(f"Columns: {list(bars.columns)}")
