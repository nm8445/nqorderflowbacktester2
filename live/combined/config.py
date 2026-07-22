"""Central configuration for the live combined-strategy system."""
from pathlib import Path

# ============ Data sources ============
DATABENTO_DATASET = "GLBX.MDP3"
DATABENTO_SYMBOL = "NQ.v.0"
DATABENTO_SCHEMA = "mbp-1"
DATABENTO_STYPE = "continuous"

# Local data archives (read for warm-start)
# Rolling 15-day cache for live warm-start (auto-managed, see rolling_cache.py).
# This is the PREFERRED source for warm-start. Falls back to the research archive
# below if the cache is missing days.
LIVE_WARMSTART_CACHE_DIR = Path("D:/trading_pythonbacktest_data/live_warmstart_cache")

# Research archive (full multi-year history). NOT pruned. Used as fallback only.
TIMEBARS_5MIN_DIRS = [
    Path("D:/trading_pythonbacktest_data/timebars_5min_5yr"),
    Path("D:/trading_pythonbacktest_data/timebars_5min"),
]
NQ_1MIN_BARS = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")

# ============ Live state ============
STATE_DIR = Path(__file__).resolve().parent / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
SETTLES_JSON = STATE_DIR / "daily_settles.json"
POSITION_STATE_JSON = STATE_DIR / "positions.json"
SESSION_LOG = STATE_DIR / "session.log"

# ============ Time / instrument ============
ET_TZ = "America/New_York"
NQ_TICK_SIZE = 0.25
NQ_POINT_VALUE = 20.0   # $/point per 1 NQ contract
PRICE_SCALE = 1_000_000_000  # Databento MBP-1 price encoding

# ============ Bar timeframes ============
BAR_5MIN_SECS = 300
BAR_20MIN_SECS = 1200
# 20-min bars anchored to midnight ET (so 19:00 lands on a bar boundary)
BAR_20MIN_ANCHOR_HOUR = 0  # midnight ET
BAR_20MIN_ANCHOR_MIN = 0

# ============ Settle capture ============
SETTLE_HOUR_ET = 16   # 16:00 ET = QQQ/SPX/NDX settle reference for gamma pipeline
SETTLE_MIN_ET = 0

# ============ Warm-start ============
# How many trading days of historical bars to feed indicators before going live.
# RV needs ~80 5-min bars for the kernel + rolling windows; B2 needs ATR seed; OD needs ATR seed.
# 15 trading days gives plenty of headroom for all three.
WARM_START_TRADING_DAYS = 15
