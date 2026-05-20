"""
Rough Vol Model — confirmed original parameters for 15-minute bar backtest.
"""

from pathlib import Path

# ── Data paths ──────────────────────────────────────────────────────────────
CACHE_DIR = Path("D:/trading_pythonbacktest_data")
TIMEBAR_CACHE_DIR = CACHE_DIR / "timebars_15min_from_ticks"
BAR_CACHE_FILE = CACHE_DIR / "15min_bars.parquet"

# ── Rough Vol Model ─────────────────────────────────────────────────────────
H          = 0.10       # Hurst exponent
KERNEL_LEN = 100        # rough driver memory in bars
ETA        = 1.0        # vol of vol
V0         = 0.0001     # baseline variance
NORM_LEN   = 200        # return normalization lookback
Z_LOOKBACK = 100        # vol z-score lookback

# ── Entry/Exit thresholds ───────────────────────────────────────────────────
HIGH_Z     = 1.0        # enter when z_vol > HIGH_Z
LOW_Z      = -1.0       # EXIT when z_vol < LOW_Z (primary exit)

# ── Trend filter and risk ───────────────────────────────────────────────────
EMA_LEN    = 50         # trend filter EMA
ATR_LEN    = 14         # ATR period
ATR_SL     = 2.0        # stop loss ATR multiple (safety net only)
ATR_TP     = 3.0        # take profit ATR multiple (safety net only)

# ── Session ─────────────────────────────────────────────────────────────────
SESSION_START = 9.0      # 9:00 AM ET
SESSION_END   = 16.0     # 3:30 PM ET
TIMEZONE      = "America/New_York"

# ── Daily limits ────────────────────────────────────────────────────────────
MAX_TRADES    = 999      # no daily trade limit
MAX_LOSS      = -2500    # max daily loss in dollars

# ── Martingale position sizing ──────────────────────────────────────────────
USE_MARTINGALE      = True
MARTINGALE_STREAK   = 2      # consecutive losses before multiplying
BASE_CONTRACTS      = 1      # starting position size
MARTINGALE_MULTIPLE = 1.5    # multiply size by this after streak
MAX_DOUBLES         = 4      # maximum number of multiplications
RESET_ON_SESSION    = True   # reset martingale counter each new day

# ── Fixed ───────────────────────────────────────────────────────────────────
POINT_VALUE  = 20.0
TICK_SIZE    = 0.25
