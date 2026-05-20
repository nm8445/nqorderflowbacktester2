# Cache Creation Scripts

Scripts for building various data caches to speed up backtesting.

## Scripts

### `build_daily_profile_cache.py` ⭐ NEW
**Creates:** Daily EOD volume profile cache  
**Session:** 6pm (prev day) to 4:59:59pm (current day)  
**Cache location:** `D:/trading_pythonbacktest_data/daily_profile_cache/`

**Metrics saved per day:**
- VAH (Value Area High)
- VAL (Value Area Low)
- POC (Point of Control)
- Total volume
- Total delta
- **Delta zones:**
  - 10 points above POC
  - 10 points below POC
  - Above VAH
  - Below VAH
  - Above VAL
  - Below VAL

**Usage:**
```python
python build_daily_profile_cache.py
```

### `build_signal_cache.py`
**Creates:** Range bar signals + refreshing volume profiles  
**Session:** 9:30am-11:00am (RTH only)  
**Cache location:** `D:/trading_pythonbacktest_data/signal_cache/`

**Contains:**
- Range bars (4-tick)
- Absorption signals
- 1-minute refreshing volume profiles
- Entry signals near VAL/VAH

### `build_signal_cache_full_session.py`
**Creates:** Range bar signals + refreshing volume profiles  
**Session:** 7pm-4:55pm (full session for prop firms)  
**Cache location:** `D:/trading_pythonbacktest_data/signal_cache_full_session/`

**Contains:**
- Same as `build_signal_cache.py` but for full session
- Used for prop firm challenge simulations

## Cache Locations

All caches stored at: `D:/trading_pythonbacktest_data/`

```
D:/trading_pythonbacktest_data/
├── daily_profile_cache/          # Daily EOD profiles (NEW)
├── signal_cache/                 # RTH signals (9:30am-11am)
├── signal_cache_full_session/    # Full session signals
└── timebars_5min/                # 5-min time bars
```

## Parallel Processing

All scripts use parallel processing by default (CPU count - 2 workers).

To disable parallel processing, edit the script and set:
```python
parallel=False
```
