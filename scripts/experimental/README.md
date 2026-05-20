# Experimental Scripts

⚠️ **WARNING**: These scripts are experimental backups and archived tests.

## Contents

### `test_rough_vol_ema_original.py`
Original rough vol + EMA strategy (before martingale was added).

**Features:**
- Regime-based exits (exit when z_vol drops below HIGH_Z)
- 5x ATR emergency stops
- No position sizing (always 1 contract)

**Archived for reference** - The main version now has martingale added.

---

## Main Script

The active version is now in `scripts/test_rough_vol_ema.py` with martingale added.
