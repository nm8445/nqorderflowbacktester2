"""
Build VWAP reaction cache with 6-point zone (instead of 3).
Saves to vwap_reaction_cache_6pt/ to keep original 3pt cache intact.
"""

import sys
from pathlib import Path

# Patch the module-level constants before importing the builder
import build_vwap_reaction_cache as builder

# Override zone and cache dir
builder.VWAP_ZONE_POINTS = 6.0
builder.VWAP_REACTION_CACHE_DIR = builder.DATA_DIR / "vwap_reaction_cache_6pt"

print(f"Zone: +/-{builder.VWAP_ZONE_POINTS} pts")
print(f"Output: {builder.VWAP_REACTION_CACHE_DIR}")

builder.build_vwap_reaction_cache("2025-03-13", "2026-04-08")
