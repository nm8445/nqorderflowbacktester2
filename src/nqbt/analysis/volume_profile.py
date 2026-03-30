"""
Developing Volume Profile for NQ orderflow analysis.

Builds a price-by-volume profile from normalized tick data with:
  - Configurable level size (default 10 ticks = 2.5 points for NQ)
  - Buy/sell volume and delta per level
  - POC (Point of Control) — highest volume level
  - Value Area (default 70%) — VAH and VAL
  - Designed for developing profiles: call build() on a growing tick slice

Session convention:
  - Daily profile runs from 6pm ET previous day through the current time
  - For backtesting, slice ticks to the desired window before calling build()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

TICK_SIZE = 0.25


@dataclass
class ProfileLevel:
    """Volume at a single price level."""
    price: float      # lower bound of level in points
    buy_vol: int = 0
    sell_vol: int = 0

    @property
    def total_vol(self) -> int:
        return self.buy_vol + self.sell_vol

    @property
    def delta(self) -> int:
        return self.buy_vol - self.sell_vol


@dataclass
class VolumeProfile:
    """
    Completed volume profile snapshot.

    Attributes
    ----------
    levels      : dict mapping level price → ProfileLevel
    poc         : price of the highest-volume level
    vah         : Value Area High (top of the highest level in the value area)
    val         : Value Area Low (bottom of the lowest level in the value area)
    total_volume: total contracts traded in this profile window
    """
    levels: dict
    poc: float
    vah: float
    val: float
    total_volume: int
    ticks_per_level: int
    value_area_pct: float

    @classmethod
    def build(
        cls,
        ticks: pd.DataFrame,
        ticks_per_level: int = 10,
        value_area_pct: float = 0.70,
    ) -> VolumeProfile:
        """
        Build a volume profile from a normalized tick DataFrame.

        Parameters
        ----------
        ticks : pd.DataFrame
            Output of normalizer.normalize() — must have price, size, aggressor columns.
        ticks_per_level : int
            Number of NQ ticks per profile level. Default 10 = 2.5 points.
        value_area_pct : float
            Fraction of total volume to include in the value area. Default 0.70.

        Returns
        -------
        VolumeProfile
        """
        level_size = round(ticks_per_level * TICK_SIZE, 10)

        if ticks.empty:
            return cls(
                levels={}, poc=0.0, vah=0.0, val=0.0,
                total_volume=0, ticks_per_level=ticks_per_level,
                value_area_pct=value_area_pct,
            )

        # Vectorized level assignment — avoids iterrows() on large tick slices
        prices    = ticks["price"].to_numpy(dtype=float)
        sizes     = ticks["size"].to_numpy(dtype=int)
        aggressors = np.asarray(ticks["aggressor"])

        level_indices = (np.round(prices / TICK_SIZE).astype(int)) // ticks_per_level
        level_prices  = np.round(level_indices * ticks_per_level * TICK_SIZE, 4)
        buy_mask      = aggressors == "buy"

        agg = pd.DataFrame({
            "level_price": level_prices,
            "buy_vol":     np.where(buy_mask, sizes, 0),
            "sell_vol":    np.where(~buy_mask, sizes, 0),
        }).groupby("level_price", sort=False).sum()

        levels: dict[float, ProfileLevel] = {
            lp: ProfileLevel(price=lp, buy_vol=int(row.buy_vol), sell_vol=int(row.sell_vol))
            for lp, row in agg.iterrows()
        }

        if not levels:
            return cls(
                levels={}, poc=0.0, vah=0.0, val=0.0,
                total_volume=0, ticks_per_level=ticks_per_level,
                value_area_pct=value_area_pct,
            )

        total_volume = sum(lv.total_vol for lv in levels.values())
        sorted_prices = sorted(levels.keys())

        # POC = level with highest total volume
        poc = max(levels, key=lambda p: levels[p].total_vol)
        poc_idx = sorted_prices.index(poc)

        # Value Area: expand outward from POC until we capture value_area_pct
        va_target = total_volume * value_area_pct
        va_volume = levels[poc].total_vol
        lo_idx = poc_idx
        hi_idx = poc_idx

        while va_volume < va_target:
            can_go_up = hi_idx + 1 < len(sorted_prices)
            can_go_dn = lo_idx - 1 >= 0

            if not can_go_up and not can_go_dn:
                break

            up_vol = levels[sorted_prices[hi_idx + 1]].total_vol if can_go_up else -1
            dn_vol = levels[sorted_prices[lo_idx - 1]].total_vol if can_go_dn else -1

            # Expand toward the higher-volume neighbor (ties go upward)
            if up_vol >= dn_vol:
                hi_idx += 1
                va_volume += up_vol
            else:
                lo_idx -= 1
                va_volume += dn_vol

        vah = sorted_prices[hi_idx] + level_size  # top edge of highest level
        val = sorted_prices[lo_idx]                # bottom edge of lowest level

        return cls(
            levels=levels,
            poc=poc,
            vah=vah,
            val=val,
            total_volume=total_volume,
            ticks_per_level=ticks_per_level,
            value_area_pct=value_area_pct,
        )

    def to_df(self) -> pd.DataFrame:
        """Return profile as a DataFrame sorted by price descending."""
        rows = [
            {
                "price":     lv.price,
                "buy_vol":   lv.buy_vol,
                "sell_vol":  lv.sell_vol,
                "total_vol": lv.total_vol,
                "delta":     lv.delta,
                "poc":       lv.price == self.poc,
                "in_va":     self.val <= lv.price < self.vah,
            }
            for lv in sorted(self.levels.values(), key=lambda l: l.price, reverse=True)
        ]
        return pd.DataFrame(rows).set_index("price")

    def summary(self) -> str:
        return (
            f"POC: {self.poc:.2f}  |  VAH: {self.vah:.2f}  |  VAL: {self.val:.2f}  |  "
            f"Total Vol: {self.total_volume:,}  |  Levels: {len(self.levels)}"
        )
