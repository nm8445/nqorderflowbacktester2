"""Capture daily 16:00 ET NQ close and persist to daily_settles.json.

The gamma builder needs the NQ price at 16:00 ET each day to map QQQ/NDX levels
to NQ price space. This module hooks into the 1-min bar stream (or any minute-level
event) and snapshots the close at the configured settle time.

The JSON file format:
    {
      "2026-05-15": 23456.75,
      "2026-05-16": 23489.50,
      ...
    }

Used by the daily gamma refresh job to look up today's NQ settle.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import pandas as pd

from live.combined.config import (
    SETTLES_JSON, SETTLE_HOUR_ET, SETTLE_MIN_ET, ET_TZ,
)
from live.combined.bar_builder import Bar


def _load_existing() -> dict:
    if not SETTLES_JSON.exists():
        return {}
    try:
        return json.loads(SETTLES_JSON.read_text())
    except Exception:
        return {}


def _persist(d: dict) -> None:
    SETTLES_JSON.parent.mkdir(parents=True, exist_ok=True)
    SETTLES_JSON.write_text(json.dumps(d, indent=2, sort_keys=True))


class SettleRecorder:
    """Subscribe me to a 5-min bar builder. I'll record the bar that closes
    at 16:00 ET (= the 15:55-16:00 5-min bar)."""

    def __init__(self):
        self.settles = _load_existing()
        self._last_captured: dt.date | None = None

    def on_bar(self, bar: Bar) -> None:
        # Bar close_time should be 16:00 ET exactly. Check.
        ct = bar.close_time
        if ct.tz is None:
            ct = ct.tz_localize("UTC").tz_convert(ET_TZ)
        else:
            ct = ct.tz_convert(ET_TZ)
        if ct.hour != SETTLE_HOUR_ET or ct.minute != SETTLE_MIN_ET:
            return
        date_key = ct.date().isoformat()
        # Avoid double-capture in case of replay or restart
        if date_key in self.settles and self._last_captured == ct.date():
            return
        self.settles[date_key] = float(bar.close)
        self._last_captured = ct.date()
        _persist(self.settles)
        print(f"[settle_recorder] captured {date_key} NQ close = {bar.close:.2f}")

    def get(self, date: dt.date) -> float | None:
        return self.settles.get(date.isoformat())
