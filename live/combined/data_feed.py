"""Databento live tick feed. Subscribes to NQ MBP-1 and dispatches trades to a callback.

Databento MBP-1 messages have:
  - action: 'T' (Trade), 'A' (Add), 'C' (Cancel), 'M' (Modify), 'F' (Fill), 'R' (Clear)
  - side:   'B' (bid lifted = buyer aggressor), 'A' (ask hit = seller aggressor), 'N' (unknown)
  - price:  int64, divide by PRICE_SCALE (1e9) to get dollars
  - size:   int
  - ts_recv: nanoseconds since epoch (UTC)

We forward only TRADE messages (action='T') to the tick callback.

Usage:
    from live.combined.data_feed import DatabentoLiveFeed
    feed = DatabentoLiveFeed(on_tick=my_callback)
    feed.start()  # blocking — runs until interrupted

    # In my_callback(ts_et, price, size, side): aggregate into bars
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable
import pandas as pd
from dotenv import load_dotenv

from live.combined.config import (
    DATABENTO_DATASET, DATABENTO_SYMBOL, DATABENTO_SCHEMA, DATABENTO_STYPE,
    PRICE_SCALE, ET_TZ,
)

log = logging.getLogger(__name__)
load_dotenv()


class DatabentoLiveFeed:
    """Live tick subscriber using Databento Live API."""

    def __init__(self, on_tick: Callable[[pd.Timestamp, float, int, str], None]):
        self.on_tick = on_tick
        self.client = None
        self._stopped = False
        self.n_ticks = 0
        self.n_trades = 0

        api_key = os.getenv("DATABENTO_API_KEY")
        if not api_key:
            raise RuntimeError("DATABENTO_API_KEY not in .env")
        self._api_key = api_key

    def start(self) -> None:
        """Connect and stream. Blocking — runs until self.stop() or exception."""
        import databento as db
        log.info(f"[data_feed] connecting to Databento live: "
                 f"{DATABENTO_DATASET} {DATABENTO_SYMBOL} {DATABENTO_SCHEMA}")
        self.client = db.Live(key=self._api_key)
        self.client.subscribe(
            dataset=DATABENTO_DATASET,
            schema=DATABENTO_SCHEMA,
            stype_in=DATABENTO_STYPE,
            symbols=[DATABENTO_SYMBOL],
        )
        # NOTE: do NOT call self.client.start() — the SDK starts on iteration.
        # Calling both raises "Cannot start iteration after streaming has started".
        log.info("[data_feed] subscribed, streaming...")

        try:
            for msg in self.client:
                if self._stopped:
                    break
                self.n_ticks += 1
                # MBP-1 has many message types. We only act on trades.
                # Trade record has 'action' = 'T' in v3 API.
                action = getattr(msg, "action", None)
                if action is None:
                    # Could be SystemMsg or ErrorMsg; skip
                    continue
                # action stored as int (84='T') or bytes (b'T') depending on version
                if isinstance(action, (bytes, bytearray)):
                    action_chr = action.decode("ascii", "ignore")
                elif isinstance(action, int):
                    action_chr = chr(action)
                else:
                    action_chr = str(action)
                if action_chr != "T":
                    continue

                # Extract trade fields
                try:
                    price_raw = msg.price
                    size = int(msg.size)
                    side_raw = msg.side
                    ts_recv_ns = msg.ts_recv  # nanoseconds since epoch
                except AttributeError:
                    continue

                price = float(price_raw) / PRICE_SCALE
                if isinstance(side_raw, (bytes, bytearray)):
                    side = side_raw.decode("ascii", "ignore")
                elif isinstance(side_raw, int):
                    side = chr(side_raw)
                else:
                    side = str(side_raw)
                if side not in ("A", "B", "N"):
                    side = "N"

                ts_et = pd.Timestamp(ts_recv_ns, unit="ns", tz="UTC").tz_convert(ET_TZ)

                self.n_trades += 1
                try:
                    self.on_tick(ts_et, price, size, side)
                except Exception as e:
                    log.exception(f"[data_feed] on_tick callback error: {e}")
        except KeyboardInterrupt:
            log.info("[data_feed] interrupted by user")
        finally:
            self.stop()

    def stop(self) -> None:
        self._stopped = True
        if self.client is not None:
            try:
                self.client.stop()
            except Exception:
                pass
            self.client = None
        log.info(f"[data_feed] stopped. Total ticks={self.n_ticks}  trades={self.n_trades}")
