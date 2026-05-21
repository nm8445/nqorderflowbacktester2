"""Databento live tick feed. Subscribes to NQ MBP-1 and dispatches trades to a callback.

Architecture (post slow-client fix, 2026-05-21):
  Two threads with a bounded tick_queue between them:

    Databento gateway
          ↓ msg
    [reader thread] — parses 'T' messages only, queue.put_nowait the raw fields
          ↓ tick_queue (200K capacity)
    [consumer thread] — pops, tz-converts, calls on_tick callback
          ↓
    on_tick callback (bar_builder → engines → coordinator → NT8 queue)

  The reader thread does ~5 microseconds of work per tick. The gateway never
  sees a slow client because the reader cannot fall behind. If the consumer
  thread falls behind (queue full), we increment a local `n_dropped` counter
  rather than letting the gateway drop our records.

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
"""
from __future__ import annotations

import logging
import os
import queue
import threading
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

# Tick queue capacity — sized large enough to absorb minute-long stalls during
# cash open bursts. ~200K ticks ≈ 5 minutes of peak-rate NQ activity.
TICK_QUEUE_MAXSIZE = 200_000
# How often the monitor thread prints queue depth (seconds)
MONITOR_INTERVAL_SEC = 60


class DatabentoLiveFeed:
    """Live tick subscriber using Databento Live API.

    Internally runs two threads:
      - reader: pulls from gateway, parses 'T' messages, enqueues raw fields
      - consumer: drains queue, builds tz-aware ts, calls on_tick callback
    """

    def __init__(self, on_tick: Callable[[pd.Timestamp, float, int, str], None]):
        self.on_tick = on_tick
        self.client = None
        self._stopped = False

        # Queue between reader and consumer.
        # Tuples: (ts_recv_ns: int, price_raw: int, size: int, side_raw: bytes|int|str)
        self.tick_queue: queue.Queue = queue.Queue(maxsize=TICK_QUEUE_MAXSIZE)

        self._reader_thread: threading.Thread | None = None
        self._consumer_thread: threading.Thread | None = None
        self._monitor_thread: threading.Thread | None = None

        # Counters
        self.n_msgs_received = 0     # all messages from gateway (including non-trades)
        self.n_trades_enqueued = 0    # 'T' messages put on queue
        self.n_trades_processed = 0   # consumer drained
        self.n_dropped = 0            # consumer fell behind, queue was full
        self.n_callback_errors = 0
        self._max_qsize_seen = 0

        api_key = os.getenv("DATABENTO_API_KEY")
        if not api_key:
            raise RuntimeError("DATABENTO_API_KEY not in .env")
        self._api_key = api_key

    def start(self) -> None:
        """Connect and stream. Blocking — runs until self.stop() or KeyboardInterrupt.
        Spawns reader, consumer, and monitor threads; main thread joins on consumer."""
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
        log.info("[data_feed] subscribed, starting reader + consumer + monitor threads...")

        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="DBN-reader", daemon=True)
        self._consumer_thread = threading.Thread(
            target=self._consumer_loop, name="DBN-consumer", daemon=True)
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, name="DBN-monitor", daemon=True)

        self._reader_thread.start()
        self._consumer_thread.start()
        self._monitor_thread.start()

        try:
            # Block the main thread on the consumer. If consumer dies, we stop.
            # The reader will keep draining the gateway and dropping ticks
            # locally rather than letting the gateway drop them.
            self._consumer_thread.join()
        except KeyboardInterrupt:
            log.info("[data_feed] interrupted by user")
        finally:
            self.stop()

    def _reader_loop(self) -> None:
        """HOT PATH. Pull messages from Databento iterator, enqueue only trades.

        Keep this loop minimal — no tz conversions, no HTTP, no I/O. Just parse
        the action byte and put_nowait. If we fall behind the gateway, the
        gateway will start dropping. With this design, the reader stays in
        lockstep with the gateway and the queue absorbs consumer-side stalls.
        """
        try:
            for msg in self.client:
                if self._stopped:
                    break
                self.n_msgs_received += 1

                # Action may be int (84='T') or bytes (b'T') depending on SDK version
                action = getattr(msg, "action", None)
                if action is None:
                    continue
                if isinstance(action, int):
                    if action != 84:    # 'T'
                        continue
                elif isinstance(action, (bytes, bytearray)):
                    if action != b"T":
                        continue
                else:
                    if str(action) != "T":
                        continue

                # Pull raw fields (no conversion) and shove on queue
                try:
                    ts_recv_ns = msg.ts_recv
                    price_raw = msg.price
                    size_raw = msg.size
                    side_raw = msg.side
                except AttributeError:
                    continue

                try:
                    self.tick_queue.put_nowait(
                        (ts_recv_ns, price_raw, size_raw, side_raw))
                    self.n_trades_enqueued += 1
                except queue.Full:
                    # Consumer fell so far behind the queue is full.
                    # Drop locally and count. If this happens often, the engine
                    # itself is too slow and we need to profile #6 in the audit.
                    self.n_dropped += 1
        except Exception as e:
            log.exception(f"[data_feed] reader thread error: {e}")
        finally:
            log.info(f"[data_feed] reader exiting (msgs={self.n_msgs_received}, "
                     f"trades={self.n_trades_enqueued}, dropped={self.n_dropped})")

    def _consumer_loop(self) -> None:
        """Drain tick_queue. All tz conversions and the on_tick callback happen here."""
        while not self._stopped:
            try:
                ts_recv_ns, price_raw, size_raw, side_raw = self.tick_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Decode side
            if isinstance(side_raw, (bytes, bytearray)):
                side = side_raw.decode("ascii", "ignore")
            elif isinstance(side_raw, int):
                side = chr(side_raw)
            else:
                side = str(side_raw)
            if side not in ("A", "B", "N"):
                side = "N"

            price = float(price_raw) / PRICE_SCALE
            size = int(size_raw)
            ts_et = pd.Timestamp(ts_recv_ns, unit="ns", tz="UTC").tz_convert(ET_TZ)

            self.n_trades_processed += 1
            try:
                self.on_tick(ts_et, price, size, side)
            except Exception as e:
                self.n_callback_errors += 1
                log.exception(f"[data_feed] on_tick callback error: {e}")

    def _monitor_loop(self) -> None:
        """Periodic queue-depth + drop-count log. Runs forever until stopped."""
        while not self._stopped:
            # Sleep first so we don't print at startup with empty stats
            time.sleep(MONITOR_INTERVAL_SEC)
            qs = self.tick_queue.qsize()
            self._max_qsize_seen = max(self._max_qsize_seen, qs)
            log.info(
                f"[mon] tick_q={qs}/{TICK_QUEUE_MAXSIZE}  "
                f"peak={self._max_qsize_seen}  "
                f"received={self.n_msgs_received:,}  "
                f"enqueued={self.n_trades_enqueued:,}  "
                f"processed={self.n_trades_processed:,}  "
                f"dropped={self.n_dropped}  "
                f"cb_errors={self.n_callback_errors}"
            )
            if self.n_dropped > 0:
                log.error(
                    f"[mon] *** {self.n_dropped} ticks DROPPED locally (queue full). "
                    f"Engine consumer is falling behind. Investigate slow path."
                )

    def stop(self) -> None:
        self._stopped = True
        if self.client is not None:
            try:
                self.client.stop()
            except Exception:
                pass
            self.client = None
        # Threads are daemonic — they'll exit when main thread exits
        log.info(f"[data_feed] stopped. msgs={self.n_msgs_received}, "
                 f"trades_enqueued={self.n_trades_enqueued}, "
                 f"processed={self.n_trades_processed}, dropped={self.n_dropped}, "
                 f"peak_qsize={self._max_qsize_seen}")
