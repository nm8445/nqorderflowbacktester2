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
# How often the contract-watch thread checks for a CME session boundary (to roll the feed's contract).
CONTRACT_WATCH_INTERVAL_SEC = 60
# Liveness watchdog: if no new gateway messages for this long DURING AN ACTIVE SESSION, the Live stream
# is dead (gateway stalled with the socket still open -> the reader blocks forever). Force a reconnect.
WATCHDOG_INTERVAL_SEC = 15
STALE_RECONNECT_SEC = 45


def _session_date_now() -> "object":
    """CME session date: if now >= 18:00 ET the session belongs to the NEXT calendar day. Used to roll
    the feed's contract at the session boundary (where the engines reset) rather than mid-session."""
    now = pd.Timestamp.now(tz=ET_TZ)
    return (now + pd.Timedelta(days=1)).date() if now.hour >= 18 else now.date()


def _market_expected_active() -> bool:
    """True when NQ Globex should be delivering ticks, so a frozen feed = a dead stream (not a quiet
    market). NQ trades Sun 18:00 ET -> Fri 17:00 ET with a daily 17:00-18:00 ET maintenance halt.
    The watchdog stays quiet outside these windows so it never false-fires during legit no-tick gaps."""
    now = pd.Timestamp.now(tz=ET_TZ)
    dow, hour = now.dayofweek, now.hour          # dow: 0=Mon .. 6=Sun
    if dow == 5:                                  # Saturday: closed
        return False
    if dow == 6:                                  # Sunday: only after the 18:00 reopen
        return hour >= 18
    if dow == 4 and hour >= 17:                   # Friday: closed for the weekend at 17:00
        return False
    if 17 <= hour < 18:                           # Mon-Thu daily maintenance halt
        return False
    return True


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
        # Subscribe to the VOLUME-LEAD contract (resolved live, e.g. 'NQU6'), not the lagging continuous
        # NQ.v.0 — so signals are computed on the SAME liquid contract the executor stamps onto orders.
        # Falls back to NQ.v.0/continuous if the resolver is unavailable, so the feed always starts.
        from live.combined.active_contract import feed_symbol
        sym, stype = feed_symbol()
        self._current_symbol, self._current_stype = sym, stype
        self._reconnect_requested = False
        self._pending_symbol = self._pending_stype = None
        log.info(f"[data_feed] connecting to Databento live: "
                 f"{DATABENTO_DATASET} {sym} ({stype}) {DATABENTO_SCHEMA}")
        self.client = db.Live(key=self._api_key)
        self._subscribe(self.client, sym, stype)
        log.info(f"[data_feed] subscribed to {sym} ({stype}), starting reader + consumer + monitor "
                 f"+ contract-watch threads...")

        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="DBN-reader", daemon=True)
        self._consumer_thread = threading.Thread(
            target=self._consumer_loop, name="DBN-consumer", daemon=True)
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, name="DBN-monitor", daemon=True)
        self._contract_thread = threading.Thread(
            target=self._contract_watch_loop, name="DBN-contract", daemon=True)
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="DBN-watchdog", daemon=True)

        self._reader_thread.start()
        self._consumer_thread.start()
        self._monitor_thread.start()
        self._contract_thread.start()
        self._watchdog_thread.start()

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
        # Outer loop so the reader can SWAP to a new Live client when the contract-watch thread flags a
        # session-boundary roll (it stops the old client to break the `for msg in client` iteration).
        while not self._stopped:
            client = self.client
            try:
                for msg in client:
                    if self._stopped or self._reconnect_requested:
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
            if self._stopped:
                break
            # Reconnect rather than silently exit. Either a roll/watchdog requested it (pending already
            # set), OR the stream ended on its own (gateway closed / StopIteration) -> reconnect to the
            # SAME contract so a clean stream-end also self-heals instead of killing the feed.
            if not self._reconnect_requested:
                log.warning("[data_feed] live stream ended with no roll pending — reconnecting to "
                            f"same contract {self._current_symbol}")
                self._pending_symbol, self._pending_stype = self._current_symbol, self._current_stype
            time.sleep(1.0)                       # small backoff so repeated failures don't hot-loop
            self._perform_reconnect()
            continue
        log.info(f"[data_feed] reader exiting (msgs={self.n_msgs_received}, "
                 f"trades={self.n_trades_enqueued}, dropped={self.n_dropped})")

    def _subscribe(self, client, sym: str, stype: str) -> None:
        client.subscribe(dataset=DATABENTO_DATASET, schema=DATABENTO_SCHEMA,
                         stype_in=stype, symbols=[sym])

    def _request_roll(self, sym: str, stype: str) -> None:
        """Flag a reconnect to (sym, stype) and stop the current client to break the reader's blocked
        `for msg in client` iteration so it reconnects. Used by both the contract-watch (real roll) and
        the liveness watchdog (reconnect to the SAME contract after a stall)."""
        self._pending_symbol, self._pending_stype = sym, stype
        self._reconnect_requested = True
        try:
            self.client.stop()
        except Exception:
            pass

    def _watchdog_loop(self) -> None:
        """Liveness guard. The reader blocks on `for msg in client` when the gateway goes silent with
        the socket still open (no exception, no StopIteration) -> the feed freezes with received frozen.
        Nothing else detects this (the contract-watch only reconnects on a contract CHANGE). If received
        stops advancing for STALE_RECONNECT_SEC during an active session, force a reconnect to the same
        contract -- stop() breaks the blocked iterator, the reader hits _perform_reconnect()."""
        last_n = self.n_msgs_received
        last_change = time.monotonic()
        while not self._stopped:
            time.sleep(WATCHDOG_INTERVAL_SEC)
            n = self.n_msgs_received
            if n != last_n:
                last_n, last_change = n, time.monotonic()
                continue
            stale = time.monotonic() - last_change
            if (stale >= STALE_RECONNECT_SEC and not self._reconnect_requested
                    and not self._stopped and _market_expected_active()):
                log.error(f"[data_feed] STALL: no gateway msgs for {stale:.0f}s (received={n:,}) "
                          f"during active session -> forcing reconnect to {self._current_symbol}")
                self._request_roll(self._current_symbol, self._current_stype)
                last_change = time.monotonic()   # hold off re-firing until the reconnect gets a chance

    def _contract_watch_loop(self) -> None:
        """At each CME session boundary (18:00 ET), re-resolve the volume-lead contract and, if it
        changed, roll the live subscription to it — so a continuously-running feed follows the quarterly
        roll with NO process restart. Rolling at the session boundary (not mid-session) keeps the
        inherent ~spread jump aligned with the engines' own session reset (B2 re-bases its overnight
        range there). Force-resolves (refresh_now) so it works even with no executor refreshing the cache."""
        from live.combined.active_contract import refresh_now
        last_session = _session_date_now()
        while not self._stopped:
            for _ in range(CONTRACT_WATCH_INTERVAL_SEC):   # interruptible sleep
                if self._stopped:
                    return
                time.sleep(1)
            sess = _session_date_now()
            if sess == last_session:
                continue                       # still the same session — nothing to do
            last_session = sess
            try:
                info = refresh_now()           # force a fresh volume resolve at the boundary
            except Exception as e:
                log.warning(f"[data_feed] contract-watch resolve failed: {e}")
                continue
            new_sym = info.get("raw") if info else None
            if new_sym and new_sym != self._current_symbol:
                log.info(f"[data_feed] SESSION ROLL: volume-lead {self._current_symbol} -> {new_sym} "
                         f"{info.get('vols', {})} — reconnecting feed at session boundary")
                self._request_roll(new_sym, "raw_symbol")

    def _perform_reconnect(self) -> None:
        """Tear down the old Live client and subscribe a fresh one to the pending (rolled) contract.
        Runs on the reader thread after the watch thread flagged a roll."""
        import databento as db
        self._reconnect_requested = False
        try:
            self.client.stop()
        except Exception:
            pass
        sym, stype = self._pending_symbol, self._pending_stype
        try:
            self.client = db.Live(key=self._api_key)
            self._subscribe(self.client, sym, stype)
            self._current_symbol, self._current_stype = sym, stype
            log.info(f"[data_feed] reconnected — now streaming {sym} ({stype})")
        except Exception as e:
            log.exception(f"[data_feed] reconnect FAILED ({sym}): {e} — stopping feed")
            self._stopped = True

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
