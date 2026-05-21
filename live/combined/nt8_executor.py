"""NT8 executor — sends tagged orders to the multi-strat NT8 addon.

Each strategy's entries/exits carry a unique tag like "RV_2026-05-19_094000_SHORT".
The NT8 addon tracks positions per tag and routes flatten commands to only
the matching position, leaving other strategies' positions untouched.

Heartbeat:
  Background thread sends POST /heartbeat every 10 sec. If NT8 misses 3
  consecutive heartbeats (30 sec stale), it auto-flattens all positions
  tagged by this Python instance (scoped failsafe).

Per-strategy execution model:
  RV    : MARKET entry + OCO bracket (stop + target) — NT8 holds; intrabar fills
  Fabio : MARKET entry + OCO bracket (ORB_Low stop + 4R target) — NT8 holds
  B2    : MARKET entry only — Python sends CLOSE_TAG on exit (bar-close exits)
  OD    : MARKET entry only — Python sends CLOSE_TAG on exit (bar-close exits)

Usage:
  from live.combined.nt8_executor import NT8Executor
  nt8 = NT8Executor(url="http://localhost:8081", instance_id="prod-01")
  nt8.start_heartbeat()
  nt8.wire_to_engines(rv, b2, od, fb)
  # ... later ...
  nt8.stop()
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

from live.combined.config import ET_TZ

NT8_URL_DEFAULT = "http://localhost:8081"   # different port from old addon
HEARTBEAT_INTERVAL_SEC = 10
HEARTBEAT_TIMEOUT_SEC = 30   # NT8's threshold for "stale" → flatten

# Async POST architecture (added 2026-05-21):
# Engine thread calls send_entry()/send_close_tag() which put_nowait on this
# queue and return instantly. A dedicated poster thread drains and does the
# actual blocking HTTP POST. This keeps the engine thread responsive to ticks
# regardless of NT8 latency.
SIGNAL_QUEUE_MAXSIZE = 10_000
POST_TIMEOUT_SEC = 5.0


@dataclass
class _OpenPosition:
    """Tracks a position we've sent to NT8."""
    tag: str
    strat: str
    direction: str
    qty: int
    entry_price: float
    entry_ts: pd.Timestamp
    has_bracket: bool = False


class NT8Executor:
    """HTTP client + heartbeat for the multi-strat NT8 addon."""

    def __init__(self, url: str = NT8_URL_DEFAULT, instance_id: Optional[str] = None,
                 contract: str = "MNQ 06-26", account: str = "Sim101"):
        self.url = url.rstrip("/")
        self.instance_id = instance_id or f"py-{uuid.uuid4().hex[:8]}"
        self.contract = contract
        self.account = account
        self.session = requests.Session()
        self._open: dict[str, _OpenPosition] = {}    # tag -> position
        self._stop_event = threading.Event()
        self._hb_thread: Optional[threading.Thread] = None
        # Async POST queue + dedicated poster thread (decouples engine from NT8 latency)
        self._signal_queue: queue.Queue = queue.Queue(maxsize=SIGNAL_QUEUE_MAXSIZE)
        self._poster_thread: Optional[threading.Thread] = None
        # Counters
        self.n_entries_sent = 0
        self.n_exits_sent = 0
        self.n_heartbeats = 0
        self.n_errors = 0
        self.n_queue_full_drops = 0
        print(f"[nt8] executor initialized (url={self.url}, instance={self.instance_id}, "
              f"contract={self.contract}, account={self.account})")
        # Start poster thread immediately so queue.put_nowait always has a drainer.
        # If the user never calls start_heartbeat(), the poster thread still runs.
        self._start_poster_thread()

    # ---------- async POST infrastructure ----------
    def _start_poster_thread(self) -> None:
        """Spawn the daemon thread that drains _signal_queue and POSTs to NT8."""
        if self._poster_thread is not None and self._poster_thread.is_alive():
            return

        def _poster_loop():
            print(f"[nt8] poster thread started")
            while not self._stop_event.is_set():
                try:
                    endpoint, payload = self._signal_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                self._post_sync(endpoint, payload)
            # Drain remaining signals on shutdown so we don't lose exits
            try:
                while True:
                    endpoint, payload = self._signal_queue.get_nowait()
                    self._post_sync(endpoint, payload)
            except queue.Empty:
                pass
            print(f"[nt8] poster thread stopped (qsize={self._signal_queue.qsize()})")

        self._poster_thread = threading.Thread(
            target=_poster_loop, daemon=True, name="nt8_poster")
        self._poster_thread.start()

    def _enqueue_post(self, endpoint: str, payload: dict) -> bool:
        """Non-blocking enqueue. Returns False only if queue is full (engine
        emitting signals faster than NT8 can accept — shouldn't ever happen
        with a 10K queue and a healthy NT8). Microseconds latency."""
        payload["instance_id"] = self.instance_id
        try:
            self._signal_queue.put_nowait((endpoint, payload))
            return True
        except queue.Full:
            self.n_queue_full_drops += 1
            print(f"  [nt8] signal queue FULL — dropping {endpoint} for {payload.get('tag','?')}")
            return False

    # ---------- low-level HTTP (runs on poster thread only) ----------
    def _post_sync(self, endpoint: str, payload: dict, timeout: float = POST_TIMEOUT_SEC) -> bool:
        """POST to NT8 and log latency. Runs on poster thread — blocking is OK."""
        full_url = f"{self.url}{endpoint}"
        try:
            t0 = time.perf_counter()
            r = self.session.post(full_url, json=payload, timeout=timeout)
            rtt_ms = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                if endpoint != "/heartbeat":
                    print(f"  [nt8] POST {endpoint} OK ({rtt_ms:.0f}ms): {payload.get('tag','')}")
                return True
            else:
                self.n_errors += 1
                print(f"  [nt8] POST {endpoint} FAIL status={r.status_code}: {r.text[:200]}")
                return False
        except requests.exceptions.ConnectionError:
            self.n_errors += 1
            print(f"  [nt8] connection FAIL ({endpoint}) — NT8 addon not running?")
            return False
        except Exception as e:
            self.n_errors += 1
            print(f"  [nt8] POST {endpoint} EXCEPTION: {e}")
            return False

    # ---------- order primitives ----------
    def send_entry(self, strat: str, direction: str, qty: int,
                    sl_price: Optional[float] = None,
                    tp_price: Optional[float] = None,
                    entry_ts: Optional[pd.Timestamp] = None,
                    entry_price: Optional[float] = None) -> Optional[str]:
        """Send MARKET entry. Optionally include stop and/or target resting orders.

        - sl_price=None, tp_price=None: no resting orders (B2 yellow / OD trailing)
        - sl_price set, tp_price set:   OCO bracket (RV, Fabio)
        - tp_price only (no stop):      target-only limit (B2 green — fixed TP)
        Returns the generated tag, or None on failure."""
        ets = entry_ts or pd.Timestamp.now(tz=ET_TZ)
        if ets.tzinfo is None:
            ets = ets.tz_localize(ET_TZ)
        tag = f"{strat}_{ets.strftime('%Y%m%d_%H%M%S')}_{direction}"
        payload = {
            "action": "ENTRY",
            "tag": tag,
            "strat": strat,
            "direction": direction.upper(),
            "quantity": int(qty),
            "order_type": "MARKET",
            "contract": self.contract,
            "account": self.account,
            "timestamp": ets.strftime("%Y-%m-%d %H:%M:%S %Z"),
        }
        has_resting = False
        if sl_price is not None:
            payload["stop_price"]   = f"{sl_price:.2f}"
            has_resting = True
        if tp_price is not None:
            payload["target_price"] = f"{tp_price:.2f}"
            has_resting = True
        # Async enqueue — never blocks the engine thread. POST happens on
        # poster thread; if it fails, the heartbeat watchdog or ResyncFromNT8
        # corrects state. We optimistically track the tag locally so subsequent
        # exit logic can find it.
        if not self._enqueue_post("/order", payload):
            return None
        self._open[tag] = _OpenPosition(
            tag=tag, strat=strat, direction=direction.upper(),
            qty=int(qty), entry_price=entry_price or 0.0,
            entry_ts=ets, has_bracket=has_resting,
        )
        self.n_entries_sent += 1
        return tag

    def send_close_tag(self, tag: str, reason: str = "") -> bool:
        """Send FLATTEN-by-tag — closes only the matching position. Async."""
        payload = {
            "action": "CLOSE_TAG",
            "tag": tag,
            "reason": reason,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if not self._enqueue_post("/order", payload):
            return False
        self._open.pop(tag, None)
        self.n_exits_sent += 1
        return True

    # ---------- engine wiring ----------
    def get_callbacks(self, rv, b2, od, fb) -> dict:
        """Return per-strategy callbacks WITHOUT subscribing them directly.
        Caller (typically the Coordinator) is responsible for routing engine
        signals through these callbacks. Lets coordinator filter blocked
        signals before they reach NT8.

        Returns {strat_name: callback(sig)}.
        """
        # Stash engine refs for the closures
        self._engines_ref = {"RV": rv, "B2": b2, "OD": od, "FB": fb}
        # ---- RV ----
        def on_rv(sig):
            d = ("LONG" if sig.direction.name == "LONG"
                 else "SHORT" if sig.direction.name == "SHORT" else "FLAT")
            if sig.event == "ENTRY" and rv.position is not None:
                p = rv.position
                self.send_entry(
                    strat="RV", direction=d, qty=1,
                    sl_price=p.stop_price, tp_price=p.target_price,
                    entry_ts=sig.timestamp, entry_price=sig.price,
                )
            elif sig.event == "EXIT":
                if sig.reason == "force_close":
                    for tag, op in list(self._open.items()):
                        if op.strat == "RV":
                            self.send_close_tag(tag, sig.reason)
                            break
        # ---- B2 ----
        def on_b2(sig):
            d = ("LONG" if sig.direction.name == "LONG"
                 else "SHORT" if sig.direction.name == "SHORT" else "FLAT")
            if sig.event == "ENTRY" and b2.position is not None:
                p = b2.position
                self.send_entry(
                    strat="B2", direction=d, qty=sig.qty,
                    sl_price=None, tp_price=p.green_val,
                    entry_ts=sig.timestamp, entry_price=sig.price,
                )
            elif sig.event == "EXIT":
                if sig.reason == "TP_FIXED":
                    for tag, op in list(self._open.items()):
                        if op.strat == "B2" and op.direction == d:
                            self._open.pop(tag, None)
                            self.n_exits_sent += 1
                            print(f"  [nt8] B2 TP_FIXED — NT8 already filled resting limit; tag cleared locally: {tag}")
                            break
                else:
                    for tag, op in list(self._open.items()):
                        if op.strat == "B2" and op.direction == d:
                            self.send_close_tag(tag, sig.reason)
                            break
        # ---- OD ----
        def on_od(sig):
            d = "LONG" if sig.direction.name == "LONG" else "FLAT"
            if sig.event == "ENTRY" and od.position is not None:
                self.send_entry(
                    strat="OD", direction=d, qty=sig.qty,
                    entry_ts=sig.timestamp, entry_price=sig.price,
                )
            elif sig.event == "EXIT":
                for tag, op in list(self._open.items()):
                    if op.strat == "OD":
                        self.send_close_tag(tag, sig.reason)
                        break
        # ---- Fabio ----
        def on_fb(sig):
            d = "LONG"
            if sig.event == "ENTRY" and fb.position is not None:
                p = fb.position
                self.send_entry(
                    strat="FB", direction=d, qty=sig.qty,
                    sl_price=p.sl_price, tp_price=p.tp_price,
                    entry_ts=sig.timestamp, entry_price=sig.price,
                )
            elif sig.event == "EXIT":
                if sig.reason == "EOD":
                    for tag, op in list(self._open.items()):
                        if op.strat == "FB":
                            self.send_close_tag(tag, sig.reason)
                            break
        print(f"[nt8] callbacks created for RV+B2+OD+Fabio (coordinator will route)")
        return {"RV": on_rv, "B2": on_b2, "OD": on_od, "FB": on_fb}

    def wire_to_engines(self, rv, b2, od, fb) -> None:
        """LEGACY: subscribes callbacks directly to engines (no coordinator).
        Kept for backward compat / debugging. Use get_callbacks() with
        Coordinator instead."""

        # Map (strat, direction, entry_ts) → tag so we can find tag on EXIT
        tag_by_position: dict[tuple, str] = {}

        def _key(strat, direction, entry_ts):
            ts = pd.Timestamp(entry_ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize(ET_TZ)
            return (strat, direction.upper(), ts.strftime("%Y-%m-%d %H:%M"))

        # ---- RV ----
        def on_rv(sig):
            d = ("LONG" if sig.direction.name == "LONG"
                 else "SHORT" if sig.direction.name == "SHORT" else "FLAT")
            if sig.event == "ENTRY" and rv.position is not None:
                p = rv.position
                tag = self.send_entry(
                    strat="RV", direction=d, qty=1,
                    sl_price=p.stop_price, tp_price=p.target_price,
                    entry_ts=sig.timestamp, entry_price=sig.price,
                )
                if tag:
                    tag_by_position[_key("RV", d, sig.timestamp)] = tag
            elif sig.event == "EXIT":
                # RV uses bracket — NT8 auto-fills the stop/target.
                # For force-close: send CLOSE_TAG to flatten.
                if sig.reason == "force_close":
                    # Find the matching open tag (most recent RV one)
                    for tag, op in list(self._open.items()):
                        if op.strat == "RV":
                            self.send_close_tag(tag, sig.reason)
                            break
        rv.subscribe(on_rv)

        # ---- B2: TP-only resting limit + Python-managed yellow stop ----
        # Green is FIXED at entry — sent as resting LIMIT so NT8 fills at the
        # exact price intrabar. Yellow is dynamic (ratchets each 20-min bar) —
        # Python handles via close_tag. On TP_FIXED exit, skip close_tag because
        # NT8's resting target already filled (otherwise we'd close twice).
        def on_b2(sig):
            d = ("LONG" if sig.direction.name == "LONG"
                 else "SHORT" if sig.direction.name == "SHORT" else "FLAT")
            if sig.event == "ENTRY" and b2.position is not None:
                p = b2.position
                tag = self.send_entry(
                    strat="B2", direction=d, qty=sig.qty,
                    sl_price=None,            # yellow is Python-managed
                    tp_price=p.green_val,     # green is fixed → NT8 resting limit
                    entry_ts=sig.timestamp, entry_price=sig.price,
                )
                if tag:
                    tag_by_position[_key("B2", d, sig.timestamp)] = tag
            elif sig.event == "EXIT":
                # On TP_FIXED, NT8's resting limit already closed the position.
                # Skip close_tag to avoid sending a double-close (which could
                # open a NEW opposite-direction position if NT8's accounting is
                # behind).
                if sig.reason == "TP_FIXED":
                    # Just clean up local state — NT8 already handled the exit
                    for tag, op in list(self._open.items()):
                        if op.strat == "B2" and op.direction == d:
                            self._open.pop(tag, None)
                            self.n_exits_sent += 1
                            print(f"  [nt8] B2 TP_FIXED — NT8 already filled resting limit; tag cleared locally: {tag}")
                            break
                else:
                    # SL_TRAIL or FORCE_CLOSE — Python needs to actively close
                    for tag, op in list(self._open.items()):
                        if op.strat == "B2" and op.direction == d:
                            self.send_close_tag(tag, sig.reason)
                            break
        b2.subscribe(on_b2)

        # ---- OD (Python-managed exits) ----
        def on_od(sig):
            d = "LONG" if sig.direction.name == "LONG" else "FLAT"
            if sig.event == "ENTRY" and od.position is not None:
                tag = self.send_entry(
                    strat="OD", direction=d, qty=sig.qty,
                    entry_ts=sig.timestamp, entry_price=sig.price,
                )
                if tag:
                    tag_by_position[_key("OD", d, sig.timestamp)] = tag
            elif sig.event == "EXIT":
                for tag, op in list(self._open.items()):
                    if op.strat == "OD":
                        self.send_close_tag(tag, sig.reason)
                        break
        od.subscribe(on_od)

        # ---- Fabio ----
        def on_fb(sig):
            d = "LONG"
            if sig.event == "ENTRY" and fb.position is not None:
                p = fb.position
                tag = self.send_entry(
                    strat="FB", direction=d, qty=sig.qty,
                    sl_price=p.sl_price, tp_price=p.tp_price,
                    entry_ts=sig.timestamp, entry_price=sig.price,
                )
                if tag:
                    tag_by_position[_key("FB", d, sig.timestamp)] = tag
            elif sig.event == "EXIT":
                # EOD exit → flatten the FB tag. SL/TP handled by NT8 bracket.
                if sig.reason == "EOD":
                    for tag, op in list(self._open.items()):
                        if op.strat == "FB":
                            self.send_close_tag(tag, sig.reason)
                            break
        fb.subscribe(on_fb)
        print(f"[nt8] wired to RV+B2+OD+Fabio engines")

    # ---------- heartbeat ----------
    def start_heartbeat(self) -> None:
        """Spawn daemon thread that POSTs /heartbeat every 10 sec."""
        if self._hb_thread is not None and self._hb_thread.is_alive():
            return
        self._stop_event.clear()

        def _hb_loop():
            print(f"[nt8] heartbeat thread started ({HEARTBEAT_INTERVAL_SEC}s interval)")
            while not self._stop_event.is_set():
                payload = {
                    "instance_id": self.instance_id,
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "open_positions": len(self._open),
                }
                try:
                    self.session.post(f"{self.url}/heartbeat", json=payload, timeout=3)
                    self.n_heartbeats += 1
                except Exception:
                    pass   # silent fail — NT8 will detect via timeout
                self._stop_event.wait(HEARTBEAT_INTERVAL_SEC)
            print(f"[nt8] heartbeat thread stopped")

        self._hb_thread = threading.Thread(target=_hb_loop, daemon=True, name="nt8_heartbeat")
        self._hb_thread.start()

    def stop(self) -> None:
        """Stop heartbeat + poster threads. Note: does NOT flatten positions —
        NT8's failsafe will detect stale heartbeats and flatten on its own
        after the timeout. For graceful shutdown that holds positions, call
        this and don't restart."""
        self._stop_event.set()
        # Wait briefly for poster to drain remaining queued exits
        if self._poster_thread is not None:
            self._poster_thread.join(timeout=10)
        if self._hb_thread is not None:
            self._hb_thread.join(timeout=5)

    def summary(self) -> dict:
        return {
            "entries_sent": self.n_entries_sent,
            "exits_sent": self.n_exits_sent,
            "heartbeats": self.n_heartbeats,
            "errors": self.n_errors,
            "queue_full_drops": self.n_queue_full_drops,
            "signal_qsize": self._signal_queue.qsize(),
            "open_positions": len(self._open),
            "open_tags": list(self._open.keys()),
        }
