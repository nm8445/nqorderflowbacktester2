"""Broadcast approved live signals from the phase-1 coordinator to the farm app.

Wired as an extra downstream callback in run_phase1.py. On each ENTRY it pulls the strat's stop from
the engine's open position (RV stop_price, OD/B2 yellow_val, FB sl_price), computes the stop distance,
and POSTs {strat, direction, stop_pts} to the farm's /api/signal on a DAEMON THREAD with a 1s timeout.

It can NEVER delay or break the coordinator: every path is wrapped, the post is async/fire-and-forget,
and if the farm app isn't running the post just fails silently. Set ENABLED=False to turn it off.
"""
from __future__ import annotations
import json
import threading
import urllib.request

ENABLED = True
FARM_SIGNAL_URL = "http://localhost:8090/api/signal"


def _stop_level(strat: str, pos):
    if strat == "RV":
        return getattr(pos, "stop_price", None)
    if strat in ("B2", "OD"):
        return getattr(pos, "yellow_val", None)
    if strat == "FB":
        return getattr(pos, "sl_price", None)
    return None


def _broadcast_async(payload: dict) -> None:
    def _send():
        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(FARM_SIGNAL_URL, data=data,
                                         headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=1.0)
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()


def make_farm_cb(strat: str, engine):
    """Downstream callback that mirrors ENTRY signals to the farm. Any error is swallowed —
    the coordinator must never be affected by the farm."""
    def cb(sig):
        try:
            if not ENABLED or getattr(sig, "event", None) != "ENTRY":
                return
            pos = getattr(engine, "position", None)
            if pos is None:
                return
            stop = _stop_level(strat, pos)
            if stop is None:
                return
            stop_pts = abs(float(sig.price) - float(stop))
            if stop_pts <= 0:
                return
            direction = sig.direction.name if hasattr(sig.direction, "name") else str(sig.direction)
            _broadcast_async({"strat": strat, "direction": direction, "stop_pts": round(stop_pts, 2)})
        except Exception:
            pass
    return cb
