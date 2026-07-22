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
    """Downstream callback that mirrors ENTRY and EXIT signals to the farm. Any error is swallowed —
    the coordinator must never be affected by the farm."""
    def cb(sig):
        try:
            if not ENABLED:
                return
            event = getattr(sig, "event", None)
            direction = sig.direction.name if hasattr(sig.direction, "name") else str(sig.direction)
            # ---- EXIT: forward so the farm can close backend-managed MILK legs, and force-close
            # gamblers + milkers (+ eval-takers) on a session force-close. The reason drives
            # milk-only vs close-everyone farm-side (force-close = force_close/FORCE_CLOSE/EOD). ----
            if event == "EXIT":
                _broadcast_async({"event": "EXIT", "strat": strat, "direction": direction,
                                  "reason": getattr(sig, "reason", "") or ""})
                return
            if event != "ENTRY":
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
            sign = 1 if str(direction).upper() == "LONG" else -1
            # Signal-anchored ABSOLUTE bracket for GAMBLERS / EVAL-TAKERS (static 1:1, eval-mode
            # run-to-completion): SL = the engine's stop (RV 2xATR / OD+B2 initial yellow / FB ORB_Low);
            # TP = a 1:1 level off the ENTRY SIGNAL price. RV's 1:1 == its native target; OD/B2/FB forced.
            payload = {"event": "ENTRY", "strat": strat, "direction": direction,
                       "stop_pts": round(stop_pts, 2), "sl_price": round(float(stop), 2)}
            tgt = getattr(pos, "target_price", None) if strat == "RV" else None
            payload["tp_price"] = (round(float(tgt), 2) if tgt is not None
                                   else round(float(sig.price) + sign * stop_pts, 2))
            # NATIVE levels so MILK routes can mirror :8081 shapes (built farm-side, per strat):
            #   B2 -> green (resting TP, no SL); FB -> native TP (1:4). RV native target already in
            #   tp_price; OD rests nothing. A missing level -> farm falls back to bracket().
            if strat == "B2":
                green = getattr(pos, "green_val", None)
                if green is not None:
                    payload["green"] = round(float(green), 2)
            elif strat == "FB":
                fbtp = getattr(pos, "tp_price", None)
                if fbtp is not None:
                    payload["fb_tp"] = round(float(fbtp), 2)
            _broadcast_async(payload)
        except Exception:
            pass
    return cb
