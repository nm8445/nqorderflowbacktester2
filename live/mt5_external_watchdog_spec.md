# MT5 External Watchdog (Layer 3) — Implementation Spec

**Status**: NOT BUILT. To be implemented when porting live system to MT5.

This is a standalone monitoring process that ensures positions get flattened if the main Python live engine dies, hangs, or loses connection to MT5. It's the MT5 equivalent of NT8's heartbeat-watchdog that lived inside the addon.

---

## Why this exists

When trading on MT5 with the "dumb executor" pattern (Python sends market entry/exit, no broker-side SL/TP), there's no NT8-style server-side fallback. If Python crashes mid-position, the position sits naked. Layers 1 + 2 reduce this risk but don't eliminate it:

| Layer | What it does | Failure mode it covers | Failure mode it MISSES |
|---|---|---|---|
| **Layer 1**: Disaster SL on entry order | Broker-side hard stop ~150pts from entry, regardless of normal exit logic | Python crashes + adverse move hits 150pts | Slow bleed where price drifts adversely but doesn't hit disaster SL |
| **Layer 2**: Python self-watchdog thread | Internal thread monitors main loop tick; force-close all positions if main loop stalls >30s | Python deadlock / infinite loop | Whole Python process killed (e.g., OOM, Ctrl-C, BSOD) |
| **Layer 3** (this doc) | Separate process pings Python, flattens if Python is dead | Whole Python process killed | Both Python AND watchdog dead simultaneously (~rare) |

Layers 1+2 alone might be enough if you trust the process — but at prop-firm sizing the cost of one runaway position can be the account. Layer 3 is cheap insurance.

---

## Architecture

```
┌────────────────────────┐         ┌──────────────────────────┐
│   Main Python engine   │         │   Watchdog process       │
│   (live/combined/      │ ←ping── │   (live/combined/        │
│    run_phase1.py)      │  every  │    mt5_watchdog.py)      │
│                        │  10s    │                          │
│   - Strategy engines   │         │   - Pings main engine    │
│   - MT5 executor       │  loses  │   - On 30s silence:      │
│   - Position state     │  ping?  │     1. Login to MT5      │
│                        │  ════>  │     2. mt5.positions_get │
└────────────────────────┘         │     3. Close all tagged  │
            │                      │        positions         │
            │ orders               │     4. Page user (SMS)   │
            ↓                      └──────────────────────────┘
        ┌─────┐                              │
        │ MT5 │ ←──────────── login + close ─┘
        └─────┘
```

Two independent processes, each connected to MT5 via the `MetaTrader5` Python package using DIFFERENT logins (or the same login if the broker permits multi-connect — most do for the same MT5 account).

---

## Watchdog process behavior

### Startup
1. Load config from `live/combined/config.py` (or new `live/combined/watchdog_config.py`)
2. Login to MT5 with watchdog credentials (read-only flag NOT supported by MT5 — needs full trading perms to close)
3. Open HTTP listener on `localhost:8090` (different port than NT8's 8081)
4. Start ping check loop

### Ping check loop (every 5s)
1. POST `/ping` to `http://localhost:8089/ping` (the main engine's health endpoint)
2. Expected response: `200 OK` with body `{"alive": true, "last_signal_ts": "..."}`
3. If response received within 5s → reset stale counter
4. If 6 consecutive misses (30s total) → **FIRE FLATTEN**

### Flatten procedure
1. Log critical event to `live/combined/state/watchdog_flatten.log` with timestamp + reason
2. Call `mt5.positions_get()` to enumerate open positions
3. Filter by tag prefix matching our naming convention (`RV_*`, `B2_*`, `OD_*`, `FB_*` in the `comment` field)
4. For each matching position:
   - `mt5.order_send({"action": TRADE_ACTION_DEAL, "type": opposite_of_position, "volume": position.volume, "position": position.ticket, ...})`
   - Log fill price
5. After all closed, write summary: `N positions flattened at TS, reason: main engine silent >30s`
6. Send SMS/email alert via SES or Twilio (low priority — implement later)
7. Continue running — main engine may come back, but watchdog should NOT auto-reopen anything

### Manual mode
- Watchdog should have a `--manual-flatten` CLI flag to immediately flatten all tagged positions without waiting for ping timeout
- Useful for emergency cases where user wants to bail before Python detects an issue

---

## Main engine changes required

The main engine (`run_phase1.py` or new HTTP wrapper) needs to expose a health endpoint:

```python
# In live/combined/run_phase1.py (or a new health_server.py)
from http.server import BaseHTTPRequestHandler, HTTPServer
import json, threading, time

last_loop_tick_ts = time.time()  # updated every main loop iteration

class HealthHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/ping":
            age_s = time.time() - last_loop_tick_ts
            alive = age_s < 30  # consider dead if main loop hasn't ticked in 30s
            self.send_response(200 if alive else 503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "alive": alive,
                "last_loop_age_s": round(age_s, 1),
                "open_positions": len(open_positions_dict),
            }).encode())

def start_health_server():
    HTTPServer(("localhost", 8089), HealthHandler).serve_forever()

threading.Thread(target=start_health_server, daemon=True).start()
```

Then in the main loop:
```python
while True:
    last_loop_tick_ts = time.time()  # touched every iteration
    # ... process bar / signal / orders ...
    time.sleep(0.1)
```

---

## Watchdog credentials & security

- Watchdog needs the same MT5 login credentials as the main engine (to flatten positions)
- Store in `live/combined/state/watchdog_creds.json` (gitignored)
- ALTERNATIVELY: have main engine pass a "kill token" to watchdog at startup that authorizes a single flatten event

---

## Multi-firm complication

Each prop firm = separate MT5 terminal + separate watchdog process:
- FundingPips watchdog → FP MT5 terminal
- FundedNext watchdog → FN MT5 terminal
- HolaPrime watchdog → HP MT5 terminal

Each watchdog pings ITS firm's main engine (different ports: 8089, 8090, 8091, etc.).

Don't try to centralize — a single watchdog connecting to 3 MT5 instances is fragile and adds points of failure.

---

## Implementation checklist

When the time comes:

- [ ] Create `live/combined/mt5_watchdog.py` with the ping check loop
- [ ] Add `/ping` HTTP endpoint to `run_phase1.py` (or a new `health_server.py` that imports the live state)
- [ ] Test: kill main engine while a position is open → verify watchdog closes it within 35s
- [ ] Test: main engine deadlock (long sleep in main loop) → verify same outcome
- [ ] Test: watchdog itself crashes → ensure main engine still runs (no dependency loop)
- [ ] Wire SMS/email alert (Twilio? SES?) for "flatten fired" event
- [ ] Document in `scaling_plan.md` once built

## Notes

- The 30s threshold matches the NT8 addon's heartbeat watchdog. Could tighten to 15s for faster reaction at the cost of false positives during brief GC pauses or bar-processing spikes.
- The watchdog must NEVER auto-reopen positions. Its only verb is "close." If main engine recovers, IT decides to reopen (subject to coordinator no-hedge rules).
- This entire layer can be skipped during paper trading (no real money at risk). Build it BEFORE the first funded MT5 account goes live.
