"""TEMPORARY one-shot monitor: watch the MT5 OD position; the moment it closes,
flatten the manually-entered NT8 position.

Context: NT8 crashed while OD was supposed to enter, so you entered NT8 manually.
MT5 received the real system OD order (magic 30001) and the live system will close
it at OD's exit (08:00 ET force-close, or yellow/green). This script mirrors that
close onto your manual NT8 trade.

Why not just POST to NT8 :8081? The addon only supports ENTRY / CLOSE_TAG (tag-
scoped). Your manual trade is UNTAGGED, so CLOSE_TAG can't touch it. Flatten paths,
in order of reliability:
  1) NT8 file-ATI OIF  -> writes CLOSEPOSITION to the `incoming` folder (no recompile,
     account+instrument scoped, tag-agnostic). PRIMARY.
  2) POST {"action":"FLATTEN"} to :8081  -> only works if you added the /flatten
     handler to NQMultiStratReceiver.cs (snippet in the chat). Best-effort.
  3) LOUD repeated alarm + banner -> guaranteed fallback so you can close by hand.

SET THE 3 CONFIG VALUES BELOW before running. Run:  python scripts/temp_mt5_to_nt8_flatten.py
Stop with Ctrl+C. Safe to delete this file afterward.
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

# ======================= CONFIG — SET THESE 2 =======================
NT8_ACCOUNT    = "MFFUSFFLX606768002"        # <-- your NT8 account name (Control Center > Accounts tab)
NT8_INSTRUMENT = "MNQ 06-26"     # <-- the instrument you manually entered on NT8 (full name)
# --------------------------------------------------------------------
OD_MAGIC       = 30001           # MT5 magic for OD (from mt5_executor.MAGIC)
MT5_SYMBOL     = os.getenv("MT5_SYMBOL", "NDX100")
NT8_INCOMING   = Path.home() / "Documents" / "NinjaTrader 8" / "incoming"
NT8_HOST, NT8_PORT = "localhost", 8081
NT8_HTTP       = f"http://{NT8_HOST}:{NT8_PORT}/order"
POLL_SEC       = 1.0
STATUS_EVERY   = 15              # print a live status line every N seconds
CLOSE_CONFIRM  = 3               # require N consecutive 'absent' reads before firing (anti-glitch)
# ====================================================================

import socket

try:
    import MetaTrader5 as mt5
except ImportError:
    sys.exit("MetaTrader5 package not installed (pip install MetaTrader5). Run on the MT5 machine.")
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
try:
    import requests
except ImportError:
    requests = None


def _log(msg: str):
    print(f"{datetime.now().strftime('%H:%M:%S')}  {msg}", flush=True)


def mt5_connect() -> bool:
    """Attach to the already-running MT5 terminal (read-only use here)."""
    if mt5.initialize():
        return True
    # Fallback: explicit login from .env
    login = os.getenv("MT5_LOGIN"); pwd = os.getenv("MT5_PASSWORD"); srv = os.getenv("MT5_SERVER")
    path = os.getenv("MT5_TERMINAL_PATH") or None
    kwargs = {}
    if path: kwargs["path"] = path
    if login and pwd and srv:
        kwargs.update(login=int(login), password=pwd, server=srv)
    if kwargs and mt5.initialize(**kwargs):
        return True
    _log(f"MT5 initialize FAILED: {mt5.last_error()}")
    return False


def get_od_position():
    """Returns (state, pos): state in {'open','flat','error'}; pos = MT5 position or None.
    Queries ALL positions (no symbol filter) and matches by magic 30001 — robust to
    broker-specific symbol naming / Market Watch selection."""
    total = mt5.positions_total()
    if total is None:
        return ("error", None)
    if total == 0:
        return ("flat", None)                 # definitively flat — no false 'error'
    poss = mt5.positions_get()                 # all open positions, no symbol filter
    if poss is None:
        return ("error", None)
    for p in poss:
        if p.magic == OD_MAGIC:
            return ("open", p)
    return ("flat", None)                      # positions exist but none are OD (magic 30001)


def dump_all_positions():
    """Print every open MT5 position so you can verify the OD one (magic/symbol)."""
    poss = mt5.positions_get()
    if poss is None:
        _log(f"  positions_get() -> None  (last_error={mt5.last_error()})")
        return
    if len(poss) == 0:
        _log("  no open MT5 positions at all.")
        return
    _log(f"  {len(poss)} open MT5 position(s):")
    for p in poss:
        side = "LONG" if p.type == 0 else "SHORT"
        flag = "  <== OD" if p.magic == OD_MAGIC else ""
        _log(f"    symbol={p.symbol} magic={p.magic} ticket={p.ticket} {side} "
             f"{p.volume}lot @ {p.price_open:.2f} P/L {p.profit:+.2f}{flag}")


def nt8_reachable() -> bool:
    """Side-effect-free TCP check that the NT8 addon is listening on :8081."""
    try:
        with socket.create_connection((NT8_HOST, NT8_PORT), timeout=2):
            return True
    except OSError:
        return False


def incoming_ok() -> bool:
    """Verify the NT8 file-ATI 'incoming' folder exists / is writable."""
    try:
        NT8_INCOMING.mkdir(parents=True, exist_ok=True)
        return os.access(NT8_INCOMING, os.W_OK)
    except Exception:
        return False


def _pos_desc(p) -> str:
    side = "LONG" if p.type == 0 else "SHORT"
    return (f"ticket={p.ticket} {side} {p.volume}lot @ {p.price_open:.2f} "
            f"now {p.price_current:.2f} P/L {p.profit:+.2f}")


def flatten_nt8():
    _log(">>> OD CLOSED ON MT5 — flattening NT8 manual position <<<")
    # 1) File ATI OIF (primary, tag-agnostic, no recompile)
    try:
        NT8_INCOMING.mkdir(parents=True, exist_ok=True)
        oif = NT8_INCOMING / f"oif_flat_{uuid.uuid4().hex[:8]}.txt"
        oif.write_text(f"CLOSEPOSITION;{NT8_ACCOUNT};{NT8_INSTRUMENT};;;;;;;;;;;\n")
        _log(f"  [OIF] wrote {oif.name}  -> CLOSEPOSITION {NT8_ACCOUNT} {NT8_INSTRUMENT}")
    except Exception as e:
        _log(f"  [OIF] FAILED: {e}")
    # 2) Best-effort POST to :8081 (only works if /flatten handler was added)
    if requests is not None:
        try:
            r = requests.post(NT8_HTTP, json={
                "action": "FLATTEN", "account": NT8_ACCOUNT,
                "instrument": NT8_INSTRUMENT,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }, timeout=3)
            _log(f"  [HTTP] POST :8081 FLATTEN -> status {r.status_code}")
        except Exception as e:
            _log(f"  [HTTP] POST :8081 failed (expected if no /flatten endpoint): {e}")


def alarm_forever():
    """Loud repeated alert until you Ctrl+C — guaranteed fallback."""
    try:
        import winsound
        beep = lambda: winsound.Beep(1000, 400)
    except Exception:
        beep = lambda: print("\a", end="", flush=True)
    n = 0
    while True:
        n += 1
        print("\n" + "!" * 64)
        print(f"  MT5 OD CLOSED — NT8 FLATTEN SENT. VERIFY NT8 IS FLAT ({NT8_ACCOUNT}/{NT8_INSTRUMENT}).")
        print(f"  If still open, CLOSE IT MANUALLY NOW.  (alert #{n})  Ctrl+C to stop.")
        print("!" * 64, flush=True)
        for _ in range(3):
            beep(); time.sleep(0.2)
        time.sleep(5)


def main():
    print("=" * 64)
    print("  TEMP MT5(OD magic 30001) -> NT8 manual-flatten monitor")
    print(f"  NT8 target : {NT8_ACCOUNT} / {NT8_INSTRUMENT}")
    print(f"  MT5 symbol : {MT5_SYMBOL}   incoming: {NT8_INCOMING}")
    print("=" * 64)
    if NT8_ACCOUNT == "Sim101":
        _log("WARNING: NT8_ACCOUNT is still the default 'Sim101' — confirm it's correct!")

    # ---- MT5 connection ----
    if not mt5_connect():
        sys.exit("Could not connect to MT5. Make sure the terminal is open/logged in.")
    ti = mt5.terminal_info()
    ai = mt5.account_info()
    _log(f"MT5 connected: terminal={'CONNECTED' if (ti and ti.connected) else 'NOT CONNECTED'}"
         f"  account={getattr(ai, 'login', '?')}  symbol={MT5_SYMBOL}")

    # ---- NT8 connectivity checks ----
    nt8_up = nt8_reachable()
    _log(f"NT8 addon :{NT8_PORT}  -> {'REACHABLE' if nt8_up else 'NOT reachable (will keep retrying)'}")
    _log(f"NT8 incoming folder    -> {'OK (writable)' if incoming_ok() else 'NOT writable: ' + str(NT8_INCOMING)}")
    if not nt8_up:
        _log("  (NT8 not up yet is fine — the OIF file path still works once NT8 reconnects.)")

    # ---- show everything currently open (diagnostic) ----
    dump_all_positions()

    # ---- initial OD position state ----
    state, pos = get_od_position()
    if state == "open":
        _log(f"OD position FOUND on MT5: {_pos_desc(pos)}. Watching for its close...")
    elif state == "flat":
        _log("No OD position on MT5 right now. Either it already closed, or wrong "
             "symbol/magic. Watching anyway — will fire if one appears then closes.")
    else:
        _log("MT5 read error on first check — will keep retrying.")

    seen_open = (state == "open")
    absent_streak = 0
    last_status = 0.0
    try:
        while True:
            time.sleep(POLL_SEC)
            state, pos = get_od_position()

            # periodic heartbeat so you can SEE it's alive + connected
            now = time.time()
            if now - last_status >= STATUS_EVERY:
                last_status = now
                if state == "open":
                    od_str = f"OD OPEN [{_pos_desc(pos)}]"
                elif state == "flat":
                    od_str = "OD flat (not open)"
                else:
                    od_str = "OD read ERROR (retrying)"
                _log(f"monitoring... {od_str} | NT8 :{NT8_PORT} "
                     f"{'UP' if nt8_reachable() else 'DOWN'} | incoming "
                     f"{'OK' if incoming_ok() else 'BAD'}")

            if state == "error":
                continue                      # transient error — never fire on this
            if state == "open":
                seen_open = True
                absent_streak = 0
                continue
            # state == 'flat'
            if not seen_open:
                continue                      # never saw it open; nothing to mirror
            absent_streak += 1
            _log(f"OD no longer present ({absent_streak}/{CLOSE_CONFIRM} confirmations)...")
            if absent_streak >= CLOSE_CONFIRM:
                flatten_nt8()
                alarm_forever()               # blocks until Ctrl+C
    except KeyboardInterrupt:
        _log("stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
