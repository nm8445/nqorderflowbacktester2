"""MT5 executor — mirror of NT8Executor interface for fanning out signals to MT5.

Designed to be a SIBLING of nt8_executor.py, not a replacement. The main engine
emits signals once; both executors receive them in parallel via get_callbacks().

Default config is intentionally tiny (0.01 lots NDX100, ~$0.10/pt exposure)
for safe verification. Real per-firm sizing comes from explicit init args or
a YAML config (later).

Per-strategy behavior (matches the prop firm rule that requires SL within 3 min):
  - RV:    entry + SL + TP (strategy provides both; pure bracket)
  - Fabio: entry + SL + TP (same)
  - B2:    entry + SL (disaster, from sl_pts_per_strat) + TP (from strategy.green_val).
           Python actively closes via close_strategy() on yellow/EOD.
  - OD:    entry + SL (disaster, from sl_pts_per_strat). No TP — Python closes.

NDX100 contract: 1 lot = $10/pt (vs MNQ $2/pt). 0.01 lots = $0.10/pt.
NDX100 and NQ move 1:1 in points but differ in absolute price (basis varies).
SL/TP distances translate directly; absolute prices must be re-anchored to the
current NDX100 tick.

Usage:
    from live.combined.mt5_executor import MT5Executor
    mt5x = MT5Executor()  # tiny-lot test defaults
    mt5x.start_heartbeat()
    cbs = mt5x.get_callbacks(rv, b2, od, fb)
    # caller wires cbs into coordinator
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

# Load MT5 credentials from .env
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Lazy import — MetaTrader5 package only available on Windows
try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    _MT5_AVAILABLE = False

from live.combined.config import ET_TZ

# ============================== defaults ==============================

# Tiny test sizes — safe for first live verification (0.01 lot NDX100 = $0.10/pt).
# Override via explicit init args or YAML config when ready to scale.
DEFAULT_LOTS = {"OD": 0.01, "B2": 0.01, "RV": 0.01, "FB": 0.01}

# Disaster SL distances per strategy (NDX/NQ points — they move 1:1).
# Used when the strategy doesn't provide its own SL price (B2, OD).
# For RV/Fabio: only used if strategy SL price is missing.
DEFAULT_SL_PTS = {"OD": 600.0, "B2": 600.0, "RV": 200.0, "FB": 150.0}

# Magic numbers per strategy (for filtering positions in MT5 audit)
MAGIC = {"OD": 30001, "B2": 30002, "RV": 30003, "FB": 30004}

HEARTBEAT_INTERVAL_SEC = 30


@dataclass
class _OpenMT5Position:
    """Tracks a position we've opened on MT5."""
    strat: str
    customTag: str
    direction: str          # "LONG" / "SHORT"
    volume: float
    entry_price: float
    entry_ts: pd.Timestamp
    ticket: int = 0         # broker-assigned ticket
    has_bracket: bool = False


class MT5Executor:
    """MT5 client. Same shape as NT8Executor — drop-in addition to the engine."""

    def __init__(self,
                 mt5_path: Optional[str] = None,
                 login: Optional[int] = None,
                 password: Optional[str] = None,
                 server: Optional[str] = None,
                 contract: Optional[str] = None,
                 lots_per_strat: Optional[dict] = None,
                 sl_pts_per_strat: Optional[dict] = None,
                 dry_run: bool = False,
                 instance_id: Optional[str] = None,
                 firm_label: str = "MT5"):
        self.mt5_path = mt5_path or os.getenv("MT5_TERMINAL_PATH", "")
        self.login = int(login if login is not None else os.getenv("MT5_LOGIN", "0"))
        self.password = password or os.getenv("MT5_PASSWORD", "")
        self.server = server or os.getenv("MT5_SERVER", "")
        # Symbol: explicit arg > env var > default. TradeMax/other brokers vary.
        self.contract = contract or os.getenv("MT5_SYMBOL", "NDX100")
        self.lots = dict(lots_per_strat) if lots_per_strat else dict(DEFAULT_LOTS)
        self.sl_pts = dict(sl_pts_per_strat) if sl_pts_per_strat else dict(DEFAULT_SL_PTS)
        self.dry_run = dry_run
        self.instance_id = instance_id or f"mt5-{uuid.uuid4().hex[:8]}"
        self.firm_label = firm_label

        self._open: dict[str, _OpenMT5Position] = {}    # strat -> position
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._hb_thread: Optional[threading.Thread] = None
        self._initialized = False
        # Broker-specific filling mode (detected during init from symbol_info)
        self._filling_mode = None

        # Counters
        self.n_entries_sent = 0
        self.n_exits_sent = 0
        self.n_heartbeats = 0
        self.n_errors = 0
        self.n_dry_run_logs = 0

        print(f"[mt5][{self.firm_label}] executor initialized "
              f"(contract={self.contract}, dry_run={self.dry_run}, "
              f"instance={self.instance_id})")
        print(f"[mt5][{self.firm_label}]   lots: {self.lots}")
        print(f"[mt5][{self.firm_label}]   SL pts: {self.sl_pts}")

        if not _MT5_AVAILABLE:
            print(f"[mt5][{self.firm_label}] WARNING: MetaTrader5 package not installed. "
                  f"Forcing dry_run=True.")
            self.dry_run = True
            return

        if not self.dry_run:
            self._init_mt5()

    def _init_mt5(self) -> bool:
        """Initialize + login to MT5 terminal."""
        try:
            ok = mt5.initialize(path=self.mt5_path) if self.mt5_path else mt5.initialize()
            if not ok:
                err = mt5.last_error()
                print(f"[mt5][{self.firm_label}] FAIL initialize: {err}")
                self.n_errors += 1
                return False
            if self.login and self.password and self.server:
                if not mt5.login(self.login, self.password, self.server):
                    print(f"[mt5][{self.firm_label}] FAIL login: {mt5.last_error()}")
                    self.n_errors += 1
                    mt5.shutdown()
                    return False
            ai = mt5.account_info()
            if ai is None:
                print(f"[mt5][{self.firm_label}] FAIL account_info: {mt5.last_error()}")
                self.n_errors += 1
                return False
            print(f"[mt5][{self.firm_label}] connected. account={ai.login} "
                  f"balance=${ai.balance:.2f} server={ai.server}")
            # Ensure contract visible in Market Watch
            info = mt5.symbol_info(self.contract)
            if not info or not info.visible:
                mt5.symbol_select(self.contract, True)
                info = mt5.symbol_info(self.contract)
            # Detect supported filling mode (brokers vary: FOK, IOC, RETURN)
            fm = info.filling_mode if info else 0
            if fm & 2:
                self._filling_mode = mt5.ORDER_FILLING_IOC
                fname = "IOC"
            elif fm & 1:
                self._filling_mode = mt5.ORDER_FILLING_FOK
                fname = "FOK"
            else:
                self._filling_mode = mt5.ORDER_FILLING_RETURN
                fname = "RETURN"
            print(f"[mt5][{self.firm_label}] filling mode: {fname} (broker bitmask={fm})")
            # Adopt any pre-existing positions matching our magics
            self._resync_from_mt5()
            self._initialized = True
            return True
        except Exception as e:
            print(f"[mt5][{self.firm_label}] init EXCEPTION: {e}")
            self.n_errors += 1
            return False

    def _resync_from_mt5(self) -> None:
        """Adopt any open positions whose magic matches our strategies."""
        try:
            positions = mt5.positions_get(symbol=self.contract) or []
            for p in positions:
                strat = next((s for s, m in MAGIC.items() if m == p.magic), None)
                if strat is None:
                    continue
                if strat in self._open:
                    continue   # already tracked
                direction = "LONG" if p.type == mt5.POSITION_TYPE_BUY else "SHORT"
                self._open[strat] = _OpenMT5Position(
                    strat=strat, customTag=p.comment or f"{strat}_RESYNC",
                    direction=direction, volume=p.volume, entry_price=p.price_open,
                    entry_ts=pd.Timestamp.now(tz=ET_TZ), ticket=p.ticket, has_bracket=False,
                )
                print(f"[mt5][{self.firm_label}] RESYNC adopted {strat} {direction} "
                      f"vol={p.volume} ticket={p.ticket}")
        except Exception as e:
            print(f"[mt5][{self.firm_label}] resync EXCEPTION: {e}")

    # ---------- low-level order sending ----------

    def _send_entry(self, strat: str, direction: str,
                    sl_price: Optional[float] = None,
                    tp_price: Optional[float] = None,
                    entry_ts: Optional[pd.Timestamp] = None,
                    signal_entry_price: Optional[float] = None) -> bool:
        """Send a MARKET entry on NDX100 with optional SL/TP.

        SL/TP are computed by translating the strategy's intended SL distance
        (in NQ-basis points) onto the current NDX100 price (which differs in
        absolute level but moves 1:1 in points)."""
        if strat in self._open:
            print(f"[mt5][{self.firm_label}] WARN entry skipped: {strat} already has open position")
            return False

        lots = self.lots.get(strat, 0.01)
        customTag = f"{strat}_{(entry_ts or pd.Timestamp.now(tz=ET_TZ)).strftime('%Y%m%d_%H%M%S')}_{direction}"

        # Compute SL/TP distances in points (relative to entry).
        sl_distance_pts = None
        tp_distance_pts = None
        if signal_entry_price is not None:
            if sl_price is not None:
                sl_distance_pts = abs(signal_entry_price - sl_price)
            if tp_price is not None:
                tp_distance_pts = abs(tp_price - signal_entry_price)
        # Use disaster SL if no strategy SL provided
        if sl_distance_pts is None or sl_distance_pts <= 0:
            sl_distance_pts = self.sl_pts.get(strat, 600.0)

        if self.dry_run:
            self.n_dry_run_logs += 1
            print(f"  [mt5-DRYRUN][{self.firm_label}][{strat}-ENTRY] {direction} "
                  f"lots={lots} sl_pts={sl_distance_pts:.1f} "
                  f"tp_pts={tp_distance_pts if tp_distance_pts else 'NONE'} tag={customTag}")
            self._open[strat] = _OpenMT5Position(
                strat=strat, customTag=customTag, direction=direction,
                volume=lots, entry_price=signal_entry_price or 0.0,
                entry_ts=entry_ts or pd.Timestamp.now(tz=ET_TZ),
                ticket=0, has_bracket=tp_distance_pts is not None,
            )
            self.n_entries_sent += 1
            return True

        # Real send
        if not self._initialized:
            print(f"[mt5][{self.firm_label}] not initialized, skipping entry")
            return False
        try:
            tick = mt5.symbol_info_tick(self.contract)
            if tick is None or tick.ask == 0 or tick.bid == 0:
                print(f"[mt5][{self.firm_label}] WARN market closed/no tick — skipping {strat} entry")
                self.n_errors += 1
                return False
            ref = tick.ask if direction == "LONG" else tick.bid
            sl_abs = ref - sl_distance_pts if direction == "LONG" else ref + sl_distance_pts
            tp_abs = (ref + tp_distance_pts if direction == "LONG" else ref - tp_distance_pts) \
                     if tp_distance_pts else 0.0

            req = {
                "action":  mt5.TRADE_ACTION_DEAL,
                "symbol":  self.contract,
                "volume":  lots,
                "type":    mt5.ORDER_TYPE_BUY if direction == "LONG" else mt5.ORDER_TYPE_SELL,
                "price":   ref,
                "sl":      sl_abs,
                "tp":      tp_abs,
                "deviation": 20,
                "magic":   MAGIC.get(strat, 30000),
                "comment": customTag[:31],   # MT5 truncates comments to 31 chars
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._filling_mode or mt5.ORDER_FILLING_RETURN,
            }
            t0 = time.perf_counter()
            res = mt5.order_send(req)
            rtt_ms = (time.perf_counter() - t0) * 1000
            if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
                rc = getattr(res, "retcode", -1) if res else -1
                msg = getattr(res, "comment", str(mt5.last_error())) if res else str(mt5.last_error())
                print(f"  [mt5][{self.firm_label}][{strat}-ENTRY] FAIL retcode={rc} {msg}")
                self.n_errors += 1
                return False
            ticket = self._find_position_ticket(customTag, MAGIC.get(strat, 0))
            self._open[strat] = _OpenMT5Position(
                strat=strat, customTag=customTag, direction=direction,
                volume=lots, entry_price=res.price,
                entry_ts=entry_ts or pd.Timestamp.now(tz=ET_TZ),
                ticket=ticket, has_bracket=tp_distance_pts is not None,
            )
            self.n_entries_sent += 1
            print(f"  [mt5][{self.firm_label}][{strat}-ENTRY] OK {direction} lots={lots} "
                  f"@ {res.price:.2f} sl={sl_abs:.2f} tp={tp_abs:.2f} "
                  f"ticket={ticket} rtt={rtt_ms:.0f}ms")
            return True
        except Exception as e:
            print(f"  [mt5][{self.firm_label}][{strat}-ENTRY] EXCEPTION: {e}")
            self.n_errors += 1
            return False

    def _find_position_ticket(self, tag: str, magic: int) -> int:
        try:
            time.sleep(0.2)  # let broker register
            positions = mt5.positions_get(symbol=self.contract) or []
            for p in positions:
                if p.magic == magic and (p.comment == tag[:31] or p.comment == tag):
                    return p.ticket
        except Exception:
            pass
        return 0

    def _close_strategy(self, strat: str, reason: str = "") -> bool:
        with self._lock:
            pos = self._open.pop(strat, None)
        if pos is None:
            print(f"  [mt5][{self.firm_label}][{strat}-CLOSE] no tracked position (already closed?)")
            return False

        if self.dry_run:
            self.n_dry_run_logs += 1
            print(f"  [mt5-DRYRUN][{self.firm_label}][{strat}-CLOSE] {pos.direction} "
                  f"lots={pos.volume} reason={reason} tag={pos.customTag}")
            self.n_exits_sent += 1
            return True

        if not self._initialized:
            print(f"[mt5][{self.firm_label}] not initialized, skipping close")
            return False

        try:
            tick = mt5.symbol_info_tick(self.contract)
            if tick is None or tick.ask == 0 or tick.bid == 0:
                print(f"[mt5][{self.firm_label}] WARN market closed — close skipped for {strat}")
                # Re-add since we can't actually close
                with self._lock:
                    self._open[strat] = pos
                return False
            # Refresh ticket via re-search (in case we never captured it)
            ticket = pos.ticket or self._find_position_ticket(pos.customTag, MAGIC.get(strat, 0))
            req = {
                "action":  mt5.TRADE_ACTION_DEAL,
                "symbol":  self.contract,
                "volume":  pos.volume,
                "type":    mt5.ORDER_TYPE_SELL if pos.direction == "LONG" else mt5.ORDER_TYPE_BUY,
                "position": ticket,        # CRITICAL: close, not new opposing
                "price":   tick.bid if pos.direction == "LONG" else tick.ask,
                "deviation": 20,
                "magic":   MAGIC.get(strat, 30000),
                "comment": (pos.customTag + "_X")[:31],
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._filling_mode or mt5.ORDER_FILLING_RETURN,
            }
            t0 = time.perf_counter()
            res = mt5.order_send(req)
            rtt_ms = (time.perf_counter() - t0) * 1000
            if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
                rc = getattr(res, "retcode", -1) if res else -1
                msg = getattr(res, "comment", str(mt5.last_error())) if res else str(mt5.last_error())
                print(f"  [mt5][{self.firm_label}][{strat}-CLOSE] FAIL retcode={rc} {msg}")
                self.n_errors += 1
                # Put back in tracking — close didn't go through
                with self._lock:
                    self._open[strat] = pos
                return False
            self.n_exits_sent += 1
            print(f"  [mt5][{self.firm_label}][{strat}-CLOSE] OK reason={reason} "
                  f"@ {res.price:.2f} rtt={rtt_ms:.0f}ms")
            return True
        except Exception as e:
            print(f"  [mt5][{self.firm_label}][{strat}-CLOSE] EXCEPTION: {e}")
            self.n_errors += 1
            with self._lock:
                self._open[strat] = pos
            return False

    # ---------- engine wiring ----------

    def get_callbacks(self, rv, b2, od, fb) -> dict:
        """Same shape as NT8Executor.get_callbacks(). Coordinator routes engine
        signals through these (plus paper logger + NT8 callbacks) after no-hedge
        validation."""
        self._engines_ref = {"RV": rv, "B2": b2, "OD": od, "FB": fb}

        def on_rv(sig):
            d = ("LONG" if sig.direction.name == "LONG"
                 else "SHORT" if sig.direction.name == "SHORT" else "FLAT")
            if sig.event == "ENTRY" and rv.position is not None:
                p = rv.position
                self._send_entry("RV", d,
                                  sl_price=p.stop_price, tp_price=p.target_price,
                                  entry_ts=sig.timestamp, signal_entry_price=sig.price)
            elif sig.event == "EXIT" and sig.reason == "force_close":
                self._close_strategy("RV", sig.reason)

        def on_b2(sig):
            d = ("LONG" if sig.direction.name == "LONG"
                 else "SHORT" if sig.direction.name == "SHORT" else "FLAT")
            if sig.event == "ENTRY" and b2.position is not None:
                p = b2.position
                # B2: green TP from strategy, SL = disaster (per config)
                self._send_entry("B2", d, sl_price=None, tp_price=p.green_val,
                                  entry_ts=sig.timestamp, signal_entry_price=sig.price)
            elif sig.event == "EXIT":
                if sig.reason == "TP_FIXED":
                    # NT8 path drops the local tag (bracket already filled).
                    # On MT5 the TP bracket also fires — broker closes, we just untrack.
                    with self._lock:
                        if "B2" in self._open:
                            self._open.pop("B2", None)
                            self.n_exits_sent += 1
                            print(f"  [mt5][{self.firm_label}][B2-TP_FIXED] bracket filled by broker; untracked")
                else:
                    self._close_strategy("B2", sig.reason)

        def on_od(sig):
            d = "LONG" if sig.direction.name == "LONG" else "FLAT"
            if sig.event == "ENTRY" and od.position is not None:
                # OD: no SL/TP from strategy — use disaster SL only.
                # NOTE: sig.qty (1 or 2 for martingale) is INTENTIONALLY ignored.
                # On 5%ers/FN prop accounts the 2c martingale doubles risk to
                # ~9% of account on a worst-case SL hit, which blows the 5%
                # daily-unrealized cap. NT8 honors martingale; MT5 doesn't.
                # If qty=2 fires, log it for visibility but still trade base lots.
                if sig.qty == 2:
                    print(f"  [mt5][{self.firm_label}][OD-MARTINGALE] engine signals 2c, "
                          f"sending base lots only ({self.lots.get('OD', 0.01)}) for prop firm safety")
                self._send_entry("OD", d, sl_price=None, tp_price=None,
                                  entry_ts=sig.timestamp, signal_entry_price=sig.price)
            elif sig.event == "EXIT":
                self._close_strategy("OD", sig.reason)

        def on_fb(sig):
            d = "LONG"
            if sig.event == "ENTRY" and fb.position is not None:
                p = fb.position
                self._send_entry("FB", d, sl_price=p.sl_price, tp_price=p.tp_price,
                                  entry_ts=sig.timestamp, signal_entry_price=sig.price)
            elif sig.event == "EXIT" and sig.reason == "EOD":
                self._close_strategy("FB", sig.reason)

        print(f"[mt5][{self.firm_label}] callbacks created for RV+B2+OD+FB")
        return {"RV": on_rv, "B2": on_b2, "OD": on_od, "FB": on_fb}

    # ---------- heartbeat (light: just a connectivity ping, no flatten-on-stale) ----------

    def start_heartbeat(self) -> None:
        if self._hb_thread is not None and self._hb_thread.is_alive():
            return
        self._stop_event.clear()

        def _hb_loop():
            print(f"[mt5][{self.firm_label}] heartbeat thread started ({HEARTBEAT_INTERVAL_SEC}s)")
            while not self._stop_event.is_set():
                if not self.dry_run and _MT5_AVAILABLE and self._initialized:
                    try:
                        ti = mt5.terminal_info()
                        if ti is None or not ti.connected:
                            print(f"[mt5][{self.firm_label}] WARN terminal not connected")
                    except Exception:
                        pass
                self.n_heartbeats += 1
                self._stop_event.wait(HEARTBEAT_INTERVAL_SEC)
            print(f"[mt5][{self.firm_label}] heartbeat thread stopped")

        self._hb_thread = threading.Thread(target=_hb_loop, daemon=True, name="mt5_heartbeat")
        self._hb_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._hb_thread:
            self._hb_thread.join(timeout=3)
        if not self.dry_run and _MT5_AVAILABLE and self._initialized:
            try:
                mt5.shutdown()
            except Exception:
                pass

    def summary(self) -> dict:
        return {
            "firm": self.firm_label,
            "entries_sent": self.n_entries_sent,
            "exits_sent": self.n_exits_sent,
            "heartbeats": self.n_heartbeats,
            "errors": self.n_errors,
            "dry_run_logs": self.n_dry_run_logs,
            "open_positions": len(self._open),
            "open_strats": list(self._open.keys()),
            "dry_run": self.dry_run,
        }
