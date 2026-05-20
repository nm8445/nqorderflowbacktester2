"""
NQ Order Flow Live Trader — Combined VWAP Strategy

=== HOW TO START TRADING ===

Step 1: Open NinjaTrader 8
  - Control Center -> New -> Add-On -> NQ Order Flow Signal Receiver
  - Check "Auto Execute" checkbox
  - Click "Start Server" (status turns green, listening on port 8080)
  - Make sure instrument is set to MNQ and account is correct

Step 2: Run this script
  - python live/live_trader.py
  - Config GUI pops up — pick Fullport, Funded, or Custom
  - Set account name to match NT8 (e.g. Sim101)
  - Click "Start Trading"
  - Script connects to Databento, warm starts from session history, then goes live

Step 3: Trading is live
  - Watch the terminal for signal logs
  - Watch NT8 for order fills
  - Ctrl+C to stop the Python side (NT8 keeps running independently)

Notes:
  - NT8 must be running BEFORE you start this script (Python sends HTTP to NT8)
  - If starting at 6pm ET (session open), use --no-warm-start since there's no history yet
  - Paper mode (--paper) logs signals but sends nothing to NT8

=== STRATEGY ===

Two complementary entry modes, shared position lock (one trade at a time):

  1. BASE (VWAP Reaction) — ADX 15-30
     - Absorption at VWAP zone (+/- 3 pts) with matching 5-min bias
     - SL 0.50x ATR, TP 1.90x ATR

  2. TRENDING (STD1 Band Reaction) — ADX 30+
     - 14 consecutive 5-min closes above/below VWAP = trending
     - Longs at upper STD1 band (uptrend), shorts at lower STD1 band (downtrend)
     - SL 1.00x ATR, TP 1.00x ATR

  Common:
  - Session VWAP (anchored 6pm ET) as dynamic support/resistance
  - Entry window: 7:10pm ET to 4:00 PM ET (14 bars warmup)
  - Force close 4:58 PM ET
  - One trade at a time, no breakeven
  - Smart order routing: limit/stop orders 9:30 AM - 12:00 PM ET only

=== USAGE ===

    python live/live_trader.py                  # GUI config, warm start
    python live/live_trader.py --paper          # paper mode (no orders)
    python live/live_trader.py --no-gui         # skip GUI, use fullport defaults
    python live/live_trader.py --no-warm-start  # skip warm start (cold start at 6pm)
    python live/live_trader.py --paper --no-gui --no-warm-start  # combine flags
"""
import argparse
import logging
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nqbt.data.schema import PRICE_SCALE
from live.bar_builder import LiveBarBuilder, LiveTimeBarBuilder
from live.signal_engine import LiveSignalEngine, Signal
from live.signal_client import (
    send_signal_to_nt8,
)
from live.warm_start import warm_start_engine
from live.config_gui import TradingConfig, calculate_contracts, show_config_gui, MNQ_POINT_VALUE
from live.account_manager import AccountManager, Account

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Data feed config
DATABENTO_API_KEY = os.environ.get("DATABENTO_API_KEY")
DATASET = "GLBX.MDP3"
SCHEMA = "mbp-1"
SYMBOL = "NQ.c.0"
STYPE = "continuous"
RANGE_TICKS = 40
TIME_BAR_MINUTES = 5
ATR_PERIOD = 14

ET = "America/New_York"

TICK_SIZE = 0.25


class LiveTrader:
    def __init__(self, config: TradingConfig, warm_start: bool = True):
        self.config = config
        self.warm_start = warm_start
        self._running = False
        self._current_price: float = 0.0


        self.bar_builder = LiveBarBuilder(range_ticks=RANGE_TICKS)
        self.time_bar_builder = LiveTimeBarBuilder(interval_minutes=TIME_BAR_MINUTES)
        self.engine = LiveSignalEngine(
            atr_period=ATR_PERIOD,
            stop_mult=config.sl_mult,
            target_mult=config.tp_mult,
            trend_stop_mult=config.trend_sl_mult,
            trend_target_mult=config.trend_tp_mult,
        )

        # Account rotation (Lucid multi-account) or single-account fallback.
        # State persists across restarts via JSON (equity/HWM/DD/best_day/rotation_idx).
        acct_list = config.accounts if config.accounts else [config.account]
        state_dir = Path(__file__).parent / "state"
        state_name = "_".join(acct_list) + ".json"
        self.account_mgr = AccountManager(acct_list, state_path=state_dir / state_name)
        self.rotation_enabled = bool(config.accounts and len(config.accounts) > 1)

        # Track the open trade's account + contracts + entry + direction for PnL calc on exit
        self._active_account: Optional[Account] = None
        self._active_contracts: int = 0
        self._active_entry_price: Optional[float] = None
        self._active_direction: int = 0  # 1 long, -1 short



    def _get_contracts(self, atr: float) -> int:
        """Calculate contract count based on config mode."""
        if self.config.contracts is not None:
            return self.config.contracts
        return calculate_contracts(atr, self.config.sl_mult, self.config.target_risk)

    def start(self):
        """Start live data stream and process ticks."""
        if not DATABENTO_API_KEY:
            log.error("DATABENTO_API_KEY not set in .env file")
            sys.exit(1)

        cfg = self.config
        if not cfg.paper:
            log.info("[trader] NT8 Add-On expected on http://localhost:8080/")
            log.info("[trader] Make sure NQ Order Flow Signal Receiver is running")

        log.info(f"[trader] Strategy: Combined VWAP (base + trending band)")
        log.info(f"[trader] Mode: {cfg.mode.upper()} | Base: SL={cfg.sl_mult}x/TP={cfg.tp_mult}x ADX 15-30 | Trend: SL={cfg.trend_sl_mult}x/TP={cfg.trend_tp_mult}x ADX 30+")
        if self.rotation_enabled:
            acct_names = " / ".join(a.name for a in self.account_mgr.accounts)
            log.info(f"[trader] Instrument: {cfg.instrument} | Rotation: {acct_names} (alternating)")
        else:
            log.info(f"[trader] Instrument: {cfg.instrument} | Account: {cfg.account}")
        if cfg.contracts:
            log.info(f"[trader] Contracts: {cfg.contracts} fixed")
        else:
            log.info(f"[trader] Contracts: dynamic (target risk ${cfg.target_risk:.0f})")
        log.info(f"[trader] Paper: {cfg.paper}")
        log.info(f"[trader] Entry window: 7:10pm-4:00pm ET | Force close: 4:58pm ET")
        log.info(f"[trader] Connecting to Databento live feed...")

        # Warm start: load overnight session data
        if self.warm_start:
            warm_start_engine(self.engine, DATABENTO_API_KEY)

        self._running = True

        try:
            import databento as db

            client = db.Live(key=DATABENTO_API_KEY)
            client.subscribe(
                dataset=DATASET,
                schema=SCHEMA,
                stype_in=STYPE,
                symbols=[SYMBOL],
            )

            log.info("[trader] Live feed connected. Waiting for ticks...")

            for record in client:
                if not self._running:
                    break
                self._on_record(record)

        except KeyboardInterrupt:
            log.info("[trader] Interrupted by user.")
        except Exception as e:
            log.error(f"[trader] Fatal error: {e}", exc_info=True)
        finally:
            self._shutdown()

    def _on_record(self, record):
        """Process one MBP-1 record from live feed."""
        action = getattr(record, "action", None)
        if hasattr(action, "value"):
            action = action.value
        if action != "T":
            return

        side = getattr(record, "side", None)
        if hasattr(side, "value"):
            side = side.value
        if side not in ("A", "B"):
            return

        price = record.price / PRICE_SCALE
        size = record.size
        ts = pd.Timestamp(record.ts_event, unit="ns", tz="UTC")
        aggressor = "sell" if side == "A" else "buy"

        # Track current price for smart routing
        self._current_price = price

        # Feed tick to engine for VWAP update + stop/target checks
        tick_output = self.engine.on_tick(price, size=size, ts=ts)
        if tick_output.signal != Signal.NONE:
            self._handle_exit(tick_output, price)
            return

        # Feed tick to time bar builder (5-min bars for ATR + bias)
        closed_time_bar = self.time_bar_builder.on_tick(ts, price, size, aggressor)
        if closed_time_bar is not None:
            self.engine.on_time_bar_close(closed_time_bar)

        # Feed tick to range bar builder (40-tick range bars for signals)
        closed_bar = self.bar_builder.on_tick(ts, price, size, aggressor)
        if closed_bar is None:
            return

        # New range bar closed — check for entry signals
        bar_output = self.engine.on_bar_close(closed_bar)
        if bar_output.signal != Signal.NONE:
            # Measure feed lag: how stale is this bar's closing tick vs wall clock?
            feed_lag_ms = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() * 1000
            self._entry_t0 = time.perf_counter()
            log.info(f"[latency] feed lag at signal: {feed_lag_ms:.1f}ms")
            self._handle_entry(bar_output, closed_bar.close)

    def _handle_entry(self, output, price):
        """Handle entry signal — decide market vs limit/stop, compute contracts."""
        sig = output.signal
        if sig not in (Signal.ENTER_LONG, Signal.ENTER_SHORT):
            return

        pos = output.position
        stop = pos.stop_price
        target = pos.target_price
        entry_price = pos.entry_price
        direction = "long" if sig == Signal.ENTER_LONG else "short"
        action = "BUY" if direction == "long" else "SELL"
        source = pos.source.upper()  # "BASE" or "TREND"

        # Reset daily PnL tracker based on entry timestamp (covers 6pm ET session roll)
        self.account_mgr.reset_daily_for_all(pos.entry_time)

        # Pick account for this trade (rotation or single)
        acct = self.account_mgr.next_account()
        if acct is None:
            log.warning(f"[signal] [{source}] ENTRY skipped — all accounts blown/passed. {self.account_mgr.status_summary()}")
            # Clear engine position so it doesn't stay locked
            self.engine._position = None
            return

        # Calculate contract count — use the right SL mult for the source
        atr = self.engine._calculate_atr()
        sl_mult = self.config.sl_mult if pos.source == "base" else self.config.trend_sl_mult
        if atr is None:
            atr = abs(entry_price - stop) / sl_mult  # fallback
        contracts = self._get_contracts(atr)
        sl_pts = atr * sl_mult
        risk_dollars = sl_pts * MNQ_POINT_VALUE * contracts

        # Remember which account owns the open trade (for PnL calc on exit)
        self._active_account = acct
        self._active_contracts = contracts
        self._active_entry_price = entry_price
        self._active_direction = 1 if direction == "long" else -1

        rot_tag = f" [acct={acct.name}]" if self.rotation_enabled else ""
        log.info(f"[signal] [{source}]{rot_tag} ENTRY {action} @ {entry_price:.2f} | "
                 f"stop={stop:.2f} target={target:.2f} | "
                 f"{contracts} MNQ (risk ${risk_dollars:.0f})")

        if self.config.paper:
            return

        if hasattr(self, "_entry_t0"):
            dt_ms = (time.perf_counter() - self._entry_t0) * 1000
            log.info(f"[latency] signal→POST decision: {dt_ms:.1f}ms")

        send_signal_to_nt8(
            action, stop, target,
            quantity=contracts,
            order_type="MARKET",
            account=acct.name,
        )

    def _handle_exit(self, output, price):
        """Handle exit signal from engine. Compute PnL, update active account."""
        sig = output.signal
        exit_price = output.bar_close if output.bar_close else price

        if sig == Signal.EXIT_STOP:
            log.info(f"[signal] EXIT STOP @ ~{exit_price:.2f}")
        elif sig == Signal.EXIT_TARGET:
            log.info(f"[signal] EXIT TARGET @ ~{exit_price:.2f}")

        # Update account PnL if we have an active account + the exit has context
        acct = self._active_account
        contracts = self._active_contracts
        if acct is not None and contracts > 0:
            # Reconstruct direction + entry from engine's last position (cleared by engine on exit,
            # so we need to re-derive from exit_price vs stop/target). Simpler: tag signal direction.
            # The engine clears _position BEFORE returning, so we can't read entry here.
            # Instead: engine now returns bar_close = stop_price or target_price on exit.
            # PnL = (exit - entry) * dir * contracts * point_value — but we lost entry.
            # Workaround: track entry + direction on _active_* at entry time.
            entry = self._active_entry_price
            direction = self._active_direction
            if entry is not None and direction != 0:
                pnl = (exit_price - entry) * direction * contracts * MNQ_POINT_VALUE
                self.account_mgr.on_trade_exit(acct, pnl)

        self._active_account = None
        self._active_contracts = 0
        self._active_entry_price = None
        self._active_direction = 0

    def _shutdown(self):
        """Cleanup on shutdown."""
        self._running = False
        log.info("[trader] Shutdown complete.")


def default_config(paper: bool = False) -> TradingConfig:
    """Default fullport config for --no-gui mode."""
    return TradingConfig(
        mode="fullport",
        sl_mult=0.50,
        tp_mult=1.90,
        trend_sl_mult=1.00,
        trend_tp_mult=1.00,
        instrument="MNQ 06-26",
        account="Sim101",
        contracts=None,
        target_risk=1000.0,
        paper=paper,
    )


def main():
    parser = argparse.ArgumentParser(description="NQ Order Flow Live Trader - VWAP Reaction")
    parser.add_argument("--paper", action="store_true", help="Paper mode - no orders sent")
    parser.add_argument("--no-gui", action="store_true", help="Skip config GUI, use fullport defaults")
    parser.add_argument("--no-warm-start", action="store_true", help="Skip warm start (start cold)")
    args = parser.parse_args()

    if args.no_gui:
        config = default_config(paper=args.paper)
    else:
        config = show_config_gui()
        if config is None:
            print("Cancelled.")
            sys.exit(0)
        # CLI --paper overrides GUI
        if args.paper:
            config.paper = True

    trader = LiveTrader(config=config, warm_start=not args.no_warm_start)
    trader.start()


if __name__ == "__main__":
    main()
