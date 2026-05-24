"""Phase 1 standalone runner — exercises data feed + bar builders + settle recorder.

Three modes:

  --replay        Feed historical pickle bars through the system. Tests bar builder
                  logic + warm-start without needing live connection.

  --warm-only     Run warm-start only, then exit. Use to verify indicator seeding works.

  --live          Connect to Databento live, stream NQ ticks, build bars in real time.
                  Warm-starts first, then streams. Ctrl+C to stop.

Usage:
    python live/combined/run_phase1.py --replay
    python live/combined/run_phase1.py --warm-only
    python live/combined/run_phase1.py --live
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

import pandas as pd

# Load DATABENTO_API_KEY from .env BEFORE any module that needs it imports
# (rolling_cache.py and data_feed.py both read it at runtime).
from dotenv import load_dotenv
load_dotenv()

# Ensure repo root is on path so 'from live.combined.xxx import yyy' works.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from live.combined.bar_builder import MultiBarBuilder, Bar
from live.combined.warm_start import feed_warm_start_to_builders
from live.combined.settle_recorder import SettleRecorder
from live.combined.config import BAR_5MIN_SECS, BAR_20MIN_SECS, WARM_START_TRADING_DAYS


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def wire_paper_engines(b5, b20):
    """Wire RV / B2 / OD / Fabio engines as paper subscribers.

    Engines subscribe to bar builders. During warm-start they receive
    historical bars to seed indicator state (ATR, EMA, kernel, gamma).
    During warm-start the _log function is SUPPRESSED so historical-bar
    signals don't pollute the live logs. Call `set_live_mode()` returned
    in the dict after warm-start completes to start logging real signals.
    """
    import csv
    from datetime import datetime
    import pandas as pd
    from live.combined.config import STATE_DIR, ET_TZ
    from live.combined.rv_engine import RVEngine, Signal as RVSig
    from live.combined.b2_engine import B2Engine, B2Signal
    from live.combined.od_engine import ODEngine, ODSignal
    from live.combined.fabio_orb_engine import FabioORBEngine, FabioSignal

    paper_dir = STATE_DIR / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    # Per-strategy CSVs (append-mode, one row per signal)
    log_files = {
        "RV": paper_dir / "live_rv_signals.csv",
        "B2": paper_dir / "live_b2_signals.csv",
        "OD": paper_dir / "live_od_signals.csv",
        "FB": paper_dir / "live_fb_signals.csv",
    }
    header = ["ts_et","strat","event","direction","price","qty","reason"]
    for p in log_files.values():
        if not p.exists():
            with open(p, "w", newline="") as f:
                csv.writer(f).writerow(header)

    # Warmup gate — flipped to True after warm-start finishes.
    # When False, _log silently absorbs signals (engines still update state).
    state = {"live": False}
    def set_live_mode():
        state["live"] = True

    def _log(strat, ev, direction, price, qty, reason, ts, extra: str = ""):
        if not state["live"]:
            return    # suppress warm-start signals
        ts_et = pd.Timestamp(ts)
        if ts_et.tzinfo is None:
            ts_et = ts_et.tz_localize(ET_TZ)
        else:
            ts_et = ts_et.tz_convert(ET_TZ)
        line = [ts_et.strftime("%Y-%m-%d %H:%M:%S %Z"), strat, ev,
                str(direction), f"{price:.2f}", str(qty), reason or ""]
        with open(log_files[strat], "a", newline="") as f:
            csv.writer(f).writerow(line)
        print(f"  [{strat}-{ev}] {ts_et.strftime('%H:%M')} {direction} @ {price:.2f} "
              f"qty={qty} {reason} {extra}".rstrip())

    # ---- RV ----
    rv = RVEngine()
    def _on_rv(sig: RVSig):
        d = "LONG" if sig.direction.name == "LONG" else "SHORT" if sig.direction.name == "SHORT" else "FLAT"
        extra = ""
        if sig.event == "ENTRY" and rv.position is not None:
            p = rv.position
            extra = f"SL={p.stop_price:.2f} TP={p.target_price:.2f}"
        _log("RV", sig.event, d, sig.price, 1, sig.reason, sig.timestamp, extra)
    # NOTE: do NOT subscribe _on_rv directly to rv. Coordinator (Phase 5)
    # will route signals through no-hedge filter then call _on_rv. See below.
    b20.subscribe(rv.on_bar)

    # ---- B2 ----
    # Phase 3b: live signal computation from 5-min bars. The B2SignalEngine
    # subscribes to 5-min bars, runs Mech-B detection + confirmation logic,
    # and queues entries into B2Engine via set_pending_entry().
    b2 = B2Engine()
    # NOTE: gamma_sign loading is handled by B2SignalEngine below
    # (it reads the parquet AND hot-reloads it on each session boundary,
    # so new gamma rows added by run_daily.py take effect without restart).
    def _on_b2(sig: B2Signal):
        d = "LONG" if sig.direction.name == "LONG" else "SHORT" if sig.direction.name == "SHORT" else "FLAT"
        extra = ""
        if sig.event == "ENTRY" and b2.position is not None:
            p = b2.position
            extra = f"yellow={p.yellow_val:.2f} green={p.green_val:.2f}"
        _log("B2", sig.event, d, sig.price, sig.qty, sig.reason, sig.timestamp, extra)
    # (coordinator routes; not subscribed directly)
    # B2 needs (bar, current_atr_y). We compute ATR online via a simple wrapper.
    from live.combined.od_engine import _ATR
    b2_atr = _ATR(length=14)
    def _b2_on_bar(bar):
        atr = b2_atr.update(bar.high, bar.low, bar.close)
        b2.on_20min_bar(bar, current_atr_y=atr if atr == atr else None)  # nan check
    b20.subscribe(_b2_on_bar)

    # B2 SIGNAL engine (Phase 3b) — consumes 5-min bars, queues entries into B2Engine
    try:
        from live.combined.b2_signal_engine import B2SignalEngine
        b2_sig = B2SignalEngine(b2_engine=b2)
        b5.subscribe(b2_sig.on_5min_bar)
        print(f"  [B2] live signal engine wired (gamma levels: {b2_sig._levels is not None}, "
              f"{0 if b2_sig._levels is None else len(b2_sig._levels)} levels for current session)")
    except Exception as e:
        print(f"  [B2] signal engine init FAILED: {e} — B2 will be entries-silent")

    # ---- OD ----
    od = ODEngine()
    def _on_od(sig: ODSignal):
        d = "LONG" if sig.direction.name == "LONG" else "FLAT"
        extra = ""
        if sig.event == "ENTRY" and od.position is not None:
            p = od.position
            extra = f"yellow={p.yellow_val:.2f} (TP green decays per bar)"
        _log("OD", sig.event, d, sig.price, sig.qty, sig.reason, sig.timestamp, extra)
    # (coordinator routes; not subscribed directly)
    b20.subscribe(od.on_20min_bar)

    # ---- Fabio ORB ----
    fb = FabioORBEngine()
    def _on_fb(sig: FabioSignal):
        d = "LONG" if sig.direction.name == "LONG" else "FLAT"
        extra = ""
        if sig.event == "ENTRY" and fb.position is not None:
            p = fb.position
            extra = f"SL={p.sl_price:.2f} TP={p.tp_price:.2f} (4R)"
        _log("FB", sig.event, d, sig.price, sig.qty, sig.reason, sig.timestamp, extra)
    # (coordinator routes; not subscribed directly)
    b5.subscribe(fb.on_5min_bar)

    print(f"  [paper] wired RV+B2+OD+Fabio (WARMUP mode — logging suppressed).")
    return {"engines": {"RV": rv, "B2": b2, "OD": od, "FB": fb},
            "log_files": log_files,
            "set_live_mode": set_live_mode,
            "paper_callbacks": {"RV": _on_rv, "B2": _on_b2, "OD": _on_od, "FB": _on_fb}}


def make_print_handler(name: str, max_to_print: int = 5):
    """Return a callback that prints the first N bars then becomes silent (just counts)."""
    state = {"count": 0}
    def handler(bar: Bar):
        state["count"] += 1
        if state["count"] <= max_to_print:
            print(f"  [{name}] bar {state['count']}: "
                  f"{bar.open_time.strftime('%Y-%m-%d %H:%M')} ET  "
                  f"O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} C={bar.close:.2f}  "
                  f"buy={bar.buy_vol} sell={bar.sell_vol} (delta={bar.delta:+})  "
                  f"ticks={bar.tick_count}")
        elif state["count"] == max_to_print + 1:
            print(f"  [{name}] ... ({state['count']}+ bars; suppressing further output)")
    handler._count = lambda: state["count"]
    return handler


def build_pipeline():
    """Wire bar builders + settle recorder. Returns (multi, b5, b20, settle_rec)."""
    multi = MultiBarBuilder()
    b5 = multi.add_timeframe(BAR_5MIN_SECS, name="5min")
    b20 = multi.add_timeframe(BAR_20MIN_SECS, name="20min")

    # Settle recorder hooks into 5-min bars
    settle_rec = SettleRecorder()
    b5.subscribe(settle_rec.on_bar)

    # Debug print handlers
    b5.subscribe(make_print_handler("5min"))
    b20.subscribe(make_print_handler("20min"))

    return multi, b5, b20, settle_rec


def run_replay():
    """Feed pickle archive through bar builders."""
    print("=" * 70)
    print("PHASE 1 REPLAY MODE — feed pickle archive through bar builders")
    print("=" * 70)
    multi, b5, b20, settle_rec = build_pipeline()

    print(f"\nWarm-start: feeding last {WARM_START_TRADING_DAYS} trading days of 5-min bars...")
    n = feed_warm_start_to_builders(multi, n_trading_days=WARM_START_TRADING_DAYS)
    print(f"  fed {n} bars")
    print(f"\nBar emission counts:")
    print(f"  5-min bars emitted:  {b5.n_bars_emitted}")
    print(f"  20-min bars emitted: {b20.n_bars_emitted}")
    print(f"  Settles captured: {len(settle_rec.settles)}")
    if settle_rec.settles:
        last_n = list(sorted(settle_rec.settles.keys()))[-5:]
        print(f"  Last 5 settle dates:")
        for d in last_n:
            print(f"    {d}: NQ close ${settle_rec.settles[d]:.2f}")
    print()
    return 0


def run_warm_only():
    """Same as replay but emphasizes warm-start verification."""
    print("=" * 70)
    print("PHASE 1 WARM-START VERIFICATION")
    print("=" * 70)
    return run_replay()


def run_live(execute_to_nt8: bool = False,
             execute_to_mt5: bool = False,
             mt5_dry_run: bool = False):
    """Connect to Databento live, warm-start, then stream.

    execute_to_nt8: if True, also send tagged orders to NT8 multi-strat addon.
        Requires NQMultiStratReceiver running on port 8081. Default False = paper only.

    execute_to_mt5: if True, ALSO send orders to MT5 (alongside NT8 if both enabled).
        Default tiny lots (0.01) for safe verification. Requires MT5 desktop app
        running and authorized on the configured broker server.

    mt5_dry_run: if True, MT5 executor logs intended orders but DOES NOT send.
        Use for routing-validation before risking real fills.
    """
    print("=" * 70)
    print("PHASE 1 LIVE MODE — Databento NQ live ticks")
    if execute_to_nt8:
        print(">>> NT8 EXECUTE MODE ENABLED <<< — signals will route to NT8 addon")
    if execute_to_mt5:
        mode = "DRY-RUN (log only)" if mt5_dry_run else "LIVE (real orders)"
        print(f">>> MT5 EXECUTE MODE ENABLED ({mode}) <<< — signals will also route to MT5")
    if not (execute_to_nt8 or execute_to_mt5):
        print("    (paper mode — signals logged to CSV, nothing sent to brokers)")
    print("Press Ctrl+C to stop")
    print("=" * 70)

    # Ensure rolling 15-day cache is fresh BEFORE warm-start.
    # Fetches any missing days from Databento Historical (which has a 30-min
    # delay, but we only ever request days that ended >24h ago so the delay
    # is never an issue). Prunes anything older than 15 trading days.
    try:
        from live.combined.rolling_cache import RollingWarmstartCache, print_status
        print("\nRefreshing rolling warm-start cache...")
        cache = RollingWarmstartCache()
        status = cache.ensure_recent_cached()
        print_status(status)
    except Exception as e:
        print(f"[run_phase1] cache refresh failed: {e} — falling back to research archive")

    multi, b5, b20, settle_rec = build_pipeline()

    # Wire strategy engines BEFORE warm-start so the historical bar replay
    # also seeds each engine's indicator state (ATR, EMA, kernel, gamma).
    # In warmup mode, _log silently absorbs signals; we flip to live AFTER
    # warm-start so only real signals get logged/printed.
    paper_logs = wire_paper_engines(b5, b20)

    print(f"\nWarm-start: feeding last {WARM_START_TRADING_DAYS} trading days of 5-min bars...")
    feed_warm_start_to_builders(multi, n_trading_days=WARM_START_TRADING_DAYS)
    print(f"  warm-start complete. 5-min bars seeded: {b5.n_bars_emitted}, "
          f"20-min bars seeded: {b20.n_bars_emitted}")
    # Engine indicator-state sanity report
    eng = paper_logs["engines"]
    print(f"  engine warmup: RV bars_seen={eng['RV'].n_bars_seen}  "
          f"B2={eng['B2'].n_bars_seen}  OD={eng['OD'].n_bars_seen}  "
          f"FB={eng['FB'].n_bars_seen}")

    # Clear any positions left over from warm-start. These are HISTORICAL
    # artifacts — phantom positions that fired during warm-start (especially
    # from force_close_current() emitting partial bars on the last day).
    # Indicators stay warmed; only positions reset so live can fire cleanly.
    reset_count = 0
    for k, e in eng.items():
        if hasattr(e, "position") and e.position is not None:
            print(f"  [warmup-reset] {k} had phantom position from warm-start — clearing")
            e.position = None
            reset_count += 1
    if reset_count == 0:
        print(f"  no phantom positions to clear")
    print()

    # Attach a LIVE-ONLY printer that fires for every bar after warm-start.
    # (The default printer suppresses after 5 bars per builder; warm-start
    #  blew past that, so live bars would emit silently otherwise.)
    def live_printer(tag):
        def cb(bar: Bar):
            print(f"  [LIVE-{tag}] {bar.open_time.strftime('%Y-%m-%d %H:%M')} ET  "
                  f"O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} C={bar.close:.2f}  "
                  f"buy={bar.buy_vol} sell={bar.sell_vol} (Δ={bar.delta:+})  "
                  f"ticks={bar.tick_count}")
        return cb
    b5.subscribe(live_printer("5min"))
    b20.subscribe(live_printer("20min"))

    # Persist every 5-min bar to today's session pickle (rolling cache).
    # Tomorrow's startup will read today's now-complete pickle as part of the
    # 15-day warm-start chain — no historical fetch needed for days the server
    # was running.
    from live.combined.bar_persistor import LiveBarPersistor
    persistor = LiveBarPersistor()
    b5.subscribe(persistor.on_bar)
    print(f"[run_phase1] LIVE printer + bar persistor attached "
          f"(cache: {persistor.cache_dir}).\n")

    # ===== Flip paper engines to LIVE mode (signals now log) =====
    paper_logs["set_live_mode"]()
    print(f"[run_phase1] PAPER engines now in LIVE mode — signals will print + log to CSV.\n")

    # ===== Build the Coordinator (Phase 5) =====
    # Routes engine signals through no-hedge filter before reaching downstream
    # consumers (paper logger, NT8 executor). Blocks SECOND entries that would
    # hedge an already-open position in another strategy.
    from live.combined.coordinator import Coordinator
    coord = Coordinator()
    coord.load_state()   # restore if fresh (< 5 min old)
    eng = paper_logs["engines"]
    paper_cbs = paper_logs["paper_callbacks"]

    # ===== Wire NT8 executor (if --execute) =====
    nt8 = None
    nt8_cbs = {"RV": None, "B2": None, "OD": None, "FB": None}
    if execute_to_nt8:
        try:
            from live.combined.nt8_executor import NT8Executor
            nt8 = NT8Executor()
            nt8_cbs = nt8.get_callbacks(eng["RV"], eng["B2"], eng["OD"], eng["FB"])
            nt8.start_heartbeat()
            print(f"[run_phase1] NT8 EXECUTE wired — orders + heartbeat active.\n")
        except Exception as e:
            print(f"[run_phase1] FAILED to wire NT8 executor: {e}")
            print(f"  Continuing in paper-only mode.")

    # ===== Wire MT5 executor (if --execute-mt5) =====
    mt5x = None
    mt5_cbs = {"RV": None, "B2": None, "OD": None, "FB": None}
    if execute_to_mt5:
        try:
            from live.combined.mt5_executor import MT5Executor
            # 5%ers 100K High Stakes config — scale 1.5 from the asymmetric MC.
            # Targets: 86% pass rate, median 68 days, 0% demotion risk
            # (5.78% daily-unrealized bust, 8.04% DD bust).
            # Source: live/combined deployment plan/the5ers_100k_challenge.csv
            # OD runs solo overnight (3% slot). B2/RV/FB share the day-session
            # 3% cap; SL distances were chosen to clear each strategy's worst MAE.
            mt5x = MT5Executor(
                dry_run=mt5_dry_run,
                firm_label="5pct-100K",
                lots_per_strat={
                    "OD": 0.75,   # 3.75 MNQ equivalent
                    "B2": 0.25,   # 1.25 MNQ equivalent
                    "RV": 0.75,   # 3.75 MNQ equivalent
                    "FB": 1.00,   # 5.00 MNQ equivalent
                },
                sl_pts_per_strat={
                    "OD": 600.0,  # worst MAE 543 pt + 57 margin
                    "B2": 600.0,  # worst MAE 550 pt + 50 margin
                    "RV": 200.0,  # worst MAE 150 pt + 50 margin
                    "FB": 150.0,  # worst MAE 100 pt + 50 margin
                },
            )
            mt5_cbs = mt5x.get_callbacks(eng["RV"], eng["B2"], eng["OD"], eng["FB"])
            mt5x.start_heartbeat()
            print(f"[run_phase1] MT5 EXECUTE wired (dry_run={mt5_dry_run}) "
                  f"@ 5%ers 100K sizing (scale 1.5).\n")
        except Exception as e:
            print(f"[run_phase1] FAILED to wire MT5 executor: {e}")
            print(f"  Continuing without MT5.")

    # Coordinator subscribes to each engine. Approved signals fan out to
    # paper logger + NT8 callback (if --execute) + MT5 callback (if --execute-mt5).
    # Blocked signals are dropped (with rollback of the engine's phantom position).
    for strat_name in ("RV", "B2", "OD", "FB"):
        downstream = [paper_cbs[strat_name]]
        if nt8_cbs[strat_name] is not None:
            downstream.append(nt8_cbs[strat_name])
        if mt5_cbs[strat_name] is not None:
            downstream.append(mt5_cbs[strat_name])
        coord.register(strat_name, eng[strat_name], *downstream)
    print(f"[run_phase1] COORDINATOR wired — no-hedge rule active across 4 strats.\n")

    # ===== Session backfill (T+30 and T+60) — fills pre-startup session gap =====
    # If you start after 18:00 ET (mid-session), Databento Historical has the
    # pre-startup bars but not until 30 min after they happened. Two passes
    # at T+30 and T+60 fetch [session_start, startup_time] and merge into
    # today's pickle (live bars win on overlap).
    try:
        from live.combined.session_backfill import schedule_session_backfill
        now_et = pd.Timestamp.now(tz="America/New_York")
        schedule_session_backfill(now_et)
    except Exception as e:
        print(f"[backfill] failed to schedule: {e}")

    from live.combined.data_feed import DatabentoLiveFeed
    feed = DatabentoLiveFeed(on_tick=multi.on_tick)

    def graceful_stop(sig, frame):
        print("\n[run_phase1] caught signal, stopping...")
        feed.stop()
    signal.signal(signal.SIGINT, graceful_stop)

    feed.start()  # blocking

    print(f"\nLive session stats:")
    print(f"  Databento msgs received: {feed.n_msgs_received}")
    print(f"  Trade ticks enqueued:    {feed.n_trades_enqueued}")
    print(f"  Trade ticks processed:   {feed.n_trades_processed}")
    print(f"  Local drops (q full):    {feed.n_dropped}")
    print(f"  Callback errors:         {feed.n_callback_errors}")
    print(f"  5-min bars emitted live: {b5.n_bars_emitted}")
    print(f"  20-min bars emitted live: {b20.n_bars_emitted}")
    print(f"  5-min bars persisted to cache: {persistor.n_bars_written}")
    print(f"  Settles captured this session: {len(settle_rec.settles)}")
    print(f"\nPaper-mode strategy stats:")
    eng = paper_logs["engines"]
    for k, e in eng.items():
        print(f"  {k:<3}  bars_seen={e.n_bars_seen:>6}  "
              f"entries={getattr(e, 'n_entries', 0):>3}  "
              f"exits={getattr(e, 'n_exits', 0):>3}")
    print(f"  Logs: {paper_logs['log_files']['RV'].parent}")
    # Coordinator stats + persist state
    cs = coord.summary()
    print(f"\nCoordinator stats:")
    print(f"  Approved: {cs['approved']}   Blocked by no-hedge: {cs['blocked']}")
    if cs['currently_open']:
        print(f"  Open positions at shutdown: {cs['currently_open']}")
    if cs['blocked_log']:
        print(f"  Last 3 blocked signals:")
        for b in cs['blocked_log'][-3:]:
            print(f"    {b['ts']}  {b['strat']} {b['direction']} blocked by "
                  f"{b['blocked_by']}={b['blocked_by_dir']}")
    try:
        coord.save_state()
        print(f"  [coord] state saved.")
    except Exception as e:
        print(f"  [coord] state save failed: {e}")

    if nt8 is not None:
        s = nt8.summary()
        print(f"\nNT8 executor stats:")
        print(f"  Entries sent: {s['entries_sent']}  Exits sent: {s['exits_sent']}")
        print(f"  Heartbeats: {s['heartbeats']}  Errors: {s['errors']}")
        print(f"  Open positions (still in NT8): {s['open_positions']}")
        if s['open_tags']:
            for t in s['open_tags']:
                print(f"    - {t}")
        nt8.stop()
    if mt5x is not None:
        s = mt5x.summary()
        print(f"\nMT5 executor stats ({s['firm']}, dry_run={s['dry_run']}):")
        print(f"  Entries sent: {s['entries_sent']}  Exits sent: {s['exits_sent']}")
        print(f"  Heartbeats: {s['heartbeats']}  Errors: {s['errors']}")
        print(f"  Dry-run logs: {s['dry_run_logs']}")
        print(f"  Open positions (tracked on MT5): {s['open_positions']}")
        if s['open_strats']:
            print(f"  Open strats: {s['open_strats']}")
        mt5x.stop()
    return 0


def main():
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--replay", action="store_true", help="Feed pickle archive only")
    g.add_argument("--warm-only", action="store_true", help="Warm-start verification only")
    g.add_argument("--live", action="store_true", help="Connect to Databento live")
    ap.add_argument("--execute", action="store_true",
                     help="ALSO send tagged orders to NT8 multi-strat addon "
                          "(requires NQMultiStratReceiver running on port 8081). "
                          "Default: paper-only (CSV logging, no orders sent).")
    ap.add_argument("--execute-mt5", action="store_true",
                     help="ALSO send orders to MT5 (FundedNext by default). "
                          "Defaults to tiny lots (0.01) for safe verification. "
                          "Requires MT5 desktop running + authorized.")
    ap.add_argument("--mt5-dry-run", action="store_true",
                     help="With --execute-mt5: log intended MT5 orders but DO NOT send. "
                          "Use to validate routing before risking real fills.")
    args = ap.parse_args()

    if args.replay:
        return run_replay()
    elif args.warm_only:
        return run_warm_only()
    elif args.live:
        return run_live(execute_to_nt8=args.execute,
                        execute_to_mt5=args.execute_mt5,
                        mt5_dry_run=args.mt5_dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
