"""Replay today's session from Databento Historical + diff vs live signals.

Workflow:
  1. Fetch all NQ trades for the requested date's CME session from Databento Historical
  2. Feed them through the same bar_builder + engines as run_phase1.py
  3. Each engine emits signals to a "replay" log
  4. Diff replay signals vs the live signals CSV (live/combined/state/paper/*.csv)
  5. Print match/mismatch report per strategy

Usage:
  python live/combined/replay_today.py                       # today's date
  python live/combined/replay_today.py --date 2026-05-19
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

from live.combined.config import (
    BAR_5MIN_SECS, BAR_20MIN_SECS, ET_TZ, STATE_DIR,
    DATABENTO_DATASET, DATABENTO_SYMBOL, DATABENTO_SCHEMA, DATABENTO_STYPE,
)
from live.combined.bar_builder import MultiBarBuilder, Bar
from live.combined.rv_engine import RVEngine
from live.combined.b2_engine import B2Engine
from live.combined.b2_signal_engine import B2SignalEngine
from live.combined.od_engine import ODEngine, _ATR
from live.combined.fabio_orb_engine import FabioORBEngine
from live.combined.warm_start import feed_warm_start_to_builders
from live.combined.config import WARM_START_TRADING_DAYS

LIVE_DIR = STATE_DIR / "paper"
REPLAY_DIR = STATE_DIR / "replay"
REPLAY_DIR.mkdir(parents=True, exist_ok=True)


def fetch_session_ticks(date: dt.date, use_cache: bool = True) -> pd.DataFrame:
    """Fetch all NQ trades for date's CME session from Databento Historical.
    Session = 18:00 ET (D-1) → 17:00 ET (D).

    Caches the result to STATE_DIR/replay/databento_<date>.parquet so reruns
    don't re-fetch (saves ~7 min per rerun)."""
    cache_path = REPLAY_DIR / f"databento_trades_{date.isoformat()}.parquet"
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        df["ts_et"] = pd.to_datetime(df["ts_et"], utc=True).dt.tz_convert(ET_TZ)
        print(f"  [replay] loaded {len(df):,} trades from cache ({cache_path.name})")
        return df

    import databento as db
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError("DATABENTO_API_KEY not set")
    client = db.Historical(api_key)

    start_et = pd.Timestamp(f"{date - dt.timedelta(days=1)} 18:00", tz=ET_TZ)
    end_et   = pd.Timestamp(f"{date} 17:00", tz=ET_TZ)
    now_safe = pd.Timestamp.now(tz=ET_TZ) - pd.Timedelta(minutes=35)
    if now_safe < end_et:
        end_et = now_safe
        print(f"  [replay] end clipped to {end_et} (Databento 30-min delay)")

    print(f"  [replay] fetching {start_et.strftime('%Y-%m-%d %H:%M ET')} -> "
          f"{end_et.strftime('%Y-%m-%d %H:%M ET')}...")
    df = client.timeseries.get_range(
        dataset=DATABENTO_DATASET, schema=DATABENTO_SCHEMA,
        symbols=[DATABENTO_SYMBOL], stype_in=DATABENTO_STYPE,
        start=start_et.tz_convert("UTC"), end=end_et.tz_convert("UTC"),
    ).to_df()
    if df.empty:
        return df
    if "ts_recv" not in df.columns:
        df = df.reset_index()
    df = df[df["action"] == "T"].copy()
    df["ts_et"] = pd.to_datetime(df["ts_recv"], utc=True).dt.tz_convert(ET_TZ)
    print(f"  [replay] fetched {len(df):,} trades")

    # Cache it (small subset of columns to keep file size down)
    out = df[["ts_et", "price", "size", "side"]].copy()
    out["ts_et"] = out["ts_et"].dt.tz_convert("UTC")   # parquet doesn't store tz well; store UTC
    out.to_parquet(cache_path, compression="zstd", index=False)
    print(f"  [replay] cached to {cache_path.name}")
    return df


def run_replay(date: dt.date) -> dict:
    """Run all 4 engines on today's historical trades. Returns dict of trades per strat."""
    ticks = fetch_session_ticks(date)
    if ticks.empty:
        print(f"  [replay] no trades returned for {date}")
        return {}

    # Build the same pipeline as run_phase1
    multi = MultiBarBuilder()
    b5 = multi.add_timeframe(BAR_5MIN_SECS, name="5min")
    b20 = multi.add_timeframe(BAR_20MIN_SECS, name="20min")

    # Engines
    rv = RVEngine()
    b2 = B2Engine()
    od = ODEngine()
    fb = FabioORBEngine()
    b2_sig = B2SignalEngine(b2_engine=b2)
    b2_atr = _ATR(length=14)

    # Trade collectors (one dict per strat)
    trades = {"RV": [], "B2": [], "OD": [], "FB": []}
    pending = {"RV": {}, "B2": {}, "OD": {}, "FB": {}}

    def make_logger(strat, pending_dict, trades_list):
        def cb(sig):
            d = ("LONG" if sig.direction.name == "LONG"
                 else "SHORT" if sig.direction.name == "SHORT" else "FLAT")
            if sig.event == "ENTRY":
                pending_dict[strat] = {"entry_ts": sig.timestamp, "entry": sig.price,
                                        "direction": d, "qty": getattr(sig, "qty", 1)}
            elif sig.event == "EXIT" and strat in pending_dict and pending_dict[strat]:
                e = pending_dict[strat]
                trades_list.append({
                    "entry_ts": e["entry_ts"], "direction": e["direction"],
                    "entry": e["entry"], "exit_ts": sig.timestamp,
                    "exit": sig.price, "reason": sig.reason, "qty": e["qty"],
                })
                pending_dict[strat] = {}
        return cb

    rv.subscribe(make_logger("RV", {"RV": {}}, trades["RV"]))
    b2.subscribe(make_logger("B2", {"B2": {}}, trades["B2"]))
    od.subscribe(make_logger("OD", {"OD": {}}, trades["OD"]))
    fb.subscribe(make_logger("FB", {"FB": {}}, trades["FB"]))

    b20.subscribe(rv.on_bar)
    def _b2_on_bar(bar):
        atr = b2_atr.update(bar.high, bar.low, bar.close)
        b2.on_20min_bar(bar, current_atr_y=atr if atr == atr else None)
    b20.subscribe(_b2_on_bar)
    b20.subscribe(od.on_20min_bar)
    b5.subscribe(b2_sig.on_5min_bar)
    b5.subscribe(fb.on_5min_bar)

    # Warm-start engines from 15-day cache (mirrors run_phase1.py).
    # RV needs 80-bar kernel + 75-bar z_vol window seeded; OD needs ATR(14)
    # warmed; without this, engines fire fewer signals than live.
    print(f"  [replay] warm-starting engines from {WARM_START_TRADING_DAYS}-day cache...")
    n_warm = feed_warm_start_to_builders(multi, n_trading_days=WARM_START_TRADING_DAYS)
    # Clear any warm-start "trades" — those are historical, not today's
    for k in trades:
        trades[k].clear()
        pending[k] = {}
    print(f"  [replay] warm-start fed {n_warm} bars; engine bar counts: "
          f"RV={rv.n_bars_seen} B2={b2.n_bars_seen} OD={od.n_bars_seen} FB={fb.n_bars_seen}")

    # Feed ticks chronologically
    print(f"  [replay] feeding {len(ticks):,} ticks through bar builders + engines...")
    side_map = {"B": "B", "A": "A", "N": "N"}
    for ts, price, size, side in zip(ticks["ts_et"], ticks["price"],
                                      ticks["size"], ticks["side"]):
        s = side_map.get(str(side), "N")
        multi.on_tick(ts, float(price), int(size), s)
    # Force-close any in-progress bars at the end
    for builder in multi.builders.values():
        builder.force_close_current()

    print(f"\n[replay] engine stats:")
    print(f"  RV: bars_seen={rv.n_bars_seen}  trades={len(trades['RV'])}")
    print(f"  B2: bars_seen={b2.n_bars_seen}  trades={len(trades['B2'])}  signals={b2_sig.n_signals_emitted}")
    print(f"  OD: bars_seen={od.n_bars_seen}  trades={len(trades['OD'])}")
    print(f"  FB: bars_seen={fb.n_bars_seen}  trades={len(trades['FB'])}")
    return trades


def save_replay_csv(trades: dict, date: dt.date) -> dict:
    """Save replay trades per strat. Returns dict of {strat: csv_path}."""
    paths = {}
    for strat, tlist in trades.items():
        path = REPLAY_DIR / f"replay_{strat.lower()}_{date.isoformat()}.csv"
        if tlist:
            df = pd.DataFrame(tlist)
            df.to_csv(path, index=False)
        paths[strat] = path
    return paths


def diff_vs_live(date: dt.date, replay_trades: dict) -> None:
    """For each strat, compare replay trades to live_*_signals.csv (filtered to today)."""
    print(f"\n{'='*70}")
    print(f"DIFF: REPLAY vs LIVE for {date}")
    print(f"{'='*70}")

    live_files = {"RV": LIVE_DIR / "live_rv_signals.csv",
                  "B2": LIVE_DIR / "live_b2_signals.csv",
                  "OD": LIVE_DIR / "live_od_signals.csv",
                  "FB": LIVE_DIR / "live_fb_signals.csv"}

    for strat in ["RV", "B2", "OD", "FB"]:
        print(f"\n--- {strat} ---")
        rep = replay_trades.get(strat, [])
        n_rep = len(rep)

        # Live CSV — pair ENTRY with subsequent EXIT for each direction-date
        lpath = live_files[strat]
        n_live_today = 0
        live_today = []
        if lpath.exists():
            try:
                lv = pd.read_csv(lpath)
                # Strip trailing timezone abbreviation (EDT/EST) — pandas deprecation
                lv["ts_et"] = lv["ts_et"].astype(str).str.replace(
                    r"\s*(EDT|EST)$", "", regex=True)
                lv["ts_et"] = pd.to_datetime(lv["ts_et"])
                lv["date"] = lv["ts_et"].dt.date
                lv_today_all = lv[lv["date"] == date].sort_values("ts_et").reset_index(drop=True)
                # Pair entries with the NEXT exit (per direction, chronologically)
                open_entry = None
                for _, r in lv_today_all.iterrows():
                    if r["event"] == "ENTRY":
                        open_entry = {"entry_ts": pd.Timestamp(r["ts_et"]),
                                       "direction": r["direction"],
                                       "entry": float(r["price"])}
                    elif r["event"] == "EXIT" and open_entry is not None:
                        live_today.append({
                            **open_entry,
                            "exit_ts": pd.Timestamp(r["ts_et"]),
                            "exit": float(r["price"]),
                            "reason": str(r["reason"]),
                        })
                        open_entry = None
                # Also count open (still-running) entries
                if open_entry is not None:
                    live_today.append({**open_entry, "exit_ts": None,
                                       "exit": None, "reason": "(open)"})
                n_live_today = len(live_today)
            except Exception as e:
                print(f"  Failed to read live CSV: {e}")

        print(f"  Replay entries: {n_rep}   Live entries today: {n_live_today}")

        if n_rep == 0 and n_live_today == 0:
            print(f"  [OK] Both zero - agreement (no signals today)")
            continue

        # Match on entry_ts (rounded to nearest minute) + direction
        def key(t):
            ts_raw = pd.Timestamp(t["entry_ts"])
            if ts_raw.tzinfo is not None:
                ts = ts_raw.tz_convert(ET_TZ)
            else:
                ts = ts_raw.tz_localize(ET_TZ)
            return (ts.strftime("%Y-%m-%d %H:%M"), str(t["direction"]).upper())
        rep_keys = {key(t): t for t in rep}
        live_keys = {key(t): t for t in live_today}
        matched = set(rep_keys) & set(live_keys)
        only_rep = set(rep_keys) - set(live_keys)
        only_live = set(live_keys) - set(rep_keys)
        print(f"  Matched: {len(matched)}   Only replay: {len(only_rep)}   Only live: {len(only_live)}")
        for k in sorted(matched):
            rp = rep_keys[k]; lv = live_keys[k]
            entry_diff = abs(rp["entry"] - lv["entry"])

            # Exit comparison — both replay and live should have exit data
            rep_exit_ts = pd.Timestamp(rp.get("exit_ts")) if rp.get("exit_ts") else None
            lv_exit_ts  = pd.Timestamp(lv.get("exit_ts")) if lv.get("exit_ts") else None
            rep_exit    = rp.get("exit")
            lv_exit     = lv.get("exit")
            rep_reason  = rp.get("reason") or ""
            lv_reason   = lv.get("reason") or ""

            # Mark — all of (entry, exit_ts, exit_price, reason) must match
            mismatches = []
            if entry_diff >= 0.5:
                mismatches.append(f"entry_diff={entry_diff:.2f}")
            if rep_exit_ts is None or lv_exit_ts is None:
                if rep_exit_ts != lv_exit_ts:
                    mismatches.append("one open, one closed")
            else:
                # Normalize tz before comparing
                rt = rep_exit_ts.tz_convert(ET_TZ) if rep_exit_ts.tzinfo else rep_exit_ts.tz_localize(ET_TZ)
                lt = lv_exit_ts.tz_convert(ET_TZ) if lv_exit_ts.tzinfo else lv_exit_ts.tz_localize(ET_TZ)
                ts_diff_min = abs((rt - lt).total_seconds()) / 60.0
                if ts_diff_min > 5:
                    mismatches.append(f"exit_ts_diff={ts_diff_min:.0f}min")
                if rep_exit is not None and lv_exit is not None:
                    px_diff = abs(rep_exit - lv_exit)
                    if px_diff >= 0.5:
                        mismatches.append(f"exit_px_diff={px_diff:.2f}")
                if rep_reason.upper() != lv_reason.upper():
                    mismatches.append(f"reason={rep_reason}|{lv_reason}")

            mark = "[OK]" if not mismatches else "[WARN]"
            entry_str = f"{rp['entry']:.2f}/{lv['entry']:.2f}"
            exit_str  = (f"{rep_exit:.2f}/{lv_exit:.2f}"
                         if rep_exit is not None and lv_exit is not None else "open/open")
            reason_str = f"{rep_reason}/{lv_reason}"
            rt_str = (rep_exit_ts.tz_convert(ET_TZ).strftime("%H:%M")
                      if rep_exit_ts is not None and rep_exit_ts.tzinfo else "--")
            lt_str = (lv_exit_ts.tz_convert(ET_TZ).strftime("%H:%M")
                      if lv_exit_ts is not None and lv_exit_ts.tzinfo else
                      lv_exit_ts.tz_localize(ET_TZ).strftime("%H:%M") if lv_exit_ts is not None else "--")
            print(f"    {mark} ENTRY {k}: rep@{entry_str}  EXIT @ rep:{rt_str} live:{lt_str}  "
                  f"px:{exit_str}  reason:{reason_str}")
            if mismatches:
                print(f"         issues: {', '.join(mismatches)}")

        for k in sorted(only_rep):
            print(f"    [REPLAY ONLY] {k} @ {rep_keys[k]['entry']:.2f}")
        for k in sorted(only_live):
            print(f"    [LIVE ONLY]   {k} @ {live_keys[k]['entry']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default=None,
                     help="Session date YYYY-MM-DD (defaults to today's CME session)")
    args = ap.parse_args()
    if args.date:
        date = dt.date.fromisoformat(args.date)
    else:
        now = pd.Timestamp.now(tz=ET_TZ)
        date = (now + pd.Timedelta(days=1)).date() if now.hour >= 18 else now.date()

    print(f"Replay date: {date}")
    trades = run_replay(date)
    save_replay_csv(trades, date)
    diff_vs_live(date, trades)


if __name__ == "__main__":
    main()
