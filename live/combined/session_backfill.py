"""Session-start backfill — fills the pre-startup gap from Databento Historical.

When you start `run_phase1.py --live` mid-session (after 18:00 ET), there's a
gap in today's pickle: from session_start (18:00 ET prior trading day) up to
your startup time. Live bars only cover startup-time onwards.

This module schedules background backfills that fetch the missing portion
from Databento Historical 30 min after Databento's processing delay window
has cleared (i.e., 30+30=60 min after startup is safest).

Two-pass design (T+30 and T+60):
  - T+30: Databento Historical has data up to (T+30)-30 = T. Fetch
          [session_start, T] and merge into pickle. Live wins on overlap.
  - T+60: same fetch, safety pass for transient errors.

Fires whenever today's session pickle has a REAL gap before startup — a cold daytime start (PC was
off overnight), a restart after downtime, or an evening start right as the session opens. Skips when
the pickle is already current (a clean restart after a continuously-running instance). No longer gated
on wall-clock hour, which used to silently skip every 00:00-17:59 ET start and leave B2 without its
overnight range.
"""
from __future__ import annotations

import datetime as dt
import os
import pickle
import threading
from pathlib import Path
from typing import Optional

import pandas as pd

from live.combined.config import (
    LIVE_WARMSTART_CACHE_DIR, ET_TZ,
    DATABENTO_DATASET, DATABENTO_SYMBOL, DATABENTO_SCHEMA, DATABENTO_STYPE,
)


def _current_session_date(now_et: pd.Timestamp) -> dt.date:
    """If now >= 18:00 ET, the session = next day's date. Else today's date."""
    if now_et.hour >= 18:
        return (now_et + pd.Timedelta(days=1)).date()
    return now_et.date()


def _session_start_et(session_date: dt.date) -> pd.Timestamp:
    """Session for date D starts at 18:00 ET (D-1)."""
    return pd.Timestamp(f"{session_date - dt.timedelta(days=1)} 18:00", tz=ET_TZ)


def _pickle_covered_until(session_date: dt.date,
                          cache_dir: Path = LIVE_WARMSTART_CACHE_DIR) -> Optional[pd.Timestamp]:
    """Open_time (ET) of the LATEST 5-min bar already on disk for this session, or None if the pickle
    is missing/empty. Lets us trigger the backfill off the real gap rather than the wall-clock hour."""
    p = cache_dir / f"nq_5min_{session_date.isoformat()}.pkl"
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            bars = pickle.load(f)
    except Exception:
        return None
    if not bars:
        return None
    last = max(pd.Timestamp(b["open_time"]) for b in bars)
    last = last.tz_localize("UTC") if last.tz is None else last.tz_convert("UTC")
    return last.tz_convert(ET_TZ)


def _fetch_historical_bars(start_et: pd.Timestamp, end_et: pd.Timestamp) -> list[dict]:
    """Fetch trades from Databento Historical and aggregate into 5-min bars."""
    import databento as db
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError("DATABENTO_API_KEY not set")
    client = db.Historical(api_key)

    start_utc = start_et.tz_convert("UTC")
    end_utc = end_et.tz_convert("UTC")

    print(f"  [backfill] fetching Databento Historical "
          f"{start_et.strftime('%Y-%m-%d %H:%M ET')} -> "
          f"{end_et.strftime('%Y-%m-%d %H:%M ET')}...")

    df = client.timeseries.get_range(
        dataset=DATABENTO_DATASET,
        schema=DATABENTO_SCHEMA,
        symbols=[DATABENTO_SYMBOL],
        stype_in=DATABENTO_STYPE,
        start=start_utc, end=end_utc,
    ).to_df()
    if df.empty:
        return []

    # Databento v3 returns ts_recv as INDEX
    if "ts_recv" not in df.columns:
        df = df.reset_index()
    df = df[df["action"] == "T"].copy()
    if df.empty:
        return []

    # Aggregate to 5-min bars
    ts = pd.to_datetime(df["ts_recv"], utc=True).dt.tz_convert(ET_TZ)
    df["bar_open_et"] = ts.dt.floor("5min")
    df["side_str"] = df["side"].astype(str)

    bars = []
    for bar_open, g in df.groupby("bar_open_et"):
        buys = int(g.loc[g["side_str"] == "B", "size"].sum())
        sells = int(g.loc[g["side_str"] == "A", "size"].sum())
        bars.append({
            "open_time": bar_open.tz_convert("UTC").tz_localize(None),
            "open": float(g["price"].iloc[0]),
            "high": float(g["price"].max()),
            "low": float(g["price"].min()),
            "close": float(g["price"].iloc[-1]),
            "buy_vol": buys,
            "sell_vol": sells,
            "tick_count": len(g),
        })
    return bars


def _merge_keep_live(historical: list[dict], live: list[dict]) -> list[dict]:
    """Merge historical + live bars. On overlap (same open_time), LIVE wins.
    Returns sorted, deduped list."""
    by_ot = {}
    # Historical first — gets overwritten by live on conflict
    for b in historical:
        ot = pd.Timestamp(b["open_time"])
        if ot.tz is not None:
            ot = ot.tz_convert("UTC").tz_localize(None)
        by_ot[ot] = b
    for b in live:
        ot = pd.Timestamp(b["open_time"])
        if ot.tz is not None:
            ot = ot.tz_convert("UTC").tz_localize(None)
        by_ot[ot] = b
    merged = sorted(by_ot.values(), key=lambda b: pd.Timestamp(b["open_time"]))
    return merged


def run_backfill_once(session_date: dt.date, startup_time_et: pd.Timestamp,
                       cache_dir: Path = LIVE_WARMSTART_CACHE_DIR) -> tuple[int, int]:
    """One backfill pass. Returns (n_fetched, n_total_after_merge)."""
    session_start = _session_start_et(session_date)
    if startup_time_et <= session_start:
        print(f"  [backfill] startup_time {startup_time_et} is at/before session_start "
              f"{session_start} — nothing to backfill")
        return (0, 0)

    pickle_path = cache_dir / f"nq_5min_{session_date.isoformat()}.pkl"

    # Load existing live-built bars
    live_bars = []
    if pickle_path.exists():
        try:
            with open(pickle_path, "rb") as f:
                live_bars = pickle.load(f)
        except Exception as e:
            print(f"  [backfill] failed to read existing pickle: {e}")
            live_bars = []

    # Fetch the gap from Databento Historical
    try:
        historical_bars = _fetch_historical_bars(session_start, startup_time_et)
    except Exception as e:
        print(f"  [backfill] historical fetch FAILED: {e}")
        return (0, len(live_bars))

    merged = _merge_keep_live(historical_bars, live_bars)

    # Atomic write
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = pickle_path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(merged, f)
    os.replace(tmp, pickle_path)

    print(f"  [backfill] merged: historical={len(historical_bars)}  "
          f"live={len(live_bars)}  -> total={len(merged)}")
    return (len(historical_bars), len(merged))


def schedule_session_backfill(startup_time_et: pd.Timestamp,
                                delay1_min: int = 30,
                                delay2_min: int = 60) -> Optional[threading.Thread]:
    """Spawn a daemon thread that runs two backfill passes at T+30 and T+60.

    Skips entirely if startup is before 18:00 ET (no new-session-in-progress).
    Returns the thread (None if not scheduled).
    """
    session_date = _current_session_date(startup_time_et)
    session_start = _session_start_et(session_date)
    if startup_time_et <= session_start:
        print(f"[backfill] startup {startup_time_et.strftime('%Y-%m-%d %H:%M ET')} is at/before "
              f"session open {session_start.strftime('%m-%d %H:%M')} — nothing to backfill")
        return None

    # Trigger off the ACTUAL gap in today's pickle, not the wall-clock hour. A prior continuously-running
    # instance persists today's pickle as it goes, so a clean restart finds it current (skip). A cold
    # daytime start (PC off overnight) finds it missing/short -> backfill the missed 18:00->startup hours.
    covered = _pickle_covered_until(session_date)
    gap_start = session_start if covered is None else covered + pd.Timedelta(minutes=5)
    gap_min = int((startup_time_et - gap_start).total_seconds() / 60)
    covered_str = covered.strftime('%m-%d %H:%M') if covered is not None else '(none on disk)'
    if gap_min < 5:
        print(f"[backfill] today's pickle already current through {covered_str} "
              f"(gap {gap_min} min) — no backfill needed")
        return None
    print(f"[backfill] startup={startup_time_et.strftime('%Y-%m-%d %H:%M ET')}  session={session_date}  "
          f"covered_through={covered_str}  gap={gap_min} min")
    print(f"[backfill] scheduled two passes: T+{delay1_min} min and T+{delay2_min} min")

    def _worker():
        import time
        # Pass 1
        time.sleep(delay1_min * 60)
        print(f"\n[backfill] PASS 1 firing at T+{delay1_min} min "
              f"({pd.Timestamp.now(tz=ET_TZ).strftime('%H:%M ET')})...")
        try:
            run_backfill_once(session_date, startup_time_et)
        except Exception as e:
            print(f"  [backfill] PASS 1 exception: {e}")
        # Pass 2
        time.sleep((delay2_min - delay1_min) * 60)
        print(f"\n[backfill] PASS 2 firing at T+{delay2_min} min "
              f"({pd.Timestamp.now(tz=ET_TZ).strftime('%H:%M ET')})...")
        try:
            run_backfill_once(session_date, startup_time_et)
        except Exception as e:
            print(f"  [backfill] PASS 2 exception: {e}")
        print(f"\n[backfill] both passes complete. Today's pickle should be continuous.\n")

    t = threading.Thread(target=_worker, daemon=True, name="session_backfill")
    t.start()
    return t


if __name__ == "__main__":
    # Manual one-shot mode — run a single backfill right now
    now_et = pd.Timestamp.now(tz=ET_TZ)
    session_date = _current_session_date(now_et)
    print(f"Manual backfill for session {session_date}, "
          f"startup_time_et=now ({now_et.strftime('%H:%M ET')})")
    run_backfill_once(session_date, now_et)
