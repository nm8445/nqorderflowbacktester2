"""Daily gamma-level pipeline — runs all 4 steps in sequence.

Steps:
  1. download_qqq_eod.py   — pull QQQ EOD greeks + open_interest from ThetaData
  2. download_ndx_eod.py   — pull NDX prices + open_interest from ThetaData
  3. menthorq_style_levels.py YYYY-MM-DD — verify level builder for one date (optional sanity check)
  4. build_menthorq_levels_bulk.py — append new days to menthorq_levels_nq.parquet

Usage:
    python scripts/thetadata/daily_pipeline/run_daily.py
    python scripts/thetadata/daily_pipeline/run_daily.py --skip-sanity
    python scripts/thetadata/daily_pipeline/run_daily.py --end-date 2026-05-15

This is the ONLY script you need to run daily. Schedule it via Windows Task
Scheduler ~17:00 ET on weekdays. On weekends it's a no-op (downloaders skip Sat/Sun).

Prereqs:
  - ThetaData Terminal running locally at http://127.0.0.1:25503
  - Active ThetaData subscription
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
THETA_HOST, THETA_PORT = "127.0.0.1", 25503
PARQUET = Path("D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet")


def terminal_up(host: str = THETA_HOST, port: int = THETA_PORT, timeout: float = 4.0) -> bool:
    """True if the ThetaData Terminal is accepting connections on :25503. The downloaders swallow
    connection errors and still exit 0, so without this pre-flight the daily task reports false
    success while fetching nothing — exactly what the 2026-06-10 Windows-update reboot caused (it
    killed the Terminal, which doesn't auto-restart, and froze the levels at the 06-08 settle)."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def run_step(name: str, cmd: list[str]) -> bool:
    """Run a step. Returns True if exit code 0."""
    print(f"\n{'='*70}\n[STEP] {name}\n{'='*70}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(THIS_DIR.parents[2]))
    elapsed = time.time() - t0
    ok = r.returncode == 0
    print(f"\n[{name}] {'OK' if ok else 'FAILED'} in {elapsed:.1f}s")
    return ok


def update_end_date(script_path: Path, new_end: dt.date) -> None:
    """Patch the END = dt.date(...) line in a downloader script."""
    text = script_path.read_text()
    import re
    new_text = re.sub(
        r"END\s*=\s*dt\.date\([^)]+\)([^\n]*)",
        f"END   = dt.date({new_end.year}, {new_end.month}, {new_end.day})  # auto-updated by run_daily.py",
        text,
        count=1,
    )
    if new_text != text:
        script_path.write_text(new_text)
        print(f"  Updated END date in {script_path.name} -> {new_end}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--end-date", type=str, default=None,
                    help="Override end date (YYYY-MM-DD). Default: today")
    ap.add_argument("--skip-sanity", action="store_true",
                    help="Skip the menthorq_style_levels.py sanity check")
    ap.add_argument("--sanity-date", type=str, default=None,
                    help="Date for sanity check (YYYY-MM-DD). Default: end_date")
    args = ap.parse_args()

    end_date = (dt.date.fromisoformat(args.end_date) if args.end_date
                else dt.date.today())
    print(f"Daily gamma pipeline. Target end date: {end_date}")

    # Pre-flight: the Terminal must be up, else the downloaders no-op silently and the task reports
    # false success. Fail loudly (non-zero) so Task Scheduler shows an error instead of a green run.
    if not terminal_up():
        print(f"\n*** ABORT: ThetaData Terminal not reachable at {THETA_HOST}:{THETA_PORT}.")
        print(f"*** Launch it (java -jar C:\\ThetaTerminal\\ThetaTerminalv3.jar) and re-run.")
        print(f"*** Gamma levels were NOT updated.")
        return 2

    # Patch download scripts' END dates so they fetch through the requested day
    update_end_date(THIS_DIR / "download_qqq_eod.py", end_date)
    update_end_date(THIS_DIR / "download_ndx_eod.py", end_date)

    # Step 1: QQQ
    if not run_step("Download QQQ EOD greeks+OI",
                    [sys.executable, str(THIS_DIR / "download_qqq_eod.py")]):
        return 1

    # Step 2: NDX
    if not run_step("Download NDX EOD prices+OI",
                    [sys.executable, str(THIS_DIR / "download_ndx_eod.py")]):
        return 1

    # Step 3: Sanity check (optional, non-fatal)
    # Auto-pick the most recent date that has ThetaData downloaded (default
    # is end_date, but if end_date is today and ThetaData isn't published yet,
    # we'd fail — so walk backwards to find a date that has data).
    if not args.skip_sanity:
        if args.sanity_date:
            sanity_date = args.sanity_date
        else:
            from pathlib import Path as _P
            qqq_root = _P("D:/trading_pythonbacktest_data/QQQ_thetadata")
            sanity_date = end_date.isoformat()
            # Walk back up to 5 days to find one with data
            for delta_days in range(5):
                check = end_date - dt.timedelta(days=delta_days)
                if (qqq_root / check.isoformat() / "greeks_eod.parquet").exists():
                    sanity_date = check.isoformat()
                    break

        ok = run_step(f"Sanity: compute levels for {sanity_date}",
                       [sys.executable, str(THIS_DIR / "menthorq_style_levels.py"),
                        sanity_date])
        if not ok:
            print(f"\nSanity check failed for {sanity_date} — CONTINUING anyway "
                  f"(bulk build will skip days without data)")
            # NOT returning — sanity is informational; bulk build is the
            # critical step and handles missing days gracefully.

    # Step 4: Incremental append via gamma_refresh.py
    # NOTE: do NOT use build_menthorq_levels_bulk.py here — it rebuilds the
    # parquet from scratch using ONLY markettick_1min_bars.parquet settles,
    # which may be stale (only updates on demand). gamma_refresh.py reads
    # settles from markettick + rolling cache + live state, and appends
    # only missing days (preserving existing rows).
    LIVE_REFRESH = THIS_DIR.parents[2] / "live" / "combined" / "gamma_refresh.py"
    if not run_step("Incremental gamma refresh (multi-source settles)",
                    [sys.executable, str(LIVE_REFRESH)]):
        return 1

    # Post-flight: confirm the parquet advanced to the last COMPLETED trading day. A silent feed failure
    # (ThetaData HTTP 478 'Invalid session ID' from a duplicate terminal, a down feed, an API change)
    # yields 0 new rows; the old calendar-day tolerance (MAX_STALE_DAYS=4) masked exactly that over a
    # weekend (e.g. parquet stuck at Fri while Mon's data was missing). Now we check against the previous
    # WEEKDAY — whose EOD greeks publish ~8am today — and surface rows_added so a stuck feed FAILS loudly.
    try:
        import pandas as pd, json as _json
        latest = pd.Timestamp(pd.read_parquet(PARQUET, columns=["date"])["date"].max()).date()
        expected = end_date - dt.timedelta(days=1)        # last weekday strictly before today
        while expected.weekday() >= 5:                    # skip Sat/Sun (holidays not modeled — a rare
            expected -= dt.timedelta(days=1)              # post-holiday false-positive beats silent pass)
        last_refresh = THIS_DIR.parents[2] / "live" / "combined" / "state" / "last_gamma_refresh.json"
        try:
            rows_added = _json.loads(last_refresh.read_text()).get("rows_added")
        except Exception:
            rows_added = None
        if latest < expected:
            print(f"\n*** FAIL: parquet latest is {latest}, but the last completed trading day is {expected}.")
            print(f"*** gamma_refresh added {rows_added} row(s) this run — {expected}'s greeks were NOT ingested.")
            print(f"*** A down or DUPLICATE ThetaData Terminal (HTTP 478 'Invalid session ID') silently yields")
            print(f"*** 0 rows. Ensure exactly ONE terminal is serving, then re-run run_daily.py.")
            return 3
        print(f"  parquet current through {latest} (>= last trading day {expected}; +{rows_added or 0} row(s) this run)")
    except Exception as e:
        print(f"  [post-check] could not verify parquet freshness: {e}")

    print(f"\n{'='*70}\nDAILY PIPELINE COMPLETE — gamma levels current through {end_date}\n{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
