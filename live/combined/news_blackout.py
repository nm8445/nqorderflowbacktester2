"""MFFU T1 news blackout — calendar + watcher thread + entry gate helper.

Per MFFU rule:
  - No open positions OR orders 2 min before/after any T1 release.
  - T1 = FOMC Meetings, FOMC Minutes, U.S. Employment Report (NFP), CPI.
  - Flex 25K/50K accounts CAN trade T1 events but must obey the 2-min flat rule.

Behavior implemented here:
  - At T-2:30 (2 min 30 sec BEFORE event): NewsBlackoutWatcher force-flattens
    all open strategy positions. Extra 30 sec gives broker fill time.
  - From T-2:00 to T+2:00: entries are blocked. The coordinator calls
    is_in_blackout() at entry time; if True, the signal is rejected.
  - After T+2:00: blackout lifts, normal trading resumes.

Calendar source: MFFU's published 2026 T1 list (user-provided).

Usage in run_phase1.py:
    from live.combined.news_blackout import NewsBlackoutWatcher, is_in_blackout
    watcher = NewsBlackoutWatcher(coord, nt8x=nt8, mt5x=mt5x)
    watcher.start()

Usage in coordinator.py:
    from live.combined.news_blackout import is_in_blackout
    in_bk, name = is_in_blackout()
    if in_bk: reject ENTRY
"""
from __future__ import annotations
import threading
from datetime import timedelta
from typing import Optional, Tuple
import pandas as pd

from live.combined.config import ET_TZ

# ============== MFFU 2026 T1 calendar ==============
# Format: (YYYY-MM-DD, HH:MM ET, event_name)
MFFU_T1_2026 = [
    # NFP (U.S. Employment Report) @ 08:30 ET
    ("2026-01-09", "08:30", "NFP"),
    ("2026-02-06", "08:30", "NFP"),
    ("2026-03-06", "08:30", "NFP"),
    ("2026-04-03", "08:30", "NFP"),
    ("2026-05-08", "08:30", "NFP"),
    ("2026-06-05", "08:30", "NFP"),
    ("2026-07-02", "08:30", "NFP"),
    ("2026-08-07", "08:30", "NFP"),
    ("2026-09-04", "08:30", "NFP"),
    ("2026-10-02", "08:30", "NFP"),
    ("2026-11-06", "08:30", "NFP"),
    ("2026-12-04", "08:30", "NFP"),
    # CPI @ 08:30 ET
    ("2026-01-13", "08:30", "CPI"),
    ("2026-02-11", "08:30", "CPI"),
    ("2026-03-11", "08:30", "CPI"),
    ("2026-04-10", "08:30", "CPI"),
    ("2026-05-12", "08:30", "CPI"),
    ("2026-06-10", "08:30", "CPI"),
    ("2026-07-14", "08:30", "CPI"),
    ("2026-08-12", "08:30", "CPI"),
    ("2026-09-11", "08:30", "CPI"),
    ("2026-10-14", "08:30", "CPI"),
    ("2026-11-10", "08:30", "CPI"),
    ("2026-12-10", "08:30", "CPI"),
    # FOMC Meeting + Minutes @ 14:00 ET
    ("2026-01-28", "14:00", "FOMC"),
    ("2026-02-18", "14:00", "FOMC_MINUTES"),
    ("2026-03-18", "14:00", "FOMC"),
    ("2026-04-08", "14:00", "FOMC_MINUTES"),
    ("2026-04-29", "14:00", "FOMC"),
    ("2026-05-20", "14:00", "FOMC_MINUTES"),
    ("2026-06-17", "14:00", "FOMC"),
    ("2026-07-08", "14:00", "FOMC_MINUTES"),
    ("2026-07-29", "14:00", "FOMC"),
    ("2026-08-19", "14:00", "FOMC_MINUTES"),
    ("2026-09-16", "14:00", "FOMC"),
    ("2026-10-07", "14:00", "FOMC_MINUTES"),
    ("2026-11-18", "14:00", "FOMC_MINUTES"),
    ("2026-12-09", "14:00", "FOMC"),
    ("2026-12-30", "14:00", "FOMC_MINUTES"),
]

# ============== Window definitions ==============
# MFFU rule: ±2 min flat around any T1.
# We flatten at T-150s (30 sec extra for fill time) and block entries T-120s to T+120s.
FLATTEN_LEAD_SEC  = 150   # force-close 2 min 30 sec BEFORE event
ENTRY_LEAD_SEC    = 120   # block new entries 2 min BEFORE event
ENTRY_TRAIL_SEC   = 120   # block new entries 2 min AFTER event


def _parse_events() -> list[tuple[pd.Timestamp, str]]:
    out = []
    for date_str, time_str, name in MFFU_T1_2026:
        dt = pd.Timestamp(f"{date_str} {time_str}:00").tz_localize(ET_TZ)
        out.append((dt, name))
    out.sort()
    return out


EVENTS = _parse_events()


def is_in_blackout(now_et: Optional[pd.Timestamp] = None) -> Tuple[bool, str]:
    """Return (True, event_name) if `now_et` is within the entry-blackout window.

    Blackout window = [event - ENTRY_LEAD_SEC, event + ENTRY_TRAIL_SEC].

    Called by coordinator._handle on every ENTRY signal. Lookahead-free
    (only compares now to scheduled event times)."""
    if now_et is None:
        now_et = pd.Timestamp.now(tz=ET_TZ)
    for ev_dt, name in EVENTS:
        # Optimization: events are sorted; bail early if we're past + window
        if ev_dt + timedelta(seconds=ENTRY_TRAIL_SEC) < now_et:
            continue
        if ev_dt - timedelta(seconds=ENTRY_LEAD_SEC) > now_et:
            break
        return True, name
    return False, ""


def next_event_within(hours: float = 48.0,
                      now_et: Optional[pd.Timestamp] = None) -> Tuple[Optional[pd.Timestamp], str]:
    """Find the next upcoming event within `hours`. Returns (event_dt, name) or (None, '')."""
    if now_et is None:
        now_et = pd.Timestamp.now(tz=ET_TZ)
    horizon = now_et + timedelta(hours=hours)
    for ev_dt, name in EVENTS:
        if now_et <= ev_dt <= horizon:
            return ev_dt, name
    return None, ""


def events_on_date(date, now_et: Optional[pd.Timestamp] = None) -> list[tuple[pd.Timestamp, str]]:
    """Return all T1 events scheduled on the given date (datetime.date or pandas Timestamp)."""
    if hasattr(date, "date"):
        date = date.date()
    return [(ev_dt, name) for ev_dt, name in EVENTS if ev_dt.date() == date]


def print_today_summary(now_et: Optional[pd.Timestamp] = None,
                         tag: str = "[news_blackout]") -> None:
    """Print a one-shot summary of today's T1 events and blackout windows.

    Called at startup and at every calendar-day rollover by the watcher."""
    if now_et is None:
        now_et = pd.Timestamp.now(tz=ET_TZ)
    today = now_et.date()
    events_today = events_on_date(today, now_et)

    print(f"\n{tag} ====== T1 NEWS CALENDAR — {today.strftime('%A %Y-%m-%d')} ======")
    if not events_today:
        print(f"{tag}   NO T1 events today. Normal trading, no blackout windows.")
        # Show next upcoming so user knows what's next
        nxt, nxt_name = next_event_within(168.0, now_et)   # next 7 days
        if nxt:
            days_ahead = (nxt.date() - today).days
            print(f"{tag}   Next T1 event: {nxt_name} on "
                  f"{nxt.strftime('%a %Y-%m-%d %H:%M ET')} ({days_ahead}d ahead)")
        print(f"{tag} ===============================================================\n")
        return

    print(f"{tag}   ⚠ {len(events_today)} T1 EVENT(S) TODAY — BLACKOUT WILL FIRE:")
    for ev_dt, name in events_today:
        flatten_at = ev_dt - timedelta(seconds=FLATTEN_LEAD_SEC)
        entry_blk_start = ev_dt - timedelta(seconds=ENTRY_LEAD_SEC)
        entry_blk_end   = ev_dt + timedelta(seconds=ENTRY_TRAIL_SEC)
        status = ""
        if now_et > entry_blk_end:
            status = "  (already past)"
        elif now_et >= entry_blk_start:
            status = "  *** BLACKOUT ACTIVE NOW ***"
        elif now_et >= flatten_at:
            status = "  *** FLATTEN WINDOW ACTIVE NOW ***"
        else:
            mins_to = (flatten_at - now_et).total_seconds() / 60
            status = f"  ({mins_to:.0f} min until flatten)"
        print(f"{tag}   • {name:<14} @ {ev_dt.strftime('%H:%M ET')}  "
              f"flatten {flatten_at.strftime('%H:%M:%S')}  "
              f"entries blocked {entry_blk_start.strftime('%H:%M:%S')}-"
              f"{entry_blk_end.strftime('%H:%M:%S')}{status}")
    print(f"{tag} ===============================================================\n")


class NewsBlackoutWatcher:
    """Background thread that wall-clock-monitors and force-flattens before T1 events.

    Responsibilities:
      - At T-FLATTEN_LEAD_SEC (2:30 before event): flatten all open strategy
        positions via direct executor calls (NT8 send_close_tag, MT5 _enqueue).
      - Clears the engines' `position` attribute so they don't try to manage
        a position that's been forcibly closed.
      - Clears coord._open_dir entries so the next entry signal isn't blocked
        by no-hedge rule against a phantom position.
      - Marks events as handled so flatten isn't repeated.

    Entry blocking is HANDLED SEPARATELY by the coordinator's _handle ENTRY
    path via is_in_blackout(). The watcher doesn't need to be running for
    entry blocking to work — they're decoupled by design.
    """

    def __init__(self, coord, nt8x=None, mt5x=None, poll_interval_sec: float = 1.0):
        self.coord = coord
        self.nt8x = nt8x
        self.mt5x = mt5x
        self.poll = poll_interval_sec
        self._handled: set[tuple[pd.Timestamp, str]] = set()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_summary_date = None   # for daily rollover detection

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        print(f"[news_blackout] watcher started.")
        # Print today's summary immediately
        now = pd.Timestamp.now(tz=ET_TZ)
        print_today_summary(now)
        self._last_summary_date = now.date()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="news_blackout")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                now = pd.Timestamp.now(tz=ET_TZ)
                # Daily summary print on calendar-date rollover
                today = now.date()
                if today != self._last_summary_date:
                    print_today_summary(now)
                    self._last_summary_date = today
                self._check(now)
            except Exception as e:
                print(f"[news_blackout] loop error: {e}")
            self._stop_event.wait(self.poll)

    def _check(self, now: pd.Timestamp) -> None:
        for ev_dt, name in EVENTS:
            if ev_dt < now - timedelta(minutes=10):
                continue
            if ev_dt > now + timedelta(minutes=10):
                break
            key = (ev_dt, name)
            if key in self._handled:
                continue
            flatten_at = ev_dt - timedelta(seconds=FLATTEN_LEAD_SEC)
            if now >= flatten_at:
                self._do_flatten(ev_dt, name)
                self._handled.add(key)

    def _do_flatten(self, ev_dt: pd.Timestamp, name: str) -> None:
        open_strats = list(self.coord._open_dir.keys())
        print(f"\n[news_blackout] === T-{FLATTEN_LEAD_SEC}s before {name} at "
              f"{ev_dt.strftime('%H:%M ET')} — flattening {len(open_strats)} "
              f"open position(s): {open_strats}")

        if not open_strats:
            print(f"[news_blackout] no open positions to flatten. Entries blocked until "
                  f"{(ev_dt + timedelta(seconds=ENTRY_TRAIL_SEC)).strftime('%H:%M:%S ET')}\n")
            return

        for strat_name in open_strats:
            engine = self.coord._engines.get(strat_name)
            if engine is not None and getattr(engine, "position", None) is not None:
                try:
                    engine.position = None
                    print(f"  [news_blackout] cleared {strat_name} engine position")
                except Exception as e:
                    print(f"  [news_blackout] failed to clear {strat_name} engine: {e}")

            self.coord._open_dir.pop(strat_name, None)

            reason = f"news_blackout_{name}"

            # NT8: find matching open tag, send CLOSE_TAG
            if self.nt8x is not None:
                sent = False
                for tag, op in list(self.nt8x._open.items()):
                    if op.strat == strat_name:
                        try:
                            self.nt8x.send_close_tag(tag, reason)
                            print(f"  [news_blackout] NT8 close sent for {strat_name} tag={tag}")
                            sent = True
                        except Exception as e:
                            print(f"  [news_blackout] NT8 close FAILED for {strat_name}: {e}")
                        break
                if not sent:
                    print(f"  [news_blackout] no NT8 tag matched for {strat_name}")

            # MT5: enqueue close (async via poster thread)
            if self.mt5x is not None:
                try:
                    self.mt5x._enqueue(("CLOSE", strat_name, reason))
                    print(f"  [news_blackout] MT5 close enqueued for {strat_name}")
                except Exception as e:
                    print(f"  [news_blackout] MT5 close FAILED for {strat_name}: {e}")

        print(f"[news_blackout] flatten complete. Entries blocked until "
              f"{(ev_dt + timedelta(seconds=ENTRY_TRAIL_SEC)).strftime('%H:%M:%S ET')}\n")
