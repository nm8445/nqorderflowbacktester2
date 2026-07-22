"""Resolve the ACTUAL futures contract the live feed's continuous symbol maps to,
so EXECUTION (the NT8 addon) trades exactly what the DATA feed is on — no roll mismatch.

Why this exists: `NQ.v.0` (volume continuous) rolls on Databento's volume crossover — a
variable, *lagging* date. The NT8 addon's own `FrontMonth()` rolls on a fixed calendar date.
Those two diverge mid-roll, which re-creates the ~300-pt data/execution mismatch. This module
lets the order path stamp each order with the feed's REAL contract month so they can't diverge.

Flow: query a recent bar of DATABENTO_SYMBOL -> instrument_id -> resolve to the raw contract
('NQM6') -> month code ('06-26'). Cached once/day in state/active_contract.json. The order path
appends it to the root: 'MNQ' -> 'MNQ 06-26'. Network is hit only on the first call of a new day;
on any failure callers fall back to the addon's own FrontMonth(), so an order is never blocked.

Run `python live/combined/active_contract.py` to print/refresh the current contract.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

from live.combined.config import (
    DATABENTO_DATASET, DATABENTO_SYMBOL, DATABENTO_STYPE, STATE_DIR,
)

CACHE = STATE_DIR / "active_contract.json"

# CME month codes (F=Jan .. Z=Dec). Quarterly NQ uses H/M/U/Z but parse any.
CODE_MONTH = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
              "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}
# Lazy root + a single month-code letter + 1-2 digit year, anchored to end so "NQM6"
# parses as root=NQ, month=M, year=6 (not root=N, month=Q).
_RAW_RE = re.compile(r"^([A-Z]+?)([FGHJKMNQUVXZ])(\d{1,2})$")


def parse_raw_to_code(raw: str) -> str | None:
    """'NQM6' / 'MNQU6' -> 'MM-YY' ('06-26'). Returns None if unparseable."""
    m = _RAW_RE.match(raw.strip().upper())
    if not m:
        return None
    month = CODE_MONTH[m.group(2)]
    yy = int(m.group(3))
    if yy >= 10:                       # explicit 2-digit year ('26')
        year = 2000 + yy
    else:                              # single-digit ('6') -> the near year ending in yy, decade-aware
        ny = dt.date.today().year      # (so 'NQH0' resolves to 2030, not 2020 — robust past 2029)
        year = ny - (ny % 10) + yy
        if year < ny - 1:              # contract digit belongs to the next decade
            year += 10
    return f"{month:02d}-{year % 100:02d}"


MONTH_TO_CODE = {v: k for k, v in CODE_MONTH.items()}   # 6 -> 'M'
NQ_QUARTERS = (3, 6, 9, 12)                             # H, M, U, Z


def _third_friday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 1)
    return d + dt.timedelta(days=(4 - d.weekday()) % 7 + 14)


def _nq_raw(year: int, month: int) -> str:
    """('NQ' + month code + single-digit year), e.g. (2026, 6) -> 'NQM6'."""
    return f"NQ{MONTH_TO_CODE[month]}{year % 10}"


def _nearest_quarterlies(today: dt.date, n: int = 2) -> list[str]:
    """The `n` nearest NQ quarterly contracts whose 3rd-Friday expiry is >= today, as raw symbols.
    On 2026-06-16 -> ['NQM6', 'NQU6'] (June + Sept)."""
    qs = []
    for yo in range(0, 2):
        for m in NQ_QUARTERS:
            yr = today.year + yo
            exp = _third_friday(yr, m)
            if exp >= today:
                qs.append((exp, yr, m))
    qs.sort()
    return [_nq_raw(yr, m) for _, yr, m in qs[:n]]


def resolve_active_detail(symbol: str = DATABENTO_SYMBOL) -> dict | None:
    """Pick the contract with the HIGHEST RECENT VOLUME among the nearest two NQ quarterlies — the true
    lead. Databento's continuous `v.0`/`c.0` LAG the real volume roll by ~2-3 days, so near expiry they
    keep reporting the dying front contract (which becomes untradeable) while the market has moved to the
    back month. Direct volume comparison tracks the actual market (and TradingView). Returns
    {'raw','code','vols'} or None on any failure. `symbol` is unused (kept for signature compat)."""
    key = os.getenv("DATABENTO_API_KEY")
    if not key:
        return None
    try:
        import pandas as pd
        import databento as db
        c = db.Historical(key)
        cands = _nearest_quarterlies(dt.date.today(), n=2)   # e.g. ['NQM6', 'NQU6']
        # Clamp the query end to Databento's published end so the variable publication lag never errors.
        rng = c.metadata.get_dataset_range(dataset=DATABENTO_DATASET)
        avail_end = pd.Timestamp(rng["schema"]["ohlcv-1m"]["end"])
        end = min(pd.Timestamp.now(tz="UTC"), avail_end)
        start = end - pd.Timedelta(hours=8)
        df = c.timeseries.get_range(
            dataset=DATABENTO_DATASET, schema="ohlcv-1m", symbols=cands,
            stype_in="raw_symbol", start=start, end=end).to_df()
        if df.empty or "symbol" not in df.columns:
            return None
        vols = df.groupby("symbol")["volume"].sum().to_dict()
        winner = max(cands, key=lambda s: vols.get(s, 0))    # highest recent volume (front on tie)
        code = parse_raw_to_code(winner)
        if not code:
            return None
        return {"raw": winner, "code": code, "vols": {s: int(vols.get(s, 0)) for s in cands}}
    except Exception:
        return None


def resolve_active(symbol: str = DATABENTO_SYMBOL) -> str | None:
    """Just the 'MM-YY' code (None on failure)."""
    d = resolve_active_detail(symbol)
    return d["code"] if d else None


def _read_cache() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except Exception:
        return {}


def get_active_code(refresh_if_stale: bool = True) -> str | None:
    """Return the feed's current contract month as 'MM-YY'. Reads today's cached value; only
    hits the network (once) when the cache is missing/old. On a resolve failure returns the last
    known code (better than nothing) or None. Never raises."""
    today = dt.date.today().isoformat()
    cached = _read_cache()
    if cached.get("date") == today and cached.get("code") and cached.get("symbol") == DATABENTO_SYMBOL:
        return cached["code"]
    if not refresh_if_stale:
        return cached.get("code")
    code = resolve_active()
    if code:
        try:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(
                {"date": today, "symbol": DATABENTO_SYMBOL, "code": code,
                 "resolved_utc": dt.datetime.now(dt.timezone.utc).isoformat()}, indent=2))
        except Exception:
            pass
        return code
    return cached.get("code")   # stale fallback


def get_active_raw(refresh_if_stale: bool = True) -> str | None:
    """The volume-lead RAW contract symbol ('NQU6'). Same cache as get_active_code; only hits the
    network when stale. Returns the last-known raw on failure (or None)."""
    cached = _read_cache()
    if cached.get("date") == dt.date.today().isoformat() and cached.get("raw"):
        return cached["raw"]
    if not refresh_if_stale:
        return cached.get("raw")
    info = refresh_now()
    return info["raw"] if info else cached.get("raw")


def feed_symbol() -> tuple[str, str]:
    """(symbol, stype_in) for the LIVE FEED + historical fetches. Returns the volume-lead RAW contract
    ('NQU6', 'raw_symbol') so the feed tracks the actually-liquid contract — NOT Databento's lagging
    continuous. Falls back to the configured continuous (NQ.v.0, 'continuous') if unresolved, so the
    feed never fails to start."""
    raw = get_active_raw()
    if raw:
        return raw, "raw_symbol"
    return DATABENTO_SYMBOL, DATABENTO_STYPE


def refresh_now(symbol: str = DATABENTO_SYMBOL) -> dict | None:
    """Force a fresh resolve (bypass the daily cache), update the cache, and return
    {'symbol','raw','code'} or None. For a BACKGROUND refresher that wants guaranteed freshness within
    its own interval — catches a roll even mid-session (e.g. the 18:00 ET CME boundary)."""
    d = resolve_active_detail(symbol)
    if d and d["code"]:
        try:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(
                {"date": dt.date.today().isoformat(), "symbol": symbol,
                 "raw": d["raw"], "code": d["code"], "vols": d.get("vols", {}),
                 "resolved_utc": dt.datetime.now(dt.timezone.utc).isoformat()}, indent=2))
        except Exception:
            pass
        return {"symbol": symbol, "raw": d["raw"], "code": d["code"], "vols": d.get("vols", {})}
    return None


if __name__ == "__main__":
    print(f"symbol  : {DATABENTO_SYMBOL}")
    print(f"resolved: {resolve_active()}")
    print(f"cached  : {get_active_code()}  -> {CACHE}")
