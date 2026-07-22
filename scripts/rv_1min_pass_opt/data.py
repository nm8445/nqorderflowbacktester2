"""Data loading for the RV 1-min pass-rate study.

- load_bars(): clean contiguous 1-min OHLC stream (markettick_1min_bars, UTC right-labeled).
- build_orderflow_flags(): per-bar long_ok/short_ok from the 1-min volumetric, using the windowed
  absorption check. CONFIG-INDEPENDENT (only WINDOW_N_TICKS / WINDOW_D), so cache once and reuse
  across stages 1-4. Keyed by bar_close_time in UTC to join the markettick bars 1:1.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from live.combined.rv_orderflow import windowed_absorption_check, NQ_TICK  # noqa: E402

DATA = Path("D:/trading_pythonbacktest_data")
BARS = DATA / "markettick_1min_bars.parquet"
VOL = DATA / "volumetric_1min_1tpl.parquet"
CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)


def load_bars(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Contiguous 1-min OHLC, UTC index (bar CLOSE time). Adds ET-local helper columns."""
    b = pd.read_parquet(BARS, columns=["open", "high", "low", "close", "volume"])
    if b.index.tz is None:
        b.index = b.index.tz_localize("UTC")
    b = b.sort_index()
    if start:
        b = b[b.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        b = b[b.index <= pd.Timestamp(end, tz="UTC")]
    et = b.index.tz_convert("America/New_York")
    b["et"] = et
    b["session_break"] = b.index.to_series().diff().gt(pd.Timedelta(minutes=1)).to_numpy()
    return b


def _flags_for_session(df: pd.DataFrame, window_n: int, window_d: float):
    """df = volumetric rows for ONE session (cols bar_open_time, high, low, level_price,
    buy_vol, sell_vol). Returns list of (bar_close_utc, long_ok, short_ok) for non-empty bars."""
    out = []
    # group rows by bar; bar_open_time is repeated per level row
    for bot, g in df.groupby("bar_open_time", sort=True):
        lp = g["level_price"].to_numpy()
        m = np.isfinite(lp)
        if not m.any():
            continue  # empty placeholder bar -> no orderflow, can't be an entry
        hi = g["high"].iloc[0]
        lo = g["low"].iloc[0]
        if not (np.isfinite(hi) and np.isfinite(lo)) or hi <= lo:
            continue
        levels = {float(p): (int(bv), int(sv))
                  for p, bv, sv in zip(lp[m], g["buy_vol"].to_numpy()[m], g["sell_vol"].to_numpy()[m])}
        long_ok, short_ok = windowed_absorption_check(hi, lo, levels, window_n, window_d)
        if long_ok or short_ok:
            close_utc = (bot + pd.Timedelta(minutes=1)).tz_convert("UTC")
            out.append((close_utc, long_ok, short_ok))
    return out


def _mags_for_session(df: pd.DataFrame, window_n: int):
    """Per bar: (bar_close_utc, best_long_mag, best_short_mag) where best_long_mag = most-NEGATIVE
    lower-half window sum, best_short_mag = most-POSITIVE upper-half window sum. Lets any threshold
    d sweep for free: long_ok(d)=best_long_mag<=-d, short_ok(d)=best_short_mag>=d."""
    out = []
    for bot, g in df.groupby("bar_open_time", sort=True):
        lp = g["level_price"].to_numpy()
        m = np.isfinite(lp)
        if not m.any():
            continue
        hi = g["high"].iloc[0]; lo = g["low"].iloc[0]
        if not (np.isfinite(hi) and np.isfinite(lo)) or hi <= lo:
            continue
        mid = (hi + lo) / 2.0
        pr = lp[m]; delta = (g["buy_vol"].to_numpy()[m] - g["sell_vol"].to_numpy()[m]).astype(float)
        best_long = 0.0; best_short = 0.0
        for half in ("low", "up"):
            sel = pr < mid if half == "low" else pr > mid
            if sel.sum() < window_n:
                continue
            p = pr[sel]; d = delta[sel]
            nslot = int(round((p.max() - p.min()) / NQ_TICK)) + 1
            if nslot < window_n:
                continue
            arr = np.zeros(nslot); arr[np.rint((p - p.min()) / NQ_TICK).astype(int)] = d
            cs = np.concatenate(([0.0], np.cumsum(arr)))
            w = cs[window_n:] - cs[:-window_n]
            if half == "low":
                best_long = float(w.min())
            else:
                best_short = float(w.max())
        close_utc = (bot + pd.Timedelta(minutes=1)).tz_convert("UTC")
        out.append((close_utc, best_long, best_short))
    return out


def build_orderflow_magnitudes(window_n: int = 8, force: bool = False) -> Path:
    """One-time pass -> cache/orderflow_mags_n{n}.parquet (bar_close_utc, best_long_mag,
    best_short_mag). Any d threshold then sweeps for free."""
    out_path = CACHE / f"orderflow_mags_n{window_n}.parquet"
    if out_path.exists() and not force:
        print(f"[mags] cached: {out_path.name}")
        return out_path
    t0 = time.time()
    print(f"[mags] reading volumetric ...", flush=True)
    v = pd.read_parquet(VOL, columns=["session_date", "bar_open_time", "high", "low",
                                      "level_price", "buy_vol", "sell_vol"])
    if pd.api.types.is_datetime64_any_dtype(v["bar_open_time"]) and v["bar_open_time"].dt.tz is None:
        v["bar_open_time"] = v["bar_open_time"].dt.tz_localize("America/New_York")
    rows = []
    sessions = list(v.groupby("session_date", sort=True))
    for i, (sd, g) in enumerate(sessions, 1):
        rows.extend(_mags_for_session(g, window_n))
        if i % 300 == 0 or i == len(sessions):
            print(f"  [{i}/{len(sessions)}] {sd}  {time.time()-t0:.0f}s", flush=True)
    out = pd.DataFrame(rows, columns=["bar_close_utc", "best_long_mag", "best_short_mag"])
    out = out.drop_duplicates("bar_close_utc").set_index("bar_close_utc").sort_index()
    out.to_parquet(out_path, compression="snappy")
    print(f"[mags] wrote {out_path.name}: {len(out):,} bars, {time.time()-t0:.0f}s")
    return out_path


def build_orderflow_flags(window_n: int = 8, window_d: float = 150, force: bool = False) -> Path:
    """One-time heavy pass -> cache/orderflow_flags_n{n}_d{d}.parquet
    (bar_close_utc, long_ok, short_ok). Only bars where some side passes are stored
    (sparse); a bar absent from the cache means both flags False."""
    out_path = CACHE / f"orderflow_flags_n{window_n}_d{int(window_d)}.parquet"
    if out_path.exists() and not force:
        print(f"[flags] cached: {out_path.name}")
        return out_path

    t0 = time.time()
    print(f"[flags] reading volumetric ({VOL.name}) ...", flush=True)
    v = pd.read_parquet(VOL, columns=["session_date", "bar_open_time", "high", "low",
                                      "level_price", "buy_vol", "sell_vol"])
    if pd.api.types.is_datetime64_any_dtype(v["bar_open_time"]) and v["bar_open_time"].dt.tz is None:
        v["bar_open_time"] = v["bar_open_time"].dt.tz_localize("America/New_York")
    print(f"[flags] {len(v):,} rows, {v['session_date'].nunique()} sessions; computing flags ...",
          flush=True)

    rows = []
    sessions = list(v.groupby("session_date", sort=True))
    for i, (sd, g) in enumerate(sessions, 1):
        rows.extend(_flags_for_session(g, window_n, window_d))
        if i % 200 == 0 or i == len(sessions):
            el = time.time() - t0
            print(f"  [{i}/{len(sessions)}] {sd}  hits={len(rows):,}  {el:.0f}s", flush=True)

    out = pd.DataFrame(rows, columns=["bar_close_utc", "long_ok", "short_ok"])
    out = out.drop_duplicates("bar_close_utc").set_index("bar_close_utc").sort_index()
    out.to_parquet(out_path, compression="snappy")
    print(f"[flags] wrote {out_path.name}: {len(out):,} bars with a passing side, "
          f"{(time.time()-t0):.0f}s")
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--d", type=float, default=150)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build_orderflow_flags(a.n, a.d, a.force)
