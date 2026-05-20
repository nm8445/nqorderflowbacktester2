"""
Build 1-minute volumetric bar cache at 1-tick-per-level resolution.

Long-format Parquet, one row per (session_date, bar_open_time, level_price) — per-minute bars.
Coarser ticks-per-level settings are just a GROUP BY at query time.

Session window: 18:00 ET (prev day) -> 16:59:59 ET = 276 bars of 5 min each.
Empty 5-min bars are preserved as placeholder rows (level_price=NaN,
OHLC=NaN, volumes=0) so the full time grid is intact for market-structure
joins.

Data source per session:
  sessions >= 2025-03-13 with Databento coverage -> Databento MBP-1 trades
  otherwise                                      -> MarketTick (fixed parser)

Databento side convention: 'B' -> buy aggressor, 'A' -> sell aggressor,
'N' -> skipped. Matches src/nqbt/data/normalizer.py.

Output: D:/trading_pythonbacktest_data/volumetric_1min_1tpl.parquet
"""
import sys
import time
import zipfile
import pickle
from pathlib import Path
from datetime import datetime, date, timedelta
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_signal_cache_from_markettick import (
    extract_trades_with_side, parse_utc_ts, get_session_date
)

# ── Config ───────────────────────────────────────────────────────────────────
ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
MT_ZIP_DIR = DATA_DIR / "NQ L2 data"
DB_PARQUET_DIR = DATA_DIR / "parquet"
DB_CONSOLIDATED = DATA_DIR / "_tmp_db_trades_consolidated.parquet"
OUTPUT = DATA_DIR / "volumetric_1min_1tpl.parquet"

DB_START_DATE = "2025-03-13"        # Databento real coverage starts here
BAR_MINUTES = 1
TICK_SIZE = 0.25                    # 1 tick = 0.25 pt = 1 level
WORKERS = 8
# Shard output to one parquet per session (simpler than one big file during
# parallel build); final step consolidates into single OUTPUT file.
SHARD_DIR = DATA_DIR / "_tmp_volumetric_1min_shards"


# ── DB consolidation ─────────────────────────────────────────────────────────
def consolidate_db_trades():
    """Read all real DB parquets, extract trade rows only, dedupe, save
    consolidated parquet. Runs once at startup."""
    if DB_CONSOLIDATED.exists():
        print(f"[db] Consolidated DB trades already at {DB_CONSOLIDATED}", flush=True)
        return

    files = sorted(DB_PARQUET_DIR.glob("NQ_c_0_mbp-1_*.parquet"))
    # Skip stray 2024 file(s) — not part of the main DB series
    files = [f for f in files if "2024" not in f.name]
    print(f"[db] Consolidating {len(files)} DB parquet files...", flush=True)

    pieces = []
    bad_files = []
    for i, f in enumerate(files, 1):
        try:
            df = pd.read_parquet(f, columns=["ts_event", "action", "side", "price", "size", "sequence"])
        except Exception as e:
            bad_files.append((f.name, f.stat().st_size, str(e)[:80]))
            print(f"  [{i}/{len(files)}] SKIP CORRUPT {f.name} ({f.stat().st_size} bytes): {str(e)[:60]}",
                  flush=True)
            continue
        # Trades only, with known aggressor
        mask = (df["action"] == "T") & (df["side"].isin(["A", "B"]))
        tr = df[mask][["ts_event", "side", "price", "size", "sequence"]].copy()
        pieces.append(tr)
        if i % 40 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {f.name}  +{len(tr):,} trades", flush=True)
    if bad_files:
        print(f"\n[db] WARNING: skipped {len(bad_files)} corrupt file(s):")
        for name, sz, err in bad_files:
            print(f"    {name}  ({sz} bytes)  {err}")

    all_tr = pd.concat(pieces, ignore_index=True)
    print(f"[db] Pre-dedupe rows: {len(all_tr):,}")
    # Dedupe on the unique trade identifiers (overlapping files yield duplicates)
    all_tr = all_tr.drop_duplicates(subset=["ts_event", "sequence", "price", "size", "side"])
    all_tr = all_tr.sort_values("ts_event").reset_index(drop=True)
    print(f"[db] Post-dedupe rows: {len(all_tr):,}")

    # Map aggressor
    all_tr["aggressor"] = all_tr["side"].map({"B": "buy", "A": "sell"})
    all_tr = all_tr[["ts_event", "price", "size", "aggressor"]].rename(columns={"ts_event": "ts"})
    all_tr["ts"] = pd.to_datetime(all_tr["ts"], utc=True)

    all_tr.to_parquet(DB_CONSOLIDATED, index=False, compression="snappy")
    print(f"[db] Saved {DB_CONSOLIDATED.name}  size={DB_CONSOLIDATED.stat().st_size/1e6:.1f} MB",
          flush=True)


# ── MT indexing ──────────────────────────────────────────────────────────────
def index_mt_csvs():
    all_entries = []
    for zf_path in sorted(MT_ZIP_DIR.glob("*.zip")):
        with zipfile.ZipFile(zf_path) as zf:
            for name in zf.namelist():
                if name.endswith(".csv"):
                    basename = name.split("/")[-1].replace(".csv", "")
                    if len(basename) == 8 and basename.isdigit():
                        all_entries.append((str(zf_path), name, basename))
    csv_by_date = defaultdict(list)
    for zf_str, csv_name, date_str in all_entries:
        csv_by_date[date_str].append((zf_str, csv_name))
    return csv_by_date


# ── Core: build bars for one session ─────────────────────────────────────────
def load_session_trades_mt(session_date_iso, csv_by_date):
    """Returns a DataFrame with columns [ts, price, size, aggressor] for the session."""
    d = datetime.strptime(session_date_iso, "%Y-%m-%d").date()
    prev = (d - timedelta(days=1)).strftime("%Y%m%d")
    curr = d.strftime("%Y%m%d")
    entries = []
    if prev in csv_by_date: entries.extend(csv_by_date[prev])
    if curr in csv_by_date: entries.extend(csv_by_date[curr])
    if not entries: return None

    state = {"best_bid": None, "best_ask": None,
             "last_trade_price": None, "last_aggressor": "buy"}
    rows = []
    for zip_path_str, csv_name in entries:
        with zipfile.ZipFile(zip_path_str) as zf:
            rows.extend(extract_trades_with_side(zf, csv_name, state=state))
    if not rows:
        return None
    # Convert list-of-tuples into DataFrame
    ts_strs = [r[0] for r in rows]
    prices = [r[1] for r in rows]
    sizes = [r[2] for r in rows]
    aggressors = [r[3] for r in rows]
    # Vectorized ts parse
    ts = pd.to_datetime([s[:14] + "." + (s[14:] + "000000")[:6] for s in ts_strs],
                         format="%Y%m%d%H%M%S.%f", utc=True)
    df = pd.DataFrame({"ts": ts, "price": prices, "size": sizes, "aggressor": aggressors})
    df = df.sort_values("ts").reset_index(drop=True)
    # Filter to session window
    d_obj = datetime.strptime(session_date_iso, "%Y-%m-%d").date()
    start_et = pd.Timestamp(d_obj - timedelta(days=1), tz=ET).replace(hour=18, minute=0, second=0)
    end_et   = pd.Timestamp(d_obj, tz=ET).replace(hour=17, minute=0, second=0)
    start_utc = start_et.tz_convert("UTC"); end_utc = end_et.tz_convert("UTC")
    df = df[(df["ts"] >= start_utc) & (df["ts"] < end_utc)]
    return df


def load_session_trades_db(session_date_iso, db_trades_path):
    """Read consolidated DB trades, filter to session window."""
    d = datetime.strptime(session_date_iso, "%Y-%m-%d").date()
    start_et = pd.Timestamp(d - timedelta(days=1), tz=ET).replace(hour=18, minute=0, second=0)
    end_et   = pd.Timestamp(d, tz=ET).replace(hour=17, minute=0, second=0)
    start_utc = start_et.tz_convert("UTC"); end_utc = end_et.tz_convert("UTC")
    df = pd.read_parquet(
        db_trades_path,
        filters=[("ts", ">=", start_utc.to_pydatetime()),
                 ("ts", "<",  end_utc.to_pydatetime())],
    )
    return df if not df.empty else None


def build_bars_from_trades(trades_df, session_date_iso):
    """Bucket trades into 5-min bars + 1-tick levels. Return long-format DataFrame.
    Includes placeholder rows for empty bars (full 276-bar grid)."""
    d = datetime.strptime(session_date_iso, "%Y-%m-%d").date()
    # Full 276-bar grid in ET, 5-min clock-aligned starting 18:00 prev day
    start_et = pd.Timestamp(d - timedelta(days=1), tz=ET).replace(hour=18, minute=0, second=0)
    end_et   = pd.Timestamp(d, tz=ET).replace(hour=17, minute=0, second=0)
    grid = pd.date_range(start_et, end_et, freq=f"{BAR_MINUTES}min", inclusive="left", tz=ET)
    # grid should have (23h / 5min) = 276 entries

    if trades_df is None or trades_df.empty:
        # All empty bars
        return pd.DataFrame({
            "session_date": session_date_iso,
            "bar_open_time": grid,
            "bar_close_time": grid + pd.Timedelta(minutes=BAR_MINUTES),
            "open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan,
            "bar_total_vol": np.int32(0),
            "level_price": np.nan,
            "buy_vol": np.int32(0), "sell_vol": np.int32(0),
        })

    # Work in ET for bucketing
    tr = trades_df.copy()
    tr["ts_et"] = tr["ts"].dt.tz_convert(ET)
    tr["bar_open_time"] = tr["ts_et"].dt.floor(f"{BAR_MINUTES}min")
    # Clip to grid bounds (shouldn't be needed if filter was correct)
    tr = tr[(tr["bar_open_time"] >= grid[0]) & (tr["bar_open_time"] <= grid[-1])]

    # OHLC + totals per bar
    ohlc = tr.groupby("bar_open_time").agg(
        open =("price", "first"),
        high =("price", "max"),
        low  =("price", "min"),
        close=("price", "last"),
        bar_total_vol=("size", "sum"),
    )

    # Level rows (1-tick resolution = raw price)
    tr_agg = (tr.assign(buy=np.where(tr["aggressor"] == "buy", tr["size"], 0),
                         sell=np.where(tr["aggressor"] == "sell", tr["size"], 0))
                .groupby(["bar_open_time", "price"])
                .agg(buy_vol=("buy", "sum"), sell_vol=("sell", "sum"))
                .reset_index()
                .rename(columns={"price": "level_price"}))

    # Attach OHLC / total to each level row
    tr_agg = tr_agg.merge(ohlc, left_on="bar_open_time", right_index=True, how="left")

    # Build placeholder rows for empty bars
    occupied = set(ohlc.index)
    empty = [g for g in grid if g not in occupied]
    if empty:
        placeholder = pd.DataFrame({
            "bar_open_time": empty,
            "level_price": np.nan,
            "buy_vol": np.int32(0),
            "sell_vol": np.int32(0),
            "open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan,
            "bar_total_vol": np.int32(0),
        })
        out = pd.concat([tr_agg, placeholder], ignore_index=True)
    else:
        out = tr_agg

    out["session_date"] = session_date_iso
    out["bar_close_time"] = out["bar_open_time"] + pd.Timedelta(minutes=BAR_MINUTES)
    out = out.sort_values(["bar_open_time", "level_price"]).reset_index(drop=True)
    # Cast
    out["buy_vol"] = out["buy_vol"].astype(np.int32)
    out["sell_vol"] = out["sell_vol"].astype(np.int32)
    out["bar_total_vol"] = out["bar_total_vol"].fillna(0).astype(np.int32)
    return out[["session_date", "bar_open_time", "bar_close_time",
                "open", "high", "low", "close", "bar_total_vol",
                "level_price", "buy_vol", "sell_vol"]]


def process_session(args):
    session_iso, source, csv_by_date, db_path = args
    shard = SHARD_DIR / f"{session_iso}.parquet"
    if shard.exists():
        return (session_iso, source, "cached", 0)

    try:
        if source == "db":
            trades = load_session_trades_db(session_iso, db_path)
        else:
            trades = load_session_trades_mt(session_iso, csv_by_date)

        bars_df = build_bars_from_trades(trades, session_iso)
        SHARD_DIR.mkdir(parents=True, exist_ok=True)
        bars_df.to_parquet(shard, index=False, compression="snappy")
        n_rows = len(bars_df)
        return (session_iso, source, "ok", n_rows)
    except Exception as e:
        return (session_iso, source, f"error:{e}", 0)


# ── Main orchestrator ────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # Step 1: consolidate DB trades
    consolidate_db_trades()

    # Step 2: index MT zips
    print("[mt] Indexing MT zips...", flush=True)
    csv_by_date = index_mt_csvs()
    mt_cal_dates = sorted(csv_by_date.keys())
    print(f"[mt] Calendar dates: {mt_cal_dates[0]} to {mt_cal_dates[-1]} ({len(mt_cal_dates)})")

    # Step 3: figure out DB coverage from consolidated file
    db_df_head = pd.read_parquet(DB_CONSOLIDATED, columns=["ts"])
    db_min = db_df_head["ts"].min(); db_max = db_df_head["ts"].max()
    print(f"[db] Consolidated DB covers ts_event {db_min} to {db_max}")
    # Session dates that DB can cover: roughly (db_min ET + 1 day) to (db_max ET)
    db_session_start = (db_min.tz_convert(ET).date() + timedelta(days=1))
    db_session_end   = db_max.tz_convert(ET).date()
    # But we enforce user cutoff: DB only from 2025-03-13 forward
    db_session_start = max(db_session_start, datetime.strptime(DB_START_DATE, "%Y-%m-%d").date())

    # Step 4: build task list
    first_cal = datetime.strptime(mt_cal_dates[0], "%Y%m%d").date()
    last_cal = db_session_end  # Build through DB's last session
    tasks = []
    cur = first_cal
    while cur <= last_cal:
        if cur.weekday() < 5:
            sess_iso = cur.strftime("%Y-%m-%d")
            shard = SHARD_DIR / f"{sess_iso}.parquet"
            if not shard.exists():
                if db_session_start <= cur <= db_session_end:
                    source = "db"; csv_ref = None
                else:
                    source = "mt"; csv_ref = csv_by_date
                tasks.append((sess_iso, source, csv_ref, str(DB_CONSOLIDATED)))
        cur += timedelta(days=1)

    mt_tasks = sum(1 for t in tasks if t[1] == "mt")
    db_tasks = sum(1 for t in tasks if t[1] == "db")
    print(f"\n[tasks] Total: {len(tasks)}  MT: {mt_tasks}  DB: {db_tasks}")
    print(f"[tasks] DB preferred range: {db_session_start} to {db_session_end}")
    print(f"[tasks] Workers: {WORKERS}", flush=True)

    if not tasks:
        print("Nothing to do.")
    else:
        # Step 5: parallel build
        SHARD_DIR.mkdir(parents=True, exist_ok=True)
        done = ok = errs = 0
        t_build = time.time()
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(process_session, t): t[0] for t in tasks}
            for fut in as_completed(futs):
                sess = futs[fut]
                try:
                    _, src, status, n_rows = fut.result()
                    done += 1
                    if status == "ok": ok += 1
                    elif status.startswith("error"): errs += 1
                    if done <= 5 or done % 50 == 0:
                        elapsed = time.time() - t_build
                        rate = done / elapsed if elapsed > 0 else 0.0
                        eta_min = ((len(tasks) - done) / rate / 60) if rate > 0 else 0.0
                        print(f"  [{done}/{len(tasks)}] {sess} [{src}] {status} rows={n_rows}  "
                              f"elapsed={elapsed:.0f}s rate={rate:.2f}/s ETA={eta_min:.1f}min",
                              flush=True)
                except Exception as e:
                    errs += 1
                    print(f"  ERROR {sess}: {e}", flush=True)
        print(f"\n[build] Done. ok={ok} errs={errs} elapsed={time.time()-t_build:.0f}s")

    # Step 6: consolidate shards into one parquet
    print(f"\n[merge] Merging {len(list(SHARD_DIR.glob('*.parquet')))} shards into {OUTPUT}...", flush=True)
    shards = sorted(SHARD_DIR.glob("*.parquet"))
    if not shards:
        print("[merge] No shards to merge.")
        return
    # Stream shards into a single parquet file via pyarrow writer
    first = pq.read_table(shards[0])
    writer = pq.ParquetWriter(OUTPUT, first.schema, compression="snappy")
    writer.write_table(first)
    for i, sh in enumerate(shards[1:], 2):
        tbl = pq.read_table(sh, schema=first.schema)
        writer.write_table(tbl)
        if i % 100 == 0: print(f"  [{i}/{len(shards)}]", flush=True)
    writer.close()
    print(f"[merge] Wrote {OUTPUT}  size={OUTPUT.stat().st_size/1e6:.1f} MB")
    print(f"\nTOTAL: {time.time()-t0:.0f}s  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
