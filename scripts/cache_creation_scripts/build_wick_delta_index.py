"""
Build wick-delta index + per-bucket wick detail from a volumetric 1-TPL cache.

Per-bar index (summary, one row per bar):
  session_date, bar_open_time, bar_close_time, open, high, low, close,
  is_bullish, is_bearish, bar_total_vol, bar_total_buy, bar_total_sell,
  bar_total_delta, bot_wick_low, bot_wick_top, bot_wick_buy, bot_wick_sell,
  bot_wick_delta, top_wick_low, top_wick_high, top_wick_buy, top_wick_sell,
  top_wick_delta.

Per-bucket table (flat, one row per (bar, wick_side, bucket_price) at 5-tick
granularity = 1.25 pts): session_date, bar_open_time, wick_side ('bot'/'top'),
bucket_price, buy, sell, delta.

Usage:
    python build_wick_delta_index.py --minutes 5
    python build_wick_delta_index.py --minutes 1
"""
import sys, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

DATA_DIR = Path("D:/trading_pythonbacktest_data")
BUCKET_PTS = 1.25       # 5 ticks
WORKERS = 6             # leave headroom for other builds running in parallel


def process_session(args):
    session, src_path = args
    df = pd.read_parquet(src_path, filters=[("session_date", "=", session)])
    if df.empty:
        return None, None

    # Metadata per bar (placeholder rows count too for the grid — OHLC is NaN there)
    bar_meta = (df.drop_duplicates("bar_open_time")
                  [["session_date", "bar_open_time", "bar_close_time",
                    "open", "high", "low", "close", "bar_total_vol"]]
                  .set_index("bar_open_time"))

    # Levels only (drop placeholder empty-bar rows)
    levels = df[df["level_price"].notna()].copy()

    # Total buy/sell per bar
    if not levels.empty:
        tot = levels.groupby("bar_open_time").agg(
            bar_total_buy=("buy_vol", "sum"),
            bar_total_sell=("sell_vol", "sum"),
        )
    else:
        tot = pd.DataFrame(columns=["bar_total_buy", "bar_total_sell"],
                            index=pd.Index([], name="bar_open_time"))

    # Body bottom / top
    levels["body_bottom"] = levels[["open", "close"]].min(axis=1)
    levels["body_top"] = levels[["open", "close"]].max(axis=1)

    # Bot wick rows (strictly below body bottom)
    bot_rows = levels[levels["level_price"] < levels["body_bottom"]]
    if not bot_rows.empty:
        bot = bot_rows.groupby("bar_open_time").agg(
            bot_wick_low=("level_price", "min"),
            bot_wick_top=("level_price", "max"),
            bot_wick_buy=("buy_vol", "sum"),
            bot_wick_sell=("sell_vol", "sum"),
        )
    else:
        bot = pd.DataFrame(columns=["bot_wick_low", "bot_wick_top",
                                     "bot_wick_buy", "bot_wick_sell"],
                            index=pd.Index([], name="bar_open_time"))

    # Top wick rows (strictly above body top)
    top_rows = levels[levels["level_price"] > levels["body_top"]]
    if not top_rows.empty:
        top = top_rows.groupby("bar_open_time").agg(
            top_wick_low=("level_price", "min"),
            top_wick_high=("level_price", "max"),
            top_wick_buy=("buy_vol", "sum"),
            top_wick_sell=("sell_vol", "sum"),
        )
    else:
        top = pd.DataFrame(columns=["top_wick_low", "top_wick_high",
                                     "top_wick_buy", "top_wick_sell"],
                            index=pd.Index([], name="bar_open_time"))

    index = bar_meta.join(tot).join(bot).join(top)

    # Fill NaN numeric cols with 0 for bars without wicks (or without trades)
    for col in ["bar_total_buy", "bar_total_sell",
                 "bot_wick_buy", "bot_wick_sell",
                 "top_wick_buy", "top_wick_sell"]:
        index[col] = index[col].fillna(0).astype("int64")

    # Delta columns
    index["bar_total_delta"] = index["bar_total_buy"] - index["bar_total_sell"]
    index["bot_wick_delta"] = index["bot_wick_buy"] - index["bot_wick_sell"]
    index["top_wick_delta"] = index["top_wick_buy"] - index["top_wick_sell"]

    # Direction flags
    index["is_bullish"] = (index["close"] > index["open"]).fillna(False)
    index["is_bearish"] = (index["close"] < index["open"]).fillna(False)

    index = index.reset_index()[[
        "session_date", "bar_open_time", "bar_close_time",
        "open", "high", "low", "close",
        "is_bullish", "is_bearish",
        "bar_total_vol", "bar_total_buy", "bar_total_sell", "bar_total_delta",
        "bot_wick_low", "bot_wick_top", "bot_wick_buy", "bot_wick_sell", "bot_wick_delta",
        "top_wick_low", "top_wick_high", "top_wick_buy", "top_wick_sell", "top_wick_delta",
    ]]

    # Per-bucket table (flat)
    bucket_pieces = []
    if not bot_rows.empty:
        br = bot_rows.copy()
        br["bucket_price"] = (np.floor(br["level_price"] / BUCKET_PTS) * BUCKET_PTS).round(4)
        br_agg = br.groupby(["bar_open_time", "bucket_price"]).agg(
            buy=("buy_vol", "sum"), sell=("sell_vol", "sum")
        ).reset_index()
        br_agg["wick_side"] = "bot"
        bucket_pieces.append(br_agg)
    if not top_rows.empty:
        tr = top_rows.copy()
        tr["bucket_price"] = (np.floor(tr["level_price"] / BUCKET_PTS) * BUCKET_PTS).round(4)
        tr_agg = tr.groupby(["bar_open_time", "bucket_price"]).agg(
            buy=("buy_vol", "sum"), sell=("sell_vol", "sum")
        ).reset_index()
        tr_agg["wick_side"] = "top"
        bucket_pieces.append(tr_agg)
    if bucket_pieces:
        buckets = pd.concat(bucket_pieces, ignore_index=True)
        buckets["delta"] = buckets["buy"] - buckets["sell"]
        buckets["session_date"] = session
        buckets = buckets[["session_date", "bar_open_time", "wick_side",
                           "bucket_price", "buy", "sell", "delta"]]
    else:
        buckets = pd.DataFrame(columns=["session_date", "bar_open_time", "wick_side",
                                         "bucket_price", "buy", "sell", "delta"])

    return index, buckets


def main(tf_min):
    src = DATA_DIR / f"volumetric_{tf_min}min_1tpl.parquet"
    out_idx = DATA_DIR / f"wick_delta_index_{tf_min}min.parquet"
    out_buckets = DATA_DIR / f"wick_delta_buckets_{tf_min}min.parquet"

    if not src.exists():
        print(f"Input not found: {src}")
        sys.exit(1)

    print(f"Source: {src.name}", flush=True)
    print(f"Output: {out_idx.name}, {out_buckets.name}", flush=True)

    sessions = sorted(pd.read_parquet(src, columns=["session_date"])["session_date"].unique())
    print(f"Sessions: {len(sessions)}", flush=True)

    idx_shards, bucket_shards = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_session, (s, str(src))): s for s in sessions}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                idx, bkt = fut.result()
                if idx is not None and not idx.empty:
                    idx_shards.append(idx)
                if bkt is not None and not bkt.empty:
                    bucket_shards.append(bkt)
            except Exception as e:
                print(f"  ERROR session {futs[fut]}: {e}", flush=True)
            if i % 100 == 0 or i < 5:
                rate = i / (time.time() - t0) if time.time() > t0 else 0
                eta = (len(sessions) - i) / rate / 60 if rate else 0
                print(f"  [{i}/{len(sessions)}]  elapsed={time.time()-t0:.0f}s  "
                      f"rate={rate:.2f}/s  ETA={eta:.1f}min", flush=True)

    print(f"\nMerging shards...", flush=True)
    idx_all = pd.concat(idx_shards, ignore_index=True).sort_values(
        ["session_date", "bar_open_time"]).reset_index(drop=True)
    buckets_all = pd.concat(bucket_shards, ignore_index=True).sort_values(
        ["session_date", "bar_open_time", "wick_side", "bucket_price"]).reset_index(drop=True)

    idx_all.to_parquet(out_idx, compression="snappy", index=False)
    buckets_all.to_parquet(out_buckets, compression="snappy", index=False)

    print(f"\nWrote:")
    print(f"  {out_idx.name}:    {out_idx.stat().st_size/1e6:.1f} MB  ({len(idx_all):,} rows)")
    print(f"  {out_buckets.name}: {out_buckets.stat().st_size/1e6:.1f} MB  ({len(buckets_all):,} rows)")
    print(f"Total elapsed: {time.time()-t0:.0f}s  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, required=True)
    args = ap.parse_args()
    main(args.minutes)
