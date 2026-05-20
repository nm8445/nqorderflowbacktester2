"""
Emit 8-type signal events from a wick delta index.

Signal types (long / short mirror pairs):
  long_1bar_abs:         single bullish bar with bot_wick_delta <= -MIN
  long_2bar_N1wick:      bar N-1 bearish + bar N-1 bot_wick_delta <= -MIN,
                         bar N bullish + bar_total_delta >= STRONG
  long_2bar_Nwick:       bar N-1 bearish, bar N bullish + bot_wick_delta <= -MIN
  long_2bar_bothwicks:   bar N-1 bearish + bot_wick_delta <= -MIN,
                         bar N bullish + bot_wick_delta <= -MIN

Short mirror: S_1bar_abs, S_2bar_N1wick, S_2bar_Nwick, S_2bar_bothwicks
  (same logic on top wick, positive delta threshold, inverse bullish/bearish)

Entry: on the signal bar's close (bar N).
Params:
  --min-wick-delta  (default 80)  threshold |wick delta| for absorption
  --strong-delta    (default 200) threshold for confirmation bar total delta (2-bar N1wick only)
  --timeframe       (default 5)   minutes; must match a built wick index
  --suffix          (default "")  output filename suffix

Output: signals_{tf}min{suffix}.parquet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("D:/trading_pythonbacktest_data")
ET = "America/New_York"


def emit_signals(idx: pd.DataFrame, min_wick: int, strong: int):
    idx = idx.sort_values(["session_date", "bar_open_time"]).reset_index(drop=True)

    # Previous-bar columns (shift within session)
    for col in ["bar_open_time", "is_bullish", "is_bearish", "bot_wick_delta",
                "top_wick_delta", "bar_total_delta", "close"]:
        idx[f"prev_{col}"] = idx.groupby("session_date")[col].shift(1)

    is_bull = idx["is_bullish"].to_numpy()
    is_bear = idx["is_bearish"].to_numpy()
    bot_d = idx["bot_wick_delta"].to_numpy()
    top_d = idx["top_wick_delta"].to_numpy()
    tot_d = idx["bar_total_delta"].to_numpy()

    prev_is_bull = idx["prev_is_bullish"].to_numpy()
    prev_is_bear = idx["prev_is_bearish"].to_numpy()
    prev_bot_d = idx["prev_bot_wick_delta"].to_numpy()
    prev_top_d = idx["prev_top_wick_delta"].to_numpy()

    rows = []

    def emit(mask, signal_type, direction):
        ids = np.where(mask)[0]
        for i in ids:
            rows.append({
                "session_date": idx.iloc[i]["session_date"],
                "signal_time": idx.iloc[i]["bar_open_time"],
                "signal_type": signal_type,
                "direction": direction,
                "entry_price": idx.iloc[i]["close"],
                "setup_bar_time": idx.iloc[i]["prev_bar_open_time"]
                                    if signal_type.startswith(("long_2bar", "short_2bar"))
                                    else idx.iloc[i]["bar_open_time"],
                "signal_bar_total_delta": tot_d[i],
                "signal_bar_bot_wick_delta": bot_d[i],
                "signal_bar_top_wick_delta": top_d[i],
                "setup_bar_bot_wick_delta": prev_bot_d[i] if signal_type.startswith(("long_2bar","short_2bar")) else np.nan,
                "setup_bar_top_wick_delta": prev_top_d[i] if signal_type.startswith(("long_2bar","short_2bar")) else np.nan,
            })

    # ── LONG signals ─────────────────────────────────────────────────────────
    mask_L1 = is_bull & (bot_d <= -min_wick)
    emit(mask_L1, "long_1bar_abs", "long")

    mask_L2_N1 = (prev_is_bear.astype(bool)) & (prev_bot_d <= -min_wick) \
                 & is_bull & (tot_d >= strong)
    emit(mask_L2_N1, "long_2bar_N1wick", "long")

    mask_L2_N = (prev_is_bear.astype(bool)) & is_bull & (bot_d <= -min_wick)
    emit(mask_L2_N, "long_2bar_Nwick", "long")

    mask_L2_both = (prev_is_bear.astype(bool)) & (prev_bot_d <= -min_wick) \
                   & is_bull & (bot_d <= -min_wick)
    emit(mask_L2_both, "long_2bar_bothwicks", "long")

    # ── SHORT signals ────────────────────────────────────────────────────────
    mask_S1 = is_bear & (top_d >= min_wick)
    emit(mask_S1, "short_1bar_abs", "short")

    mask_S2_N1 = (prev_is_bull.astype(bool)) & (prev_top_d >= min_wick) \
                 & is_bear & (tot_d <= -strong)
    emit(mask_S2_N1, "short_2bar_N1wick", "short")

    mask_S2_N = (prev_is_bull.astype(bool)) & is_bear & (top_d >= min_wick)
    emit(mask_S2_N, "short_2bar_Nwick", "short")

    mask_S2_both = (prev_is_bull.astype(bool)) & (prev_top_d >= min_wick) \
                   & is_bear & (top_d >= min_wick)
    emit(mask_S2_both, "short_2bar_bothwicks", "short")

    sig_df = pd.DataFrame(rows)
    if sig_df.empty:
        return sig_df
    sig_df = sig_df.sort_values(["signal_time", "signal_type"]).reset_index(drop=True)
    return sig_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-wick-delta", type=int, default=80)
    ap.add_argument("--strong-delta", type=int, default=200)
    ap.add_argument("--timeframe", type=int, default=5)
    ap.add_argument("--suffix", type=str, default="")
    args = ap.parse_args()

    t0 = time.time()
    src = DATA_DIR / f"wick_delta_index_{args.timeframe}min.parquet"
    print(f"Loading {src.name}...")
    idx = pd.read_parquet(src)
    idx["bar_open_time"] = pd.to_datetime(idx["bar_open_time"])
    if idx["bar_open_time"].dt.tz is None:
        idx["bar_open_time"] = idx["bar_open_time"].dt.tz_localize("UTC").dt.tz_convert(ET)
    else:
        idx["bar_open_time"] = idx["bar_open_time"].dt.tz_convert(ET)
    print(f"  rows: {len(idx):,}")

    print(f"Emitting signals (min_wick={args.min_wick_delta}, strong={args.strong_delta})...")
    sig = emit_signals(idx, args.min_wick_delta, args.strong_delta)

    print(f"\nSignals by type:")
    if not sig.empty:
        for stype, c in sig["signal_type"].value_counts().items():
            print(f"  {stype:25s}  {c:>8,}")
    print(f"  total: {len(sig):,}")

    suffix = args.suffix if args.suffix else ""
    out = DATA_DIR / f"signals_{args.timeframe}min{suffix}.parquet"
    sig.to_parquet(out, compression="snappy", index=False)
    print(f"\nWrote {out.name}  ({out.stat().st_size/1e6:.1f} MB, {len(sig):,} rows)")
    print(f"Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
