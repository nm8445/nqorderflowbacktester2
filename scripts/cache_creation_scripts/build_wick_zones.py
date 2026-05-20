"""
Build wick zones (support + resistance) and lifecycle events from the 5-min
wick-delta index.

Zone formation run rule:
  - run starts on a bar with qualifying wick absorption (support run: bullish OR
    bearish close with bot_wick_delta <= -MIN_WICK_DELTA; resistance: top wick
    with top_wick_delta >= +MIN_WICK_DELTA)
  - next bar extends the run ONLY if it also qualifies AND its wick RANGE
    overlaps the running zone range
  - run ends the first time either condition fails
  - zone "locks" when run ends, IF run length >= MIN_RUN AND:
      support: at least one bullish-close bar in the run
                AND no body close below zone_low during the run
      resistance: at least one bearish-close bar
                AND no body close above zone_high during the run

Zone lifecycle tracked forward from formation:
  - any subsequent bar whose [low, high] overlaps [zone_low, zone_high]  -> touch event
  - any subsequent bar whose body (close) crosses through zone boundary -> invalidation
    (eth_break if the bar is in ETH, rth_break if RTH)
  - 14 trading days since last touch (clock resets on each touch) -> staleness death

Output:
  wick_zones_5min{suffix}.parquet   (zone lifecycle, one row per zone)
  wick_zone_touches_5min{suffix}.parquet (touch events, one row per touch)

Usage:
  python build_wick_zones.py --min-wick-delta 80 --min-run 2 --stale-days 14
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("D:/trading_pythonbacktest_data")
ET = "America/New_York"


def build_zones(idx: pd.DataFrame, min_wick_delta: int, min_run: int,
                displacement_bars: int = 0, displacement_atr: float = 0.0):
    """Walk bars chronologically, emit finalized zones + touches.

    If displacement_bars > 0 and displacement_atr > 0, a candidate zone is
    only kept if within `displacement_bars` bars after the run ends, at least
    one bar closes outside the zone by `displacement_atr * ATR(14)` in the
    favorable direction (above zone_high for support; below zone_low for
    resistance). This filters out stalling clusters where price hasn't
    actually moved away from the zone yet.
    """
    idx = idx.sort_values("bar_open_time").reset_index(drop=True)

    # Precompute arrays for speed
    n = len(idx)
    bot_delta = idx["bot_wick_delta"].to_numpy()
    top_delta = idx["top_wick_delta"].to_numpy()
    bot_low = idx["bot_wick_low"].to_numpy()
    bot_top = idx["bot_wick_top"].to_numpy()
    top_low = idx["top_wick_low"].to_numpy()
    top_high = idx["top_wick_high"].to_numpy()
    open_ = idx["open"].to_numpy()
    high_ = idx["high"].to_numpy()
    low_ = idx["low"].to_numpy()
    close_ = idx["close"].to_numpy()
    is_bull = idx["is_bullish"].to_numpy()
    is_bear = idx["is_bearish"].to_numpy()
    ts = idx["bar_open_time"].to_numpy()

    # ATR(14) for displacement check
    atr_arr = np.full(n, np.nan)
    if displacement_bars > 0 and displacement_atr > 0:
        prev_close = idx["close"].shift(1).to_numpy()
        tr = np.maximum.reduce([
            np.where(np.isnan(idx["high"]) | np.isnan(idx["low"]), np.nan,
                     idx["high"] - idx["low"]),
            np.abs(idx["high"].to_numpy() - prev_close),
            np.abs(idx["low"].to_numpy() - prev_close),
        ])
        atr_arr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()

    def displacement_ok(zone_low, zone_high, last_run_idx, ztype):
        """Is there displacement after the run ends?"""
        if displacement_bars <= 0 or displacement_atr <= 0:
            return True
        atr_val = atr_arr[last_run_idx]
        if np.isnan(atr_val):
            return False
        threshold = displacement_atr * atr_val
        # Check the next `displacement_bars` bars (after last_run_idx)
        for k in range(1, displacement_bars + 1):
            j = last_run_idx + k
            if j >= n:
                break
            c = close_[j]
            if np.isnan(c):
                continue
            if ztype == "support" and c >= zone_high + threshold:
                return True
            if ztype == "resistance" and c <= zone_low - threshold:
                return True
        return False

    zones = []

    # ── pass 1: detect formation runs for SUPPORT zones ────────────────────
    # A bar qualifies for support run if bot_wick_delta <= -min_wick_delta
    run_start = -1
    run_low = run_high = None
    run_has_bullish = False
    run_bar_ids = []
    for i in range(n):
        # Does bar i qualify?
        qualifies = bot_delta[i] <= -min_wick_delta and not np.isnan(bot_low[i])
        if not qualifies:
            # Close out any open run
            if (run_start >= 0 and len(run_bar_ids) >= min_run and run_has_bullish
                and displacement_ok(run_low, run_high, run_bar_ids[-1], "support")):
                zones.append({
                    "zone_type": "support",
                    "formed_at": ts[run_bar_ids[-1]],
                    "zone_low": run_low, "zone_high": run_high,
                    "run_bars": len(run_bar_ids),
                    "run_start_time": ts[run_bar_ids[0]],
                })
            run_start = -1; run_low = run_high = None
            run_has_bullish = False; run_bar_ids = []
            continue
        wick_low_i = bot_low[i]; wick_top_i = bot_top[i]
        if run_start < 0:
            # Start new run
            run_start = i
            run_low = wick_low_i; run_high = wick_top_i
            run_has_bullish = bool(is_bull[i])
            run_bar_ids = [i]
        else:
            # Extend only if wick overlaps current zone range
            # overlap condition: wick_top_i >= run_low AND wick_low_i <= run_high
            if wick_top_i >= run_low and wick_low_i <= run_high:
                run_low = min(run_low, wick_low_i)
                run_high = max(run_high, wick_top_i)
                run_has_bullish = run_has_bullish or bool(is_bull[i])
                run_bar_ids.append(i)
            else:
                # Close previous run, start fresh on this bar
                if (len(run_bar_ids) >= min_run and run_has_bullish
                    and displacement_ok(run_low, run_high, run_bar_ids[-1], "support")):
                    zones.append({
                        "zone_type": "support",
                        "formed_at": ts[run_bar_ids[-1]],
                        "zone_low": run_low, "zone_high": run_high,
                        "run_bars": len(run_bar_ids),
                        "run_start_time": ts[run_bar_ids[0]],
                    })
                run_start = i
                run_low = wick_low_i; run_high = wick_top_i
                run_has_bullish = bool(is_bull[i])
                run_bar_ids = [i]
        # Check run invalidation: body close below run_low during formation
        if not np.isnan(close_[i]) and close_[i] < run_low:
            # Run is tainted; discard
            run_start = -1; run_low = run_high = None
            run_has_bullish = False; run_bar_ids = []
    # Flush last run
    if (run_start >= 0 and len(run_bar_ids) >= min_run and run_has_bullish
        and displacement_ok(run_low, run_high, run_bar_ids[-1], "support")):
        zones.append({
            "zone_type": "support",
            "formed_at": ts[run_bar_ids[-1]],
            "zone_low": run_low, "zone_high": run_high,
            "run_bars": len(run_bar_ids),
            "run_start_time": ts[run_bar_ids[0]],
        })

    # ── pass 2: detect formation runs for RESISTANCE zones ─────────────────
    run_start = -1; run_low = run_high = None
    run_has_bearish = False; run_bar_ids = []
    for i in range(n):
        qualifies = top_delta[i] >= min_wick_delta and not np.isnan(top_low[i])
        if not qualifies:
            if (run_start >= 0 and len(run_bar_ids) >= min_run and run_has_bearish
                and displacement_ok(run_low, run_high, run_bar_ids[-1], "resistance")):
                zones.append({
                    "zone_type": "resistance",
                    "formed_at": ts[run_bar_ids[-1]],
                    "zone_low": run_low, "zone_high": run_high,
                    "run_bars": len(run_bar_ids),
                    "run_start_time": ts[run_bar_ids[0]],
                })
            run_start = -1; run_low = run_high = None
            run_has_bearish = False; run_bar_ids = []
            continue
        wick_low_i = top_low[i]; wick_high_i = top_high[i]
        if run_start < 0:
            run_start = i
            run_low = wick_low_i; run_high = wick_high_i
            run_has_bearish = bool(is_bear[i])
            run_bar_ids = [i]
        else:
            if wick_high_i >= run_low and wick_low_i <= run_high:
                run_low = min(run_low, wick_low_i)
                run_high = max(run_high, wick_high_i)
                run_has_bearish = run_has_bearish or bool(is_bear[i])
                run_bar_ids.append(i)
            else:
                if (len(run_bar_ids) >= min_run and run_has_bearish
                    and displacement_ok(run_low, run_high, run_bar_ids[-1], "resistance")):
                    zones.append({
                        "zone_type": "resistance",
                        "formed_at": ts[run_bar_ids[-1]],
                        "zone_low": run_low, "zone_high": run_high,
                        "run_bars": len(run_bar_ids),
                        "run_start_time": ts[run_bar_ids[0]],
                    })
                run_start = i
                run_low = wick_low_i; run_high = wick_high_i
                run_has_bearish = bool(is_bear[i])
                run_bar_ids = [i]
        if not np.isnan(close_[i]) and close_[i] > run_high:
            run_start = -1; run_low = run_high = None
            run_has_bearish = False; run_bar_ids = []
    if (run_start >= 0 and len(run_bar_ids) >= min_run and run_has_bearish
        and displacement_ok(run_low, run_high, run_bar_ids[-1], "resistance")):
        zones.append({
            "zone_type": "resistance",
            "formed_at": ts[run_bar_ids[-1]],
            "zone_low": run_low, "zone_high": run_high,
            "run_bars": len(run_bar_ids),
            "run_start_time": ts[run_bar_ids[0]],
        })

    zones_df = pd.DataFrame(zones)
    if zones_df.empty:
        return zones_df, pd.DataFrame()
    zones_df = zones_df.sort_values("formed_at").reset_index(drop=True)
    zones_df["zone_id"] = range(len(zones_df))
    return zones_df, idx


def track_lifecycle(zones_df: pd.DataFrame, idx: pd.DataFrame,
                    stale_days: int, is_rth_map: dict):
    """For each zone, scan forward and record touches + invalidation."""
    if zones_df.empty:
        return zones_df, pd.DataFrame()

    idx = idx.sort_values("bar_open_time").reset_index(drop=True)
    ts = idx["bar_open_time"].to_numpy()
    high_ = idx["high"].to_numpy()
    low_ = idx["low"].to_numpy()
    close_ = idx["close"].to_numpy()
    sess = idx["session_date"].to_numpy()

    # Index from timestamp -> row index for fast lookup
    ts_series = pd.Series(np.arange(len(idx)), index=pd.to_datetime(idx["bar_open_time"]))

    # Unique trading sessions sorted
    unique_sessions = sorted(pd.unique(sess))
    sess_to_ord = {s: i for i, s in enumerate(unique_sessions)}

    touches_all = []
    inv_times = []
    inv_reasons = []
    last_touch_times = []

    for _, z in zones_df.iterrows():
        form_ts = pd.Timestamp(z["formed_at"])
        form_pos = ts_series.get(form_ts)
        if form_pos is None: form_pos = ts_series.index.searchsorted(form_ts)
        start_pos = form_pos + 1
        zlow = z["zone_low"]; zhigh = z["zone_high"]
        ztype = z["zone_type"]

        zone_touches = []
        last_touch_ord = sess_to_ord.get(str(pd.Timestamp(form_ts).date()),
                                           sess_to_ord.get(z.get("formation_session", None), 0))
        inv_t = None; inv_r = None

        for j in range(start_pos, len(idx)):
            h = high_[j]; l = low_[j]; c = close_[j]
            if np.isnan(h) or np.isnan(l) or np.isnan(c):
                continue
            # Touch: bar's [low, high] overlaps zone
            if h >= zlow and l <= zhigh:
                zone_touches.append({
                    "zone_id": int(z["zone_id"]),
                    "touch_time": ts[j],
                    "touch_low": l, "touch_high": h, "touch_close": c,
                    "is_rth": is_rth_map.get(pd.Timestamp(ts[j]).tz_convert(ET).strftime("%H:%M"), False),
                })
                last_touch_ord = sess_to_ord.get(str(sess[j]), last_touch_ord)

            # Invalidation: body close through zone (close < zone_low for support, > zone_high for resistance)
            if ztype == "support" and c < zlow:
                # ETH vs RTH
                hm = pd.Timestamp(ts[j]).tz_convert(ET).strftime("%H:%M")
                inv_t = ts[j]
                inv_r = "rth_break" if is_rth_map.get(hm, False) else "eth_break"
                break
            if ztype == "resistance" and c > zhigh:
                hm = pd.Timestamp(ts[j]).tz_convert(ET).strftime("%H:%M")
                inv_t = ts[j]
                inv_r = "rth_break" if is_rth_map.get(hm, False) else "eth_break"
                break

            # Staleness: N trading days since last touch
            cur_sess = str(sess[j])
            cur_ord = sess_to_ord.get(cur_sess)
            if cur_ord is not None and last_touch_ord is not None:
                if cur_ord - last_touch_ord > stale_days:
                    inv_t = ts[j]
                    inv_r = "staleness"
                    break

        touches_all.extend(zone_touches)
        if zone_touches:
            last_touch_times.append(zone_touches[-1]["touch_time"])
        else:
            last_touch_times.append(None)
        inv_times.append(inv_t); inv_reasons.append(inv_r)

    zones_df = zones_df.copy()
    zones_df["last_touch_time"] = last_touch_times
    zones_df["invalidation_time"] = inv_times
    zones_df["invalidation_reason"] = inv_reasons
    zones_df["is_active_at_eod"] = zones_df["invalidation_time"].isna()

    touches_df = pd.DataFrame(touches_all)
    return zones_df, touches_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-wick-delta", type=int, default=80)
    ap.add_argument("--min-run", type=int, default=2)
    ap.add_argument("--stale-days", type=int, default=14)
    ap.add_argument("--displacement-bars", type=int, default=0,
                    help="If >0, require a bar within this many bars after run "
                         "end to close >= displacement_atr*ATR(14) outside zone.")
    ap.add_argument("--displacement-atr", type=float, default=0.0,
                    help="ATR multiple for displacement filter (e.g. 0.5).")
    ap.add_argument("--suffix", type=str, default="")
    args = ap.parse_args()

    t0 = time.time()
    print(f"Loading wick delta index (5min)...")
    idx = pd.read_parquet(DATA_DIR / "wick_delta_index_5min.parquet")
    idx["bar_open_time"] = pd.to_datetime(idx["bar_open_time"])
    if idx["bar_open_time"].dt.tz is None:
        idx["bar_open_time"] = idx["bar_open_time"].dt.tz_localize("UTC").dt.tz_convert(ET)
    else:
        idx["bar_open_time"] = idx["bar_open_time"].dt.tz_convert(ET)

    # RTH hour-minute map (string -> bool). A bar is RTH if its hm in [09:30, 16:59].
    hm_all = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(60)]
    is_rth_map = {hm: ("09:30" <= hm <= "16:59") for hm in hm_all}

    print(f"  rows: {len(idx):,}")
    print(f"Detecting zone formations (min_wick_delta={args.min_wick_delta}, "
          f"min_run={args.min_run}, displacement_bars={args.displacement_bars}, "
          f"displacement_atr={args.displacement_atr})...")
    zones_df, idx_sorted = build_zones(idx, args.min_wick_delta, args.min_run,
                                        args.displacement_bars, args.displacement_atr)
    print(f"  zones detected: {len(zones_df):,}")
    if not zones_df.empty:
        print(f"    support:    {(zones_df['zone_type']=='support').sum():,}")
        print(f"    resistance: {(zones_df['zone_type']=='resistance').sum():,}")

    print(f"Tracking zone lifecycle (stale_days={args.stale_days})...")
    zones_df, touches_df = track_lifecycle(zones_df, idx_sorted, args.stale_days, is_rth_map)

    if not zones_df.empty:
        # Invalidation reason counts
        rc = zones_df["invalidation_reason"].fillna("still_active").value_counts()
        print("  invalidation reasons:")
        for r, c in rc.items():
            print(f"    {r}: {c:,}")

    suffix = args.suffix if args.suffix else ""
    out_zones = DATA_DIR / f"wick_zones_5min{suffix}.parquet"
    out_touches = DATA_DIR / f"wick_zone_touches_5min{suffix}.parquet"
    zones_df.to_parquet(out_zones, compression="snappy", index=False)
    touches_df.to_parquet(out_touches, compression="snappy", index=False)
    print(f"\nWrote {out_zones.name}    ({out_zones.stat().st_size/1e6:.1f} MB, {len(zones_df):,} zones)")
    print(f"      {out_touches.name}  ({out_touches.stat().st_size/1e6:.1f} MB, {len(touches_df):,} touches)")
    print(f"Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
