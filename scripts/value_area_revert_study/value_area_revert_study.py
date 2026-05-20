"""Value-area mean-reversion study (RTH-only VA from cached profile snapshots).

Theory:
  If today's premarket-last-print opens ABOVE prev-day RTH VAH, today's RTH
  session has elevated probability of trading DOWN to touch prev-day VAH.
  Symmetric for opens BELOW prev-day VAL.

Distance metric:
  distance_pct = (open - vah_prev) / (vah_prev - val_prev)   [if above VAH]
  distance_pct = (val_prev - open) / (vah_prev - val_prev)   [if below VAL]

Prev-day RTH VA is computed from the cached profile snapshots:
  D:/trading_pythonbacktest_data/cache/profiles/{date}_refresh_minutes=1.pkl
  Each pickle is a list of per-minute snapshots; each snapshot has a
  VolumeProfile object with cumulative-through-session per-level buy/sell
  volume. RTH-only volume = levels at 16:00 ET snapshot MINUS levels at
  09:30 ET snapshot, then VA at 68% around the RTH POC.

Today's "open" = close of last 1-min bar with timestamp < 09:30 ET.
"Return" = any 5-min RTH bar's LOW <= vah_prev (if open above) or
           HIGH >= val_prev (if open below) during today's 9:30-16:00 ET.
"""
from __future__ import annotations

import datetime as dt
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

VOL5M     = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
M1_BARS   = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
PROFILES  = Path("D:/trading_pythonbacktest_data/cache/profiles")
OUT_TXT   = Path(__file__).parent.parent / "overnight range strat" / "tradelogs" / "robust_configs" / "value_area_revert_study.txt"
CACHE_VA  = Path(__file__).parent / "prev_day_rth_va.parquet"
ET        = "America/New_York"

RTH_START = dt.time(9, 30)
RTH_END   = dt.time(16, 0)
VA_PCT    = 0.68                       # matches the cached profiles' setting
MIN_VA_WIDTH = 20.0
MIN_RTH_BARS = 30
DISTANCE_BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 35),
                    (35, 50), (50, 75), (75, 100), (100, 9999)]
BUCKET_LABELS = ["0-5%", "5-10%", "10-20%", "20-35%",
                 "35-50%", "50-75%", "75-100%", "100%+"]


# ---------- VA via snapshot subtraction ----------

def _levels_dict(profile_obj) -> dict[float, float]:
    """Extract {price: buy_vol+sell_vol} from a VolumeProfile snapshot."""
    out = {}
    levels = getattr(profile_obj, "levels", None)
    if not levels:
        return out
    for price, plvl in levels.items():
        out[float(price)] = float(getattr(plvl, "buy_vol", 0)) + float(getattr(plvl, "sell_vol", 0))
    return out


def _find_snapshot_at(snapshots, target_et: pd.Timestamp):
    """Find the snapshot whose refresh_time is the latest <= target_et."""
    best = None
    for s in snapshots:
        ts = pd.Timestamp(s["refresh_time"])
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        ts_et = ts.tz_convert(ET)
        if ts_et <= target_et:
            if best is None or ts_et > pd.Timestamp(best["refresh_time"]).tz_convert(ET):
                best = s
        else:
            break  # snapshots are time-ordered (build sequence)
    return best


def compute_rth_va_from_profile(pkl_path: Path, session_date: dt.date,
                                  va_pct: float = VA_PCT
                                  ) -> tuple[float, float, float, float]:
    """Return (vah, val, poc, total_rth_vol). NaN if unable."""
    if not pkl_path.exists():
        return (np.nan, np.nan, np.nan, 0)
    try:
        with open(pkl_path, "rb") as f:
            snaps = pickle.load(f)
    except Exception:
        return (np.nan, np.nan, np.nan, 0)
    if not snaps:
        return (np.nan, np.nan, np.nan, 0)

    # Targets: 09:30 ET and 16:00 ET on session_date
    open_target  = pd.Timestamp(f"{session_date} {RTH_START}", tz=ET)
    close_target = pd.Timestamp(f"{session_date} {RTH_END}",   tz=ET)
    snap_open  = _find_snapshot_at(snaps, open_target)
    snap_close = _find_snapshot_at(snaps, close_target)
    if snap_open is None or snap_close is None:
        return (np.nan, np.nan, np.nan, 0)
    # If the "open" snapshot equals the "close" snapshot, no RTH volume captured
    if pd.Timestamp(snap_open["refresh_time"]).tz_localize(None) >= pd.Timestamp(snap_close["refresh_time"]).tz_localize(None):
        return (np.nan, np.nan, np.nan, 0)

    lvls_open  = _levels_dict(snap_open["profile"])
    lvls_close = _levels_dict(snap_close["profile"])
    # RTH-only per-level vol = close - open
    rth = {}
    for p, v in lvls_close.items():
        diff = v - lvls_open.get(p, 0.0)
        if diff > 0:
            rth[p] = diff
    if not rth:
        return (np.nan, np.nan, np.nan, 0)

    # Compute VA at va_pct around POC
    prices = np.array(sorted(rth.keys()))
    vols   = np.array([rth[p] for p in prices], dtype=float)
    total  = float(vols.sum())
    if total <= 0:
        return (np.nan, np.nan, np.nan, 0)
    poc_idx = int(np.argmax(vols))
    poc_price = float(prices[poc_idx])
    target_vol = total * va_pct
    lo = hi = poc_idx
    accum = float(vols[poc_idx])
    while accum < target_vol and (lo > 0 or hi < len(prices) - 1):
        up   = float(vols[hi + 1]) if hi < len(prices) - 1 else -1.0
        down = float(vols[lo - 1]) if lo > 0 else -1.0
        if up < 0 and down < 0:
            break
        if up >= down:
            hi += 1; accum += up
        else:
            lo -= 1; accum += down
    return (float(prices[hi]), float(prices[lo]), poc_price, int(total))


def build_rth_va_table():
    if CACHE_VA.exists():
        print(f"loading cached RTH VA table from {CACHE_VA}...")
        return pd.read_parquet(CACHE_VA)
    print(f"computing RTH-only VA per session day from {PROFILES} ...")
    rows = []
    pkls = sorted(PROFILES.glob("*_refresh_minutes=1.pkl"))
    print(f"  {len(pkls)} profile files")
    for i, p in enumerate(pkls):
        # date is everything before the first underscore
        date_str = p.name.split("_refresh_minutes")[0]
        try:
            d = dt.date.fromisoformat(date_str)
        except ValueError:
            continue
        vah, val, poc, total = compute_rth_va_from_profile(p, d)
        rows.append({"date": d, "vah": vah, "val": val, "poc": poc,
                     "va_width": vah - val if np.isfinite(vah) and np.isfinite(val) else np.nan,
                     "total_rth_vol": total})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(pkls)} done")
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df.to_parquet(CACHE_VA)
    print(f"saved {CACHE_VA}")
    return df


# ---------- premarket open ----------

def build_premarket_open():
    print("loading 1-min bars for premarket-last-print ...")
    df = pd.read_parquet(M1_BARS, columns=["close"])
    idx = pd.to_datetime(df.index, utc=True).tz_convert(ET)
    df = df.set_index(idx).sort_index()
    df["date"] = df.index.date
    df["t"]    = df.index.time
    pre = df[df["t"] < RTH_START].copy()
    last_pre = pre.groupby("date").tail(1)
    out = last_pre[["date", "close"]].rename(columns={"close": "open_price"})
    print(f"  {len(out)} dates with premarket-last-print")
    return out


# ---------- RTH 5-min bars for return-walk ----------

def build_rth_bars():
    print("loading 5-min bars for RTH-walk ...")
    df = pd.read_parquet(VOL5M, columns=["session_date", "bar_open_time",
                                          "open", "high", "low", "close"])
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
    bot = pd.to_datetime(df["bar_open_time"], utc=True).dt.tz_convert(ET)
    rth_mask = (bot.dt.time >= RTH_START) & (bot.dt.time < RTH_END)
    df = df[rth_mask].copy()
    df = df.drop_duplicates(subset=["session_date", "bar_open_time"]).reset_index(drop=True)
    df["bar_open_time"] = bot[rth_mask].values[:len(df)]
    print(f"  {len(df):,} unique RTH bar rows")
    return df


# ---------- bucketing & stats ----------

def bucket_pct(p: float) -> str:
    for (lo, hi), lab in zip(DISTANCE_BUCKETS, BUCKET_LABELS):
        if lo <= p < hi:
            return lab
    return BUCKET_LABELS[-1]


def main():
    va = build_rth_va_table()
    pre = build_premarket_open()
    rth = build_rth_bars()

    # prev-day lookup
    dates_with_va = va.dropna(subset=["vah","val"]).sort_values("date")["date"].tolist()
    prev_lookup = {dates_with_va[i]: dates_with_va[i-1] for i in range(1, len(dates_with_va))}

    va_idx  = va.set_index("date")
    pre_idx = pre.set_index("date")

    rth_by_day = {d: g.sort_values("bar_open_time").reset_index(drop=True)
                  for d, g in rth.groupby("session_date")}

    print("joining today's open + prev-day VA + walking RTH bars ...")
    rows = []
    for today in sorted(rth_by_day.keys()):
        prev_d = prev_lookup.get(today)
        if prev_d is None or prev_d not in va_idx.index:
            continue
        vah_prev = float(va_idx.loc[prev_d, "vah"])
        val_prev = float(va_idx.loc[prev_d, "val"])
        if not (np.isfinite(vah_prev) and np.isfinite(val_prev)):
            continue
        width_prev = vah_prev - val_prev
        if width_prev < MIN_VA_WIDTH:
            continue
        if today not in pre_idx.index:
            continue
        open_price = float(pre_idx.loc[today, "open_price"])
        if not np.isfinite(open_price):
            continue
        bars = rth_by_day[today]
        if len(bars) < MIN_RTH_BARS:
            continue

        if open_price > vah_prev:
            direction = "ABOVE_VAH"
            distance  = open_price - vah_prev
            pct       = distance / width_prev * 100
            target    = vah_prev
            touch_mask = bars["low"].values <= target
            mfe = open_price - float(bars["low"].min())
            mae = float(bars["high"].max()) - open_price
        elif open_price < val_prev:
            direction = "BELOW_VAL"
            distance  = val_prev - open_price
            pct       = distance / width_prev * 100
            target    = val_prev
            touch_mask = bars["high"].values >= target
            mfe = float(bars["high"].max()) - open_price
            mae = open_price - float(bars["low"].min())
        else:
            continue

        if touch_mask.any():
            touched = True
            first_idx = int(np.argmax(touch_mask))
            minutes_to_touch = (first_idx + 1) * 5
        else:
            touched = False
            minutes_to_touch = None

        rows.append({
            "date": today, "direction": direction,
            "open_price": open_price, "vah_prev": vah_prev, "val_prev": val_prev,
            "width_prev": width_prev, "distance_pts": distance, "distance_pct": pct,
            "touched": touched, "minutes_to_touch": minutes_to_touch,
            "mfe_pts": mfe, "mae_pts": mae,
        })
    df = pd.DataFrame(rows)
    print(f"  {len(df)} qualifying days  "
          f"(above={len(df[df['direction']=='ABOVE_VAH'])}, below={len(df[df['direction']=='BELOW_VAL'])})")

    df["bucket"] = df["distance_pct"].apply(bucket_pct)
    df["period"] = np.where(df["date"] < dt.date(2025, 1, 1), "IS", "OOS")

    L = []
    L.append("=" * 200)
    L.append("PREV-DAY RTH VA MEAN-REVERSION STUDY")
    L.append("=" * 200)
    L.append(f"Premarket-last-print as today's 'open'   |   RTH window = {RTH_START}-{RTH_END} ET")
    L.append(f"VA: {int(VA_PCT*100)}% volume area, RTH-only (snapshot subtraction from cached profiles)")
    L.append(f"Sample: {df['date'].min()} -> {df['date'].max()}")
    L.append(f"Qualifying days: {len(df)}   "
             f"ABOVE_VAH={len(df[df['direction']=='ABOVE_VAH'])}   "
             f"BELOW_VAL={len(df[df['direction']=='BELOW_VAL'])}")
    L.append(f"Filters: VA width >= {MIN_VA_WIDTH} pts, RTH bars >= {MIN_RTH_BARS}")
    L.append("")

    def per_bucket(sub, label):
        out = (sub.groupby("bucket", observed=True)
                .agg(n=("touched","count"),
                     ret_pct=("touched", lambda s: s.mean()*100),
                     median_min=("minutes_to_touch", lambda s: s.dropna().median()),
                     median_dist_pct=("distance_pct","median"),
                     median_dist_pts=("distance_pts","median"),
                     median_mfe=("mfe_pts","median"),
                     median_mae=("mae_pts","median"))
                .reindex(BUCKET_LABELS).fillna(0))
        L.append("")
        L.append(f"--- {label} ---")
        L.append(out.to_string(float_format=lambda x: f"{x:.2f}"))

    L.append("=" * 200)
    L.append("ABOVE_VAH (open above prev VAH -- expect drop to VAH)")
    L.append("=" * 200)
    above = df[df["direction"] == "ABOVE_VAH"]
    per_bucket(above, "ALL")
    per_bucket(above[above["period"]=="IS"],  "IS (Dec 2020 -> Dec 2024)")
    per_bucket(above[above["period"]=="OOS"], "OOS (2025+)")

    L.append("")
    L.append("=" * 200)
    L.append("BELOW_VAL (open below prev VAL -- expect rise to VAL)")
    L.append("=" * 200)
    below = df[df["direction"] == "BELOW_VAL"]
    per_bucket(below, "ALL")
    per_bucket(below[below["period"]=="IS"],  "IS")
    per_bucket(below[below["period"]=="OOS"], "OOS")

    L.append("")
    L.append("=" * 200)
    L.append("COMBINED (above + below mirrored) -- return rate per distance bucket")
    L.append("=" * 200)
    per_bucket(df, "ALL combined")
    per_bucket(df[df["period"]=="IS"],  "IS combined")
    per_bucket(df[df["period"]=="OOS"], "OOS combined")

    L.append("")
    L.append("=" * 200)
    L.append("NO-RETURN DAYS -- how far past the line did price keep going?")
    L.append("=" * 200)
    no_ret = df[~df["touched"]].copy()
    if not no_ret.empty:
        nr_buck = (no_ret.groupby(["direction","bucket"], observed=True)
                       .agg(n=("touched","count"),
                            median_mae=("mae_pts","median"),
                            p75_mae=("mae_pts", lambda s: s.quantile(0.75)),
                            max_mae=("mae_pts","max"))
                       .round(2))
        L.append(nr_buck.to_string())

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
