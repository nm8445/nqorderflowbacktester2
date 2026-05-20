"""Overnight noise band study with gamma regime filter.

Bands:
  - Anchor = NQ price at 5:00 PM ET (yesterday's cash close)
  - Sigma = std of overnight returns (5pm prev day -> 9:30 ET today) over 14-day
    and 90-day lookbacks.  Computed as a single scalar per day per lookback.
  - upper_band = anchor * (1 + sigma)
  - lower_band = anchor * (1 - sigma)
  - Bands are STATIC for the entire RTH session (no expansion during the day).

Trigger:
  - At each 5-min bar close from 9:30 to 17:00 ET, check if close > upper or
    < lower. Record the FIRST break of each side and time of first break.
  - Day classification: above_only / below_only / both / inside

Outcome:
  - day_ret_pts = NQ_close_17:00 - NQ_open_9:30 (signed pts)
  - We test whether the day continues in the direction of the break, conditional
    on the gamma regime at 9:30 ET.

Cohorts: classification x gamma regime (QQQ + NDX)
Output:
  - per-day parquet at .../noise band studies/scripts/overnight_band_per_day.parquet
  - printed cohort tables with t-stats and p-values
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

NQ_1MIN     = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
GAMMA_PATH  = Path("D:/trading_pythonbacktest_data/QQQ_thetadata/study_hvl0dte_intraday_regime.parquet")
OUT_DIR     = Path(__file__).parent
PER_DAY_OUT = OUT_DIR / "overnight_band_per_day.parquet"

LOOKBACKS = [14, 90]   # both windows requested


def load_nq_et() -> pd.Series:
    df = pd.read_parquet(NQ_1MIN)
    idx = df.index
    if idx.tz is None: idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert("America/New_York")
    return df["close"].sort_index()


def at_time(nq: pd.Series, date: dt.date, h: int, m: int) -> float:
    target = pd.Timestamp(year=date.year, month=date.month, day=date.day,
                          hour=h, minute=m, tz="America/New_York")
    win = nq.loc[target - pd.Timedelta(minutes=2): target + pd.Timedelta(minutes=1)]
    return float(win.iloc[-1]) if not win.empty else np.nan


def build_per_day_table() -> pd.DataFrame:
    print("loading NQ 1-min bars (UTC -> ET)...")
    nq = load_nq_et()
    print(f"  range: {nq.index.min()} -> {nq.index.max()}")

    # All trading dates (Mon-Fri) within the data range
    all_dates = sorted(set(nq.index.date.tolist()))
    weekdays = [d for d in all_dates if d.weekday() < 5]
    print(f"weekdays in range: {len(weekdays)}")

    # 5pm prev day and 9:30 today for each
    print("computing per-day open/close/anchor...")
    rows = []
    for d in weekdays:
        anchor_5pm = at_time(nq, d, 16, 59)  # use 16:59 to grab the last RTH-session minute
        # The "yesterday's 5pm close" is actually today's 16:59 since the RTH closes there.
        # For the OVERNIGHT band we need to anchor to the close that PRECEDES today's open.
        # That means: for trading day D, anchor = 16:59 ET on the previous trading day.
        # We'll wire this up after we have all close prices.
        rth_open  = at_time(nq, d, 9, 30)
        rth_close = at_time(nq, d, 17, 0)
        rows.append({"date": d, "close_1659": anchor_5pm,
                     "open_930": rth_open, "close_1700": rth_close})
    base = pd.DataFrame(rows).dropna()
    print(f"  base rows: {len(base)}")

    # For each day D, anchor = previous day's close_1659
    base = base.sort_values("date").reset_index(drop=True)
    base["anchor"] = base["close_1659"].shift(1)
    base["overnight_ret"] = (base["open_930"] - base["anchor"]) / base["anchor"]

    # Compute rolling sigma for each lookback (uses prior overnight_ret, NOT today's)
    for lb in LOOKBACKS:
        # Rolling std on overnight_ret with lookback ending the day BEFORE today
        # i.e., today's sigma uses overnight returns from days [D-lb-1, D-1]
        sig = base["overnight_ret"].shift(1).rolling(lb).std()
        base[f"sigma_{lb}d"] = sig
        base[f"upper_band_{lb}d"] = base["anchor"] * (1.0 + sig)
        base[f"lower_band_{lb}d"] = base["anchor"] * (1.0 - sig)

    # Now scan 5-min RTH bars for band breaks
    print("scanning 5-min bars for band breaks...")
    # Resample close to 5-min within RTH only
    rth_mask = (nq.index.time >= dt.time(9, 30)) & (nq.index.time <= dt.time(17, 0))
    rth_nq = nq[rth_mask]
    # 5-min close = use last value within each 5-min bin
    rth_5m = rth_nq.resample("5min", origin="epoch").last().dropna()
    rth_5m = rth_5m[(rth_5m.index.time >= dt.time(9, 30)) &
                    (rth_5m.index.time <= dt.time(17, 0))]
    rth_5m_by_date = {d: g for d, g in rth_5m.groupby(rth_5m.index.date)}

    # For each day, classify break behavior for both lookbacks
    out_rows = []
    t0 = time.time()
    base = base.dropna(subset=["anchor","sigma_14d","sigma_90d"])
    for i, r in enumerate(base.itertuples(index=False), 1):
        d = r.date
        bars = rth_5m_by_date.get(d)
        if bars is None or len(bars) < 5:
            continue
        row_out = {
            "date": d,
            "anchor":      float(r.anchor),
            "open_930":    float(r.open_930),
            "close_1700":  float(r.close_1700),
            "day_ret_pts": float(r.close_1700 - r.open_930),
            "overnight_ret": float(r.overnight_ret),
            "sigma_14d":   float(r.sigma_14d),
            "sigma_90d":   float(r.sigma_90d),
            "upper_band_14d": float(r.upper_band_14d),
            "lower_band_14d": float(r.lower_band_14d),
            "upper_band_90d": float(r.upper_band_90d),
            "lower_band_90d": float(r.lower_band_90d),
        }
        # Detect first breaks for each lookback
        for lb in LOOKBACKS:
            up = r.upper_band_14d if lb == 14 else r.upper_band_90d
            lo = r.lower_band_14d if lb == 14 else r.lower_band_90d
            broke_up_idx = bars.index[bars > up]
            broke_dn_idx = bars.index[bars < lo]
            first_up = broke_up_idx[0] if len(broke_up_idx) else None
            first_dn = broke_dn_idx[0] if len(broke_dn_idx) else None
            if first_up and first_dn:
                cls = "above_first" if first_up < first_dn else "below_first"
                cls_simple = "both"
                first_break = min(first_up, first_dn)
            elif first_up:
                cls = "above_only"; cls_simple = "above_only"
                first_break = first_up
            elif first_dn:
                cls = "below_only"; cls_simple = "below_only"
                first_break = first_dn
            else:
                cls = "inside"; cls_simple = "inside"
                first_break = None
            entry_price = float(bars.loc[first_break]) if first_break is not None else np.nan
            row_out[f"break_class_{lb}d"]      = cls
            row_out[f"break_class_simple_{lb}d"] = cls_simple
            row_out[f"first_break_time_{lb}d"]  = first_break.strftime("%H:%M") if first_break else None
            row_out[f"entry_price_{lb}d"]       = entry_price
            row_out[f"ret_from_entry_{lb}d"]    = float(r.close_1700 - entry_price) if np.isfinite(entry_price) else np.nan
        out_rows.append(row_out)
        if i % 200 == 0:
            print(f"  {i}/{len(base)}  elapsed={time.time()-t0:.0f}s  rows={len(out_rows)}")

    return pd.DataFrame(out_rows)


def attach_gamma(df: pd.DataFrame) -> pd.DataFrame:
    g = pd.read_parquet(GAMMA_PATH)
    g["date"] = pd.to_datetime(g["date"]).dt.date
    return df.merge(
        g[["date","qqq_regime_open","ndx_regime_open"]],
        on="date", how="left",
    )


# ------------------------------ Cohort reporting ------------------------------

def cohort_stats(sub: pd.DataFrame, ret_col: str = "ret_from_entry_14d",
                 direction: str = "long") -> dict:
    """Direction = 'long' (profit if ret > 0) or 'short' (profit if ret < 0).
    For 'inside' / 'both' we just report sign-agnostic mean drift."""
    n = len(sub)
    if n < 2:
        return {"n": n}
    rets = sub[ret_col].dropna().values
    n = len(rets)
    if n < 2:
        return {"n": n}
    if direction == "long":
        p_profit = float((rets > 0).mean())
        mean = float(rets.mean())
        t, p = ss.ttest_1samp(rets, 0)  # null: zero drift
    elif direction == "short":
        p_profit = float((rets < 0).mean())  # short profitable when price drops
        mean = float(-rets.mean())  # report short PnL = -ret
        t, p = ss.ttest_1samp(-rets, 0)  # null: zero short PnL
    else:  # 'flat' — just descriptive
        p_profit = float((rets > 0).mean())
        mean = float(rets.mean())
        t, p = ss.ttest_1samp(rets, 0)
    return {"n": n, "p_profit": p_profit, "mean": mean,
            "t": float(t), "p": float(p)}


def fmt_row(label, st):
    if st.get("n", 0) < 2:
        return f"  {label:<55}  n={st.get('n',0):>4}  insufficient"
    sig = "  ***" if st["p"] < 0.001 else ("   **" if st["p"] < 0.01 else
          ("    *" if st["p"] < 0.05 else "     "))
    return (f"  {label:<55}  n={st['n']:>4}  P(profit)={st['p_profit']:.1%}  "
            f"mean_pnl={st['mean']:+7.2f} pts  t={st['t']:+5.2f}  p={st['p']:.4f}{sig}")


def report(df: pd.DataFrame):
    print(f"\nfull sample after gamma merge: {len(df)} rows")
    print(f"date range: {df['date'].min()} -> {df['date'].max()}")
    print()
    print("Entry: 5-min close that first breaks the static band")
    print("Exit:  NQ close at 17:00 ET")
    print("PnL is from break-time entry to close. Long for above_only,")
    print("short for below_only. 'inside'/'both' = no clean trade signal.")
    print()
    print("Significance markers: * p<0.05, ** p<0.01, *** p<0.001")

    for lb in LOOKBACKS:
        cls_col = f"break_class_simple_{lb}d"
        ret_col = f"ret_from_entry_{lb}d"
        print()
        print("=" * 95)
        print(f"LOOKBACK = {lb} days")
        print("=" * 95)
        counts = df[cls_col].value_counts()
        print(f"break-class counts: {dict(counts)}")
        print()

        for source, regime_col in [("QQQ", "qqq_regime_open"),
                                    ("NDX", "ndx_regime_open")]:
            print(f"--- {source}-derived gamma regime ---")
            for cls, direction in [("above_only", "long"),
                                    ("below_only", "short"),
                                    ("both", "long"),       # report long PnL just for reference
                                    ("inside", "long")]:
                sub = df[df[cls_col] == cls]
                print(fmt_row(f"  {cls:<11} ALL gamma  ({direction})",
                              cohort_stats(sub, ret_col, direction)))
                for regime in ["pos", "neg"]:
                    sub_r = sub[sub[regime_col] == regime]
                    print(fmt_row(f"    {cls:<11} + {source}_{regime}-gamma",
                                  cohort_stats(sub_r, ret_col, direction)))
            print()


def main():
    df = build_per_day_table()
    df = attach_gamma(df)
    df.to_parquet(PER_DAY_OUT, compression="zstd", index=False)
    print(f"\nwrote {PER_DAY_OUT}  ({len(df)} rows)")
    report(df)


if __name__ == "__main__":
    sys.exit(main())
