"""
Filter analysis for the LOCKED overnight drift config.

Tests three filters:
  1) Skip FOMC nights (the night of FOMC announcement + the night before)
  2) Skip MAG7 earnings nights (post-close earnings release)
  3) Pre-entry delta filter: cumulative buy_vol - sell_vol from 18:00 to 19:00 ET
     on the entry day. Does positive delta predict trade success? What
     threshold matters?

Inputs:
  live/overnight drift/trades.csv  (1,357 trades from locked config)
  D:/trading_pythonbacktest_data/timebars_5min_5yr/  (historical 5-min bars with buy/sell split)
  D:/trading_pythonbacktest_data/timebars_5min/      (recent 5-min bars)
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("C:/trading/nqorderflowbacktester/live/overnight drift")
TZ = "America/New_York"

# ---------------- FOMC meeting dates (announcement days, 2020-2026) ----------------
FOMC_DATES = pd.to_datetime([
    "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29",
])


# ---------------- MAG7 earnings — quarterly post-close releases ----------------
# Hand-curated from public earnings calendars 2020-2026
MAG7_EARNINGS: dict[str, list[str]] = {
    "AAPL": ["2021-01-27","2021-04-28","2021-07-27","2021-10-28",
             "2022-01-27","2022-04-28","2022-07-28","2022-10-27",
             "2023-02-02","2023-05-04","2023-08-03","2023-11-02",
             "2024-02-01","2024-05-02","2024-08-01","2024-10-31",
             "2025-01-30","2025-05-01","2025-07-31","2025-10-30",
             "2026-01-29","2026-04-30"],
    "MSFT": ["2021-01-26","2021-04-27","2021-07-27","2021-10-26",
             "2022-01-25","2022-04-26","2022-07-26","2022-10-25",
             "2023-01-24","2023-04-25","2023-07-25","2023-10-24",
             "2024-01-30","2024-04-25","2024-07-30","2024-10-30",
             "2025-01-29","2025-04-30","2025-07-30","2025-10-29",
             "2026-01-28","2026-04-29"],
    "GOOGL": ["2021-02-02","2021-04-27","2021-07-27","2021-10-26",
              "2022-02-01","2022-04-26","2022-07-26","2022-10-25",
              "2023-02-02","2023-04-25","2023-07-25","2023-10-24",
              "2024-01-30","2024-04-25","2024-07-23","2024-10-29",
              "2025-02-04","2025-04-24","2025-07-23","2025-10-28",
              "2026-02-03","2026-04-28"],
    "AMZN": ["2021-02-02","2021-04-29","2021-07-29","2021-10-28",
             "2022-02-03","2022-04-28","2022-07-28","2022-10-27",
             "2023-02-02","2023-04-27","2023-08-03","2023-10-26",
             "2024-02-01","2024-04-30","2024-08-01","2024-10-31",
             "2025-02-06","2025-05-01","2025-07-31","2025-10-30",
             "2026-02-05","2026-04-30"],
    "META": ["2021-01-27","2021-04-28","2021-07-28","2021-10-25",
             "2022-02-02","2022-04-27","2022-07-27","2022-10-26",
             "2023-02-01","2023-04-26","2023-07-26","2023-10-25",
             "2024-01-31","2024-04-24","2024-07-31","2024-10-30",
             "2025-01-29","2025-04-30","2025-07-30","2025-10-29",
             "2026-01-28","2026-04-29"],
    "NVDA": ["2021-02-24","2021-05-26","2021-08-18","2021-11-17",
             "2022-02-16","2022-05-25","2022-08-24","2022-11-16",
             "2023-02-22","2023-05-24","2023-08-23","2023-11-21",
             "2024-02-21","2024-05-22","2024-08-28","2024-11-20",
             "2025-02-26","2025-05-28","2025-08-27","2025-11-19",
             "2026-02-25","2026-05-27"],
    "TSLA": ["2021-01-27","2021-04-26","2021-07-26","2021-10-20",
             "2022-01-26","2022-04-20","2022-07-20","2022-10-19",
             "2023-01-25","2023-04-19","2023-07-19","2023-10-18",
             "2024-01-24","2024-04-23","2024-07-23","2024-10-23",
             "2025-01-29","2025-04-22","2025-07-23","2025-10-22",
             "2026-01-28","2026-04-22"],
}
EARNINGS_DATES = pd.to_datetime(sorted({d for dates in MAG7_EARNINGS.values() for d in dates}))


# ---------------- 5-min bar load (buy_vol / sell_vol per bar) ----------------

def load_5min_buy_sell() -> pd.DataFrame:
    """Load 5-min bars with buy_vol/sell_vol from both pickle folders.

    Returns DataFrame indexed by ET timestamp with cols [buy_vol, sell_vol].
    """
    rows = []
    for folder in [
        "D:/trading_pythonbacktest_data/timebars_5min_5yr",
        "D:/trading_pythonbacktest_data/timebars_5min",
    ]:
        files = sorted(os.listdir(folder))
        for f in files:
            if not f.startswith("timebars_5min_") or not f.endswith(".pkl"):
                continue
            with open(os.path.join(folder, f), "rb") as fh:
                day = pickle.load(fh)
            for bar in day:
                if "buy_vol" in bar and "sell_vol" in bar:
                    rows.append({
                        "ts": bar["open_time"],
                        "buy_vol": bar["buy_vol"],
                        "sell_vol": bar["sell_vol"],
                    })
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index = df.index.tz_convert(TZ)
    return df


def compute_pre_entry_delta(trades: pd.DataFrame, bars5: pd.DataFrame) -> pd.Series:
    """For each trade, sum buy_vol - sell_vol from 18:00 to 19:00 ET on the
    entry day. Returns Series aligned to trades."""
    bars5 = bars5.copy()
    bars5["delta"] = bars5["buy_vol"] - bars5["sell_vol"]
    bars5["total"] = bars5["buy_vol"] + bars5["sell_vol"]
    # Build a per-session-date lookup of the hourly delta
    et_dt = bars5.index
    in_window = (et_dt.time >= pd.Timestamp("18:00").time()) & (et_dt.time < pd.Timestamp("19:00").time())
    win = bars5[in_window].copy()
    win["session_date"] = win.index.normalize().tz_localize(None)
    hourly = win.groupby("session_date").agg(
        delta=("delta", "sum"),
        total=("total", "sum"),
    )
    hourly["delta_pct"] = hourly["delta"] / hourly["total"] * 100
    return trades["session_date"].map(hourly["delta"]), trades["session_date"].map(hourly["delta_pct"]), trades["session_date"].map(hourly["total"])


def stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"trades": 0, "wr": np.nan, "PF": np.nan, "gross": 0.0, "avg": np.nan, "mdd": 0.0}
    pnl = df["pnl_dollars"].to_numpy()
    wins = (pnl > 0).sum()
    gw = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    c = np.cumsum(pnl); peak = np.maximum.accumulate(c)
    mdd = float((c - peak).min())
    return {"trades": int(len(pnl)), "wr": float(wins / len(pnl) * 100),
            "PF": float(gw / gl) if gl > 0 else float("inf"),
            "gross": float(pnl.sum()), "avg": float(pnl.mean()), "mdd": mdd}


def main() -> None:
    trades = pd.read_csv(OUT_DIR / "trades.csv")
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["session_date"] = trades["entry_time"].dt.tz_convert(TZ).dt.normalize().dt.tz_localize(None)
    trades = trades.sort_values("session_date").reset_index(drop=True)
    base = stats(trades)
    print(f"BASELINE — all {base['trades']} trades")
    print(f"  WR {base['wr']:.2f}%  PF {base['PF']:.3f}  gross ${base['gross']:,.0f}  MaxDD ${base['mdd']:,.0f}\n")

    # ----------- 1) FOMC filter -----------
    fomc_set = set(FOMC_DATES.normalize())
    eve_set = set((FOMC_DATES - pd.Timedelta(days=1)).normalize())
    trades["is_fomc_day"] = trades["session_date"].isin(fomc_set)
    trades["is_fomc_eve"] = trades["session_date"].isin(eve_set)
    trades["is_fomc_window"] = trades["is_fomc_day"] | trades["is_fomc_eve"]

    print("=== FILTER 1: FOMC ===")
    print(f"FOMC announcement dates in sample: {sum(trades['is_fomc_day'])}")
    print(f"FOMC eve dates in sample:          {sum(trades['is_fomc_eve'])}")
    print(f"Total FOMC-window dates skipped:   {sum(trades['is_fomc_window'])}")
    for label, mask in [
        ("FOMC day only", trades["is_fomc_day"]),
        ("FOMC eve only", trades["is_fomc_eve"]),
        ("FOMC day + eve combined", trades["is_fomc_window"]),
    ]:
        on = stats(trades[mask])
        off = stats(trades[~mask])
        print(f"\n  {label}: {on['trades']} trades hit, {off['trades']} survive filter")
        print(f"    Hit nights:   WR {on['wr']:.1f}%  PF {on['PF']:.2f}  avg ${on['avg']:.0f}  gross ${on['gross']:,.0f}")
        print(f"    SKIP outcome: WR {off['wr']:.1f}%  PF {off['PF']:.2f}  avg ${off['avg']:.0f}  gross ${off['gross']:,.0f}")
        print(f"    delta vs baseline gross: ${off['gross'] - base['gross']:+,.0f}")

    # ----------- 2) Earnings filter -----------
    print("\n=== FILTER 2: MAG7 earnings ===")
    earn_set = set(EARNINGS_DATES.normalize())
    trades["is_earn"] = trades["session_date"].isin(earn_set)
    on = stats(trades[trades["is_earn"]])
    off = stats(trades[~trades["is_earn"]])
    print(f"Earnings nights in sample: {trades['is_earn'].sum()}")
    print(f"  Hit nights:   WR {on['wr']:.1f}%  PF {on['PF']:.2f}  avg ${on['avg']:.0f}  gross ${on['gross']:,.0f}")
    print(f"  SKIP outcome: WR {off['wr']:.1f}%  PF {off['PF']:.2f}  avg ${off['avg']:.0f}  gross ${off['gross']:,.0f}")
    print(f"  delta vs baseline gross: ${off['gross'] - base['gross']:+,.0f}")

    # Combined FOMC + earnings
    combined = trades["is_fomc_window"] | trades["is_earn"]
    on = stats(trades[combined])
    off = stats(trades[~combined])
    print(f"\nCombined FOMC + earnings: {combined.sum()} skipped")
    print(f"  Hit nights:   WR {on['wr']:.1f}%  PF {on['PF']:.2f}  avg ${on['avg']:.0f}  gross ${on['gross']:,.0f}")
    print(f"  SKIP outcome: WR {off['wr']:.1f}%  PF {off['PF']:.2f}  avg ${off['avg']:.0f}  gross ${off['gross']:,.0f}")

    # ----------- 3) Pre-entry delta filter -----------
    print("\n=== FILTER 3: 18:00-19:00 ET cumulative delta ===")
    print("Loading 5-min buy/sell bars (this takes ~30s)...", flush=True)
    bars5 = load_5min_buy_sell()
    print(f"  5-min bars loaded: {len(bars5):,}")
    delta, delta_pct, total = compute_pre_entry_delta(trades, bars5)
    trades["pre_delta"] = delta
    trades["pre_delta_pct"] = delta_pct
    trades["pre_total"] = total
    have = trades["pre_delta"].notna().sum()
    print(f"  Trades with delta data: {have}/{len(trades)}")

    # Quintile analysis
    sub = trades.dropna(subset=["pre_delta"]).copy()
    sub["delta_quintile"] = pd.qcut(sub["pre_delta"], 5, labels=["Q1 (most neg)", "Q2", "Q3", "Q4", "Q5 (most pos)"])
    print("\nBy delta quintile:")
    print(f"  {'Bucket':<18s}  {'cutoff':>12s}  {'trades':>6s}  {'WR%':>5s}  {'PF':>5s}  {'avg_$':>6s}  {'gross_$':>10s}")
    for q in sub["delta_quintile"].cat.categories:
        g = sub[sub["delta_quintile"] == q]
        s = stats(g)
        lo, hi = g["pre_delta"].min(), g["pre_delta"].max()
        print(f"  {str(q):<18s}  [{lo:>5.0f},{hi:>5.0f}]  {s['trades']:>6d}  {s['wr']:>5.1f}  {s['PF']:>5.2f}  {s['avg']:>6.0f}  {s['gross']:>10,.0f}")

    print("\nSign split (negative vs positive delta):")
    for label, mask in [
        ("delta < 0", sub["pre_delta"] < 0),
        ("delta > 0", sub["pre_delta"] > 0),
        ("delta >= +500", sub["pre_delta"] >= 500),
        ("delta >= +1000", sub["pre_delta"] >= 1000),
        ("delta >= +2000", sub["pre_delta"] >= 2000),
        ("delta <= -500", sub["pre_delta"] <= -500),
        ("delta <= -1000", sub["pre_delta"] <= -1000),
    ]:
        g = sub[mask]
        s = stats(g)
        if s["trades"]:
            print(f"  {label:<16s}  trades={s['trades']:>4d}  WR={s['wr']:>5.1f}  PF={s['PF']:>5.2f}  avg=${s['avg']:>6.0f}  gross=${s['gross']:>9,.0f}")

    print("\nBy delta_pct (normalized, removes volume effect):")
    sub["delta_pct_quintile"] = pd.qcut(sub["pre_delta_pct"], 5,
                                          labels=["Q1 (most -)", "Q2", "Q3", "Q4", "Q5 (most +)"])
    for q in sub["delta_pct_quintile"].cat.categories:
        g = sub[sub["delta_pct_quintile"] == q]
        s = stats(g)
        lo, hi = g["pre_delta_pct"].min(), g["pre_delta_pct"].max()
        print(f"  {str(q):<16s}  [{lo:>+5.1f}%, {hi:>+5.1f}%]  trades={s['trades']:>3}  WR={s['wr']:>5.1f}  PF={s['PF']:>5.2f}  avg=${s['avg']:>6.0f}")

    # Correlation
    from scipy import stats as ss
    r, p = ss.spearmanr(sub["pre_delta"], sub["pnl_dollars"])
    r_pct, p_pct = ss.spearmanr(sub["pre_delta_pct"], sub["pnl_dollars"])
    print(f"\nSpearman(delta, pnl):     r={r:.3f}  p={p:.3f}")
    print(f"Spearman(delta%, pnl):    r={r_pct:.3f}  p={p_pct:.3f}")

    trades.to_csv(OUT_DIR / "trades_with_filters.csv", index=False)
    print(f"\nSaved -> {OUT_DIR / 'trades_with_filters.csv'}")


if __name__ == "__main__":
    main()
