"""MenthorQ-style intraday gamma levels for QQQ on a given date.

ThetaData Standard plan blocks intraday gamma. We pull first_order greeks (IV)
at 10-min bars for every expiration <=MAX_DTE, compute Black-Scholes gamma
locally, merge with the day's open_interest, and produce levels at each of the
14 MenthorQ snapshot times.

NQ conversion: per-day settle ratio (intraday ratio drift is sub-bp).

Usage:  python intraday_levels_2026_04_28.py [YYYY-MM-DD] [--include-0dte]
Output: scripts/thetadata/intraday_levels_<date>.txt
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from scipy.stats import norm

MAX_DTE = 45
INTERVAL = "10m"
R = 0.05
# Per-date settle ratios (NQ at 16:59 ET / QQQ underlying_price at 17:15 ET)
DAILY_RATIO = {
    dt.date(2026, 4, 22): 41.378,
    dt.date(2026, 4, 23): 41.360,
    dt.date(2026, 4, 27): 41.279,
    dt.date(2026, 4, 28): 41.353,
    dt.date(2026, 4, 29): 41.037,
    dt.date(2026, 4, 30): 41.331,
}

BASE = "http://127.0.0.1:25503/v3"
TIMEout_path = 120.0

SNAPSHOTS = [
    ("08:00", "pre-market"),
    ("09:50", "first intraday"),
    ("10:20", ""), ("10:50", ""),
    ("11:20", ""), ("11:50", ""),
    ("12:20", ""), ("12:50", ""),
    ("13:20", ""), ("13:50", ""),
    ("14:20", ""), ("14:50", ""),
    ("15:20", ""), ("15:50", "final hour"),
]


def get_ndjson(path: str, params: dict) -> pd.DataFrame:
    r = httpx.get(f"{BASE}{path}",
                  params={**params, "format": "ndjson"}, timeout=TIMEout_path)
    if r.status_code == 472 or not r.text.strip():
        return pd.DataFrame()
    r.raise_for_status()
    return pd.read_json(io.StringIO(r.text), lines=True)


def bs_gamma(S, K, T, sigma, r):
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def gex_per_strike(snap: pd.DataFrame) -> pd.DataFrame:
    if snap.empty:
        return pd.DataFrame()
    df = snap.copy()
    df["signed_gex"] = df["gamma"] * df["open_interest"].fillna(0) * 100
    df.loc[df["right"].str.upper() == "PUT", "signed_gex"] *= -1
    spot = df["underlying_price"].iloc[0]
    df["signed_gex"] *= spot ** 2
    return (df.groupby("strike")["signed_gex"].sum()
              .reset_index().rename(columns={"signed_gex": "net_gex"})
              .sort_values("strike"))


def extract_levels(gex: pd.DataFrame, spot: float) -> dict:
    if gex.empty:
        return {}
    above = gex[gex["strike"] >= spot]
    below = gex[gex["strike"] <= spot]
    cr = float(above.loc[above["net_gex"].idxmax(), "strike"]) if not above.empty else None
    ps = float(below.loc[below["net_gex"].idxmin(), "strike"]) if not below.empty else None

    s = gex.sort_values("strike").reset_index(drop=True).copy()
    s["cum"] = s["net_gex"].cumsum()
    sign = s["cum"].apply(lambda v: 1 if v > 0 else -1)
    flip = s[sign.diff().abs() > 0]
    hvl = float(flip.iloc[0]["strike"]) if not flip.empty else None

    top10 = (gex.reindex(gex["net_gex"].abs().sort_values(ascending=False).index).head(10))
    return {
        "call_resistance": cr, "put_support": ps, "hvl": hvl,
        "top10": list(top10[["strike", "net_gex"]].itertuples(index=False, name=None)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", nargs="?", default="2026-04-28",
                    help="ISO date (default 2026-04-28)")
    ap.add_argument("--include-0dte", action="store_true",
                    help="Include same-day expiration (dte==0) in the chain")
    ap.add_argument("--ratio", type=float, default=None,
                    help="Override NQ/QQQ ratio (otherwise looked up by date)")
    args = ap.parse_args()

    DATE = dt.date.fromisoformat(args.date)
    nq_ratio = args.ratio if args.ratio else DAILY_RATIO.get(DATE, 41.30)
    out_path = Path(__file__).parent / f"intraday_levels_{DATE.isoformat()}.txt"

    t0 = time.time()
    date_s = DATE.strftime("%Y%m%d")

    # 1. Expirations within 45 DTE
    exps_df = get_ndjson("/option/list/expirations", {"symbol": "QQQ"})
    exps = pd.to_datetime(exps_df["expiration"]).dt.date.tolist()
    target_exps = [e for e in exps if DATE <= e <= DATE + dt.timedelta(days=MAX_DTE)]
    print(f"expirations to fetch: {len(target_exps)}")

    # 2. Loop expirations and pull first_order greeks at 10m
    parts = []
    for i, exp in enumerate(target_exps, 1):
        df = get_ndjson("/option/history/greeks/first_order", {
            "symbol": "QQQ", "expiration": exp.strftime("%Y%m%d"),
            "start_date": date_s, "end_date": date_s, "interval": INTERVAL,
        })
        if not df.empty:
            df["_exp_obj"] = exp
            parts.append(df)
        if i % 5 == 0 or i == len(target_exps):
            print(f"  fetched {i}/{len(target_exps)}  elapsed={(time.time()-t0):.0f}s")
    greeks = pd.concat(parts, ignore_index=True)
    print(f"\ntotal greek rows: {len(greeks)}")

    # 3. OI snapshot for the day (wildcard works)
    oi = get_ndjson("/option/history/open_interest", {
        "symbol": "QQQ", "expiration": "*",
        "start_date": date_s, "end_date": date_s,
    })
    oi["expiration"] = pd.to_datetime(oi["expiration"])
    print(f"OI rows: {len(oi)}")

    # 4. Compute T, BS gamma
    greeks["expiration"] = pd.to_datetime(greeks["expiration"])
    greeks["timestamp"] = pd.to_datetime(greeks["timestamp"])
    # ThetaData option timestamps are ET-naive. Treat as ET wall time.
    exp_close = greeks["expiration"] + pd.Timedelta(hours=16)
    greeks["T"] = (exp_close - greeks["timestamp"]).dt.total_seconds() / (365.25 * 86400)
    valid = (greeks["implied_vol"] > 0) & (greeks["T"] > 0)
    g = greeks[valid].copy()
    g["gamma"] = bs_gamma(g["underlying_price"], g["strike"], g["T"],
                          g["implied_vol"], R)
    g = g.dropna(subset=["gamma"])
    g = g[g["gamma"].abs() < 100]  # drop pathological extreme-OTM rows

    # 5. Merge OI
    chain = g.merge(oi[["strike", "right", "expiration", "open_interest"]],
                    on=["strike", "right", "expiration"], how="left")

    # 6. DTE filter
    chain["dte"] = (chain["expiration"] - pd.Timestamp(DATE)).dt.days
    dte_min = 0 if args.include_0dte else 1
    chain = chain[(chain["dte"] >= dte_min) & (chain["dte"] <= MAX_DTE)]
    print(f"chain rows after DTE filter (>={dte_min}, <={MAX_DTE}): {len(chain)}")

    chain["hhmm"] = chain["timestamp"].dt.strftime("%H:%M")
    bars_present = sorted(chain["hhmm"].unique())
    print(f"\nintraday bars present: {bars_present[:10]} ... {bars_present[-5:]}")

    lines = []
    lines.append(f"=== QQQ INTRADAY GAMMA LEVELS  {DATE} ({DATE.strftime('%a')}) ===")
    lines.append(f"  Method: first_order greeks @ {INTERVAL} bars; BS gamma local (r={R})")
    lines.append(f"  DTE filter: {dte_min} <= dte <= {MAX_DTE}"
                 f"  ({'0DTE included' if args.include_0dte else '0DTE excluded'})")
    lines.append(f"  NQ conversion ratio ({DATE} settle): {nq_ratio:.3f}")
    lines.append(f"  Format: 'QQQ_strike  NQ_equiv  (signed_gex)'")
    lines.append("")

    for snap, label in SNAPSHOTS:
        sub = chain[chain["hhmm"] == snap]
        if sub.empty:
            lines.append(f"--- {snap} ET {('(' + label + ')') if label else ''} ---")
            lines.append(f"    no bar at {snap}; nearest bars in dataset: "
                         f"{[b for b in bars_present if abs(int(b[:2])*60+int(b[3:]) - (int(snap[:2])*60+int(snap[3:])))<=10]}")
            lines.append("")
            continue

        spot = float(sub["underlying_price"].iloc[0])
        gex = gex_per_strike(sub)
        lv = extract_levels(gex, spot)
        ratio_label = f"{label}" if label else ""
        lines.append(f"--- {snap} ET {('(' + ratio_label + ')') if ratio_label else ''} ---")
        lines.append(f"  QQQ spot: {spot:.2f}    NQ equiv: {spot * nq_ratio:.2f}")
        for name in ("call_resistance", "put_support", "hvl"):
            v = lv.get(name)
            if v is None:
                lines.append(f"  {name:<22}  --")
            else:
                lines.append(f"  {name:<22}  QQQ {v:>7.2f}  NQ {v * nq_ratio:>9.2f}")
        for i, (k, gv) in enumerate(lv.get("top10", []), 1):
            sign = "+" if gv >= 0 else "-"
            lines.append(f"  GEX_{i:<2}                 QQQ {k:>7.2f}  NQ {k * nq_ratio:>9.2f}  "
                         f"({sign}{abs(gv):.2e})")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path}  ({out_path.stat().st_size} bytes)")
    print(f"total elapsed: {(time.time() - t0):.0f}s")


if __name__ == "__main__":
    sys.exit(main())
