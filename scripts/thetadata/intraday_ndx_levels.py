"""MenthorQ-style intraday NDX gamma levels for a given date.

Pulls option/history/quote (NDX + NDXP) at 10-min bars on Standard plan, derives
NDX spot via put-call parity per snapshot, inverts BS for IV, computes gamma,
and produces 14 MenthorQ-style snapshots through the day. NQ-equivalent strikes
are shown using the day's NDX-NQ basis.

Usage:  python intraday_ndx_levels.py [YYYY-MM-DD] [--include-0dte]
Output: scripts/thetadata/intraday_ndx_levels_<date>.txt
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
Q = 0.006
ROOTS = ("NDX", "NDXP")
BASE = "http://127.0.0.1:25503/v3"
TIMEOUT = 120.0

# Daily NDX-NQ basis (NQ_settle - NDX_parity) — small per-day variation.
# These match what build_ndx_levels.py produces for the same dates.
DAILY_BASIS = {
    dt.date(2026, 4, 23): 98.9,
    dt.date(2026, 4, 27): 99.4,
    dt.date(2026, 4, 28): 105.7,
    dt.date(2026, 4, 29): 109.1,
    dt.date(2026, 4, 30): 106.4,
}

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


# ---------- HTTP ----------

def get_ndjson(path, params):
    r = httpx.get(f"{BASE}{path}", params={**params, "format": "ndjson"}, timeout=TIMEOUT)
    if r.status_code == 472 or not r.text.strip():
        return pd.DataFrame()
    r.raise_for_status()
    return pd.read_json(io.StringIO(r.text), lines=True)


def list_expirations(symbol):
    df = get_ndjson("/option/list/expirations", {"symbol": symbol})
    if df.empty: return []
    return pd.to_datetime(df["expiration"]).dt.date.tolist()


# ---------- BS + IV ----------

def _d1(S, K, T, sigma):
    return (np.log(S/K) + (R - Q + 0.5*sigma**2) * T) / (sigma * np.sqrt(T))

def bs_call(S, K, T, sigma):
    d1 = _d1(S, K, T, sigma); d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-Q*T)*norm.cdf(d1) - K*np.exp(-R*T)*norm.cdf(d2)

def bs_put(S, K, T, sigma):
    d1 = _d1(S, K, T, sigma); d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-R*T)*norm.cdf(-d2) - S*np.exp(-Q*T)*norm.cdf(-d1)

def bs_vega(S, K, T, sigma):
    return S*np.exp(-Q*T)*norm.pdf(_d1(S, K, T, sigma))*np.sqrt(T)

def bs_gamma(S, K, T, sigma):
    return np.exp(-Q*T)*norm.pdf(_d1(S, K, T, sigma)) / (S*sigma*np.sqrt(T))


def iv_vec(price, S, K, T, is_call, init=0.30, max_iter=15):
    sigma = np.full_like(price, init, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(max_iter):
            bs_p = np.where(is_call, bs_call(S, K, T, sigma), bs_put(S, K, T, sigma))
            diff = bs_p - price
            v = bs_vega(S, K, T, sigma)
            step = np.divide(diff, v, out=np.zeros_like(diff), where=(v > 1e-8))
            sigma = np.clip(sigma - step, 1e-4, 5.0)
        bs_p = np.where(is_call, bs_call(S, K, T, sigma), bs_put(S, K, T, sigma))
        bad = (np.abs(bs_p - price) > np.maximum(0.10, 0.01 * price))
    sigma[bad] = np.nan
    return sigma


# ---------- Spot via parity at a snapshot ----------

def derive_spot(snap_chain: pd.DataFrame) -> tuple[float, int]:
    df = snap_chain[(snap_chain["bid"] > 0) & (snap_chain["ask"] > 0)].copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2
    short = df[(df["dte"] > 0) & (df["dte"] <= 36)]
    calls = short[short["right"].str.upper() == "CALL"][["root","strike","expiration","dte","mid"]].rename(columns={"mid":"C"})
    puts = short[short["right"].str.upper() == "PUT"][["root","strike","expiration","dte","mid"]].rename(columns={"mid":"P"})
    pairs = calls.merge(puts, on=["root","strike","expiration","dte"])
    if pairs.empty:
        return float("nan"), 0
    T = pairs["dte"].values / 365.25
    S = ((pairs["C"] - pairs["P"]) * np.exp(R * T) + pairs["strike"]) / np.exp(-Q * T)
    return float(np.median(S)), len(pairs)


# ---------- Level extraction ----------

def extract_levels(by_strike: pd.DataFrame, spot: float) -> dict:
    if by_strike.empty: return {}
    above = by_strike[by_strike["strike"] >= spot]
    below = by_strike[by_strike["strike"] <= spot]
    cr = float(above.loc[above["net_gex"].idxmax(), "strike"]) if not above.empty else None
    ps = float(below.loc[below["net_gex"].idxmin(), "strike"]) if not below.empty else None

    s = by_strike.sort_values("strike").reset_index(drop=True).copy()
    s["cum"] = s["net_gex"].cumsum()
    flips = []
    for i in range(1, len(s)):
        if (s.iloc[i-1]["cum"] > 0) != (s.iloc[i]["cum"] > 0):
            flips.append(float(s.iloc[i]["strike"]))
    hvl = min(flips, key=lambda k: abs(k-spot)) if flips else None

    top10 = (by_strike.reindex(by_strike["net_gex"].abs().sort_values(ascending=False).index)
             .head(10)[["strike","net_gex"]])
    return {
        "call_resistance": cr, "put_support": ps, "hvl": hvl,
        "top10": list(top10.itertuples(index=False, name=None)),
    }


# ---------- Driver ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", nargs="?", default="2026-04-28")
    ap.add_argument("--include-0dte", action="store_true")
    args = ap.parse_args()

    DATE = dt.date.fromisoformat(args.date)
    basis = DAILY_BASIS.get(DATE, 105.0)
    out_path = Path(__file__).parent / f"intraday_ndx_levels_{DATE.isoformat()}.txt"
    t0 = time.time()
    date_s = DATE.strftime("%Y%m%d")

    # 1. Expirations within MAX_DTE for both roots
    target_exps = []  # list of (root, exp_date)
    for root in ROOTS:
        exps = list_expirations(root)
        for e in exps:
            if DATE <= e <= DATE + dt.timedelta(days=MAX_DTE):
                target_exps.append((root, e))
    print(f"expirations to fetch: {len(target_exps)}  ({len(ROOTS)} roots × ~{len(target_exps)//len(ROOTS)} ea)")

    # 2. Pull intraday quotes per (root, exp)
    parts = []
    for i, (root, exp) in enumerate(target_exps, 1):
        df = get_ndjson("/option/history/quote", {
            "symbol": root, "expiration": exp.strftime("%Y%m%d"),
            "start_date": date_s, "end_date": date_s, "interval": INTERVAL,
        })
        if not df.empty:
            df["root"] = root
            df["expiration"] = pd.to_datetime(df["expiration"])
            parts.append(df)
        if i % 10 == 0 or i == len(target_exps):
            print(f"  fetched {i}/{len(target_exps)}  elapsed={(time.time()-t0):.0f}s")
    quotes = pd.concat(parts, ignore_index=True)
    print(f"\ntotal quote rows: {len(quotes)}")

    # 3. OI per root (one wildcard call each)
    oi_parts = []
    for root in ROOTS:
        oi_df = get_ndjson("/option/history/open_interest", {
            "symbol": root, "expiration": "*",
            "start_date": date_s, "end_date": date_s,
        })
        if not oi_df.empty:
            oi_df["root"] = root
            oi_df["expiration"] = pd.to_datetime(oi_df["expiration"])
            oi_parts.append(oi_df)
    oi = pd.concat(oi_parts, ignore_index=True)
    print(f"OI rows: {len(oi)}")

    # 4. Merge OI into quotes
    quotes["timestamp"] = pd.to_datetime(quotes["timestamp"])
    chain = quotes.merge(oi[["root","strike","right","expiration","open_interest"]],
                         on=["root","strike","right","expiration"], how="left")
    chain["dte"] = (chain["expiration"] - pd.Timestamp(DATE)).dt.days
    dte_min = 0 if args.include_0dte else 1
    chain = chain[(chain["dte"] >= dte_min) & (chain["dte"] <= MAX_DTE)]
    chain["hhmm"] = chain["timestamp"].dt.strftime("%H:%M")
    bars_present = sorted(chain["hhmm"].unique())
    print(f"chain rows after DTE filter ({dte_min}..{MAX_DTE}): {len(chain)}")
    print(f"intraday bars: {bars_present[:6]}...{bars_present[-3:]}")

    # 5. For each snapshot time, compute levels
    lines = []
    lines.append(f"=== NDX INTRADAY GAMMA LEVELS  {DATE} ({DATE.strftime('%a')}) ===")
    lines.append(f"  Method: option/history/quote @ {INTERVAL} bars, NDX + NDXP combined")
    lines.append(f"          Spot via put-call parity (per-snapshot consensus)")
    lines.append(f"          IV via vectorized Newton-Raphson; BS gamma local (r={R}, q={Q})")
    lines.append(f"  DTE filter: {dte_min} <= dte <= {MAX_DTE}  "
                 f"({'0DTE included' if args.include_0dte else '0DTE excluded'})")
    lines.append(f"  NDX-NQ basis ({DATE} EOD): +{basis:.1f} pts (NQ ~ NDX strike + basis)")
    lines.append(f"  Format: 'NDX_strike  NQ_equiv  (signed_gex)'")
    lines.append("")

    for snap, label in SNAPSHOTS:
        sub = chain[chain["hhmm"] == snap]
        if sub.empty:
            lines.append(f"--- {snap} ET {('(' + label + ')') if label else ''} ---")
            lines.append(f"    no bar at {snap}")
            lines.append("")
            continue

        spot, n_pairs = derive_spot(sub)
        if not np.isfinite(spot):
            lines.append(f"--- {snap} ET {('(' + label + ')') if label else ''} ---")
            lines.append(f"    insufficient call/put pairs for spot derivation")
            lines.append("")
            continue

        sub = sub[(sub["bid"] > 0) & (sub["ask"] > 0)].copy()
        sub["mid"] = (sub["bid"] + sub["ask"]) / 2
        T = sub["dte"].values / 365.25
        K = sub["strike"].values.astype(float)
        P = sub["mid"].values.astype(float)
        is_call = (sub["right"].str.upper() == "CALL").values
        S_arr = np.full_like(K, spot)
        sub["iv"] = iv_vec(P, S_arr, K, T, is_call)
        sub = sub.dropna(subset=["iv"])
        if sub.empty:
            lines.append(f"--- {snap} ET ---  IV inversion failed for all options")
            lines.append("")
            continue

        sub["gamma"] = bs_gamma(spot, sub["strike"].values,
                                sub["dte"].values/365.25, sub["iv"].values)
        sub["signed_gex"] = sub["gamma"] * sub["open_interest"].fillna(0) * 100 * spot**2
        sub.loc[sub["right"].str.upper() == "PUT", "signed_gex"] *= -1
        by_strike = (sub.groupby("strike")["signed_gex"].sum()
                     .reset_index().rename(columns={"signed_gex":"net_gex"})
                     .sort_values("strike"))
        lv = extract_levels(by_strike, spot)

        lines.append(f"--- {snap} ET {('(' + label + ')') if label else ''} ---")
        lines.append(f"  NDX spot (parity): {spot:.2f}    NQ equiv: {spot + basis:.2f}    n_pairs={n_pairs}")
        for name in ("call_resistance", "put_support", "hvl"):
            v = lv.get(name)
            if v is None:
                lines.append(f"  {name:<22}  --")
            else:
                lines.append(f"  {name:<22}  NDX {v:>8.0f}  NQ {v + basis:>9.0f}")
        for i, (k, gv) in enumerate(lv.get("top10", []), 1):
            sign = "+" if gv >= 0 else "-"
            lines.append(f"  GEX_{i:<2}                 NDX {k:>8.0f}  NQ {k + basis:>9.0f}  "
                         f"({sign}{abs(gv):.2e})")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out_path}  ({out_path.stat().st_size} bytes)")
    print(f"total elapsed: {(time.time() - t0):.0f}s")


if __name__ == "__main__":
    sys.exit(main())
