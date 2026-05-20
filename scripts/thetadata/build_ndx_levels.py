"""Compute NDX-native gamma levels from the local NDX/NDXP price archive.

ThetaData Standard plan blocks index-options greeks, so we derive them locally:
  1. Pair calls/puts at same (root, strike, expiration); per-pair parity spot
     S = (C_mid - P_mid) e^(rT) + K, dividend-adjusted.  Median across short-DTE
     pairs gives a consensus daily spot.
  2. Vectorized Newton-Raphson IV inversion using vega.
  3. Black-Scholes gamma from IV.
  4. Net GEX per strike (calls +, puts -, summed across roots/expirations).
  5. Extract call_resistance, put_support, HVL (closest sign flip to spot),
     GEX_1..10 (top 10 strikes by |net GEX|).

Output: D:/trading_pythonbacktest_data/NDX_thetadata/ndx_levels.parquet
        (one row per trading day with all 13 levels in NDX strike units;
         NDX strikes ≈ NQ price within the futures basis).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path("D:/trading_pythonbacktest_data/NDX_thetadata")
OUTPUT = ROOT / "ndx_levels.parquet"
R = 0.05    # short-rate proxy
Q = 0.006   # NDX dividend yield approx


# ---------------------------- BS primitives ----------------------------

def _d1(S, K, T, sigma):
    return (np.log(S / K) + (R - Q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def bs_call(S, K, T, sigma):
    d1 = _d1(S, K, T, sigma); d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-Q * T) * norm.cdf(d1) - K * np.exp(-R * T) * norm.cdf(d2)


def bs_put(S, K, T, sigma):
    d1 = _d1(S, K, T, sigma); d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-R * T) * norm.cdf(-d2) - S * np.exp(-Q * T) * norm.cdf(-d1)


def bs_vega(S, K, T, sigma):
    return S * np.exp(-Q * T) * norm.pdf(_d1(S, K, T, sigma)) * np.sqrt(T)


def bs_gamma(S, K, T, sigma):
    return np.exp(-Q * T) * norm.pdf(_d1(S, K, T, sigma)) / (S * sigma * np.sqrt(T))


# ---------------------------- IV inversion ----------------------------

def implied_vol_vec(price, S, K, T, is_call, init=0.30, max_iter=15):
    """Vectorized Newton-Raphson. Returns NaN where unconverged or vega vanishes."""
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


# ---------------------------- Spot via parity ----------------------------

def derive_spot(eod: pd.DataFrame, snap_date: pd.Timestamp,
                dte_max: int = 36) -> tuple[float, int]:
    df = eod[(eod["bid"] > 0) & (eod["ask"] > 0)].copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["dte"] = (df["expiration"] - snap_date).dt.days
    df = df[(df["dte"] > 0) & (df["dte"] <= dte_max)]

    calls = df[df["right"].str.upper() == "CALL"][
        ["root", "strike", "expiration", "dte", "mid"]].rename(columns={"mid": "C"})
    puts = df[df["right"].str.upper() == "PUT"][
        ["root", "strike", "expiration", "dte", "mid"]].rename(columns={"mid": "P"})
    pairs = calls.merge(puts, on=["root", "strike", "expiration", "dte"])
    if pairs.empty:
        return float("nan"), 0
    T = pairs["dte"].values / 365.25
    S = ((pairs["C"] - pairs["P"]) * np.exp(R * T) + pairs["strike"]) / np.exp(-Q * T)
    return float(np.median(S)), len(pairs)


# ---------------------------- Level extraction ----------------------------

def extract_levels(by_strike: pd.DataFrame, spot: float) -> dict:
    if by_strike.empty:
        return {}
    above = by_strike[by_strike["strike"] >= spot]
    below = by_strike[by_strike["strike"] <= spot]
    cr = float(above.loc[above["net_gex"].idxmax(), "strike"]) if not above.empty else None
    ps = float(below.loc[below["net_gex"].idxmin(), "strike"]) if not below.empty else None

    s = by_strike.sort_values("strike").reset_index(drop=True).copy()
    s["cum"] = s["net_gex"].cumsum()
    flips = []
    for i in range(1, len(s)):
        a, b = s.iloc[i - 1]["cum"], s.iloc[i]["cum"]
        if (a > 0) != (b > 0):
            flips.append(float(s.iloc[i]["strike"]))
    hvl = min(flips, key=lambda k: abs(k - spot)) if flips else None

    top10 = by_strike.reindex(by_strike["net_gex"].abs()
                              .sort_values(ascending=False).index).head(10)
    out = {"call_resistance": cr, "put_support": ps, "hvl": hvl}
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        out[f"GEX_{i}"] = float(row["strike"])
    return out


# ---------------------------- Per-day pipeline ----------------------------

def levels_for_day(date_dir: Path, max_dte: int) -> dict | None:
    eod_path = date_dir / "eod.parquet"
    oi_path = date_dir / "oi.parquet"
    if not eod_path.exists() or not oi_path.exists():
        return None

    eod = pd.read_parquet(eod_path)
    oi = pd.read_parquet(oi_path)
    eod["expiration"] = pd.to_datetime(eod["expiration"])
    oi["expiration"] = pd.to_datetime(oi["expiration"])
    snap_date = pd.to_datetime(date_dir.name)

    spot, n_pairs = derive_spot(eod, snap_date)
    if not np.isfinite(spot):
        return None

    chain = eod.merge(
        oi[["root", "strike", "right", "expiration", "open_interest"]],
        on=["root", "strike", "right", "expiration"], how="left",
    )
    chain["mid"] = (chain["bid"] + chain["ask"]) / 2
    chain = chain[(chain["bid"] > 0) & (chain["ask"] > 0) & (chain["mid"] > 0)]
    chain["dte"] = (chain["expiration"] - snap_date).dt.days
    if max_dte > 0:
        chain = chain[(chain["dte"] > 0) & (chain["dte"] <= max_dte)]
    else:
        chain = chain[chain["dte"] > 0]
    if chain.empty:
        return None

    T = chain["dte"].values / 365.25
    K = chain["strike"].values.astype(float)
    P = chain["mid"].values.astype(float)
    is_call = (chain["right"].str.upper() == "CALL").values
    S_arr = np.full_like(K, spot)

    iv = implied_vol_vec(P, S_arr, K, T, is_call)
    chain["iv"] = iv
    chain = chain.dropna(subset=["iv"])
    if chain.empty:
        return None

    chain["gamma"] = bs_gamma(spot, chain["strike"].values,
                              chain["dte"].values / 365.25, chain["iv"].values)
    chain["signed_gex"] = chain["gamma"] * chain["open_interest"].fillna(0) * 100 * spot ** 2
    chain.loc[chain["right"].str.upper() == "PUT", "signed_gex"] *= -1

    by_strike = (chain.groupby("strike")["signed_gex"].sum()
                 .reset_index().rename(columns={"signed_gex": "net_gex"})
                 .sort_values("strike"))

    out = {
        "date": dt.date.fromisoformat(date_dir.name),
        "ndx_spot": spot,
        "n_parity_pairs": n_pairs,
        "n_options": len(chain),
        **extract_levels(by_strike, spot),
    }
    return out


# ---------------------------- Driver ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dte", type=int, default=45,
                    help="DTE filter; -1 for full chain (default 45)")
    args = ap.parse_args()

    date_dirs = sorted([p for p in ROOT.iterdir()
                        if p.is_dir() and len(p.name) == 10 and p.name[4] == "-"])
    print(f"processing {len(date_dirs)} days, max_dte={args.max_dte}")

    rows = []; errs = 0
    t0 = time.time()
    for i, dd in enumerate(date_dirs, 1):
        try:
            r0 = levels_for_day(dd, args.max_dte)
            if r0:
                rows.append(r0)
        except Exception as e:
            errs += 1
            print(f"  {dd.name}: ERROR {type(e).__name__}: {e}")
        if i % 100 == 0 or i == len(date_dirs):
            print(f"  {i}/{len(date_dirs)}  elapsed={(time.time()-t0):.0f}s  "
                  f"rows={len(rows)}  errs={errs}")

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df.to_parquet(OUTPUT, compression="zstd", index=False)
    print(f"\nwrote {OUTPUT}  ({len(df)} rows)")

    # Quick sanity print
    print(f"\nlast 3 rows:")
    cols = ["date", "ndx_spot", "n_parity_pairs", "n_options",
            "call_resistance", "put_support", "hvl", "GEX_1", "GEX_2", "GEX_3"]
    print(df[cols].tail(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
