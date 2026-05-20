"""Validate locally-computed Black-Scholes gamma against ThetaData's reported gamma.

Why this matters: on the Standard plan, intraday `first_order` greeks return IV but
NOT gamma (gamma is in the Pro-only `second_order` endpoint). To produce intraday
gamma levels on Standard, we compute gamma ourselves from BS using IV, spot, strike,
T, and r. This script confirms the math by comparing our computed gamma to
ThetaData's gamma in the EOD greeks payload (which we DO have on Standard).

Usage:
    python validate_bs_gamma.py [YYYY-MM-DD]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

DATA_ROOT = Path("D:/trading_pythonbacktest_data/QQQ_thetadata")
DEFAULT_DATE = "2024-01-22"  # arbitrary date downloaded early in the run
DEFAULT_R = 0.05  # short-rate proxy; we'll grid-search anyway


def bs_gamma(S, K, T, sigma, r):
    """Standard BS gamma. Vectorized."""
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def gamma_from_d1(S, T, sigma, d1):
    """Gamma using ThetaData's reported d1 — bypasses any disagreement on r/T conventions."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def load_day(date_str: str) -> pd.DataFrame:
    p = DATA_ROOT / date_str / "greeks_eod.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — wait for download to reach this date")
    df = pd.read_parquet(p)
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["underlying_timestamp"] = pd.to_datetime(df["underlying_timestamp"])
    return df


def report(df: pd.DataFrame, label: str, mask: pd.Series | None = None):
    sub = df if mask is None else df[mask]
    if sub.empty:
        print(f"  {label}: no rows")
        return
    err = sub["gamma_pred"] - sub["gamma"]
    abs_err = err.abs()
    rel = (abs_err / sub["gamma"].abs().replace(0, np.nan)).dropna()
    valid = sub[(sub["gamma"].abs() > 1e-9) & sub["gamma_pred"].notna()]
    if len(valid) > 1:
        corr = valid["gamma"].corr(valid["gamma_pred"])
    else:
        corr = float("nan")
    print(f"  {label:<25} n={len(sub):<6}  "
          f"mae={abs_err.mean():.6f}  "
          f"max_err={abs_err.max():.6f}  "
          f"median_rel={rel.median():.4%}  "
          f"corr={corr:.6f}")


def main():
    date_str = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATE
    print(f"Validating BS gamma on {date_str}\n")

    df = load_day(date_str)
    print(f"loaded {len(df)} rows")
    print(f"columns we use: implied_vol, underlying_price, strike, expiration, "
          f"underlying_timestamp, d1, gamma\n")

    # Time to expiration in years (calendar 365)
    df["T"] = ((df["expiration"] + pd.Timedelta(hours=16)
                - df["underlying_timestamp"]).dt.total_seconds()
               / (365.25 * 24 * 3600))

    # DTE bucket for breakdown
    df["dte"] = ((df["expiration"] - df["underlying_timestamp"].dt.normalize())
                 .dt.days)

    # Drop rows where IV/T is non-positive — gamma is undefined
    valid = (df["implied_vol"] > 0) & (df["T"] > 0) & df["d1"].notna()
    df = df[valid].copy()
    print(f"after filtering invalid IV/T: {len(df)} rows\n")

    # Method A: use ThetaData's d1 (validates the formula independent of r/T choice)
    df["gamma_pred"] = gamma_from_d1(df["underlying_price"], df["T"],
                                     df["implied_vol"], df["d1"])
    print("Method A — using ThetaData's d1:")
    report(df, "all rows")
    report(df, "0 < DTE <= 7",  (df["dte"] > 0) & (df["dte"] <= 7))
    report(df, "8 <= DTE <= 30", (df["dte"] >= 8) & (df["dte"] <= 30))
    report(df, "31 <= DTE <= 90",(df["dte"] >= 31) & (df["dte"] <= 90))
    report(df, "DTE > 90",       df["dte"] > 90)

    # Method B: full BS from scratch with r grid
    print("\nMethod B — full BS, sweeping r to find best fit:")
    for r in [0.0, 0.02, 0.04, 0.05, 0.053, 0.06]:
        df["gamma_pred"] = bs_gamma(df["underlying_price"], df["strike"],
                                    df["T"], df["implied_vol"], r)
        report(df, f"r={r:.3f}")


if __name__ == "__main__":
    main()
