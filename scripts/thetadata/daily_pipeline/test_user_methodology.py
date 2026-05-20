"""Test the user's claimed original methodology against the signal parquet.

User's claimed methodology:
  - NQ settle at 16:00 ET (current matches)
  - HVL from FULL expiry chain (current uses dynamic near-DTE for HVL_0DTE col)
  - GEX ranking: combined |Net GEX|/max + |Net DEX|/max (current matches)
  - QQQ GEX 1-10: 0-1 DTE (current matches)
  - NDX GEX 1-10: FULL expiry (current uses 0-1 DTE — DIVERGES)

Test: compute levels for 2024-09-25 with user's methodology, compare to signal
parquet's near_level values for that date.

Also test alternative (NDX 0-1 DTE) just to be thorough.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import menthorq_style_levels as mq
import build_menthorq_levels_bulk as bulk

TEST_DATE = dt.date(2024, 9, 25)


def load_signal_near_levels(date: dt.date) -> set[float]:
    sig_path = Path("C:/trading/nqorderflowbacktester/scripts/overnight range strat/scripts/parquets/entry_signal_trades.parquet")
    df = pd.read_parquet(sig_path)
    df_d = df[pd.to_datetime(df["date"]).dt.date == date]
    return set(df_d["near_level"].dropna().values)


def compute_user_methodology(date: dt.date, ndx_dte_filter: tuple[int, int] = (0, 45)) -> dict:
    """Compute levels per user's stated methodology.

    ndx_dte_filter: (0,45) for user's claim of NDX all-expiry,
                    or (0,1) to test if NDX was actually 0-1 DTE.
    """
    settles = bulk.load_nq_settles()
    if date not in settles:
        raise ValueError(f"No NQ settle for {date}")
    nq_settle = settles[date]

    # QQQ — main = full chain (0-45)
    q_main, q_spot, q_iv = mq.qqq_strikes(date, (0, 45))
    q_em = mq.expected_move(q_spot, q_iv)
    qqq_ratio = nq_settle / q_spot

    # HVL: FULL chain (user's method)
    q_hvl = mq.hvl_from_gex(q_main, q_spot)

    # CR, PS: full chain
    q_cr, q_ps, _ = mq.menthorq_levels(q_main, q_spot, q_em)

    # QQQ GEX 1-10: 0-1 DTE
    q_0dte, _, _ = mq.qqq_strikes(date, (0, 1))
    _, _, q_gex = mq.menthorq_levels(q_0dte, q_spot, q_em,
                                       exclude_extra={q_hvl,
                                                       q_cr[0] if q_cr else None,
                                                       q_ps[0] if q_ps else None})

    # NDX
    n_main, n_spot, n_iv = mq.ndx_strikes(date, (0, 45))
    n_em = mq.expected_move(n_spot, n_iv)
    ndx_basis = nq_settle - n_spot
    # CR, PS: full chain
    n_cr, n_ps, _ = mq.menthorq_levels(n_main, n_spot, n_em)
    # GEX 1-10: per ndx_dte_filter (user says full chain, but we also test 0-1)
    n_chain_for_gex, _, _ = mq.ndx_strikes(date, ndx_dte_filter)
    _, _, n_gex = mq.menthorq_levels(n_chain_for_gex, n_spot, n_em,
                                       exclude_extra={n_cr[0] if n_cr else None,
                                                       n_ps[0] if n_ps else None})

    # Map to NQ
    levels = {}
    levels["qqq_hvl"] = q_hvl * qqq_ratio if q_hvl else None
    levels["qqq_cr"] = q_cr[0] * qqq_ratio if q_cr else None
    levels["qqq_ps"] = q_ps[0] * qqq_ratio if q_ps else None
    for i, item in enumerate(q_gex[:10], 1):
        strike = item[0]
        levels[f"qqq_g{i}"] = strike * qqq_ratio
    levels["ndx_cr"] = n_cr[0] + ndx_basis if n_cr else None
    levels["ndx_ps"] = n_ps[0] + ndx_basis if n_ps else None
    for i, item in enumerate(n_gex[:10], 1):
        strike = item[0]
        levels[f"ndx_g{i}"] = strike + ndx_basis
    return {"qqq_spot": q_spot, "ndx_spot": n_spot, "qqq_ratio": qqq_ratio,
            "ndx_basis": ndx_basis, "levels": levels}


def main():
    print(f"Testing {TEST_DATE}")
    print("=" * 70)
    signal_levels = load_signal_near_levels(TEST_DATE)
    print(f"\nSignal parquet near_level values for {TEST_DATE}:")
    for v in sorted(signal_levels):
        print(f"  {v:.4f}")

    print()
    for label, ndx_dte in [
        ("User method (NDX ALL expiry)", (0, 45)),
        ("Alternative (NDX 0-1 DTE)",     (0, 1)),
    ]:
        print(f"\n--- {label} ---")
        try:
            result = compute_user_methodology(TEST_DATE, ndx_dte_filter=ndx_dte)
            print(f"  qqq_ratio={result['qqq_ratio']:.4f}  ndx_basis={result['ndx_basis']:.2f}")
            computed_set = set()
            for name, v in result["levels"].items():
                if v is None or pd.isna(v):
                    continue
                computed_set.add(round(v, 4))
                marker = "  <-- MATCHES SIGNAL" if any(abs(v - s) < 0.5 for s in signal_levels) else ""
                print(f"    {name:>10}: {v:>15,.4f}{marker}")
            matched = sum(1 for v in computed_set if any(abs(v - s) < 0.5 for s in signal_levels))
            print(f"  Matched {matched} of {len(signal_levels)} signal near_levels")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
