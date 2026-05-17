"""Bulk-build MenthorQ-style gamma levels for QQQ and NDX, mapped to NQ.

Reuses the algorithm from menthorq_style_levels.py but iterates over the full
ThetaData archive and outputs ONE parquet with all 24 levels per day in NQ
price space (12 from QQQ, 12 from NDX).

Methodology (MenthorQ docs):
  - Window: 1D Expected Move = spot * ATM_IV * sqrt(1/252)
  - Call Resistance: max calls-only gamma at strike >= spot
  - Put Support:     max puts-only  gamma at strike <  spot
  - GEX 1..10:       0-1 DTE chain only, top by |Net GEX|/max + |Net DEX|/max,
                     excluding CR, PS, HVL, HVL_0DTE strikes

NQ mapping:
  - QQQ ratio = NQ_settle / QQQ_spot, computed per day (NQ price at 17:15 ET)
  - NDX basis = NQ_settle - NDX_spot, computed per day

Output: D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet
"""

from __future__ import annotations

import datetime as dt
import sys
import time
import traceback
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")

sys.path.insert(0, str(Path(__file__).parent))
import menthorq_style_levels as mq  # noqa: E402

QQQ_ROOT = Path("D:/trading_pythonbacktest_data/QQQ_thetadata")
NDX_ROOT = Path("D:/trading_pythonbacktest_data/NDX_thetadata")
NQ_1MIN  = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
OUT      = Path("D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet")

SETTLE_HOUR = 16
SETTLE_MIN  = 5   # close of 1-min bar (16:05 → 16:06) — matches old signal parquet's NQ snapshot


def load_nq_settles() -> dict[dt.date, float]:
    """Map date -> NQ price at 16:05 ET on that date.

    This selects the close of the 1-min bar opening at 16:04 ET (i.e., the bar
    that closes at 16:05 ET) — equivalent to the close of the 5-min bar opening
    at 16:00 ET (16:00 → 16:05 window). This matches the original B2 signal
    generation methodology (which produced the `entry_signal_trades.parquet`)."""
    nq = pd.read_parquet(NQ_1MIN)
    idx = nq.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    nq.index = idx.tz_convert("America/New_York")
    nq = nq.sort_index()
    mask = (nq.index.hour == SETTLE_HOUR) & (nq.index.minute == SETTLE_MIN)
    seq = nq.loc[mask]["close"]
    return {ts.date(): float(p) for ts, p in seq.items()}


def date_dirs(root: Path) -> set[dt.date]:
    out = set()
    for p in root.iterdir():
        if not p.is_dir():
            continue
        try:
            out.add(dt.date.fromisoformat(p.name))
        except ValueError:
            pass
    return out


def levels_to_nq(cr, ps, gex_list, scale_fn) -> dict:
    """Apply scale_fn (NQ mapping) to a level set. Returns dict with 12 NQ values."""
    out = {}
    out["cr"] = scale_fn(cr[0]) if cr else np.nan
    out["ps"] = scale_fn(ps[0]) if ps else np.nan
    for i in range(1, 11):
        if i <= len(gex_list):
            k = gex_list[i - 1][0]
            out[f"g{i}"] = scale_fn(k)
        else:
            out[f"g{i}"] = np.nan
    return out


def hvl_extended(by_strike, spot):
    """Same as mq.hvl_from_gex but no distance constraint — finds the
    cumulative-GEX sign-flip wherever it occurs. Returns (strike, dist_pct)
    or (None, None) if no flip exists (shouldn't happen on real chains)."""
    if by_strike.empty: return None, None
    s = by_strike.sort_values("strike").reset_index(drop=True).copy()
    s["cum"] = s["net_gex"].cumsum()
    flips = []
    for i in range(1, len(s)):
        K = float(s.iloc[i]["strike"])
        if (s.iloc[i-1]["cum"] > 0) != (s.iloc[i]["cum"] > 0):
            flips.append(K)
    if not flips:
        return None, None
    nearest = min(flips, key=lambda k: abs(k - spot))
    return nearest, abs(nearest - spot) / spot


def gamma_sign_at_spot(by_strike, spot):
    """Sign of cumulative GEX at strikes immediately around spot.
    Returns +1 if cum_GEX(spot) > 0 (positive gamma), -1 if < 0, 0 if zero/unknown."""
    if by_strike.empty: return 0
    s = by_strike.sort_values("strike").reset_index(drop=True).copy()
    s["cum"] = s["net_gex"].cumsum()
    s["dist"] = (s["strike"] - spot).abs()
    nearest = s.iloc[s["dist"].idxmin()]
    cum = nearest["cum"]
    if cum > 0: return 1
    if cum < 0: return -1
    return 0


def process_one(date: dt.date, nq_settle: float, qqq_hvl_dte_filter):
    """Compute MenthorQ levels mapped to NQ.

    qqq_hvl_dte_filter : (lo, hi) DTE filter to use for QQQ HVL 0DTE,
        computed by caller from next-trading-day perspective so the saved
        HVL applies to NTD's session (lookahead-free for live trading).
    """
    # QQQ — main (0-45 DTE) for CR, PS, IV/EM, full-chain HVL
    q_main, q_spot, q_iv = mq.qqq_strikes(date, (0, 45))
    q_em = mq.expected_move(q_spot, q_iv)

    # QQQ — DTE filter for "HVL 0DTE valid for next trading day"
    q_0dte, _, _ = mq.qqq_strikes(date, qqq_hvl_dte_filter)

    # HVL variants
    q_hvl   = mq.hvl_from_gex(q_main, q_spot)        # local full chain (±5% of spot)
    q_hvl_0 = mq.hvl_from_gex(q_0dte, q_spot)        # local 0-1 DTE-on-NTD chain
    q_hvl_ext, q_hvl_dist = hvl_extended(q_main, q_spot)   # full chain, no distance constraint
    q_gamma_sign = gamma_sign_at_spot(q_main, q_spot)      # sign of cum_GEX at spot

    # LEGACY MODE: matches original build_nq_levels.py methodology used to generate
    # entry_signal_trades.parquet. CR/PS = max/min net_gex, GEX 1..10 by |net_gex|
    # with NO exclusions, NO EM window restriction.
    q_cr, q_ps, q_gex = mq.menthorq_levels(q_0dte, q_spot, q_em, legacy_mode=True)
    # All-expiry variant kept for reference (still legacy ranking)
    _, _, q_gex_alldte = mq.menthorq_levels(q_main, q_spot, q_em, legacy_mode=True)

    # NDX — per user's documented methodology:
    #   - CR, PS, GEX 1..10 ALL use FULL CHAIN (0-45 DTE)
    #   - User: "NDX GEX 1-10 was all expiry because 0-1 DTE NDX options didn't exist
    #     reliably across the full backtest period (2020-2023)."
    n_main, n_spot, n_iv = mq.ndx_strikes(date, (0, 45))
    n_em = mq.expected_move(n_spot, n_iv)
    n_hvl   = mq.hvl_from_gex(n_main, n_spot)        # used internally for exclusion
    # LEGACY MODE for NDX too — matches build_ndx_levels.py methodology
    n_cr, n_ps, n_gex = mq.menthorq_levels(n_main, n_spot, n_em, legacy_mode=True)

    qqq_ratio = nq_settle / q_spot if q_spot and q_spot > 0 else np.nan
    ndx_basis = nq_settle - n_spot if n_spot and np.isfinite(n_spot) else np.nan

    q_nq = levels_to_nq(q_cr, q_ps, q_gex, lambda x: x * qqq_ratio)
    n_nq = levels_to_nq(n_cr, n_ps, n_gex, lambda x: x + ndx_basis)

    row = {"date": date,
           "qqq_spot": q_spot, "ndx_spot": n_spot, "nq_settle": nq_settle,
           "qqq_ratio": qqq_ratio, "ndx_basis": ndx_basis,
           "qqq_iv": q_iv, "ndx_iv": n_iv,
           "qqq_em": q_em, "ndx_em": n_em,
           # QQQ HVL columns (NDX HVLs intentionally dropped — expirations inconsistent pre-2023)
           "qqq_hvl_nq":          (q_hvl     * qqq_ratio) if q_hvl     is not None and np.isfinite(qqq_ratio) else np.nan,
           "qqq_hvl_0dte_nq":     (q_hvl_0   * qqq_ratio) if q_hvl_0   is not None and np.isfinite(qqq_ratio) else np.nan,
           "qqq_hvl_extended_nq": (q_hvl_ext * qqq_ratio) if q_hvl_ext is not None and np.isfinite(qqq_ratio) else np.nan,
           "qqq_hvl_dist_pct":    q_hvl_dist if q_hvl_dist is not None else np.nan,
           "qqq_gamma_sign":      int(q_gamma_sign),
           "qqq_hvl_dte_filter":  f"({qqq_hvl_dte_filter[0]},{qqq_hvl_dte_filter[1]})",
           }
    for k, v in q_nq.items():
        row[f"qqq_{k}_nq"] = v
    for k, v in n_nq.items():
        row[f"ndx_{k}_nq"] = v
    # NEW: 0-45 DTE (all-expiry) GEX levels for QQQ, mapped to NQ
    for i in range(1, 11):
        if i <= len(q_gex_alldte) and np.isfinite(qqq_ratio):
            row[f"qqq_g{i}_alldte_nq"] = q_gex_alldte[i - 1][0] * qqq_ratio
        else:
            row[f"qqq_g{i}_alldte_nq"] = np.nan
    return row


def main():
    print("loading NQ 17:15 settles...")
    settles = load_nq_settles()
    print(f"  settles available for {len(settles)} dates  "
          f"({min(settles)} -> {max(settles)})")

    qqq_avail = date_dirs(QQQ_ROOT)
    ndx_avail = date_dirs(NDX_ROOT)
    settle_dates = set(settles.keys())
    common = sorted(qqq_avail & ndx_avail & settle_dates)
    print(f"  common dates (QQQ AND NDX AND NQ settles): {len(common)}")
    print(f"  range: {common[0]} -> {common[-1]}")

    # Build "next trading day" lookup so QQQ HVL 0DTE filter targets NTD's session.
    # NTD is the next trading day with available data; on Friday EOD that's Monday.
    common_set = set(common)
    sorted_common = sorted(common_set)
    next_td = {}
    for i, d in enumerate(sorted_common[:-1]):
        next_td[d] = sorted_common[i + 1]
    # last day has no next trading day in the dataset; default to (0, 1) DTE filter
    next_td[sorted_common[-1]] = sorted_common[-1] + dt.timedelta(days=1)

    rows = []
    skipped = []
    t0 = time.time()
    for i, d in enumerate(common, 1):
        # Compute next-trading-day DTE filter for QQQ HVL 0DTE
        ntd = next_td.get(d, d + dt.timedelta(days=1))
        shift = (ntd - d).days
        qqq_hvl_filter = (shift, shift + 1)
        try:
            row = process_one(d, settles[d], qqq_hvl_filter)
            rows.append(row)
        except Exception as e:
            skipped.append((d, str(e)[:80]))
        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(common) - i) / rate
            print(f"  {i}/{len(common)}  elapsed={elapsed:.0f}s  rate={rate:.1f}/s  "
                  f"eta={eta:.0f}s  rows={len(rows)}  skipped={len(skipped)}")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df.to_parquet(OUT, compression="zstd", index=False)
    print(f"\nwrote {OUT}  ({len(df)} rows, {len(df.columns)} cols)")
    if skipped:
        print(f"\nskipped {len(skipped)} dates:")
        for d, msg in skipped[:20]:
            print(f"  {d}: {msg}")


if __name__ == "__main__":
    sys.exit(main())
