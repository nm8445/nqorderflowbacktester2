"""Build MenthorQ-style intraday gamma levels for QQQ (0-1 DTE chain).

For each trading day:
  - 3 pre-market rows  (08:00, 09:00, 09:30)  show prev-day EOD anchor
  - 13 RTH rows        (09:50, 10:20, ..., 15:50)  live intraday recompute

Methodology:
  - Chain: 0-1 DTE expirations only (today's expiry + next available expiry)
  - Pre-market anchor: prev-day EOD greeks/OI, filtered to {today, next_td} expirations,
    levels computed once and carried into 08:00/09:00/09:30 rows
  - Intraday: ThetaData first_order greeks @ 10m bars, BS gamma local,
    multiplied by prev-day EOD OI (held fixed all day)
  - HVL: +/-5% of live spot gate; carry-forward indefinitely until next in-band flip
  - CR / PS / GEX 1-10: per MenthorQ standard (1D EM window, no extra gate)

NQ conversion: per-day settle ratio from menthorq_levels_nq.parquet.

Output:  D:/trading_pythonbacktest_data/qqq_intraday_levels.parquet
Skipped: D:/trading_pythonbacktest_data/qqq_intraday_levels_skipped.parquet

Resumable: dates already in rollup are skipped on rerun.

Usage:
  python build_qqq_intraday_levels.py                                  # full backfill
  python build_qqq_intraday_levels.py --start 2024-12-01 --end 2024-12-31
  python build_qqq_intraday_levels.py --workers 4
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import sys
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

QQQ_ROOT = Path("D:/trading_pythonbacktest_data/QQQ_thetadata")
LEVELS_NQ = Path("D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet")
ROLLUP    = Path("D:/trading_pythonbacktest_data/qqq_intraday_levels.parquet")
SKIPPED   = Path("D:/trading_pythonbacktest_data/qqq_intraday_levels_skipped.parquet")
THETA_BASE = "http://127.0.0.1:25503/v3"
THETA_TIMEOUT = 300.0       # bumped 120 -> 300 (some chains need it)
THETA_MAX_RETRIES = 1       # 1 retry on timeout with fresh client
THETA_RETRY_BACKOFF = 5.0   # seconds before retry

R = 0.05            # risk-free rate
Q_QQQ = 0.005       # dividend yield
INTERVAL = "10m"    # ThetaData bar interval
HVL_GATE_PCT = 0.05 # +/-5% of live spot

PREMARKET_TIMES = ["08:00", "09:00", "09:30"]
RTH_SNAPSHOTS = ["09:50", "10:20", "10:50", "11:20", "11:50",
                 "12:20", "12:50", "13:20", "13:50",
                 "14:20", "14:50", "15:20", "15:50"]
ALL_TIMES = PREMARKET_TIMES + RTH_SNAPSHOTS  # 16 rows/day


# --------------------------------------------------------------------------
# BS primitives
# --------------------------------------------------------------------------

def _d1(S, K, T, sigma, q=Q_QQQ):
    return (np.log(S / K) + (R - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def bs_gamma(S, K, T, sigma, q=Q_QQQ):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.exp(-q * T) * norm.pdf(_d1(S, K, T, sigma, q)) / (S * sigma * np.sqrt(T))


def bs_call_delta(S, K, T, sigma, q=Q_QQQ):
    return np.exp(-q * T) * norm.cdf(_d1(S, K, T, sigma, q))


def bs_put_delta(S, K, T, sigma, q=Q_QQQ):
    return -np.exp(-q * T) * norm.cdf(-_d1(S, K, T, sigma, q))


# --------------------------------------------------------------------------
# Level extraction (1D EM window for CR/PS/GEX, +/-5% for HVL)
# --------------------------------------------------------------------------

def atm_iv_from_chain(chain: pd.DataFrame, spot: float, iv_col: str = "implied_vol") -> float:
    """Mean IV of 4 nearest strikes at the shortest non-zero DTE expiration."""
    sub = chain[chain["dte"] > 0].copy()
    if sub.empty:
        # fall back to 0DTE if no positive DTE
        sub = chain.copy()
    if sub.empty:
        return float("nan")
    min_dte = int(sub["dte"].min())
    sub = sub[sub["dte"] == min_dte].copy()
    sub["dist"] = (sub["strike"] - spot).abs()
    near = sub.nsmallest(4, "dist")
    return float(near[iv_col].mean())


def expected_move(spot: float, iv: float) -> float:
    if not np.isfinite(iv) or iv <= 0:
        return float("nan")
    return spot * iv * np.sqrt(1.0 / 252.0)


def aggregate_by_strike(chain: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Group chain rows into per-strike net_gex, net_dex, call_only_*, put_only_* columns."""
    if chain.empty:
        return pd.DataFrame(columns=["strike", "net_gex", "net_dex",
                                     "call_only_gex", "put_only_gex",
                                     "call_only_dex", "put_only_dex"])
    df = chain.copy()
    df["right_norm"] = df["right"].astype(str).str.upper().str[0]
    is_call = (df["right_norm"] == "C")
    df["gex_abs"]    = df["gamma"] * df["open_interest"].fillna(0) * 100 * spot ** 2
    df["signed_gex"] = np.where(is_call, df["gex_abs"], -df["gex_abs"])
    df["signed_dex"] = df["delta"] * df["open_interest"].fillna(0) * 100 * spot
    df["call_only_gex"] = np.where(is_call, df["gex_abs"], 0.0)
    df["put_only_gex"]  = np.where(is_call, 0.0,         df["gex_abs"])
    df["call_only_dex"] = np.where(is_call, df["signed_dex"], 0.0)
    df["put_only_dex"]  = np.where(is_call, 0.0,         df["signed_dex"])
    return (df.groupby("strike").agg(net_gex=("signed_gex", "sum"),
                                     net_dex=("signed_dex", "sum"),
                                     call_only_gex=("call_only_gex", "sum"),
                                     put_only_gex=("put_only_gex", "sum"),
                                     call_only_dex=("call_only_dex", "sum"),
                                     put_only_dex=("put_only_dex", "sum"))
              .reset_index()
              .sort_values("strike")
              .reset_index(drop=True))


def hvl_with_gate(by_strike: pd.DataFrame, spot: float,
                  gate_pct: float = HVL_GATE_PCT
                  ) -> tuple[float | None, float | None, int, int]:
    """Returns (hvl_gated, hvl_extended, n_total_flips, n_inband_flips).
    hvl_gated    = nearest in-band flip (or None if no flip within +/-gate_pct)
    hvl_extended = nearest flip across full chain regardless of distance (or None if no flips at all)
    """
    if by_strike.empty:
        return None, None, 0, 0
    s = by_strike.sort_values("strike").reset_index(drop=True).copy()
    s["cum"] = s["net_gex"].cumsum()
    band = spot * gate_pct
    flips_total = []
    flips_inband = []
    for i in range(1, len(s)):
        K = float(s.iloc[i]["strike"])
        if (s.iloc[i - 1]["cum"] > 0) != (s.iloc[i]["cum"] > 0):
            flips_total.append(K)
            if abs(K - spot) <= band:
                flips_inband.append(K)
    hvl_gated    = min(flips_inband, key=lambda k: abs(k - spot)) if flips_inband else None
    hvl_extended = min(flips_total,  key=lambda k: abs(k - spot)) if flips_total  else None
    return hvl_gated, hvl_extended, len(flips_total), len(flips_inband)


def cr_ps_gex(by_strike: pd.DataFrame, spot: float, em: float,
              hvl_strike: float | None) -> tuple[tuple | None, tuple | None, list]:
    """Returns (cr, ps, gex_top10).
    cr = (strike, call_only_gex_value)
    ps = (strike, -put_only_gex_value)   negative sign indicates puts
    gex_top10 = list of (strike, net_gex, net_dex, call_only_dex, put_only_dex)
    Uses 1D EM window. CR/PS strict (no extra gate beyond EM).
    """
    if by_strike.empty or not np.isfinite(em):
        return None, None, []
    win = by_strike[(by_strike["strike"] >= spot - em) &
                    (by_strike["strike"] <= spot + em)].copy()
    if win.empty:
        return None, None, []

    above = win[win["strike"] >= spot]
    below = win[win["strike"] <  spot]
    cr_row = above.loc[above["call_only_gex"].idxmax()] if not above.empty and above["call_only_gex"].max() > 0 else None
    ps_row = below.loc[below["put_only_gex"].idxmax()]  if not below.empty and below["put_only_gex"].max()  > 0 else None
    cr_strike = float(cr_row["strike"]) if cr_row is not None else None
    ps_strike = float(ps_row["strike"]) if ps_row is not None else None

    excluded = {cr_strike, ps_strike, hvl_strike}
    excluded.discard(None)
    rest = win[~win["strike"].isin(excluded)].copy()
    if rest.empty:
        gex_list = []
    else:
        rest["abs_gex"] = rest["net_gex"].abs()
        rest["abs_dex"] = rest["net_dex"].abs()
        max_gex = rest["abs_gex"].max() or 1.0
        max_dex = rest["abs_dex"].max() or 1.0
        rest["score"] = rest["abs_gex"] / max_gex + rest["abs_dex"] / max_dex
        gex_top = rest.sort_values("score", ascending=False).head(10)
        gex_list = list(gex_top[["strike", "net_gex", "net_dex",
                                  "call_only_dex", "put_only_dex"]]
                        .itertuples(index=False, name=None))

    cr = (cr_strike, float(cr_row["call_only_gex"])) if cr_row is not None else None
    ps = (ps_strike, -float(ps_row["put_only_gex"])) if ps_row is not None else None
    return cr, ps, gex_list


def gamma_sign_at_spot(by_strike: pd.DataFrame, spot: float) -> int:
    if by_strike.empty:
        return 0
    s = by_strike.sort_values("strike").reset_index(drop=True).copy()
    s["cum"] = s["net_gex"].cumsum()
    s["dist"] = (s["strike"] - spot).abs()
    nearest = s.iloc[s["dist"].idxmin()]
    cum = nearest["cum"]
    return 1 if cum > 0 else (-1 if cum < 0 else 0)


def cum_gex_at_spot(by_strike: pd.DataFrame, spot: float) -> float:
    if by_strike.empty:
        return float("nan")
    s = by_strike.sort_values("strike").reset_index(drop=True).copy()
    s["cum"] = s["net_gex"].cumsum()
    s["dist"] = (s["strike"] - spot).abs()
    return float(s.iloc[s["dist"].idxmin()]["cum"])


# --------------------------------------------------------------------------
# Trading-day utilities
# --------------------------------------------------------------------------

def trading_dates_available() -> list[dt.date]:
    """Dates with both QQQ_thetadata folder and a row in menthorq_levels_nq.parquet."""
    qqq = set()
    for p in QQQ_ROOT.iterdir():
        if not p.is_dir():
            continue
        try:
            qqq.add(dt.date.fromisoformat(p.name))
        except ValueError:
            pass
    if not LEVELS_NQ.exists():
        return sorted(qqq)
    mq = pd.read_parquet(LEVELS_NQ)
    mq_dates = set(pd.to_datetime(mq["date"]).dt.date.tolist())
    return sorted(qqq & mq_dates)


def next_trading_day(d: dt.date, all_dates: list[dt.date]) -> dt.date | None:
    """Next date in all_dates after d. None if d is the last."""
    for x in all_dates:
        if x > d:
            return x
    return None


def prior_trading_day(d: dt.date, all_dates: list[dt.date]) -> dt.date | None:
    prev = None
    for x in all_dates:
        if x < d:
            prev = x
        else:
            break
    return prev


# --------------------------------------------------------------------------
# Pre-market anchor: build levels from prev-day EOD greeks/OI
# --------------------------------------------------------------------------

DTE_WINDOW_DAYS = 4  # 0-1 DTE intent + slack for weekends/legacy weekly schedule


def select_target_expirations(available_exps: set, today: dt.date) -> list[dt.date]:
    """Pick expirations to represent the 0-1 DTE chain on `today`.
    Returns expirations in [today, today + DTE_WINDOW_DAYS], up to 2 nearest.
    Handles:
      - pre-2022 Mon/Wed/Fri-only weekly schedule (Tue/Thu fall back to next available)
      - Friday with no Sat/Sun expiry (next is Mon, 3 cal days out)
      - holiday-shortened weeks
    """
    candidates = sorted(e for e in available_exps
                        if today <= e <= today + dt.timedelta(days=DTE_WINDOW_DAYS))
    return candidates[:2]


def load_prev_day_chain(prev_d: dt.date, today: dt.date
                        ) -> tuple[pd.DataFrame, float, list[dt.date]]:
    """Load prev_d's greeks + OI, filter to nearest 0-1 DTE-eligible expirations,
    return (per-strike aggregated DataFrame, prev_spot, target_exps used)."""
    root = QQQ_ROOT / prev_d.isoformat()
    g = pd.read_parquet(root / "greeks_eod.parquet")
    o = pd.read_parquet(root / "open_interest.parquet")
    g["expiration"] = pd.to_datetime(g["expiration"]).dt.date
    o["expiration"] = pd.to_datetime(o["expiration"]).dt.date

    available = set(g["expiration"].unique())
    target_exps = select_target_expirations(available, today)
    if not target_exps:
        return pd.DataFrame(), float("nan"), []

    g = g[g["expiration"].isin(target_exps)].copy()
    o = o[o["expiration"].isin(target_exps)].copy()
    if g.empty:
        return pd.DataFrame(), float("nan"), []

    # Use prev-day spot as anchor for the pre-market display
    prev_spot = float(g["underlying_price"].iloc[0])

    # Merge OI onto greeks; normalize right column for merge
    g["right"] = g["right"].astype(str).str.upper()
    o["right"] = o["right"].astype(str).str.upper()
    merged = g.merge(o[["strike", "right", "expiration", "open_interest"]],
                     on=["strike", "right", "expiration"], how="left")
    merged["open_interest"] = merged["open_interest"].fillna(0)
    return merged, prev_spot, target_exps


# --------------------------------------------------------------------------
# ThetaData fetch
# --------------------------------------------------------------------------

def theta_get_ndjson(path: str, params: dict) -> pd.DataFrame:
    """GET ndjson with timeout + 1 retry on read timeout (fresh client per attempt)."""
    last_err = None
    for attempt in range(THETA_MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=THETA_TIMEOUT) as client:
                r = client.get(f"{THETA_BASE}{path}",
                               params={**params, "format": "ndjson"})
            if r.status_code == 472 or not r.text.strip():
                return pd.DataFrame()
            r.raise_for_status()
            return pd.read_json(io.StringIO(r.text), lines=True)
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as e:
            last_err = e
            if attempt < THETA_MAX_RETRIES:
                time.sleep(THETA_RETRY_BACKOFF)
                continue
            raise


def fetch_intraday_greeks(date: dt.date, expirations: list[dt.date]) -> pd.DataFrame:
    """Fetch first_order greeks @ 10-min bars for each expiration. Concat results."""
    parts = []
    for exp in expirations:
        df = theta_get_ndjson("/option/history/greeks/first_order", {
            "symbol": "QQQ",
            "expiration": exp.strftime("%Y%m%d"),
            "start_date": date.strftime("%Y%m%d"),
            "end_date":   date.strftime("%Y%m%d"),
            "interval":   INTERVAL,
        })
        if not df.empty:
            df["_exp"] = exp
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["expiration"] = pd.to_datetime(out["expiration"]).dt.date
    out["right"] = out["right"].astype(str).str.upper()
    return out


# --------------------------------------------------------------------------
# Per-snapshot level computation
# --------------------------------------------------------------------------

def levels_at_snapshot(chain: pd.DataFrame, prev_hvl: float | None,
                       prev_hvl_ts) -> dict:
    """Compute CR/PS/HVL/GEX at a single snapshot. Apply HVL carry-forward.
    `chain` must have columns: strike, right, dte, T_yrs, implied_vol,
                                underlying_price, gamma, delta, open_interest.
    `prev_hvl` / `prev_hvl_ts` carry the last live HVL value/time forward.
    Returns dict with all level fields + diagnostics + HVL source/last_live_ts.
    """
    if chain.empty:
        return {"hvl_source": "no_data", "_carried_hvl": prev_hvl, "_carried_ts": prev_hvl_ts}
    spot = float(chain["underlying_price"].iloc[0])

    by_strike = aggregate_by_strike(chain, spot)
    iv = atm_iv_from_chain(chain, spot, iv_col="implied_vol")
    em = expected_move(spot, iv)

    hvl_live, hvl_ext, n_flips_total, n_flips_inband = hvl_with_gate(by_strike, spot)
    if hvl_live is not None:
        hvl_used = hvl_live
        hvl_source = "live"
        hvl_last_live_ts = chain["timestamp"].iloc[0] if "timestamp" in chain.columns else None
    elif prev_hvl is not None:
        hvl_used = prev_hvl
        hvl_source = "carry"
        hvl_last_live_ts = prev_hvl_ts
    else:
        hvl_used = None
        hvl_source = "no_flip_ever"
        hvl_last_live_ts = None

    cr, ps, gex_list = cr_ps_gex(by_strike, spot, em, hvl_used)
    cgs = cum_gex_at_spot(by_strike, spot)
    gsign = gamma_sign_at_spot(by_strike, spot)

    out = {
        "spot_qqq": spot,
        "atm_iv": iv,
        "em_qqq": em,
        "hvl_qqq": hvl_used,
        "hvl_extended_qqq": hvl_ext,
        "hvl_source": hvl_source,
        "hvl_last_live_ts": hvl_last_live_ts,
        "hvl_n_flips_total": n_flips_total,
        "hvl_n_flips_inband": n_flips_inband,
        "cr_qqq": cr[0] if cr else None,
        "cr_call_only_gex": cr[1] if cr else None,
        "ps_qqq": ps[0] if ps else None,
        "ps_put_only_gex": ps[1] if ps else None,
        "cum_gex_at_spot": cgs,
        "gamma_sign": gsign,
        "n_strikes": int(len(by_strike)),
        "_carried_hvl": hvl_live if hvl_live is not None else prev_hvl,
        "_carried_ts": hvl_last_live_ts if hvl_live is not None else prev_hvl_ts,
    }

    # Top-10 GEX strikes with values, NaN-padded
    for i in range(1, 11):
        if i <= len(gex_list):
            k, gex, dex, c_dex, p_dex = gex_list[i - 1]
            out[f"gex{i}_qqq"]    = float(k)
            out[f"gex{i}_signed"] = float(gex)
            out[f"gex{i}_dex"]    = float(dex)
        else:
            out[f"gex{i}_qqq"]    = np.nan
            out[f"gex{i}_signed"] = np.nan
            out[f"gex{i}_dex"]    = np.nan
    return out


def add_nq_columns(row: dict, ratio: float) -> dict:
    """Multiply QQQ-space level columns by ratio to get NQ-equivalent."""
    out = dict(row)
    for k in ["spot", "hvl", "hvl_extended", "cr", "ps"]:
        q = out.get(f"{k}_qqq")
        out[f"{k}_nq"] = (q * ratio) if (q is not None and np.isfinite(q)) else np.nan
    for i in range(1, 11):
        q = out.get(f"gex{i}_qqq")
        out[f"gex{i}_nq"] = (q * ratio) if (q is not None and np.isfinite(q)) else np.nan
    return out


# --------------------------------------------------------------------------
# Per-day driver
# --------------------------------------------------------------------------

def process_one_day(date: dt.date, prev_d: dt.date, next_td: dt.date | None,
                    qqq_ratio: float) -> tuple[list[dict], str | None]:
    """Returns (rows, error_msg). rows is empty + error_msg set on failure."""
    try:
        # ----- 1. Pre-market anchor: prev-day EOD chain, expirations within
        # [today, today + DTE_WINDOW_DAYS]. Handles pre-2022 Mon/Wed/Fri schedule
        # and Friday->Monday weekend gaps automatically.
        prev_chain, prev_spot, target_exps = load_prev_day_chain(prev_d, date)
        if prev_chain.empty:
            return [], f"no prev-day expirations within {DTE_WINDOW_DAYS} cal days of {date}"

        prev_chain["dte"] = (pd.to_datetime(prev_chain["expiration"])
                              - pd.Timestamp(prev_d)).dt.days
        prev_chain_today = prev_chain.copy()
        # Use prev_chain's underlying_price as spot
        anchor = levels_at_snapshot(prev_chain_today, prev_hvl=None, prev_hvl_ts=None)
        if anchor.get("hvl_source") == "no_data":
            return [], "anchor produced no_data"

        # Track HVL carry-forward state
        carried_hvl = anchor.get("_carried_hvl")
        carried_ts  = anchor.get("_carried_ts")
        # Mark anchor source explicitly
        anchor["hvl_source"] = "eod_anchor" if anchor["hvl_qqq"] is not None else anchor["hvl_source"]
        anchor["hvl_last_live_ts"] = pd.Timestamp(prev_d)  # anchor is timestamped at prev EOD

        # Build pre-market rows: 08:00, 09:00, 09:30 — same anchor values
        rows = []
        for tstr in PREMARKET_TIMES:
            ts = pd.Timestamp.combine(date, dt.time.fromisoformat(tstr + ":00"))
            r = dict(anchor)
            r["date"] = date
            r["timestamp_et"] = ts
            r["snapshot_label"] = tstr
            r["session"] = "pre-market"
            r["qqq_ratio"] = qqq_ratio
            r = add_nq_columns(r, qqq_ratio)
            r.pop("_carried_hvl", None)
            r.pop("_carried_ts", None)
            rows.append(r)

        # ----- 2. Intraday: fetch ThetaData greeks for the same target expirations
        # selected from the prev-day chain. Pre-2022 era this may be a single
        # expiration (e.g. Wednesday) on a Tue/Thu where today itself isn't an expiry.
        greeks = fetch_intraday_greeks(date, target_exps)
        if greeks.empty:
            return [], "no intraday greeks"

        # Compute T per row using each option's expiration close (16:00 ET)
        exp_close = pd.to_datetime(greeks["expiration"]) + pd.Timedelta(hours=16)
        greeks["T_yrs"] = ((exp_close - greeks["timestamp"]).dt.total_seconds()
                            / (365.25 * 86400))
        valid = (greeks["T_yrs"] > 0) & (greeks["implied_vol"] > 0)
        greeks = greeks[valid].copy()
        if greeks.empty:
            return [], "no valid greeks rows after filter"

        # Compute BS gamma + delta locally
        S = greeks["underlying_price"].values
        K = greeks["strike"].values
        T = greeks["T_yrs"].values
        sig = greeks["implied_vol"].values
        is_call = (greeks["right"] == "CALL").values
        greeks["gamma"] = bs_gamma(S, K, T, sig)
        greeks["delta"] = np.where(is_call, bs_call_delta(S, K, T, sig),
                                    bs_put_delta(S, K, T, sig))
        greeks = greeks.dropna(subset=["gamma", "delta"])
        greeks = greeks[greeks["gamma"].abs() < 100]  # drop pathological extreme-OTM

        # ----- 3. Merge prev-day OI (from local archive) onto intraday greeks
        prev_oi = pd.read_parquet(QQQ_ROOT / prev_d.isoformat() / "open_interest.parquet")
        prev_oi["expiration"] = pd.to_datetime(prev_oi["expiration"]).dt.date
        prev_oi = prev_oi[prev_oi["expiration"].isin(target_exps)].copy()
        prev_oi["right"] = prev_oi["right"].astype(str).str.upper()
        chain = greeks.merge(
            prev_oi[["strike", "right", "expiration", "open_interest"]],
            on=["strike", "right", "expiration"], how="left",
        )
        chain["open_interest"] = chain["open_interest"].fillna(0)
        chain["dte"] = (pd.to_datetime(chain["expiration"])
                         - pd.Timestamp(date)).dt.days
        chain["hhmm"] = chain["timestamp"].dt.strftime("%H:%M")

        # ----- 4. Per-snapshot loop with HVL carry-forward
        for tstr in RTH_SNAPSHOTS:
            sub = chain[chain["hhmm"] == tstr]
            if sub.empty:
                # No bar at this time — emit a row with NaN + carry forward HVL
                ts = pd.Timestamp.combine(date, dt.time.fromisoformat(tstr + ":00"))
                r = {
                    "date": date,
                    "timestamp_et": ts,
                    "snapshot_label": tstr,
                    "session": "rth",
                    "qqq_ratio": qqq_ratio,
                    "spot_qqq": np.nan, "spot_nq": np.nan,
                    "atm_iv": np.nan, "em_qqq": np.nan,
                    "hvl_qqq": carried_hvl,
                    "hvl_nq": (carried_hvl * qqq_ratio) if carried_hvl is not None else np.nan,
                    "hvl_extended_qqq": np.nan, "hvl_extended_nq": np.nan,
                    "hvl_source": "carry" if carried_hvl is not None else "no_data",
                    "hvl_last_live_ts": carried_ts,
                    "hvl_n_flips_total": 0, "hvl_n_flips_inband": 0,
                    "cr_qqq": np.nan, "cr_nq": np.nan, "cr_call_only_gex": np.nan,
                    "ps_qqq": np.nan, "ps_nq": np.nan, "ps_put_only_gex": np.nan,
                    "cum_gex_at_spot": np.nan, "gamma_sign": 0, "n_strikes": 0,
                }
                for i in range(1, 11):
                    r[f"gex{i}_qqq"] = np.nan
                    r[f"gex{i}_signed"] = np.nan
                    r[f"gex{i}_dex"] = np.nan
                    r[f"gex{i}_nq"] = np.nan
                rows.append(r)
                continue
            res = levels_at_snapshot(sub, prev_hvl=carried_hvl, prev_hvl_ts=carried_ts)
            ts = pd.Timestamp.combine(date, dt.time.fromisoformat(tstr + ":00"))
            r = dict(res)
            r["date"] = date
            r["timestamp_et"] = ts
            r["snapshot_label"] = tstr
            r["session"] = "rth"
            r["qqq_ratio"] = qqq_ratio
            r = add_nq_columns(r, qqq_ratio)
            carried_hvl = r.pop("_carried_hvl", carried_hvl)
            carried_ts  = r.pop("_carried_ts", carried_ts)
            rows.append(r)

        return rows, None
    except Exception as e:
        return [], f"{type(e).__name__}: {str(e)[:200]}"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default=None, help="ISO date YYYY-MM-DD")
    ap.add_argument("--end",   type=str, default=None, help="ISO date YYYY-MM-DD")
    ap.add_argument("--workers", type=int, default=2,
                    help="ThreadPool workers (default 2; was 4, lowered to reduce ThetaData backpressure)")
    ap.add_argument("--retry-skipped", action="store_true",
                    help="Re-process dates currently in the skipped sidecar parquet")
    ap.add_argument("--rebuild", action="store_true",
                    help="Ignore existing rollup; rebuild from scratch")
    args = ap.parse_args()

    print("loading available trading dates...")
    all_dates = trading_dates_available()
    print(f"  {len(all_dates)} dates with QQQ_thetadata + menthorq_levels_nq.parquet entry")
    print(f"  range: {all_dates[0]} -> {all_dates[-1]}")

    start = dt.date.fromisoformat(args.start) if args.start else all_dates[0]
    end   = dt.date.fromisoformat(args.end)   if args.end   else all_dates[-1]
    target = [d for d in all_dates if start <= d <= end]
    print(f"  target window: {start} -> {end}  ({len(target)} dates)")

    # Load existing rollup for resume
    existing_done = set()
    if ROLLUP.exists() and not args.rebuild:
        prev = pd.read_parquet(ROLLUP)
        existing_done = set(pd.to_datetime(prev["date"]).dt.date.tolist())
        print(f"  existing rollup: {len(prev)} rows over {len(existing_done)} dates")
    else:
        prev = pd.DataFrame()

    if args.retry_skipped:
        if not SKIPPED.exists():
            print("  --retry-skipped given but no skipped sidecar found; nothing to retry")
            return
        sk_in = pd.read_parquet(SKIPPED)
        sk_in["date"] = pd.to_datetime(sk_in["date"]).dt.date
        retry_dates = set(sk_in["date"].tolist()) & set(target)
        todo = sorted(retry_dates - existing_done)
        print(f"  retry-skipped mode: {len(todo)} dates from sidecar within {start}->{end}")
    else:
        todo = [d for d in target if d not in existing_done]
        print(f"  to process: {len(todo)} new dates")
    if not todo:
        print("nothing to do.")
        return

    # Look up qqq_ratio per date from menthorq_levels_nq.parquet
    mq = pd.read_parquet(LEVELS_NQ)
    mq["date"] = pd.to_datetime(mq["date"]).dt.date
    ratio_map = dict(zip(mq["date"], mq["qqq_ratio"]))

    new_rows = []
    skipped = []
    succeeded_dates = set()

    def work(date):
        prev_d = prior_trading_day(date, all_dates)
        if prev_d is None:
            return date, [], "no prior trading day in dataset"
        next_td = next_trading_day(date, all_dates)
        ratio = ratio_map.get(date)
        if ratio is None or not np.isfinite(ratio):
            return date, [], f"no qqq_ratio for {date}"
        rows, err = process_one_day(date, prev_d, next_td, float(ratio))
        return date, rows, err

    t0 = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(work, d): d for d in todo}
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                date, rows, err = fut.result()
            except Exception as e:
                date, rows, err = d, [], f"future raise: {e}"
            completed += 1
            if err:
                skipped.append({"date": date, "reason": err})
                print(f"  [{completed:>4}/{len(todo)}]  SKIP {date}  {err[:80]}")
            else:
                new_rows.extend(rows)
                succeeded_dates.add(date)
                if completed % 10 == 0 or completed == len(todo):
                    rate = completed / max(time.time() - t0, 0.01)
                    eta = (len(todo) - completed) / rate
                    print(f"  [{completed:>4}/{len(todo)}]  ok    {date}  "
                          f"rate={rate:.2f}/s  eta={eta:.0f}s  rows={len(new_rows)}")

    if not new_rows and not skipped:
        print("no rows produced and no skips logged.")
        return

    # Concat with existing rollup
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # Convert timestamp_et to ET-aware once
        if "timestamp_et" in new_df.columns:
            try:
                new_df["timestamp_et"] = pd.to_datetime(new_df["timestamp_et"]).dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
            except Exception:
                pass
        if not prev.empty:
            new_df = pd.concat([prev, new_df], ignore_index=True)
        # Sort and dedupe
        new_df["date"] = pd.to_datetime(new_df["date"]).dt.date
        new_df = (new_df.sort_values(["date", "timestamp_et"])
                        .drop_duplicates(subset=["date", "snapshot_label"], keep="last")
                        .reset_index(drop=True))
        new_df.to_parquet(ROLLUP, compression="zstd", index=False)
        print(f"\nwrote {ROLLUP}  ({len(new_df)} total rows, {new_df['date'].nunique()} dates)")

    # Update skipped sidecar: remove succeeded, add new skips
    sk_existing = pd.DataFrame(columns=["date", "reason"])
    if SKIPPED.exists():
        sk_existing = pd.read_parquet(SKIPPED)
        sk_existing["date"] = pd.to_datetime(sk_existing["date"]).dt.date
    sk_existing = sk_existing[~sk_existing["date"].isin(succeeded_dates)]
    if skipped:
        sk_new = pd.DataFrame(skipped)
        sk_new["date"] = pd.to_datetime(sk_new["date"]).dt.date
        sk_combined = (pd.concat([sk_existing, sk_new], ignore_index=True)
                         .drop_duplicates(subset=["date"], keep="last"))
    else:
        sk_combined = sk_existing
    if not sk_combined.empty:
        sk_combined.to_parquet(SKIPPED, compression="zstd", index=False)
        print(f"wrote {SKIPPED}  ({len(sk_combined)} dates remaining in sidecar)")
    elif SKIPPED.exists():
        SKIPPED.unlink()
        print(f"all sidecar skips resolved; deleted {SKIPPED}")

    print(f"\nelapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
