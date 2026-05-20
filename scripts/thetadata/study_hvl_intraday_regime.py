"""Intraday regime tracking based on the prior EOD cumulative-GEX curve.

For each trading day D:
  1. Load prior trading day P's QQQ greeks + OI parquets, filter to expirations
     on D and D+1, compute the cum_gex(K) curve for the 0-1 DTE chain.
  2. Same for NDX with parity + IV inversion.
  3. Pull intraday NQ prices at 5-min bars from 9:30 to 17:00 ET.
  4. For each bar, look up cum_gex(current_qqq_or_ndx_strike) on the static
     curve and classify regime at that minute.
  5. Aggregate per-day: time spent in each regime, regime at open/close,
     number of regime flips, transition timing.

Output: D:/trading_pythonbacktest_data/QQQ_thetadata/study_hvl0dte_intraday_regime.parquet
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

QQQ_ROOT = Path("D:/trading_pythonbacktest_data/QQQ_thetadata")
NDX_ROOT = Path("D:/trading_pythonbacktest_data/NDX_thetadata")
NQ_1MIN  = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
NQ_15MIN = Path("D:/trading_pythonbacktest_data/15min_bars.parquet")
OUT      = QQQ_ROOT / "study_hvl0dte_intraday_regime.parquet"

R = 0.05; Q_NDX = 0.006

# Intraday bar grid (ET): 5-min from 9:30 to 17:00
BAR_TIMES = []
t = pd.Timestamp("2000-01-01 09:30:00")
end = pd.Timestamp("2000-01-01 17:00:00")
while t <= end:
    BAR_TIMES.append((t.hour, t.minute))
    t = t + pd.Timedelta(minutes=5)


# --------------- BS helpers (NDX) ---------------

def _d1(S,K,T,sigma,q): return (np.log(S/K)+(R-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
def bs_call(S,K,T,sigma,q):
    d1=_d1(S,K,T,sigma,q); d2=d1-sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1)-K*np.exp(-R*T)*norm.cdf(d2)
def bs_put(S,K,T,sigma,q):
    d1=_d1(S,K,T,sigma,q); d2=d1-sigma*np.sqrt(T)
    return K*np.exp(-R*T)*norm.cdf(-d2)-S*np.exp(-q*T)*norm.cdf(-d1)
def bs_vega(S,K,T,sigma,q):
    return S*np.exp(-q*T)*norm.pdf(_d1(S,K,T,sigma,q))*np.sqrt(T)
def bs_gamma(S,K,T,sigma,q):
    return np.exp(-q*T)*norm.pdf(_d1(S,K,T,sigma,q))/(S*sigma*np.sqrt(T))

def iv_vec(price,S,K,T,is_call,q):
    sigma = np.full_like(price, 0.30, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(15):
            bp = np.where(is_call, bs_call(S,K,T,sigma,q), bs_put(S,K,T,sigma,q))
            d = bp-price; v = bs_vega(S,K,T,sigma,q)
            step = np.divide(d, v, out=np.zeros_like(d), where=(v>1e-8))
            sigma = np.clip(sigma-step, 1e-4, 5.0)
        bp = np.where(is_call, bs_call(S,K,T,sigma,q), bs_put(S,K,T,sigma,q))
        bad = (np.abs(bp-price) > np.maximum(0.10, 0.01*price))
    sigma[bad] = np.nan
    return sigma


# --------------- Build cum_gex curves ---------------

def qqq_cum_gex_curve(prev_date: dt.date, today: dt.date) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Returns (strikes, cum_gex, qqq_spot) — strikes sorted ascending, cum_gex aligned."""
    g_path = QQQ_ROOT / prev_date.isoformat() / "greeks_eod.parquet"
    o_path = QQQ_ROOT / prev_date.isoformat() / "open_interest.parquet"
    if not g_path.exists() or not o_path.exists(): return None
    g = pd.read_parquet(g_path); o = pd.read_parquet(o_path)
    g["expiration"] = pd.to_datetime(g["expiration"])
    o["expiration"] = pd.to_datetime(o["expiration"])
    end = today + dt.timedelta(days=2)
    g = g[(g["expiration"].dt.date >= today) & (g["expiration"].dt.date <= end)]
    if g.empty: return None
    chain = g.merge(o[["strike","right","expiration","open_interest"]],
                    on=["strike","right","expiration"], how="left")
    spot = float(chain["underlying_price"].iloc[0])
    chain["signed_gex"] = chain["gamma"]*chain["open_interest"].fillna(0)*100*spot**2
    chain.loc[chain["right"].str.upper()=="PUT","signed_gex"] *= -1
    by = (chain.groupby("strike")["signed_gex"].sum()
          .reset_index().rename(columns={"signed_gex":"net_gex"}).sort_values("strike"))
    strikes = by["strike"].values
    cum = np.cumsum(by["net_gex"].values)
    return strikes, cum, spot


def ndx_cum_gex_curve(prev_date: dt.date, today: dt.date) -> tuple[np.ndarray, np.ndarray, float] | None:
    eod_path = NDX_ROOT / prev_date.isoformat() / "eod.parquet"
    oi_path  = NDX_ROOT / prev_date.isoformat() / "oi.parquet"
    if not eod_path.exists() or not oi_path.exists(): return None
    eod_full = pd.read_parquet(eod_path); oi = pd.read_parquet(oi_path)
    eod_full["expiration"] = pd.to_datetime(eod_full["expiration"])
    oi["expiration"] = pd.to_datetime(oi["expiration"])
    end = today + dt.timedelta(days=2)
    eod = eod_full[(eod_full["expiration"].dt.date >= today) & (eod_full["expiration"].dt.date <= end)]
    if eod.empty: return None

    # Parity spot from full chain (short-DTE pairs)
    snap = pd.to_datetime(prev_date)
    df = eod_full[(eod_full["bid"]>0) & (eod_full["ask"]>0)].copy()
    df["mid"] = (df["bid"]+df["ask"])/2
    df["dte"] = (df["expiration"]-snap).dt.days
    df = df[(df["dte"]>0) & (df["dte"]<=36)]
    calls = df[df["right"].str.upper()=="CALL"][["root","strike","expiration","dte","mid"]].rename(columns={"mid":"C"})
    puts  = df[df["right"].str.upper()=="PUT"][["root","strike","expiration","dte","mid"]].rename(columns={"mid":"P"})
    pairs = calls.merge(puts, on=["root","strike","expiration","dte"])
    if pairs.empty: return None
    T = pairs["dte"].values/365.25
    S = ((pairs["C"]-pairs["P"])*np.exp(R*T)+pairs["strike"])/np.exp(-Q_NDX*T)
    spot = float(np.median(S))

    chain = eod.merge(oi[["root","strike","right","expiration","open_interest"]],
                      on=["root","strike","right","expiration"], how="left")
    chain["mid"] = (chain["bid"]+chain["ask"])/2
    chain = chain[(chain["bid"]>0) & (chain["ask"]>0) & (chain["mid"]>0)]
    chain["dte"] = (chain["expiration"]-snap).dt.days
    if chain.empty: return None
    T = chain["dte"].values/365.25
    K = chain["strike"].values.astype(float)
    P = chain["mid"].values.astype(float)
    is_call = (chain["right"].str.upper()=="CALL").values
    S_arr = np.full_like(K, spot)
    chain["iv"] = iv_vec(P, S_arr, K, T, is_call, Q_NDX)
    chain = chain.dropna(subset=["iv"])
    if chain.empty: return None
    chain["gamma"] = bs_gamma(spot, chain["strike"].values,
                              chain["dte"].values/365.25, chain["iv"].values, Q_NDX)
    chain["signed_gex"] = chain["gamma"]*chain["open_interest"].fillna(0)*100*spot**2
    chain.loc[chain["right"].str.upper()=="PUT","signed_gex"] *= -1
    by = (chain.groupby("strike")["signed_gex"].sum()
          .reset_index().rename(columns={"signed_gex":"net_gex"}).sort_values("strike"))
    strikes = by["strike"].values
    cum = np.cumsum(by["net_gex"].values)
    return strikes, cum, spot


# --------------- NQ price loader ---------------

def load_nq() -> pd.Series:
    parts = []
    if NQ_1MIN.exists():
        m1 = pd.read_parquet(NQ_1MIN)
        idx = m1.index
        if idx.tz is None: idx = idx.tz_localize("UTC")
        m1.index = idx.tz_convert("America/New_York")
        parts.append(m1["close"].rename("nq"))
    if NQ_15MIN.exists():
        m15 = pd.read_parquet(NQ_15MIN)
        idx = m15.index
        if idx.tz is None: idx = idx.tz_localize("America/New_York")
        else: idx = idx.tz_convert("America/New_York")
        m15.index = idx
        parts.append(m15["close"].rename("nq"))
    nq = pd.concat(parts).sort_index()
    nq = nq[~nq.index.duplicated(keep="first")]
    return nq


def nq_at_each_bar(nq: pd.Series, date: dt.date) -> pd.Series:
    """Return NQ price at each 5-min ET bar from 9:30 to 17:00 on `date`.
    Uses backfill (most recent prior tick within 5 min)."""
    targets = [pd.Timestamp(year=date.year, month=date.month, day=date.day,
                            hour=h, minute=m, tz="America/New_York")
               for (h, m) in BAR_TIMES]
    out = []
    for t in targets:
        win = nq.loc[t - pd.Timedelta(minutes=5):t]
        out.append(float(win.iloc[-1]) if not win.empty else np.nan)
    return pd.Series(out, index=targets)


# --------------- Per-day intraday regime ---------------

def regime_label(cum_gex_at_price: float) -> str:
    if cum_gex_at_price > 0: return "pos"
    if cum_gex_at_price < 0: return "neg"
    return "zero"


def daily_regime_stats(strikes: np.ndarray, cum: np.ndarray,
                       prices_in_strike_units: pd.Series) -> dict:
    """Given the static cum_gex curve and a sequence of intraday prices in
    the curve's strike units, return time-in-regime stats plus regime tags
    at hourly ET checkpoints."""
    valid = prices_in_strike_units.dropna()
    if valid.empty: return {}
    cum_at = np.interp(valid.values, strikes, cum,
                       left=cum[0], right=cum[-1])

    n = len(cum_at)
    pos_mask = cum_at > 0
    neg_mask = cum_at < 0
    n_pos = int(pos_mask.sum()); n_neg = int(neg_mask.sum())

    sign = np.sign(cum_at)
    flips = int(np.sum(np.diff(sign) != 0))

    out = {
        "n_bars":        n,
        "pct_pos":       n_pos / n,
        "pct_neg":       n_neg / n,
        "regime_open":   "pos" if cum_at[0] > 0 else ("neg" if cum_at[0] < 0 else "zero"),
        "regime_close":  "pos" if cum_at[-1] > 0 else ("neg" if cum_at[-1] < 0 else "zero"),
        "n_flips":       flips,
        "min_cum":       float(cum_at.min()),
        "max_cum":       float(cum_at.max()),
        "mean_cum":      float(cum_at.mean()),
    }

    # Hourly checkpoint regime tags: 10:00, 11:00, 12:00, 13:00, 14:00, 15:00 ET
    bar_index = {(t.hour, t.minute): i for i, t in enumerate(valid.index)}
    for h in [10, 11, 12, 13, 14, 15]:
        idx = bar_index.get((h, 0))
        if idx is None:
            out[f"regime_at_{h:02d}"] = None
        else:
            v = cum_at[idx]
            out[f"regime_at_{h:02d}"] = "pos" if v > 0 else ("neg" if v < 0 else "zero")
    return out


# --------------- Driver ---------------

def trading_dirs() -> list[dt.date]:
    out = []
    for p in QQQ_ROOT.iterdir():
        if not (p.is_dir() and len(p.name)==10 and p.name[4]=='-'): continue
        try:
            d = dt.date.fromisoformat(p.name)
            if (NDX_ROOT / p.name / "eod.parquet").exists():
                out.append(d)
        except: pass
    return sorted(out)


def main():
    print("loading NQ price series...")
    nq = load_nq()
    print(f"  NQ range: {nq.index.min()} -> {nq.index.max()}\n")

    days = trading_dirs()
    print(f"days available: {len(days)}")

    rows = []
    t0 = time.time()
    for i, d in enumerate(days):
        if i == 0: continue
        prev = days[i-1]

        # Build curves
        try:
            qcurve = qqq_cum_gex_curve(prev, d)
            ncurve = ndx_cum_gex_curve(prev, d)
        except Exception as e:
            print(f"  {d}: ERROR build curve {type(e).__name__}: {e}")
            continue

        # Intraday NQ prices at 5-min bars
        prices_nq = nq_at_each_bar(nq, d)
        if prices_nq.dropna().empty:
            continue

        # We need to convert NQ to QQQ-strike or NDX-strike units to look up on the curve
        # Use prev day's settle to derive ratio/basis (ratio = nq_settle / qqq_settle)
        nq_prev_close = nq.loc[(nq.index.date == prev) &
                               (nq.index.hour < 18)]
        if nq_prev_close.empty:
            qqq_ratio = np.nan; ndx_basis = np.nan
        else:
            nq_prev = float(nq_prev_close.iloc[-1])
            qqq_ratio = nq_prev / qcurve[2] if qcurve else np.nan
            ndx_basis = nq_prev - ncurve[2] if ncurve else np.nan

        row = {"date": d}
        row["nq_open_930"]  = float(prices_nq.iloc[0])  if not pd.isna(prices_nq.iloc[0])  else np.nan
        row["nq_close_500"] = float(prices_nq.iloc[-1]) if not pd.isna(prices_nq.iloc[-1]) else np.nan

        # QQQ regime stats
        if qcurve and np.isfinite(qqq_ratio):
            qqq_prices = prices_nq / qqq_ratio  # NQ -> QQQ strike units
            qstats = daily_regime_stats(qcurve[0], qcurve[1], qqq_prices)
            for k, v in qstats.items():
                row[f"qqq_{k}"] = v

        # NDX regime stats
        if ncurve and np.isfinite(ndx_basis):
            ndx_prices = prices_nq - ndx_basis  # NQ -> NDX strike units
            nstats = daily_regime_stats(ncurve[0], ncurve[1], ndx_prices)
            for k, v in nstats.items():
                row[f"ndx_{k}"] = v

        rows.append(row)
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(days)}  elapsed={time.time()-t0:.0f}s  rows={len(rows)}")

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["nq_open_930","nq_close_500"])
    df["ret"] = (df["nq_close_500"] - df["nq_open_930"]) / df["nq_open_930"]
    df["abs_ret"] = df["ret"].abs()
    df.to_parquet(OUT, compression="zstd", index=False)
    print(f"\nwrote {OUT}  ({len(df)} rows)")


if __name__ == "__main__":
    sys.exit(main())
