"""Study: does price tend to mean-revert (close > open) when above HVL 0DTE at 9:30 AM?

For each trading day D:
  - Prev day P's QQQ greeks + OI -> HVL 0DTE for the 0/1 DTE chain (expirations D, D+1)
  - Prev day P's NDX prices -> HVL 0DTE via IV inversion + BS gamma
  - Convert both to NQ space using P's settle ratio (QQQ) / basis (NDX)
  - NQ at 9:30 ET (open) and 17:00 ET (close) from price parquets
  - Compare cohorts (above HVL vs below) for hit rate, mean return, |return|

Outputs: D:/trading_pythonbacktest_data/QQQ_thetadata/study_hvl0dte_meanrev.parquet
        plus a printed summary table.
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
OUT      = QQQ_ROOT / "study_hvl0dte_meanrev.parquet"

R = 0.05; Q_NDX = 0.006


# ------------------------------ BS helpers (NDX) ------------------------------

def _d1(S, K, T, sigma, q):
    return (np.log(S/K) + (R - q + 0.5*sigma**2) * T) / (sigma * np.sqrt(T))
def bs_call(S,K,T,sigma,q):
    d1=_d1(S,K,T,sigma,q); d2=d1-sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-R*T)*norm.cdf(d2)
def bs_put(S,K,T,sigma,q):
    d1=_d1(S,K,T,sigma,q); d2=d1-sigma*np.sqrt(T)
    return K*np.exp(-R*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
def bs_vega(S,K,T,sigma,q):
    return S*np.exp(-q*T)*norm.pdf(_d1(S,K,T,sigma,q))*np.sqrt(T)
def bs_gamma(S,K,T,sigma,q):
    return np.exp(-q*T)*norm.pdf(_d1(S,K,T,sigma,q))/(S*sigma*np.sqrt(T))

def iv_vec(price,S,K,T,is_call,q):
    sigma = np.full_like(price, 0.30, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(15):
            bp = np.where(is_call, bs_call(S,K,T,sigma,q), bs_put(S,K,T,sigma,q))
            d = bp - price
            v = bs_vega(S,K,T,sigma,q)
            step = np.divide(d, v, out=np.zeros_like(d), where=(v>1e-8))
            sigma = np.clip(sigma - step, 1e-4, 5.0)
        bp = np.where(is_call, bs_call(S,K,T,sigma,q), bs_put(S,K,T,sigma,q))
        bad = (np.abs(bp - price) > np.maximum(0.10, 0.01*price))
    sigma[bad] = np.nan
    return sigma


# ------------------------------ HVL helpers ------------------------------

def hvl_from_strikes(by_strike: pd.DataFrame, spot: float, max_dist_pct=0.05) -> float | None:
    if by_strike.empty: return None
    s = by_strike.sort_values("strike").reset_index(drop=True).copy()
    s["cum"] = s["net_gex"].cumsum()
    band = spot * max_dist_pct
    flips = []
    for i in range(1, len(s)):
        K = float(s.iloc[i]["strike"])
        if abs(K - spot) > band: continue
        if (s.iloc[i-1]["cum"] > 0) != (s.iloc[i]["cum"] > 0):
            flips.append(K)
    return min(flips, key=lambda k: abs(k-spot)) if flips else None


def classify_regime(by_strike: pd.DataFrame, spot: float, max_dist_pct=0.05) -> tuple[str, float | None]:
    """Classify the 0-1 DTE chain near spot into one of 4 regimes:
       - 'no_data'      : empty chain or no rows in window
       - 'above_flip'   : flip exists in window, spot is ABOVE the flip strike
       - 'below_flip'   : flip exists in window, spot is BELOW the flip strike
       - 'deep_pos'     : no flip; entire ±max_dist_pct band has positive cumulative GEX
       - 'deep_neg'     : no flip; entire band has negative cumulative GEX
       - 'mixed_no_flip': edge case (shouldn't normally happen)
    Returns (regime_label, hvl_strike_or_none)."""
    if by_strike.empty:
        return "no_data", None
    s = by_strike.sort_values("strike").reset_index(drop=True).copy()
    s["cum"] = s["net_gex"].cumsum()
    band = spot * max_dist_pct
    near = s[(s["strike"] >= spot - band) & (s["strike"] <= spot + band)].copy()
    if near.empty:
        return "no_data", None

    flips = []
    for i in range(1, len(s)):
        K = float(s.iloc[i]["strike"])
        if abs(K - spot) > band: continue
        if (s.iloc[i-1]["cum"] > 0) != (s.iloc[i]["cum"] > 0):
            flips.append(K)
    if flips:
        hvl = min(flips, key=lambda k: abs(k - spot))
        regime = "above_flip" if spot > hvl else "below_flip"
        return regime, float(hvl)

    pos = (near["cum"] > 0).all()
    neg = (near["cum"] < 0).all()
    if pos:  return "deep_pos", None
    if neg:  return "deep_neg", None
    return "mixed_no_flip", None


def qqq_hvl_0dte(prev_date: dt.date, today: dt.date) -> tuple[float | None, float | None, str]:
    """Returns (qqq_hvl_strike, qqq_spot_at_prev_settle, regime_label).
    Uses prev_date's parquet, filtered to expirations on today and today+1."""
    g_path = QQQ_ROOT / prev_date.isoformat() / "greeks_eod.parquet"
    o_path = QQQ_ROOT / prev_date.isoformat() / "open_interest.parquet"
    if not g_path.exists() or not o_path.exists(): return None, None, "no_data"
    g = pd.read_parquet(g_path); o = pd.read_parquet(o_path)
    g["expiration"] = pd.to_datetime(g["expiration"])
    o["expiration"] = pd.to_datetime(o["expiration"])
    end = today + dt.timedelta(days=2)
    g = g[(g["expiration"].dt.date >= today) & (g["expiration"].dt.date <= end)]
    if g.empty: return None, None, "no_data"
    chain = g.merge(o[["strike","right","expiration","open_interest"]],
                    on=["strike","right","expiration"], how="left")
    spot = float(chain["underlying_price"].iloc[0])
    chain["signed_gex"] = chain["gamma"] * chain["open_interest"].fillna(0) * 100 * spot**2
    chain.loc[chain["right"].str.upper()=="PUT","signed_gex"] *= -1
    by = (chain.groupby("strike")["signed_gex"].sum()
          .reset_index().rename(columns={"signed_gex":"net_gex"}).sort_values("strike"))
    regime, hvl = classify_regime(by, spot)
    return hvl, spot, regime


def ndx_hvl_0dte(prev_date: dt.date, today: dt.date) -> tuple[float | None, float | None, str]:
    eod_path = NDX_ROOT / prev_date.isoformat() / "eod.parquet"
    oi_path  = NDX_ROOT / prev_date.isoformat() / "oi.parquet"
    if not eod_path.exists() or not oi_path.exists(): return None, None, "no_data"
    eod = pd.read_parquet(eod_path); oi = pd.read_parquet(oi_path)
    eod["expiration"] = pd.to_datetime(eod["expiration"])
    oi["expiration"] = pd.to_datetime(oi["expiration"])
    end = today + dt.timedelta(days=2)
    eod = eod[(eod["expiration"].dt.date >= today) & (eod["expiration"].dt.date <= end)]
    if eod.empty: return None, None, "no_data"

    # parity spot from short-DTE pairs in the prev day's full chain
    full_eod = pd.read_parquet(eod_path)
    full_eod["expiration"] = pd.to_datetime(full_eod["expiration"])
    snap = pd.to_datetime(prev_date)
    df = full_eod[(full_eod["bid"]>0) & (full_eod["ask"]>0)].copy()
    df["mid"] = (df["bid"]+df["ask"])/2
    df["dte"] = (df["expiration"]-snap).dt.days
    df = df[(df["dte"]>0) & (df["dte"]<=36)]
    calls = df[df["right"].str.upper()=="CALL"][["root","strike","expiration","dte","mid"]].rename(columns={"mid":"C"})
    puts  = df[df["right"].str.upper()=="PUT"][["root","strike","expiration","dte","mid"]].rename(columns={"mid":"P"})
    pairs = calls.merge(puts, on=["root","strike","expiration","dte"])
    if pairs.empty: return None, None, "no_data"
    T = pairs["dte"].values/365.25
    S = ((pairs["C"]-pairs["P"])*np.exp(R*T)+pairs["strike"])/np.exp(-Q_NDX*T)
    spot = float(np.median(S))

    # Now compute gamma on the 0-1 DTE subset
    chain = eod.merge(oi[["root","strike","right","expiration","open_interest"]],
                      on=["root","strike","right","expiration"], how="left")
    chain["mid"] = (chain["bid"]+chain["ask"])/2
    chain = chain[(chain["bid"]>0) & (chain["ask"]>0) & (chain["mid"]>0)]
    chain["dte"] = (chain["expiration"]-snap).dt.days
    if chain.empty: return None, spot, "no_data"
    T = chain["dte"].values/365.25
    K = chain["strike"].values.astype(float)
    P = chain["mid"].values.astype(float)
    is_call = (chain["right"].str.upper()=="CALL").values
    S_arr = np.full_like(K, spot)
    chain["iv"] = iv_vec(P, S_arr, K, T, is_call, Q_NDX)
    chain = chain.dropna(subset=["iv"])
    if chain.empty: return None, spot, "no_data"
    chain["gamma"] = bs_gamma(spot, chain["strike"].values,
                              chain["dte"].values/365.25, chain["iv"].values, Q_NDX)
    chain["signed_gex"] = chain["gamma"]*chain["open_interest"].fillna(0)*100*spot**2
    chain.loc[chain["right"].str.upper()=="PUT","signed_gex"] *= -1
    by = (chain.groupby("strike")["signed_gex"].sum()
          .reset_index().rename(columns={"signed_gex":"net_gex"}).sort_values("strike"))
    regime, hvl = classify_regime(by, spot)
    return hvl, spot, regime


# ------------------------------ NQ price loader ------------------------------

def load_nq_series() -> pd.Series:
    """Combined ET-indexed NQ close series from markettick (1m) + 15min bars."""
    parts = []
    if NQ_1MIN.exists():
        m1 = pd.read_parquet(NQ_1MIN)
        idx = m1.index
        if idx.tz is None: idx = idx.tz_localize("UTC")
        m1.index = idx.tz_convert("America/New_York")
        parts.append(m1["close"].rename("nq_close"))
    if NQ_15MIN.exists():
        m15 = pd.read_parquet(NQ_15MIN)
        idx = m15.index
        if idx.tz is None: idx = idx.tz_localize("America/New_York")
        else: idx = idx.tz_convert("America/New_York")
        m15.index = idx
        parts.append(m15["close"].rename("nq_close"))
    nq = pd.concat(parts).sort_index()
    nq = nq[~nq.index.duplicated(keep="first")]
    return nq


def nq_at(nq: pd.Series, date: dt.date, hour: int, minute: int,
          lookback_min: int = 30) -> float:
    target = pd.Timestamp(year=date.year, month=date.month, day=date.day,
                          hour=hour, minute=minute, tz="America/New_York")
    win = nq.loc[target - pd.Timedelta(minutes=lookback_min) : target]
    return float(win.iloc[-1]) if not win.empty else float("nan")


# ------------------------------ Driver ------------------------------

def trading_dirs() -> list[dt.date]:
    """Days where both QQQ and NDX have parquet (prev day available)."""
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
    nq = load_nq_series()
    print(f"  range: {nq.index.min()} -> {nq.index.max()}  bars: {len(nq)}")

    days = trading_dirs()
    print(f"trading days available: {len(days)}")

    # Map each day to its previous trading day (data day, not calendar day)
    rows = []
    t0 = time.time()
    for i, d in enumerate(days):
        if i == 0: continue  # need prior
        prev = days[i-1]
        try:
            qqq_h, qqq_s, qqq_regime = qqq_hvl_0dte(prev, d)
            ndx_h, ndx_s, ndx_regime = ndx_hvl_0dte(prev, d)
        except Exception as e:
            print(f"  {d}: ERROR {type(e).__name__}: {e}")
            continue
        # Get NQ ratio/basis from prev day
        # ratio = NQ at prev settle / QQQ spot at prev settle
        # We approximate using nq at 17:00 ET on prev day
        nq_prev_close = nq_at(nq, prev, 17, 0, lookback_min=60)
        if not (np.isfinite(nq_prev_close) and qqq_s and qqq_s > 0):
            qqq_ratio = np.nan; ndx_basis = np.nan
        else:
            qqq_ratio = nq_prev_close / qqq_s
            ndx_basis = nq_prev_close - ndx_s if (ndx_s and ndx_s > 0) else np.nan

        nq_open = nq_at(nq, d, 9, 30, lookback_min=10)
        nq_close = nq_at(nq, d, 17, 0, lookback_min=60)

        qqq_hvl_nq = qqq_h * qqq_ratio if (qqq_h and np.isfinite(qqq_ratio)) else np.nan
        ndx_hvl_nq = ndx_h + ndx_basis if (ndx_h and np.isfinite(ndx_basis)) else np.nan

        rows.append({
            "date": d,
            "nq_open_930": nq_open,
            "nq_close_500": nq_close,
            "qqq_hvl_0dte_strike": qqq_h,
            "qqq_hvl_0dte_nq": qqq_hvl_nq,
            "qqq_regime": qqq_regime,
            "ndx_hvl_0dte_strike": ndx_h,
            "ndx_hvl_0dte_nq": ndx_hvl_nq,
            "ndx_regime": ndx_regime,
            "qqq_ratio_used": qqq_ratio,
            "ndx_basis_used": ndx_basis,
        })
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(days)}  elapsed={time.time()-t0:.0f}s  rows={len(rows)}")

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["nq_open_930","nq_close_500"])
    df["ret"] = (df["nq_close_500"] - df["nq_open_930"]) / df["nq_open_930"]
    df["abs_ret"] = df["ret"].abs()
    df.to_parquet(OUT, compression="zstd", index=False)
    print(f"\nwrote {OUT}  ({len(df)} rows)")

    # ------------------------------ Analysis ------------------------------

    def report(label, mask):
        sub = df[mask]
        n = len(sub)
        if n == 0:
            print(f"  {label:<55} n=0"); return
        p_pos = (sub["ret"] > 0).mean()
        m = sub["ret"].mean(); md = sub["ret"].median()
        std = sub["ret"].std()
        absm = sub["abs_ret"].mean()
        absmd = sub["abs_ret"].median()
        t = m / (std/np.sqrt(n)) if std else float("nan")
        print(f"  {label:<55} n={n:>5}  P(close>open)={p_pos:.2%}  "
              f"mean={m:+.3%}  med={md:+.3%}  |ret|_mean={absm:.3%}  |ret|_med={absmd:.3%}  "
              f"t={t:+.2f}")

    print("\n=== Headline ===")
    report("BASE  (all valid days)", df["ret"].notna())

    for src_label, hvl_col in [("QQQ-derived HVL 0DTE", "qqq_hvl_0dte_nq"),
                                ("NDX-derived HVL 0DTE", "ndx_hvl_0dte_nq")]:
        print(f"\n=== {src_label} ===")
        valid = df[hvl_col].notna() & df["ret"].notna()
        df_v = df[valid].copy()
        df_v["above"] = df_v["nq_open_930"] > df_v[hvl_col]
        df_v["above_buf"] = df_v["nq_open_930"] > df_v[hvl_col] * 1.001  # 0.1% buffer

        print("--- Strict comparison (open > HVL) ---")
        report("ABOVE HVL @ 9:30",  df_v["above"])
        report("BELOW HVL @ 9:30", ~df_v["above"])
        print("--- Buffered comparison (open > HVL * 1.001) ---")
        report("ABOVE HVL+0.1% @ 9:30",  df_v["above_buf"])
        report("BELOW HVL-0.1% @ 9:30 (i.e. NOT above buffered)", ~df_v["above_buf"])

        # Mean reversion test: |ret| should be smaller in positive-gamma cohort
        a = df_v[df_v["above"]]["abs_ret"]
        b = df_v[~df_v["above"]]["abs_ret"]
        if len(a) > 1 and len(b) > 1:
            from scipy import stats as ss
            t, p = ss.ttest_ind(a, b, equal_var=False)
            print(f"\n  |ret| test (above vs below):  "
                  f"above_mean={a.mean():.3%}  below_mean={b.mean():.3%}  "
                  f"t={t:+.2f}  p={p:.4f}")


if __name__ == "__main__":
    sys.exit(main())
