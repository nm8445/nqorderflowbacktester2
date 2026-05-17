"""MenthorQ-style gamma levels — full algorithm per official docs.

Documented methodology (https://menthorq.com/guide/):
  - Window: 1D Expected Move = spot * ATM_IV * sqrt(1/252)
  - Call Resistance = strike with most net call gamma (windowed)
  - Put Support     = strike with most net put gamma (windowed)
  - HVL             = inflection point of cumulative-GEX curve (sign flip)
  - GEX 1..10       = top strikes by combined Net GEX + Net DEX,
                      EXCLUDING the CR and PS strikes; within 1D EM window

Net GEX  = gamma * OI * 100 * spot^2     (calls positive, puts negated)
Net DEX  = delta * OI * 100 * spot       (call delta natural +, put delta natural -)

Usage:  python menthorq_style_levels.py [YYYY-MM-DD] [--em-mult 1.0]
Output: scripts/thetadata/menthorq_style_<date>.txt
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

R = 0.05; Q_NDX = 0.006; Q_QQQ = 0.005

DAILY_RATIO_QQQ = {
    dt.date(2026, 4, 22): 41.378,
    dt.date(2026, 4, 23): 41.360,
    dt.date(2026, 4, 27): 41.279,
    dt.date(2026, 4, 28): 41.353,
    dt.date(2026, 4, 29): 41.037,
    dt.date(2026, 4, 30): 41.331,
}
DAILY_BASIS_NDX = {
    dt.date(2026, 4, 23): 98.9,
    dt.date(2026, 4, 27): 99.4,
    dt.date(2026, 4, 28): 105.7,
    dt.date(2026, 4, 29): 109.1,
    dt.date(2026, 4, 30): 106.4,
}


# --------------- BS primitives ---------------

def _d1(S, K, T, sigma, q):
    return (np.log(S/K) + (R - q + 0.5*sigma**2) * T) / (sigma * np.sqrt(T))

def bs_call(S, K, T, sigma, q):
    d1 = _d1(S, K, T, sigma, q); d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-R*T)*norm.cdf(d2)

def bs_put(S, K, T, sigma, q):
    d1 = _d1(S, K, T, sigma, q); d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-R*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def bs_vega(S, K, T, sigma, q):
    return S*np.exp(-q*T)*norm.pdf(_d1(S, K, T, sigma, q))*np.sqrt(T)

def bs_gamma(S, K, T, sigma, q):
    return np.exp(-q*T)*norm.pdf(_d1(S, K, T, sigma, q)) / (S*sigma*np.sqrt(T))

def bs_call_delta(S, K, T, sigma, q):
    return np.exp(-q*T) * norm.cdf(_d1(S, K, T, sigma, q))

def bs_put_delta(S, K, T, sigma, q):
    return -np.exp(-q*T) * norm.cdf(-_d1(S, K, T, sigma, q))

def iv_vec(price, S, K, T, is_call, q):
    sigma = np.full_like(price, 0.30, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(15):
            bp = np.where(is_call, bs_call(S,K,T,sigma,q), bs_put(S,K,T,sigma,q))
            d = bp - price
            v = bs_vega(S,K,T,sigma,q)
            step = np.divide(d, v, out=np.zeros_like(d), where=(v > 1e-8))
            sigma = np.clip(sigma - step, 1e-4, 5.0)
        bp = np.where(is_call, bs_call(S,K,T,sigma,q), bs_put(S,K,T,sigma,q))
        bad = (np.abs(bp - price) > np.maximum(0.10, 0.01*price))
    sigma[bad] = np.nan
    return sigma


# --------------- 1D Expected Move ---------------

def atm_iv(chain: pd.DataFrame, spot: float, iv_col: str) -> float:
    """Mean IV of 4 closest strikes at the shortest non-zero DTE."""
    sub = chain[chain["dte"] > 0].copy()
    if sub.empty: return float("nan")
    min_dte = int(sub["dte"].min())
    sub = sub[sub["dte"] == min_dte].copy()
    sub["dist"] = (sub["strike"] - spot).abs()
    near = sub.nsmallest(4, "dist")
    return float(near[iv_col].mean())


def expected_move(spot: float, iv: float) -> float:
    if not np.isfinite(iv): return float("nan")
    return spot * iv * np.sqrt(1.0 / 252.0)


# --------------- HVL ---------------

def hvl_from_gex(by_strike: pd.DataFrame, spot: float, max_dist_pct: float = 0.05) -> float | None:
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


# --------------- Level extraction (MenthorQ convention) ---------------

def menthorq_levels(by_strike: pd.DataFrame, spot: float, em: float,
                    exclude_extra: set | None = None,
                    legacy_mode: bool = False):
    """by_strike must have columns: strike, net_gex, net_dex,
    call_only_gex, put_only_gex, call_only_dex, put_only_dex.

    Returns (call_resistance, put_support, gex_list).

    legacy_mode=False (default — MenthorQ docs):
      - 1D EM window restriction (spot ± em)
      - CR = max calls-only gamma above spot
      - PS = max puts-only gamma below spot
      - GEX 1..10 ranked by combined |net_gex|/max + |net_dex|/max
      - GEX 1..10 EXCLUDES CR, PS, and any `exclude_extra` strikes

    legacy_mode=True (matches scripts/thetadata/build_ndx_levels.py + build_nq_levels.py):
      - NO EM window restriction (uses full chain in `by_strike`)
      - CR = max net_gex above spot (any strike, not calls-only)
      - PS = min net_gex below spot (most negative, not puts-only)
      - GEX 1..10 ranked by |net_gex| only
      - NO exclusions of CR/PS/HVL — they appear in GEX list if top-ranked
    """
    if by_strike.empty:
        return None, None, []

    if legacy_mode:
        # OLD methodology used to generate entry_signal_trades.parquet:
        #   - No EM window restriction
        #   - CR = max net_gex strike above spot (any strike, not calls-only)
        #   - PS = min net_gex strike below spot (most negative, not puts-only)
        #   - GEX 1..10 by |net_gex| only, EXCLUDES CR/PS (user clarification:
        #     "cr and ps are separate from gex 1 to 10 levels")
        above = by_strike[by_strike["strike"] >= spot]
        below = by_strike[by_strike["strike"] <  spot]
        cr_row = above.loc[above["net_gex"].idxmax()] if not above.empty else None
        ps_row = below.loc[below["net_gex"].idxmin()] if not below.empty else None
        cr_strike = float(cr_row["strike"]) if cr_row is not None else None
        ps_strike = float(ps_row["strike"]) if ps_row is not None else None

        # Exclude CR and PS strikes from GEX 1..10
        excluded = {cr_strike, ps_strike}
        if exclude_extra:
            excluded |= set(exclude_extra)
        excluded.discard(None)

        rest = by_strike[~by_strike["strike"].isin(excluded)].copy()
        rest["abs_gex"] = rest["net_gex"].abs()
        gex_top = rest.sort_values("abs_gex", ascending=False).head(10)

        cr = (cr_strike, float(cr_row["net_gex"])) if cr_row is not None else None
        ps = (ps_strike, float(ps_row["net_gex"])) if ps_row is not None else None
        cols = ["strike", "net_gex", "net_dex", "call_only_dex", "put_only_dex"]
        gex_list = list(gex_top[cols].itertuples(index=False, name=None))
        return cr, ps, gex_list

    # Default (MenthorQ docs) — 1D EM window required
    if not np.isfinite(em):
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

    excluded = {cr_strike, ps_strike}
    if exclude_extra:
        excluded |= set(exclude_extra)
    excluded.discard(None)

    rest = win[~win["strike"].isin(excluded)].copy()
    rest["abs_gex"] = rest["net_gex"].abs()
    rest["abs_dex"] = rest["net_dex"].abs()
    max_gex = rest["abs_gex"].max() or 1.0
    max_dex = rest["abs_dex"].max() or 1.0
    rest["score"] = rest["abs_gex"] / max_gex + rest["abs_dex"] / max_dex
    gex_top = rest.sort_values("score", ascending=False).head(10)

    cr = (cr_strike, float(cr_row["call_only_gex"])) if cr_row is not None else None
    ps = (ps_strike, -float(ps_row["put_only_gex"])) if ps_row is not None else None
    cols = ["strike", "net_gex", "net_dex", "call_only_dex", "put_only_dex"]
    gex_list = list(gex_top[cols].itertuples(index=False, name=None))
    return cr, ps, gex_list


# --------------- QQQ pipeline ---------------

def qqq_strikes(date: dt.date, dte_filter: tuple[int, int]) -> tuple[pd.DataFrame, float, float]:
    """Returns (by_strike with [strike, net_gex, net_dex], spot, atm_iv)."""
    root = Path(f"D:/trading_pythonbacktest_data/QQQ_thetadata/{date.isoformat()}")
    g = pd.read_parquet(root / "greeks_eod.parquet")
    o = pd.read_parquet(root / "open_interest.parquet")
    g["expiration"] = pd.to_datetime(g["expiration"])
    o["expiration"] = pd.to_datetime(o["expiration"])
    g["dte"] = (g["expiration"] - pd.Timestamp(date)).dt.days
    lo, hi = dte_filter
    g = g[(g["dte"] >= lo) & (g["dte"] <= hi)]
    spot = float(g["underlying_price"].iloc[0])
    iv = atm_iv(g, spot, "implied_vol")

    chain = g.merge(o[["strike","right","expiration","open_interest"]],
                    on=["strike","right","expiration"], how="left")
    chain["gex_abs"] = chain["gamma"] * chain["open_interest"].fillna(0) * 100 * spot**2
    chain["signed_gex"] = chain["gex_abs"]
    chain.loc[chain["right"].str.upper() == "PUT", "signed_gex"] *= -1
    chain["signed_dex"] = chain["delta"] * chain["open_interest"].fillna(0) * 100 * spot
    is_call = (chain["right"].str.upper() == "CALL")
    chain["call_only_gex"] = np.where(is_call, chain["gex_abs"], 0.0)
    chain["put_only_gex"]  = np.where(is_call, 0.0, chain["gex_abs"])
    chain["call_only_dex"] = np.where(is_call, chain["signed_dex"], 0.0)
    chain["put_only_dex"]  = np.where(is_call, 0.0, chain["signed_dex"])
    grouped = (chain.groupby("strike").agg(net_gex=("signed_gex","sum"),
                                            net_dex=("signed_dex","sum"),
                                            call_only_gex=("call_only_gex","sum"),
                                            put_only_gex=("put_only_gex","sum"),
                                            call_only_dex=("call_only_dex","sum"),
                                            put_only_dex=("put_only_dex","sum"))
               .reset_index().sort_values("strike"))
    return grouped, spot, iv


# --------------- NDX pipeline ---------------

def ndx_derive_spot(eod, snap_date, dte_max=36):
    df = eod[(eod["bid"] > 0) & (eod["ask"] > 0)].copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["dte"] = (df["expiration"] - snap_date).dt.days
    df = df[(df["dte"] > 0) & (df["dte"] <= dte_max)]
    calls = df[df["right"].str.upper() == "CALL"][["root","strike","expiration","dte","mid"]].rename(columns={"mid":"C"})
    puts  = df[df["right"].str.upper() == "PUT"][["root","strike","expiration","dte","mid"]].rename(columns={"mid":"P"})
    pairs = calls.merge(puts, on=["root","strike","expiration","dte"])
    if pairs.empty: return float("nan")
    T = pairs["dte"].values / 365.25
    S = ((pairs["C"] - pairs["P"]) * np.exp(R*T) + pairs["strike"]) / np.exp(-Q_NDX*T)
    return float(np.median(S))


def ndx_strikes(date: dt.date, dte_filter: tuple[int, int]) -> tuple[pd.DataFrame, float, float]:
    root = Path(f"D:/trading_pythonbacktest_data/NDX_thetadata/{date.isoformat()}")
    eod = pd.read_parquet(root / "eod.parquet")
    oi  = pd.read_parquet(root / "oi.parquet")
    eod["expiration"] = pd.to_datetime(eod["expiration"])
    oi["expiration"]  = pd.to_datetime(oi["expiration"])
    snap = pd.to_datetime(date)
    spot = ndx_derive_spot(eod, snap)

    chain = eod.merge(oi[["root","strike","right","expiration","open_interest"]],
                      on=["root","strike","right","expiration"], how="left")
    chain["mid"] = (chain["bid"] + chain["ask"]) / 2
    chain = chain[(chain["bid"] > 0) & (chain["ask"] > 0) & (chain["mid"] > 0)]
    chain["dte"] = (chain["expiration"] - snap).dt.days
    lo, hi = dte_filter
    chain = chain[(chain["dte"] >= lo) & (chain["dte"] <= hi)]
    if chain.empty:
        return pd.DataFrame(columns=["strike","net_gex","net_dex","call_only_gex","put_only_gex","call_only_dex","put_only_dex"]), spot, float("nan")

    T = chain["dte"].values / 365.25
    K = chain["strike"].values.astype(float)
    P = chain["mid"].values.astype(float)
    is_call = (chain["right"].str.upper() == "CALL").values
    S_arr = np.full_like(K, spot)
    chain["iv"] = iv_vec(P, S_arr, K, T, is_call, Q_NDX)
    chain = chain.dropna(subset=["iv"])

    iv = atm_iv(chain, spot, "iv")

    chain["gamma"] = bs_gamma(spot, chain["strike"].values,
                              chain["dte"].values/365.25, chain["iv"].values, Q_NDX)
    # Compute delta locally — call_delta for calls, put_delta for puts
    is_c = (chain["right"].str.upper() == "CALL").values
    chain["delta"] = np.where(
        is_c,
        bs_call_delta(spot, chain["strike"].values,
                      chain["dte"].values/365.25, chain["iv"].values, Q_NDX),
        bs_put_delta(spot, chain["strike"].values,
                     chain["dte"].values/365.25, chain["iv"].values, Q_NDX),
    )

    chain["gex_abs"] = chain["gamma"] * chain["open_interest"].fillna(0) * 100 * spot**2
    chain["signed_gex"] = chain["gex_abs"]
    chain.loc[chain["right"].str.upper() == "PUT", "signed_gex"] *= -1
    chain["signed_dex"] = chain["delta"] * chain["open_interest"].fillna(0) * 100 * spot
    is_c2 = (chain["right"].str.upper() == "CALL")
    chain["call_only_gex"] = np.where(is_c2, chain["gex_abs"], 0.0)
    chain["put_only_gex"]  = np.where(is_c2, 0.0, chain["gex_abs"])
    chain["call_only_dex"] = np.where(is_c2, chain["signed_dex"], 0.0)
    chain["put_only_dex"]  = np.where(is_c2, 0.0, chain["signed_dex"])
    grouped = (chain.groupby("strike").agg(net_gex=("signed_gex","sum"),
                                            net_dex=("signed_dex","sum"),
                                            call_only_gex=("call_only_gex","sum"),
                                            put_only_gex=("put_only_gex","sum"),
                                            call_only_dex=("call_only_dex","sum"),
                                            put_only_dex=("put_only_dex","sum"))
               .reset_index().sort_values("strike"))
    return grouped, spot, iv


# --------------- Output formatter ---------------

def fmt_section(title, spot, iv, em, hvl, hvl_0dte, cr, ps, gex_list, scale_label, scale_fn):
    lines = [f"--- {title} ---",
             f"  spot:             {spot:.2f}    NQ-equiv: {scale_fn(spot):.2f}    ({scale_label})",
             f"  ATM IV:           {iv:.4f}",
             f"  1D Expected Move: +/- {em:.2f}    window: [{spot-em:.2f}, {spot+em:.2f}]",
             f"  HVL:              {f'K={hvl:.1f}  NQ={scale_fn(hvl):.0f}' if hvl else '--'}",
             f"  HVL 0DTE:         {f'K={hvl_0dte:.1f}  NQ={scale_fn(hvl_0dte):.0f}' if hvl_0dte else '--'}",
             f"  Call Resistance:  "
             f"{f'K={cr[0]:.1f}  NQ={scale_fn(cr[0]):.0f}  (calls-only gex +{cr[1]:.2e})' if cr else '--'}",
             f"  Put Support:      "
             f"{f'K={ps[0]:.1f}  NQ={scale_fn(ps[0]):.0f}  (puts-only gex {ps[1]:+.2e})' if ps else '--'}",
             f"  GEX 1..10  (0-1 DTE chain, within 1D EM, top combined |Net GEX|+|Net DEX|, excl CR & PS):"]
    for i, (k, gex, dex, c_dex, p_dex) in enumerate(gex_list, 1):
        gs = "+" if gex >= 0 else ""
        ds = "+" if dex >= 0 else ""
        lines.append(f"    GEX_{i:<2}  K={k:>8.1f}  NQ={scale_fn(k):>8.0f}  "
                     f"gex={gs}{gex:.2e}  net_dex={ds}{dex:.2e}  "
                     f"call_dex=+{c_dex:.2e}  put_dex={p_dex:+.2e}")
    return "\n".join(lines)


# --------------- Driver ---------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", nargs="?", default="2026-04-23")
    ap.add_argument("--em-mult", type=float, default=1.0,
                    help="Multiplier on 1D EM window (default 1.0)")
    args = ap.parse_args()
    DATE = dt.date.fromisoformat(args.date)
    out_path = Path(__file__).parent / f"menthorq_style_{DATE.isoformat()}.txt"

    print(f"computing MenthorQ-style levels for {DATE}...")

    # CR/PS use full chain (per docs, "across all expirations")
    # GEX 1..10 uses 0-1 DTE only (per docs, "1D Exp Range")
    # GEX 1..10 also excludes HVL and HVL 0DTE strikes (named anchor levels)
    q_main, q_spot, q_iv = qqq_strikes(DATE, (0, 45))
    q_0dte, _, _         = qqq_strikes(DATE, (0, 1))
    q_em = expected_move(q_spot, q_iv) * args.em_mult
    q_hvl    = hvl_from_gex(q_main, q_spot)
    q_hvl_0  = hvl_from_gex(q_0dte, q_spot)
    q_cr, q_ps, _   = menthorq_levels(q_main, q_spot, q_em)
    _, _, q_gex     = menthorq_levels(q_0dte, q_spot, q_em,
                                      exclude_extra={q_hvl, q_hvl_0,
                                                     q_cr[0] if q_cr else None,
                                                     q_ps[0] if q_ps else None})

    n_main, n_spot, n_iv = ndx_strikes(DATE, (0, 45))
    n_0dte, _, _         = ndx_strikes(DATE, (0, 1))
    n_em = expected_move(n_spot, n_iv) * args.em_mult
    n_hvl    = hvl_from_gex(n_main, n_spot)
    n_hvl_0  = hvl_from_gex(n_0dte, n_spot)
    n_cr, n_ps, _   = menthorq_levels(n_main, n_spot, n_em)
    _, _, n_gex     = menthorq_levels(n_0dte, n_spot, n_em,
                                      exclude_extra={n_hvl, n_hvl_0,
                                                     n_cr[0] if n_cr else None,
                                                     n_ps[0] if n_ps else None})

    ratio = DAILY_RATIO_QQQ.get(DATE, 41.30)
    basis = DAILY_BASIS_NDX.get(DATE, 100.0)

    lines = [f"=== MENTHORQ-STYLE GAMMA LEVELS  {DATE} ({DATE.strftime('%a')}) ===",
             f"  Window: 1D Expected Move = spot * ATM_IV * sqrt(1/252)",
             f"  Score:  |Net GEX| + spot * |Net DEX|  (excludes CR and PS strikes)",
             f"  HVL:    cumulative GEX zero-crossing within +/-5% of spot",
             f"  QQQ ratio: {ratio:.3f}  |  NDX-NQ basis: +{basis:.1f}",
             ""]
    lines.append(fmt_section("QQQ", q_spot, q_iv, q_em, q_hvl, q_hvl_0, q_cr, q_ps, q_gex,
                             f"NQ = QQQ * {ratio:.3f}",
                             lambda x: x * ratio))
    lines.append("")
    lines.append(fmt_section("NDX (NDX + NDXP combined)", n_spot, n_iv, n_em,
                             n_hvl, n_hvl_0, n_cr, n_ps, n_gex,
                             f"NQ = NDX + {basis:.1f}",
                             lambda x: x + basis))

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out_path}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
