"""Value-tier validation (calibrated): do MenthorQ levels reproduce from LOCALLY-derived gamma
(Value tier, price-only) vs ThetaData's API gamma (Standard)? And how big is any drift in NQ points?

Calibration: ThetaData's implied T (backed out of gamma/IV/d1) is ~dte/357 (not dte/365.25), and it
uses precise intraday T for near-dated options. We test T conventions and the production DTE filters.

Paths per day (data we already have on Standard lets us diff all of them):
  A  API gamma                          — the current levels (baseline)
  B  bs_gamma(API IV, calibrated T)     — gamma FORMULA test
  C  bs_gamma(IV backed out of price, calibrated T)  — the FULL VALUE path (price only)

Levels: CR, PS (from the (1,2)-DTE chain, legacy ranking = production), HVL (0-45 DTE). Reports the
exact-strike match rate AND the drift converted to NQ points (via qqq_ratio = nq_settle / qqq_spot).

Run:  python scripts/thetadata/daily_pipeline/validate_value_tier_levels.py [N_DAYS] [DAYCOUNT]
"""
from __future__ import annotations
import sys
from pathlib import Path
import datetime as dt
import warnings; warnings.filterwarnings("ignore"); import numpy as np; np.seterr(all="ignore")
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import menthorq_style_levels as mq  # noqa: E402
from menthorq_style_levels import bs_gamma, iv_vec, atm_iv, expected_move, Q_QQQ  # noqa: E402

QQQ_ROOT = Path("D:/trading_pythonbacktest_data/QQQ_thetadata")
NQ_1MIN = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
DAYCOUNT = float(sys.argv[2]) if len(sys.argv) > 2 else 357.0   # ThetaData's implied day-count


def nq_settles():
    nq = pd.read_parquet(NQ_1MIN)
    idx = nq.index
    if idx.tz is None: idx = idx.tz_localize("UTC")
    nq.index = idx.tz_convert("America/New_York"); nq = nq.sort_index()
    m = (nq.index.hour == 16) & (nq.index.minute == 5)
    return {ts.date(): float(p) for ts, p in nq.loc[m]["close"].items()}


def by_strike(g, o, date, dte_filter, mode):
    g = g.copy(); g["expiration"] = pd.to_datetime(g["expiration"])
    g["dte"] = (g["expiration"] - pd.Timestamp(date)).dt.days
    lo, hi = dte_filter
    g = g[(g["dte"] >= lo) & (g["dte"] <= hi)]
    if g.empty: return None, None, None
    spot = float(g["underlying_price"].iloc[0])
    iv_atm = atm_iv(g, spot, "implied_vol")
    o = o.copy(); o["expiration"] = pd.to_datetime(o["expiration"])
    c = g.merge(o[["strike", "right", "expiration", "open_interest"]], on=["strike", "right", "expiration"], how="left")
    S, K = spot, c["strike"].values
    T = np.maximum(c["dte"].values, 0.15) / DAYCOUNT     # calibrated T; 0-DTE floored to ~4h
    is_call = (c["right"].str.upper() == "CALL").values
    if mode == "A":
        gam = c["gamma"].values
    elif mode == "B":
        gam = bs_gamma(S, K, T, c["implied_vol"].values, Q_QQQ)
    else:  # C: back out IV from price
        P = c["close"].values if mode == "C_close" else ((c["bid"] + c["ask"]) / 2).values
        gam = bs_gamma(S, K, T, iv_vec(P, S, K, T, is_call, Q_QQQ), Q_QQQ)
    gex = np.nan_to_num(gam) * c["open_interest"].fillna(0).values * 100 * spot ** 2
    c["signed_gex"] = np.where(is_call, gex, -gex)
    grp = c.groupby("strike").agg(net_gex=("signed_gex", "sum")).reset_index().sort_values("strike")
    for col in ("net_dex", "call_only_dex", "put_only_dex"): grp[col] = 0.0
    return grp, spot, expected_move(spot, iv_atm)


def levels_for(g, o, date, mode):
    bs, spot, em = by_strike(g, o, date, (1, 2), mode)       # CR/PS: production near-term filter
    cr, ps, gex = mq.menthorq_levels(bs, spot, em, legacy_mode=True) if bs is not None else (None, None, [])
    bs45, s45, _ = by_strike(g, o, date, (0, 45), mode)
    hvl = mq.hvl_from_gex(bs45, s45) if bs45 is not None else None
    return {"CR": cr[0] if cr else None, "PS": ps[0] if ps else None, "HVL": hvl}, spot


def main():
    n_days = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    settles = nq_settles()
    days = sorted([dt.date.fromisoformat(p.name) for p in QQQ_ROOT.iterdir()
                   if p.is_dir() and (p / "greeks_eod.parquet").exists() and dt.date.fromisoformat(p.name) in settles])[-n_days:]
    print(f"Calibrated T = dte/{DAYCOUNT:.0f}. {len(days)} days ({days[0]}->{days[-1]}).")
    print("Match = same STRIKE as API (path A). Drift shown in NQ points (QQQ diff x ratio).\n")
    ok = {"B": 0, "C_close": 0, "C_mid": 0}
    nqdiff = {"B": [], "C_close": [], "C_mid": []}
    tot = 0
    for d in days:
        r = QQQ_ROOT / d.isoformat()
        try:
            g = pd.read_parquet(r / "greeks_eod.parquet"); o = pd.read_parquet(r / "open_interest.parquet")
        except Exception: continue
        (la, spot) = levels_for(g, o, d, "A")
        ratio = settles[d] / spot if spot else np.nan
        tot += 1
        for m in ("B", "C_close", "C_mid"):
            lm, _ = levels_for(g, o, d, m)
            ok[m] += (lm == la)
            for k in ("CR", "PS", "HVL"):
                if la[k] is not None and lm[k] is not None and np.isfinite(ratio):
                    nqdiff[m].append(abs(lm[k] - la[k]) * ratio)   # NQ pts; 0 where the strike matched
    print(f"of {tot} days, all 3 levels (CR/PS/HVL) IDENTICAL to Standard:")
    for m in ("B", "C_close", "C_mid"):
        arr = np.array([x for x in nqdiff[m] if np.isfinite(x)])
        mism = arr[arr > 0]                       # only the level-comparisons that actually differ
        lvl_match = 100 * (arr == 0).mean()
        print(f"  {m:8s}: {ok[m]:>2}/{tot} days exact ({100*ok[m]/tot:>3.0f}%) | "
              f"per-LEVEL match {lvl_match:.0f}% | when a level differs: median {np.median(mism) if len(mism) else 0:.0f} NQpt, "
              f"p90 {np.percentile(mism,90) if len(mism) else 0:.0f}pt, max {mism.max() if len(mism) else 0:.0f}pt")


if __name__ == "__main__":
    main()
