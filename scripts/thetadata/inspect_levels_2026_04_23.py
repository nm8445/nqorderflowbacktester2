"""Look at the QQQ gamma levels on 2026-04-23 and convert to NQ space.
Targets NQ 27,155 (user's interest). Shows nearby levels by DTE bucket and
ranks every distinct strike by |net_gex| weight."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATE = dt.date(2026, 4, 23)
NQ_TARGET = 27155.0
RATIO_FALLBACK = 41.3  # used if NQ-at-settle isn't available
DATA_ROOT = Path("D:/trading_pythonbacktest_data/QQQ_thetadata")


def gex_per_strike(chain: pd.DataFrame) -> pd.DataFrame:
    if chain.empty: return pd.DataFrame()
    df = chain.copy()
    df["signed_gex"] = df["gamma"] * df["open_interest"].fillna(0) * 100
    df.loc[df["right"].str.upper() == "PUT", "signed_gex"] *= -1
    spot = df["underlying_price"].iloc[0]
    df["signed_gex"] *= spot ** 2
    return (df.groupby("strike")["signed_gex"].sum()
              .reset_index().rename(columns={"signed_gex": "net_gex"})
              .sort_values("strike"))


def main():
    g = pd.read_parquet(DATA_ROOT / DATE.isoformat() / "greeks_eod.parquet")
    o = pd.read_parquet(DATA_ROOT / DATE.isoformat() / "open_interest.parquet")
    g["expiration"] = pd.to_datetime(g["expiration"])
    o["expiration"] = pd.to_datetime(o["expiration"])
    g["dte"] = (g["expiration"] - pd.Timestamp(DATE)).dt.days

    spot = g["underlying_price"].iloc[0]
    qqq_target = NQ_TARGET / RATIO_FALLBACK
    print(f"Date:       {DATE}")
    print(f"QQQ spot:   {spot:.2f}")
    print(f"NQ target:  {NQ_TARGET}")
    print(f"QQQ-equiv target (using ratio {RATIO_FALLBACK}): {qqq_target:.2f}")
    print(f"  (= NQ_target / ratio)")
    print()

    chain_full = g.merge(o[["strike","right","expiration","open_interest"]],
                         on=["strike","right","expiration"], how="left")

    # Per-DTE-bucket analysis
    for label, mask in [
        ("0DTE only",         g["dte"] == 0),
        ("1-7 DTE",           (g["dte"] >= 1) & (g["dte"] <= 7)),
        ("8-30 DTE",          (g["dte"] >= 8) & (g["dte"] <= 30)),
        ("31-45 DTE",         (g["dte"] >= 31) & (g["dte"] <= 45)),
        ("ALL <=45 DTE",      g["dte"] <= 45),
        ("ALL <=7 DTE",       g["dte"] <= 7),
    ]:
        sub = g[mask].merge(o[["strike","right","expiration","open_interest"]],
                            on=["strike","right","expiration"], how="left")
        if sub.empty:
            print(f"--- {label} ---  (no rows)")
            continue
        gex = gex_per_strike(sub)
        gex["abs_gex"] = gex["net_gex"].abs()
        # 5 strikes nearest the QQQ-equiv target
        gex["dist_to_target"] = (gex["strike"] - qqq_target).abs()
        near = gex.sort_values("dist_to_target").head(8)
        # 5 strikes with largest |net_gex|
        top = gex.sort_values("abs_gex", ascending=False).head(8)

        print(f"--- {label}  (n_strikes={len(gex)}) ---")
        print(f"  {'strikes near QQQ '+f'{qqq_target:.2f}':<48} | top |GEX| strikes")
        for n_row, t_row in zip(near.itertuples(), top.itertuples()):
            n_qqq = n_row.strike;  n_nq = n_qqq * RATIO_FALLBACK
            t_qqq = t_row.strike;  t_nq = t_qqq * RATIO_FALLBACK
            n_sign = "+" if n_row.net_gex > 0 else "-"
            t_sign = "+" if t_row.net_gex > 0 else "-"
            print(f"  K={n_qqq:>6.1f} (NQ {n_nq:>8.0f})  "
                  f"gex={n_sign}{abs(n_row.net_gex):>10.2e}  "
                  f"d={n_row.dist_to_target:>4.1f}    | "
                  f"K={t_qqq:>6.1f} (NQ {t_nq:>8.0f})  "
                  f"gex={t_sign}{abs(t_row.net_gex):>10.2e}")
        print()


if __name__ == "__main__":
    main()
