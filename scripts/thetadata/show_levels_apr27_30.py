"""Print QQQ gamma levels for 4/27..4/30, converted to NQ via the day's actual ratio.
Two views per day: (a) <=45 DTE aggregate, (b) 0DTE only."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import databento as db
import pandas as pd

QQQ_ROOT = Path("D:/trading_pythonbacktest_data/QQQ_thetadata")
DBN_ROOT = Path("D:/trading_pythonbacktest_data/dbn")
DATES = [dt.date(2026, 4, d) for d in (27, 28, 29, 30)]


def nq_at_settle(date: dt.date) -> float | None:
    """Last NQ trade in 16:45-17:00 ET window (just before futures daily break)."""
    end_et = pd.Timestamp(year=date.year, month=date.month, day=date.day,
                          hour=17, minute=0, tz="America/New_York")
    start = (end_et - pd.Timedelta(minutes=15)).tz_convert("UTC")
    end = end_et.tz_convert("UTC")
    p = DBN_ROOT / f"{date.isoformat()}.dbn"
    if not p.exists():
        return None
    df = db.DBNStore.from_file(str(p)).to_df()
    trades = df.loc[start:end]
    trades = trades[trades["action"] == "T"]
    if trades.empty:
        return None
    return float(trades.iloc[-1]["price"])


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


def extract_levels(gex: pd.DataFrame, spot: float) -> dict:
    if gex.empty: return {}
    above = gex[gex["strike"] >= spot]
    below = gex[gex["strike"] <= spot]
    cr = float(above.loc[above["net_gex"].idxmax(), "strike"]) if not above.empty else None
    ps = float(below.loc[below["net_gex"].idxmin(), "strike"]) if not below.empty else None

    s = gex.sort_values("strike").reset_index(drop=True).copy()
    s["cum"] = s["net_gex"].cumsum()
    sign = s["cum"].apply(lambda v: 1 if v > 0 else -1)
    flip = s[sign.diff().abs() > 0]
    hvl = float(flip.iloc[0]["strike"]) if not flip.empty else None

    top10 = (gex.reindex(gex["net_gex"].abs().sort_values(ascending=False).index).head(10))
    return {
        "call_resistance": cr, "put_support": ps, "hvl": hvl,
        "top10": list(top10[["strike", "net_gex"]].itertuples(index=False, name=None)),
    }


def levels_for(date: dt.date, dte_filter: str):
    g = pd.read_parquet(QQQ_ROOT / date.isoformat() / "greeks_eod.parquet")
    o = pd.read_parquet(QQQ_ROOT / date.isoformat() / "open_interest.parquet")
    g["expiration"] = pd.to_datetime(g["expiration"])
    o["expiration"] = pd.to_datetime(o["expiration"])
    g["dte"] = (g["expiration"] - pd.Timestamp(date)).dt.days

    if dte_filter == "0dte":
        g = g[g["dte"] == 0]
    elif dte_filter == "le45":
        g = g[(g["dte"] > 0) & (g["dte"] <= 45)]
    if g.empty:
        return None, None

    chain = g.merge(o[["strike", "right", "expiration", "open_interest"]],
                    on=["strike", "right", "expiration"], how="left")
    spot = float(chain["underlying_price"].iloc[0])
    return spot, extract_levels(gex_per_strike(chain), spot)


def fmt_row(name, qqq_lvl, ratio):
    if qqq_lvl is None:
        return f"  {name:<22} QQQ ----    NQ -----"
    return f"  {name:<22} QQQ {qqq_lvl:>7.2f}  NQ {qqq_lvl * ratio:>9.2f}"


def main():
    for date in DATES:
        nq = nq_at_settle(date)
        spot45, lv45 = levels_for(date, "le45")
        spot0,  lv0  = levels_for(date, "0dte")

        if nq is None or spot45 is None:
            print(f"\n=== {date} ({date.strftime('%a')})  -- skipped (data missing) ---")
            continue

        ratio = nq / spot45
        print(f"\n=== {date} ({date.strftime('%a')}) ===")
        print(f"  QQQ spot {spot45:.2f}   NQ at settle {nq:.2f}   ratio {ratio:.3f}")

        print(f"\n  ---- <=45 DTE aggregate ----")
        print(fmt_row("call_resistance", lv45["call_resistance"], ratio))
        print(fmt_row("put_support", lv45["put_support"], ratio))
        print(fmt_row("hvl", lv45["hvl"], ratio))
        for i, (k, gex) in enumerate(lv45["top10"], 1):
            sign = "+" if gex >= 0 else "-"
            print(f"  GEX_{i:<2}                 QQQ {k:>7.2f}  NQ {k * ratio:>9.2f}  "
                  f"({sign}{abs(gex):.2e})")

        if lv0 is None:
            print("\n  ---- 0DTE: no rows ----")
        else:
            print(f"\n  ---- 0DTE only ----")
            print(fmt_row("call_resistance", lv0["call_resistance"], ratio))
            print(fmt_row("put_support", lv0["put_support"], ratio))
            print(fmt_row("hvl", lv0["hvl"], ratio))
            for i, (k, gex) in enumerate(lv0["top10"], 1):
                sign = "+" if gex >= 0 else "-"
                print(f"  GEX_{i:<2}                 QQQ {k:>7.2f}  NQ {k * ratio:>9.2f}  "
                      f"({sign}{abs(gex):.2e})")


if __name__ == "__main__":
    main()
