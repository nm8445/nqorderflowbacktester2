"""Build NQ-converted gamma levels from the QQQ ThetaData archive.

For each trading day:
  1. Load QQQ EOD greeks + open_interest parquets
  2. Compute net GEX per strike, extract 13 levels (call_res, put_sup, hvl, GEX_1..10)
  3. Look up NQ front-month price at 17:15 ET (the QQQ settle time)
  4. Convert every level via qqq_level * (nq_spot / qqq_spot)

Output: D:/trading_pythonbacktest_data/QQQ_thetadata/nq_levels.parquet
       (one row per trading day with both QQQ and NQ levels)

Optional DTE filter via --max-dte (default 45 — keeps standard MenthorQ-style window
and excludes far-dated LEAP gamma that dilutes near-term levels).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

QQQ_ROOT = Path("D:/trading_pythonbacktest_data/QQQ_thetadata")
NQ_1MIN  = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
NQ_15MIN = Path("D:/trading_pythonbacktest_data/15min_bars.parquet")
OUTPUT   = QQQ_ROOT / "nq_levels.parquet"

SETTLE_HOUR = 17
SETTLE_MIN  = 15


def load_nq_prices() -> pd.Series:
    """Build a unified ET-indexed NQ close-price series from the two parquet sources.
    1-min MarketTick wins where they overlap (finer granularity)."""
    parts = []

    if NQ_1MIN.exists():
        m1 = pd.read_parquet(NQ_1MIN)
        idx = m1.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        m1.index = idx.tz_convert("America/New_York")
        parts.append(m1["close"].rename("nq_close"))

    if NQ_15MIN.exists():
        m15 = pd.read_parquet(NQ_15MIN)
        idx = m15.index
        if idx.tz is None:
            idx = idx.tz_localize("America/New_York")
        else:
            idx = idx.tz_convert("America/New_York")
        m15.index = idx
        parts.append(m15["close"].rename("nq_close"))

    if not parts:
        raise FileNotFoundError("no NQ price sources found")

    nq = pd.concat(parts).sort_index()
    nq = nq[~nq.index.duplicated(keep="first")]
    return nq


def nq_at_settle(nq: pd.Series, date: dt.date) -> float:
    """NQ price at 17:15 ET on `date`. Falls back to nearest prior bar within 30 min."""
    target = pd.Timestamp(year=date.year, month=date.month, day=date.day,
                          hour=SETTLE_HOUR, minute=SETTLE_MIN,
                          tz="America/New_York")
    window = nq.loc[target - pd.Timedelta(minutes=30) : target]
    if window.empty:
        return float("nan")
    return float(window.iloc[-1])


def compute_gex_by_strike(chain: pd.DataFrame) -> pd.DataFrame:
    """Net GEX per strike. Calls positive, puts negated (dealer perspective)."""
    if chain.empty or "gamma" not in chain.columns:
        return pd.DataFrame()
    df = chain.copy()
    df["signed_gex"] = df["gamma"] * df["open_interest"].fillna(0) * 100
    df.loc[df["right"].str.upper() == "PUT", "signed_gex"] *= -1
    spot = df["underlying_price"].iloc[0]
    df["signed_gex"] *= spot ** 2
    return (df.groupby("strike")["signed_gex"].sum()
              .reset_index().rename(columns={"signed_gex": "net_gex"})
              .sort_values("strike"))


def extract_levels(gex_by_strike: pd.DataFrame, spot: float) -> dict:
    if gex_by_strike.empty:
        return {}
    above = gex_by_strike[gex_by_strike["strike"] >= spot]
    below = gex_by_strike[gex_by_strike["strike"] <= spot]

    call_resistance = (above.loc[above["net_gex"].idxmax(), "strike"]
                       if not above.empty else None)
    put_support = (below.loc[below["net_gex"].idxmin(), "strike"]
                   if not below.empty else None)

    sorted_g = gex_by_strike.sort_values("strike").reset_index(drop=True)
    sorted_g["cum_gex"] = sorted_g["net_gex"].cumsum()
    sign = sorted_g["cum_gex"].apply(lambda v: 1 if v > 0 else -1)
    flip_rows = sorted_g[sign.diff().abs() > 0]
    hvl = float(flip_rows.iloc[0]["strike"]) if not flip_rows.empty else None

    top10 = (gex_by_strike.reindex(
                gex_by_strike["net_gex"].abs().sort_values(ascending=False).index)
             .head(10))
    gex_levels = {f"GEX_{i+1}": float(s) for i, s in enumerate(top10["strike"].values)}

    return {
        "call_resistance": float(call_resistance) if call_resistance is not None else None,
        "put_support":     float(put_support) if put_support is not None else None,
        "hvl":             hvl,
        **gex_levels,
    }


def levels_for_day(date_dir: Path, max_dte: int) -> dict | None:
    g_path = date_dir / "greeks_eod.parquet"
    o_path = date_dir / "open_interest.parquet"
    if not g_path.exists() or not o_path.exists():
        return None

    greeks = pd.read_parquet(g_path)
    oi = pd.read_parquet(o_path)
    greeks["expiration"] = pd.to_datetime(greeks["expiration"])
    oi["expiration"] = pd.to_datetime(oi["expiration"])

    # Snapshot date
    snap_date = pd.to_datetime(date_dir.name)
    greeks["dte"] = (greeks["expiration"] - snap_date).dt.days

    # Filter to (1, max_dte] — exclude 0DTE (post-settle, evaporates)
    if max_dte is not None:
        greeks = greeks[(greeks["dte"] > 0) & (greeks["dte"] <= max_dte)]

    if greeks.empty:
        return None

    chain = greeks.merge(
        oi[["strike", "right", "expiration", "open_interest"]],
        on=["strike", "right", "expiration"], how="left",
    )

    spot = float(chain["underlying_price"].iloc[0])
    gex = compute_gex_by_strike(chain)
    levels = extract_levels(gex, spot)
    return {"qqq_spot": spot, **levels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dte", type=int, default=45,
                    help="Filter to expirations within N days (default 45). "
                         "Pass -1 for full chain.")
    args = ap.parse_args()
    max_dte = None if args.max_dte == -1 else args.max_dte

    print(f"loading NQ price series...")
    nq = load_nq_prices()
    print(f"  NQ: {len(nq)} bars  range {nq.index.min()} -> {nq.index.max()}")

    date_dirs = sorted([p for p in QQQ_ROOT.iterdir()
                        if p.is_dir() and len(p.name) == 10 and p.name[4] == "-"])
    print(f"  QQQ days: {len(date_dirs)}")
    print(f"  max_dte filter: {max_dte}")
    print()

    rows = []
    missing_nq = []
    for i, dd in enumerate(date_dirs, 1):
        try:
            qqq = levels_for_day(dd, max_dte)
        except Exception as e:
            print(f"[{i}/{len(date_dirs)}] {dd.name} ERROR: {type(e).__name__}: {e}")
            continue
        if qqq is None:
            continue

        date = dt.date.fromisoformat(dd.name)
        nq_spot = nq_at_settle(nq, date)
        if pd.isna(nq_spot):
            missing_nq.append(dd.name)

        ratio = nq_spot / qqq["qqq_spot"] if qqq["qqq_spot"] else float("nan")

        row = {"date": date, "qqq_spot": qqq["qqq_spot"],
               "nq_spot": nq_spot, "ratio": ratio}
        for key, qqq_lvl in qqq.items():
            if key == "qqq_spot":
                continue
            row[f"qqq_{key}"] = qqq_lvl
            row[f"nq_{key}"] = qqq_lvl * ratio if qqq_lvl is not None else None
        rows.append(row)

        if i % 100 == 0:
            print(f"[{i:>4}/{len(date_dirs)}] {dd.name}  rows={len(rows)}  "
                  f"missing_nq={len(missing_nq)}")

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df.to_parquet(OUTPUT, compression="zstd", index=False)

    print(f"\nwrote {OUTPUT}  ({len(df)} rows)")
    print(f"missing NQ price on {len(missing_nq)} days")
    if missing_nq:
        print(f"  first/last missing: {missing_nq[0]} ... {missing_nq[-1]}")
    print(f"\nsample (last 3 rows):")
    cols = ["date", "qqq_spot", "nq_spot", "ratio",
            "qqq_call_resistance", "nq_call_resistance",
            "qqq_put_support", "nq_put_support", "qqq_hvl", "nq_hvl"]
    print(df[cols].tail(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
