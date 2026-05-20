"""
Compute MAE (Maximum Adverse Excursion in $, NQ basis) per trade for all 3 strategies.
Augments combined_3way_trades.csv with mae_$ column.

MAE_$ < 0 by construction (worst unrealized loss point during the trade).
"""
from __future__ import annotations
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

NQ_PT = 20.0
TZ = "America/New_York"

# Bars sources
TIMEBARS_DIRS = [
    Path("D:/trading_pythonbacktest_data/timebars_5min_5yr"),
    Path("D:/trading_pythonbacktest_data/timebars_5min"),
]

OD_TRADES = "C:/trading/nqorderflowbacktester/live/overnight drift/trades.csv"
RV_TRADES = "C:/trading/nqorderflowbacktester/scripts/rough vol orderflow/results/inspect_v3_N400_v3_trades.csv"
B2_TRADES = "C:/trading/nqorderflowbacktester/scripts/overnight range strat/tradelogs/robust_configs/locked_v2_k08_lock045_mart_fc_filtered_trades.csv"
COMBINED  = "C:/trading/nqorderflowbacktester/scripts/rough vol orderflow/results/combined_3way_trades.csv"

OUT_COMBINED_WITH_MAE = "C:/trading/nqorderflowbacktester/scripts/rough vol orderflow/results/combined_3way_trades_with_mae.csv"


def build_20min_bars():
    files_by_date = {}
    for d in TIMEBARS_DIRS:
        for f in sorted(d.glob("timebars_5min_202*.pkl")):
            files_by_date[f.stem] = f
    frames = []
    for stem in sorted(files_by_date.keys()):
        with open(files_by_date[stem], "rb") as fh:
            bars = pickle.load(fh)
        if not bars: continue
        rows = [{"timestamp": b["open_time"], "open": b["open"], "high": b["high"],
                 "low": b["low"], "close": b["close"]} for b in bars]
        df5 = pd.DataFrame(rows).set_index("timestamp").sort_index()
        df5["group"] = df5.index.floor("20min")
        agg = df5.groupby("group").agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"),
        )
        agg.index += pd.Timedelta(minutes=20)
        frames.append(agg)
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    # Index may be DatetimeIndex (with tz) or regular Index. Normalize:
    df.index = pd.DatetimeIndex([
        (t.tz_convert(TZ) if hasattr(t, "tz_convert") and getattr(t, "tzinfo", None) is not None
         else pd.Timestamp(t).tz_localize("UTC").tz_convert(TZ))
        for t in df.index
    ])
    return df


def compute_mae_for_log(bars, trades, direction_col, entry_price_col, qty_col,
                         entry_col, exit_col, sign_map):
    """For each trade, find worst adverse excursion in $ (NQ basis $20/pt, x qty).
       sign_map: {'LONG': 1, 'SHORT': -1}
    """
    bars_idx = bars.index
    highs = bars["high"].values
    lows = bars["low"].values
    mae_list = []
    for _, t in trades.iterrows():
        ent_ts = pd.Timestamp(t[entry_col])
        ex_ts  = pd.Timestamp(t[exit_col])
        if ent_ts.tz is None: ent_ts = ent_ts.tz_localize(TZ)
        else: ent_ts = ent_ts.tz_convert(TZ)
        if ex_ts.tz is None: ex_ts = ex_ts.tz_localize(TZ)
        else: ex_ts = ex_ts.tz_convert(TZ)
        ei = bars_idx.searchsorted(ent_ts)
        xi = bars_idx.searchsorted(ex_ts) + 1
        if ei >= len(bars_idx) or xi <= ei:
            mae_list.append(0.0); continue
        sign = sign_map[t[direction_col]]
        ep = float(t[entry_price_col])
        qty = float(t[qty_col]) if qty_col else 1.0
        if sign > 0:
            worst = lows[ei:xi].min()
            mae_pts = worst - ep  # negative for adverse
        else:
            worst = highs[ei:xi].max()
            mae_pts = -(worst - ep)
        mae_list.append(float(mae_pts) * NQ_PT * qty)
    return mae_list


def derive_b2_entry_prices(bars, b2_trades):
    """B2 has no entry_price column. The strategy enters at the open of the
       5-min bar AFTER confirmation, but we approximate using the 20-min bar
       close at entry_ts (the closest available reference)."""
    bars_idx = bars.index
    closes = bars["close"].values
    entry_prices = []
    for _, t in b2_trades.iterrows():
        ent_ts = pd.Timestamp(t["entry_ts"])
        if ent_ts.tz is None: ent_ts = ent_ts.tz_localize(TZ)
        else: ent_ts = ent_ts.tz_convert(TZ)
        # use prior 20-min bar close as proxy
        ei = max(0, bars_idx.searchsorted(ent_ts) - 1)
        entry_prices.append(float(closes[ei]))
    return entry_prices


def main():
    print("Building 20-min bars...")
    bars = build_20min_bars()
    print(f"  bars: {len(bars):,}  range: {bars.index.min()} -> {bars.index.max()}\n")

    # --- OD ---
    print("Computing MAE for OD...")
    od = pd.read_csv(OD_TRADES)
    od["entry_time"] = pd.to_datetime(od["entry_time"], utc=True).dt.tz_convert(TZ)
    od["exit_time"]  = pd.to_datetime(od["exit_time"], utc=True).dt.tz_convert(TZ)
    od["direction"] = "LONG"  # OD is always long
    od["mae_$"] = compute_mae_for_log(bars, od, "direction", "entry_price", "qty",
                                       "entry_time", "exit_time", {"LONG": 1, "SHORT": -1})
    print(f"  OD: median MAE ${np.median(od['mae_$']):.0f}  worst ${od['mae_$'].min():.0f}")

    # --- RV ---
    print("Computing MAE for RV...")
    rv = pd.read_csv(RV_TRADES)
    rv["entry_ts"] = pd.to_datetime(rv["entry_ts"], utc=True).dt.tz_convert(TZ)
    rv["exit_ts"]  = pd.to_datetime(rv["exit_ts"], utc=True).dt.tz_convert(TZ)
    rv["qty"] = 1
    rv["mae_$"] = compute_mae_for_log(bars, rv, "side", "entry_price", "qty",
                                       "entry_ts", "exit_ts", {"LONG": 1, "SHORT": -1})
    print(f"  RV: median MAE ${np.median(rv['mae_$']):.0f}  worst ${rv['mae_$'].min():.0f}")

    # --- B2 ---
    print("Computing MAE for B2...")
    b2 = pd.read_csv(B2_TRADES)
    b2["entry_ts"] = pd.to_datetime(b2["entry_ts"], utc=True).dt.tz_convert(TZ)
    b2["exit_ts"]  = pd.to_datetime(b2["exit_ts"], utc=True).dt.tz_convert(TZ)
    b2["entry_price"] = derive_b2_entry_prices(bars, b2)
    # B2's 'pnl' is in NQ pts. scaled_pnl is in MNQ $. For MAE we need NQ $.
    # NQ basis MAE: same as if size=1 contract NQ.
    b2["qty"] = 1
    b2["mae_$_per_contract_NQ"] = compute_mae_for_log(bars, b2, "direction", "entry_price", "qty",
                                                       "entry_ts", "exit_ts", {"LONG": 1, "SHORT": -1})
    # Scale by 'size' column for martingale:
    b2["mae_$"] = b2["mae_$_per_contract_NQ"] * b2["size"]
    print(f"  B2: median MAE ${np.median(b2['mae_$']):.0f}  worst ${b2['mae_$'].min():.0f}")

    # --- Build augmented combined log ---
    print("\nMerging into combined log...")
    combined = pd.read_csv(COMBINED)
    combined["entry_ts"] = pd.to_datetime(combined["entry_ts"], utc=True).dt.tz_convert(TZ)
    combined["exit_ts"]  = pd.to_datetime(combined["exit_ts"], utc=True).dt.tz_convert(TZ)

    # Build a lookup: (strat, entry_ts rounded to second) -> mae_$
    def key(ts):
        return ts.replace(microsecond=0)

    mae_map = {}
    for _, r in od.iterrows():
        mae_map[("OD", key(r["entry_time"]))] = r["mae_$"]
    for _, r in rv.iterrows():
        mae_map[("RV", key(r["entry_ts"]))] = r["mae_$"]
    for _, r in b2.iterrows():
        mae_map[("B2", key(r["entry_ts"]))] = r["mae_$"]

    mae_col = []
    misses = 0
    for _, r in combined.iterrows():
        v = mae_map.get((r["strat"], key(r["entry_ts"])))
        if v is None:
            misses += 1
            mae_col.append(0.0)
        else:
            mae_col.append(v)
    combined["mae_$"] = mae_col
    combined["mae_minus_pnl_$"] = combined["mae_$"] - combined["pnl_$"]  # how much WORSE than realized
    print(f"  matched: {len(combined) - misses} / {len(combined)}  (misses={misses})")
    print(f"  median MAE ${np.median(combined['mae_$']):.0f}  worst ${combined['mae_$'].min():.0f}")
    print(f"  median (MAE - PnL) (extra unrealized pain) ${np.median(combined['mae_minus_pnl_$']):.0f}")
    print(f"  worst single trade MAE: ${combined['mae_$'].min():.0f}  on {combined.loc[combined['mae_$'].idxmin(), 'entry_ts']}")

    combined.to_csv(OUT_COMBINED_WITH_MAE, index=False)
    print(f"\nSaved -> {OUT_COMBINED_WITH_MAE}")


if __name__ == "__main__":
    main()
