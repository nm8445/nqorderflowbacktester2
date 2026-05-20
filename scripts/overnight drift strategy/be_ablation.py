"""
BE rule ablation: run the strategy with use_be=True and use_be=False and
compare. Then go further and look at each BE-stopped trade in the BE=True run
and compute what would have happened to it without the BE stop -- the "BE
saved me" vs "BE cost me" attribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overnight_drift_strategy import (  # noqa: E402
    StrategyParams,
    build_full_20min_series,
    run_backtest,
    trades_to_df,
)

PARQUET_PATH = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
PICKLE_FOLDER = "D:/trading_pythonbacktest_data/timebars_5min"


def summarise(name: str, df: pd.DataFrame) -> None:
    pnl = df["pnl_dollars"].sum()
    wr = (df["pnl_dollars"] > 0).mean() * 100
    gw = df.loc[df["pnl_dollars"] > 0, "pnl_dollars"].sum()
    gl = abs(df.loc[df["pnl_dollars"] < 0, "pnl_dollars"].sum())
    pf = gw / gl if gl else float("inf")
    print(
        f"{name:>14}  trades={len(df):>5}  total=${pnl:>10,.0f}  "
        f"avg=${df['pnl_dollars'].mean():>7,.1f}  win%={wr:5.1f}  PF={pf:4.2f}"
    )
    print(f"               exits: {df['reason'].value_counts().to_dict()}")


def main() -> None:
    bars = build_full_20min_series(PARQUET_PATH, PICKLE_FOLDER)
    print(f"bars: {len(bars):,}  range: {bars.index.min()} -> {bars.index.max()}\n")

    trades_on = trades_to_df(run_backtest(bars, StrategyParams(use_be=True)))
    trades_off = trades_to_df(run_backtest(bars, StrategyParams(use_be=False)))

    summarise("BE ON", trades_on)
    summarise("BE OFF", trades_off)
    diff = trades_on["pnl_dollars"].sum() - trades_off["pnl_dollars"].sum()
    print(f"\nNet effect of BE: ${diff:,.0f}  (positive = BE helps)")

    # ---- Match by entry_time and attribute per trade ----
    on = trades_on.set_index("entry_time")
    off = trades_off.set_index("entry_time")
    common = on.index.intersection(off.index)
    on_c = on.loc[common]
    off_c = off.loc[common]
    delta = (on_c["pnl_dollars"] - off_c["pnl_dollars"]).rename("delta_$")

    cmp = pd.DataFrame(
        {
            "on_reason": on_c["reason"],
            "off_reason": off_c["reason"],
            "on_pnl": on_c["pnl_dollars"],
            "off_pnl": off_c["pnl_dollars"],
            "delta_$": delta,
        }
    )

    # Trades where BE actually fired
    be_fired = cmp[cmp["on_reason"] == "BE Stop"]
    print(f"\nTrades where BE fired: {len(be_fired)}")
    print(f"  net delta from BE-fired trades: ${be_fired['delta_$'].sum():,.0f}")
    print(
        f"  saved (delta>0): {(be_fired['delta_$'] > 0).sum()}  "
        f"cost (delta<0): {(be_fired['delta_$'] < 0).sum()}  "
        f"flat: {(be_fired['delta_$'] == 0).sum()}"
    )
    print(
        f"  total saved $: {be_fired.loc[be_fired['delta_$'] > 0, 'delta_$'].sum():,.0f}  "
        f"total cost $: {be_fired.loc[be_fired['delta_$'] < 0, 'delta_$'].sum():,.0f}"
    )
    print("\n  Where the BE-fired trades would have ended without BE:")
    print(be_fired["off_reason"].value_counts().to_string())
    print("\n  Avg P&L of those trades in the BE-OFF universe (would-have-been):")
    print(
        be_fired.groupby("off_reason")["off_pnl"].agg(["count", "mean", "sum"]).round(0).to_string()
    )

    # All trades that differ at all
    differ = cmp[cmp["delta_$"] != 0]
    print(f"\nTotal trades whose outcome differs with BE on: {len(differ)}  "
          f"(net ${differ['delta_$'].sum():,.0f})")


if __name__ == "__main__":
    main()
