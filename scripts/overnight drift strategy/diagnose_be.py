"""
BE-rule diagnostic. Verifies:
  1) BE arm time and BE exit time are SPACED OUT (no same-bar arm-and-fire).
  2) Gap-down entries that breach yellow on bar 1 exit as real losses, not as
     $0 BE stops.
  3) Distribution of bars between entry -> BE arm and BE arm -> BE exit.
  4) Spot-checks: prints the bar tape for a handful of BE-stopped trades so
     the logic can be eyeballed.
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
    rma_atr,
    run_backtest,
    trades_to_df,
)

PARQUET_PATH = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
PICKLE_FOLDER = "D:/trading_pythonbacktest_data/timebars_5min"


def main() -> None:
    bars = build_full_20min_series(PARQUET_PATH, PICKLE_FOLDER)
    params = StrategyParams()
    trades = trades_to_df(run_backtest(bars, params))

    trades["entry_time"] = pd.to_datetime(trades["entry_time"]).dt.tz_convert(
        "America/New_York"
    )
    trades["exit_time"] = pd.to_datetime(trades["exit_time"]).dt.tz_convert(
        "America/New_York"
    )
    trades["be_arm_time"] = pd.to_datetime(trades["be_arm_time"], utc=True, errors="coerce")
    has_arm = trades["be_arm_time"].notna()
    trades.loc[has_arm, "be_arm_time"] = trades.loc[has_arm, "be_arm_time"].dt.tz_convert(
        "America/New_York"
    )

    print(f"Total trades: {len(trades)}")
    print(f"BE-Stop exits: {(trades['reason'] == 'BE Stop').sum()}")
    print(f"Trades where BE armed (any reason): {has_arm.sum()}")

    armed = trades[has_arm].copy()
    armed["bars_arm_to_exit"] = armed["bars_held"] - armed["be_arm_bar"]

    print("\n--- Spacing: bars between BE arm and exit ---")
    print("(0 would indicate a same-bar arm+fire bug)")
    print(armed["bars_arm_to_exit"].value_counts().sort_index().head(15).to_string())
    same_bar = (armed["bars_arm_to_exit"] == 0).sum()
    print(f"\nSame-bar arm+exit count: {same_bar}  <-- should be 0")

    be_exits = trades[trades["reason"] == "BE Stop"].copy()
    be_exits["bars_arm_to_exit"] = be_exits["bars_held"] - be_exits["be_arm_bar"]
    print("\n--- BE-Stop exits: bars from arm to exit ---")
    print(be_exits["bars_arm_to_exit"].describe().round(2).to_string())
    print("\nBE-Stop bars-held distribution (entry -> exit):")
    print(be_exits["bars_held"].describe().round(2).to_string())
    print("\nBE-arm-bar (entry -> arm) distribution:")
    print(be_exits["be_arm_bar"].describe().round(2).to_string())

    # Bar 1 BE arm + Bar 2 exit?  Earliest possible.
    earliest = be_exits[(be_exits["be_arm_bar"] == 1) & (be_exits["bars_held"] == 2)]
    print(f"\nEarliest possible (arm bar 1, exit bar 2): {len(earliest)} trades")

    # ------------------------------------------------------------------
    # Gap-down sanity: trades whose first post-entry bar's low went well
    # below entry. Those should NOT exit as $0 BE -- they should exit as
    # SL Yellow at a real loss.
    # ------------------------------------------------------------------
    bars_et = bars  # already in NY tz
    bar_idx = bars_et.index
    pos = {ts: i for i, ts in enumerate(bar_idx)}

    gap_results = []
    for _, t in trades.iterrows():
        ent_ts = t["entry_time"]
        if ent_ts not in pos:
            continue
        i = pos[ent_ts]
        if i + 1 >= len(bars_et):
            continue
        b0 = bars_et.iloc[i]
        b1 = bars_et.iloc[i + 1]
        gap_pts = b1["low"] - b0["close"]
        gap_results.append(
            {
                "entry_time": ent_ts,
                "entry_close": b0["close"],
                "next_low": b1["low"],
                "next_open": b1["open"],
                "next_close": b1["close"],
                "gap_low_pts": gap_pts,
                "exit_reason": t["reason"],
                "pnl_dollars": t["pnl_dollars"],
                "exit_time": t["exit_time"],
                "be_arm_bar": t["be_arm_bar"],
            }
        )
    gap_df = pd.DataFrame(gap_results)
    sharp = gap_df[gap_df["gap_low_pts"] < -10].sort_values("gap_low_pts")
    print(f"\n--- Sharp gap-down on bar+1 (low - entry_close < -10 pts) ---")
    print(f"Count: {len(sharp)}")
    print("\nExit reason breakdown for those entries:")
    print(sharp["exit_reason"].value_counts().to_string())
    print("\nP&L stats by exit reason:")
    print(sharp.groupby("exit_reason")["pnl_dollars"].describe().round(0).to_string())

    # Surface any BE Stop with a sharp gap (suspicious)
    suspicious = sharp[sharp["exit_reason"] == "BE Stop"]
    print(f"\nBE Stop exits among sharp gap-downs: {len(suspicious)}")
    if len(suspicious):
        print("First 5:")
        print(
            suspicious.head(5)[
                [
                    "entry_time",
                    "entry_close",
                    "next_low",
                    "next_close",
                    "gap_low_pts",
                    "be_arm_bar",
                    "exit_time",
                ]
            ].to_string(index=False)
        )

    # ------------------------------------------------------------------
    # Spot-check: print bar tape for 3 BE-stopped trades
    # ------------------------------------------------------------------
    print("\n\n=== Spot-check: bar tape for sample BE-stopped trades ===")
    samples = be_exits.sample(min(3, len(be_exits)), random_state=7)
    atr_y = rma_atr(bars["high"], bars["low"], bars["close"], params.yellow_atr_len)
    for _, t in samples.iterrows():
        ent_ts = t["entry_time"]
        i = pos[ent_ts]
        ext_ts = t["exit_time"]
        j = pos[ext_ts]
        print(f"\nEntry {ent_ts}  exit {ext_ts}  reason={t['reason']}  pnl=${t['pnl_dollars']:.0f}")
        print(f"  BE arm bar = {t['be_arm_bar']}  arm time = {t['be_arm_time']}")
        slc = bars.iloc[i : j + 1].copy()
        slc["atr_y"] = atr_y.iloc[i : j + 1]
        print(slc[["open", "high", "low", "close", "atr_y"]].round(2).to_string())


if __name__ == "__main__":
    main()
