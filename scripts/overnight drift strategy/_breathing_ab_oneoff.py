"""One-off: A/B the new breathing_trail yellow mode vs locked pure_ratchet."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overnight_drift_strategy import (
    StrategyParams, build_full_20min_series, run_backtest, trades_to_df,
)

PARQUET_PATH = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
PICKLE_FOLDER = "D:/trading_pythonbacktest_data/timebars_5min"


def stats(df):
    if df.empty:
        return {}
    pnl = df["pnl_dollars"].values
    wins = pnl[pnl > 0]; losses = pnl[pnl < 0]
    eq = np.cumsum(pnl); mdd = float((eq - np.maximum.accumulate(eq)).min())
    daily = pd.Series(pnl, index=df["entry_time"].dt.date.values).groupby(level=0).sum()
    sharpe = daily.mean()/daily.std()*np.sqrt(252) if daily.std() > 0 else 0.0
    return {"n": len(df), "wr": (pnl > 0).mean()*100,
            "pf": wins.sum()/abs(losses.sum()) if losses.sum() else np.inf,
            "total_$": pnl.sum(), "avg_$": pnl.mean(),
            "avg_win_$": wins.mean() if len(wins) else 0,
            "avg_loss_$": losses.mean() if len(losses) else 0,
            "max_dd_$": mdd, "sharpe": sharpe, "avg_bars": df["bars_held"].mean(),
            "tp%": (df["reason"] == "TP Green").mean()*100,
            "sl%": (df["reason"] == "SL Yellow").mean()*100,
            "be%": (df["reason"] == "BE Stop").mean()*100,
            "fc%": (df["reason"] == "Force Close").mean()*100}


def run_cfg(bars, label, **kw):
    p = StrategyParams(**kw)
    df = trades_to_df(run_backtest(bars, p))
    if not df.empty:
        df["entry_time"] = pd.to_datetime(df["entry_time"]).dt.tz_convert("America/New_York")
    s = stats(df); s["cfg"] = label
    return s


def main():
    bars = build_full_20min_series(PARQUET_PATH, PICKLE_FOLDER)
    rows = [
        run_cfg(bars, "pure_ratchet (locked)", yellow_mode="pure_ratchet"),
        run_cfg(bars, "breathing gap2.0 up2 dn1 floor70", yellow_mode="breathing_trail"),
        run_cfg(bars, "breathing gap1.7 up2 dn1 floor60", yellow_mode="breathing_trail",
                min_gap_atr=1.7, trail_up_step=2.0, trail_down_step=1.0, max_room_pts=60),
        run_cfg(bars, "breathing gap2.5 up3 dn1.5 floor90", yellow_mode="breathing_trail",
                min_gap_atr=2.5, trail_up_step=3.0, trail_down_step=1.5, max_room_pts=90),
    ]
    out = pd.DataFrame(rows).set_index("cfg")
    cols = ["n","wr","pf","total_$","avg_$","avg_win_$","avg_loss_$","max_dd_$",
            "sharpe","avg_bars","tp%","sl%","be%","fc%"]
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(out[cols].round(2).to_string())


if __name__ == "__main__":
    main()
