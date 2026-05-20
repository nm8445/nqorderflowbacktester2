"""
Prop firm pass-rate optimizer.

Strategy: every weekday at 19:00 ET, long 1 NQ contract. Fixed-point TP / SL.
Force-close at 08:00 ET.

Account rules (Apex-style $50K eval):
  - Start P&L = 0
  - Trailing drawdown = $2000 below the running peak (peak = max of running P&L,
    floored at 0 -- i.e. trailing only starts to bite once you've gained ground;
    DD is from peak realized P&L). Account "busts" if cum P&L <= peak - $2000.
  - Profit target = $3000 (configurable).
  - Max trades per eval = 30 (configurable).
  - "Pass" = cum P&L reaches target *before* busting and within max trades.
  - Cost per account = $100. Payout when passing (handled separately).

For each SL in {25, 35, 50, 75, 100, 150, 200} NQ points:
  - Compute per-trade P&L stream across the full 5.5 yrs.
  - For each possible start day (across all trades), simulate an account.
  - Pass rate = passes / total starts.
  - Split into IS (2020-12 .. 2023-12) and OOS (2024-01 .. 2026-05).
  - EV per account = P(pass) * payout - cost.
  - 5-account EV = 5 * EV per account.
"""

from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overnight_drift_strategy import build_full_20min_series  # noqa: E402

NQ_POINT_VALUE = 20.0
PARQUET_PATH = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
PICKLE_FOLDER = "D:/trading_pythonbacktest_data/timebars_5min"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)
TZ = "America/New_York"

# Account params
TRAILING_DD = 2000.0
PROFIT_TARGET = 3000.0
MAX_TRADES = 30
COST_PER_ACCOUNT = 100.0
PAYOUT_ON_PASS = 1500.0  # net payout assumption (will sweep this if wanted)

# Strategy params
ENTRY_T = time(19, 0)
FORCE_T = time(8, 0)
TP_POINTS = 50.0  # $1000 per contract


def run_fixed(bars: pd.DataFrame, sl_points: float) -> pd.DataFrame:
    """Run the long-at-19:00 / TP+50pt / SL-X pt / force-close-08:00 strategy."""
    o = bars["open"].values
    h = bars["high"].values
    l = bars["low"].values
    c = bars["close"].values
    idx = bars.index

    rows = []
    in_pos = False
    entry = np.nan
    tp = np.nan
    sl = np.nan
    et = None
    for i in range(len(bars)):
        ts = idx[i]
        t_local = ts.time()
        if not in_pos and t_local == ENTRY_T:
            in_pos = True
            entry = c[i]
            tp = entry + TP_POINTS
            sl = entry - sl_points
            et = ts
            continue
        if in_pos:
            exit_p = np.nan
            reason = ""
            # Gap-through logic (open beyond level => fill at open)
            if o[i] <= sl:
                exit_p, reason = o[i], "SL"
            elif o[i] >= tp:
                exit_p, reason = o[i], "TP"
            else:
                hit_sl = l[i] <= sl
                hit_tp = h[i] >= tp
                if hit_sl and hit_tp:
                    exit_p, reason = sl, "SL"  # worst-case
                elif hit_sl:
                    exit_p, reason = sl, "SL"
                elif hit_tp:
                    exit_p, reason = tp, "TP"
                elif t_local == FORCE_T:
                    exit_p, reason = c[i], "Force"
            if not np.isnan(exit_p):
                rows.append(
                    {
                        "entry_time": et,
                        "exit_time": ts,
                        "entry": entry,
                        "exit": exit_p,
                        "reason": reason,
                        "pnl_$": (exit_p - entry) * NQ_POINT_VALUE,
                    }
                )
                in_pos = False
                entry = np.nan
    return pd.DataFrame(rows)


def simulate_account(pnls: np.ndarray, target: float, dd: float, max_trades: int) -> tuple[str, int]:
    """Walk a P&L stream forward. Return (outcome, n_trades_used).

    outcome: 'pass' | 'bust' | 'timeout'
    """
    cum = 0.0
    peak = 0.0
    for i, p in enumerate(pnls[:max_trades]):
        cum += p
        if cum > peak:
            peak = cum
        if cum <= peak - dd:
            return "bust", i + 1
        if cum >= target:
            return "pass", i + 1
    return "timeout", min(len(pnls), max_trades)


def sweep_sl(bars: pd.DataFrame) -> None:
    is_end = pd.Timestamp("2024-01-01", tz=TZ)

    sl_values = [25, 35, 50, 75, 100, 150, 200]
    rows = []
    detail = {}

    for sl in sl_values:
        trades = run_fixed(bars, sl)
        trades["entry_time"] = pd.to_datetime(trades["entry_time"]).dt.tz_convert(TZ)
        trades = trades.sort_values("entry_time").reset_index(drop=True)

        # Per-trade summary
        wr = (trades["pnl_$"] > 0).mean() * 100
        tp_cnt = (trades["reason"] == "TP").sum()
        sl_cnt = (trades["reason"] == "SL").sum()
        fc_cnt = (trades["reason"] == "Force").sum()
        avg_pnl = trades["pnl_$"].mean()
        worst = trades["pnl_$"].min()

        # Split IS/OOS
        is_mask = trades["entry_time"] < is_end
        is_trades = trades[is_mask].reset_index(drop=True)
        oos_trades = trades[~is_mask].reset_index(drop=True)

        # Run account sim starting at every trade index (rolling)
        def evals(trade_df: pd.DataFrame) -> dict:
            pnls = trade_df["pnl_$"].to_numpy()
            n = len(pnls)
            if n < MAX_TRADES // 2:
                return {"starts": 0, "pass": 0, "bust": 0, "timeout": 0, "pass_rate": np.nan}
            outcomes = {"pass": 0, "bust": 0, "timeout": 0}
            for s in range(n - 5):  # need at least 5 trades available
                stream = pnls[s:]
                outcome, _ = simulate_account(stream, PROFIT_TARGET, TRAILING_DD, MAX_TRADES)
                outcomes[outcome] += 1
            tot = sum(outcomes.values())
            return {"starts": tot, **outcomes, "pass_rate": outcomes["pass"] / tot * 100 if tot else np.nan}

        is_res = evals(is_trades)
        oos_res = evals(oos_trades)
        all_res = evals(trades)

        row = {
            "SL_pts": sl,
            "SL_$": sl * NQ_POINT_VALUE,
            "RR": round(TP_POINTS / sl, 2),
            "total_trades": len(trades),
            "win%": round(wr, 1),
            "TP": tp_cnt,
            "SL": sl_cnt,
            "Force": fc_cnt,
            "avg_$/trade": round(avg_pnl, 1),
            "worst_$/trade": round(worst, 0),
            "IS_pass%": round(is_res["pass_rate"], 1),
            "IS_bust%": round(is_res["bust"] / is_res["starts"] * 100, 1) if is_res["starts"] else np.nan,
            "OOS_pass%": round(oos_res["pass_rate"], 1),
            "OOS_bust%": round(oos_res["bust"] / oos_res["starts"] * 100, 1) if oos_res["starts"] else np.nan,
            "ALL_pass%": round(all_res["pass_rate"], 1),
            "EV_per_acct_$": round(all_res["pass_rate"] / 100 * PAYOUT_ON_PASS - COST_PER_ACCOUNT, 1),
            "EV_5_accts_$": round(5 * (all_res["pass_rate"] / 100 * PAYOUT_ON_PASS - COST_PER_ACCOUNT), 1),
            "P_>=1_of_5_pass%": round(
                (1 - (1 - all_res["pass_rate"] / 100) ** 5) * 100, 1
            ),
        }
        rows.append(row)
        detail[sl] = {"trades": trades, "is": is_res, "oos": oos_res, "all": all_res}

    out = pd.DataFrame(rows)
    print(f"\n{'='*120}")
    print(f"Prop firm sim: TP=+50pt (\$1000), trailing DD=\${TRAILING_DD:.0f}, "
          f"target=\${PROFIT_TARGET:.0f}, max {MAX_TRADES} trades/eval")
    print(f"Cost=\${COST_PER_ACCOUNT:.0f}/acct, payout=\${PAYOUT_ON_PASS:.0f} (assumed)")
    print(f"{'='*120}")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(out.to_string(index=False))

    # Robustness check: pass rate must be similar in IS and OOS
    out["IS_OOS_diff"] = out["IS_pass%"] - out["OOS_pass%"]
    print(f"\n=== Robustness check (IS - OOS pass rate; closer to 0 is more stable) ===")
    print(out[["SL_pts", "IS_pass%", "OOS_pass%", "IS_OOS_diff", "ALL_pass%", "EV_5_accts_$"]].to_string(index=False))

    out.to_csv(OUT_DIR / "prop_firm_sweep.csv", index=False)
    print(f"\nSaved -> {OUT_DIR / 'prop_firm_sweep.csv'}")


def main() -> None:
    print("Loading bars...", flush=True)
    bars = build_full_20min_series(PARQUET_PATH, PICKLE_FOLDER)
    print(f"  bars: {len(bars):,}  range: {bars.index.min()} -> {bars.index.max()}", flush=True)
    sweep_sl(bars)


if __name__ == "__main__":
    main()
