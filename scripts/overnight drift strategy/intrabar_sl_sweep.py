"""
Intrabar tick-stop SL variant for the overnight drift strategy.

Original behavior:
  SL Yellow fires when: close <= yellow_val AND close < open  (bar-close trigger)

New behavior:
  SL fires INTRABAR when: low <= (prev_yellow - cushion_pts)
    - Gap-through: if bar open is already <= sl_level, fill at open
    - Else fill at sl_level
  This catches the situations where the bar closes far below yellow because
  the close-trigger only fires at end of bar -- a real-time stop fires when
  the level is touched.

We sweep cushion_pts from 1.0 to 10.0 in 0.5 increments (19 values).
TP and Force Close logic unchanged. BE off. Martingale 1/2.

Config (the robust pick from the constrained sweep):
  yellow_atr_len   = 14
  yellow_atr_mult  = 1.30
  yellow_mode      = pure_ratchet
  green_atr_len    = 14
  green_atr_mult   = 1.00
  green_base       = 82.5
  green_decay      = 1.5
  red_intercept    = 0.0
  red_drift        = 0.45

Same fold structure as before.
"""

from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overnight_drift_strategy import build_full_20min_series, rma_atr  # noqa: E402

NQ_POINT_VALUE = 20.0
PARQUET = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
PICKLES = "D:/trading_pythonbacktest_data/timebars_5min"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
TZ = "America/New_York"

FOLDS = [
    ("Pre", "2020-12-01", "2022-11-30"),
    ("F1",  "2022-12-01", "2023-11-30"),
    ("F2",  "2023-12-01", "2024-11-30"),
    ("F3",  "2024-12-01", "2025-11-30"),
    ("F4",  "2025-12-01", "2026-05-31"),
]


def run_intrabar_sl(
    bars: pd.DataFrame,
    sl_cushion: float,
    y_atr_len: int = 14,
    y_mult: float = 1.30,
    g_atr_len: int = 14,
    g_mult: float = 1.00,
    g_base: float = 82.5,
    g_decay: float = 1.5,
    red_intercept: float = 0.0,
    red_drift: float = 0.45,
    entry_t: time = time(19, 0),
    force_t: time = time(8, 0),
    use_martingale: bool = True,
    base_qty: int = 1,
    loss_qty: int = 2,
) -> pd.DataFrame:
    atr_y = rma_atr(bars["high"], bars["low"], bars["close"], y_atr_len).values
    atr_g = rma_atr(bars["high"], bars["low"], bars["close"], g_atr_len).values
    o = bars["open"].values
    h = bars["high"].values
    l = bars["low"].values
    c = bars["close"].values
    idx = bars.index

    rows = []
    in_pos = False
    entry_price = np.nan
    entry_idx = -1
    entry_qty = 0
    yellow_val = np.nan
    prev_yellow = np.nan
    marti_state = 0
    next_qty = base_qty

    for i in range(len(bars)):
        ts = idx[i]
        t_local = ts.time()
        ay = atr_y[i]
        ag = atr_g[i]

        # Entry
        if not in_pos and t_local == entry_t and not np.isnan(ay):
            if use_martingale:
                if marti_state == 0:
                    next_qty = base_qty
                elif marti_state == 1:
                    next_qty = loss_qty
                else:
                    next_qty = base_qty
            else:
                next_qty = base_qty
            in_pos = True
            entry_price = c[i]
            entry_idx = i
            entry_qty = int(next_qty)
            yellow_val = entry_price - y_mult * ay
            prev_yellow = np.nan
            continue

        if in_pos:
            bars_in_trade = i - entry_idx

            # Compute current bar's yellow (pure_ratchet)
            raw_yellow = c[i] - y_mult * ay if not np.isnan(ay) else np.nan
            if not np.isnan(prev_yellow) and not np.isnan(raw_yellow):
                yellow_val = max(prev_yellow, raw_yellow)
            elif not np.isnan(raw_yellow):
                yellow_val = raw_yellow
            # Compute red and green
            red_val = entry_price + red_intercept + red_drift * bars_in_trade
            green_val = red_val + g_base - g_decay * bars_in_trade + g_mult * ag if not np.isnan(ag) else np.nan

            exited = False
            exit_price = np.nan
            reason = ""

            # 1. NEW SL: intrabar at (prev_yellow - cushion)
            if not np.isnan(prev_yellow):
                sl_level = prev_yellow - sl_cushion
                if o[i] <= sl_level:
                    exit_price = o[i]  # gap-through
                    reason = "SL Yellow"
                    exited = True
                elif l[i] <= sl_level:
                    exit_price = sl_level
                    reason = "SL Yellow"
                    exited = True

            # 2. TP Green: high >= green, exit at bar close (Pine semantics)
            if not exited and not np.isnan(green_val) and h[i] >= green_val:
                exit_price = c[i]
                reason = "TP Green"
                exited = True

            # 3. Force close
            if not exited and t_local == force_t:
                exit_price = c[i]
                reason = "Force Close"
                exited = True

            if exited:
                pnl_pts = exit_price - entry_price
                pnl_dollars = pnl_pts * NQ_POINT_VALUE * entry_qty
                rows.append(
                    {
                        "entry_time": idx[entry_idx],
                        "exit_time": ts,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "qty": entry_qty,
                        "reason": reason,
                        "bars_held": bars_in_trade,
                        "pnl_points": pnl_pts,
                        "pnl_dollars": pnl_dollars,
                    }
                )
                # Marti update
                last_was_loss = pnl_dollars < 0
                if marti_state == 0:
                    marti_state = 1 if last_was_loss else 0
                elif marti_state == 1:
                    marti_state = 2
                else:
                    marti_state = 1 if last_was_loss else 0
                in_pos = False
                entry_price = np.nan
                entry_idx = -1
                entry_qty = 0
                yellow_val = np.nan
                prev_yellow = np.nan
                continue

            prev_yellow = yellow_val
    return pd.DataFrame(rows)


def stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"trades": 0, "win%": np.nan, "PF": np.nan, "gross": 0.0, "avg": np.nan}
    pnl = df["pnl_dollars"].values
    wins = (pnl > 0).sum()
    gw = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    return {
        "trades": int(len(pnl)),
        "win%": float(wins / len(pnl) * 100),
        "PF": float(gw / gl) if gl > 0 else float("inf"),
        "gross": float(pnl.sum()),
        "avg": float(pnl.mean()),
    }


def maxdd(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    s = df.sort_values("entry_time").copy()
    s["c"] = s["pnl_dollars"].cumsum()
    s["p"] = s["c"].cummax()
    return float((s["c"] - s["p"]).min())


def main() -> None:
    print("Loading bars...", flush=True)
    bars = build_full_20min_series(PARQUET, PICKLES)
    print(f"  bars: {len(bars):,}  range: {bars.index.min()} -> {bars.index.max()}\n", flush=True)

    cushions = [round(1.0 + 0.5 * i, 1) for i in range(19)]
    rows = []
    for cu in cushions:
        df = run_intrabar_sl(bars, sl_cushion=cu)
        df["entry_time"] = pd.to_datetime(df["entry_time"]).dt.tz_convert(TZ)
        s_all = stats(df)
        dd = maxdd(df)
        row = {
            "cushion_pts": cu,
            "trades": s_all["trades"],
            "win%": s_all["win%"],
            "PF": s_all["PF"],
            "gross": s_all["gross"],
            "avg_$/tr": s_all["avg"],
            "MaxDD": dd,
            "TP_hits": int((df["reason"] == "TP Green").sum()),
            "SL_hits": int((df["reason"] == "SL Yellow").sum()),
            "Force_hits": int((df["reason"] == "Force Close").sum()),
        }
        # Per-fold PF
        pfs = []
        for lbl, lo, hi in FOLDS:
            m = (df["entry_time"] >= pd.Timestamp(lo, tz=TZ)) & (df["entry_time"] <= pd.Timestamp(hi + " 23:59:59", tz=TZ))
            sub = df[m]
            s = stats(sub)
            row[f"{lbl}_PF"] = s["PF"]
            row[f"{lbl}_gross"] = s["gross"]
            pfs.append(s["PF"])
        finite = [p for p in pfs if np.isfinite(p)]
        row["min_fold_PF"] = min(finite) if finite else np.nan
        rows.append(row)

    g = pd.DataFrame(rows)

    print("=== All-period stats by cushion (pts below prev_yellow) ===")
    show = g[["cushion_pts", "trades", "win%", "PF", "gross", "MaxDD",
              "TP_hits", "SL_hits", "Force_hits", "min_fold_PF"]].copy()
    with pd.option_context("display.width", 200):
        print(show.round(2).to_string(index=False))

    print("\n=== Per-fold PF ===")
    fold_show = g[["cushion_pts", "Pre_PF", "F1_PF", "F2_PF", "F3_PF", "F4_PF", "PF", "min_fold_PF"]].copy()
    with pd.option_context("display.width", 200):
        print(fold_show.round(2).to_string(index=False))

    print("\n=== Per-fold gross $ ===")
    fold_gross = g[["cushion_pts", "Pre_gross", "F1_gross", "F2_gross", "F3_gross", "F4_gross", "gross"]].copy()
    with pd.option_context("display.width", 200):
        print(fold_gross.round(0).to_string(index=False))

    # Baseline (original bar-close SL, same config)
    print("\n=== Baseline (original close-based SL Yellow, same config) ===")
    from overnight_drift_strategy import StrategyParams, run_backtest, trades_to_df
    p = StrategyParams(
        yellow_atr_len=14, yellow_atr_mult=1.30, yellow_drift=0.0, yellow_mode="pure_ratchet",
        green_atr_len=14, green_atr_mult=1.00, green_base=82.5, green_decay=1.5,
        red_intercept=0.0, red_drift=0.45,
        use_be=False, use_martingale=True, base_qty=1, loss_qty=2,
    )
    base_df = trades_to_df(run_backtest(bars, p))
    base_df["entry_time"] = pd.to_datetime(base_df["entry_time"]).dt.tz_convert(TZ)
    b = stats(base_df)
    print(f"  trades={b['trades']}  win%={b['win%']:.1f}  PF={b['PF']:.2f}  gross=${b['gross']:,.0f}  MaxDD=${maxdd(base_df):,.0f}")
    base_pfs = []
    for lbl, lo, hi in FOLDS:
        m = (base_df["entry_time"] >= pd.Timestamp(lo, tz=TZ)) & (base_df["entry_time"] <= pd.Timestamp(hi + " 23:59:59", tz=TZ))
        sub = base_df[m]
        bs = stats(sub)
        base_pfs.append(bs["PF"])
        print(f"  {lbl}: PF={bs['PF']:.2f} gross=${bs['gross']:,.0f}")

    g.to_csv(OUT / "intrabar_sl_sweep.csv", index=False)
    print(f"\nSaved -> {OUT / 'intrabar_sl_sweep.csv'}")


if __name__ == "__main__":
    main()
