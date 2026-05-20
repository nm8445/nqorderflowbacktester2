"""
Fabio Valentini ORB — 5-min NQ backtest with REAL delta (buy_vol - sell_vol).

Long-only ORB breakout: 08:30-09:00 ET opening range, entries on bar-close above
ORB_High while inside 09:00-14:00 ET trading window. SL = ORB_Low,
TP = entry + TP_RR_Ratio * (entry - ORB_Low). EOD flat at 14:00 ET.

Splits dates 60/40 chronologically. Sweeps:
    TP_RR_Ratio   in [0.50, 0.75, ..., 5.00]
    DeltaThreshold in [0, 25, 50, ..., 500] (contracts)

Outputs written to D:/trading_pythonbacktest_data/fabio orb/:
    trades_IS{RUN_SUFFIX}.csv, trades_OOS.csv
    sweep_summary_IS.csv, sweep_summary_OOS.csv
    is_vs_oos_top20.csv
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

VOL_PARQUET = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
TRADE_BARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min_trades_5yr")
OUT_DIR = Path("D:/trading_pythonbacktest_data/fabio orb")
ET = "America/New_York"

# Fabio defaults — Trade_End cutoff overridable via env FABIO_TRADE_END (e.g. 1500)
# DELTA_SOURCE: "contracts" (volumetric parquet, buy_vol-sell_vol)
#               "upticks" (trade-bar pkls, uptick_count-downtick_count)
import os as _os
ORB_START_HHMM = 830
ORB_END_HHMM = 900
TRADE_END_HHMM = int(_os.environ.get("FABIO_TRADE_END", "1400"))
DELTA_SOURCE = _os.environ.get("FABIO_DELTA", "contracts").lower()
RUN_SUFFIX = f"_te{TRADE_END_HHMM}_{DELTA_SOURCE}"

# Cost model
TICK_SIZE = 0.25
TICK_VALUE = 5.0       # $/contract/tick
DOLLARS_PER_POINT = TICK_VALUE / TICK_SIZE   # = 20
SLIP_TICKS_PER_SIDE = 1
COMMISSION_RT = 5.0    # $/contract round trip


# ============================== data loading ==============================

def _load_from_volumetric_parquet() -> pd.DataFrame:
    df = pd.read_parquet(VOL_PARQUET)
    agg = df.groupby("bar_open_time", as_index=False).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"),
    )
    agg["delta_value"] = agg["buy_vol"] - agg["sell_vol"]   # contracts
    return agg


def _load_from_trade_bar_pkls() -> pd.DataFrame:
    """Load all per-day trade-bar pkls (built by build_trade_5min_bars.py).
    Bars are UTC-aware; convert to ET. Delta = uptick_count - downtick_count (trade counts)."""
    rows = []
    for f in sorted(TRADE_BARS_DIR.glob("timebars_5min_*.pkl")):
        try:
            with open(f, "rb") as fp:
                rows.extend(pickle.load(fp))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["bar_open_time"] = pd.to_datetime(df["open_time"]).dt.tz_convert(ET)
    df["delta_value"] = df["uptick_count"].astype("float64") - df["downtick_count"].astype("float64")
    return df[["bar_open_time", "open", "high", "low", "close", "delta_value"]]


def load_bars() -> dict[pd.Timestamp, pd.DataFrame]:
    """Switchable loader. Returns per-session-date DataFrames with cols:
       bar_open_time, close_et, hhmm, open/high/low/close, delta_value."""
    if DELTA_SOURCE == "upticks":
        agg = _load_from_trade_bar_pkls()
    else:
        agg = _load_from_volumetric_parquet()
    agg = agg.dropna(subset=["open", "high", "low", "close"])
    agg["close_et"] = agg["bar_open_time"] + pd.Timedelta(minutes=5)
    agg["hhmm"] = agg["close_et"].dt.hour * 100 + agg["close_et"].dt.minute
    agg = agg[(agg["hhmm"] > ORB_START_HHMM) & (agg["hhmm"] <= TRADE_END_HHMM)].copy()
    if agg.empty:
        return {}
    agg["session_date"] = agg["close_et"].dt.normalize()
    out: dict[pd.Timestamp, pd.DataFrame] = {}
    for sd, sub in agg.groupby("session_date"):
        sub = sub.sort_values("close_et").reset_index(drop=True)
        out[pd.Timestamp(sd).tz_localize(None)] = sub
    return out


def precompute_days(bars_by_date: dict[pd.Timestamp, pd.DataFrame]) -> dict[pd.Timestamp, dict]:
    """Per-day: lock in ORB (08:30-09:00 ET), extract numpy arrays for post-ORB bars."""
    days: dict[pd.Timestamp, dict] = {}
    for d, df in bars_by_date.items():
        in_orb_mask = (df["hhmm"] > ORB_START_HHMM) & (df["hhmm"] <= ORB_END_HHMM)
        if not in_orb_mask.any():
            continue
        in_orb = df[in_orb_mask]
        orb_high = float(in_orb["high"].max())
        orb_low = float(in_orb["low"].min())
        post = df[(df["hhmm"] > ORB_END_HHMM) & (df["hhmm"] <= TRADE_END_HHMM)]
        if post.empty:
            continue
        days[d] = {
            "orb_high": orb_high,
            "orb_low": orb_low,
            "hhmm": post["hhmm"].to_numpy(),
            "high": post["high"].to_numpy(dtype=np.float64),
            "low": post["low"].to_numpy(dtype=np.float64),
            "close": post["close"].to_numpy(dtype=np.float64),
            # Bar delta — either contracts (volumetric) or trade-count (upticks) depending on DELTA_SOURCE
            "delta": post["delta_value"].to_numpy(dtype=np.float64),
            "close_et": post["close_et"].to_numpy(),
        }
    return days


# ============================== single-day backtest ==============================

def run_day(day: dict, tp_rr: float, delta_thr: float) -> dict | None:
    """Fabio ORB on one preprocessed session. Returns trade dict or None."""
    oh = day["orb_high"]
    ol = day["orb_low"]
    hhmm = day["hhmm"]
    high = day["high"]
    low = day["low"]
    close = day["close"]
    delta = day["delta"]
    etime = day["close_et"]
    n = len(hhmm)

    # Find first qualifying entry
    entry_idx = -1
    entry_price = sl = tp = 0.0
    entry_time = None
    for i in range(n):
        # entries allowed through the bar that closes at TRADE_END (inclusive of 14:00)
        if hhmm[i] > TRADE_END_HHMM:
            break
        if close[i] > oh and delta[i] >= delta_thr:
            ep = close[i]
            if ol < ep:
                entry_idx = i
                entry_price = ep
                sl = ol
                tp = ep + tp_rr * (ep - ol)
                entry_time = etime[i]
                break

    if entry_idx < 0:
        return None

    # Simulate exits on subsequent bars
    for j in range(entry_idx + 1, n):
        if hhmm[j] >= TRADE_END_HHMM:
            return _close(entry_price, close[j], entry_time, etime[j], "EOD",
                          sl, tp, tp_rr, delta_thr, oh, ol)
        hit_sl = low[j] <= sl
        hit_tp = high[j] >= tp
        if hit_sl and hit_tp:
            # conservative: assume SL hit first
            return _close(entry_price, sl, entry_time, etime[j], "SL_TP_same_bar",
                          sl, tp, tp_rr, delta_thr, oh, ol)
        if hit_sl:
            return _close(entry_price, sl, entry_time, etime[j], "SL",
                          sl, tp, tp_rr, delta_thr, oh, ol)
        if hit_tp:
            return _close(entry_price, tp, entry_time, etime[j], "TP",
                          sl, tp, tp_rr, delta_thr, oh, ol)

    # Bars ran out — flat at last close
    return _close(entry_price, close[-1], entry_time, etime[-1], "EOD_LAST",
                  sl, tp, tp_rr, delta_thr, oh, ol)


def _close(ep, xp, etime, xtime, reason, sl, tp, tp_rr, dthr, oh, ol) -> dict:
    raw_pts = xp - ep
    slip_pts = SLIP_TICKS_PER_SIDE * TICK_SIZE * 2  # round trip
    gross_dollars = raw_pts * DOLLARS_PER_POINT
    slip_dollars = slip_pts * DOLLARS_PER_POINT
    net_dollars = gross_dollars - slip_dollars - COMMISSION_RT
    return {
        "entry_time": pd.Timestamp(etime),
        "exit_time": pd.Timestamp(xtime),
        "entry": ep,
        "exit": xp,
        "sl": sl,
        "tp": tp,
        "tp_rr": tp_rr,
        "delta_thr": dthr,
        "orb_high": oh,
        "orb_low": ol,
        "risk_pts": ep - ol,
        "raw_pts": raw_pts,
        "gross_dollars": gross_dollars,
        "net_dollars": net_dollars,
        "reason": reason,
    }


# ============================== sweep ==============================

def run_sweep(days: dict, date_keys: list[pd.Timestamp],
              tp_grid: np.ndarray, delta_grid: np.ndarray,
              label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (all_trades_df, summary_df)."""
    summary_rows = []
    all_trade_rows = []
    n_configs = len(tp_grid) * len(delta_grid)
    cfg_i = 0
    t0 = time.time()
    for tp_rr in tp_grid:
        for dthr in delta_grid:
            cfg_i += 1
            trades = []
            for d in date_keys:
                t = run_day(days[d], float(tp_rr), float(dthr))
                if t is not None:
                    t["session_date"] = d.date()
                    t["split"] = label
                    trades.append(t)
            n = len(trades)
            if n > 0:
                df = pd.DataFrame(trades)
                wins = int((df["net_dollars"] > 0).sum())
                summary_rows.append({
                    "tp_rr": float(tp_rr),
                    "delta_thr": float(dthr),
                    "n_trades": n,
                    "wins": wins,
                    "win_rate": wins / n,
                    "gross_$": float(df["gross_dollars"].sum()),
                    "net_$": float(df["net_dollars"].sum()),
                    "avg_net_$": float(df["net_dollars"].mean()),
                    "median_net_$": float(df["net_dollars"].median()),
                    "max_dd_$": _max_dd(df["net_dollars"]),
                    "sharpe_per_trade": _sharpe(df["net_dollars"]),
                    "profit_factor": _pf(df["net_dollars"]),
                    "n_tp": int((df["reason"] == "TP").sum()),
                    "n_sl": int((df["reason"].isin(["SL", "SL_TP_same_bar"])).sum()),
                    "n_eod": int((df["reason"].isin(["EOD", "EOD_LAST"])).sum()),
                })
                all_trade_rows.extend(trades)
            else:
                summary_rows.append({
                    "tp_rr": float(tp_rr), "delta_thr": float(dthr),
                    "n_trades": 0, "wins": 0, "win_rate": float("nan"),
                    "gross_$": 0.0, "net_$": 0.0, "avg_net_$": float("nan"),
                    "median_net_$": float("nan"), "max_dd_$": 0.0,
                    "sharpe_per_trade": float("nan"), "profit_factor": float("nan"),
                    "n_tp": 0, "n_sl": 0, "n_eod": 0,
                })
            if cfg_i % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [{label}] {cfg_i}/{n_configs} configs ({elapsed:.1f}s)", flush=True)
    return pd.DataFrame(all_trade_rows), pd.DataFrame(summary_rows)


# ============================== stats helpers ==============================

def _max_dd(pnls: pd.Series) -> float:
    eq = pnls.cumsum()
    peak = eq.cummax()
    dd = eq - peak
    return float(dd.min()) if len(dd) else 0.0


def _sharpe(pnls: pd.Series) -> float:
    s = pnls.std(ddof=0)
    if s == 0 or len(pnls) < 2 or np.isnan(s):
        return float("nan")
    return float(pnls.mean() / s)


def _pf(pnls: pd.Series) -> float:
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    if losses <= 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


# ============================== main ==============================

def main() -> None:
    print("Loading 5-min bars...", flush=True)
    bars_by_date = load_bars()
    days = precompute_days(bars_by_date)
    date_keys = sorted(days.keys())
    print(f"  {len(date_keys)} session days from {date_keys[0].date()} to {date_keys[-1].date()}", flush=True)

    split_idx = int(len(date_keys) * 0.6)
    is_dates = date_keys[:split_idx]
    oos_dates = date_keys[split_idx:]
    print(f"IS : {is_dates[0].date()} -> {is_dates[-1].date()}  ({len(is_dates)} days)", flush=True)
    print(f"OOS: {oos_dates[0].date()} -> {oos_dates[-1].date()} ({len(oos_dates)} days)", flush=True)

    tp_grid = np.round(np.arange(0.50, 5.0 + 1e-9, 0.25), 2)
    delta_grid = np.arange(0, 500 + 1, 25)
    print(f"Grid: {len(tp_grid)} TP x {len(delta_grid)} delta = {len(tp_grid)*len(delta_grid)} configs", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nRunning IS sweep...", flush=True)
    is_trades, is_summary = run_sweep(days, is_dates, tp_grid, delta_grid, "IS")
    is_trades.to_csv(OUT_DIR / f"trades_IS{RUN_SUFFIX}.csv", index=False)
    is_summary.to_csv(OUT_DIR / f"sweep_summary_IS{RUN_SUFFIX}.csv", index=False)

    print("\nRunning OOS sweep...", flush=True)
    oos_trades, oos_summary = run_sweep(days, oos_dates, tp_grid, delta_grid, "OOS")
    oos_trades.to_csv(OUT_DIR / f"trades_OOS{RUN_SUFFIX}.csv", index=False)
    oos_summary.to_csv(OUT_DIR / f"sweep_summary_OOS{RUN_SUFFIX}.csv", index=False)

    # IS top 20 vs OOS
    is_sorted = is_summary.sort_values("net_$", ascending=False).reset_index(drop=True)
    top = is_sorted.head(20).merge(oos_summary, on=["tp_rr", "delta_thr"], suffixes=("_IS", "_OOS"))
    top.to_csv(OUT_DIR / f"is_vs_oos_top20{RUN_SUFFIX}.csv", index=False)

    pd.options.display.width = 200
    pd.options.display.max_columns = 30

    print("\n=== Top 20 IS configs (by net $) -- with OOS results ===")
    cols = ["tp_rr", "delta_thr",
            "n_trades_IS", "win_rate_IS", "net_$_IS", "max_dd_$_IS", "profit_factor_IS",
            "n_trades_OOS", "win_rate_OOS", "net_$_OOS", "max_dd_$_OOS", "profit_factor_OOS"]
    print(top[cols].round(3).to_string(index=False))

    print("\n=== Top 5 IS by Sharpe ===")
    print(is_summary.sort_values("sharpe_per_trade", ascending=False).head(5)
          [["tp_rr", "delta_thr", "n_trades", "win_rate", "net_$", "sharpe_per_trade", "profit_factor"]]
          .round(3).to_string(index=False))

    print("\n=== Top 5 IS by Profit Factor (min 30 trades) ===")
    pf_pool = is_summary[is_summary["n_trades"] >= 30]
    print(pf_pool.sort_values("profit_factor", ascending=False).head(5)
          [["tp_rr", "delta_thr", "n_trades", "win_rate", "net_$", "sharpe_per_trade", "profit_factor"]]
          .round(3).to_string(index=False))

    # Highlight Fabio's published default: TP=1.0, DeltaThreshold=200 (UpTicks-DownTicks)
    print("\n=== Fabio published default (TP_RR=1.0, DeltaThreshold=200) ===")
    def_full = pd.concat([
        is_summary.assign(split="IS")[is_summary["tp_rr"].eq(1.0) & is_summary["delta_thr"].eq(200.0)],
        oos_summary.assign(split="OOS")[oos_summary["tp_rr"].eq(1.0) & oos_summary["delta_thr"].eq(200.0)],
    ])
    if not def_full.empty:
        print(def_full[["split", "n_trades", "win_rate", "net_$", "profit_factor",
                        "max_dd_$", "n_tp", "n_sl", "n_eod"]].round(3).to_string(index=False))
    # Combined full sample (concatenate IS + OOS trades and recompute)
    full_trades = pd.concat([is_trades, oos_trades], ignore_index=True)
    fab = full_trades[(full_trades["tp_rr"].eq(1.0)) & (full_trades["delta_thr"].eq(200.0))]
    if not fab.empty:
        wins = int((fab["net_dollars"] > 0).sum())
        n = len(fab)
        wins_dollars = fab.loc[fab["net_dollars"] > 0, "net_dollars"].sum()
        loss_dollars = -fab.loc[fab["net_dollars"] < 0, "net_dollars"].sum()
        print(f"\nFull-sample Fabio default: n={n}, wins={wins} ({100*wins/n:.1f}%), "
              f"net=${fab['net_dollars'].sum():,.0f}, "
              f"gross=${fab['gross_dollars'].sum():,.0f}, "
              f"PF={wins_dollars/loss_dollars:.3f}")

    print(f"\nWrote outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
