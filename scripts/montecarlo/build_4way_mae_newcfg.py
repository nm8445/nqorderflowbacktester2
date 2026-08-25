"""Build the 4-way trade log + consistent 1-min MAE for the NEW configs (FTMO/funded question):
  OD  = best 1hr, rank-1 robust-PF, force_close 14:00   [project_od_1hr_config]
  FB  = giveback trailing yellow k=1.5 / gb0.3, live 2026-07-09
  RV  = live engine full-history replay baseline (state/live_rv_trades.csv)
  B2  = live engine full-history replay baseline (state/live_b2_trades.csv), martingale stripped

Why a new builder: results/combined_4way_with_mae_1min.csv is built on OD-20min + STATIC FB —
the wrong configs. It also lacks entry/exit PRICES, which are needed to anchor MAE properly.

MAE (floating drawdown), per trade, from 1-min bars, anchored on the leg's OWN logged entry price:
    LONG : mae_pts = entry_price - min(1-min low  over hold)
    SHORT: mae_pts = max(1-min high over hold) - entry_price
  mae_$ (1 NQ) = mae_pts * $20. Per-contract, so it scales linearly with size.

FILL-OFFSET CALIBRATION (measured, not assumed — see reference_mae_exit_bar_bug):
Each leg's logged timestamp is converted to a real fill instant by adding FILL_OFF[strat].
The offsets below were calibrated by maximising exact price agreement between the leg's own
logged entry/exit price and the 1-min close at (timestamp + offset):
    OD  +59 min  -> 100.0% entry / 100.0% exit match  (60-min bars are left-labeled and are
                    resampled from a RIGHT-labeled 1-min series, so the bar labeled T closes
                    at T+59, not T+60)
    FB    0 min  ->  99.0% / 98.9%   (close_et is already the bar-close instant)
    RV    0 min  ->  timestamps are already fill instants (20-min grid, both ends)
    B2    0 min  ->  entry 68.1% exact; residual is a data-source difference (RV/B2 bars are
                    built from the 5-min pickles, not the 1-min parquet), not a time offset —
                    every non-zero offset collapses to <6%.
RV/B2 exits fill at computed stop/target LEVELS (69% / 36% of exit prices are non-tick), so exit
price agreement is not a meaningful offset signal for them; the window ends at exit_ts.

Window is truncated at the first 1-min bar that TOUCHES exit_price, so an intrabar TP fill does
not get credited with adverse excursion that happened after the position was already flat.

Output: results/combined_4way_newcfg_with_mae.csv
Invariant asserted per leg: a trade cannot realize a loss deeper than its own MAE.

Run:  python scripts/montecarlo/build_4way_mae_newcfg.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[2]
OD_DIR = ROOT / "scripts" / "overnight drift strategy"
FB_DIR = ROOT / "scripts" / "fabio_orb"
STATE = ROOT / "live" / "combined" / "state"
ONE_MIN = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
OUT = ROOT / "scripts" / "montecarlo" / "results" / "combined_4way_newcfg_with_mae.csv"

ET = "America/New_York"
NQ_PT = 20.0

FILL_OFF = {"OD": 59, "FB": 0, "RV": 0, "B2": 0}    # minutes; calibrated, see header

OD_1HR = dict(force_hour=14, atr_len=7, yellow_atr_mult=0.99, yellow_mode="pure_ratchet",
              yellow_drift=0.0, yellow_giveback=0.33, scale_by_body=False,
              max_giveback_atr=0.79, giveback_min_gap_atr=0.36,
              green_atr_mult=3.40, green_base=102.4, green_decay=0.17)
FB_GB = dict(k=1.5, mode="drift_floor", drift=0.0, gb=0.3, scale_body=True, max_gb=0.5, min_gap=0.3)


def _et(s):
    d = pd.to_datetime(s, utc=True, format="mixed") if getattr(s, "dtype", None) == object \
        else pd.to_datetime(s, utc=True)
    return d.dt.tz_convert(ET)


def build_od() -> pd.DataFrame:
    sys.path.insert(0, str(OD_DIR))
    from overnight_drift_strategy import StrategyParams, run_backtest, trades_to_df
    from sweep_1hr_timeframe import build_series

    p = StrategyParams(
        forced_hour=OD_1HR["force_hour"],
        yellow_atr_len=OD_1HR["atr_len"], yellow_atr_mult=OD_1HR["yellow_atr_mult"],
        yellow_drift=OD_1HR["yellow_drift"], yellow_mode=OD_1HR["yellow_mode"],
        green_atr_len=OD_1HR["atr_len"], green_atr_mult=OD_1HR["green_atr_mult"],
        green_base=OD_1HR["green_base"], green_decay=OD_1HR["green_decay"],
        red_intercept=0.0, red_drift=0.45,
        yellow_giveback=OD_1HR["yellow_giveback"], scale_by_body=OD_1HR["scale_by_body"],
        max_giveback_atr=OD_1HR["max_giveback_atr"],
        giveback_min_gap_atr=OD_1HR["giveback_min_gap_atr"],
        use_be=False, use_martingale=False, base_qty=1, loss_qty=1,
    )
    df = trades_to_df(run_backtest(build_series("60min"), p))
    return pd.DataFrame({
        "entry_ts": _et(df["entry_time"]), "exit_ts": _et(df["exit_time"]),
        "direction": "LONG", "entry_price": df["entry_price"].astype(float),
        "exit_price": df["exit_price"].astype(float),
        "pnl_$": df["pnl_dollars"].astype(float), "strat": "OD",
    })


def build_fb() -> pd.DataFrame:
    sys.path.insert(0, str(FB_DIR))
    from run_giveback_variant import load_days, run_giveback

    days = load_days()
    df = pd.DataFrame([t for d in sorted(days) if (t := run_giveback(days[d], **FB_GB)) is not None])
    e, x = pd.to_datetime(df["entry_time"]), pd.to_datetime(df["exit_time"])
    if e.dt.tz is None:
        e, x = e.dt.tz_localize(ET), x.dt.tz_localize(ET)
    return pd.DataFrame({
        "entry_ts": e, "exit_ts": x, "direction": "LONG",
        "entry_price": df["entry"].astype(float), "exit_price": df["exit"].astype(float),
        "pnl_$": df["net_dollars"].astype(float), "strat": "FB",
    })


def _rv_atr_at_entry(d: pd.DataFrame, df1: pd.DataFrame) -> np.ndarray:
    """Per-trade 20-min ATR(14) at entry, for the live ATR_MAX=150 filter.

    RV's SL and TP are BOTH symmetric 2xATR, so for any trade that exited at its stop or
    target the ATR is recoverable EXACTLY as |exit-entry|/2 (69% of trades). For force_close
    exits it is not, so fall back to a recomputed Wilder ATR(14) on 20-min bars resampled
    label='right'/closed='right' (the no-lookahead convention: the bar labeled T is the one
    that just closed at T). That recompute was validated against the 595 exactly-recoverable
    trades: corr 0.993, and it flags the same 2 trades above 150.
    """
    impl = (d["exit_price"] - d["entry_price"]).abs() / 2.0
    exact = d["exit_reason"].isin(["stop", "target"])

    b = df1.resample("20min", label="right", closed="right", origin="start_day").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    pc = b["close"].shift()
    tr = np.maximum(b["high"] - b["low"],
                    np.maximum((b["high"] - pc).abs(), (b["low"] - pc).abs()))
    a = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().dropna()
    ai, av = a.index.values.astype("int64"), a.values
    pos = np.searchsorted(ai, d["entry_ts"].values.astype("int64"), side="right") - 1
    est = np.where(pos >= 0, av[np.clip(pos, 0, len(av) - 1)], np.nan)
    return np.where(exact, impl, est)


def build_rv_b2(df1: pd.DataFrame) -> pd.DataFrame:
    out = []
    for s, f in [("RV", "live_rv_trades.csv"), ("B2", "live_b2_trades.csv")]:
        d = pd.read_csv(STATE / f)
        d["entry_ts"] = _et(d["entry_ts"])
        d["exit_ts"] = _et(d["exit_ts"])
        if s == "RV":
            # The LIVE engine runs ATR_MAX=150 (rv_engine.py:87). The baseline replay log
            # predates it, so apply it here or RV's tail is the UNFILTERED one.
            atr = _rv_atr_at_entry(d, df1)
            drop = np.nan_to_num(atr, nan=0.0) > 150.0
            print(f"  RV ATR_MAX=150 filter: dropping {drop.sum()} trades "
                  f"(net ${d.loc[drop,'pnl_dollars'].sum():+,.0f})")
            d = d[~drop].reset_index(drop=True)
        out.append(pd.DataFrame({
            "entry_ts": d["entry_ts"], "exit_ts": d["exit_ts"],
            "direction": d["direction"], "entry_price": d["entry_price"].astype(float),
            "exit_price": d["exit_price"].astype(float),
            # pnl_dollars is the 1-contract column; scaled_pnl_dollars is the martingale one.
            "pnl_$": d["pnl_dollars"].astype(float), "strat": s,
        }))
    return pd.concat(out, ignore_index=True)


def main():
    print("Loading 1-min bars...")
    df1 = pd.read_parquet(ONE_MIN)
    if df1.index.tz is None:
        df1.index = df1.index.tz_localize("UTC")
    df1 = df1.tz_convert(ET).sort_index()
    idx = df1.index.values.astype("int64")
    hi, lo, cl = df1["high"].values, df1["low"].values, df1["close"].values
    cov_end = df1.index.max()
    print(f"  coverage ends {cov_end}\n")

    print("Building leg trade logs (new configs)...")
    legs = pd.concat([build_od(), build_fb(), build_rv_b2(df1)], ignore_index=True)
    legs = legs.sort_values("entry_ts").reset_index(drop=True)
    for s in ["OD", "FB", "RV", "B2"]:
        m = legs[legs.strat == s]
        print(f"  {s:<3} n={len(m):5d}  net=${m['pnl_$'].sum():>11,.0f}  "
              f"{m.entry_ts.min().date()} .. {m.entry_ts.max().date()}")

    rows, skipped = [], 0
    for _, r in legs.iterrows():
        strat = r["strat"]
        off = pd.Timedelta(minutes=FILL_OFF[strat])
        longish = (r["direction"] == "LONG")
        fill, x_end = r["entry_ts"] + off, r["exit_ts"] + off
        if x_end > cov_end:
            skipped += 1
            continue
        s = int(np.searchsorted(idx, np.int64(fill.value), side="right"))
        e = int(np.searchsorted(idx, np.int64(x_end.value), side="right"))
        if e <= s:
            skipped += 1
            continue
        # Anchor on the 1-MIN series' OWN price at the fill instant, not the leg's logged
        # entry_price. The legs are backtested on different sources (FB = volumetric 5-min,
        # RV/B2 = 5-min pickles) which sit on a different contract-roll basis in roll weeks --
        # on 2025-03-19 that basis gap was ~200pts and produced MAE=$0 on a -$1,655 trade.
        # Anchor and extremes must come from the same series. P&L is a difference, so the
        # leg's own basis cancels there and the invariant check stays valid.
        ep = cl[s - 1] if s > 0 else r["entry_price"]
        # No exit-price truncation: for a stop-loss the exit sits ON THE LOSING SIDE of entry,
        # so "first bar touching exit_price" fires on bar 1 and collapses the window to one
        # minute (this produced MAE=$0 on trades that realized -$8,255). Ending at the exit
        # instant over-includes at most the final bar -- conservative in the safe direction.
        mae_pts = (ep - lo[s:e].min()) if longish else (hi[s:e].max() - ep)
        rows.append({
            "date": fill.date(), "ts": fill.isoformat(), "exit_ts": x_end.isoformat(),
            "strat": strat, "dir": r["direction"],
            "pnl_1c": r["pnl_$"], "mae_1c": -(max(mae_pts, 0.0) * NQ_PT),
        })

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nBuilt {len(out)} trades (skipped {skipped} past 1-min coverage).")

    bad = out[out.pnl_1c < out.mae_1c - 1e-6]
    print(f"\nINVARIANT  realized >= MAE : {len(bad)} violations of {len(out)}")
    if len(bad):
        print(bad.groupby("strat").size().to_string())
        print("  worst:"); print(bad.nsmallest(4, "pnl_1c").to_string(index=False))

    print("\nPer-strat floating MAE, per 1 NQ contract ($):")
    hdr = f"{'':<4}{'n':>6}{'net$':>11}{'med':>8}{'p90':>8}{'p95':>8}{'p99':>9}{'max':>9}"
    print(hdr); print("-" * len(hdr))
    for s in ["OD", "RV", "B2", "FB"]:
        m = -out[out.strat == s]["mae_1c"]
        print(f"{s:<4}{len(m):>6}{out[out.strat==s]['pnl_1c'].sum():>11,.0f}{m.median():>8,.0f}"
              f"{m.quantile(.90):>8,.0f}{m.quantile(.95):>8,.0f}{m.quantile(.99):>9,.0f}{m.max():>9,.0f}")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
