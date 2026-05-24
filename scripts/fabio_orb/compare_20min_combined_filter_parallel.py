"""Parallel version (6 workers) of the combined-filter sweep.

Pre-computes per-day numpy arrays once, ships to worker processes via
ProcessPoolExecutor initializer. Each worker runs configs against the
shared per-day data using tight numpy loops (no pandas in the hot path).
"""
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import itertools
import time
import pandas as pd
import numpy as np

VOL_PARQUET = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
OUT_CSV = Path("C:/trading/nqorderflowbacktester/scripts/fabio_orb/20min_combined_filter_sweep.csv")
ET = "America/New_York"

ORB_START_HHMM = 830
ORB_END_HHMM   = 900
TRADE_END_HHMM = 1400
EOD_HHMM       = 1400

DELTA_GRID = [0, 100, 200, 300, 400, 500, 600, 800]
ABS_GRID   = [0, 30, 40, 50, 60, 70, 80, 100]
TP_GRID    = [round(x, 2) for x in np.arange(1.0, 4.01, 0.25)]
ABS_MIN_LEVELS = 2

TICK = 0.25
DOLLARS_PER_PT = 20.0
SLIP_PTS_RT = 0.5     # 1 tick per side
COMM_RT = 5.0

N_WORKERS = 6

# Worker-process globals (set by initializer)
_PER_DAY = None


def load_and_prepare() -> dict:
    """Load parquet, build 5-min and 20-min bars, return per-day numpy bundles."""
    print(f"[{time.strftime('%H:%M:%S')}] Loading {VOL_PARQUET.name}...")
    df = pd.read_parquet(VOL_PARQUET)
    df["bar_open_time"] = pd.to_datetime(df["bar_open_time"]).dt.tz_convert(ET)
    df["bar_close_time"] = df["bar_open_time"] + pd.Timedelta(minutes=5)
    df["hhmm"] = df["bar_close_time"].dt.hour * 100 + df["bar_close_time"].dt.minute
    df["date"] = df["bar_close_time"].dt.date
    df["bar_open_20min"] = df["bar_open_time"].dt.floor("20min")

    bars5 = df.drop_duplicates("bar_open_time")[
        ["bar_open_time", "bar_open_20min", "open", "high", "low", "close", "hhmm", "date"]
    ].sort_values("bar_open_time").reset_index(drop=True)
    print(f"[{time.strftime('%H:%M:%S')}]   5-min bars: {len(bars5):,}")

    # 20-min OHLC
    ohlc20 = bars5.groupby("bar_open_20min", as_index=False).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    )
    ohlc20["bar_close_20min"] = ohlc20["bar_open_20min"] + pd.Timedelta(minutes=20)
    ohlc20["hhmm"] = ohlc20["bar_close_20min"].dt.hour * 100 + ohlc20["bar_close_20min"].dt.minute
    ohlc20["date"] = ohlc20["bar_close_20min"].dt.date

    # Per-level 20-min aggregate
    lvl20 = df.groupby(["bar_open_20min", "level_price"], as_index=False).agg(
        buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"))
    tot = lvl20.groupby("bar_open_20min", as_index=False).agg(
        total_buy=("buy_vol", "sum"), total_sell=("sell_vol", "sum"))
    tot["total_delta"] = tot["total_buy"] - tot["total_sell"]
    ohlc20 = ohlc20.merge(tot[["bar_open_20min", "total_delta"]], on="bar_open_20min", how="left")

    lvl20 = lvl20.merge(ohlc20[["bar_open_20min", "high", "low"]], on="bar_open_20min", how="left")
    lvl20["mid"] = (lvl20["high"] + lvl20["low"]) / 2.0
    lvl20["seller_pressure"] = lvl20["sell_vol"] - lvl20["buy_vol"]
    lower = lvl20[lvl20["level_price"] <= lvl20["mid"]].copy()

    print(f"[{time.strftime('%H:%M:%S')}]   Computing absorption counts at {len(ABS_GRID)} thresholds...")
    for T in ABS_GRID:
        col = f"abs_count_{T}"
        if T == 0:
            ohlc20[col] = 999
            continue
        passes = lower[lower["seller_pressure"] >= T].groupby("bar_open_20min").size()
        ohlc20 = ohlc20.merge(passes.rename(col), left_on="bar_open_20min", right_index=True, how="left")
        ohlc20[col] = ohlc20[col].fillna(0).astype(int)
    print(f"[{time.strftime('%H:%M:%S')}]   20-min bars: {len(ohlc20):,}")

    # Build per-day bundle
    print(f"[{time.strftime('%H:%M:%S')}] Building per-day numpy bundles...")
    bars5_by_day = {d: g for d, g in bars5.groupby("date")}
    bars20_by_day = {d: g for d, g in ohlc20.groupby("date")}
    per_day = {}
    for d, b5 in bars5_by_day.items():
        b20 = bars20_by_day.get(d)
        if b20 is None: continue
        orb_mask = (b5["hhmm"] > ORB_START_HHMM) & (b5["hhmm"] <= ORB_END_HHMM)
        orb_bars = b5[orb_mask]
        if len(orb_bars) == 0: continue
        post20 = b20[(b20["hhmm"] > ORB_END_HHMM) & (b20["hhmm"] <= TRADE_END_HHMM)]
        if len(post20) == 0: continue

        per_day[d] = {
            "orb_high": float(orb_bars["high"].max()),
            "orb_low":  float(orb_bars["low"].min()),
            "b20_close": post20["close"].values.astype(np.float64),
            "b20_hhmm":  post20["hhmm"].values.astype(np.int32),
            "b20_delta": post20["total_delta"].values.astype(np.float64),
            "b20_close_time": post20["bar_close_20min"].values.astype("datetime64[ns]"),
            "b20_abs": {T: post20[f"abs_count_{T}"].values.astype(np.int32) for T in ABS_GRID},
            "b5_open_time": b5["bar_open_time"].values.astype("datetime64[ns]"),
            "b5_high":  b5["high"].values.astype(np.float64),
            "b5_low":   b5["low"].values.astype(np.float64),
            "b5_close": b5["close"].values.astype(np.float64),
            "b5_hhmm":  b5["hhmm"].values.astype(np.int32),
        }
    print(f"[{time.strftime('%H:%M:%S')}]   {len(per_day):,} session days prepared")
    return per_day


def _init_worker(per_day):
    global _PER_DAY
    _PER_DAY = per_day


def _run_config(args):
    T_delta, T_abs, tp_rr = args
    trades_pnl = []
    n_tp = n_sl = n_eod = 0
    for d, dat in _PER_DAY.items():
        b20_close = dat["b20_close"]
        b20_hhmm  = dat["b20_hhmm"]
        b20_delta = dat["b20_delta"]
        b20_abs   = dat["b20_abs"][T_abs]
        orb_high  = dat["orb_high"]
        orb_low   = dat["orb_low"]

        entry_idx = -1
        for i in range(len(b20_close)):
            if b20_hhmm[i] > TRADE_END_HHMM: break
            if b20_close[i] <= orb_high: continue
            if orb_low >= b20_close[i]: continue
            if b20_delta[i] < T_delta: continue
            if b20_abs[i] < ABS_MIN_LEVELS: continue
            entry_idx = i
            break
        if entry_idx < 0: continue

        entry_price = b20_close[entry_idx]
        entry_time  = dat["b20_close_time"][entry_idx]
        # Skip if entry bar OHLC is NaN (sparse-volume / session-break bars)
        if not np.isfinite(entry_price) or not np.isfinite(orb_low):
            continue

        b5_open = dat["b5_open_time"]
        start_5 = int(np.searchsorted(b5_open, entry_time, side="left"))
        b5_low   = dat["b5_low"]
        b5_high  = dat["b5_high"]
        b5_close = dat["b5_close"]
        b5_hhmm  = dat["b5_hhmm"]

        sl = orb_low
        risk = entry_price - sl
        tp = entry_price + tp_rr * risk
        exit_price = entry_price
        reason = "EOD"

        for j in range(start_5, len(b5_close)):
            if b5_low[j] <= sl:
                exit_price = sl; reason = "SL"; break
            if b5_high[j] >= tp:
                exit_price = tp; reason = "TP"; break
            if b5_hhmm[j] >= EOD_HHMM:
                exit_price = b5_close[j]; reason = "EOD"; break

        if not np.isfinite(exit_price):
            continue
        pnl = (exit_price - entry_price - SLIP_PTS_RT) * DOLLARS_PER_PT - COMM_RT
        trades_pnl.append(pnl)
        if reason == "TP": n_tp += 1
        elif reason == "SL": n_sl += 1
        else: n_eod += 1

    n = len(trades_pnl)
    if n == 0:
        return {"T_delta": T_delta, "T_abs": T_abs, "tp_rr": tp_rr,
                "n_trades": 0, "wr%": 0, "net_$": 0, "PF": 0, "MaxDD_$": 0,
                "avg_$": 0, "TP%": 0, "SL%": 0, "EOD%": 0}
    arr = np.array(trades_pnl)
    wins = arr[arr > 0]; losses = arr[arr < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else float("inf")
    cum = arr.cumsum()
    dd = (cum - np.maximum.accumulate(cum)).min()
    return {
        "T_delta": T_delta, "T_abs": T_abs, "tp_rr": tp_rr,
        "n_trades": n,
        "wr%": round(len(wins) / n * 100, 1),
        "net_$": round(arr.sum(), 0),
        "PF": round(pf, 2),
        "MaxDD_$": round(dd, 0),
        "avg_$": round(arr.mean(), 1),
        "TP%": round(n_tp / n * 100, 1),
        "SL%": round(n_sl / n * 100, 1),
        "EOD%": round(n_eod / n * 100, 1),
    }


def main():
    t0 = time.time()
    per_day = load_and_prepare()
    t_prep = time.time() - t0

    configs = list(itertools.product(DELTA_GRID, ABS_GRID, TP_GRID))
    print(f"\n[{time.strftime('%H:%M:%S')}] Dispatching {len(configs)} configs to {N_WORKERS} workers...")
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker, initargs=(per_day,)) as ex:
        results = list(ex.map(_run_config, configs, chunksize=8))
    t_sweep = time.time() - t1
    print(f"[{time.strftime('%H:%M:%S')}] Sweep done in {t_sweep:.1f}s (prep was {t_prep:.1f}s)")

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV} ({len(df)} rows)")

    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)

    print("\n=== USER-REQUESTED: DELTA 600 + ABSORPTION 60 (sweep TP) ===")
    spec = df[(df["T_delta"] == 600) & (df["T_abs"] == 60)].sort_values("net_$", ascending=False)
    print(spec.to_string(index=False))

    print("\n=== HEATMAP: best NET $ per (delta, absorption) (best TP picked) ===")
    bp = (df.loc[df.groupby(["T_delta", "T_abs"])["net_$"].idxmax()]
            .pivot(index="T_delta", columns="T_abs", values="net_$"))
    print(bp.fillna(0).astype(int).to_string())

    print("\n=== HEATMAP: # trades at that best-TP cell ===")
    bn = (df.loc[df.groupby(["T_delta", "T_abs"])["net_$"].idxmax()]
            .pivot(index="T_delta", columns="T_abs", values="n_trades"))
    print(bn.fillna(0).astype(int).to_string())

    print("\n=== HEATMAP: PF at that best-TP cell ===")
    bf = (df.loc[df.groupby(["T_delta", "T_abs"])["net_$"].idxmax()]
            .pivot(index="T_delta", columns="T_abs", values="PF"))
    print(bf.fillna(0).round(2).to_string())

    print("\n=== TOP 10 (n_trades >= 100) ===")
    top = df[df["n_trades"] >= 100].sort_values("net_$", ascending=False).head(10)
    print(top.to_string(index=False))

    print("\n=== COMPARISON ===")
    print(f"  Locked 5-min Fabio:      709 trades, 53.2% WR, $151,265 net, PF 1.33, MaxDD -$20,850")
    print(f"  DELTA 600 only (best):   866 trades, 43.0% WR, $134,471 net, PF 1.25, MaxDD -$23,362")
    print(f"  ABSORPTION 60 only:      751 trades, 49.1% WR, $129,710 net, PF 1.28, MaxDD -$29,085")
    best = df[df["n_trades"] >= 100].sort_values("net_$", ascending=False).iloc[0]
    print(f"  BEST COMBO:              {best['n_trades']} trades, {best['wr%']}% WR, "
          f"${best['net_$']:,.0f}, PF {best['PF']}, MaxDD ${best['MaxDD_$']:,.0f}  "
          f"(D={best['T_delta']}, A={best['T_abs']}, TP={best['tp_rr']})")


if __name__ == "__main__":
    main()
