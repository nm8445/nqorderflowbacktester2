"""Experimental (variant): EMA trend filter on each strat's NATIVE timeframe.

OD enters on 20-min bars; B2 + Fabio fire on 5-min bars (their live signal engines). So filter each on
its own timeframe: OD = 20-min EMA, B2 = 5-min EMA, FB = 5-min EMA. Same rule + sweep as the 20-min-all
variant (no lookahead = last completed bar's close vs EMA):
  OD/FB (long-only): keep LONG if close > EMA
  B2  (long/short):  keep LONG if close > EMA, SHORT if close < EMA
Sweep EMA span 50..400 step 10, IS (oldest 60%) + OOS (newest 40%), parallel.

Run: python scripts/experimental_questions/ema_trend_filter_native_tf.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
TRADES = ROOT / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"
PARQ = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
ET = "America/New_York"
PERIODS = list(range(50, 401, 10))
STRAT_TF = {"OD": "20min", "B2": "5min", "FB": "5min"}   # native timeframe per strat

_close = {"5min": None, "20min": None}     # set by pool initializer
_trades = None                              # strat -> (idx[], sign[], pnl[], is_oos[])


def _load():
    b = pd.read_parquet(PARQ, columns=["close"])
    if b.index.tz is None:
        b.index = b.index.tz_localize("UTC")
    b.index = b.index.tz_convert(ET)
    b = b.sort_index()
    bars = {}
    for tf in ("5min", "20min"):
        c = b["close"].resample(tf, label="right", closed="right").last().dropna()
        bars[tf] = (c.values.astype(float), c.index.values.astype("int64"))

    t = pd.read_csv(TRADES)
    t["entry_ts"] = pd.to_datetime(t["entry_ts"], utc=True).dt.tz_convert(ET)
    t = t[t["strat"].isin(STRAT_TF)].sort_values("entry_ts").reset_index(drop=True)
    dates = np.sort(t["entry_ts"].dt.date.unique())
    cutoff = dates[int(0.6 * len(dates))]
    ent = t["entry_ts"].values.astype("int64")

    trades = {}
    for s, tf in STRAT_TF.items():
        g = t[t["strat"] == s]
        e = g["entry_ts"].values.astype("int64")
        idx = np.searchsorted(bars[tf][1], e, "right") - 1
        ok = idx >= 0
        trades[s] = (idx[ok],
                     np.where(g["direction"].values[ok] == "LONG", 1, -1),
                     g["pnl_$"].values[ok],
                     g["entry_ts"].dt.date.values[ok] >= cutoff)
    close = {tf: bars[tf][0] for tf in bars}
    return close, trades


def _init(close, trades):
    global _close, _trades
    _close, _trades = close, trades


def _evaluate(args):
    strat, period = args
    close = _close[STRAT_TF[strat]]
    ema = pd.Series(close).ewm(span=period, adjust=False).mean().values
    idx, sign, pnl, oos = _trades[strat]
    up = close[idx] > ema[idx]
    keep = np.where(sign == 1, up, ~up)
    kis, kos = keep & ~oos, keep & oos
    return (strat, period, float(pnl[kis].sum()), int(kis.sum()),
            float(pnl[kos].sum()), int(kos.sum()))


def main():
    print("Loading bars + trades ...")
    close, trades = _load()
    base = {}
    for s in STRAT_TF:
        idx, sign, pnl, oos = trades[s]
        base[s] = (float(pnl[~oos].sum()), int((~oos).sum()), float(pnl[oos].sum()), int(oos.sum()))

    combos = [(s, p) for s in STRAT_TF for p in PERIODS]
    with ProcessPoolExecutor(initializer=_init, initargs=(close, trades)) as ex:
        results = list(ex.map(_evaluate, combos))

    by_strat = {s: [] for s in STRAT_TF}
    for r in results:
        by_strat[r[0]].append(r)

    for s in STRAT_TF:
        b_is, b_in, b_os, b_on = base[s]
        print(f"\n=== {s} ({STRAT_TF[s]} EMA) ===  BASELINE: IS ${b_is:,.0f} ({b_in} tr)  |  OOS ${b_os:,.0f} ({b_on} tr)")
        rows = sorted(by_strat[s], key=lambda r: r[4], reverse=True)
        print(f"  {'EMA':>4} {'IS net':>10} {'IS tr':>6} {'OOS net':>10} {'OOS tr':>7} "
              f"{'IS vs base':>11} {'OOS vs base':>12}")
        for r in rows[:6]:
            _, p, isn, isnn, osn, osnn = r
            print(f"  {p:>4} {isn:>10,.0f} {isnn:>6} {osn:>10,.0f} {osnn:>7} "
                  f"{isn - b_is:>+11,.0f} {osn - b_os:>+12,.0f}")
        best = max(by_strat[s], key=lambda r: min(r[2] - b_is, r[4] - b_os))
        if best[2] > b_is and best[4] > b_os:
            print(f"  -> best ROBUST (improves IS & OOS): EMA {best[1]}  "
                  f"(IS {best[2]-b_is:+,.0f}, OOS {best[4]-b_os:+,.0f})")
        else:
            print(f"  -> NO EMA improves both IS and OOS - filter doesn't robustly help {s}")


if __name__ == "__main__":
    main()
