"""Experimental: does a 20-min EMA trend filter improve B2 / OD / Fabio?

For each strat, keep a trade only if it agrees with the EMA trend at entry. No lookahead: use the last
COMPLETED 20-min bar's close vs its EMA at the trade's entry.
  OD  (long-only):  keep LONG if close > EMA
  FB  (long-only):  keep LONG if close > EMA
  B2  (long/short): keep LONG if close > EMA, SHORT if close < EMA
Sweep EMA span 50..400 step 10. Report baseline (no filter) vs the best EMAs, IS (oldest 60%) + OOS
(newest 40%) by entry date. Parallel over (strat, period).

Run: python scripts/experimental_questions/ema_trend_filter_sweep.py
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
STRATS = ["OD", "B2", "FB"]

_close = None          # 20-min close array (set by pool initializer)
_trades = None         # dict: strat -> (idx[], sign[], pnl[], is_oos[])


def _load():
    b = pd.read_parquet(PARQ, columns=["close"])
    if b.index.tz is None:
        b.index = b.index.tz_localize("UTC")
    b.index = b.index.tz_convert(ET)
    b = b.sort_index()
    c20 = b["close"].resample("20min", label="right", closed="right").last().dropna()
    close = c20.values.astype(float)
    btimes = c20.index.values.astype("int64")

    t = pd.read_csv(TRADES)
    t["entry_ts"] = pd.to_datetime(t["entry_ts"], utc=True).dt.tz_convert(ET)
    t = t[t["strat"].isin(STRATS)].sort_values("entry_ts").reset_index(drop=True)
    ent = t["entry_ts"].values.astype("int64")
    t["idx"] = np.searchsorted(btimes, ent, "right") - 1     # last completed 20m bar at/before entry
    t = t[t["idx"] >= 0]
    dates = np.sort(t["entry_ts"].dt.date.unique())
    cutoff = dates[int(0.6 * len(dates))]                    # 60/40 IS/OOS by date
    t["is_oos"] = t["entry_ts"].dt.date.values >= cutoff
    t["sign"] = np.where(t["direction"] == "LONG", 1, -1)

    trades = {s: (g["idx"].values, g["sign"].values, g["pnl_$"].values, g["is_oos"].values)
              for s, g in t.groupby("strat")}
    return close, trades


def _init(close, trades):
    global _close, _trades
    _close, _trades = close, trades


def _evaluate(args):
    strat, period = args
    ema = pd.Series(_close).ewm(span=period, adjust=False).mean().values
    idx, sign, pnl, oos = _trades[strat]
    up = _close[idx] > ema[idx]
    keep = np.where(sign == 1, up, ~up)                      # LONG wants up, SHORT wants down
    kis, kos = keep & ~oos, keep & oos
    return (strat, period, float(pnl[kis].sum()), int(kis.sum()),
            float(pnl[kos].sum()), int(kos.sum()))


def main():
    print("Loading bars + trades ...")
    close, trades = _load()
    base = {}
    for s in STRATS:
        idx, sign, pnl, oos = trades[s]
        base[s] = (float(pnl[~oos].sum()), int((~oos).sum()), float(pnl[oos].sum()), int(oos.sum()))

    combos = [(s, p) for s in STRATS for p in PERIODS]
    with ProcessPoolExecutor(initializer=_init, initargs=(close, trades)) as ex:
        results = list(ex.map(_evaluate, combos))

    by_strat = {s: [] for s in STRATS}
    for r in results:
        by_strat[r[0]].append(r)

    for s in STRATS:
        b_is, b_in, b_os, b_on = base[s]
        print(f"\n=== {s} ===  BASELINE no-filter: IS ${b_is:,.0f} ({b_in} tr)  |  OOS ${b_os:,.0f} ({b_on} tr)")
        rows = sorted(by_strat[s], key=lambda r: r[4], reverse=True)   # rank by OOS net
        print(f"  {'EMA':>4} {'IS net':>10} {'IS tr':>6} {'OOS net':>10} {'OOS tr':>7} "
              f"{'IS vs base':>11} {'OOS vs base':>12}")
        for r in rows[:6]:
            _, p, isn, isnn, osn, osnn = r
            print(f"  {p:>4} {isn:>10,.0f} {isnn:>6} {osn:>10,.0f} {osnn:>7} "
                  f"{isn - b_is:>+11,.0f} {osn - b_os:>+12,.0f}")
        best = max(by_strat[s], key=lambda r: min(r[2] - b_is, r[4] - b_os))   # robust: improve both
        if best[2] > b_is and best[4] > b_os:
            print(f"  -> best ROBUST (improves IS & OOS): EMA {best[1]}  "
                  f"(IS {best[2]-b_is:+,.0f}, OOS {best[4]-b_os:+,.0f})")
        else:
            print(f"  -> NO EMA improves both IS and OOS - filter doesn't robustly help {s}")


if __name__ == "__main__":
    main()
