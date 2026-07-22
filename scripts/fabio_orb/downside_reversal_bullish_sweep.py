"""Fabio downside-reversal LONG — bullish-close confirmation sweep.

Reversal long (long-only): the day has BROKEN below ORB_Low (a real breakdown), then `n_bull` consecutive
BULLISH candles (close > open) confirm the turn, with delta >= `dthr` on the entry bar. Enter LONG at
the close;  SL = entry - ORB_range;  TP = entry + 4 * ORB_range. Same window/skip-09:30/14:00 EOD,
one trade/day (first valid). Sweep n_bull x delta to see whether green-close confirmation beats the
old 'keep closing below ORB_low' rule (which stopped out 63% of the time).

Run: python scripts/fabio_orb/downside_reversal_bullish_sweep.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np, pandas as pd
from run_final_config import (VOL_PARQUET, ORB_START_HHMM, ORB_END_HHMM, TRADE_END_HHMM,
                              SKIP_BUCKET_HHMM, DPP, SLIP, COMM, TICK_SIZE)


def load_days_open():
    df = pd.read_parquet(VOL_PARQUET)
    agg = df.groupby("bar_open_time", as_index=False).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"),
    ).dropna(subset=["open", "high", "low", "close"])
    agg["close_et"] = agg["bar_open_time"] + pd.Timedelta(minutes=5)
    agg["hhmm"] = agg["close_et"].dt.hour * 100 + agg["close_et"].dt.minute
    agg = agg[(agg["hhmm"] > ORB_START_HHMM) & (agg["hhmm"] <= TRADE_END_HHMM)].copy()
    agg["delta"] = agg["buy_vol"] - agg["sell_vol"]
    agg["session_date"] = agg["close_et"].dt.normalize()
    days = {}
    for sd, sub in agg.groupby("session_date"):
        sub = sub.sort_values("close_et").reset_index(drop=True)
        in_orb = sub[(sub["hhmm"] > ORB_START_HHMM) & (sub["hhmm"] <= ORB_END_HHMM)]
        post = sub[(sub["hhmm"] > ORB_END_HHMM) & (sub["hhmm"] <= TRADE_END_HHMM)]
        if in_orb.empty or post.empty: continue
        days[pd.Timestamp(sd)] = {
            "orb_high": float(in_orb["high"].max()), "orb_low": float(in_orb["low"].min()),
            "hhmm": post["hhmm"].to_numpy(), "open": post["open"].to_numpy(np.float64),
            "high": post["high"].to_numpy(np.float64), "low": post["low"].to_numpy(np.float64),
            "close": post["close"].to_numpy(np.float64), "delta": post["delta"].to_numpy(np.float64),
        }
    return days


def sim_exit_raw(day, i, ep, sl, tp):
    hhmm = day["hhmm"]; high = day["high"]; low = day["low"]; close = day["close"]; n = len(hhmm)
    for j in range(i + 1, n):
        if hhmm[j] >= TRADE_END_HHMM: return close[j] - ep
        if low[j] <= sl: return sl - ep        # SL first on a same-bar tie (conservative)
        if high[j] >= tp: return tp - ep
    return close[-1] - ep


def find_rev(day, n_bull, dthr):
    ol = day["orb_low"]; rng = day["orb_high"] - ol
    hhmm = day["hhmm"]; opn = day["open"]; close = day["close"]; low = day["low"]; delta = day["delta"]
    for i in range(n_bull - 1, len(hhmm)):
        if hhmm[i] > TRADE_END_HHMM: break
        if hhmm[i] == SKIP_BUCKET_HHMM: continue
        if np.min(low[:i + 1]) >= ol: continue                       # require a breakdown below ORB_low
        if not all(close[i - k] > opn[i - k] for k in range(n_bull)): continue  # N green candles
        if delta[i] < dthr: continue
        ep = close[i]
        return (i, ep, ep - rng, ep + 4.0 * rng)
    return None


def run(days, n_bull, dthr):
    nets = []
    for d in sorted(days):
        pick = find_rev(days[d], n_bull, dthr)
        if pick is None: continue
        i, ep, sl, tp = pick
        raw = sim_exit_raw(days[d], i, ep, sl, tp)
        nets.append(raw * DPP - SLIP * TICK_SIZE * 2 * DPP - COMM)
    return np.array(nets)


def stats(nets):
    if len(nets) == 0: return None
    n = len(nets); wins = int((nets > 0).sum())
    wd = nets[nets > 0].sum(); ld = -nets[nets < 0].sum()
    pf = wd / ld if ld > 0 else float("inf")
    eq = np.cumsum(nets); maxdd = (eq - np.maximum.accumulate(eq)).min()
    return dict(n=n, wr=100 * wins / n, net=nets.sum(), pf=pf, maxdd=maxdd, avg=nets.mean())


def main():
    days = load_days_open()
    print(f"{len(days)} days  (baseline UP book = 709 tr / 53.7% / $157,965 / PF 1.347 / DD -$20,240)\n")
    print(f"  {'Nbull':>5} {'delta':>6} {'trades':>7} {'WR%':>6} {'net$':>10} {'PF':>6} {'MaxDD$':>10} {'avg$':>6}")
    for n_bull in (2, 3, 4):
        for dthr in (0, 100, 200, 300, 400, 500):
            s = stats(run(days, n_bull, float(dthr)))
            if s:
                print(f"  {n_bull:>5} {dthr:>6} {s['n']:>7} {s['wr']:>5.0f}% {s['net']:>10,.0f} "
                      f"{s['pf']:>6.2f} {s['maxdd']:>10,.0f} {s['avg']:>6.0f}")
        print()


if __name__ == "__main__":
    main()
