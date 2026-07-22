"""Stage 0: harness validation + throughput.

  1. Load 1-min bars, compute rough-vol features (live default RVConfig).
  2. 60/40 chronological IS/OOS split (by trading date; warmup lives inside IS).
  3. Signal funnel + candidate-bars/day vs HIGH_Z (the throughput question).
  4. Backtest (session 19:00->16:00, hard 16:00 force-close, orderflow d=40) -> actual trades/day,
     win rate, exit-reason mix, IS vs OOS.
  5. Smoke-test the eval-MC on the IS trade pool.

This is a measurement/validation pass, NOT optimization. Default config = current live 20-min RVConfig
applied to 1-min bars (lookbacks NOT yet re-tuned -- that's stage 1).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # project root
sys.path.insert(0, str(HERE))

from live.combined.rv_features import RVConfig
from features import compute_features
from data import load_bars, build_orderflow_flags
from backtest import run, BTParams, _in_session
import eval_mc

ORDERFLOW_D = 40
SPLIT_FRAC = 0.60


def main():
    t0 = time.time()
    print("=== STAGE 0: harness validation + throughput ===\n")

    bars = load_bars()
    print(f"bars: {len(bars):,}  {bars.index.min()} -> {bars.index.max()}  ({time.time()-t0:.0f}s)")

    cfg = RVConfig()
    feat = compute_features(bars["close"].to_numpy(), bars["high"].to_numpy(),
                            bars["low"].to_numpy(), cfg)
    print(f"features computed (cfg={cfg})  ({time.time()-t0:.0f}s)\n")

    # ---- 60/40 chronological split by trading date (naive ET wall time throughout) ----
    et = bars["et"]
    et_naive = et.dt.tz_localize(None)
    dates = np.sort(np.unique(et_naive.dt.normalize().to_numpy()))
    split_date = pd.Timestamp(dates[int(len(dates) * SPLIT_FRAC)])
    print(f"IS/OOS split @ {split_date.date()}  (IS {pd.Timestamp(dates[0]).date()} -> "
          f"{split_date.date()}, OOS {split_date.date()} -> {pd.Timestamp(dates[-1]).date()})")

    # ---- signal funnel vs HIGH_Z (z_vol gate, pre-orderflow) ----
    z = feat["z_vol"]; ema = feat["ema"]; atr = feat["atr"]
    tmin = (et.dt.hour * 60 + et.dt.minute).to_numpy()
    insess = _in_session(tmin)
    n_days = len(dates)
    base = insess & np.isfinite(z) & np.isfinite(ema) & (atr > 0) & (atr <= 150.0) \
           & ((bars["close"].to_numpy() > ema) | (bars["close"].to_numpy() < ema))
    print(f"\nin-session ready bars: {int((insess & np.isfinite(z)).sum()):,}  "
          f"({n_days} trading days)")
    print("\ncandidate BARS/day vs HIGH_Z (z_vol gate + EMA dir + ATR ok, PRE-orderflow):")
    print(f"  {'HIGH_Z':>6} {'cand/day':>9} {'%in-sess':>9}")
    insess_ready = int((insess & np.isfinite(z)).sum())
    for hz in (1.0, 1.5, 2.0, 2.5, 3.0):
        c = int((base & (z > hz)).sum())
        print(f"  {hz:>6.1f} {c/n_days:>9.2f} {100*c/insess_ready:>8.2f}%")

    # ---- orderflow flags ----
    fpath = build_orderflow_flags(8, ORDERFLOW_D)
    flags = pd.read_parquet(fpath)
    lo = flags["long_ok"].reindex(bars.index, fill_value=False).to_numpy()
    so = flags["short_ok"].reindex(bars.index, fill_value=False).to_numpy()
    print(f"\norderflow flags (d={ORDERFLOW_D}): long_ok {lo.mean()*100:.1f}% / short_ok {so.mean()*100:.1f}% of bars")

    # ---- backtest at a couple HIGH_Z (actual trades, one-position-at-a-time) ----
    print(f"\nbacktest (k=2.0, session 19:00->16:00, hard 16:00 FC, orderflow d={ORDERFLOW_D}):")
    print(f"  {'HIGH_Z':>6} {'trades':>7} {'IS':>6} {'OOS':>6} {'tr/day':>7} {'win%':>6} "
          f"{'tp%':>5} {'sl%':>5} {'fc%':>5} {'EV$':>7}")
    pools = {}
    for hz in (1.5, 2.0, 2.5):
        tr = run(bars, feat, BTParams(high_z=hz, k=2.0, atr_max=150.0, use_orderflow=True), lo, so)
        if tr.empty:
            print(f"  {hz:>6.1f}  (no trades)"); continue
        ent = pd.to_datetime(tr["entry_ts"]).dt.tz_convert("America/New_York").dt.tz_localize(None)
        is_mask = (ent < split_date).reset_index(drop=True)
        rc = tr["reason"].value_counts(normalize=True)
        win = (tr["pnl_$"] > 0).mean()
        print(f"  {hz:>6.1f} {len(tr):>7} {is_mask.sum():>6} {(~is_mask).sum():>6} "
              f"{len(tr)/n_days:>7.2f} {win*100:>5.1f} "
              f"{rc.get('target',0)*100:>4.0f} {rc.get('stop',0)*100:>4.0f} "
              f"{rc.get('force_close',0)*100:>4.0f} {tr['pnl_$'].mean():>7.0f}")
        pools[hz] = (tr, is_mask)

    # ---- eval-MC smoke test on IS pool @ HIGH_Z=2.0 ----
    if 2.0 in pools:
        tr, is_mask = pools[2.0]
        ispool = tr[is_mask.values]
        res = eval_mc.run_mc(ispool["pnl_$"].to_numpy(), ispool["mae_$"].to_numpy(),
                             n_sims=20000, trades_per_day=1)
        print(f"\neval-MC smoke (IS pool, HIGH_Z=2.0, 1 trade/day, 20k sims):")
        print(f"  pool n={res['n_trades_pool']} win={res['win_rate']*100:.1f}% EV/trade=${res['ev_per_trade']:.0f}")
        print(f"  PASS={res['pass']*100:.1f}%  BUST={res['bust']*100:.1f}%  "
              f"TIMEOUT={res['timeout']*100:.1f}%  median {res['median_days']:.0f}d  p90 {res['p90_days']:.0f}d")

    print(f"\nTOTAL {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
