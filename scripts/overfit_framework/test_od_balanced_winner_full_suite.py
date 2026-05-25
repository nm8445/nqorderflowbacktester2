"""Full 5-test overfit framework for the "balanced winner" config from the 3D sweep.

Config under test:
  green_base=160, green_decay=2.0   (validated by test_od_160_20_full_suite.py)
  yellow_suppress_bars=30           (was 25)
  yellow_atr_mult=1.4               (was 1.30)
  green_atr_mult=1.5                (was 1.00)

3D-sweep result for this cell (from sweep_yellow_atr_3d.py, validated IS/OOS):
  ALL net $342,910   MDD -$35,395
  IS  +$36,545 vs anchor   OOS +$15,925 vs anchor   (2.3:1 IS:OOS ratio — balanced)

If this passes the full 5-test framework with quality similar to the 160/2.0
anchor (4/5 strict + 1 borderline), it's a deployable upgrade.
"""
from __future__ import annotations
import sys
import time
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

OD_DIR = Path("C:/trading/nqorderflowbacktester/scripts/overnight drift strategy")
sys.path.insert(0, str(OD_DIR))

CFG_BAL = dict(
    yellow_atr_len=14, yellow_atr_mult=1.40,         # was 1.30
    yellow_drift=0.0, yellow_mode="pure_ratchet",
    green_atr_len=14, green_atr_mult=1.50,           # was 1.00
    green_base=160.0, green_decay=2.0,
    red_intercept=0.0, red_drift=0.45,
    use_be=False, use_martingale=True, base_qty=1, loss_qty=2,
    tp_intrabar_fill=False,
    yellow_suppress_bars=30,                          # was 25
)

N_MC = 10_000
N_BOOT = 10_000
N_PERMS = 1000

_OD_BARS = None


def _od_init():
    global _OD_BARS
    import overnight_drift_strategy as ods
    PARQUET = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
    PICKLES = "D:/trading_pythonbacktest_data/timebars_5min"
    _OD_BARS = ods.build_full_20min_series(PARQUET, PICKLES)


def compute_pf(p):
    w = p[p > 0].sum(); l = -p[p < 0].sum()
    return float(w / l) if l > 0 else 99.0


def compute_mdd(cum):
    return float((cum - np.maximum.accumulate(cum)).min())


def _stats(p):
    if len(p) == 0: return dict(trades=0, pnl=0.0, pf=0.0, mdd=0.0, wr=0.0)
    w = p[p > 0]; l = p[p < 0]
    pf = float(w.sum() / abs(l.sum())) if len(l) else 99.0
    cum = np.cumsum(p); mdd = float((cum - np.maximum.accumulate(cum)).min())
    return dict(trades=len(p), pnl=float(p.sum()), pf=pf, mdd=mdd, wr=100.0*len(w)/len(p))


def od_run(cfg):
    import overnight_drift_strategy as ods
    trades = ods.run_backtest(_OD_BARS, ods.StrategyParams(**cfg))
    pnls = np.array([t.pnl_dollars for t in trades])
    return _stats(pnls), trades


def tests_2_3_4(trades, rng):
    df = pd.DataFrame([{
        "entry_time": t.entry_time, "exit_time": t.exit_time,
        "entry_price": t.entry_price, "exit_price": t.exit_price,
        "qty": t.qty, "pnl_dollars": t.pnl_dollars,
    } for t in trades])
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True).dt.tz_convert(None)
    df = df.sort_values("entry_time").reset_index(drop=True)
    pnl = df["pnl_dollars"].to_numpy()
    real_pnl = pnl.sum(); real_pf = compute_pf(pnl); real_mdd = compute_mdd(np.cumsum(pnl))

    out = []
    out.append(f"\n### Overnight Drift BALANCED WINNER (160/2.0 + ys=30 + yatr=1.4 + gatr=1.5)  ({len(pnl)} trades)")
    out.append("=" * 80)
    out.append(f"Baseline: PnL=${real_pnl:+,.0f}  PF={real_pf:.3f}  MDD=${real_mdd:+,.0f}")

    n = len(pnl)
    mdds = np.empty(N_MC, dtype=np.float64)
    for i in range(N_MC):
        p = rng.permutation(pnl)
        mdds[i] = compute_mdd(np.cumsum(p))
    pct_real_better = float((mdds <= real_mdd).mean() * 100)
    out.append("\n--- Test 3: Monte Carlo Order Shuffling (10k perms) ---")
    out.append(f"  Real MDD:                   ${real_mdd:+,.0f}")
    out.append(f"  Median shuffled MDD:        ${np.median(mdds):+,.0f}")
    out.append(f"  5%-95% shuffled MDD band:   ${np.quantile(mdds,0.05):+,.0f}  ...  ${np.quantile(mdds,0.95):+,.0f}")
    out.append(f"  Real MDD better than {pct_real_better:.1f}% of random orderings")
    out.append(f"  Verdict (need >50%): {'PASS' if pct_real_better >= 50 else 'FAIL'}")

    idx = rng.integers(0, n, size=(N_BOOT, n))
    samples = pnl[idx]
    pnl_dist = samples.sum(axis=1)
    pos = np.where(samples > 0, samples, 0).sum(axis=1)
    neg = -np.where(samples < 0, samples, 0).sum(axis=1)
    pf_dist = np.where(neg > 0, pos / neg, 99.0)
    out.append("\n--- Test 4: Bootstrap CI (10k resamples) ---")
    out.append(f"  Real PnL:        ${real_pnl:+,.0f}")
    out.append(f"  95% CI PnL:      ${np.quantile(pnl_dist,0.025):+,.0f}  ...  ${np.quantile(pnl_dist,0.975):+,.0f}")
    out.append(f"  95% CI PF:       {np.quantile(pf_dist,0.025):.3f}  ...  {np.quantile(pf_dist,0.975):.3f}")
    out.append(f"  P(losing PnL):   {float((pnl_dist <= 0).mean()*100):.2f}%")
    out.append(f"  P(PF <= 1.0):    {float((pf_dist <= 1.0).mean()*100):.2f}%")
    verdict_4 = float((pnl_dist <= 0).mean()*100) < 1.0 and np.quantile(pnl_dist,0.025) > 0
    out.append(f"  Verdict (P(losing)<1% AND CI lower>0): {'PASS' if verdict_4 else 'FAIL'}")

    df["yr_month"] = pd.to_datetime(df["entry_time"]).dt.to_period("M")
    periods = sorted(df["yr_month"].unique())
    windows = []
    for end_p in periods[11:]:
        start_p = end_p - 11
        win = df[(df["yr_month"] >= start_p) & (df["yr_month"] <= end_p)]
        if len(win) < 30: continue
        p = win["pnl_dollars"].to_numpy()
        windows.append({"end": str(end_p), "pf": compute_pf(p), "pnl": float(p.sum())})
    wf = pd.DataFrame(windows)
    out.append("\n--- Test 2: Walk-Forward (rolling 12-mo PF) ---")
    out.append(f"  Windows tested:    {len(wf)}")
    out.append(f"  PF > 1.0:          {(wf['pf']>1.0).sum()} ({100*(wf['pf']>1).mean():.1f}%)")
    out.append(f"  PF > 1.1:          {(wf['pf']>1.1).sum()} ({100*(wf['pf']>1.1).mean():.1f}%)")
    out.append(f"  Min window PF:     {wf['pf'].min():.3f}  (window ending {wf.loc[wf['pf'].idxmin(),'end']})")
    out.append(f"  Median window PF:  {wf['pf'].median():.3f}")
    verdict_2 = (wf['pf']>1).mean() >= 0.75
    out.append(f"  Verdict (75%+ PF>1): {'PASS' if verdict_2 else 'FAIL'}")
    return "\n".join(out)


def test_1_parameter_stability():
    base, _ = od_run(CFG_BAL)
    out = ["\n--- Test 1: Parameter Stability (1D sweeps around balanced winner) ---"]
    out.append(f"  CENTER: PnL=${base['pnl']:+,.0f}  PF={base['pf']:.3f}  MDD=${base['mdd']:+,.0f}  trades={base['trades']}")

    sweeps = {
        "green_base":       [140.0, 150.0, 155.0, 160.0, 170.0, 180.0, 200.0],
        "green_decay":      [1.5, 1.75, 2.0, 2.25, 2.5],
        "yellow_suppress_bars": [25, 28, 30, 32, 35],
        "yellow_atr_mult":  [1.2, 1.3, 1.4, 1.5],
        "green_atr_mult":   [1.0, 1.25, 1.5, 1.75],
    }
    overall_pass = True
    for param, values in sweeps.items():
        out.append(f"\n  Sweep {param}: {values}")
        rows = []
        for v in values:
            cfg = dict(CFG_BAL); cfg[param] = v
            m, _ = od_run(cfg)
            tag = "  *CENTER*" if v == CFG_BAL[param] else ""
            out.append(f"    {param}={v}  trades={m['trades']:>5}  PnL=${m['pnl']:>+11,.0f}  PF={m['pf']:.3f}  MDD=${m['mdd']:>+10,.0f}{tag}")
            rows.append(m)
        all_profitable = all(r["pnl"] > 0 for r in rows)
        all_pf = all(r["pf"] > 1.0 for r in rows)
        verdict = "PASS" if all_profitable and all_pf else "FAIL"
        if not (all_profitable and all_pf): overall_pass = False
        pnls = [r["pnl"] for r in rows]
        out.append(f"    -> {verdict}  PnL range relative to center: "
                   f"{min(pnls)/base['pnl']:.2f}x .. {max(pnls)/base['pnl']:.2f}x")
    out.append(f"\n  OVERALL Test 1: {'PASS' if overall_pass else 'FAIL'}")
    return "\n".join(out)


def _od_simulate_random_dir(rng, params_dict):
    from datetime import time as dtime
    import overnight_drift_strategy as ods
    bars = _OD_BARS
    atr_y = ods.rma_atr(bars["high"], bars["low"], bars["close"], 14)
    atr_g = atr_y
    entry_t = dtime(19, 0); force_t = dtime(8, 0)
    YM = params_dict["yellow_atr_mult"]; GM = params_dict["green_atr_mult"]
    GB = params_dict["green_base"]; GD = params_dict["green_decay"]
    RI = params_dict["red_intercept"]; RD = params_dict["red_drift"]
    YSUP = params_dict["yellow_suppress_bars"]

    idx = bars.index
    o = bars["open"].values; h = bars["high"].values
    l = bars["low"].values; c = bars["close"].values
    pos = 0; entry_price = 0.0; bars_in_trade = 0; prev_yellow = np.nan
    pnls = []

    for i in range(len(bars)):
        ts = idx[i]; local_t = ts.time()
        ay = atr_y.iloc[i]; ag = atr_g.iloc[i]
        if pos != 0:
            bars_in_trade += 1
            sign = pos
            raw_yellow = c[i] - sign * YM * ay if not np.isnan(ay) else np.nan
            if np.isnan(prev_yellow):
                yellow_val = raw_yellow
            else:
                yellow_val = max(prev_yellow, raw_yellow) if sign > 0 else min(prev_yellow, raw_yellow)
            red_val = entry_price + sign * (RI + RD * bars_in_trade)
            green_val = red_val + sign * (GB - GD * bars_in_trade) + sign * (GM * ag if not np.isnan(ag) else 0.0)
            exited = False; exit_price = np.nan
            if not np.isnan(green_val):
                if sign == 1 and h[i] >= green_val:
                    exit_price = c[i]; exited = True
                elif sign == -1 and l[i] <= green_val:
                    exit_price = c[i]; exited = True
            if (not exited and not np.isnan(yellow_val)
                    and bars_in_trade >= YSUP):
                if sign == 1 and c[i] <= yellow_val and c[i] < o[i]:
                    exit_price = c[i]; exited = True
                elif sign == -1 and c[i] >= yellow_val and c[i] > o[i]:
                    exit_price = c[i]; exited = True
            if not exited and local_t == force_t:
                exit_price = c[i]; exited = True
            if exited:
                pnls.append((exit_price - entry_price) * sign * 20.0)
                pos = 0; prev_yellow = np.nan; bars_in_trade = 0
                continue
            prev_yellow = yellow_val
        if pos == 0 and local_t == entry_t:
            pos = 1 if rng.random() < 0.5 else -1
            entry_price = c[i]; prev_yellow = np.nan; bars_in_trade = 0
    return float(sum(pnls))


def _perm_worker(args):
    seed, params_dict = args
    rng = np.random.default_rng(seed)
    return _od_simulate_random_dir(rng, params_dict)


def _perm_worker_init():
    _od_init()


def test_5_direction_permutation(n_workers=6):
    out = [f"\n--- Test 5: Direction Permutation ({N_PERMS} perms) ---"]
    class AllLong:
        def random(self): return 0.0
    real_pnl = _od_simulate_random_dir(AllLong(), CFG_BAL)
    out.append(f"  Real PnL (all long, no-mart at 1c): ${real_pnl:+,.0f}")
    args = [(s, CFG_BAL) for s in range(N_PERMS)]
    t0 = time.time()
    with mp.Pool(n_workers, initializer=_perm_worker_init) as pool:
        pnls = list(pool.imap_unordered(_perm_worker, args, chunksize=4))
    pnls = np.array(pnls)
    out.append(f"  ({time.time()-t0:.0f}s for {N_PERMS} perms across {n_workers} workers)")
    out.append(f"  Perm median:     ${np.median(pnls):+,.0f}")
    out.append(f"  Perm 5/95%:      ${np.quantile(pnls,0.05):+,.0f}  ...  ${np.quantile(pnls,0.95):+,.0f}")
    out.append(f"  Perm 99%:        ${np.quantile(pnls,0.99):+,.0f}")
    pct_real_higher = float((pnls <= real_pnl).mean() * 100)
    p_value = float((pnls >= real_pnl).mean())
    out.append(f"  Real beats {pct_real_higher:.2f}% of perms")
    out.append(f"  p-value: {p_value:.4f}")
    out.append(f"  Verdict (p<0.01): {'PASS' if p_value < 0.01 else 'FAIL'}")
    np.save(RESULTS / "od_balanced_winner_perm_pnls.npy", pnls)
    return "\n".join(out)


def main():
    mp.freeze_support()
    rng = np.random.default_rng(42)
    print("Loading bars...")
    _od_init()

    print("\n[1/3] Generating trades for balanced-winner config...")
    base_stats, trades = od_run(CFG_BAL)
    print(f"  {base_stats['trades']} trades, PnL=${base_stats['pnl']:+,.0f} "
          f"PF={base_stats['pf']:.3f} MDD=${base_stats['mdd']:+,.0f}")

    print("\n[2/3] Tests 2, 3, 4 from generated trades...")
    out_234 = tests_2_3_4(trades, rng)

    print("\n[3/3a] Test 1 parameter stability...")
    out_1 = test_1_parameter_stability()

    print("\n[3/3b] Test 5 direction permutation (parallel)...")
    out_5 = test_5_direction_permutation(n_workers=6)

    final = "\n".join([
        "=" * 80,
        "OVERFIT FRAMEWORK — OD BALANCED WINNER",
        "  green_base=160, green_decay=2.0, yellow_suppress=30, yellow_atr=1.4, green_atr=1.5",
        "=" * 80,
        out_234, out_1, out_5,
    ])
    with open(RESULTS / "od_balanced_winner_full_suite_summary.txt", "w", encoding="utf-8") as f:
        f.write(final)
    print("\n" + final)
    print(f"\nSaved -> {RESULTS / 'od_balanced_winner_full_suite_summary.txt'}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
