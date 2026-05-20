"""
Tests 3 (MC Shuffle), 4 (Bootstrap), 2 (Walk-Forward / Rolling OOS) for all 3 strats.
All vectorized — no engine reruns needed.

Run: python test_3_4_2_tradelog_tests.py
Saves: results/test_3_4_2_summary.txt + per-test detail files.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# -- Trade log paths and pnl-column names --
STRATS = [
    {
        "name": "Rough Vol Orderflow",
        "csv": "C:/trading/nqorderflowbacktester/scripts/rough vol orderflow/results/inspect_v3_N400_v3_trades.csv",
        "pnl_col": "pnl_dollars",
        "entry_col": "entry_ts",
    },
    {
        "name": "Overnight Drift",
        "csv": "C:/trading/nqorderflowbacktester/live/overnight drift/trades.csv",
        "pnl_col": "pnl_dollars",
        "entry_col": "entry_time",
    },
    {
        "name": "OHI/OLO B2 (MNQ)",
        "csv": "C:/trading/nqorderflowbacktester/scripts/overnight range strat/tradelogs/robust_configs/locked_v2_k08_lock045_mart_fc_filtered_trades.csv",
        "pnl_col": "scaled_pnl",
        "entry_col": "entry_ts",
    },
]

N_MC = 10_000
N_BOOT = 10_000
RNG = np.random.default_rng(42)


def load_pnl(strat):
    df = pd.read_csv(strat["csv"])
    df[strat["entry_col"]] = pd.to_datetime(df[strat["entry_col"]], utc=False)
    df = df.sort_values(strat["entry_col"]).reset_index(drop=True)
    pnl = df[strat["pnl_col"]].to_numpy()
    return df, pnl


def compute_mdd(cum):
    """MDD of equity curve `cum` (cumulative pnl)."""
    return float((cum - np.maximum.accumulate(cum)).min())


def compute_pf(pnls):
    w = pnls[pnls > 0].sum()
    l = -pnls[pnls < 0].sum()
    return float(w / l) if l > 0 else 99.0


# ---------------- Test 3: MC Shuffle ----------------
def test_3_mc_shuffle(pnls, n_perms=N_MC, rng=RNG):
    """Shuffle trade order, compute MDD distribution. Real MDD percentile-rank."""
    n = len(pnls)
    # Vectorized shuffle: build a matrix where each row is a permutation
    # Memory-efficient: process in chunks
    real_mdd = compute_mdd(np.cumsum(pnls))
    chunk = min(2000, n_perms)
    mdds = np.empty(n_perms, dtype=np.float64)
    for start in range(0, n_perms, chunk):
        end = min(start + chunk, n_perms)
        nrows = end - start
        # Generate permutations
        perms = np.empty((nrows, n), dtype=pnls.dtype)
        for i in range(nrows):
            perms[i] = rng.permutation(pnls)
        cum = np.cumsum(perms, axis=1)
        run_max = np.maximum.accumulate(cum, axis=1)
        mdd_per = (cum - run_max).min(axis=1)
        mdds[start:end] = mdd_per
    pct_real_better = float((mdds <= real_mdd).mean() * 100)  # fraction of perms with WORSE (more neg) MDD
    pct_real_worse = 100 - pct_real_better
    return dict(
        real_mdd=real_mdd,
        mdd_median=float(np.median(mdds)),
        mdd_q05=float(np.quantile(mdds, 0.05)),
        mdd_q95=float(np.quantile(mdds, 0.95)),
        pct_perms_with_worse_mdd=pct_real_better,
        pct_perms_with_better_mdd=pct_real_worse,
    )


# ---------------- Test 4: Bootstrap ----------------
def test_4_bootstrap(pnls, n_boot=N_BOOT, rng=RNG):
    """Resample with replacement, compute distribution of total PnL and PF."""
    n = len(pnls)
    real_pnl = float(pnls.sum())
    real_pf = compute_pf(pnls)
    # Vectorized resample
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = pnls[idx]
    pnl_dist = samples.sum(axis=1)
    # PF per resample
    pos = np.where(samples > 0, samples, 0).sum(axis=1)
    neg = -np.where(samples < 0, samples, 0).sum(axis=1)
    pf_dist = np.where(neg > 0, pos / neg, 99.0)
    return dict(
        real_pnl=real_pnl,
        real_pf=real_pf,
        pnl_mean=float(pnl_dist.mean()),
        pnl_median=float(np.median(pnl_dist)),
        pnl_q025=float(np.quantile(pnl_dist, 0.025)),
        pnl_q975=float(np.quantile(pnl_dist, 0.975)),
        pf_q025=float(np.quantile(pf_dist, 0.025)),
        pf_q975=float(np.quantile(pf_dist, 0.975)),
        prob_losing=float((pnl_dist <= 0).mean() * 100),
        prob_pf_under_1=float((pf_dist <= 1.0).mean() * 100),
    )


# ---------------- Test 2: Walk-Forward / Rolling OOS ----------------
def test_2_walk_forward(df, entry_col, pnl_col):
    """For locked configs, we can't refit. Instead: rolling 12-month PF / PnL."""
    df = df.copy()
    ts = pd.to_datetime(df[entry_col], utc=True).dt.tz_convert(None)
    df["yr_month"] = ts.dt.to_period("M")
    df["yr"] = ts.dt.year
    # Rolling 12 calendar-month windows ending on each year-month
    periods = sorted(df["yr_month"].unique())
    windows = []
    for end_p in periods[11:]:  # need 12 prior months
        start_p = end_p - 11
        win = df[(df["yr_month"] >= start_p) & (df["yr_month"] <= end_p)]
        if len(win) < 30:  # skip tiny windows
            continue
        pnl = win[pnl_col].to_numpy()
        windows.append({
            "window_end": str(end_p),
            "trades": len(win),
            "pf": compute_pf(pnl),
            "pnl": float(pnl.sum()),
            "wr": float((pnl > 0).mean() * 100),
        })
    return pd.DataFrame(windows)


def main():
    out_lines = []
    out_lines.append("=" * 100)
    out_lines.append("OVERFIT FRAMEWORK — Tests 2, 3, 4 (trade-log based)")
    out_lines.append("=" * 100)

    for strat in STRATS:
        print(f"\n>>> {strat['name']}")
        df, pnl = load_pnl(strat)
        out_lines.append(f"\n\n### {strat['name']}   ({len(pnl)} trades)")
        out_lines.append("=" * 100)

        # Baseline
        real_pnl = pnl.sum()
        real_pf = compute_pf(pnl)
        real_mdd = compute_mdd(np.cumsum(pnl))
        out_lines.append(f"\nBaseline: trades={len(pnl)}  total_pnl=${real_pnl:+,.0f}  "
                          f"PF={real_pf:.3f}  MDD=${real_mdd:+,.0f}")

        # Test 3: MC Shuffle
        t3 = test_3_mc_shuffle(pnl)
        out_lines.append("\n--- Test 3: Monte Carlo Order Shuffling (10k perms) ---")
        out_lines.append(f"  Real MDD:                   ${t3['real_mdd']:+,.0f}")
        out_lines.append(f"  Median shuffled MDD:        ${t3['mdd_median']:+,.0f}")
        out_lines.append(f"  5%-95% shuffled MDD band:   ${t3['mdd_q05']:+,.0f}  ... ${t3['mdd_q95']:+,.0f}")
        out_lines.append(f"  Real MDD better than {t3['pct_perms_with_worse_mdd']:.1f}% of random orderings")
        verdict_3 = "PASS" if t3['pct_perms_with_worse_mdd'] >= 50 else "FAIL"
        out_lines.append(f"  Verdict (need >50%): {verdict_3}")

        # Test 4: Bootstrap
        t4 = test_4_bootstrap(pnl)
        out_lines.append("\n--- Test 4: Bootstrap CI (10k resamples) ---")
        out_lines.append(f"  Real PnL:        ${t4['real_pnl']:+,.0f}")
        out_lines.append(f"  Mean resampled:  ${t4['pnl_mean']:+,.0f}")
        out_lines.append(f"  95% CI PnL:      ${t4['pnl_q025']:+,.0f}  ... ${t4['pnl_q975']:+,.0f}")
        out_lines.append(f"  95% CI PF:       {t4['pf_q025']:.3f}  ...  {t4['pf_q975']:.3f}")
        out_lines.append(f"  P(losing PnL):   {t4['prob_losing']:.2f}%")
        out_lines.append(f"  P(PF <= 1.0):    {t4['prob_pf_under_1']:.2f}%")
        verdict_4 = "PASS" if t4['prob_losing'] < 1.0 and t4['pnl_q025'] > 0 else "FAIL"
        out_lines.append(f"  Verdict (need P(losing)<1% AND CI lower>0): {verdict_4}")

        # Test 2: Walk-Forward / Rolling
        wf = test_2_walk_forward(df, strat["entry_col"], strat["pnl_col"])
        if len(wf) == 0:
            out_lines.append("\n--- Test 2: Walk-Forward — INSUFFICIENT WINDOWS ---")
        else:
            out_lines.append("\n--- Test 2: Walk-Forward (rolling 12-month PF, locked config) ---")
            out_lines.append(f"  Windows tested:           {len(wf)}")
            out_lines.append(f"  Windows with PF > 1.0:    {(wf['pf'] > 1.0).sum()} ({100*(wf['pf']>1).mean():.1f}%)")
            out_lines.append(f"  Windows with PF > 1.1:    {(wf['pf'] > 1.1).sum()} ({100*(wf['pf']>1.1).mean():.1f}%)")
            out_lines.append(f"  Min window PF:            {wf['pf'].min():.3f}")
            out_lines.append(f"  Max window PF:            {wf['pf'].max():.3f}")
            out_lines.append(f"  Median window PF:         {wf['pf'].median():.3f}")
            out_lines.append(f"  Worst window:             {wf.loc[wf['pf'].idxmin(), 'window_end']}  PF={wf['pf'].min():.3f}  PnL=${wf.loc[wf['pf'].idxmin(),'pnl']:+,.0f}")
            verdict_2 = "PASS" if (wf['pf'] > 1.0).mean() >= 0.75 else "FAIL"
            out_lines.append(f"  Verdict (need 75%+ windows PF>1): {verdict_2}")
            # Save detail
            wf_path = RESULTS / f"walkforward_{strat['name'].replace('/','_').replace(' ','_')}.csv"
            wf.to_csv(wf_path, index=False)

    # Write summary
    with open(RESULTS / "test_3_4_2_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print("\n".join(out_lines))
    print(f"\nSaved to {RESULTS / 'test_3_4_2_summary.txt'}")


if __name__ == "__main__":
    main()
