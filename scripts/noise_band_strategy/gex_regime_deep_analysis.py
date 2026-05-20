"""
Deep GEX regime analysis for noise-band strategy.

1. Bootstrap test for long/short asymmetry by regime
2. EOD survival rate by regime
3. Year-by-year Sharpe by regime (confounder check)
4. Combined directional filter test (longs-only in -GEX, shorts-only in +GEX)

Usage:
    python -u scripts/noise_band_strategy/gex_regime_deep_analysis.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

DIX_CSV = Path("D:/trading_pythonbacktest_data/NQ L2 data/DIX.csv")
TRADES_CSV = Path("C:/trading/nqorderflowbacktester/results/noise_band/trades_quantitativo.csv")

N_BOOTSTRAP = 10_000
RNG = np.random.default_rng(42)


def load_gex():
    df = pd.read_csv(DIX_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["gex_for_next_day"] = df["gex"].values
    df["trade_date"] = df["date"].shift(-1)
    df = df.dropna(subset=["trade_date"])
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df[["trade_date", "gex_for_next_day"]].copy()


def load_trades():
    df = pd.read_csv(TRADES_CSV)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df["trade_date"] = df["entry_time"].dt.tz_convert("America/New_York").dt.date
    return df


def daily_sharpe(pnls_series, trade_dates):
    """Compute annualized Sharpe from trade-level PnLs grouped by date."""
    daily = {}
    for pnl, d in zip(pnls_series, trade_dates):
        daily[d] = daily.get(d, 0) + pnl
    arr = np.array(list(daily.values()))
    if len(arr) < 2 or arr.std() == 0:
        return 0.0
    return (arr.mean() / arr.std()) * np.sqrt(252)


# ═══════════════════════════════════════════════════════════════════════════
# 1. BOOTSTRAP: Long/short asymmetry by regime
# ═══════════════════════════════════════════════════════════════���═══════════
def bootstrap_directional_asymmetry(matched):
    print("=" * 70)
    print("1. BOOTSTRAP: Long/short PnL asymmetry by GEX regime")
    print("=" * 70)

    for regime_name, sub in [
        ("+GEX", matched[matched["gex_for_next_day"] > 0]),
        ("-GEX", matched[matched["gex_for_next_day"] <= 0]),
    ]:
        longs = sub[sub["direction"] == "long"]["net_pnl"].values
        shorts = sub[sub["direction"] == "short"]["net_pnl"].values
        observed_diff = longs.mean() - shorts.mean()

        # Bootstrap: resample each direction independently, compute diff
        boot_diffs = np.empty(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            bl = RNG.choice(longs, size=len(longs), replace=True)
            bs = RNG.choice(shorts, size=len(shorts), replace=True)
            boot_diffs[i] = bl.mean() - bs.mean()

        ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
        # P-value: fraction of bootstrap samples where sign flips
        if observed_diff > 0:
            p_val = (boot_diffs <= 0).mean()
        else:
            p_val = (boot_diffs >= 0).mean()

        sig = "YES" if p_val < 0.05 else "NO"
        print(f"\n  {regime_name} regime ({len(sub)} trades):")
        print(f"    Longs:  n={len(longs)}  avg=${longs.mean():+,.0f}")
        print(f"    Shorts: n={len(shorts)}  avg=${shorts.mean():+,.0f}")
        print(f"    Observed diff (long - short): ${observed_diff:+,.0f}")
        print(f"    95% CI: [${ci_lo:+,.0f}, ${ci_hi:+,.0f}]")
        print(f"    P-value: {p_val:.4f}  Significant at 5%: {sig}")

    # Also test: is the regime effect itself significant?
    # Compare avg PnL in +GEX vs -GEX via bootstrap
    print(f"\n  --- Overall regime effect ---")
    pos_pnls = matched[matched["gex_for_next_day"] > 0]["net_pnl"].values
    neg_pnls = matched[matched["gex_for_next_day"] <= 0]["net_pnl"].values
    obs = pos_pnls.mean() - neg_pnls.mean()
    boot = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        bp = RNG.choice(pos_pnls, size=len(pos_pnls), replace=True)
        bn = RNG.choice(neg_pnls, size=len(neg_pnls), replace=True)
        boot[i] = bp.mean() - bn.mean()
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    p_val = (boot <= 0).mean() if obs > 0 else (boot >= 0).mean()
    print(f"    +GEX avg: ${pos_pnls.mean():+,.0f}  -GEX avg: ${neg_pnls.mean():+,.0f}")
    print(f"    Diff: ${obs:+,.0f}  95% CI: [${ci_lo:+,.0f}, ${ci_hi:+,.0f}]")
    print(f"    P-value: {p_val:.4f}  (tests if +GEX != -GEX)")


# ═══════════════════════════════════════════════════════════════════════════
# 2. EOD SURVIVAL RATE BY REGIME
# ═══════════════════════════════════════════════════════════════════════════
def eod_survival_by_regime(matched):
    print(f"\n{'=' * 70}")
    print("2. EOD SURVIVAL RATE BY GEX REGIME")
    print("=" * 70)

    for regime_name, sub in [
        ("+GEX", matched[matched["gex_for_next_day"] > 0]),
        ("-GEX", matched[matched["gex_for_next_day"] <= 0]),
    ]:
        total = len(sub)
        eod = (sub["exit_reason"] == "eod_close").sum()
        band = (sub["exit_reason"] == "band_stop").sum()
        vwap = (sub["exit_reason"] == "vwap_stop").sum()

        eod_pnl = sub[sub["exit_reason"] == "eod_close"]["net_pnl"]
        stopped_pnl = sub[sub["exit_reason"] != "eod_close"]["net_pnl"]

        print(f"\n  {regime_name} ({total} trades):")
        print(f"    EOD close:  {eod:>4} ({100*eod/total:.1f}%)  avg=${eod_pnl.mean():+,.0f}")
        print(f"    Band stop:  {band:>4} ({100*band/total:.1f}%)  avg=${sub[sub['exit_reason']=='band_stop']['net_pnl'].mean():+,.0f}")
        print(f"    VWAP stop:  {vwap:>4} ({100*vwap/total:.1f}%)  avg=${sub[sub['exit_reason']=='vwap_stop']['net_pnl'].mean():+,.0f}")
        print(f"    Survival rate (reach EOD): {100*eod/total:.1f}%")

    # Statistical test on survival rates
    pos = matched[matched["gex_for_next_day"] > 0]
    neg = matched[matched["gex_for_next_day"] <= 0]
    pos_surv = (pos["exit_reason"] == "eod_close").mean()
    neg_surv = (neg["exit_reason"] == "eod_close").mean()
    diff = pos_surv - neg_surv
    print(f"\n  Survival rate gap: {100*diff:+.1f}pp  (+GEX {100*pos_surv:.1f}% vs -GEX {100*neg_surv:.1f}%)")

    # Bootstrap the survival rate difference
    boot_diffs = np.empty(N_BOOTSTRAP)
    pos_eod = (pos["exit_reason"] == "eod_close").values.astype(float)
    neg_eod = (neg["exit_reason"] == "eod_close").values.astype(float)
    for i in range(N_BOOTSTRAP):
        bp = RNG.choice(pos_eod, size=len(pos_eod), replace=True).mean()
        bn = RNG.choice(neg_eod, size=len(neg_eod), replace=True).mean()
        boot_diffs[i] = bp - bn
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    p_val = (boot_diffs <= 0).mean() if diff > 0 else (boot_diffs >= 0).mean()
    print(f"  Bootstrap 95% CI: [{100*ci_lo:+.1f}pp, {100*ci_hi:+.1f}pp]  p={p_val:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. YEAR-BY-YEAR SHARPE BY REGIME (confounder check)
# ═══════════════════════════════════════════════════════════════════════════
def yearly_sharpe_by_regime(matched):
    print(f"\n{'=' * 70}")
    print("3. YEAR-BY-YEAR SHARPE BY REGIME (confounder check)")
    print("=" * 70)

    matched = matched.copy()
    matched["year"] = pd.to_datetime(matched["entry_time"]).dt.year

    print(f"\n  {'Year':<6} {'+GEX Sharpe':>12} {'+GEX Trades':>12} {'-GEX Sharpe':>12} {'-GEX Trades':>12} {'Diff':>8} {'Winner':>8}")
    print("  " + "-" * 72)

    yearly_results = []
    for year in sorted(matched["year"].unique()):
        yr_data = matched[matched["year"] == year]
        pos = yr_data[yr_data["gex_for_next_day"] > 0]
        neg = yr_data[yr_data["gex_for_next_day"] <= 0]

        pos_sharpe = daily_sharpe(pos["net_pnl"].values, pos["trade_date"].values) if len(pos) > 5 else float("nan")
        neg_sharpe = daily_sharpe(neg["net_pnl"].values, neg["trade_date"].values) if len(neg) > 5 else float("nan")

        diff = pos_sharpe - neg_sharpe if not (np.isnan(pos_sharpe) or np.isnan(neg_sharpe)) else float("nan")
        winner = ""
        if not np.isnan(diff):
            winner = "+GEX" if diff > 0 else "-GEX"

        pos_s = f"{pos_sharpe:.2f}" if not np.isnan(pos_sharpe) else "n/a"
        neg_s = f"{neg_sharpe:.2f}" if not np.isnan(neg_sharpe) else "n/a"
        diff_s = f"{diff:+.2f}" if not np.isnan(diff) else "n/a"

        print(f"  {year:<6} {pos_s:>12} {len(pos):>12} {neg_s:>12} {len(neg):>12} {diff_s:>8} {winner:>8}")
        yearly_results.append({"year": year, "pos_sharpe": pos_sharpe, "neg_sharpe": neg_sharpe})

    # Count how many years +GEX beats -GEX
    valid = [(r["pos_sharpe"], r["neg_sharpe"]) for r in yearly_results
             if not np.isnan(r["pos_sharpe"]) and not np.isnan(r["neg_sharpe"])]
    pos_wins = sum(1 for p, n in valid if p > n)
    print(f"\n  +GEX higher Sharpe in {pos_wins}/{len(valid)} years with enough data")
    if len(valid) >= 3:
        consistency = "CONSISTENT" if pos_wins >= len(valid) * 0.6 else "INCONSISTENT"
        print(f"  Verdict: {consistency} regime effect")


# ═══════════════════════════════════════════════════════════════════════════
# 4. COMBINED DIRECTIONAL FILTER TEST
# ═══════════════════════════════════════════════════════════════════════════
def combined_filter_test(matched):
    print(f"\n{'=' * 70}")
    print("4. COMBINED DIRECTIONAL FILTER: shorts-only in +GEX, longs-only in -GEX")
    print("=" * 70)

    # Baseline: all trades
    all_pnls = matched["net_pnl"].values
    all_dates = matched["trade_date"].values
    base_sharpe = daily_sharpe(all_pnls, all_dates)
    base_cum = all_pnls.cumsum()
    base_dd = (base_cum - np.maximum.accumulate(base_cum)).min()
    base_w = all_pnls[all_pnls > 0]
    base_l = all_pnls[all_pnls < 0]
    base_pf = base_w.sum() / abs(base_l.sum()) if len(base_l) else 99

    # Filtered: shorts in +GEX, longs in -GEX
    pos_shorts = matched[(matched["gex_for_next_day"] > 0) & (matched["direction"] == "short")]
    neg_longs = matched[(matched["gex_for_next_day"] <= 0) & (matched["direction"] == "long")]
    filtered = pd.concat([pos_shorts, neg_longs]).sort_values("entry_time")

    f_pnls = filtered["net_pnl"].values
    f_dates = filtered["trade_date"].values
    f_sharpe = daily_sharpe(f_pnls, f_dates)
    f_cum = f_pnls.cumsum()
    f_dd = (f_cum - np.maximum.accumulate(f_cum)).min() if len(f_cum) else 0
    f_w = f_pnls[f_pnls > 0]
    f_l = f_pnls[f_pnls < 0]
    f_pf = f_w.sum() / abs(f_l.sum()) if len(f_l) else 99

    # Opposite filter for comparison: longs in +GEX, shorts in -GEX
    pos_longs = matched[(matched["gex_for_next_day"] > 0) & (matched["direction"] == "long")]
    neg_shorts = matched[(matched["gex_for_next_day"] <= 0) & (matched["direction"] == "short")]
    opposite = pd.concat([pos_longs, neg_shorts]).sort_values("entry_time")

    o_pnls = opposite["net_pnl"].values
    o_dates = opposite["trade_date"].values
    o_sharpe = daily_sharpe(o_pnls, o_dates)
    o_cum = o_pnls.cumsum()
    o_dd = (o_cum - np.maximum.accumulate(o_cum)).min() if len(o_cum) else 0
    o_w = o_pnls[o_pnls > 0]
    o_l = o_pnls[o_pnls < 0]
    o_pf = o_w.sum() / abs(o_l.sum()) if len(o_l) else 99

    print(f"\n  {'Metric':<20} {'Baseline':>14} {'GEX Filter':>14} {'Opposite':>14}")
    print("  " + "-" * 60)
    print(f"  {'Trades':<20} {len(all_pnls):>14,} {len(f_pnls):>14,} {len(o_pnls):>14,}")
    print(f"  {'Sharpe':<20} {base_sharpe:>14.2f} {f_sharpe:>14.2f} {o_sharpe:>14.2f}")
    print(f"  {'PF':<20} {base_pf:>14.2f} {f_pf:>14.2f} {o_pf:>14.2f}")
    print(f"  {'WR':<20} {100*len(base_w)/len(all_pnls):>13.1f}% {100*len(f_w)/len(f_pnls):>13.1f}% {100*len(o_w)/len(o_pnls):>13.1f}%")
    print(f"  {'Total PnL':<20} ${all_pnls.sum():>+13,.0f} ${f_pnls.sum():>+13,.0f} ${o_pnls.sum():>+13,.0f}")
    print(f"  {'Avg PnL':<20} ${all_pnls.mean():>+13,.0f} ${f_pnls.mean():>+13,.0f} ${o_pnls.mean():>+13,.0f}")
    print(f"  {'Max DD':<20} ${base_dd:>13,.0f} ${f_dd:>13,.0f} ${o_dd:>13,.0f}")
    print(f"  {'Avg Win':<20} ${base_w.mean():>+13,.0f} ${f_w.mean():>+13,.0f} ${o_w.mean():>+13,.0f}")
    print(f"  {'Avg Loss':<20} ${base_l.mean():>+13,.0f} ${f_l.mean():>+13,.0f} ${o_l.mean():>+13,.0f}")

    # PnL per trade per regime direction combo
    print(f"\n  Breakdown of filtered trades:")
    for label, sub in [("Shorts in +GEX", pos_shorts), ("Longs in -GEX", neg_longs)]:
        p = sub["net_pnl"].values
        w = p[p > 0]
        l_a = p[p < 0]
        pf = w.sum() / abs(l_a.sum()) if len(l_a) and l_a.sum() != 0 else 99
        wr = 100 * len(w) / len(p) if len(p) else 0
        print(f"    {label:<20} {len(p):>4} trades  PF={pf:.2f}  WR={wr:.1f}%  PnL=${p.sum():+,.0f}  Avg=${p.mean():+,.0f}")

    # Yearly breakdown of filtered strategy
    print(f"\n  Filtered strategy yearly:")
    filtered_c = filtered.copy()
    filtered_c["year"] = pd.to_datetime(filtered_c["entry_time"]).dt.year
    print(f"  {'Year':<6} {'Trades':>7} {'PF':>6} {'WR':>6} {'PnL':>12} {'Sharpe':>8}")
    print("  " + "-" * 50)
    for year in sorted(filtered_c["year"].unique()):
        ys = filtered_c[filtered_c["year"] == year]
        yp = ys["net_pnl"].values
        yw = yp[yp > 0]; yl = yp[yp < 0]
        ypf = yw.sum() / abs(yl.sum()) if len(yl) and yl.sum() != 0 else 99
        ywr = 100 * len(yw) / len(yp) if len(yp) else 0
        ys_sharpe = daily_sharpe(yp, ys["trade_date"].values)
        print(f"  {year:<6} {len(yp):>7} {ypf:>6.2f} {ywr:>5.1f}% ${yp.sum():>+10,.0f} {ys_sharpe:>8.2f}")


def main():
    print("Loading data...", flush=True)
    gex_df = load_gex()
    trades = load_trades()
    matched = trades.merge(gex_df, on="trade_date", how="left").dropna(subset=["gex_for_next_day"])
    print(f"  {len(matched)} trades matched with GEX data\n")

    bootstrap_directional_asymmetry(matched)
    eod_survival_by_regime(matched)
    yearly_sharpe_by_regime(matched)
    combined_filter_test(matched)


if __name__ == "__main__":
    main()
