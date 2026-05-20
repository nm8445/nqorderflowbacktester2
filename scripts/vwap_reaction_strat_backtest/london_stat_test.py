"""
Statistical significance test: TP 1.90 vs TP 2.10 in London session.
Bootstrap CI, permutation test, power analysis, variance comparison.
"""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from adx_filter_common import (
    ET, DATA_DIR, build_adx_lookup, get_adx_at_time, passes_adx_filter,
)

VWAP_REACTION_CACHE_DIR = DATA_DIR / "vwap_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
POINT_VALUE = 20.0
ENTRY_CUTOFF = "16:00"
FORCE_CLOSE  = "16:58"

N_BOOT = 10_000
N_PERM = 10_000
SEED = 42


def get_session(hour, minute):
    t = hour * 60 + minute
    if t >= 19 * 60 or t < 2 * 60:
        return "Asia"
    elif t < 9 * 60 + 30:
        return "London"
    elif t < 16 * 60:
        return "New York"
    else:
        return "Evening"


def load_days(adx_lookup):
    start = datetime.strptime("2025-03-13", "%Y-%m-%d").date()
    end = datetime.strptime("2026-04-08", "%Y-%m-%d").date()
    days = []
    for cache_file in sorted(VWAP_REACTION_CACHE_DIR.glob("*.pkl")):
        date_str = cache_file.stem
        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if file_date < start or file_date > end:
            continue
        with open(cache_file, 'rb') as f:
            vd = pickle.load(f)
        signals = vd["signals"]
        if not signals:
            continue
        scf = SIGNAL_CACHE_DIR / f"{date_str}.pkl"
        if not scf.exists():
            continue
        with open(scf, 'rb') as f:
            sd = pickle.load(f)
        tagged = []
        for signal in signals:
            ct = signal["confirm_time"]
            if hasattr(ct, 'tz_convert'):
                cet = ct.tz_convert(ET)
            else:
                cet = pd.Timestamp(ct, tz='UTC').tz_convert(ET)
            adx_val = get_adx_at_time(cet, adx_lookup)
            signal["_adx_pass"] = passes_adx_filter(adx_val)
            signal["_confirm_et"] = cet
            tagged.append(signal)
        days.append((date_str, tagged, sd["bars"]))
    return days


def run_backtest_session(days, sl_mult, tp_mult, target_session):
    """Return array of pnl_dollars for trades in target_session."""
    pnls = []
    for date_str, signals, bars in days:
        last_exit_time = None
        for signal in signals:
            if not signal["_adx_pass"]:
                continue
            ep = signal["entry_price"]; d = signal["direction"]; atr = signal["atr"]
            cbi = signal["bar_index"] + 1
            if atr is None or atr <= 0:
                continue
            cet = signal["_confirm_et"]
            ct = signal["confirm_time"]
            if "16:00" <= cet.strftime("%H:%M") < "19:10":
                continue
            if last_exit_time is not None and ct <= last_exit_time:
                continue

            sl = ep - atr * sl_mult if d == "long" else ep + atr * sl_mult
            tp = ep + atr * tp_mult if d == "long" else ep - atr * tp_mult
            if d == "long" and (tp <= ep or sl >= ep): continue
            if d == "short" and (tp >= ep or sl <= ep): continue

            exit_price = None; exit_time = None
            for j in range(cbi + 1, len(bars)):
                bar = bars[j]
                if not bar.closed: continue
                bct = bar.close_time
                if hasattr(bct, 'tz_convert'): bet = bct.tz_convert(ET)
                else: bet = pd.Timestamp(bct, tz='UTC').tz_convert(ET)
                if bet.strftime("%H:%M") >= FORCE_CLOSE:
                    exit_price = bar.close; exit_time = bct; break
                if d == "short":
                    if bar.high >= sl: exit_price = sl; exit_time = bct; break
                    if bar.low <= tp: exit_price = tp; exit_time = bct; break
                else:
                    if bar.low <= sl: exit_price = sl; exit_time = bct; break
                    if bar.high >= tp: exit_price = tp; exit_time = bct; break
            if exit_price is None:
                exit_price = bars[-1].close; exit_time = bars[-1].close_time

            pnl = (ep - exit_price) if d == "short" else (exit_price - ep)
            last_exit_time = exit_time

            session = get_session(cet.hour, cet.minute)
            if session == target_session:
                pnls.append(pnl * POINT_VALUE)

    return np.array(pnls)


if __name__ == "__main__":
    print("Building ADX lookup...")
    adx_lookup = build_adx_lookup()
    days = load_days(adx_lookup)

    # Get London trades for both configs
    lon_190 = run_backtest_session(days, 0.50, 1.90, "London")
    lon_210 = run_backtest_session(days, 0.50, 2.10, "London")

    # Also get NY trades for variance comparison
    ny_190 = run_backtest_session(days, 0.50, 1.90, "New York")
    ny_210 = run_backtest_session(days, 0.50, 2.10, "New York")

    exp_190 = lon_190.mean()
    exp_210 = lon_210.mean()
    obs_diff = exp_190 - exp_210

    print(f"\n{'='*80}")
    print(f"  London Session: TP 1.90 vs TP 2.10 (ADX 15-30, SL 0.50)")
    print(f"{'='*80}")
    print(f"  TP 1.90: {len(lon_190)} trades, exp ${exp_190:+.1f}/trade, std ${lon_190.std():.1f}")
    print(f"  TP 2.10: {len(lon_210)} trades, exp ${exp_210:+.1f}/trade, std ${lon_210.std():.1f}")
    print(f"  Observed difference: ${obs_diff:+.1f}/trade")

    rng = np.random.default_rng(SEED)

    # =========================================================================
    # 1. BOOTSTRAP CONFIDENCE INTERVALS
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  1. BOOTSTRAP CONFIDENCE INTERVALS (10,000 resamples)")
    print(f"{'='*80}")

    boot_190 = np.zeros(N_BOOT)
    boot_210 = np.zeros(N_BOOT)
    boot_diff = np.zeros(N_BOOT)

    for i in range(N_BOOT):
        idx_190 = rng.integers(0, len(lon_190), size=len(lon_190))
        idx_210 = rng.integers(0, len(lon_210), size=len(lon_210))
        boot_190[i] = lon_190[idx_190].mean()
        boot_210[i] = lon_210[idx_210].mean()
        boot_diff[i] = boot_190[i] - boot_210[i]

    ci_190 = np.percentile(boot_190, [2.5, 97.5])
    ci_210 = np.percentile(boot_210, [2.5, 97.5])
    ci_diff = np.percentile(boot_diff, [2.5, 97.5])

    print(f"\n  TP 1.90 expectancy 95% CI: [${ci_190[0]:+.1f}, ${ci_190[1]:+.1f}]")
    print(f"  TP 2.10 expectancy 95% CI: [${ci_210[0]:+.1f}, ${ci_210[1]:+.1f}]")
    print(f"  Difference (1.90 - 2.10) 95% CI: [${ci_diff[0]:+.1f}, ${ci_diff[1]:+.1f}]")

    # Check overlap
    overlap_low = max(ci_190[0], ci_210[0])
    overlap_high = min(ci_190[1], ci_210[1])
    if overlap_low < overlap_high:
        overlap_range = overlap_high - overlap_low
        total_range_190 = ci_190[1] - ci_190[0]
        total_range_210 = ci_210[1] - ci_210[0]
        avg_range = (total_range_190 + total_range_210) / 2
        overlap_pct = overlap_range / avg_range * 100
        print(f"\n  CIs OVERLAP by ${overlap_range:.1f} ({overlap_pct:.0f}% of avg CI width)")
        print(f"  --> Difference is NOT statistically reliable")
    else:
        print(f"\n  CIs DO NOT overlap")
        print(f"  --> Difference may be meaningful")

    # Does difference CI contain zero?
    if ci_diff[0] <= 0 <= ci_diff[1]:
        print(f"  Difference CI contains $0 --> Cannot conclude TP 1.90 > TP 2.10")
    else:
        direction = "TP 1.90 > TP 2.10" if ci_diff[0] > 0 else "TP 2.10 > TP 1.90"
        print(f"  Difference CI excludes $0 --> {direction} with 95% confidence")

    # =========================================================================
    # 2. PERMUTATION TEST
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  2. PERMUTATION TEST (10,000 shuffles)")
    print(f"{'='*80}")

    # These are paired trades from the same signals but different TP levels
    # Use the shorter array length for pairing
    n_paired = min(len(lon_190), len(lon_210))
    paired_diff = lon_190[:n_paired] - lon_210[:n_paired]
    obs_paired_mean = paired_diff.mean()

    # Permutation: randomly flip signs of differences
    perm_diffs = np.zeros(N_PERM)
    for i in range(N_PERM):
        signs = rng.choice([-1, 1], size=n_paired)
        perm_diffs[i] = (paired_diff * signs).mean()

    p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_paired_mean))

    print(f"\n  Paired trades: {n_paired}")
    print(f"  Observed mean paired difference: ${obs_paired_mean:+.1f}/trade")
    print(f"  P-value (two-tailed): {p_value:.4f}")

    if p_value > 0.10:
        print(f"  --> p = {p_value:.3f} > 0.10: CANNOT reject null hypothesis")
        print(f"     The difference is indistinguishable from random noise")
    elif p_value > 0.05:
        print(f"  --> p = {p_value:.3f}: Weak evidence, not significant at 5%")
    else:
        print(f"  --> p = {p_value:.4f}: Statistically significant at 5%")

    # Also run unpooled permutation for completeness
    pooled = np.concatenate([lon_190, lon_210])
    unpooled_diffs = np.zeros(N_PERM)
    n1 = len(lon_190)
    for i in range(N_PERM):
        rng.shuffle(pooled)
        unpooled_diffs[i] = pooled[:n1].mean() - pooled[n1:].mean()

    p_unpooled = np.mean(np.abs(unpooled_diffs) >= np.abs(obs_diff))
    print(f"\n  Unpooled permutation p-value: {p_unpooled:.4f}")

    # =========================================================================
    # 3. MINIMUM SAMPLE SIZE (POWER ANALYSIS)
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  3. MINIMUM SAMPLE SIZE FOR 80% POWER")
    print(f"{'='*80}")

    # Effect size: $16/trade difference
    # Pooled std from both groups
    pooled_std = np.sqrt((lon_190.var() + lon_210.var()) / 2)
    effect = abs(obs_diff)
    cohen_d = effect / pooled_std

    print(f"\n  Observed difference: ${effect:.1f}/trade")
    print(f"  Pooled std: ${pooled_std:.1f}")
    print(f"  Cohen's d: {cohen_d:.3f} ({'negligible' if cohen_d < 0.2 else 'small' if cohen_d < 0.5 else 'medium' if cohen_d < 0.8 else 'large'})")

    # For two-sample t-test, n per group for 80% power at alpha=0.05:
    # n = 2 * ((z_alpha/2 + z_beta) / d)^2
    # z_0.025 = 1.96, z_0.20 = 0.842
    if cohen_d > 0:
        n_required = 2 * ((1.96 + 0.842) / cohen_d) ** 2
        n_required = int(np.ceil(n_required))
        # At ~82 London trades per year
        trades_per_year = len(lon_190)  # approximate
        years_needed = n_required / trades_per_year if trades_per_year > 0 else float('inf')

        print(f"\n  Required trades per group (80% power, alpha=0.05): {n_required}")
        print(f"  Current London trades: {len(lon_190)}")
        print(f"  At ~{trades_per_year} London trades/year: {years_needed:.1f} years of data needed")
    else:
        print(f"\n  Effect size is zero -- infinite sample needed")

    # =========================================================================
    # 4. VARIANCE COMPARISON: London diff vs NY noise
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  4. WITHIN-SESSION VARIANCE vs BETWEEN-CONFIG VARIANCE")
    print(f"{'='*80}")

    # Bootstrap 82-trade subsets of NY data to see normal noise in expectancy
    n_london = len(lon_190)
    ny_boot_diffs = np.zeros(N_BOOT)
    for i in range(N_BOOT):
        idx1 = rng.integers(0, len(ny_190), size=n_london)
        idx2 = rng.integers(0, len(ny_210), size=n_london)
        ny_boot_diffs[i] = ny_190[idx1].mean() - ny_210[idx2].mean()

    ny_noise_std = ny_boot_diffs.std()
    ny_noise_ci = np.percentile(np.abs(ny_boot_diffs), [50, 95])

    print(f"\n  NY session: {len(ny_190)} / {len(ny_210)} trades")
    print(f"  NY actual expectancy diff: ${ny_190.mean() - ny_210.mean():+.1f}/trade")
    print(f"\n  Bootstrapped 82-trade NY expectancy differences:")
    print(f"    Std of random diffs: ${ny_noise_std:.1f}")
    print(f"    Median |diff|: ${ny_noise_ci[0]:.1f}")
    print(f"    95th pct |diff|: ${ny_noise_ci[1]:.1f}")
    print(f"\n  London observed diff: ${abs(obs_diff):.1f}")
    print(f"  NY noise 95th pct:    ${ny_noise_ci[1]:.1f}")

    if abs(obs_diff) < ny_noise_ci[1]:
        print(f"\n  --> London diff (${abs(obs_diff):.1f}) is SMALLER than normal")
        print(f"     82-trade noise (${ny_noise_ci[1]:.1f} at 95th pct)")
        print(f"     This is completely within normal sampling variation")
    else:
        print(f"\n  --> London diff exceeds 95th pct of normal noise")
        print(f"     May indicate a real session-specific effect")

    # Overall London variance vs NY variance
    print(f"\n  Per-trade std:")
    print(f"    London TP 1.90: ${lon_190.std():.1f}")
    print(f"    London TP 2.10: ${lon_210.std():.1f}")
    print(f"    NY TP 1.90:     ${ny_190.std():.1f}")
    print(f"    NY TP 2.10:     ${ny_210.std():.1f}")

    # =========================================================================
    # VERDICT
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  VERDICT")
    print(f"{'='*80}")
    verdicts = []
    if ci_diff[0] <= 0 <= ci_diff[1]:
        verdicts.append("Bootstrap: difference CI contains $0")
    if p_value > 0.10:
        verdicts.append(f"Permutation: p={p_value:.3f} (not significant)")
    if cohen_d < 0.2:
        verdicts.append(f"Effect size: Cohen's d={cohen_d:.3f} (negligible)")
    if abs(obs_diff) < ny_noise_ci[1]:
        verdicts.append("Diff is within normal 82-trade sampling noise")

    if len(verdicts) >= 3:
        print(f"\n  CONCLUSION: The $16/trade London difference is NOISE.")
        print(f"  TP 1.90 and TP 2.10 perform indistinguishably in London.")
    elif len(verdicts) >= 2:
        print(f"\n  CONCLUSION: Likely noise, insufficient evidence for a real difference.")
    else:
        print(f"\n  CONCLUSION: Some evidence of a real difference, but needs more data.")

    for v in verdicts:
        print(f"    - {v}")

    if cohen_d > 0:
        print(f"\n  To confirm: need ~{n_required} London trades ({years_needed:.1f} years)")
