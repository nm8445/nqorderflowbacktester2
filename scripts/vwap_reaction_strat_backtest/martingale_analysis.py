"""
Martingale Activation Threshold Analysis.

Two configs: SL 0.50/TP 1.90 ADX, SL 0.50/TP 2.10 ADX.
Account: $50k, $2k EOD trailing DD (floor fixed at $48k until equity hits $52k,
then trails $2k below HWM). Base risk $500/trade (1%).
Martingale: after N consecutive losses, double bet. Cap at 8x (3 doublings).

Parts:
1. Sweep N = flat, 1-6
2. Monte Carlo (10k sims, 1k trades each)
3. IS/OOS validation (60/40 split)
4. Walk-forward (150-trade window, 50-trade forward)
5. Overfitting diagnostics
6. Summary table
7. HTML report with plots
"""

import pickle
import sys
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from adx_filter_common import (
    ET, DATA_DIR, build_adx_lookup, get_adx_at_time, passes_adx_filter,
)

VWAP_REACTION_CACHE_DIR = DATA_DIR / "vwap_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"

ENTRY_CUTOFF = "16:00"
FORCE_CLOSE  = "16:58"
MNQ_PV = 2.0

# Account rules
ACCOUNT_SIZE = 50_000
TRAIL_DD = 4_500
TRAIL_ACTIVATE = 4_500  # DD only trails after +$4.5k profit
BASE_RISK = 150  # 0.3% of account
MAX_MARTINGALE = 8  # 3 doublings: 1x -> 2x -> 4x -> 8x

N_SIMS = 10_000
MC_TRADES = 1_000
SEED = 42

# IS/OOS split
IS_FRAC = 0.60

# Walk-forward
WF_OPT_WINDOW = 150
WF_OOS_WINDOW = 50

# Minimum streak occurrences to recommend
MIN_STREAK_OCCURRENCES = 15


def load_days_with_adx(adx_lookup):
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
        tagged_signals = []
        for signal in signals:
            ct = signal["confirm_time"]
            if hasattr(ct, 'tz_convert'):
                cet = ct.tz_convert(ET)
            else:
                cet = pd.Timestamp(ct, tz='UTC').tz_convert(ET)
            adx_val = get_adx_at_time(cet, adx_lookup)
            signal["_adx_pass"] = passes_adx_filter(adx_val)
            signal["_confirm_et"] = cet
            tagged_signals.append(signal)
        days.append((date_str, tagged_signals, sd["bars"]))
    return days


def collect_trades_r_multiples(days, sl_mult, tp_mult):
    """
    Run backtest with ADX filter, return trades as R-multiples.
    R = pnl_points / sl_points. Win at TP = tp_mult/sl_mult R. Loss at SL = -1.0 R.
    Also returns raw arrays for analysis.
    """
    trades_r = []  # R-multiples

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

            sl_pts = atr * sl_mult
            if d == "long":
                sl = ep - sl_pts; tp = ep + atr * tp_mult
                if tp <= ep or sl >= ep: continue
            else:
                sl = ep + sl_pts; tp = ep - atr * tp_mult
                if tp >= ep or sl <= ep: continue

            exp = None; ext = None
            for j in range(cbi + 1, len(bars)):
                bar = bars[j]
                if not bar.closed: continue
                bct = bar.close_time
                if hasattr(bct, 'tz_convert'): bet = bct.tz_convert(ET)
                else: bet = pd.Timestamp(bct, tz='UTC').tz_convert(ET)
                if bet.strftime("%H:%M") >= FORCE_CLOSE:
                    exp = bar.close; ext = bct; break
                if d == "short":
                    if bar.high >= sl: exp = sl; ext = bct; break
                    if bar.low <= tp: exp = tp; ext = bct; break
                else:
                    if bar.low <= sl: exp = sl; ext = bct; break
                    if bar.high >= tp: exp = tp; ext = bct; break
            if exp is None:
                exp = bars[-1].close; ext = bars[-1].close_time

            pnl = (ep - exp) if d == "short" else (exp - ep)
            r_mult = pnl / sl_pts
            trades_r.append(r_mult)
            last_exit_time = ext

    return np.array(trades_r, dtype=np.float64)


def get_martingale_multiplier(consec_losses, n_threshold):
    """
    Get bet multiplier.
    n_threshold=0 means flat (always 1x).
    n_threshold=N means after N consecutive losses, start doubling.
    """
    if n_threshold == 0:
        return 1
    if consec_losses < n_threshold:
        return 1
    doublings = consec_losses - n_threshold
    mult = 2 ** min(doublings, 3)  # cap at 8x (2^3)
    return min(mult, MAX_MARTINGALE)


def simulate_equity(trades_r, n_threshold, base_risk=BASE_RISK):
    """
    Simulate equity curve for a sequence of trades (R-multiples).
    Returns: equity_curve, ruin_occurred, ruin_trade_idx
    """
    equity = float(ACCOUNT_SIZE)
    hwm = float(ACCOUNT_SIZE)
    dd_floor = float(ACCOUNT_SIZE - TRAIL_DD)
    trailing_active = False
    consec_losses = 0

    curve = [equity]
    ruin = False
    ruin_idx = -1

    for i, r in enumerate(trades_r):
        mult = get_martingale_multiplier(consec_losses, n_threshold)
        pnl = r * base_risk * mult

        equity += pnl

        # Update streak
        if r < 0:
            consec_losses += 1
        else:
            consec_losses = 0

        # Trailing DD logic
        if not trailing_active:
            if equity >= ACCOUNT_SIZE + TRAIL_ACTIVATE:
                trailing_active = True
                hwm = equity
                dd_floor = hwm - TRAIL_DD
        else:
            if equity > hwm:
                hwm = equity
                dd_floor = hwm - TRAIL_DD

        # Check ruin
        if equity <= dd_floor:
            ruin = True
            ruin_idx = i + 1
            curve.append(equity)
            break

        curve.append(equity)

    return np.array(curve), ruin, ruin_idx


def monte_carlo_sim(wr, avg_win_r, n_threshold, rng):
    """
    MC simulation with binary outcomes.
    Returns dict of stats across N_SIMS runs.
    """
    final_equities = []
    max_drawdowns = []
    ruin_count = 0
    ruin_trade_idxs = []
    sharpe_returns = []

    for _ in range(N_SIMS):
        equity = float(ACCOUNT_SIZE)
        hwm_eq = float(ACCOUNT_SIZE)
        dd_floor = float(ACCOUNT_SIZE - TRAIL_DD)
        trailing_active = False
        consec_losses = 0
        max_dd = 0.0
        ruin = False
        ruin_idx = -1

        trade_returns = []

        outcomes = rng.random(MC_TRADES)

        for t in range(MC_TRADES):
            mult = get_martingale_multiplier(consec_losses, n_threshold)
            is_win = outcomes[t] < wr

            if is_win:
                pnl = avg_win_r * BASE_RISK * mult
                consec_losses = 0
            else:
                pnl = -1.0 * BASE_RISK * mult
                consec_losses += 1

            equity += pnl
            trade_returns.append(pnl)

            # Trailing DD
            if not trailing_active:
                if equity >= ACCOUNT_SIZE + TRAIL_ACTIVATE:
                    trailing_active = True
                    hwm_eq = equity
                    dd_floor = hwm_eq - TRAIL_DD
            else:
                if equity > hwm_eq:
                    hwm_eq = equity
                    dd_floor = hwm_eq - TRAIL_DD

            # Track max DD from peak
            if equity < hwm_eq:
                dd = hwm_eq - equity
                if dd > max_dd:
                    max_dd = dd

            if equity <= dd_floor:
                ruin = True
                ruin_idx = t + 1
                break

        final_equities.append(equity)
        max_drawdowns.append(max_dd)
        if ruin:
            ruin_count += 1
            ruin_trade_idxs.append(ruin_idx)

        # Sharpe from trade returns
        tr = np.array(trade_returns)
        if len(tr) > 1 and tr.std() > 0:
            sharpe_returns.append(tr.mean() / tr.std() * np.sqrt(252))
        else:
            sharpe_returns.append(0.0)

    fe = np.array(final_equities)
    return {
        "avg_return": fe.mean() - ACCOUNT_SIZE,
        "median_return": np.median(fe) - ACCOUNT_SIZE,
        "mean_equity": fe.mean(),
        "median_equity": np.median(fe),
        "max_dd": np.mean(max_drawdowns),
        "ruin_pct": ruin_count / N_SIMS * 100,
        "sharpe": np.mean(sharpe_returns),
        "avg_ruin_trade": np.mean(ruin_trade_idxs) if ruin_trade_idxs else MC_TRADES,
        "equity_std": fe.std(),
        "equity_p5": np.percentile(fe, 5),
        "equity_p25": np.percentile(fe, 25),
        "equity_p50": np.percentile(fe, 50),
        "equity_p75": np.percentile(fe, 75),
        "equity_p95": np.percentile(fe, 95),
        "final_equities": fe,
        "ruin_trade_idxs": np.array(ruin_trade_idxs) if ruin_trade_idxs else np.array([]),
    }


def compute_sharpe_from_trades(trades_r, n_threshold):
    """Compute Sharpe ratio from actual trade R-multiples with Martingale."""
    consec_losses = 0
    pnls = []
    for r in trades_r:
        mult = get_martingale_multiplier(consec_losses, n_threshold)
        pnl = r * BASE_RISK * mult
        pnls.append(pnl)
        if r < 0:
            consec_losses += 1
        else:
            consec_losses = 0
    arr = np.array(pnls)
    if len(arr) < 2 or arr.std() == 0:
        return 0.0
    return arr.mean() / arr.std() * np.sqrt(252)


def count_loss_streaks(trades_r):
    """Count loss streaks by length. Returns dict {length: count}."""
    streaks = {}
    current = 0
    for r in trades_r:
        if r < 0:
            current += 1
        else:
            if current > 0:
                streaks[current] = streaks.get(current, 0) + 1
            current = 0
    if current > 0:
        streaks[current] = streaks.get(current, 0) + 1
    return streaks


def count_streaks_gte(streaks_dict, n):
    """Count number of loss streaks of length >= n."""
    return sum(count for length, count in streaks_dict.items() if length >= n)


def walk_forward_analysis(trades_r, opt_window=WF_OPT_WINDOW, oos_window=WF_OOS_WINDOW):
    """
    Rolling walk-forward: optimize N on opt_window, test on oos_window.
    Returns list of (window_idx, optimal_n, is_sharpe, oos_sharpe, oos_return).
    """
    results = []
    start = 0
    window_idx = 0

    while start + opt_window + oos_window <= len(trades_r):
        is_trades = trades_r[start: start + opt_window]
        oos_trades = trades_r[start + opt_window: start + opt_window + oos_window]

        # Optimize: find N with best Sharpe on IS
        best_n = 0
        best_sharpe = -999
        for n in range(7):  # 0=flat, 1-6
            s = compute_sharpe_from_trades(is_trades, n)
            if s > best_sharpe:
                best_sharpe = s
                best_n = n

        # Evaluate on OOS
        oos_sharpe = compute_sharpe_from_trades(oos_trades, best_n)
        _, oos_curve_end = simulate_equity(oos_trades, best_n)[:2]
        oos_eq_curve = simulate_equity(oos_trades, best_n)[0]
        oos_return = oos_eq_curve[-1] - ACCOUNT_SIZE

        is_sharpe_val = best_sharpe

        results.append({
            "window": window_idx,
            "start": start,
            "optimal_n": best_n,
            "is_sharpe": is_sharpe_val,
            "oos_sharpe": oos_sharpe,
            "oos_return": oos_return,
        })

        start += oos_window
        window_idx += 1

    return results


def analyze_config(config_name, trades_r, wr, avg_win_r, rng_seed=SEED):
    """Full analysis for one config. Returns all results."""
    print(f"\n{'='*100}")
    print(f"  {config_name}")
    print(f"  {len(trades_r)} trades | WR: {wr*100:.1f}% | Avg Win R: {avg_win_r:.2f} | Avg Loss R: -1.00")
    print(f"  WR 95% CI: [{wr - 1.96*math.sqrt(wr*(1-wr)/len(trades_r)):.3f}, "
          f"{wr + 1.96*math.sqrt(wr*(1-wr)/len(trades_r)):.3f}]")
    print(f"{'='*100}")

    n_values = list(range(7))  # 0=flat, 1-6
    results = {}

    # --- Part 1 & 2: MC Simulation ---
    print("\n  PART 1-2: Monte Carlo Simulation (10k sims, 1k trades)")
    print(f"  {'N':>3s}  {'Avg Ret':>10s}  {'Med Ret':>10s}  {'Avg MaxDD':>10s}  {'Sharpe':>8s}  "
          f"{'Ruin%':>7s}  {'Avg Ruin@':>9s}  {'Eq StdDev':>10s}")
    print(f"  {'-'*80}")

    mc_results = {}
    for n in n_values:
        rng = np.random.default_rng(rng_seed + n)
        label = "flat" if n == 0 else f"N={n}"
        mc = monte_carlo_sim(wr, avg_win_r, n, rng)
        mc_results[n] = mc

        ruin_at = f"{mc['avg_ruin_trade']:.0f}" if mc['ruin_pct'] > 0 else "n/a"
        print(f"  {label:>4s}  ${mc['avg_return']:>+9,.0f}  ${mc['median_return']:>+9,.0f}  "
              f"${mc['max_dd']:>9,.0f}  {mc['sharpe']:>7.2f}  {mc['ruin_pct']:>6.1f}%  "
              f"{ruin_at:>9s}  ${mc['equity_std']:>9,.0f}")

    # --- Part 3: IS/OOS Validation ---
    n_is = int(len(trades_r) * IS_FRAC)
    is_trades = trades_r[:n_is]
    oos_trades = trades_r[n_is:]
    print(f"\n  PART 3: IS/OOS Validation (IS: {len(is_trades)} trades, OOS: {len(oos_trades)} trades)")

    is_oos = {}
    best_is_n = 0
    best_is_sharpe = -999
    print(f"  {'N':>3s}  {'IS Sharpe':>10s}  {'OOS Sharpe':>11s}  {'IS/OOS Ratio':>13s}  "
          f"{'IS Return':>10s}  {'OOS Return':>11s}  {'Degradation':>12s}")
    print(f"  {'-'*80}")

    for n in n_values:
        label = "flat" if n == 0 else f"N={n}"
        is_sharpe = compute_sharpe_from_trades(is_trades, n)
        oos_sharpe = compute_sharpe_from_trades(oos_trades, n)

        is_curve = simulate_equity(is_trades, n)[0]
        oos_curve = simulate_equity(oos_trades, n)[0]
        is_ret = is_curve[-1] - ACCOUNT_SIZE
        oos_ret = oos_curve[-1] - ACCOUNT_SIZE

        ratio = is_sharpe / oos_sharpe if oos_sharpe != 0 else float('inf')
        degradation = ""
        if oos_sharpe < is_sharpe * 0.5 and is_sharpe > 0:
            degradation = "** SIGNIFICANT **"
        elif oos_sharpe < is_sharpe * 0.7 and is_sharpe > 0:
            degradation = "* moderate *"

        is_oos[n] = {
            "is_sharpe": is_sharpe, "oos_sharpe": oos_sharpe,
            "ratio": ratio, "is_ret": is_ret, "oos_ret": oos_ret,
        }

        if is_sharpe > best_is_sharpe:
            best_is_sharpe = is_sharpe
            best_is_n = n

        print(f"  {label:>4s}  {is_sharpe:>10.2f}  {oos_sharpe:>11.2f}  {ratio:>13.2f}  "
              f"${is_ret:>+9,.0f}  ${oos_ret:>+10,.0f}  {degradation}")

    print(f"\n  IS-optimal N = {best_is_n} (Sharpe {best_is_sharpe:.2f})")
    print(f"  OOS performance at N={best_is_n}: Sharpe {is_oos[best_is_n]['oos_sharpe']:.2f}, "
          f"Return ${is_oos[best_is_n]['oos_ret']:+,.0f}")

    # --- Part 4: Walk-Forward ---
    wf = walk_forward_analysis(trades_r)
    print(f"\n  PART 4: Walk-Forward Analysis ({WF_OPT_WINDOW}-trade opt / {WF_OOS_WINDOW}-trade OOS)")

    if len(wf) < 3:
        print(f"  WARNING: Only {len(wf)} windows available (< 3). Results may be unreliable.")

    if wf:
        opt_ns = [w["optimal_n"] for w in wf]
        n_stability = np.std(opt_ns)
        print(f"  Windows: {len(wf)}")
        print(f"  Optimal N per window: {opt_ns}")
        print(f"  Stability (stddev of optimal N): {n_stability:.2f}")
        for w in wf:
            label_n = "flat" if w["optimal_n"] == 0 else f"N={w['optimal_n']}"
            print(f"    Window {w['window']}: optimal={label_n}, "
                  f"IS Sharpe={w['is_sharpe']:.2f}, OOS Sharpe={w['oos_sharpe']:.2f}, "
                  f"OOS Ret=${w['oos_return']:+,.0f}")
    else:
        opt_ns = []
        n_stability = float('nan')
        print("  No windows available!")

    # --- Part 5: Overfitting Diagnostics ---
    print(f"\n  PART 5: Overfitting Diagnostics")
    streaks = count_loss_streaks(trades_r)
    is_streaks = count_loss_streaks(is_trades)

    print(f"\n  Loss streak distribution (full dataset, {len(trades_r)} trades):")
    for length in sorted(streaks.keys()):
        print(f"    Streak length {length}: {streaks[length]} occurrences")

    print(f"\n  Streak occurrences >= N (in-sample, {len(is_trades)} trades):")
    streak_counts = {}
    for n in range(1, 7):
        count = count_streaks_gte(is_streaks, n)
        streak_counts[n] = count
        flag = " << INSUFFICIENT SAMPLE" if count < MIN_STREAK_OCCURRENCES else ""
        print(f"    Streaks >= {n}: {count}{flag}")

    # --- Part 6: Summary Table ---
    print(f"\n  PART 6: Summary Table")
    print(f"  {'N':>4s}  {'MC Ret':>10s}  {'MC Med':>10s}  {'MC MaxDD':>10s}  {'MC Sharpe':>10s}  "
          f"{'Ruin%':>7s}  {'IS/OOS Sh':>10s}  {'Streaks':>8s}  {'Stable':>8s}  {'Recommend':>10s}")
    print(f"  {'-'*100}")

    recommendations = {}
    for n in n_values:
        label = "flat" if n == 0 else f"N={n}"
        mc = mc_results[n]
        iso = is_oos[n]

        # Streak count (for N>=1)
        if n == 0:
            sc = "-"
            sc_val = 999
        else:
            sc_val = streak_counts.get(n, 0)
            sc = str(sc_val)

        # Stability from walk-forward
        stab = f"{n_stability:.2f}" if not math.isnan(n_stability) else "n/a"

        # Recommendation logic
        flags = []
        if n > 0 and sc_val < MIN_STREAK_OCCURRENCES:
            flags.append("low_sample")
        if iso["oos_sharpe"] < iso["is_sharpe"] * 0.5 and iso["is_sharpe"] > 0:
            flags.append("overfit")
        if mc["ruin_pct"] > 50:
            flags.append("high_ruin")

        if not flags:
            recommend = "OK"
        else:
            recommend = ",".join(flags)

        recommendations[n] = {
            "flags": flags,
            "mc": mc,
            "is_oos": iso,
            "streak_count": sc_val,
        }

        ratio_str = f"{iso['ratio']:.2f}" if iso['ratio'] != float('inf') else "inf"

        print(f"  {label:>4s}  ${mc['avg_return']:>+9,.0f}  ${mc['median_return']:>+9,.0f}  "
              f"${mc['max_dd']:>9,.0f}  {mc['sharpe']:>10.2f}  {mc['ruin_pct']:>6.1f}%  "
              f"{ratio_str:>10s}  {sc:>8s}  {stab:>8s}  {recommend:>10s}")

    # Find best N without red flags
    best_n_rec = None
    best_n_score = -999
    for n in n_values:
        rec = recommendations[n]
        if rec["flags"]:
            continue
        # Score: Sharpe * (1 - ruin_pct/100) — risk-adjusted
        score = rec["mc"]["sharpe"] * (1 - rec["mc"]["ruin_pct"] / 100)
        if score > best_n_score:
            best_n_score = score
            best_n_rec = n

    if best_n_rec is not None:
        label = "flat" if best_n_rec == 0 else f"N={best_n_rec}"
        print(f"\n  >>> RECOMMENDED: {label} (best risk-adjusted without overfitting flags)")
    else:
        print(f"\n  >>> WARNING: All N values have red flags. Flat betting safest default.")

    return {
        "config_name": config_name,
        "n_trades": len(trades_r),
        "wr": wr,
        "avg_win_r": avg_win_r,
        "mc_results": mc_results,
        "is_oos": is_oos,
        "walk_forward": wf,
        "streaks": streaks,
        "is_streaks": is_streaks,
        "streak_counts": streak_counts,
        "n_stability": n_stability,
        "recommendations": recommendations,
        "best_n": best_n_rec,
        "is_trades": is_trades,
        "oos_trades": oos_trades,
    }


def generate_html_report(all_results, output_path):
    """Generate comprehensive HTML report with charts."""

    # Prepare data for charts
    configs_json = []
    for res in all_results:
        cfg = {
            "name": res["config_name"],
            "n_trades": res["n_trades"],
            "wr": round(res["wr"] * 100, 1),
            "avg_win_r": round(res["avg_win_r"], 2),
            "mc": {},
            "is_oos": {},
            "walk_forward": [],
            "streaks": {str(k): v for k, v in res["streaks"].items()},
            "streak_counts": {str(k): v for k, v in res["streak_counts"].items()},
            "best_n": res["best_n"],
            "n_stability": round(res["n_stability"], 2) if not math.isnan(res["n_stability"]) else None,
        }

        for n in range(7):
            mc = res["mc_results"][n]
            cfg["mc"][str(n)] = {
                "avg_return": round(mc["avg_return"]),
                "median_return": round(mc["median_return"]),
                "max_dd": round(mc["max_dd"]),
                "sharpe": round(mc["sharpe"], 2),
                "ruin_pct": round(mc["ruin_pct"], 1),
                "avg_ruin_trade": round(mc["avg_ruin_trade"]),
                "p5": round(mc["equity_p5"]),
                "p25": round(mc["equity_p25"]),
                "p50": round(mc["equity_p50"]),
                "p75": round(mc["equity_p75"]),
                "p95": round(mc["equity_p95"]),
            }

            iso = res["is_oos"][n]
            cfg["is_oos"][str(n)] = {
                "is_sharpe": round(iso["is_sharpe"], 2),
                "oos_sharpe": round(iso["oos_sharpe"], 2),
                "ratio": round(iso["ratio"], 2) if iso["ratio"] != float('inf') else 99,
                "is_ret": round(iso["is_ret"]),
                "oos_ret": round(iso["oos_ret"]),
            }

        for w in res["walk_forward"]:
            cfg["walk_forward"].append({
                "window": w["window"],
                "optimal_n": w["optimal_n"],
                "is_sharpe": round(w["is_sharpe"], 2),
                "oos_sharpe": round(w["oos_sharpe"], 2),
            })

        # Survival curves: for each N, what % of sims still alive at trade T
        survival = {}
        for n in range(7):
            mc = res["mc_results"][n]
            ruin_idxs = mc["ruin_trade_idxs"]
            if len(ruin_idxs) == 0:
                survival[str(n)] = [100.0] * 20  # all survive
            else:
                checkpoints = list(range(50, 1001, 50))
                surv = []
                for cp in checkpoints:
                    alive = N_SIMS - np.sum(ruin_idxs <= cp)
                    surv.append(round(alive / N_SIMS * 100, 1))
                survival[str(n)] = surv
        cfg["survival"] = survival

        # Equity curve samples for IS/OOS
        is_curves = {}
        oos_curves = {}
        for n in range(7):
            is_c = simulate_equity(res["is_trades"], n)[0]
            oos_c = simulate_equity(res["oos_trades"], n)[0]
            is_curves[str(n)] = [round(float(x)) for x in is_c]
            oos_curves[str(n)] = [round(float(x)) for x in oos_c]
        cfg["is_curves"] = is_curves
        cfg["oos_curves"] = oos_curves

        configs_json.append(cfg)

    data_json = json.dumps(configs_json)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Martingale Activation Threshold Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0a0f; color: #e0e0e0; padding: 20px; }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        h1 {{ color: #00d4ff; font-size: 28px; margin-bottom: 5px; }}
        h2 {{ color: #ff6b35; font-size: 22px; margin: 30px 0 15px; border-bottom: 1px solid #333; padding-bottom: 8px; }}
        h3 {{ color: #00d4ff; font-size: 18px; margin: 20px 0 10px; }}
        .subtitle {{ color: #888; font-size: 14px; margin-bottom: 25px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
        .chart-box {{ background: #151520; border: 1px solid #2a2a3a; border-radius: 8px; padding: 15px; }}
        .chart-box canvas {{ width: 100% !important; height: 350px !important; }}
        .full-width {{ grid-column: 1 / -1; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 13px; }}
        th {{ background: #1a1a2e; color: #00d4ff; padding: 8px 10px; text-align: right; border: 1px solid #2a2a3a; }}
        td {{ padding: 8px 10px; text-align: right; border: 1px solid #2a2a3a; }}
        tr:nth-child(even) {{ background: #12121e; }}
        tr:hover {{ background: #1a1a30; }}
        .highlight {{ background: #0a2a0a !important; border-left: 3px solid #00ff88; }}
        .flag {{ color: #ff4444; font-weight: bold; }}
        .ok {{ color: #00ff88; }}
        .warn {{ color: #ffaa00; }}
        .config-section {{ margin: 40px 0; padding: 20px; background: #0d0d18; border: 1px solid #2a2a3a; border-radius: 10px; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 15px 0; }}
        .stat-card {{ background: #151520; border: 1px solid #2a2a3a; border-radius: 6px; padding: 12px; text-align: center; }}
        .stat-card .value {{ font-size: 24px; font-weight: bold; color: #00d4ff; }}
        .stat-card .label {{ font-size: 11px; color: #888; margin-top: 4px; }}
        .note {{ background: #1a1a0a; border-left: 3px solid #ffaa00; padding: 10px 15px; margin: 15px 0; font-size: 13px; color: #ccaa00; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Martingale Activation Threshold Analysis</h1>
    <div class="subtitle">
        VWAP Reaction Strategy | ADX 15-30 Filter | $50k Account | $2k Trailing DD (activates at +$2k profit) |
        Base Risk: $500/trade (1%) | Martingale Cap: 8x | 10,000 MC sims x 1,000 trades
    </div>

    <div class="note">
        <strong>Confidence note:</strong> Win rates estimated from ~350-380 historical trades.
        95% confidence intervals shown per config. Monte Carlo assumes these rates are accurate.
        With only ~350 trades, streak counts for N >= 4 are inherently small.
        Be conservative in interpretation.
    </div>

    <div id="report"></div>
</div>

<script>
const DATA = {data_json};
const COLORS = ['#888888', '#00d4ff', '#ff6b35', '#00ff88', '#ff4488', '#aa44ff', '#ffdd00'];
const N_LABELS = ['Flat', 'N=1', 'N=2', 'N=3', 'N=4', 'N=5', 'N=6'];

function fmt(n) {{ return n >= 0 ? '+$' + n.toLocaleString() : '-$' + Math.abs(n).toLocaleString(); }}

function renderConfig(cfg, idx) {{
    const div = document.createElement('div');
    div.className = 'config-section';

    const wrCI = (1.96 * Math.sqrt((cfg.wr/100)*(1-cfg.wr/100)/cfg.n_trades) * 100).toFixed(1);

    let html = `<h2>${{cfg.name}}</h2>
    <div class="stat-grid">
        <div class="stat-card"><div class="value">${{cfg.n_trades}}</div><div class="label">Trades</div></div>
        <div class="stat-card"><div class="value">${{cfg.wr}}%</div><div class="label">Win Rate (95% CI: +/-${{wrCI}}%)</div></div>
        <div class="stat-card"><div class="value">${{cfg.avg_win_r}}R</div><div class="label">Avg Win (R-multiple)</div></div>
        <div class="stat-card"><div class="value">${{cfg.best_n !== null ? (cfg.best_n === 0 ? 'Flat' : 'N=' + cfg.best_n) : 'None'}}</div><div class="label">Recommended N</div></div>
    </div>`;

    // Summary table
    html += `<h3>Summary Table</h3><table><tr>
        <th>N</th><th>MC Avg Ret</th><th>MC Med Ret</th><th>MC Avg MaxDD</th><th>MC Sharpe</th>
        <th>Ruin %</th><th>Avg Ruin @</th><th>IS Sharpe</th><th>OOS Sharpe</th><th>IS/OOS</th>
        <th>Streaks >= N</th><th>Stability</th><th>Status</th></tr>`;

    for (let n = 0; n <= 6; n++) {{
        const mc = cfg.mc[n];
        const iso = cfg.is_oos[n];
        const isRec = cfg.best_n === n;
        const sc = n === 0 ? '-' : (cfg.streak_counts[n] || 0);
        const scVal = n === 0 ? 999 : (cfg.streak_counts[n] || 0);
        const stab = cfg.n_stability !== null ? cfg.n_stability.toFixed(2) : 'n/a';

        let status = '<span class="ok">OK</span>';
        let flags = [];
        if (n > 0 && scVal < 15) flags.push('low_sample');
        if (iso.oos_sharpe < iso.is_sharpe * 0.5 && iso.is_sharpe > 0) flags.push('overfit');
        if (mc.ruin_pct > 50) flags.push('high_ruin');
        if (flags.length > 0) status = '<span class="flag">' + flags.join(', ') + '</span>';

        const cls = isRec ? ' class="highlight"' : '';
        html += `<tr${{cls}}>
            <td>${{N_LABELS[n]}}</td>
            <td>${{fmt(mc.avg_return)}}</td><td>${{fmt(mc.median_return)}}</td>
            <td>$$${{mc.max_dd.toLocaleString()}}</td><td>${{mc.sharpe.toFixed(2)}}</td>
            <td>${{mc.ruin_pct.toFixed(1)}}%</td><td>${{mc.ruin_pct > 0 ? mc.avg_ruin_trade : 'n/a'}}</td>
            <td>${{iso.is_sharpe.toFixed(2)}}</td><td>${{iso.oos_sharpe.toFixed(2)}}</td>
            <td>${{iso.ratio.toFixed(2)}}</td>
            <td>${{sc}}${{(n > 0 && scVal < 15) ? ' <span class="flag">!</span>' : ''}}</td>
            <td>${{stab}}</td><td>${{status}}</td></tr>`;
    }}
    html += '</table>';

    // Walk-forward table
    if (cfg.walk_forward.length > 0) {{
        html += `<h3>Walk-Forward Windows</h3>`;
        if (cfg.walk_forward.length < 3) {{
            html += `<div class="note">Only ${{cfg.walk_forward.length}} windows available. Insufficient for reliable stability assessment.</div>`;
        }}
        html += `<table><tr><th>Window</th><th>Optimal N</th><th>IS Sharpe</th><th>OOS Sharpe</th></tr>`;
        cfg.walk_forward.forEach(w => {{
            html += `<tr><td>${{w.window}}</td><td>${{N_LABELS[w.optimal_n]}}</td>
                <td>${{w.is_sharpe.toFixed(2)}}</td><td>${{w.oos_sharpe.toFixed(2)}}</td></tr>`;
        }});
        html += `</table>`;
    }}

    // Streak distribution
    html += `<h3>Loss Streak Distribution (Full Dataset)</h3><table><tr><th>Streak Length</th><th>Count</th><th>Cumulative >= N</th></tr>`;
    const maxStreak = Math.max(...Object.keys(cfg.streaks).map(Number));
    for (let s = 1; s <= Math.max(maxStreak, 6); s++) {{
        const cnt = cfg.streaks[s] || 0;
        let cum = 0;
        for (let k = s; k <= maxStreak; k++) cum += (cfg.streaks[k] || 0);
        const flag = (s >= 1 && cum < 15) ? ' <span class="warn">low sample</span>' : '';
        html += `<tr><td>${{s}}</td><td>${{cnt}}</td><td>${{cum}}${{flag}}</td></tr>`;
    }}
    html += '</table>';

    // Charts
    html += `<div class="grid">
        <div class="chart-box"><h3>Equity Distribution by N (MC)</h3><canvas id="box${{idx}}"></canvas></div>
        <div class="chart-box"><h3>Risk of Ruin vs N</h3><canvas id="ruin${{idx}}"></canvas></div>
        <div class="chart-box"><h3>Survival Curve (% Alive at Trade #)</h3><canvas id="surv${{idx}}"></canvas></div>
        <div class="chart-box"><h3>IS vs OOS Sharpe by N</h3><canvas id="sharpe${{idx}}"></canvas></div>
        <div class="chart-box"><h3>Historical Equity Curve (In-Sample)</h3><canvas id="eqIS${{idx}}"></canvas></div>
        <div class="chart-box"><h3>Historical Equity Curve (Out-of-Sample)</h3><canvas id="eqOOS${{idx}}"></canvas></div>
    </div>`;

    div.innerHTML = html;
    document.getElementById('report').appendChild(div);

    // --- Render Charts ---

    // Box plot approximation (p5, p25, p50, p75, p95)
    const boxData = [];
    for (let n = 0; n <= 6; n++) {{
        const mc = cfg.mc[n];
        boxData.push({{
            label: N_LABELS[n],
            p5: mc.p5 - 50000, p25: mc.p25 - 50000, p50: mc.p50 - 50000,
            p75: mc.p75 - 50000, p95: mc.p95 - 50000, avg: mc.avg_return
        }});
    }}

    new Chart(document.getElementById('box' + idx), {{
        type: 'bar',
        data: {{
            labels: N_LABELS,
            datasets: [
                {{ label: 'P5-P25', data: boxData.map(b => b.p25 - b.p5), backgroundColor: 'rgba(0,212,255,0.15)',
                   stack: 'box', borderWidth: 0 }},
                {{ label: 'P25-P50', data: boxData.map(b => b.p50 - b.p25), backgroundColor: 'rgba(0,212,255,0.4)',
                   stack: 'box', borderWidth: 0 }},
                {{ label: 'P50-P75', data: boxData.map(b => b.p75 - b.p50), backgroundColor: 'rgba(0,255,136,0.4)',
                   stack: 'box', borderWidth: 0 }},
                {{ label: 'P75-P95', data: boxData.map(b => b.p95 - b.p75), backgroundColor: 'rgba(0,255,136,0.15)',
                   stack: 'box', borderWidth: 0 }},
                {{ label: 'Median', data: boxData.map(b => b.p50), type: 'line',
                   borderColor: '#ffffff', pointBackgroundColor: '#ffffff', borderWidth: 2, pointRadius: 5, fill: false }},
            ]
        }},
        options: {{
            responsive: true,
            plugins: {{
                legend: {{ display: true, labels: {{ color: '#aaa', font: {{ size: 10 }} }} }},
                tooltip: {{ callbacks: {{ label: ctx => '$' + ctx.raw.toLocaleString() }} }}
            }},
            scales: {{
                x: {{ stacked: true, ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }} }},
                y: {{ stacked: true, ticks: {{ color: '#aaa', callback: v => '$' + v.toLocaleString() }},
                      grid: {{ color: '#222' }}, title: {{ display: true, text: 'Return ($)', color: '#888' }} }}
            }}
        }}
    }});

    // Ruin chart
    new Chart(document.getElementById('ruin' + idx), {{
        type: 'bar',
        data: {{
            labels: N_LABELS,
            datasets: [{{ label: 'Risk of Ruin %', data: Object.values(cfg.mc).map(m => m.ruin_pct),
                backgroundColor: Object.values(cfg.mc).map(m => m.ruin_pct > 50 ? '#ff4444' : m.ruin_pct > 20 ? '#ffaa00' : '#00ff88'),
                borderWidth: 0 }}]
        }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ display: false }} }},
            scales: {{
                x: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }} }},
                y: {{ ticks: {{ color: '#aaa', callback: v => v + '%' }}, grid: {{ color: '#222' }},
                      title: {{ display: true, text: 'Ruin %', color: '#888' }}, min: 0, max: 100 }}
            }}
        }}
    }});

    // Survival curves
    const survCheckpoints = Array.from({{length: 20}}, (_, i) => (i + 1) * 50);
    const survDatasets = [];
    for (let n = 0; n <= 6; n++) {{
        survDatasets.push({{
            label: N_LABELS[n],
            data: cfg.survival[n],
            borderColor: COLORS[n],
            backgroundColor: 'transparent',
            borderWidth: n === (cfg.best_n || 0) ? 3 : 1.5,
            pointRadius: 0,
            tension: 0.3,
        }});
    }}
    new Chart(document.getElementById('surv' + idx), {{
        type: 'line',
        data: {{ labels: survCheckpoints, datasets: survDatasets }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ labels: {{ color: '#aaa', font: {{ size: 10 }} }} }} }},
            scales: {{
                x: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }},
                      title: {{ display: true, text: 'Trade #', color: '#888' }} }},
                y: {{ ticks: {{ color: '#aaa', callback: v => v + '%' }}, grid: {{ color: '#222' }},
                      title: {{ display: true, text: '% Sims Alive', color: '#888' }}, min: 0, max: 100 }}
            }}
        }}
    }});

    // IS vs OOS Sharpe
    const isSharpes = [], oosSharpes = [];
    for (let n = 0; n <= 6; n++) {{
        isSharpes.push(cfg.is_oos[n].is_sharpe);
        oosSharpes.push(cfg.is_oos[n].oos_sharpe);
    }}
    new Chart(document.getElementById('sharpe' + idx), {{
        type: 'bar',
        data: {{
            labels: N_LABELS,
            datasets: [
                {{ label: 'In-Sample Sharpe', data: isSharpes, backgroundColor: 'rgba(0,212,255,0.6)', borderWidth: 0 }},
                {{ label: 'Out-of-Sample Sharpe', data: oosSharpes, backgroundColor: 'rgba(255,107,53,0.6)', borderWidth: 0 }},
            ]
        }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ labels: {{ color: '#aaa' }} }} }},
            scales: {{
                x: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }} }},
                y: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }},
                      title: {{ display: true, text: 'Sharpe Ratio', color: '#888' }} }}
            }}
        }}
    }});

    // IS equity curves
    const isDatasets = [];
    for (let n = 0; n <= 6; n++) {{
        isDatasets.push({{
            label: N_LABELS[n],
            data: cfg.is_curves[n].map(v => v - 50000),
            borderColor: COLORS[n],
            backgroundColor: 'transparent',
            borderWidth: n === (cfg.best_n || 0) ? 3 : 1,
            pointRadius: 0,
            tension: 0.1,
        }});
    }}
    new Chart(document.getElementById('eqIS' + idx), {{
        type: 'line',
        data: {{ labels: cfg.is_curves['0'].map((_, i) => i), datasets: isDatasets }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ labels: {{ color: '#aaa', font: {{ size: 10 }} }} }} }},
            scales: {{
                x: {{ ticks: {{ color: '#aaa', maxTicksLimit: 10 }}, grid: {{ color: '#222' }},
                      title: {{ display: true, text: 'Trade #', color: '#888' }} }},
                y: {{ ticks: {{ color: '#aaa', callback: v => '$' + v.toLocaleString() }}, grid: {{ color: '#222' }},
                      title: {{ display: true, text: 'P&L ($)', color: '#888' }} }}
            }}
        }}
    }});

    // OOS equity curves
    const oosDatasets = [];
    for (let n = 0; n <= 6; n++) {{
        oosDatasets.push({{
            label: N_LABELS[n],
            data: cfg.oos_curves[n].map(v => v - 50000),
            borderColor: COLORS[n],
            backgroundColor: 'transparent',
            borderWidth: n === (cfg.best_n || 0) ? 3 : 1,
            pointRadius: 0,
            tension: 0.1,
        }});
    }}
    new Chart(document.getElementById('eqOOS' + idx), {{
        type: 'line',
        data: {{ labels: cfg.oos_curves['0'].map((_, i) => i), datasets: oosDatasets }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ labels: {{ color: '#aaa', font: {{ size: 10 }} }} }} }},
            scales: {{
                x: {{ ticks: {{ color: '#aaa', maxTicksLimit: 10 }}, grid: {{ color: '#222' }},
                      title: {{ display: true, text: 'Trade #', color: '#888' }} }},
                y: {{ ticks: {{ color: '#aaa', callback: v => '$' + v.toLocaleString() }}, grid: {{ color: '#222' }},
                      title: {{ display: true, text: 'P&L ($)', color: '#888' }} }}
            }}
        }}
    }});
}}

// Comparison chart
function renderComparison() {{
    if (DATA.length < 2) return;

    const div = document.createElement('div');
    div.className = 'config-section';
    div.innerHTML = `<h2>Side-by-Side Comparison</h2>
        <div class="grid">
            <div class="chart-box"><h3>Risk of Ruin: Config Comparison</h3><canvas id="cmpRuin"></canvas></div>
            <div class="chart-box"><h3>MC Sharpe: Config Comparison</h3><canvas id="cmpSharpe"></canvas></div>
        </div>`;
    document.getElementById('report').appendChild(div);

    // Ruin comparison
    new Chart(document.getElementById('cmpRuin'), {{
        type: 'bar',
        data: {{
            labels: N_LABELS,
            datasets: DATA.map((cfg, i) => ({{
                label: cfg.name,
                data: Object.values(cfg.mc).map(m => m.ruin_pct),
                backgroundColor: i === 0 ? 'rgba(0,212,255,0.6)' : 'rgba(255,107,53,0.6)',
                borderWidth: 0,
            }}))
        }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ labels: {{ color: '#aaa' }} }} }},
            scales: {{
                x: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }} }},
                y: {{ ticks: {{ color: '#aaa', callback: v => v + '%' }}, grid: {{ color: '#222' }},
                      min: 0, max: 100 }}
            }}
        }}
    }});

    // Sharpe comparison
    new Chart(document.getElementById('cmpSharpe'), {{
        type: 'bar',
        data: {{
            labels: N_LABELS,
            datasets: DATA.map((cfg, i) => ({{
                label: cfg.name,
                data: Object.values(cfg.mc).map(m => m.sharpe),
                backgroundColor: i === 0 ? 'rgba(0,212,255,0.6)' : 'rgba(255,107,53,0.6)',
                borderWidth: 0,
            }}))
        }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ labels: {{ color: '#aaa' }} }} }},
            scales: {{
                x: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }} }},
                y: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }} }}
            }}
        }}
    }});
}}

DATA.forEach((cfg, i) => renderConfig(cfg, i));
renderComparison();
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nHTML report saved to {output_path}")


if __name__ == "__main__":
    print("Building ADX lookup...")
    adx_lookup = build_adx_lookup()
    print(f"  {len(adx_lookup)} rows")

    print("Loading data...")
    days = load_days_with_adx(adx_lookup)
    print(f"  {len(days)} trading days loaded")

    configs = [
        (0.50, 1.90, "SL 0.50 / TP 1.90 (ADX 15-30)"),
        (0.50, 2.10, "SL 0.50 / TP 2.10 (ADX 15-30)"),
    ]

    all_results = []
    for sl_mult, tp_mult, name in configs:
        trades_r = collect_trades_r_multiples(days, sl_mult, tp_mult)
        n_total = len(trades_r)
        wins = np.sum(trades_r > 0)
        wr = wins / n_total
        avg_win_r = trades_r[trades_r > 0].mean()

        result = analyze_config(name, trades_r, wr, avg_win_r)
        all_results.append(result)
        sys.stdout.flush()

    # Generate HTML report
    output = Path("results/html/martingale_analysis.html")
    generate_html_report(all_results, output)

    print("\nDone.")
