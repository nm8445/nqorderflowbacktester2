"""
Monte Carlo prop firm simulation — Combined VWAP Strategy.

Base (ADX 15-30, SL 0.50x, TP 1.90x @ VWAP) + Trending (ADX 30+, SL 1.00x, TP 1.00x @ STD1 band)
Both strategies merged per day, shared position lock (one trade at a time).

Part 1: Challenge pass rate (MNQ 1-9)
Part 2: Funded payouts + avg days per payout (MNQ 1-9)
Part 3: Slippage resistance
Part 4: Summary — optimal contracts for challenge vs funded

Usage:
    python -u scripts/vwap_reaction_strat_backtest/montecarlo_combined.py
"""

import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from adx_filter_common import (
    ET, DATA_DIR, build_adx_lookup, get_adx_at_time, passes_adx_filter,
    LUNCH_START, LUNCH_END,
)
from trending_vwap_atr_grid import (
    load_pickle, load_5min_bars,
    precompute_day_state, detect_signals as detect_trending_signals,
)

VWAP_REACTION_CACHE_DIR = DATA_DIR / "vwap_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
VWAP_PRICE_CACHE_DIR = DATA_DIR / "vwap_cache"
TIMEBARS_DIR = DATA_DIR / "timebars_5min"

POINT_VALUE = 20.0
MNQ_PV = 2.0

ENTRY_CUTOFF = "16:00"
FORCE_CLOSE = "16:58"

# Strategy configs
BASE_SL = 0.50
BASE_TP = 1.90
TREND_SL = 1.00
TREND_TP = 1.00
TREND_STD = 1
TREND_LB = 14
TREND_ADX_MIN = 30.0

# Prop firm params
ACCOUNT_SIZE = 50_000
TRAIL_DD = 2_000
PROFIT_TARGET = 3_000
FUNDED_BUFFER = 2_000

N_SIMS = 10_000
MAX_DAYS_CHALLENGE = 500
MAX_DAYS_FUNDED = 1500
SEED = 42

START_DATE = "2025-03-13"
END_DATE = "2026-04-08"


def collect_combined_trades(adx_lookup):
    """
    Run combined backtest (base + trending), return trade pool with PnL and MAE.
    Matches generate_combined_html.py logic exactly.
    """
    start = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end = datetime.strptime(END_DATE, "%Y-%m-%d").date()

    all_trade_pnls = []
    all_trade_maes = []
    daily_trade_counts = []  # trades per calendar day (including 0-trade days)
    all_dates = set()

    daily_pnls = {}
    daily_maes = {}

    for cache_file in sorted(VWAP_REACTION_CACHE_DIR.glob("*.pkl")):
        date_str = cache_file.stem
        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if file_date < start or file_date > end:
            continue
        all_dates.add(date_str)

        # Load range bars
        signal_file = SIGNAL_CACHE_DIR / f"{date_str}.pkl"
        if not signal_file.exists():
            continue
        signal_data = load_pickle(signal_file)
        bars = signal_data["bars"]
        if not bars:
            continue

        # Base VWAP signals
        vwap_data = load_pickle(cache_file)
        base_signals = vwap_data["signals"]

        # Trending band signals
        vwap_price_file = VWAP_PRICE_CACHE_DIR / f"{date_str}.pkl"
        tb_file = TIMEBARS_DIR / f"timebars_5min_{date_str.replace('-', '_')}.pkl"
        trend_signals = []

        if vwap_price_file.exists() and tb_file.exists():
            vwap_df = load_pickle(vwap_price_file)
            bars_5min = load_5min_bars(date_str)
            if bars_5min is not None:
                day_state = precompute_day_state(bars_5min, vwap_df, TREND_LB, TREND_STD)
                trend_signals = detect_trending_signals(bars, day_state)
                for s in trend_signals:
                    s["date"] = date_str

        # Build unified signal list
        all_signals = []

        for s in base_signals:
            ct = s["confirm_time"]
            ct_et = ct.tz_convert(ET) if hasattr(ct, "tz_convert") else pd.Timestamp(ct, tz="UTC").tz_convert(ET)
            if "16:00" <= ct_et.strftime("%H:%M") < "19:10":
                continue
            if LUNCH_START <= ct_et.strftime("%H:%M") < LUNCH_END:
                continue
            atr = s["atr"]
            if atr is None or atr <= 0:
                continue
            adx_val = get_adx_at_time(ct_et, adx_lookup)
            if not passes_adx_filter(adx_val):
                continue
            ep = s["entry_price"]
            d = s["direction"]
            if d == "long":
                sl = ep - atr * BASE_SL; tp = ep + atr * BASE_TP
                if tp <= ep or sl >= ep: continue
            else:
                sl = ep + atr * BASE_SL; tp = ep - atr * BASE_TP
                if tp >= ep or sl <= ep: continue
            all_signals.append({
                "source": "base", "bar_index": s["bar_index"],
                "confirm_time": ct, "confirm_time_et": ct_et,
                "entry_price": ep, "direction": d, "sl": sl, "tp": tp,
            })

        for s in trend_signals:
            ct = s["confirm_time"]
            ct_et = ct.tz_convert(ET) if hasattr(ct, "tz_convert") else pd.Timestamp(ct, tz="UTC").tz_convert(ET)
            if "16:00" <= ct_et.strftime("%H:%M") < "19:10":
                continue
            if LUNCH_START <= ct_et.strftime("%H:%M") < LUNCH_END:
                continue
            atr = s["atr"]
            if atr is None or atr <= 0:
                continue
            adx_val = get_adx_at_time(ct_et, adx_lookup)
            if np.isnan(adx_val) or adx_val < TREND_ADX_MIN:
                continue
            ep = s["entry_price"]
            d = s["direction"]
            if d == "long":
                sl = ep - atr * TREND_SL; tp = ep + atr * TREND_TP
            else:
                sl = ep + atr * TREND_SL; tp = ep - atr * TREND_TP
            all_signals.append({
                "source": "trend", "bar_index": s["bar_index"],
                "confirm_time": ct, "confirm_time_et": ct_et,
                "entry_price": ep, "direction": d, "sl": sl, "tp": tp,
            })

        all_signals.sort(key=lambda x: x["confirm_time"])

        # Simulate with shared position lock
        day_trades_pnl = []
        day_trades_mae = []
        last_exit_time = None

        for sig in all_signals:
            if last_exit_time is not None and sig["confirm_time"] <= last_exit_time:
                continue

            ep = sig["entry_price"]
            d = sig["direction"]
            sl_price = sig["sl"]
            tp_price = sig["tp"]
            ci = sig["bar_index"] + 1

            exit_price = None
            exit_time = None
            worst_adverse = 0.0

            for j in range(ci + 1, len(bars)):
                bar = bars[j]
                if not bar.closed:
                    continue

                # Track MAE
                if d == "long":
                    adverse = ep - bar.low
                else:
                    adverse = bar.high - ep
                if adverse > worst_adverse:
                    worst_adverse = adverse

                bct = bar.close_time
                bet = bct.tz_convert(ET) if hasattr(bct, "tz_convert") else pd.Timestamp(bct, tz="UTC").tz_convert(ET)
                if bet.strftime("%H:%M") >= FORCE_CLOSE:
                    exit_price = bar.close; exit_time = bct; break
                if d == "long":
                    if bar.low <= sl_price: exit_price = sl_price; exit_time = bct; break
                    if bar.high >= tp_price: exit_price = tp_price; exit_time = bct; break
                else:
                    if bar.high >= sl_price: exit_price = sl_price; exit_time = bct; break
                    if bar.low <= tp_price: exit_price = tp_price; exit_time = bct; break

            if exit_price is None:
                exit_price = bars[-1].close; exit_time = bars[-1].close_time

            pnl = (exit_price - ep) if d == "long" else (ep - exit_price)
            day_trades_pnl.append(pnl)
            day_trades_mae.append(worst_adverse)
            last_exit_time = exit_time

        if day_trades_pnl:
            daily_pnls[date_str] = day_trades_pnl
            daily_maes[date_str] = day_trades_mae

    # Build arrays in date order, with 0-trade days included
    for d in sorted(all_dates):
        if d in daily_pnls:
            daily_trade_counts.append(len(daily_pnls[d]))
            all_trade_pnls.extend(daily_pnls[d])
            all_trade_maes.extend(daily_maes[d])
        else:
            daily_trade_counts.append(0)

    return (np.array(all_trade_pnls, dtype=np.float64),
            np.array(all_trade_maes, dtype=np.float64),
            np.array(daily_trade_counts, dtype=np.int32))


def sim_challenge(trade_pool, mae_pool, day_count_dist, pv, contracts, rng):
    trades_per_day = rng.choice(day_count_dist, size=(N_SIMS, MAX_DAYS_CHALLENGE))
    total_needed = int(trades_per_day.sum())
    idx = rng.integers(0, len(trade_pool), size=total_needed)
    all_pnls = trade_pool[idx]
    all_maes = mae_pool[idx]

    passes = 0
    pass_days = []
    blowups = 0
    cursor = 0

    for sim in range(N_SIMS):
        equity = float(ACCOUNT_SIZE)
        hwm = float(ACCOUNT_SIZE)
        dd_level = float(ACCOUNT_SIZE - TRAIL_DD)
        passed = False
        blown = False
        day = 0

        for di in range(MAX_DAYS_CHALLENGE):
            nt = int(trades_per_day[sim, di])
            if nt == 0:
                day += 1
                continue

            for _ in range(nt):
                trade_pnl = all_pnls[cursor] * pv * contracts
                trade_mae = all_maes[cursor] * pv * contracts
                cursor += 1
                if equity - trade_mae <= dd_level:
                    blown = True; break
                equity += trade_pnl
                if equity <= dd_level:
                    blown = True; break

            if blown:
                for rd in range(di + 1, MAX_DAYS_CHALLENGE):
                    cursor += int(trades_per_day[sim, rd])
                break

            day += 1
            if equity > hwm:
                hwm = equity; dd_level = hwm - TRAIL_DD
            if equity <= dd_level:
                blown = True
                for rd in range(di + 1, MAX_DAYS_CHALLENGE):
                    cursor += int(trades_per_day[sim, rd])
                break

            total_profit = equity - ACCOUNT_SIZE
            if total_profit >= PROFIT_TARGET:
                passed = True
                for rd in range(di + 1, MAX_DAYS_CHALLENGE):
                    cursor += int(trades_per_day[sim, rd])
                break

        if passed:
            passes += 1
            pass_days.append(day)
        if blown:
            blowups += 1

    return (passes / N_SIMS * 100,
            np.mean(pass_days) if pass_days else 0,
            np.median(pass_days) if pass_days else 0,
            blowups / N_SIMS * 100)


def sim_funded(trade_pool, mae_pool, day_count_dist, pv, contracts, rng):
    trades_per_day = rng.choice(day_count_dist, size=(N_SIMS, MAX_DAYS_FUNDED))
    total_needed = int(trades_per_day.sum())
    idx = rng.integers(0, len(trade_pool), size=total_needed)
    all_pnls = trade_pool[idx]
    all_maes = mae_pool[idx]

    all_payouts = []
    all_dpp = []
    blowups = 0
    cursor = 0

    for sim in range(N_SIMS):
        equity = float(ACCOUNT_SIZE)
        hwm = float(ACCOUNT_SIZE)
        dd_level = float(ACCOUNT_SIZE - TRAIL_DD)
        dd_locked = False
        payouts = 0
        cycle_start = 0
        blown = False

        for di in range(MAX_DAYS_FUNDED):
            nt = int(trades_per_day[sim, di])
            if nt == 0:
                continue

            for _ in range(nt):
                trade_pnl = all_pnls[cursor] * pv * contracts
                trade_mae = all_maes[cursor] * pv * contracts
                cursor += 1
                if equity - trade_mae <= dd_level:
                    blown = True; break
                equity += trade_pnl
                if equity <= dd_level:
                    blown = True; break

            if blown:
                for rd in range(di + 1, MAX_DAYS_FUNDED):
                    cursor += int(trades_per_day[sim, rd])
                break

            if equity > hwm:
                hwm = equity
                if not dd_locked:
                    dd_level = hwm - TRAIL_DD
                    if dd_level >= ACCOUNT_SIZE:
                        dd_level = float(ACCOUNT_SIZE)
                        dd_locked = True

            if equity <= dd_level:
                blown = True
                for rd in range(di + 1, MAX_DAYS_FUNDED):
                    cursor += int(trades_per_day[sim, rd])
                break

            buffer_eq = ACCOUNT_SIZE + FUNDED_BUFFER
            profit = equity - buffer_eq
            if profit >= PROFIT_TARGET:
                payouts += 1
                all_dpp.append(di + 1 - cycle_start)
                equity = buffer_eq
                hwm = buffer_eq
                cycle_start = di + 1

        all_payouts.append(payouts)
        if blown:
            blowups += 1

    return (np.mean(all_payouts),
            np.mean(all_dpp) if all_dpp else 0,
            np.median(all_dpp) if all_dpp else 0,
            blowups / N_SIMS * 100)


def main():
    print("=" * 100)
    print("  MONTE CARLO PROP FIRM SIMULATION -- COMBINED VWAP STRATEGY")
    print(f"  Base: SL {BASE_SL}x / TP {BASE_TP}x (ADX 15-30) @ VWAP")
    print(f"  Trend: SL {TREND_SL}x / TP {TREND_TP}x (ADX 30+) @ STD{TREND_STD} band, LB={TREND_LB}")
    print(f"  Account: ${ACCOUNT_SIZE:,} | Trail DD: ${TRAIL_DD:,} | Target: ${PROFIT_TARGET:,}")
    print(f"  {N_SIMS:,} simulations | MNQ sizing ($2/pt)")
    print("=" * 100)
    sys.stdout.flush()

    print("\nBuilding ADX lookup...", end=" ", flush=True)
    adx_lookup = build_adx_lookup()
    print(f"done ({len(adx_lookup)} rows)")

    print("Collecting combined trades...", end=" ", flush=True)
    trade_pool, mae_pool, day_counts = collect_combined_trades(adx_lookup)
    print(f"done")

    # Stats
    n = len(trade_pool)
    w = trade_pool[trade_pool > 0]
    l = trade_pool[trade_pool < 0]
    wr = len(w) / n * 100
    gp = w.sum() * POINT_VALUE
    gl = abs(l.sum()) * POINT_VALUE if len(l) else 1
    pf = gp / gl
    exp_nq = trade_pool.mean() * POINT_VALUE
    total_nq = trade_pool.sum() * POINT_VALUE
    cum = np.cumsum(trade_pool) * POINT_VALUE
    dd_nq = float((cum - np.maximum.accumulate(cum)).min())

    trading_days = np.sum(day_counts > 0)
    avg_trades_per_day = np.mean(day_counts[day_counts > 0]) if trading_days > 0 else 0

    print(f"\n  Combined trade pool:")
    print(f"    {n} trades over {len(day_counts)} calendar days ({trading_days} trading days)")
    print(f"    Avg trades/trading day: {avg_trades_per_day:.1f}")
    print(f"    WR: {wr:.1f}% | PF: {pf:.2f} | Exp: ${exp_nq:+.0f}/trade (1 NQ)")
    print(f"    Total P&L: ${total_nq:+,.0f} (1 NQ) | Max DD: ${dd_nq:+,.0f}")
    print(f"    MNQ equiv: Total ${total_nq*0.1:+,.0f} | DD ${dd_nq*0.1:+,.0f}")

    rng = np.random.default_rng(SEED)
    contract_range = list(range(1, 10))

    # ===================================================================
    # PART 1: Challenge pass rate
    # ===================================================================
    print(f"\n{'=' * 100}")
    print(f"  PART 1: CHALLENGE PASS RATE -- MNQ 1-9")
    print(f"  Account: ${ACCOUNT_SIZE:,} | Trail DD: ${TRAIL_DD:,} | Target: ${PROFIT_TARGET:,} | {N_SIMS:,} sims")
    print(f"{'=' * 100}")
    print(f"  {'MNQ':>4} {'Pass%':>7} {'AvgDays':>8} {'MedDays':>8} {'Blow%':>7} {'Risk/Trade':>11}")
    print(f"  {'----':>4} {'-------':>7} {'--------':>8} {'--------':>8} {'-------':>7} {'-----------':>11}")
    sys.stdout.flush()

    challenge_results = []
    for c in contract_range:
        rate, avg_d, med_d, blow = sim_challenge(trade_pool, mae_pool, day_counts, MNQ_PV, c, rng)
        avg_risk = abs(trade_pool.mean()) * MNQ_PV * c
        print(f"  {c:>4} {rate:>6.1f}% {avg_d:>8.0f} {med_d:>8.0f} {blow:>6.1f}% ${avg_risk:>10.0f}", flush=True)
        challenge_results.append({"contracts": c, "pass": rate, "avg_days": avg_d, "med_days": med_d, "blow": blow})

    # ===================================================================
    # PART 2: Funded payouts
    # ===================================================================
    print(f"\n{'=' * 100}")
    print(f"  PART 2: FUNDED PAYOUTS -- MNQ 1-9")
    print(f"  DD locks at ${ACCOUNT_SIZE:,} | Buffer: ${FUNDED_BUFFER:,} | Payout: ${PROFIT_TARGET:,}")
    print(f"{'=' * 100}")
    print(f"  {'MNQ':>4} {'AvgPay':>7} {'AvgD/P':>7} {'MedD/P':>7} {'Blow%':>7} {'NetLifetime':>12} {'$/month':>8}")
    print(f"  {'----':>4} {'-------':>7} {'-------':>7} {'-------':>7} {'-------':>7} {'------------':>12} {'--------':>8}")
    sys.stdout.flush()

    funded_results = []
    for c in contract_range:
        avg_pay, avg_dpp, med_dpp, blow = sim_funded(trade_pool, mae_pool, day_counts, MNQ_PV, c, rng)
        net = avg_pay * PROFIT_TARGET - TRAIL_DD
        monthly = (PROFIT_TARGET / avg_dpp * 21) if avg_dpp > 0 else 0  # ~21 trading days/month
        print(f"  {c:>4} {avg_pay:>7.1f} {avg_dpp:>7.0f} {med_dpp:>7.0f} {blow:>6.1f}% ${net:>+11,.0f} ${monthly:>7,.0f}", flush=True)
        funded_results.append({"contracts": c, "payouts": avg_pay, "avg_dpp": avg_dpp, "med_dpp": med_dpp, "blow": blow, "net": net, "monthly": monthly})

    # ===================================================================
    # PART 3: Slippage resistance
    # ===================================================================
    print(f"\n{'=' * 100}")
    print(f"  PART 3: SLIPPAGE RESISTANCE")
    print(f"  Challenge pass rate with 0.25/0.50/1.00 pt roundtrip slippage")
    print(f"{'=' * 100}")

    slippages = [0.25, 0.50, 1.00]
    print(f"  {'MNQ':>4}", end="")
    for slip in slippages:
        print(f" | {slip}pt Pass%  Blow%", end="")
    print()
    sys.stdout.flush()

    for c in contract_range:
        line = f"  {c:>4}"
        for slip in slippages:
            slipped = trade_pool - slip
            rate, _, _, blow = sim_challenge(slipped, mae_pool, day_counts, MNQ_PV, c, rng)
            line += f" | {rate:>10.1f}% {blow:>5.1f}%"
        print(line, flush=True)

    # ===================================================================
    # PART 4: Summary
    # ===================================================================
    print(f"\n{'=' * 100}")
    print(f"  PART 4: RECOMMENDATIONS")
    print(f"{'=' * 100}")

    # Best for challenge
    best_ch = max(challenge_results, key=lambda x: x["pass"])
    print(f"\n  CHALLENGE (passing the eval):")
    print(f"    Best pass rate: MNQ x{best_ch['contracts']} -> {best_ch['pass']:.1f}% pass, {best_ch['med_days']:.0f} median days, {best_ch['blow']:.1f}% blow")

    # Find sweet spot: high pass + low blow
    safe_ch = [r for r in challenge_results if r["blow"] < 30]
    if safe_ch:
        best_safe = max(safe_ch, key=lambda x: x["pass"])
        print(f"    Safest (blow < 30%): MNQ x{best_safe['contracts']} -> {best_safe['pass']:.1f}% pass, {best_safe['med_days']:.0f} median days, {best_safe['blow']:.1f}% blow")

    # Best for funded
    best_fd = max(funded_results, key=lambda x: x["net"])
    print(f"\n  FUNDED (maximizing payouts):")
    print(f"    Max net lifetime: MNQ x{best_fd['contracts']} -> {best_fd['payouts']:.1f} payouts, ${best_fd['net']:+,.0f} net, ~${best_fd['monthly']:,.0f}/month")
    print(f"    Avg {best_fd['avg_dpp']:.0f} days per payout (median {best_fd['med_dpp']:.0f}), {best_fd['blow']:.1f}% blow risk")

    safe_fd = [r for r in funded_results if r["blow"] < 20]
    if safe_fd:
        best_safe_fd = max(safe_fd, key=lambda x: x["net"])
        print(f"    Safest (blow < 20%): MNQ x{best_safe_fd['contracts']} -> {best_safe_fd['payouts']:.1f} payouts, ${best_safe_fd['net']:+,.0f} net, ~${best_safe_fd['monthly']:,.0f}/month")
        print(f"    Avg {best_safe_fd['avg_dpp']:.0f} days per payout (median {best_safe_fd['med_dpp']:.0f}), {best_safe_fd['blow']:.1f}% blow risk")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
