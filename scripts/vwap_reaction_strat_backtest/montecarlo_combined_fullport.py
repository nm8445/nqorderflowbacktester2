"""
Monte Carlo: Combined VWAP Strategy — Fullport Multi-Account Challenge.

Combines:
- Combined trade pool (Base ADX 15-30 + Trending ADX 30+)
- Multi-account fullport rotation ($1k risk/trade, dynamic MNQ sizing)

Rules:
- Buy N challenge accounts ($50k each)
- Risk ~$1,000 per trade (contracts = floor($1000 / (SL_pts * $2)))
- Daily loss limit: $1,000 per account
- Max trailing DD: $2,000 per account
- Profit target: $3,000
- One account per trade, rotate on daily-lock or DD blow
- Success = at least 1 account passes

Usage:
    python -u scripts/vwap_reaction_strat_backtest/montecarlo_combined_fullport.py
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
DAILY_LOSS_LIM = 1_000
TRAIL_DD = 2_000
PROFIT_TARGET = 3_000
TARGET_RISK = 1_000  # dollars risked per trade

N_SIMS = 10_000
MAX_DAYS = 500
SEED = 42

START_DATE = "2025-03-13"
END_DATE = "2026-04-08"


def collect_combined_trades_with_sl(adx_lookup):
    """
    Collect combined (base + trending) trades with PnL, MAE, and SL distance.
    Returns per-day dict and total day count.
    Each trade = (pnl_points, mae_points, sl_points).
    """
    start = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end = datetime.strptime(END_DATE, "%Y-%m-%d").date()

    daily_trades = {}
    all_dates = set()

    for cache_file in sorted(VWAP_REACTION_CACHE_DIR.glob("*.pkl")):
        date_str = cache_file.stem
        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if file_date < start or file_date > end:
            continue
        all_dates.add(date_str)

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

        # Build unified signal list with SL info
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
            sl_pts = atr * BASE_SL
            tp_pts = atr * BASE_TP
            if d == "long":
                sl = ep - sl_pts; tp = ep + tp_pts
                if tp <= ep or sl >= ep: continue
            else:
                sl = ep + sl_pts; tp = ep - tp_pts
                if tp >= ep or sl <= ep: continue
            all_signals.append({
                "source": "base", "bar_index": s["bar_index"],
                "confirm_time": ct, "confirm_time_et": ct_et,
                "entry_price": ep, "direction": d, "sl": sl, "tp": tp,
                "sl_pts": sl_pts,
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
            sl_pts = atr * TREND_SL
            tp_pts = atr * TREND_TP
            if d == "long":
                sl = ep - sl_pts; tp = ep + tp_pts
            else:
                sl = ep + sl_pts; tp = ep - tp_pts
            all_signals.append({
                "source": "trend", "bar_index": s["bar_index"],
                "confirm_time": ct, "confirm_time_et": ct_et,
                "entry_price": ep, "direction": d, "sl": sl, "tp": tp,
                "sl_pts": sl_pts,
            })

        all_signals.sort(key=lambda x: x["confirm_time"])

        # Simulate with shared position lock
        day_trades = []
        last_exit_time = None

        for sig in all_signals:
            if last_exit_time is not None and sig["confirm_time"] <= last_exit_time:
                continue

            ep = sig["entry_price"]
            d = sig["direction"]
            sl_price = sig["sl"]
            tp_price = sig["tp"]
            sl_pts = sig["sl_pts"]
            ci = sig["bar_index"] + 1

            exit_price = None
            exit_time = None
            worst_adverse = 0.0

            for j in range(ci + 1, len(bars)):
                bar = bars[j]
                if not bar.closed:
                    continue
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
            day_trades.append((pnl, worst_adverse, sl_pts))
            last_exit_time = exit_time

        if day_trades:
            daily_trades[date_str] = day_trades

    return daily_trades, len(all_dates)


def build_trade_pool(daily_trades):
    """Flatten daily trades into numpy array + day count distribution."""
    all_trades = []
    day_counts = []
    for date_str in sorted(daily_trades.keys()):
        trades = daily_trades[date_str]
        day_counts.append(len(trades))
        all_trades.extend(trades)

    trade_pool = np.array(all_trades, dtype=np.float64)  # (N, 3): pnl, mae, sl_pts
    day_count_dist = np.array(day_counts, dtype=np.int32)
    return trade_pool, day_count_dist


def sim_multi_account(trade_pool, day_count_dist, n_accounts, rng):
    """
    Fullport multi-account rotation simulation.
    Dynamic sizing: contracts = floor($1000 / (sl_pts * $2)).
    """
    pnl_arr = trade_pool[:, 0]
    mae_arr = trade_pool[:, 1]
    sl_arr = trade_pool[:, 2]

    passed_any = 0
    all_blown = 0
    pass_days_list = []
    accounts_used_list = []
    pass_counts = np.zeros(n_accounts + 1, dtype=int)

    for sim in range(N_SIMS):
        equities = np.full(n_accounts, float(ACCOUNT_SIZE))
        hwms = np.full(n_accounts, float(ACCOUNT_SIZE))
        dd_floors = np.full(n_accounts, float(ACCOUNT_SIZE - TRAIL_DD))
        alive = np.ones(n_accounts, dtype=bool)
        daily_loss = np.zeros(n_accounts)
        daily_locked = np.zeros(n_accounts, dtype=bool)

        sim_passed = False
        sim_pass_day = 0
        n_passed = 0
        accounts_touched = set()

        for day in range(MAX_DAYS):
            daily_loss[:] = 0
            daily_locked[:] = False

            n_trades_today = rng.choice(day_count_dist)
            if n_trades_today == 0:
                continue

            trade_idxs = rng.integers(0, len(trade_pool), size=n_trades_today)

            for ti in trade_idxs:
                pnl_pts = pnl_arr[ti]
                mae_pts = mae_arr[ti]
                sl_pts = sl_arr[ti]

                # Pick first available account
                acct = -1
                for a in range(n_accounts):
                    if alive[a] and not daily_locked[a]:
                        acct = a
                        break
                if acct == -1:
                    break

                accounts_touched.add(acct)

                # Dynamic sizing: risk ~$1000
                sl_dollars_per_con = sl_pts * MNQ_PV
                if sl_dollars_per_con <= 0:
                    continue
                contracts = int(TARGET_RISK / sl_dollars_per_con)
                if contracts < 1:
                    contracts = 1

                trade_pnl_dollars = pnl_pts * MNQ_PV * contracts
                trade_mae_dollars = mae_pts * MNQ_PV * contracts

                # Check unrealized blowup (MAE)
                if equities[acct] - trade_mae_dollars <= dd_floors[acct]:
                    alive[acct] = False
                    continue

                # Check if MAE would breach daily loss limit
                if daily_loss[acct] + trade_mae_dollars >= DAILY_LOSS_LIM:
                    daily_locked[acct] = True
                    equities[acct] += trade_pnl_dollars
                    daily_loss[acct] += max(0, -trade_pnl_dollars)
                    if equities[acct] <= dd_floors[acct]:
                        alive[acct] = False
                    elif equities[acct] > hwms[acct]:
                        hwms[acct] = equities[acct]
                        dd_floors[acct] = hwms[acct] - TRAIL_DD
                    if equities[acct] - ACCOUNT_SIZE >= PROFIT_TARGET and alive[acct]:
                        n_passed += 1
                        alive[acct] = False
                    continue

                # Apply trade
                equities[acct] += trade_pnl_dollars
                if trade_pnl_dollars < 0:
                    daily_loss[acct] += abs(trade_pnl_dollars)

                if equities[acct] <= dd_floors[acct]:
                    alive[acct] = False
                    continue

                if equities[acct] > hwms[acct]:
                    hwms[acct] = equities[acct]
                    dd_floors[acct] = hwms[acct] - TRAIL_DD

                if daily_loss[acct] >= DAILY_LOSS_LIM:
                    daily_locked[acct] = True

                if equities[acct] - ACCOUNT_SIZE >= PROFIT_TARGET:
                    n_passed += 1
                    alive[acct] = False

            if not any(alive):
                break

            if n_passed > 0 and not sim_passed:
                sim_passed = True
                sim_pass_day = day + 1

        if sim_passed or n_passed > 0:
            passed_any += 1
            pass_days_list.append(sim_pass_day if sim_pass_day > 0 else day + 1)
        if not any(alive) and n_passed == 0:
            all_blown += 1
        accounts_used_list.append(len(accounts_touched))
        pass_counts[n_passed] += 1

    return {
        "pass_rate": passed_any / N_SIMS * 100,
        "avg_days": np.mean(pass_days_list) if pass_days_list else 0,
        "all_blown_rate": all_blown / N_SIMS * 100,
        "avg_accounts_used": np.mean(accounts_used_list),
        "pass_counts": pass_counts,
    }


def sim_challenge_fixed(trade_pool, day_count_dist, contracts, rng):
    """Fixed-size challenge sim (no rotation, single account)."""
    pnl_arr = trade_pool[:, 0]
    mae_arr = trade_pool[:, 1]

    passes = 0
    blowups = 0
    pass_days = []

    for sim in range(N_SIMS):
        equity = float(ACCOUNT_SIZE)
        hwm = float(ACCOUNT_SIZE)
        dd_level = float(ACCOUNT_SIZE - TRAIL_DD)
        passed = False
        blown = False

        for day in range(MAX_DAYS):
            nt = rng.choice(day_count_dist)
            if nt == 0:
                continue
            idxs = rng.integers(0, len(trade_pool), size=nt)
            for ti in idxs:
                trade_pnl = pnl_arr[ti] * MNQ_PV * contracts
                trade_mae = mae_arr[ti] * MNQ_PV * contracts
                if equity - trade_mae <= dd_level:
                    blown = True; break
                equity += trade_pnl
                if equity <= dd_level:
                    blown = True; break
            if blown:
                break
            if equity > hwm:
                hwm = equity; dd_level = hwm - TRAIL_DD
            if equity <= dd_level:
                blown = True; break
            if equity - ACCOUNT_SIZE >= PROFIT_TARGET:
                passed = True; pass_days.append(day + 1); break

        if passed:
            passes += 1
        if blown:
            blowups += 1

    return {
        "pass_rate": passes / N_SIMS * 100,
        "avg_days": np.mean(pass_days) if pass_days else 0,
        "blow_rate": blowups / N_SIMS * 100,
    }


def main():
    print("=" * 110)
    print("  MONTE CARLO -- COMBINED VWAP STRATEGY -- FULLPORT MULTI-ACCOUNT")
    print(f"  Base: SL {BASE_SL}x / TP {BASE_TP}x (ADX 15-30) @ VWAP")
    print(f"  Trend: SL {TREND_SL}x / TP {TREND_TP}x (ADX 30+) @ STD{TREND_STD} band, LB={TREND_LB}")
    print(f"  Fullport: risk ${TARGET_RISK}/trade | Daily loss limit: ${DAILY_LOSS_LIM}")
    print(f"  Account: ${ACCOUNT_SIZE:,} | Trail DD: ${TRAIL_DD:,} | Target: ${PROFIT_TARGET:,}")
    print(f"  {N_SIMS:,} sims per config")
    print("=" * 110)
    sys.stdout.flush()

    print("\nBuilding ADX lookup...", end=" ", flush=True)
    adx_lookup = build_adx_lookup()
    print(f"done ({len(adx_lookup)} rows)")

    print("Collecting combined trades (base + trending)...", end=" ", flush=True)
    daily_trades, n_total_days = collect_combined_trades_with_sl(adx_lookup)
    trade_pool, day_count_dist = build_trade_pool(daily_trades)
    print(f"done")

    # Trade pool stats
    n = len(trade_pool)
    pnls = trade_pool[:, 0]
    maes = trade_pool[:, 1]
    sls = trade_pool[:, 2]
    wins = pnls > 0
    wr = np.sum(wins) / n * 100

    # Dynamic sizing stats
    sl_dollars = sls * MNQ_PV
    contracts = np.maximum(1, np.floor(TARGET_RISK / sl_dollars)).astype(int)
    actual_risk = sls * MNQ_PV * contracts
    actual_pnl = pnls * MNQ_PV * contracts

    n_base = 0
    n_trend = 0
    for dt in daily_trades.values():
        n_base += len(dt)  # we can't easily separate, just show total
    trading_days = len(daily_trades)

    print(f"\n  Combined trade pool:")
    print(f"    {n} trades over {n_total_days} calendar days ({trading_days} trading days)")
    print(f"    Avg trades/trading day: {n / trading_days:.1f}")
    print(f"    WR: {wr:.1f}%")
    print(f"    NQ equiv (1 contract): PF {abs(pnls[pnls > 0].sum()) / abs(pnls[pnls < 0].sum()):.2f} | Total ${pnls.sum() * POINT_VALUE:+,.0f}")
    print(f"    Fullport dynamic sizing:")
    print(f"      Contracts per trade: min={contracts.min()} median={int(np.median(contracts))} max={contracts.max()}")
    print(f"      Risk per trade: ${actual_risk.mean():.0f} avg (${actual_risk.min():.0f}-${actual_risk.max():.0f})")
    print(f"      Avg win: ${actual_pnl[actual_pnl > 0].mean():.0f} | Avg loss: ${abs(actual_pnl[actual_pnl < 0].mean()):.0f}")
    sys.stdout.flush()

    rng = np.random.default_rng(SEED)

    # ===================================================================
    # PART 1: Fixed-size challenge (1 account, MNQ 1-9) — for comparison
    # ===================================================================
    print(f"\n{'=' * 110}")
    print(f"  PART 1: FIXED-SIZE CHALLENGE (1 account, no rotation)")
    print(f"{'=' * 110}")
    print(f"  {'MNQ':>4} {'Pass%':>7} {'AvgDays':>8} {'Blow%':>7}")
    print(f"  {'----':>4} {'-------':>7} {'--------':>8} {'-------':>7}")
    sys.stdout.flush()

    for c in range(1, 10):
        r = sim_challenge_fixed(trade_pool, day_count_dist, c, rng)
        print(f"  {c:>4} {r['pass_rate']:>6.1f}% {r['avg_days']:>8.0f} {r['blow_rate']:>6.1f}%", flush=True)

    # ===================================================================
    # PART 2: Fullport multi-account rotation
    # ===================================================================
    account_counts = [1, 2, 3, 4, 5, 6, 8, 10]

    print(f"\n{'=' * 110}")
    print(f"  PART 2: FULLPORT MULTI-ACCOUNT ROTATION")
    print(f"  Risk ${TARGET_RISK}/trade | Dynamic MNQ sizing | Rotate on daily lock")
    print(f"{'=' * 110}")
    print(f"  {'Accounts':>8}  {'Pass>=1':>8}  {'Avg Days':>9}  {'All Blown':>10}  {'Avg Accts Used':>15}")
    print(f"  {'-' * 65}")
    sys.stdout.flush()

    for n_accts in account_counts:
        rng_acct = np.random.default_rng(SEED)
        result = sim_multi_account(trade_pool, day_count_dist, n_accts, rng_acct)
        print(f"  {n_accts:>8d}  {result['pass_rate']:>7.1f}%  {result['avg_days']:>8.1f}d  "
              f"{result['all_blown_rate']:>9.1f}%  {result['avg_accounts_used']:>14.1f}", flush=True)

    # ===================================================================
    # PART 3: Cost-benefit analysis
    # ===================================================================
    print(f"\n{'=' * 110}")
    print(f"  PART 3: COST-BENEFIT (assuming $150 per account)")
    print(f"{'=' * 110}")
    ACCOUNT_COST = 150  # typical prop firm challenge fee
    print(f"  {'Accounts':>8}  {'Cost':>7}  {'Pass>=1':>8}  {'EV per attempt':>15}  {'Attempts to pass':>17}")
    print(f"  {'-' * 70}")
    sys.stdout.flush()

    for n_accts in account_counts:
        rng_ev = np.random.default_rng(SEED)
        result = sim_multi_account(trade_pool, day_count_dist, n_accts, rng_ev)
        cost = n_accts * ACCOUNT_COST
        pass_rate = result["pass_rate"] / 100
        if pass_rate > 0:
            attempts_to_pass = 1 / pass_rate
            ev = pass_rate * PROFIT_TARGET - cost  # simplified EV
        else:
            attempts_to_pass = float('inf')
            ev = -cost
        total_cost_to_pass = cost * attempts_to_pass if pass_rate > 0 else float('inf')
        print(f"  {n_accts:>8d}  ${cost:>5}  {result['pass_rate']:>7.1f}%  ${ev:>+13,.0f}  "
              f"{attempts_to_pass:>16.1f}x", flush=True)

    print(f"\n{'=' * 110}")
    print(f"  PART 4: RECOMMENDATIONS")
    print(f"{'=' * 110}")
    print(f"\n  Strategy: Combined VWAP (base + trending)")
    print(f"  Fullport rotation with ~${TARGET_RISK} risk/trade")
    print(f"\n  Check results above for optimal account count.")
    sys.stdout.flush()

    print("\nDone.")


if __name__ == "__main__":
    main()
