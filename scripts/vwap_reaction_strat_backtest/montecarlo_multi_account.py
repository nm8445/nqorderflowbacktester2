"""
Monte Carlo: multi-account fullport prop firm challenge.

Rules:
- Buy N challenge accounts ($50k each)
- Daily loss limit: $1,000 per account
- Max trailing DD: $2,000 per account
- Profit target: $3,000
- Risk per trade: ~$1,000 (dynamic MNQ sizing based on SL distance)
- Trade ONE account per trade; rotate on daily-lock
- If daily loss hit → lock that account for the day, switch to next
- If max DD hit → account dead
- If profit target hit → account passed, done
- Success = at least 1 account passes

Tests with and without ADX 15-30 filter.
"""

import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from adx_filter_common import (
    ET, DATA_DIR, build_adx_lookup, get_adx_at_time, passes_adx_filter,
    LUNCH_START, LUNCH_END,
)

VWAP_REACTION_CACHE_DIR = DATA_DIR / "vwap_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"

ENTRY_CUTOFF = "16:00"
FORCE_CLOSE  = "16:58"

MNQ_PV = 2.0

ACCOUNT_SIZE    = 50_000
DAILY_LOSS_LIM  = 1_000
TRAIL_DD        = 2_000
PROFIT_TARGET   = 3_000
TARGET_RISK     = 1_000  # dollars risked per trade

N_SIMS = 10_000
MAX_DAYS = 500
SEED = 42


def load_days_with_adx(adx_lookup):
    """Load all days, pre-tag signals with ADX pass/fail."""
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


def collect_trades_with_atr(days, sl_mult, tp_mult, use_adx):
    """
    Run backtest, return per-day list of trades.
    Each trade = (pnl_points, mae_points, sl_points) where sl_points = atr * sl_mult.
    Returns: daily_trades dict {day_idx: [(pnl, mae, sl_pts), ...]}, n_total_days
    """
    all_dates = set()
    daily_trades = {}

    for date_str, signals, bars in days:
        all_dates.add(date_str)
        day_trades = []
        last_exit_time = None

        for signal in signals:
            if use_adx and not signal["_adx_pass"]:
                continue
            ep = signal["entry_price"]; d = signal["direction"]; atr = signal["atr"]
            cbi = signal["bar_index"] + 1
            if atr is None or atr <= 0:
                continue
            cet = signal["_confirm_et"]
            ct = signal["confirm_time"]
            if "16:00" <= cet.strftime("%H:%M") < "19:10":
                continue
            if LUNCH_START <= cet.strftime("%H:%M") < LUNCH_END:
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
            worst_adverse = 0.0
            for j in range(cbi + 1, len(bars)):
                bar = bars[j]
                if not bar.closed: continue
                if d == "long":
                    adverse = ep - bar.low
                else:
                    adverse = bar.high - ep
                if adverse > worst_adverse:
                    worst_adverse = adverse

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
            day_trades.append((pnl, worst_adverse, sl_pts))
            last_exit_time = ext

        if day_trades:
            daily_trades[date_str] = day_trades

    return daily_trades, len(all_dates)


def sim_multi_account(trade_pool, day_count_dist, n_accounts, rng):
    """
    trade_pool: np.array of (pnl_pts, mae_pts, sl_pts) per trade
    day_count_dist: np.array of trades-per-day counts
    n_accounts: number of challenge accounts purchased

    Returns: (pass_at_least_one_rate, avg_days_to_pass, all_blown_rate,
              avg_accounts_used, pass_count_dist)
    """
    pnl_arr = trade_pool[:, 0]
    mae_arr = trade_pool[:, 1]
    sl_arr  = trade_pool[:, 2]

    passed_any = 0
    all_blown = 0
    pass_days_list = []
    accounts_used_list = []
    pass_counts = np.zeros(n_accounts + 1, dtype=int)  # how many accounts passed per sim

    for sim in range(N_SIMS):
        # Account state
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
            # Reset daily state
            daily_loss[:] = 0
            daily_locked[:] = False

            n_trades_today = rng.choice(day_count_dist)
            if n_trades_today == 0:
                continue

            trade_idxs = rng.integers(0, len(trade_pool), size=n_trades_today)

            for ti in trade_idxs:
                pnl_pts = pnl_arr[ti]
                mae_pts = mae_arr[ti]
                sl_pts  = sl_arr[ti]

                # Pick first available account (alive + not daily-locked)
                acct = -1
                for a in range(n_accounts):
                    if alive[a] and not daily_locked[a]:
                        acct = a
                        break
                if acct == -1:
                    break  # all accounts locked or dead for today

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
                    # Would breach daily limit on unrealized — lock account
                    daily_locked[acct] = True
                    # Still apply the trade (it happened, just triggered the lock)
                    equities[acct] += trade_pnl_dollars
                    daily_loss[acct] += max(0, -trade_pnl_dollars)
                    if equities[acct] <= dd_floors[acct]:
                        alive[acct] = False
                    elif equities[acct] > hwms[acct]:
                        hwms[acct] = equities[acct]
                        dd_floors[acct] = hwms[acct] - TRAIL_DD
                    if equities[acct] - ACCOUNT_SIZE >= PROFIT_TARGET and alive[acct]:
                        n_passed += 1
                        alive[acct] = False  # done with this account
                    continue

                # Apply trade
                equities[acct] += trade_pnl_dollars
                if trade_pnl_dollars < 0:
                    daily_loss[acct] += abs(trade_pnl_dollars)

                # Check DD blowup after trade
                if equities[acct] <= dd_floors[acct]:
                    alive[acct] = False
                    continue

                # Update HWM
                if equities[acct] > hwms[acct]:
                    hwms[acct] = equities[acct]
                    dd_floors[acct] = hwms[acct] - TRAIL_DD

                # Check daily loss lock
                if daily_loss[acct] >= DAILY_LOSS_LIM:
                    daily_locked[acct] = True

                # Check pass
                if equities[acct] - ACCOUNT_SIZE >= PROFIT_TARGET:
                    n_passed += 1
                    alive[acct] = False  # done, passed

            # Check if all accounts dead
            if not any(alive):
                break

            # If at least one passed, record and stop
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
    # Include zero-trade days
    return trade_pool, day_count_dist


if __name__ == "__main__":
    print("Building ADX lookup...")
    adx_lookup = build_adx_lookup()
    print(f"  {len(adx_lookup)} rows")

    print("Loading data...")
    days = load_days_with_adx(adx_lookup)
    print(f"  {len(days)} trading days loaded")

    # Configs to test
    configs = [
        (0.50, 1.90, "SL 0.50 / TP 1.90"),
        (0.50, 1.60, "SL 0.50 / TP 1.60"),
        (0.50, 2.10, "SL 0.50 / TP 2.10"),
        (0.90, 2.60, "SL 0.90 / TP 2.60"),
        (1.00, 2.00, "SL 1.00 / TP 2.00"),
    ]

    # Account counts to test
    account_counts = [1, 2, 3, 4, 5, 6, 8, 10]

    for use_adx in [True, False]:
        label = "ADX 15-30" if use_adx else "No filter"
        print(f"\n{'='*120}")
        print(f"  {label}")
        print(f"{'='*120}")
        print(f"  Fullport: risk ${TARGET_RISK}/trade | Daily loss limit: ${DAILY_LOSS_LIM}")
        print(f"  Account: ${ACCOUNT_SIZE:,} | Trail DD: ${TRAIL_DD:,} | Target: ${PROFIT_TARGET:,}")
        print(f"  {N_SIMS:,} sims per config per account count")

        for sl_mult, tp_mult, config_name in configs:
            daily_trades, n_days = collect_trades_with_atr(days, sl_mult, tp_mult, use_adx)
            trade_pool, day_count_dist = build_trade_pool(daily_trades)

            if len(trade_pool) == 0:
                print(f"\n  {config_name}: No trades")
                continue

            n_trades = len(trade_pool)
            wins = np.sum(trade_pool[:, 0] > 0)
            wr = wins / n_trades * 100

            # Show sizing stats
            sl_dollars = trade_pool[:, 2] * MNQ_PV
            contracts = np.maximum(1, np.floor(TARGET_RISK / sl_dollars)).astype(int)
            actual_risk = trade_pool[:, 2] * MNQ_PV * contracts
            actual_pnl = trade_pool[:, 0] * MNQ_PV * contracts

            print(f"\n  --- {config_name} ({label}) ---")
            print(f"  Trades: {n_trades} | WR: {wr:.1f}%")
            print(f"  Contracts per trade: min={contracts.min()} median={int(np.median(contracts))} max={contracts.max()}")
            print(f"  Risk per trade: ${actual_risk.mean():.0f} avg (${actual_risk.min():.0f}-${actual_risk.max():.0f})")
            print(f"  Win $: avg ${actual_pnl[actual_pnl > 0].mean():.0f} | Loss $: avg ${abs(actual_pnl[actual_pnl < 0].mean()):.0f}")

            print(f"\n  {'Accounts':>8s}  {'Pass>=1':>8s}  {'Avg Days':>9s}  {'All Blown':>10s}  {'Avg Accts Used':>15s}")
            print(f"  {'-'*60}")

            for n_accts in account_counts:
                rng = np.random.default_rng(SEED)
                result = sim_multi_account(trade_pool, day_count_dist, n_accts, rng)
                print(f"  {n_accts:>8d}  {result['pass_rate']:>7.1f}%  {result['avg_days']:>8.1f}d  "
                      f"{result['all_blown_rate']:>9.1f}%  {result['avg_accounts_used']:>14.1f}")

            sys.stdout.flush()

    print("\nDone.")
