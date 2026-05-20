"""
Monte Carlo prop firm simulation with ADX 15-30 filter.
Tests top 50 positive-RR configs from robust params CSV.
MNQ 1-9 contracts. Focus on slippage resistance, speed, blow-up risk.
"""

import pickle
import sys
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

ACCOUNT_SIZE   = 50_000
TRAIL_DD       = 2_000
PROFIT_TARGET  = 3_000
FUNDED_BUFFER  = 2_000

N_SIMS = 5_000
MAX_DAYS_CHALLENGE = 500
MAX_DAYS_FUNDED = 1500
SEED = 42

MNQ_PV = 2.0


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

        # Pre-tag each signal with ADX pass/fail
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


def collect_trades_for_config(days, sl_mult, tp_mult):
    """Run backtest with ADX filter, return (trade_pnl, mae, day_counts)."""
    all_dates = set()
    daily_pnls = {}
    daily_maes = {}

    for date_str, signals, bars in days:
        all_dates.add(date_str)
        day_trades = []
        day_maes = []
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

            if d == "long":
                sl = ep - atr * sl_mult; tp = ep + atr * tp_mult
                if tp <= ep or sl >= ep: continue
            else:
                sl = ep + atr * sl_mult; tp = ep - atr * tp_mult
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
            day_trades.append(pnl)
            day_maes.append(worst_adverse)
            last_exit_time = ext

        if day_trades:
            daily_pnls[date_str] = day_trades
            daily_maes[date_str] = day_maes

    all_pnl = []
    all_mae = []
    day_counts = []
    for d in sorted(all_dates):
        if d in daily_pnls:
            day_counts.append(len(daily_pnls[d]))
            all_pnl.extend(daily_pnls[d])
            all_mae.extend(daily_maes[d])
        else:
            day_counts.append(0)

    return (np.array(all_pnl, dtype=np.float64),
            np.array(all_mae, dtype=np.float64),
            np.array(day_counts, dtype=np.int32))


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
            blowups / N_SIMS * 100)


def get_top_configs(n=50):
    """Get top N positive-RR configs sorted by R:R descending."""
    csv_path = Path("C:/trading/nqorderflowbacktester/results/csv/monte_carlo_robust_params_results.csv")
    df = pd.read_csv(csv_path)
    combos = df.drop_duplicates(subset=['stop_mult', 'tp_mult'])[['stop_mult', 'tp_mult']].copy()
    combos['rr'] = combos['tp_mult'] / combos['stop_mult']
    pos_rr = combos[combos['rr'] > 1.0].sort_values('rr', ascending=False).head(n)
    return list(zip(pos_rr['stop_mult'], pos_rr['tp_mult']))


def main():
    print("Building ADX lookup...", flush=True)
    adx_lookup = build_adx_lookup()
    print(f"  {len(adx_lookup)} rows", flush=True)

    print("Loading data with ADX tags...", flush=True)
    days = load_days_with_adx(adx_lookup)
    print(f"  {len(days)} trading days loaded\n", flush=True)

    configs = get_top_configs(50)
    print(f"Testing {len(configs)} configs with positive R:R\n", flush=True)

    rng = np.random.default_rng(SEED)

    # MNQ 1-9 contracts
    contract_range = list(range(1, 10))

    # ═══════════════════════════════════════════════════════════════════
    # PART 1: Backtest stats for all 50 configs with ADX filter
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 110)
    print("PART 1: ADX 15-30 FILTERED BACKTEST STATS (Top 50 positive R:R configs)")
    print("=" * 110)
    print(f"{'SL':>5s} {'TP':>5s} {'R:R':>5s} {'Trades':>6s} {'WR%':>6s} {'PF':>6s} {'Exp$':>8s} {'Total$':>10s} {'DD$':>10s} {'Ret/DD':>7s}")
    print("-" * 110)

    config_data = {}
    results_rows = []

    for sl, tp in configs:
        trade_pool, mae_pool, day_counts = collect_trades_for_config(days, sl, tp)
        n = len(trade_pool)
        if n < 10:
            continue

        w = trade_pool[trade_pool > 0]; l = trade_pool[trade_pool < 0]
        wr = len(w) / n * 100
        gp = w.sum() * POINT_VALUE if len(w) else 0
        gl = abs(l.sum()) * POINT_VALUE if len(l) else 1
        pf = gp / gl if gl > 0 else 999
        exp_d = trade_pool.mean() * POINT_VALUE
        total_d = trade_pool.sum() * POINT_VALUE
        cum = np.cumsum(trade_pool) * POINT_VALUE
        dd = float((cum - np.maximum.accumulate(cum)).min())
        ret_dd = abs(total_d / dd) if dd != 0 else 0

        print(f"{sl:5.2f} {tp:5.2f} {tp/sl:5.2f} {n:6d} {wr:5.1f}% {pf:6.2f} {exp_d:+8.0f} {total_d:+10,.0f} {dd:+10,.0f} {ret_dd:7.2f}")

        config_data[(sl, tp)] = (trade_pool, mae_pool, day_counts)
        results_rows.append({
            'sl': sl, 'tp': tp, 'rr': tp/sl, 'trades': n, 'wr': wr, 'pf': pf,
            'exp_nq': exp_d, 'total_nq': total_d, 'dd_nq': dd, 'ret_dd': ret_dd,
        })

    sys.stdout.flush()

    # ═══════════════════════════════════════════════════════════════════
    # PART 2: Challenge pass rate — MNQ 1-9 for each config
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 140}")
    print(f"PART 2: CHALLENGE PASS RATE — MNQ 1-9 contracts")
    print(f"Account: ${ACCOUNT_SIZE:,} | Trail DD: ${TRAIL_DD:,} | Target: ${PROFIT_TARGET:,} | {N_SIMS} sims")
    print(f"{'=' * 140}")

    header = f"{'SL':>5s} {'TP':>5s} {'R:R':>5s} {'#':>4s} |"
    for c in contract_range:
        header += f" {c}con"
    header += " |"
    for c in contract_range:
        header += f" {c}d"
    header += " |"
    for c in contract_range:
        header += f" {c}b%"
    print(header)
    print("-" * len(header))
    sys.stdout.flush()

    challenge_data = []

    for sl, tp in configs:
        if (sl, tp) not in config_data:
            continue
        trade_pool, mae_pool, day_counts = config_data[(sl, tp)]
        n = len(trade_pool)
        rr = tp / sl

        line = f"{sl:5.2f} {tp:5.2f} {rr:5.2f} {n:4d} |"
        rates = []; avg_days_list = []; blowup_list = []

        for c in contract_range:
            rate, avg_days, blowup = sim_challenge(trade_pool, mae_pool, day_counts, MNQ_PV, c, rng)
            rates.append(rate)
            avg_days_list.append(avg_days)
            blowup_list.append(blowup)

        for r in rates:
            line += f" {r:3.0f}%"
        line += " |"
        for d in avg_days_list:
            line += f" {d:3.0f}"
        line += " |"
        for b in blowup_list:
            line += f" {b:3.0f}"

        print(line, flush=True)

        challenge_data.append({
            'sl': sl, 'tp': tp, 'trades': n,
            **{f'pass_{c}': rates[i] for i, c in enumerate(contract_range)},
            **{f'days_{c}': avg_days_list[i] for i, c in enumerate(contract_range)},
            **{f'blow_{c}': blowup_list[i] for i, c in enumerate(contract_range)},
        })

    # ═══════════════════════════════════════════════════════════════════
    # PART 3: Funded payouts — MNQ 1-9 for each config
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 140}")
    print(f"PART 3: FUNDED PAYOUTS — MNQ 1-9 contracts")
    print(f"DD locks at ${ACCOUNT_SIZE:,} | Buffer: ${FUNDED_BUFFER:,} | Payout: ${PROFIT_TARGET:,}")
    print(f"{'=' * 140}")

    header2 = f"{'SL':>5s} {'TP':>5s} {'R:R':>5s} |"
    for c in contract_range:
        header2 += f" {c}pay"
    header2 += " |"
    for c in contract_range:
        header2 += f" {c}d/p"
    header2 += " |"
    for c in contract_range:
        header2 += f" {c}b%"
    header2 += " |"
    for c in contract_range:
        header2 += f"  {c}net"
    print(header2)
    print("-" * len(header2))
    sys.stdout.flush()

    funded_data = []

    for sl, tp in configs:
        if (sl, tp) not in config_data:
            continue
        trade_pool, mae_pool, day_counts = config_data[(sl, tp)]
        rr = tp / sl

        line = f"{sl:5.2f} {tp:5.2f} {rr:5.2f} |"
        payouts_list = []; dpp_list = []; blowup_list = []

        for c in contract_range:
            avg_pay, avg_dpp, blowup = sim_funded(trade_pool, mae_pool, day_counts, MNQ_PV, c, rng)
            payouts_list.append(avg_pay)
            dpp_list.append(avg_dpp)
            blowup_list.append(blowup)

        for p in payouts_list:
            line += f" {p:3.1f}"
        line += " |"
        for d in dpp_list:
            line += f" {d:4.0f}"
        line += " |"
        for b in blowup_list:
            line += f" {b:3.0f}"
        line += " |"
        for p in payouts_list:
            net = p * PROFIT_TARGET - TRAIL_DD
            line += f" {net:+5.0f}"

        print(line, flush=True)

        funded_data.append({
            'sl': sl, 'tp': tp,
            **{f'payouts_{c}': payouts_list[i] for i, c in enumerate(contract_range)},
            **{f'dpp_{c}': dpp_list[i] for i, c in enumerate(contract_range)},
            **{f'fblowup_{c}': blowup_list[i] for i, c in enumerate(contract_range)},
        })

    # ═══════════════════════════════════════════════════════════════════
    # PART 4: Slippage sensitivity — add 0.5pt, 1pt, 1.5pt slippage
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 140}")
    print(f"PART 4: SLIPPAGE RESISTANCE — Challenge pass rate with 0.5/1.0/1.5 pt slippage")
    print(f"Testing best configs across MNQ contract sizes")
    print(f"{'=' * 140}")

    # Pick configs that had > 50% pass rate at any contract size
    good_configs = []
    for cd in challenge_data:
        max_pass = max(cd[f'pass_{c}'] for c in contract_range)
        if max_pass >= 50:
            good_configs.append((cd['sl'], cd['tp']))

    if not good_configs:
        # Fallback to top 15 by pass rate at 3 contracts
        sorted_cd = sorted(challenge_data, key=lambda x: x.get('pass_3', 0), reverse=True)
        good_configs = [(x['sl'], x['tp']) for x in sorted_cd[:15]]

    slippages = [0.5, 1.0, 1.5]
    test_contracts = [2, 3, 4, 5, 6]

    for slip in slippages:
        print(f"\n  --- Slippage: {slip} pt/trade ---")
        header_s = f"  {'SL':>5s} {'TP':>5s} |"
        for c in test_contracts:
            header_s += f" {c}con"
        print(header_s)

        for sl, tp in good_configs:
            if (sl, tp) not in config_data:
                continue
            trade_pool, mae_pool, day_counts = config_data[(sl, tp)]
            # Apply slippage: subtract from each trade
            slipped_pool = trade_pool - slip

            line = f"  {sl:5.2f} {tp:5.2f} |"
            for c in test_contracts:
                rate, _, _ = sim_challenge(slipped_pool, mae_pool, day_counts, MNQ_PV, c, rng)
                line += f" {rate:3.0f}%"
            print(line, flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # PART 5: Summary — best configs by criteria
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 140}")
    print(f"PART 5: BEST CONFIGS SUMMARY")
    print(f"{'=' * 140}")

    # Merge challenge + funded data
    for c in contract_range:
        print(f"\n  --- MNQ x{c} ---")
        # Sort by pass rate
        sorted_ch = sorted(challenge_data, key=lambda x: x.get(f'pass_{c}', 0), reverse=True)
        best = sorted_ch[0] if sorted_ch else None
        if best:
            fd = next((f for f in funded_data if f['sl'] == best['sl'] and f['tp'] == best['tp']), None)
            pay = fd[f'payouts_{c}'] if fd else 0
            dpp = fd[f'dpp_{c}'] if fd else 0
            net = pay * PROFIT_TARGET - TRAIL_DD
            print(f"    Best pass rate: SL={best['sl']:.2f} TP={best['tp']:.2f} | "
                  f"Pass {best[f'pass_{c}']:.0f}% | Avg {best[f'days_{c}']:.0f}d | "
                  f"Blow {best[f'blow_{c}']:.0f}% | "
                  f"Funded: {pay:.1f} payouts, {dpp:.0f} d/p, net ${net:+,.0f}")

        # Sort by lowest blowup
        sorted_safe = sorted(challenge_data, key=lambda x: x.get(f'blow_{c}', 100))
        safest = sorted_safe[0] if sorted_safe else None
        if safest and safest != best:
            fd = next((f for f in funded_data if f['sl'] == safest['sl'] and f['tp'] == safest['tp']), None)
            pay = fd[f'payouts_{c}'] if fd else 0
            net = pay * PROFIT_TARGET - TRAIL_DD
            print(f"    Lowest blowup: SL={safest['sl']:.2f} TP={safest['tp']:.2f} | "
                  f"Pass {safest[f'pass_{c}']:.0f}% | Blow {safest[f'blow_{c}']:.0f}% | "
                  f"Net ${net:+,.0f}")

        # Sort by fastest pass
        has_pass = [x for x in challenge_data if x.get(f'pass_{c}', 0) > 30]
        if has_pass:
            sorted_fast = sorted(has_pass, key=lambda x: x.get(f'days_{c}', 999))
            fastest = sorted_fast[0]
            if fastest != best:
                print(f"    Fastest pass:   SL={fastest['sl']:.2f} TP={fastest['tp']:.2f} | "
                      f"Pass {fastest[f'pass_{c}']:.0f}% | Avg {fastest[f'days_{c}']:.0f}d | "
                      f"Blow {fastest[f'blow_{c}']:.0f}%")

    # Save CSV
    out_dir = Path("results/csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Challenge results
    ch_rows = []
    for cd in challenge_data:
        for c in contract_range:
            ch_rows.append({
                'stop_mult': cd['sl'], 'tp_mult': cd['tp'],
                'contract_type': 'MNQ', 'num_contracts': c,
                'challenge_pass_rate': cd[f'pass_{c}'],
                'challenge_avg_days': cd[f'days_{c}'],
                'challenge_blowup_rate': cd[f'blow_{c}'],
            })
    ch_df = pd.DataFrame(ch_rows)

    # Funded results
    fd_rows = []
    for fd in funded_data:
        for c in contract_range:
            fd_rows.append({
                'stop_mult': fd['sl'], 'tp_mult': fd['tp'],
                'contract_type': 'MNQ', 'num_contracts': c,
                'funded_avg_payouts': fd[f'payouts_{c}'],
                'funded_days_per_payout': fd[f'dpp_{c}'],
                'funded_blowup_rate': fd[f'fblowup_{c}'],
                'funded_net_lifetime': fd[f'payouts_{c}'] * PROFIT_TARGET - TRAIL_DD,
            })
    fd_df = pd.DataFrame(fd_rows)

    merged = ch_df.merge(fd_df, on=['stop_mult', 'tp_mult', 'contract_type', 'num_contracts'])
    merged.to_csv(out_dir / "monte_carlo_adx_mnq_results.csv", index=False)
    print(f"\nResults saved to {out_dir / 'monte_carlo_adx_mnq_results.csv'}")


if __name__ == "__main__":
    main()
