"""
Monte Carlo -- Instant Funding Account Simulation

$50k Account Rules:
  - Trailing DD: $2k (floor starts at $48k, trails up with HWM)
  - DD stops trailing once $2k profit reached (floor locks at $50k)
  - Payout 1 profit goal: $3,000
  - Payout 2+ profit goal: $2,000
  - Max payout: $2,000 (payouts 1-3), $2,500 (payout 4+)
  - 20% consistency: no single day's profit > 20% of total accumulated profit
  - If consistency not met at goal, keep trading until total profit dilutes the big day

Config: SL 0.50 / TP 2.10 | ADX 15-30
"""

import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from adx_filter_common import (
    ET, DATA_DIR, build_adx_lookup, get_adx_at_time, passes_adx_filter,
)

VWAP_REACTION_CACHE_DIR = DATA_DIR / "vwap_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"

# Account rules
ACCOUNT_SIZE = 50_000
TRAILING_DD = 2_000
DD_LOCK_PROFIT = 2_000
FIRST_PAYOUT_GOAL = 3_000
NEXT_PAYOUT_GOAL = 2_000
MAX_PAYOUT_1_3 = 2_000
MAX_PAYOUT_4 = 2_500
CONSISTENCY_PCT = 0.20
MAX_TRADING_DAYS = 1500

# Strategy config
SL_MULT = 0.50
TP_MULT = 2.10
ENTRY_CUTOFF = "16:00"
FORCE_CLOSE = "16:58"
MNQ_POINT_VALUE = 2.0
TARGET_RISK = 1000.0

N_SIMS = 5000


def build_daily_trades(days_data, mnq_count=None):
    daily_results = []
    for date_str, signals, bars in days_data:
        last_exit_time = None
        day_pnl = 0.0
        day_trades = []
        for signal in signals:
            if not signal["_adx_pass"]:
                continue
            ep = signal["entry_price"]
            d = signal["direction"]
            atr = signal["atr"]
            cbi = signal["bar_index"] + 1
            if atr is None or atr <= 0:
                continue
            cet = signal["_confirm_et"]
            ct = signal["confirm_time"]
            if "16:00" <= cet.strftime("%H:%M") < "19:10":
                continue
            if last_exit_time is not None and ct <= last_exit_time:
                continue
            sl = ep - atr * SL_MULT if d == "long" else ep + atr * SL_MULT
            tp = ep + atr * TP_MULT if d == "long" else ep - atr * TP_MULT
            if d == "long" and (tp <= ep or sl >= ep):
                continue
            if d == "short" and (tp >= ep or sl <= ep):
                continue
            if mnq_count is None:
                sl_pts = atr * SL_MULT
                contracts = max(1, int(TARGET_RISK / (sl_pts * MNQ_POINT_VALUE)))
            else:
                contracts = mnq_count
            exit_price = None
            exit_time = None
            for j in range(cbi + 1, len(bars)):
                bar = bars[j]
                if not bar.closed:
                    continue
                bct = bar.close_time
                if hasattr(bct, 'tz_convert'):
                    bet = bct.tz_convert(ET)
                else:
                    bet = pd.Timestamp(bct, tz='UTC').tz_convert(ET)
                if bet.strftime("%H:%M") >= FORCE_CLOSE:
                    exit_price = bar.close
                    exit_time = bct
                    break
                if d == "short":
                    if bar.high >= sl:
                        exit_price = sl
                        exit_time = bct
                        break
                    if bar.low <= tp:
                        exit_price = tp
                        exit_time = bct
                        break
                else:
                    if bar.low <= sl:
                        exit_price = sl
                        exit_time = bct
                        break
                    if bar.high >= tp:
                        exit_price = tp
                        exit_time = bct
                        break
            if exit_price is None:
                exit_price = bars[-1].close
                exit_time = bars[-1].close_time
            pnl_pts = (ep - exit_price) if d == "short" else (exit_price - ep)
            pnl_dollars = pnl_pts * MNQ_POINT_VALUE * contracts
            day_pnl += pnl_dollars
            last_exit_time = exit_time
            day_trades.append(pnl_dollars)
        if day_trades:
            daily_results.append({"date": date_str, "pnl": day_pnl, "trades": day_trades})
    return daily_results


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


def sim_instant_funding(daily_pool, n_sims):
    no_trade_pct = 1.0 - (len(daily_pool) / 260)

    results = {
        "got_payout": 0,         # got at least 1 payout (success!)
        "blown_no_payout": 0,    # blew up before any payout
        "blown_after_payout": 0, # blew up after collecting payouts (still profitable)
        "timed_out": 0,
        "days_to_first": [],
        "total_payouts": [],
        "total_earned": [],
        "days_alive": [],        # how long account survived
    }

    for _ in range(n_sims):
        equity = ACCOUNT_SIZE
        hwm = ACCOUNT_SIZE
        dd_floor = ACCOUNT_SIZE - TRAILING_DD
        dd_locked = False
        payout_count = 0
        total_earned = 0.0
        day_count = 0
        blown = False
        first_payout_day = None

        positive_day_pnls = []
        total_profit_for_consistency = 0.0
        profit_since_last_payout = 0.0

        while day_count < MAX_TRADING_DAYS:
            day_count += 1

            if random.random() < no_trade_pct:
                continue

            day = random.choice(daily_pool)
            day_pnl = day["pnl"]

            equity += day_pnl
            profit_since_last_payout += day_pnl
            if day_pnl > 0:
                total_profit_for_consistency += day_pnl
                positive_day_pnls.append(day_pnl)

            # Update trailing DD
            if equity > hwm:
                hwm = equity
                if not dd_locked:
                    dd_floor = hwm - TRAILING_DD
                    if equity >= ACCOUNT_SIZE + DD_LOCK_PROFIT:
                        dd_floor = ACCOUNT_SIZE
                        dd_locked = True

            # Check blowup
            if equity <= dd_floor:
                blown = True
                break

            # Check payout eligibility
            goal = FIRST_PAYOUT_GOAL if payout_count == 0 else NEXT_PAYOUT_GOAL
            if profit_since_last_payout >= goal:
                if positive_day_pnls:
                    max_day = max(positive_day_pnls)
                    consistent = max_day <= CONSISTENCY_PCT * total_profit_for_consistency
                else:
                    consistent = False

                if consistent:
                    if payout_count < 3:
                        max_payout = MAX_PAYOUT_1_3
                    else:
                        max_payout = MAX_PAYOUT_4

                    payout_amt = min(max_payout, profit_since_last_payout)
                    payout_amt = max(1000, payout_amt)

                    equity -= payout_amt
                    total_earned += payout_amt
                    payout_count += 1

                    if first_payout_day is None:
                        first_payout_day = day_count

                    profit_since_last_payout = equity - ACCOUNT_SIZE
                    hwm = equity
                    if not dd_locked:
                        dd_floor = hwm - TRAILING_DD

        # Categorize outcome
        if payout_count > 0:
            results["got_payout"] += 1
            results["days_to_first"].append(first_payout_day)
            results["total_payouts"].append(payout_count)
            results["total_earned"].append(total_earned)
            results["days_alive"].append(day_count)
            if blown:
                results["blown_after_payout"] += 1
        elif blown:
            results["blown_no_payout"] += 1
            results["days_alive"].append(day_count)
        else:
            results["timed_out"] += 1
            results["days_alive"].append(day_count)

    return results


if __name__ == "__main__":
    print("=" * 120)
    print("MONTE CARLO -- INSTANT FUNDING ACCOUNT ($50k)")
    print("=" * 120)
    print(f"Trailing DD: ${TRAILING_DD:,} (locks at +${DD_LOCK_PROFIT:,} profit, floor = ${ACCOUNT_SIZE:,})")
    print(f"Payout 1 goal: ${FIRST_PAYOUT_GOAL:,} | Payout 2+ goal: ${NEXT_PAYOUT_GOAL:,}")
    print(f"Max payout: ${MAX_PAYOUT_1_3:,} (1-3) / ${MAX_PAYOUT_4:,} (4+)")
    print(f"20% consistency: biggest winning day <= 20% of total accumulated profit")
    print(f"Config: SL {SL_MULT} / TP {TP_MULT} | ADX 15-30 | {N_SIMS:,} sims | {MAX_TRADING_DAYS} max days")
    print()

    print("Building ADX lookup...")
    adx_lookup = build_adx_lookup()
    days_data = load_days(adx_lookup)
    print(f"  {len(days_data)} trading days loaded")
    print()

    mnq_sizes = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 33]

    print("Building trade data per MNQ size...")
    size_data = {}
    for mnq in mnq_sizes:
        daily = build_daily_trades(days_data, mnq_count=mnq)
        pnls = [d["pnl"] for d in daily]
        all_t = []
        for d in daily:
            all_t.extend(d["trades"])
        wins = [t for t in all_t if t > 0]
        losses = [t for t in all_t if t < 0]
        size_data[mnq] = {"daily": daily, "avg_win": np.mean(wins) if wins else 0, "avg_loss": np.mean(losses) if losses else 0, "max_day": max(pnls), "min_day": min(pnls)}
        print(f"  {mnq:3d} MNQ: avg win ${np.mean(wins) if wins else 0:+,.0f}, avg loss ${np.mean(losses) if losses else 0:+,.0f}, "
              f"best day ${max(pnls):+,.0f}, worst day ${min(pnls):+,.0f}")
    daily_fp = build_daily_trades(days_data, mnq_count=None)
    fp_pnls = [d["pnl"] for d in daily_fp]
    print(f"   FP MNQ: best day ${max(fp_pnls):+,.0f}, worst day ${min(fp_pnls):+,.0f}")
    print()

    # Run simulations
    print("=" * 120)
    header = (f"{'Size':>6s} | {'Payout%':>7s} | {'No Pay':>7s} | {'Blown':>7s} | {'Timeout':>7s} | "
              f"{'Avg Days':>8s} | {'Med Days':>8s} | {'Avg #Pay':>8s} | {'Avg Earned':>10s} | "
              f"{'Avg Life':>8s} | {'Risk/Tr':>8s}")
    print(header)
    print("-" * 120)

    # Fullport
    res = sim_instant_funding(daily_fp, N_SIMS)
    t = N_SIMS
    avg_d = f"{np.mean(res['days_to_first']):.0f}" if res["days_to_first"] else "N/A"
    med_d = f"{np.median(res['days_to_first']):.0f}" if res["days_to_first"] else "N/A"
    avg_p = f"{np.mean(res['total_payouts']):.1f}" if res["total_payouts"] else "0"
    avg_e = f"${np.mean(res['total_earned']):,.0f}" if res["total_earned"] else "$0"
    avg_l = f"{np.mean(res['days_alive']):.0f}"
    print(f"{'FP':>6s} | {res['got_payout']/t*100:6.1f}% | {res['blown_no_payout']/t*100:6.1f}% | "
          f"{res['blown_after_payout']/t*100:6.1f}% | {res['timed_out']/t*100:6.1f}% | "
          f"{avg_d:>8s} | {med_d:>8s} | {avg_p:>8s} | {avg_e:>10s} | {avg_l:>8s} | {'~$1000':>8s}")

    for mnq in mnq_sizes:
        daily = size_data[mnq]["daily"]
        res = sim_instant_funding(daily, N_SIMS)
        sl_risk = 7.5 * MNQ_POINT_VALUE * mnq

        avg_d = f"{np.mean(res['days_to_first']):.0f}" if res["days_to_first"] else "N/A"
        med_d = f"{np.median(res['days_to_first']):.0f}" if res["days_to_first"] else "N/A"
        avg_p = f"{np.mean(res['total_payouts']):.1f}" if res["total_payouts"] else "0"
        avg_e = f"${np.mean(res['total_earned']):,.0f}" if res["total_earned"] else "$0"
        avg_l = f"{np.mean(res['days_alive']):.0f}"

        print(f"{mnq:5d}x | {res['got_payout']/t*100:6.1f}% | {res['blown_no_payout']/t*100:6.1f}% | "
              f"{res['blown_after_payout']/t*100:6.1f}% | {res['timed_out']/t*100:6.1f}% | "
              f"{avg_d:>8s} | {med_d:>8s} | {avg_p:>8s} | {avg_e:>10s} | {avg_l:>8s} | ${sl_risk:>6,.0f}")

    print()

    # Summary for best sizes
    print("=" * 120)
    print("DETAILED BREAKDOWN — Best Sizes")
    print("=" * 120)
    for mnq in [2, 3, 4, 5, 6, 8]:
        daily = size_data[mnq]["daily"]
        res = sim_instant_funding(daily, N_SIMS)
        t = N_SIMS
        got = res["got_payout"]
        bap = res["blown_after_payout"]
        bnp = res["blown_no_payout"]

        print(f"\n  {mnq} MNQ (risk ${7.5*MNQ_POINT_VALUE*mnq:,.0f}/trade, TP hit ~${7.5*TP_MULT/SL_MULT*MNQ_POINT_VALUE*mnq:,.0f}):")
        print(f"    Got at least 1 payout: {got/t*100:.1f}%")
        if got > 0:
            print(f"      Avg payouts collected: {np.mean(res['total_payouts']):.1f}")
            print(f"      Avg total earned: ${np.mean(res['total_earned']):,.0f}")
            print(f"      Avg days to first payout: {np.mean(res['days_to_first']):.0f}")
            print(f"      Of those, {bap} ({bap/got*100:.0f}%) eventually blew up after collecting payouts")
        print(f"    Blew up before any payout: {bnp/t*100:.1f}%")
        print(f"    Avg account lifetime: {np.mean(res['days_alive']):.0f} days")

    print()
    print("=" * 120)
    print("KEY INSIGHT:")
    print("  After each payout, buffer = equity - $50k floor. Payouts shrink the buffer.")
    print("  The account survives by collecting payouts BEFORE the inevitable blowup.")
    print("  Success metric = total $ earned, not survival. All accounts eventually blow up.")
    print("=" * 120)
