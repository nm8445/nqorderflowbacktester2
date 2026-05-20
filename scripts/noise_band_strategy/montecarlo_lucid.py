"""
Monte Carlo — Lucid Funded account simulation for noise-band strategy.

Uses the buffer k=0.5 backtest trade pool, converted to MNQ ($2/pt).
Sweeps 1-10 MNQ contracts to find optimal sizing.

Lucid Rules:
  - $50k account, $2k trailing DD (never locks — trails forever as HWM - $2k)
  - Profit target: $3k (challenge pass)
  - Consistency: best day <= 50% of total profit at pass time
  - After passing: funded account, same DD rules
  - Payout: withdraw profit above $52k buffer, min $1k, max $2k per withdrawal
  - Daily loss limit: none (only trailing DD)

Trade pool: noise-band buffer k=0.5, 1 NQ contract backtest → extract point moves
  → rescale to MNQ ($2/pt) × N contracts with MNQ fee structure.
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict

ONEMN_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
SIGMA_CACHE_DIR = Path("D:/trading_pythonbacktest_data/noise_band_sigma_cache")
ET = "America/New_York"

# Backtest uses NQ to build the point-move pool
NQ_POINT_VALUE = 20.0
NQ_TICK_SIZE = 0.25
NQ_COMMISSION_PER_SIDE = 0.85
NQ_EXCHANGE_FEE_PER_SIDE = 1.40
SLIPPAGE_TICKS_PER_SIDE = 0.25
BUFFER_K = 0.5
NQ_RT_COST = (NQ_COMMISSION_PER_SIDE + NQ_EXCHANGE_FEE_PER_SIDE) * 2  # $4.50

# MNQ fee structure
MNQ_POINT_VALUE = 2.0
MNQ_COMMISSION_PER_SIDE = 0.62
MNQ_EXCHANGE_FEE_PER_SIDE = 0.47
MNQ_RT_COST = (MNQ_COMMISSION_PER_SIDE + MNQ_EXCHANGE_FEE_PER_SIDE) * 2  # $2.18

# Lucid rules
ACCOUNT_SIZE = 50_000.0
TRAILING_DD = 2_000.0
PROFIT_TARGET = 3_000.0
CONSISTENCY_RATIO = 0.50
PAYOUT_BUFFER = 2_000.0
MIN_PAYOUT = 1_000.0
MAX_PAYOUT = 2_000.0

N_SIMS = 10_000
MAX_DAYS = 500
SEED = 42


def load_data():
    df = pd.read_parquet(ONEMN_PARQUET)
    new_idx = []
    for t in df.index:
        if hasattr(t, "tz_convert") and t.tzinfo:
            new_idx.append(t.tz_convert(ET))
        else:
            new_idx.append(pd.Timestamp(t).tz_localize("UTC").tz_convert(ET))
    df.index = pd.DatetimeIndex(new_idx)
    return df


def compute_realized_vol(daily_df, today_idx, window=14):
    if today_idx < window + 1:
        return 0.15
    start = max(0, today_idx - window)
    subset = daily_df.iloc[start:today_idx]
    log_rets = np.log(subset["close"] / subset["close"].shift(1)).dropna()
    return log_rets.std() * np.sqrt(252) if len(log_rets) >= 5 else 0.15


def build_trade_pool():
    """
    Run buffer k=0.5 backtest with 1 NQ contract, extract point moves per trade.
    Returns daily_point_moves: list of lists, each inner list = point moves for one day.
    Point moves include slippage but NOT commissions/exchange fees — those get applied
    per-instrument when scaling to MNQ.
    """
    print("Loading 1-min bars...", flush=True)
    df = load_data()
    hm = df.index.strftime("%H:%M")
    session_df = df[(hm >= "09:30") & (hm <= "16:45")].copy()

    trading_dates = sorted(set(session_df.index.date))
    bars_by_date = {}
    for d in trading_dates:
        bars_by_date[d] = session_df[session_df.index.date == d]

    print("Loading sigma cache...", flush=True)
    with open(SIGMA_CACHE_DIR / "sigma_lookback_90.pkl", "rb") as f:
        sigma_cache = pickle.load(f)

    entry_times = set()
    h, m = 10, 0
    while (h, m) <= (15, 30):
        entry_times.add(f"{h:02d}:{m:02d}")
        m += 30
        if m >= 60:
            h += m // 60
            m = m % 60

    # Collect point moves grouped by day
    daily_point_moves = []

    for day_num, today in enumerate(trading_dates):
        if day_num < 90:
            continue
        today_bars = bars_by_date.get(today)
        if today_bars is None or len(today_bars) < 30:
            continue

        today_open = today_bars.iloc[0]["open"]
        prev_day_idx = day_num - 1
        while prev_day_idx >= 0 and trading_dates[prev_day_idx] not in bars_by_date:
            prev_day_idx -= 1
        if prev_day_idx < 0:
            continue
        yesterday_close = bars_by_date[trading_dates[prev_day_idx]].iloc[-1]["close"]

        sigma = sigma_cache.get(today)
        if not sigma:
            continue

        upper_anchor = max(today_open, yesterday_close)
        lower_anchor = min(today_open, yesterday_close)
        last_sigma = max(sigma.values())

        in_pos = False
        direction = ""
        entry_price = 0.0
        trailing_stop = 0.0
        bars_held = 0
        vwap_pv = 0.0
        vwap_v = 0.0
        day_moves = []  # point moves (signed, after slippage, before costs)

        for bar_i in range(len(today_bars)):
            bar = today_bars.iloc[bar_i]
            hm_str = today_bars.index[bar_i].strftime("%H:%M")

            tp = (bar["high"] + bar["low"] + bar["close"]) / 3.0
            bv = bar["volume"] if bar["volume"] > 0 else 1
            vwap_pv += tp * bv
            vwap_v += bv
            vwap = vwap_pv / vwap_v if vwap_v > 0 else bar["close"]

            sig = sigma.get(hm_str)
            if sig is None or sig <= 0:
                sig = last_sigma
            upper_band = upper_anchor * (1.0 + sig)
            lower_band = lower_anchor * (1.0 - sig)

            if in_pos:
                bars_held += 1
                if hm_str >= "16:45":
                    ep_out = bar["close"]
                    if direction == "long":
                        ep_out -= SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                        day_moves.append(ep_out - entry_price)
                    else:
                        ep_out += SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                        day_moves.append(entry_price - ep_out)
                    in_pos = False
                    continue

                if direction == "long":
                    band_sl = upper_band - BUFFER_K * sig * upper_anchor
                    new_stop = max(band_sl, vwap)
                    trailing_stop = max(trailing_stop, new_stop)
                    if bar["low"] <= trailing_stop:
                        ep_out = bar["open"] if bar["open"] <= trailing_stop else trailing_stop
                        ep_out -= SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                        day_moves.append(ep_out - entry_price)
                        in_pos = False
                        continue
                else:
                    band_sl = lower_band + BUFFER_K * sig * lower_anchor
                    new_stop = min(band_sl, vwap)
                    trailing_stop = min(trailing_stop, new_stop)
                    if bar["high"] >= trailing_stop:
                        ep_out = bar["open"] if bar["open"] >= trailing_stop else trailing_stop
                        ep_out += SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                        day_moves.append(entry_price - ep_out)
                        in_pos = False
                        continue

            if not in_pos and hm_str in entry_times:
                price = bar["close"]
                if price > upper_band and bar_i + 1 < len(today_bars):
                    nb = today_bars.iloc[bar_i + 1]
                    entry_price = nb["open"] + SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                    in_pos = True
                    direction = "long"
                    bars_held = 0
                    band_sl = upper_band - BUFFER_K * sig * upper_anchor
                    trailing_stop = max(band_sl, vwap)
                elif price < lower_band and bar_i + 1 < len(today_bars):
                    nb = today_bars.iloc[bar_i + 1]
                    entry_price = nb["open"] - SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                    in_pos = True
                    direction = "short"
                    bars_held = 0
                    band_sl = lower_band + BUFFER_K * sig * lower_anchor
                    trailing_stop = min(band_sl, vwap)

        if in_pos:
            last_bar = today_bars.iloc[-1]
            ep_out = last_bar["close"]
            if direction == "long":
                ep_out -= SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                day_moves.append(ep_out - entry_price)
            else:
                ep_out += SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                day_moves.append(entry_price - ep_out)

        daily_point_moves.append(day_moves)

    return daily_point_moves


def scale_day_to_mnq(day_point_moves, n_contracts):
    """Convert point moves to MNQ dollar PnL for n_contracts."""
    pnls = []
    for pt_move in day_point_moves:
        gross = pt_move * MNQ_POINT_VALUE * n_contracts
        cost = MNQ_RT_COST * n_contracts
        pnls.append(gross - cost)
    return pnls


def run_sim_challenge(daily_point_moves, n_contracts, rng):
    """
    Simulate challenge + funded phases with N MNQ contracts.
    """
    n_days_pool = len(daily_point_moves)

    results = []
    for sim in range(N_SIMS):
        day_order = rng.integers(0, n_days_pool, size=MAX_DAYS)

        equity = ACCOUNT_SIZE
        hwm = ACCOUNT_SIZE
        dd_floor = ACCOUNT_SIZE - TRAILING_DD
        best_day_pnl = 0.0
        total_profit = 0.0
        phase = "challenge"
        payouts = 0
        total_withdrawn = 0.0
        days_to_pass = 0
        days_to_first_payout = 0
        passed = False
        got_payout = False
        blown = False

        for di in range(MAX_DAYS):
            day_moves = daily_point_moves[day_order[di]]
            if not day_moves:
                if not passed:
                    days_to_pass += 1
                if not got_payout:
                    days_to_first_payout += 1
                continue

            day_pnl = 0.0
            for pt_move in day_moves:
                trade_pnl = pt_move * MNQ_POINT_VALUE * n_contracts - MNQ_RT_COST * n_contracts
                equity += trade_pnl
                day_pnl += trade_pnl

                if equity <= dd_floor:
                    blown = True
                    break

            if blown:
                break

            if equity > hwm:
                hwm = equity
                new_floor = hwm - TRAILING_DD
                dd_floor = max(dd_floor, new_floor)

            if equity <= dd_floor:
                blown = True
                break

            if day_pnl > best_day_pnl:
                best_day_pnl = day_pnl
            if day_pnl > 0:
                total_profit += day_pnl

            if not passed:
                days_to_pass += 1
            if not got_payout:
                days_to_first_payout += 1

            if phase == "challenge":
                if equity >= ACCOUNT_SIZE + PROFIT_TARGET:
                    if total_profit > 0 and best_day_pnl <= CONSISTENCY_RATIO * total_profit:
                        passed = True
                        phase = "funded"
                        best_day_pnl = 0.0

            if phase == "funded":
                profit_above_buffer = equity - (ACCOUNT_SIZE + PAYOUT_BUFFER)
                if profit_above_buffer >= MIN_PAYOUT:
                    withdraw = min(profit_above_buffer, MAX_PAYOUT)
                    payouts += 1
                    total_withdrawn += withdraw
                    if not got_payout:
                        got_payout = True
                    equity -= withdraw
                    hwm = equity
                    best_day_pnl = 0.0

        results.append({
            "passed": passed,
            "blown": blown,
            "payouts": payouts,
            "total_withdrawn": total_withdrawn,
            "days_to_pass": days_to_pass if passed else -1,
            "days_to_first_payout": days_to_first_payout if got_payout else -1,
            "got_payout": got_payout,
            "final_equity": equity,
        })

    return results


def run_sim_instant_funded(daily_point_moves, n_contracts, rng):
    """
    Simulate instant funded (skip challenge) with N MNQ contracts.
    """
    n_days_pool = len(daily_point_moves)
    results = []

    for sim in range(N_SIMS):
        day_order = rng.integers(0, n_days_pool, size=MAX_DAYS)

        equity = ACCOUNT_SIZE
        hwm = ACCOUNT_SIZE
        dd_floor = ACCOUNT_SIZE - TRAILING_DD
        best_day_pnl = 0.0
        payouts = 0
        total_withdrawn = 0.0
        days_to_first_payout = 0
        got_payout = False
        blown = False

        for di in range(MAX_DAYS):
            day_moves = daily_point_moves[day_order[di]]
            if not day_moves:
                if not got_payout:
                    days_to_first_payout += 1
                continue

            day_pnl = 0.0
            for pt_move in day_moves:
                trade_pnl = pt_move * MNQ_POINT_VALUE * n_contracts - MNQ_RT_COST * n_contracts
                equity += trade_pnl
                day_pnl += trade_pnl
                if equity <= dd_floor:
                    blown = True
                    break

            if blown:
                break

            if equity > hwm:
                hwm = equity
                new_floor = hwm - TRAILING_DD
                dd_floor = max(dd_floor, new_floor)

            if equity <= dd_floor:
                blown = True
                break

            if day_pnl > best_day_pnl:
                best_day_pnl = day_pnl

            if not got_payout:
                days_to_first_payout += 1

            profit_above_buffer = equity - (ACCOUNT_SIZE + PAYOUT_BUFFER)
            if profit_above_buffer >= MIN_PAYOUT:
                withdraw = min(profit_above_buffer, MAX_PAYOUT)
                payouts += 1
                total_withdrawn += withdraw
                if not got_payout:
                    got_payout = True
                equity -= withdraw
                hwm = equity
                best_day_pnl = 0.0

        results.append({
            "blown": blown,
            "payouts": payouts,
            "total_withdrawn": total_withdrawn,
            "days_to_first_payout": days_to_first_payout if got_payout else -1,
            "got_payout": got_payout,
            "final_equity": equity,
        })

    return results


def summarize(results, show_pass=False):
    """Return summary dict from sim results."""
    n = len(results)
    blown = sum(1 for r in results if r["blown"])
    got_payout = [r for r in results if r["got_payout"]]
    all_withdrawn = [r["total_withdrawn"] for r in results]

    s = {
        "blown_pct": 100 * blown / n,
        "payout_pct": 100 * len(got_payout) / n,
        "avg_withdrawn": np.mean(all_withdrawn),
        "median_withdrawn": np.median(all_withdrawn),
        "avg_payouts": np.mean([r["payouts"] for r in results]),
    }

    if show_pass:
        passed = sum(1 for r in results if r["passed"])
        s["pass_pct"] = 100 * passed / n
        pass_days = [r["days_to_pass"] for r in results if r["passed"]]
        s["avg_days_pass"] = np.mean(pass_days) if pass_days else -1

    if got_payout:
        payout_days = [r["days_to_first_payout"] for r in got_payout]
        s["avg_days_1st_pay"] = np.mean(payout_days)
    else:
        s["avg_days_1st_pay"] = -1

    return s


def main():
    daily_point_moves = build_trade_pool()

    # Trade pool stats (show as 1 MNQ for reference)
    all_moves = []
    for dm in daily_point_moves:
        all_moves.extend(dm)
    all_moves = np.array(all_moves)

    # Show stats for 1 MNQ
    pnls_1mnq = all_moves * MNQ_POINT_VALUE - MNQ_RT_COST
    n = len(pnls_1mnq)
    w = pnls_1mnq[pnls_1mnq > 0]
    l = pnls_1mnq[pnls_1mnq < 0]
    trade_days = [dm for dm in daily_point_moves if dm]
    trades_per_day = np.array([len(dm) for dm in trade_days])

    print(f"\nTrade pool (1 MNQ, $2/pt): {n} trades across {len(daily_point_moves)} days")
    print(f"  WR: {100*len(w)/n:.1f}%  PF: {w.sum()/abs(l.sum()):.2f}")
    print(f"  Avg: ${pnls_1mnq.mean():+,.2f}  Avg win: ${w.mean():+,.2f}  Avg loss: ${l.mean():+,.2f}")
    print(f"  Days with trades: {len(trade_days)}/{len(daily_point_moves)}")
    print(f"  Trades/day: {trades_per_day.mean():.1f} avg, {np.median(trades_per_day):.0f} median")
    print(f"  MNQ costs: ${MNQ_COMMISSION_PER_SIDE}/side comm + ${MNQ_EXCHANGE_FEE_PER_SIDE}/side exch = ${MNQ_RT_COST:.2f}/RT")

    rng = np.random.default_rng(SEED)
    contract_range = range(1, 11)

    # ──────────────────────────────────────────────────
    # Challenge + Funded sweep
    # ──────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"LUCID CHALLENGE + FUNDED — MNQ sweep 1-10 contracts, buffer k=0.5")
    print(f"$50k account | $2k trailing DD (never locks) | $3k target | 50% consistency")
    print(f"{N_SIMS:,} sims × {MAX_DAYS} max days each")
    print(f"{'='*100}")

    challenge_summaries = {}
    for nc in contract_range:
        print(f"  {nc} MNQ...", end="", flush=True)
        res = run_sim_challenge(daily_point_moves, nc, rng)
        challenge_summaries[nc] = summarize(res, show_pass=True)
        print(f" pass={challenge_summaries[nc]['pass_pct']:.1f}%  "
              f"payout={challenge_summaries[nc]['payout_pct']:.1f}%  "
              f"blown={challenge_summaries[nc]['blown_pct']:.1f}%  "
              f"avg withdrawn=${challenge_summaries[nc]['avg_withdrawn']:,.0f}")

    print(f"\n{'MNQ':>4} | {'Pass%':>6} | {'Payout%':>7} | {'Blown%':>6} | "
          f"{'AvgDays':>7} | {'AvgPay#':>7} | {'Avg $Withdrawn':>14} | {'Med $Withdrawn':>14}")
    print("-" * 100)
    for nc in contract_range:
        s = challenge_summaries[nc]
        days_str = f"{s['avg_days_pass']:.0f}" if s['avg_days_pass'] > 0 else "—"
        print(f"{nc:>4} | {s['pass_pct']:>5.1f}% | {s['payout_pct']:>6.1f}% | {s['blown_pct']:>5.1f}% | "
              f"{days_str:>7} | {s['avg_payouts']:>7.1f} | ${s['avg_withdrawn']:>13,.0f} | ${s['median_withdrawn']:>13,.0f}")

    # ──────────────────────────────────────────────────
    # Instant Funded sweep
    # ──────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"LUCID INSTANT FUNDED — MNQ sweep 1-10 contracts, buffer k=0.5")
    print(f"$50k account | $2k trailing DD (never locks) | no challenge | straight to payouts")
    print(f"{N_SIMS:,} sims × {MAX_DAYS} max days each")
    print(f"{'='*100}")

    instant_summaries = {}
    for nc in contract_range:
        print(f"  {nc} MNQ...", end="", flush=True)
        res = run_sim_instant_funded(daily_point_moves, nc, rng)
        instant_summaries[nc] = summarize(res, show_pass=False)
        print(f" payout={instant_summaries[nc]['payout_pct']:.1f}%  "
              f"blown={instant_summaries[nc]['blown_pct']:.1f}%  "
              f"avg withdrawn=${instant_summaries[nc]['avg_withdrawn']:,.0f}")

    print(f"\n{'MNQ':>4} | {'Payout%':>7} | {'Blown%':>6} | {'Days1stPay':>10} | "
          f"{'AvgPay#':>7} | {'Avg $Withdrawn':>14} | {'Med $Withdrawn':>14}")
    print("-" * 90)
    for nc in contract_range:
        s = instant_summaries[nc]
        days_str = f"{s['avg_days_1st_pay']:.0f}" if s['avg_days_1st_pay'] > 0 else "—"
        print(f"{nc:>4} | {s['payout_pct']:>6.1f}% | {s['blown_pct']:>5.1f}% | "
              f"{days_str:>10} | {s['avg_payouts']:>7.1f} | ${s['avg_withdrawn']:>13,.0f} | ${s['median_withdrawn']:>13,.0f}")


if __name__ == "__main__":
    main()
