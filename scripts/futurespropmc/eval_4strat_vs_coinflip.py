"""MC: Eval pass rate — running 4-strat live signals vs coinflip gambler's ruin.

4-strat path:
  Bootstrap daily P&L from live combined trades (OD+RV+B2 at 1 NQ basis).
  Run day-by-day until cumulative P&L hits +$3000 (PASS) or -$2000 (BLOW).
  Record: pass rate, avg/median days, distribution.

Coinflip gambler's ruin:
  Bet $1500, 1:1 RR, target +$3000, blow at -$2000.
  Random direction (coinflip) gives WR=50% before fees; ~48% with slippage.

Outputs:
  - Pass rate comparison
  - Days-to-outcome distribution
  - Per-funded eval cost
  - Annual net at the user's 30-day cycle
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TRADES_CSV = "live/combined deployment plan/combined_trades.csv"
EVAL_TARGET = 3000
EVAL_DD = -2000
N_TRIALS = 50_000
RNG = np.random.default_rng(42)


def load_daily_pnl():
    df = pd.read_csv(TRADES_CSV)
    daily = df.groupby("date")["pnl_$"].sum().values
    return daily


def bootstrap_4strat(daily_pnl, n_trials=N_TRIALS, max_days=60):
    """Random-sample daily P&L with replacement until pass/blow/max."""
    results = []
    for _ in range(n_trials):
        cum = 0.0
        days = 0
        while days < max_days:
            cum += RNG.choice(daily_pnl)
            days += 1
            if cum >= EVAL_TARGET:
                results.append((True, days, cum)); break
            if cum <= EVAL_DD:
                results.append((False, days, cum)); break
        else:
            results.append((False, days, cum))  # max_days reached as "fail"
    return pd.DataFrame(results, columns=["pass", "days", "final_cum"])


def coinflip_gamblers_ruin(wr=0.50, n_trials=N_TRIALS, max_trades=20):
    """1:1 RR at $1500, target +$3000, blow at -$2000."""
    bet = 1500
    results = []
    for _ in range(n_trials):
        cum = 0.0
        trades = 0
        while trades < max_trades:
            cum += bet if RNG.random() < wr else -bet
            trades += 1
            if cum >= EVAL_TARGET:
                results.append((True, trades, cum)); break
            if cum <= EVAL_DD:
                results.append((False, trades, cum)); break
        else:
            results.append((False, trades, cum))
    return pd.DataFrame(results, columns=["pass", "trades", "final_cum"])


def summarize(df: pd.DataFrame, time_col: str, label: str) -> dict:
    pass_df = df[df["pass"]]
    blow_df = df[~df["pass"]]
    return dict(
        label=label,
        pass_rate=df["pass"].mean(),
        avg_time_to_pass=pass_df[time_col].mean() if len(pass_df) > 0 else np.nan,
        median_time_to_pass=pass_df[time_col].median() if len(pass_df) > 0 else np.nan,
        avg_time_to_blow=blow_df[time_col].mean() if len(blow_df) > 0 else np.nan,
    )


def main():
    print("=" * 90)
    print("MC: 4-STRAT (live OD+RV+B2 at 1 NQ) vs COINFLIP GAMBLER'S RUIN — eval pass")
    print("=" * 90)

    daily = load_daily_pnl()
    print(f"\nLoaded {len(daily)} live trading days from {TRADES_CSV}")
    print(f"  Daily P&L: mean=${daily.mean():,.0f}  std=${daily.std():,.0f}  "
          f"min=${daily.min():,.0f}  max=${daily.max():,.0f}")
    print(f"  Win days: {(daily > 0).sum()}/{len(daily)} = {(daily > 0).mean()*100:.1f}%")

    print("\n" + "-" * 90)
    print("4-STRAT BOOTSTRAP")
    print("-" * 90)
    df_4s = bootstrap_4strat(daily)
    s_4s = summarize(df_4s, "days", "4-strat (live OD+RV+B2)")
    print(f"  Pass rate:           {s_4s['pass_rate']*100:.1f}%")
    print(f"  Avg days to pass:    {s_4s['avg_time_to_pass']:.1f}")
    print(f"  Median days to pass: {s_4s['median_time_to_pass']:.0f}")
    print(f"  Avg days to blow:    {s_4s['avg_time_to_blow']:.1f}")

    # Days-to-pass histogram
    pass_df = df_4s[df_4s["pass"]]
    print(f"\n  Distribution of days-to-pass (only passing trials):")
    pct = pass_df["days"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    print(pct.to_string())

    print("\n" + "-" * 90)
    print("COINFLIP GAMBLER'S RUIN (WR=50%)")
    print("-" * 90)
    df_cf = coinflip_gamblers_ruin(wr=0.50)
    s_cf = summarize(df_cf, "trades", "Coinflip (WR=50%)")
    print(f"  Pass rate:             {s_cf['pass_rate']*100:.1f}%")
    print(f"  Avg trades to pass:    {s_cf['avg_time_to_pass']:.2f}")
    print(f"  Median trades to pass: {s_cf['median_time_to_pass']:.0f}")

    print("\n" + "-" * 90)
    print("COINFLIP GAMBLER'S RUIN (WR=48%, slippage drag)")
    print("-" * 90)
    df_cf48 = coinflip_gamblers_ruin(wr=0.48)
    s_cf48 = summarize(df_cf48, "trades", "Coinflip (WR=48%)")
    print(f"  Pass rate:             {s_cf48['pass_rate']*100:.1f}%")
    print(f"  Avg trades to pass:    {s_cf48['avg_time_to_pass']:.2f}")
    print(f"  Median trades to pass: {s_cf48['median_time_to_pass']:.0f}")

    print("\n" + "=" * 90)
    print("COMPARISON")
    print("=" * 90)
    print(f"{'Approach':<35} {'Pass rate':>10} {'Time to pass':>15} {'$/funded':>12}")
    print("-" * 80)
    eval_cost = 100
    for s in [s_4s, s_cf, s_cf48]:
        cost_per_funded = eval_cost / s["pass_rate"]
        tt = f"{s['avg_time_to_pass']:.1f} units" if not np.isnan(s['avg_time_to_pass']) else "n/a"
        print(f"{s['label']:<35} {s['pass_rate']*100:>9.1f}% {tt:>15} {f'${cost_per_funded:.0f}':>12}")

    # Annual cycles + net comparison
    print("\n" + "=" * 90)
    print("ANNUAL NET (10 funded accounts per cycle, $11.75K revenue, $100 eval cost)")
    print("=" * 90)

    # For 4-strat: cycle = 10 × avg_days_to_pass trading days (sequential per account)
    # OR: run in parallel across multiple eval slots → 1× avg_days_to_pass + overhead
    # User said "3 days max" implying parallel firing
    # 4-strat needs 7+ days even passing — can't be parallelized as fast
    # Assume PARALLEL: cycle time = avg_days_to_pass + 5 days (grind + payout)
    avg_days = s_4s['avg_time_to_pass'] if not np.isnan(s_4s['avg_time_to_pass']) else 30
    cycle_4s = avg_days + 5
    cycles_4s = 250 / cycle_4s

    # Coinflip: 3 days pass + 5 days grind = 8 day cycle
    cycle_cf = 3 + 5
    cycles_cf = 250 / cycle_cf

    # Eval count to get 10 funded = 10 / pass_rate
    for s, cycle_days, n_cycles in [
        (s_4s, cycle_4s, cycles_4s),
        (s_cf, cycle_cf, cycles_cf),
        (s_cf48, cycle_cf, cycles_cf),
    ]:
        evals_per_cycle = 10 / s["pass_rate"]
        eval_cost_cycle = evals_per_cycle * 100
        net_per_cycle = 11750 - eval_cost_cycle
        annual_net = net_per_cycle * n_cycles
        print(f"  {s['label']:<32} cycle={cycle_days:.0f}d  "
              f"evals/cycle={evals_per_cycle:.0f}  "
              f"net/cycle=${net_per_cycle:,.0f}  "
              f"cycles/yr={n_cycles:.1f}  "
              f"ANNUAL=${annual_net:,.0f}")


if __name__ == "__main__":
    main()
