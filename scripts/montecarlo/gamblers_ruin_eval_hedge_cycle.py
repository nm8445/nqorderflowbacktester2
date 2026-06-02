"""MC: Gambler's-ruin eval + 1:2 hedge + 5-day grind + payout cycle.

User's plan:
- EVAL: bet $1.5K at 1:1 RR until $3K profit. Ruin = 2 net losses (DD $2K).
  Cost: $100/eval avg. Goal: 10 funded accounts per cycle.
- HEDGE: 10 funded -> 5 pairs. Risk $2K to make $4K (1:2 RR). Per pair:
  always 1 blown, 1 survivor with +$2K or +$4K profit cushion.
- 5-DAY GRIND: 4-strat at 1 MNQ to bank 5 winning days. Small blow risk.
- PAYOUT 1: withdraw $2K/survivor (100% split on small payouts).
- MILK: continue $1K hedges + $1.5K withdrawals until all accounts blown.
- CYCLE: as soon as payout lands, reinvest into next 10 evals.

Outputs: per-cycle net $, sensitivity to WR/TP-rate/grind-blow, annualized $.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ---------- EVAL PHASE ----------
EVAL_COST          = 100        # avg cost per eval (incl. resets/discounts)
EVAL_BASE_WR       = 0.50       # 1:1 RR random NQ win rate
EVAL_TARGET        = 3000       # eval profit target
EVAL_BET           = 1500       # bet size per trade
EVAL_DD            = 2000       # max DD (blow on 2 net losses)
N_TARGET_FUNDED    = 10

# ---------- HEDGE PHASE ----------
HEDGE_TP_RATE      = 0.30       # P(surviving account hits +$4K TP)
HEDGE_TP_PROFIT    = 4000
HEDGE_SL_PROFIT    = 2000       # default outcome: SL on losing side, +$2K on winning

# ---------- GRIND PHASE ----------
GRIND_BLOW_PROB    = 0.05       # P(blow during 5-day grind at 1 MNQ)

# ---------- PAYOUT ----------
PAYOUT_AMOUNT      = 2000
PROFIT_SPLIT       = 1.00       # 100% on first small payout (Tradeify/TopStep/MFFU)

# ---------- MILK PHASE (second wave) ----------
MILK_HEDGE_BET     = 1000       # post-payout hedge bet
MILK_WITHDRAW      = 1500       # withdraw per available account per round
MILK_MAX_ROUNDS    = 4          # cap rounds before assuming all blow

# ---------- CYCLE TIMING ----------
DAYS_PER_CYCLE     = 30
TRADING_DAYS_YR    = 250


def eval_pass_prob(wr: float) -> float:
    """Closed-form pass prob for the 3-state gambler's ruin.

    States: -1500, 0, +1500. Pass at +3000, ruin at -3000.
    Recurrence:
      P(0)     = wr*P(+1500) + (1-wr)*P(-1500)
      P(+1500) = wr + (1-wr)*P(0)
      P(-1500) = wr*P(0)
    => P(0) = wr^2 / (1 - 2*wr*(1-wr))
    """
    return wr**2 / (1 - 2 * wr * (1 - wr))


def run_one_cycle(wr: float, hedge_tp_rate: float, grind_blow: float,
                  rng: np.random.Generator) -> dict:
    """One full cycle: evals -> 10 funded -> hedge -> grind -> payout -> milk."""
    p_pass = eval_pass_prob(wr)

    # === EVAL: buy evals until N_TARGET_FUNDED pass ===
    # Negative binomial: n_evals = sum of geometric(p_pass), N_TARGET_FUNDED times
    n_evals = int(rng.negative_binomial(N_TARGET_FUNDED, p_pass) + N_TARGET_FUNDED)
    eval_cost = n_evals * EVAL_COST

    # === HEDGE: 5 pairs of 2 -> 5 survivors ===
    n_pairs = N_TARGET_FUNDED // 2
    tp_hits = rng.binomial(n_pairs, hedge_tp_rate)
    # Each survivor's cushion: +$4K if TP, +$2K otherwise
    hedge_cushion_total = tp_hits * HEDGE_TP_PROFIT + (n_pairs - tp_hits) * HEDGE_SL_PROFIT
    n_survivors_after_hedge = n_pairs  # always 1 per pair

    # === GRIND: 5 winning days at 1 MNQ ===
    n_survivors_after_grind = rng.binomial(n_survivors_after_hedge, 1 - grind_blow)

    # === FIRST PAYOUT: $2K per survivor ===
    first_payout_revenue = n_survivors_after_grind * PAYOUT_AMOUNT * PROFIT_SPLIT

    # Track residual buffer per surviving account (after $2K payout)
    # If hedge TP'd (+$4K), residual = $4K - $2K = $2K
    # If hedge SL'd (+$2K), residual = $2K - $2K = $0
    # Distribute TP hits across survivors after grind
    if n_survivors_after_grind > 0:
        prob_tp = tp_hits / n_pairs if n_pairs > 0 else 0
        n_tp_survivors = rng.binomial(n_survivors_after_grind, prob_tp)
    else:
        n_tp_survivors = 0
    n_sl_survivors = n_survivors_after_grind - n_tp_survivors

    # === MILK PHASE ===
    # SL-survivors ($0 buffer): next $1K hedge - half blow immediately, half gain $1K
    # TP-survivors ($2K buffer): can survive $1K hedge loss, gain $1K profit possibility
    # Conservative model: each milk round halves the surviving pool and adds $1.5K withdrawal per survivor
    milk_revenue = 0
    # Pool starts at $2K-buffer survivors only (SL-buffer ones blow on first hedge ~50%)
    pool_2k = n_tp_survivors
    pool_0k = n_sl_survivors  # these mostly blow on first hedge
    # First milk round: $1K hedge across all
    # Hedge winners: pool_0k * 0.5 + pool_2k * 0.5 wins, gain $1K
    # Hedge losers: blow if buffer < $1K (all pool_0k losers) or stay alive at $1K buffer (pool_2k losers)
    n_winners = rng.binomial(pool_0k + pool_2k, 0.5)
    # Winners now have +$1K each (pool_0k_winners) or +$3K each (pool_2k_winners)
    # Withdraw $1.5K from anyone with >= $1.5K available:
    #   pool_0k_winners have $1K - can't withdraw
    #   pool_2k_winners have $3K - can withdraw $1.5K
    n_pool_2k_winners = rng.binomial(pool_2k, 0.5) if pool_2k > 0 else 0
    n_pool_0k_winners = n_winners - n_pool_2k_winners if n_winners >= n_pool_2k_winners else 0
    milk_revenue += n_pool_2k_winners * MILK_WITHDRAW
    # After withdrawal: pool_2k_winners have $1.5K buffer, pool_0k_winners have $1K buffer
    # Losers from pool_2k stay alive at $1K buffer; losers from pool_0k blown
    surviving_pool = n_pool_2k_winners + n_pool_0k_winners + (pool_2k - n_pool_2k_winners)
    # Second milk round: hedge $1K across remaining survivors
    if surviving_pool > 0:
        n_round2_winners = rng.binomial(surviving_pool, 0.5)
        # Most have $1-1.5K buffer; another $1K hedge will blow ~half the losers
        # Winners gain $1K, then withdraw whatever they can ($1.5K if buffer >=$1.5K)
        milk_revenue += n_round2_winners * MILK_WITHDRAW * 0.5  # conservative: half can withdraw
        # Pool collapses fast - assume all blow within 2-3 more rounds
    # All accounts eventually blow

    total_revenue = first_payout_revenue + milk_revenue
    net_pnl = total_revenue - eval_cost

    return dict(
        n_evals=n_evals, eval_cost=eval_cost,
        n_survivors_after_hedge=n_survivors_after_hedge,
        hedge_cushion_total=hedge_cushion_total,
        tp_hits=tp_hits,
        n_survivors_after_grind=n_survivors_after_grind,
        first_payout_revenue=first_payout_revenue,
        milk_revenue=milk_revenue,
        total_revenue=total_revenue,
        net_pnl=net_pnl,
    )


def run_mc(wr: float, hedge_tp_rate: float, grind_blow: float, n_trials: int = 10_000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = [run_one_cycle(wr, hedge_tp_rate, grind_blow, rng) for _ in range(n_trials)]
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, label: str) -> dict:
    return dict(
        label=label,
        mean_evals=df["n_evals"].mean(),
        mean_eval_cost=df["eval_cost"].mean(),
        mean_first_payout=df["first_payout_revenue"].mean(),
        mean_milk=df["milk_revenue"].mean(),
        mean_total_rev=df["total_revenue"].mean(),
        mean_net=df["net_pnl"].mean(),
        std_net=df["net_pnl"].std(),
        p10_net=df["net_pnl"].quantile(0.10),
        p50_net=df["net_pnl"].quantile(0.50),
        p90_net=df["net_pnl"].quantile(0.90),
        p_loss=(df["net_pnl"] < 0).mean(),
    )


def main():
    print("=" * 100)
    print("MC: Gambler's-Ruin Eval + 1:2 Hedge + 5-Day Grind + Payout Cycle")
    print("=" * 100)

    # ---- Baseline ----
    print("\n--- BASELINE PARAMS ---")
    print(f"  EVAL: bet ${EVAL_BET}, target ${EVAL_TARGET}, DD ${EVAL_DD}, cost ${EVAL_COST}/eval")
    print(f"  WR={EVAL_BASE_WR:.0%}, pass_prob={eval_pass_prob(EVAL_BASE_WR):.1%}")
    print(f"  HEDGE: 1:2 RR, TP_rate={HEDGE_TP_RATE:.0%}, n_target_funded={N_TARGET_FUNDED}")
    print(f"  GRIND: blow_prob={GRIND_BLOW_PROB:.0%}")
    print(f"  PAYOUT: ${PAYOUT_AMOUNT} per survivor, profit_split={PROFIT_SPLIT:.0%}")

    df = run_mc(EVAL_BASE_WR, HEDGE_TP_RATE, GRIND_BLOW_PROB, n_trials=20_000)
    base = summarize(df, "BASELINE (WR=50%, TP=30%, blow=5%)")

    print("\n--- BASELINE RESULTS (20,000 trials) ---")
    print(f"  Avg evals to get 10 funded:    {base['mean_evals']:.1f}")
    print(f"  Avg eval cost:                 ${base['mean_eval_cost']:,.0f}")
    print(f"  Avg first payout revenue:      ${base['mean_first_payout']:,.0f}")
    print(f"  Avg milk revenue:              ${base['mean_milk']:,.0f}")
    print(f"  Avg total revenue:             ${base['mean_total_rev']:,.0f}")
    print(f"  AVG NET PER CYCLE:             ${base['mean_net']:,.0f}")
    print(f"    (std ${base['std_net']:,.0f}, p10 ${base['p10_net']:,.0f}, p50 ${base['p50_net']:,.0f}, p90 ${base['p90_net']:,.0f})")
    print(f"  P(losing cycle):               {base['p_loss']:.1%}")
    cycles_per_year = TRADING_DAYS_YR / DAYS_PER_CYCLE
    print(f"\n  Cycles per year ({DAYS_PER_CYCLE}d/cycle): {cycles_per_year:.1f}")
    print(f"  AVG ANNUAL NET:                ${base['mean_net'] * cycles_per_year:,.0f}")

    # ---- Sensitivity sweeps ----
    print("\n" + "=" * 100)
    print("SENSITIVITY: WR (eval pass rate driver)")
    print("=" * 100)
    rows = []
    for wr in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        df_wr = run_mc(wr, HEDGE_TP_RATE, GRIND_BLOW_PROB, n_trials=10_000)
        s = summarize(df_wr, f"WR={wr:.0%}")
        s["pass_prob"] = eval_pass_prob(wr)
        s["annual_net"] = s["mean_net"] * cycles_per_year
        rows.append(s)
    df_wr_sweep = pd.DataFrame(rows)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
    cols = ["label", "pass_prob", "mean_evals", "mean_eval_cost", "mean_total_rev", "mean_net", "p10_net", "p90_net", "annual_net"]
    print(df_wr_sweep[cols].to_string(index=False, float_format=lambda x: f"{x:,.0f}" if abs(x) > 10 else f"{x:.3f}"))

    print("\n" + "=" * 100)
    print("SENSITIVITY: HEDGE TP rate (P(survivor hits +$4K vs +$2K))")
    print("=" * 100)
    rows = []
    for tp in [0.10, 0.20, 0.30, 0.40, 0.50]:
        df_tp = run_mc(EVAL_BASE_WR, tp, GRIND_BLOW_PROB, n_trials=10_000)
        s = summarize(df_tp, f"TP_rate={tp:.0%}")
        s["annual_net"] = s["mean_net"] * cycles_per_year
        rows.append(s)
    df_tp_sweep = pd.DataFrame(rows)
    cols2 = ["label", "mean_total_rev", "mean_milk", "mean_net", "annual_net"]
    print(df_tp_sweep[cols2].to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    print("\n" + "=" * 100)
    print("SENSITIVITY: 5-day grind blow rate")
    print("=" * 100)
    rows = []
    for bp in [0.02, 0.05, 0.10, 0.15, 0.20]:
        df_bp = run_mc(EVAL_BASE_WR, HEDGE_TP_RATE, bp, n_trials=10_000)
        s = summarize(df_bp, f"grind_blow={bp:.0%}")
        s["annual_net"] = s["mean_net"] * cycles_per_year
        rows.append(s)
    df_bp_sweep = pd.DataFrame(rows)
    cols3 = ["label", "mean_first_payout", "mean_milk", "mean_net", "annual_net"]
    print(df_bp_sweep[cols3].to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    print("\n" + "=" * 100)
    print("DISTRIBUTION OF EVALS NEEDED (baseline, WR=50%)")
    print("=" * 100)
    pct = df["n_evals"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(pct.to_string())

    # Save all results
    df.to_csv("scripts/montecarlo/results/gamblers_ruin_baseline_trials.csv", index=False)
    df_wr_sweep.to_csv("scripts/montecarlo/results/gamblers_ruin_wr_sweep.csv", index=False)
    print("\nSaved results to scripts/montecarlo/results/")


if __name__ == "__main__":
    from pathlib import Path
    Path("scripts/montecarlo/results").mkdir(parents=True, exist_ok=True)
    main()
