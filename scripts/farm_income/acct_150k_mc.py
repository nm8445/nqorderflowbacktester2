"""150k futures account MC ($4,500 trailing-lock DD, $9,000 eval target, ~$300 eval fee).

Mirrors the 50k farm-income framework (farm_income_mc.py / funded_target_risk_sweep.py) at 150k scale,
re-using the SCALE-FREE R-multiple distributions + per-1-MNQ daily packs -> no new backtests.

  GAMBLE config = 5-strat (4-way OD/RV/B2/FB + MEC) high-risk 1:1, used for BOTH the eval pass and the
    funded buffer-build.  MILK = 4-way combined at a fixed optimal MNQ (swept here).

Three deliverables:
  1. EVAL optimal per-trade risk that maximizes pass rate, for the 40% and 50% consistency rules.
     (1:1 bracket => per-trade hit-rate is scale-free; risk moves pass% only via DD/risk & target/risk.
      The consistency rule is a CEILING on risk: biggest day = risk <= rule% x $9,000.)
  2. FUNDED $/account: gamble->de-risk->milk, withdraw 50% of profit (capped $3k/$4k/$5k by firm),
     winning day >= $250, 5 winning days/payout, 2 payouts, 90% split. Optimal milk MNQ swept.
  3. Calculator inputs (pass%, $/funded, gamble-trades/cycle, milk-days/cycle) for the 150k preset.

User-confirmed rules (2026-06-27): split 90% | 5 winning days/payout | 2 payouts max.
Run: python scripts/farm_income/acct_150k_mc.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scripts.propfirm_milking.common import eval_bracket
from scripts.propfirm_milking.entries import build_entry_packs
from farm_income_mc import load_eval_R, load_daily_1mnq

# ---- 150k rule constants -----------------------------------------------------------------------
START, DD, TARGET = 150_000.0, 4_500.0, 9_000.0     # profit terms => start=0, floor trails to -DD then locks at 0
LOCK = 4_500.0              # funded de-risk point: gamble to here -> banks the DD, floor locks at start
WITHDRAW_TARGET = 9_000.0   # funded milk-to-here, then withdraw min(50%, cap), staying >= +$4,500
EVAL_FEE = 300.0
SPLIT = 0.90                 # trader keeps 90% of each withdrawal
WIN_DAY = 250.0              # a funded "winning day" = realized >= this
WINS_NEEDED = 5             # winning days per payout
MAX_WD = 2                   # payouts before retirement
MIN_WD = 500.0               # stop if a withdrawal would be < this
CAPS = [3_000.0, 4_000.0, 5_000.0]   # firm withdrawal caps (50% of profit, capped here)
TRADING_DAYS_YR = 251
N = 40_000


# ---- 5-strat (4-way + MEC) 1:1 R pool ----------------------------------------------------------
def mec_R() -> np.ndarray:
    """MEC per-trade R (atr28 / sl1.75, 1:1) — same build as add_mec_funded.py."""
    packs = build_entry_packs("MEC", atr_lens=[28], verbose=False)
    tr = eval_bracket(packs, atr_len=28, sl_mult=1.75, tp_mult=1.75)
    return (tr["pnl_pts"] / tr["sl_dist_pts"]).to_numpy()


def five_strat_R() -> np.ndarray:
    """4-way farm-config 1:1 combined R (cached csv) + MEC R = the 5-strat gamble pool."""
    return np.concatenate([load_eval_R(), mec_R()])


# ---- eval (= the gamble to +$9k under the consistency-capped risk) -----------------------------
def eval_sim(poolR, risk, rng, max_days=150):
    """One signal/day at 1:1 risk `risk`; reach +$9,000 before the trailing-lock floor.
    Floor = min(0, peak - DD) (trails to -DD, locks at 0 once +DD banked). Returns (pass?, days).
    Consistency is satisfied by construction whenever risk <= rule% x $9,000 (biggest day = risk)."""
    profit = 0.0; peak = 0.0; days = 0
    while days < max_days:
        days += 1
        profit += poolR[rng.integers(poolR.size)] * risk
        peak = max(peak, profit)
        if profit <= min(0.0, peak - DD) + 1e-9:
            return False, days
        if profit >= TARGET:
            return True, days
    return False, days


def eval_pass(poolR, risk, n=80_000, seed=7):
    rng = np.random.default_rng(seed)
    out = [eval_sim(poolR, risk, rng) for _ in range(n)]
    ok = np.array([o[0] for o in out]); dys = np.array([o[1] for o in out])
    return ok.mean(), (np.median(dys[ok]) if ok.any() else float("nan")), dys.mean()


# ---- funded: gamble to +$4,500 lock -> milk to +$9k -> withdraw 50% (capped), stay >= +$4,500 ----
def gamble_to_lock(pool_d, rng):
    """Gamble (5-strat 1:1) from 0 until profit >= LOCK (de-risk: banks the DD, floor locks at start)
    or blow. Floor trails min(0, peak - DD). Returns (profit|None, trades)."""
    p = 0.0; peak = 0.0; t = 0
    for _ in range(400):
        t += 1
        p += pool_d[rng.integers(pool_d.size)]
        peak = max(peak, p)
        if p <= min(0.0, peak - DD) + 1e-9:
            return None, t
        if p >= LOCK:
            return p, t
    return None, t


def milk_to_target(p, daily_mnq, rng):
    """Milk the 4-way daily P&L (chosen MNQ) from p up to >= WITHDRAW_TARGET AND >= WINS_NEEDED winning
    days (>= $250). Floor locked at 0 -> a red streak that loses the whole +profit blows it. The milk
    starts from only +$4,500 of cushion, so MNQ is a real risk knob here. Returns (profit|None, days)."""
    days = 0; wins = 0; n = daily_mnq.size
    while p < WITHDRAW_TARGET or wins < WINS_NEEDED:
        days += 1
        pnl = daily_mnq[rng.integers(n)]
        p += pnl
        if p <= 1e-9:
            return None, days
        if pnl >= WIN_DAY:
            wins += 1
    return p, days


def funded_life(pool_d, daily_mnq, cap, rng):
    """$ pocketed (after split) + gamble trades + milk days for one funded account. Lock-and-milk:
    gamble to +$4,500, then for each payout milk to +$9k and withdraw min(50%, cap)."""
    cash = 0.0; mdays = 0
    p, gtr = gamble_to_lock(pool_d, rng)
    if p is None:
        return 0.0, gtr, mdays
    for _ in range(MAX_WD):
        p2, md = milk_to_target(p, daily_mnq, rng); mdays += md
        if p2 is None:
            return cash, gtr, mdays
        p = p2
        w = min(0.5 * p, cap)
        if w < MIN_WD:
            return cash, gtr, mdays
        cash += w * SPLIT; p -= w
    return cash, gtr, mdays


def funded_stats(pool_d, daily_1mnq, mnq, cap, n=N, seed=5):
    rng = np.random.default_rng(seed)
    daily_mnq = daily_1mnq * mnq
    cash = np.empty(n); gtr = np.empty(n); md = np.empty(n)
    for i in range(n):
        cash[i], gtr[i], md[i] = funded_life(pool_d, daily_mnq, cap, rng)
    paid = cash > 0
    return dict(ef=cash.mean(), ppay=paid.mean(),
                gtr=gtr.mean(), md=md[paid].mean() if paid.any() else 0.0,
                days_p80=float(np.percentile(gtr + md, 80)))


# ---- cohort economics: per-eval EV, evals for a target, lanes x firms variance -----------------
def batch_year(pool_d, daily_mnq, cap, n_lanes, n_firms, p_pass, n_cycles, rng):
    """One simulated YEAR of the paired structure: n_lanes decorrelated lanes, each lane = n_firms
    accounts on IDENTICAL signals (lane simulated ONCE, counted n_firms times). Fee on every account."""
    annual = 0.0
    for _ in range(n_cycles):
        for _ in range(n_lanes):
            ext = funded_life(pool_d, daily_mnq, cap, rng)[0] if rng.random() < p_pass else 0.0
            annual += n_firms * ext - n_firms * EVAL_FEE
    return annual


def paired_stats(pool_d, daily_mnq, cap, n_lanes, n_firms, p_pass, cycles_yr, n_years=4000, seed=11):
    rng = np.random.default_rng(seed)
    nc = max(1, int(round(cycles_yr)))
    v = np.array([batch_year(pool_d, daily_mnq, cap, n_lanes, n_firms, p_pass, nc, rng)
                  for _ in range(n_years)])
    m = v.mean()
    return dict(mean=m, p10=np.percentile(v, 10), p50=np.percentile(v, 50), p90=np.percentile(v, 90),
                ploss=(v < 0).mean(), cv=v.std() / m if m else float("nan"))


def main():
    poolR = five_strat_R()
    daily = load_daily_1mnq()               # 4-way, 1 MNQ, firm-day aggregated
    print(f"5-strat gamble pool: n={poolR.size}  WR={(poolR>0).mean()*100:.1f}%  meanR={poolR.mean():+.3f}")
    print(f"4-way milk daily (1 MNQ): n={daily.size}  mean=${daily.mean():.0f}  "
          f">=$250={(daily>=WIN_DAY).mean()*100:.0f}%  red={(daily<0).mean()*100:.0f}%\n")

    # 1) EVAL optimal risk per consistency rule -------------------------------------------------
    print("=" * 70)
    print("1) EVAL pass rate vs per-trade risk  (150k, $4.5k DD, +$9k target, 1:1)")
    print("=" * 70)
    fine = [1500, 1750, 2000, 2250, 2500, 2750, 3000, 3300, 3600, 4000, 4500]
    rules = {"50% consistency (risk<=$4,500)": (0.50, [r for r in fine if r <= 4500]),
             "40% consistency (risk<=$3,600)": (0.40, [r for r in fine if r <= 3600])}
    eval_best = {}
    for label, (cons, grid) in rules.items():
        print(f"\n  {label}:")
        print(f"    {'risk$':>6} {'DD/risk':>8} {'tgt/risk':>9} {'pass%':>7} {'med days':>9}")
        best = (None, -1.0)
        for risk in grid:
            p, med, _ = eval_pass(poolR, float(risk), n=150_000)
            mark = ""
            if p > best[1]:
                best = (risk, p); mark = "  <- best so far"
            print(f"    {risk:>6} {DD/risk:>8.2f} {TARGET/risk:>9.2f} {p*100:>6.1f}% {med:>9.0f}{mark}")
        eval_best[label] = best
        print(f"    => OPTIMAL risk ${best[0]:,} : pass {best[1]*100:.1f}%")

    # 2) FUNDED: gamble to +$4,500 lock -> milk to +$9k -> withdraw 50% (capped) -----------------
    print("\n" + "=" * 70)
    print("2) FUNDED lock-and-milk: gamble to +$4,500, milk to +$9k, withdraw min(50%, cap), x2")
    print("=" * 70)
    GAMBLE_RISK = 3600.0
    pool_d = poolR * GAMBLE_RISK
    reach = np.mean([gamble_to_lock(pool_d, np.random.default_rng(s))[0] is not None for s in range(8000)])
    print(f"  funded gamble risk ${GAMBLE_RISK:,.0f} -> P(reach +$4,500 lock) ~ {reach*100:.0f}%  "
          f"(vs ~37% to reach +$9k)\n")
    # MNQ optimum is by INCOME RATE (ef per calendar day), not raw $/funded: 1 MNQ maxes $/funded but
    # takes ~180 milk days/acct; higher MNQ trades a little $/funded for far shorter milk -> more cycles.
    TD_YR = TRADING_DAYS_YR
    MNQ_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    funded_best = {}
    for cap in CAPS:
        wd = min(0.5 * WITHDRAW_TARGET, cap)
        print(f"  ${cap:,.0f} cap -> withdraw ${wd:,.0f}/payout, keep +${WITHDRAW_TARGET-wd:,.0f} buffer:")
        print(f"    {'mnq':>4} {'$/funded':>9} {'P(>=1 pay)':>11} {'milk days':>10} {'$/yr/acct':>10}")
        rows = []
        for mnq in MNQ_GRID:
            st = funded_stats(pool_d, daily, mnq, cap)
            inc = st["ef"] * TD_YR / (st["gtr"] + st["md"]) if (st["gtr"] + st["md"]) else 0.0
            rows.append((mnq, st, inc))
        best = max(rows, key=lambda r: r[2])     # income-rate optimum
        for mnq, st, inc in rows:
            mark = "  <- opt (income)" if mnq == best[0] else ""
            print(f"    {mnq:>4} {st['ef']:>9,.0f} {st['ppay']*100:>10.0f}% {st['md']:>10.1f} {inc:>10,.0f}{mark}")
        funded_best[cap] = (best[0], best[1]["ef"], best[1])
        print()

    # 3) Calculator preset inputs ----------------------------------------------------------------
    print("=" * 70)
    print("3) CALCULATOR PRESET inputs (150k) — eval@$4,000 risk, optimal milk MNQ per cap")
    print("=" * 70)
    e50 = eval_best["50% consistency (risk<=$4,500)"]
    p50, med50, _ = eval_pass(poolR, float(e50[0]), n=150_000)
    print(f"  eval pass rate = {p50*100:.0f}%  (risk ${e50[0]:,}, median {med50:.0f} td)   eval fee = ${EVAL_FEE:.0f}\n")
    print(f"    {'cap$':>5} {'opt MNQ':>8} {'$/funded':>9} {'gamble tr/cyc':>14} {'milk days/cyc':>14}")
    for cap in CAPS:
        mnq, ef, st = funded_best[cap]
        print(f"    {cap:>5.0f} {mnq:>8} {ef:>9,.0f} {st['gtr']:>14.1f} {st['md']:>14.1f}")

    # 4) COHORT income readout (how many 150k evals -> a target $/yr) -----------------------------
    print("\n" + "=" * 70)
    print("4) COHORT income — 150k, $5k cap, 6 MNQ milk, eval pass 37%")
    print("=" * 70)
    MILK_MNQ = 6; cap = 5000.0
    st = funded_stats(pool_d, daily, MILK_MNQ, cap)
    per_funded = st["ef"]; days_p80 = st["days_p80"]
    per_eval = p50 * per_funded - EVAL_FEE                       # EV of one eval slot per cycle
    cycle_days = med50 + days_p80                                # eval days + funded extraction (p80)
    cycles_yr = TRADING_DAYS_YR / cycle_days
    ann_per_eval = per_eval * cycles_yr
    print(f"  $/funded ${per_funded:,.0f}  |  per-eval EV = {p50*100:.0f}% x ${per_funded:,.0f} - ${EVAL_FEE:.0f} "
          f"= ${per_eval:,.0f}")
    print(f"  cycle ~{cycle_days:.0f} td (eval {med50:.0f} + funded p80 {days_p80:.0f}) -> {cycles_yr:.1f} cycles/yr")
    print(f"  annual per eval slot = ${ann_per_eval:,.0f}  (re-buying the eval each cycle)\n")
    for target in (100_000, 200_000):
        evals = target / ann_per_eval if ann_per_eval > 0 else float("inf")
        print(f"  for ${target:,}/yr  -> ~{evals:.0f} eval slots/cycle  "
              f"(~{evals*cycles_yr:.0f} evals/yr, ${evals*cycles_yr*EVAL_FEE:,.0f} fees)")
    print(f"\n  Variance — L decorrelated lanes x 3 firms (correlated within lane):")
    print(f"    {'lanes':>5} {'evals/cyc':>10} {'mean $/yr':>11} {'p10':>9} {'p90':>9} {'CV':>5} {'P(loss)':>8}")
    for L in (2, 3, 5, 10):
        ps = paired_stats(pool_d, daily * MILK_MNQ, cap, n_lanes=L, n_firms=3,
                          p_pass=p50, cycles_yr=cycles_yr)
        print(f"    {L:>5} {L*3:>10} {ps['mean']:>11,.0f} {ps['p10']:>9,.0f} {ps['p90']:>9,.0f} "
              f"{ps['cv']:>5.2f} {ps['ploss']*100:>7.0f}%")
    print("  (throughput-real caps from signals/day are in prop_income_calculator.html — 150k presets.)")


if __name__ == "__main__":
    main()
