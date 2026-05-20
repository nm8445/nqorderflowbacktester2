"""
Monte Carlo: Lucid 2-account alternating + single + copytrade funded comparison.

Rules modeled:
  - $50k challenge account, $2k trailing DD, $3k profit target
  - EOD DD: HWM only updates at end of day; intraday equity CAN still blow the
    account if MAE breaches the floor (unrealized equity blow).
  - Trailing DD locks once HWM reaches $50k (no further trailing once past start).
  - 50% consistency: best single day must be <= 50% of total realized profit
    (required to pass challenge ONLY; funded has no consistency rule).
  - Funded payout: weekly sweep, withdraw any excess above $2,100 buffer.
  - Slippage: 2 tick entry + 2 tick exit = 1.0 pt total drag per trade (default)

Strategy: VWAP reaction base (ADX 15-30) + trending band (ADX >= 25).

Output:
  Part 1: Challenge pass rate — 2-account alternating, MNQ 1-10
  Part 2: Funded payouts — single account vs 2-acct alternate vs 2-acct copytrade
  Part 3: Slippage sensitivity — 1-10 ticks total slip
  Markdown report: docs/MONTECARLO_REPORT.md

Usage:
    python -u scripts/vwap_reaction_strat_backtest/montecarlo_lucid_2acct_alternate.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import montecarlo_apex_full as base
# Override trend ADX threshold inside the import
base.TREND_ADX_MIN = 25.0

from montecarlo_apex_full import (
    collect_combined_trades, build_adx_lookup,
    MNQ_PV, ACCOUNT_SIZE, TRAIL_DD, PROFIT_TARGET,
    CONSISTENCY_RULE, FUNDED_BUFFER, ACCOUNT_COST,
    N_SIMS, MAX_DAYS_CHALLENGE, MAX_DAYS_FUNDED, SEED,
)

TICK = 0.25
ENTRY_SLIP_TICKS = 2
EXIT_SLIP_TICKS = 2
SLIP_POINTS = (ENTRY_SLIP_TICKS + EXIT_SLIP_TICKS) * TICK  # total per-trade drag

N = N_SIMS
DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs"


# =====================================================================
# Challenge sim -- 2-account alternating, EOD DD
# =====================================================================

def sim_alternate_challenge(pnl_pool, mae_pool, day_counts, contracts, rng, slip_pts=SLIP_POINTS):
    """
    2 accounts A/B alternating per trade. Pass when EITHER account hits target.
    Returns (pass_rate, avg_pass_days, blow_rate).
    """
    pnl_adj = (pnl_pool - slip_pts) * MNQ_PV * contracts
    mae_adj = mae_pool * MNQ_PV * contracts

    trades_per_day = rng.choice(day_counts, size=(N, MAX_DAYS_CHALLENGE))
    total_needed = int(trades_per_day.sum())
    idx = rng.integers(0, len(pnl_pool), size=total_needed)
    pnls = pnl_adj[idx]
    maes = mae_adj[idx]

    passes = 0
    blown = 0
    pass_days = []
    cursor = 0

    for sim in range(N):
        eq = [float(ACCOUNT_SIZE), float(ACCOUNT_SIZE)]
        hwm = [float(ACCOUNT_SIZE), float(ACCOUNT_SIZE)]
        floor = [float(ACCOUNT_SIZE - TRAIL_DD), float(ACCOUNT_SIZE - TRAIL_DD)]
        locked = [False, False]
        alive = [True, True]
        best_day = [0.0, 0.0]
        total_profit = [0.0, 0.0]
        passed_acct = -1
        toggle = 0

        for di in range(MAX_DAYS_CHALLENGE):
            nt = int(trades_per_day[sim, di])
            day_pnl = [0.0, 0.0]

            for _ in range(nt):
                tp = pnls[cursor]
                tm = maes[cursor]
                cursor += 1

                target = -1
                for step in range(2):
                    cand = (toggle + step) % 2
                    if alive[cand]:
                        target = cand
                        break
                if target == -1:
                    continue
                toggle = (target + 1) % 2

                a = target
                if eq[a] - tm <= floor[a]:
                    alive[a] = False
                    continue
                eq[a] += tp
                day_pnl[a] += tp
                if tp > 0:
                    total_profit[a] += tp
                if eq[a] <= floor[a]:
                    alive[a] = False

            for a in range(2):
                if not alive[a]:
                    continue
                if day_pnl[a] > best_day[a]:
                    best_day[a] = day_pnl[a]
                if not locked[a] and eq[a] > hwm[a]:
                    hwm[a] = eq[a]
                    if hwm[a] >= ACCOUNT_SIZE:
                        locked[a] = True
                        floor[a] = ACCOUNT_SIZE - TRAIL_DD
                    else:
                        floor[a] = hwm[a] - TRAIL_DD
                if eq[a] - ACCOUNT_SIZE >= PROFIT_TARGET:
                    if total_profit[a] > 0 and best_day[a] <= CONSISTENCY_RULE * total_profit[a]:
                        passed_acct = a
                        break
            if passed_acct >= 0:
                passes += 1
                pass_days.append(di + 1)
                break
            if not alive[0] and not alive[1]:
                blown += 1
                break

    return (
        passes / N,
        float(np.mean(pass_days)) if pass_days else 0.0,
        blown / N,
    )


# =====================================================================
# Funded: single account (no rotation)
# =====================================================================

def sim_single_funded(pnl_pool, mae_pool, day_counts, contracts, rng, slip_pts=SLIP_POINTS):
    """
    Single-account funded sim. 1 year (252 days), weekly payout sweeps.
    No consistency rule. DD locked at $50k floor.
    Returns (payout_chance%, avg_payouts, avg_days_to_first, blow%, avg_$/payout).
    """
    pnl_adj = (pnl_pool - slip_pts) * MNQ_PV * contracts
    mae_adj = mae_pool * MNQ_PV * contracts

    DAYS = 252
    trades_per_day = rng.choice(day_counts, size=(N, DAYS))
    total_needed = int(trades_per_day.sum())
    idx = rng.integers(0, len(pnl_pool), size=total_needed)
    pnls = pnl_adj[idx]
    maes = mae_adj[idx]

    all_payouts = np.zeros(N, dtype=np.int32)
    all_payout_dollars = np.zeros(N)
    first_payout_day = np.full(N, -1, dtype=np.int32)
    blowups = 0
    cursor = 0

    for sim in range(N):
        equity = float(ACCOUNT_SIZE)
        floor = float(ACCOUNT_SIZE - TRAIL_DD)
        blown = False

        for di in range(DAYS):
            nt = int(trades_per_day[sim, di])
            for _ in range(nt):
                tp = pnls[cursor]; tm = maes[cursor]; cursor += 1
                if equity - tm <= floor:
                    blown = True; break
                equity += tp
                if equity <= floor:
                    blown = True; break
            if blown:
                for rd in range(di + 1, DAYS):
                    cursor += int(trades_per_day[sim, rd])
                break

            if (di + 1) % 5 == 0:
                excess = equity - (ACCOUNT_SIZE + FUNDED_BUFFER)
                if excess > 0:
                    all_payouts[sim] += 1
                    all_payout_dollars[sim] += excess
                    if first_payout_day[sim] == -1:
                        first_payout_day[sim] = di + 1
                    equity -= excess

        if blown:
            blowups += 1

    got_payout = np.sum(all_payouts > 0)
    payout_chance = got_payout / N * 100
    avg_payouts = float(np.mean(all_payouts))
    valid_first = first_payout_day[first_payout_day >= 0]
    avg_days_first = float(np.mean(valid_first)) if len(valid_first) else 0
    with_payouts = all_payouts > 0
    avg_per_payout = float(np.mean(all_payout_dollars[with_payouts] / all_payouts[with_payouts])) if got_payout else 0
    blow_pct = blowups / N * 100

    return payout_chance, avg_payouts, avg_days_first, blow_pct, avg_per_payout


# =====================================================================
# Funded: 2-account alternating
# =====================================================================

def sim_alternate_funded(pnl_pool, mae_pool, day_counts, contracts, rng, slip_pts=SLIP_POINTS):
    """
    2 alternating accounts funded sim. 1 year, weekly payout sweeps per account.
    No consistency rule. DD locked at $50k floor per account.
    Returns (payout_chance%, avg_payouts, avg_days_to_first, blow%, avg_$/payout).
    """
    pnl_adj = (pnl_pool - slip_pts) * MNQ_PV * contracts
    mae_adj = mae_pool * MNQ_PV * contracts

    DAYS = 252
    trades_per_day = rng.choice(day_counts, size=(N, DAYS))
    total_needed = int(trades_per_day.sum())
    idx = rng.integers(0, len(pnl_pool), size=total_needed)
    pnls = pnl_adj[idx]
    maes = mae_adj[idx]

    all_payouts = np.zeros(N, dtype=np.int32)
    all_payout_dollars = np.zeros(N)
    first_payout_day = np.full(N, -1, dtype=np.int32)
    blowups = 0
    cursor = 0

    for sim in range(N):
        eq = [float(ACCOUNT_SIZE), float(ACCOUNT_SIZE)]
        floor = [float(ACCOUNT_SIZE - TRAIL_DD), float(ACCOUNT_SIZE - TRAIL_DD)]
        alive = [True, True]
        toggle = 0

        for di in range(DAYS):
            nt = int(trades_per_day[sim, di])
            for _ in range(nt):
                tp = pnls[cursor]; tm = maes[cursor]; cursor += 1
                target = -1
                for step in range(2):
                    cand = (toggle + step) % 2
                    if alive[cand]:
                        target = cand
                        break
                if target == -1:
                    continue
                toggle = (target + 1) % 2
                a = target
                if eq[a] - tm <= floor[a]:
                    alive[a] = False
                    continue
                eq[a] += tp
                if eq[a] <= floor[a]:
                    alive[a] = False

            if (di + 1) % 5 == 0:
                for a in range(2):
                    if not alive[a]:
                        continue
                    excess = eq[a] - (ACCOUNT_SIZE + FUNDED_BUFFER)
                    if excess > 0:
                        all_payouts[sim] += 1
                        all_payout_dollars[sim] += excess
                        if first_payout_day[sim] == -1:
                            first_payout_day[sim] = di + 1
                        eq[a] -= excess

            if not alive[0] and not alive[1]:
                for rd in range(di + 1, DAYS):
                    cursor += int(trades_per_day[sim, rd])
                break

        if not alive[0] and not alive[1]:
            blowups += 1

    got_payout = np.sum(all_payouts > 0)
    payout_chance = got_payout / N * 100
    avg_payouts = float(np.mean(all_payouts))
    valid_first = first_payout_day[first_payout_day >= 0]
    avg_days_first = float(np.mean(valid_first)) if len(valid_first) else 0
    with_payouts = all_payouts > 0
    avg_per_payout = float(np.mean(all_payout_dollars[with_payouts] / all_payouts[with_payouts])) if got_payout else 0
    blow_pct = blowups / N * 100

    return payout_chance, avg_payouts, avg_days_first, blow_pct, avg_per_payout


# =====================================================================
# Funded: 2-account copytrade (both take every trade)
# =====================================================================

def sim_copytrade_funded(pnl_pool, mae_pool, day_counts, contracts, rng, slip_pts=SLIP_POINTS):
    """
    2 accounts take EVERY trade simultaneously (copytrade).
    Each account has independent equity/DD. Weekly payout per account.
    No consistency rule.
    Returns (payout_chance%, avg_payouts, avg_days_to_first, blow%, avg_$/payout).
    """
    pnl_adj = (pnl_pool - slip_pts) * MNQ_PV * contracts
    mae_adj = mae_pool * MNQ_PV * contracts

    DAYS = 252
    trades_per_day = rng.choice(day_counts, size=(N, DAYS))
    total_needed = int(trades_per_day.sum())
    idx = rng.integers(0, len(pnl_pool), size=total_needed)
    pnls = pnl_adj[idx]
    maes = mae_adj[idx]

    all_payouts = np.zeros(N, dtype=np.int32)
    all_payout_dollars = np.zeros(N)
    first_payout_day = np.full(N, -1, dtype=np.int32)
    blowups = 0
    cursor = 0

    for sim in range(N):
        eq = [float(ACCOUNT_SIZE), float(ACCOUNT_SIZE)]
        floor = [float(ACCOUNT_SIZE - TRAIL_DD), float(ACCOUNT_SIZE - TRAIL_DD)]
        alive = [True, True]

        for di in range(DAYS):
            nt = int(trades_per_day[sim, di])
            for _ in range(nt):
                tp = pnls[cursor]; tm = maes[cursor]; cursor += 1
                # Both accounts take the same trade
                for a in range(2):
                    if not alive[a]:
                        continue
                    if eq[a] - tm <= floor[a]:
                        alive[a] = False
                        continue
                    eq[a] += tp
                    if eq[a] <= floor[a]:
                        alive[a] = False

            if (di + 1) % 5 == 0:
                for a in range(2):
                    if not alive[a]:
                        continue
                    excess = eq[a] - (ACCOUNT_SIZE + FUNDED_BUFFER)
                    if excess > 0:
                        all_payouts[sim] += 1
                        all_payout_dollars[sim] += excess
                        if first_payout_day[sim] == -1:
                            first_payout_day[sim] = di + 1
                        eq[a] -= excess

            if not alive[0] and not alive[1]:
                for rd in range(di + 1, DAYS):
                    cursor += int(trades_per_day[sim, rd])
                break

        if not alive[0] and not alive[1]:
            blowups += 1

    got_payout = np.sum(all_payouts > 0)
    payout_chance = got_payout / N * 100
    avg_payouts = float(np.mean(all_payouts))
    valid_first = first_payout_day[first_payout_day >= 0]
    avg_days_first = float(np.mean(valid_first)) if len(valid_first) else 0
    with_payouts = all_payouts > 0
    avg_per_payout = float(np.mean(all_payout_dollars[with_payouts] / all_payouts[with_payouts])) if got_payout else 0
    blow_pct = blowups / N * 100

    return payout_chance, avg_payouts, avg_days_first, blow_pct, avg_per_payout


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 100)
    print("  LUCID MONTE CARLO — COMPREHENSIVE REPORT")
    print(f"  ADX >= 25 trending | slip: {ENTRY_SLIP_TICKS}t entry + {EXIT_SLIP_TICKS}t exit = {SLIP_POINTS} pts drag")
    print(f"  EOD DD (HWM updates at close) | {N:,} sims | Funded buffer: ${FUNDED_BUFFER:,.0f}")
    print("=" * 100)

    print("\nBuilding ADX lookup + collecting trades...")
    adx_lookup = build_adx_lookup()
    pnl_pool, mae_pool, sl_pool, day_counts = collect_combined_trades(adx_lookup)
    print(f"  {len(pnl_pool)} trades in pool | {len(day_counts)} days | "
          f"avg {day_counts.mean():.2f} trades/day | avg PnL {pnl_pool.mean():+.3f} pts "
          f"(after slip: {pnl_pool.mean() - SLIP_POINTS:+.3f} pts)")

    avg_sl_pts = float(np.mean(sl_pool))
    md = []  # markdown lines

    md.append("# Lucid Monte Carlo Simulation Report")
    md.append("")
    md.append(f"**Strategy:** Combined VWAP (base ADX 15-30 + trending ADX >= 25)")
    md.append(f"**Trade pool:** {len(pnl_pool)} trades | {len(day_counts)} days | avg {day_counts.mean():.1f} trades/day | avg PnL {pnl_pool.mean():+.3f} pts")
    md.append(f"**Simulations:** {N:,} per cell | Seed: {SEED}")
    md.append(f"**Lucid rules:** $50k account, $2k trailing DD (EOD HWM), $3k target, 50% consistency (challenge only)")
    md.append(f"**Funded:** no consistency rule, weekly payout sweep, ${FUNDED_BUFFER:,.0f} buffer kept")
    md.append(f"**Default slippage:** {ENTRY_SLIP_TICKS}t entry + {EXIT_SLIP_TICKS}t exit = {SLIP_POINTS} pts total drag")
    md.append("")

    # ── PART 1: Challenge pass rates ─────────────────────────────────
    rng_ch = np.random.default_rng(SEED)
    print("\n" + "=" * 100)
    print("  PART 1: CHALLENGE PASS RATE (2-account alternating, pass if EITHER hits $3k + consistency)")
    print("=" * 100)
    print(f"  {'MNQ':>4} | {'Pass%':>7} | {'Avg Days':>9} | {'Blow%':>6} | {'Risk/trade':>12}")
    print("  " + "-" * 55)

    md.append("## Part 1: Challenge Pass Rates (2-Account Alternating)")
    md.append("")
    md.append("Pass when EITHER account hits $3k profit target with 50% consistency rule.")
    md.append("")
    md.append("| MNQ | Pass % | Avg Days | Blow % | Risk/Trade |")
    md.append("|----:|-------:|---------:|-------:|-----------:|")

    best_ch = None
    for mnq in range(1, 11):
        pr, ad, br = sim_alternate_challenge(pnl_pool, mae_pool, day_counts, mnq, rng_ch)
        risk = mnq * avg_sl_pts * MNQ_PV
        print(f"  {mnq:>4} | {pr*100:>6.1f}% | {ad:>9.1f} | {br*100:>5.1f}% | ${risk:>10,.0f}")
        md.append(f"| {mnq} | {pr*100:.1f}% | {ad:.0f} | {br*100:.1f}% | ${risk:,.0f} |")
        if best_ch is None or pr > best_ch[1] or (pr == best_ch[1] and ad < best_ch[2]):
            best_ch = (mnq, pr, ad)

    md.append("")

    # ── PART 2a: Funded — single account ─────────────────────────────
    print("\n" + "=" * 100)
    print("  PART 2a: FUNDED — SINGLE ACCOUNT (1 year, weekly payout, no consistency)")
    print("=" * 100)
    print(f"  {'MNQ':>4} | {'Payout%':>8} | {'Avg Payouts':>11} | {'Days to 1st':>11} | {'Avg $/PO':>10} | {'Blow%':>6}")
    print("  " + "-" * 70)

    md.append("## Part 2a: Funded Payouts -- Single Account (1 Year)")
    md.append("")
    md.append("Weekly payout sweep. No consistency rule. DD locked at $50k.")
    md.append("")
    md.append("| MNQ | Payout Chance | Avg Payouts | Days to 1st | Avg $/Payout | Blow % |")
    md.append("|----:|--------------:|------------:|------------:|-------------:|-------:|")

    rng_f1 = np.random.default_rng(SEED + 1)
    best_f1 = None
    for mnq in range(1, 11):
        pc, ap, adf, bp, adp = sim_single_funded(pnl_pool, mae_pool, day_counts, mnq, rng_f1)
        print(f"  {mnq:>4} | {pc:>7.1f}% | {ap:>11.1f} | {adf:>11.0f} | ${adp:>8,.0f} | {bp:>5.1f}%")
        md.append(f"| {mnq} | {pc:.1f}% | {ap:.1f} | {adf:.0f} | ${adp:,.0f} | {bp:.1f}% |")
        if best_f1 is None or ap > best_f1[1]:
            best_f1 = (mnq, ap, adp)

    md.append("")

    # ── PART 2b: Funded — 2-account alternating ────���─────────────────
    print("\n" + "=" * 100)
    print("  PART 2b: FUNDED — 2-ACCOUNT ALTERNATING (1 year, weekly payout per account)")
    print("=" * 100)
    print(f"  {'MNQ':>4} | {'Payout%':>8} | {'Avg Payouts':>11} | {'Days to 1st':>11} | {'Avg $/PO':>10} | {'Blow%':>6}")
    print("  " + "-" * 70)

    md.append("## Part 2b: Funded Payouts -- 2-Account Alternating (1 Year)")
    md.append("")
    md.append("Trades alternate between accounts. Weekly payout per account independently.")
    md.append("")
    md.append("| MNQ | Payout Chance | Avg Payouts | Days to 1st | Avg $/Payout | Blow % |")
    md.append("|----:|--------------:|------------:|------------:|-------------:|-------:|")

    rng_f2 = np.random.default_rng(SEED + 2)
    best_f2 = None
    for mnq in range(1, 11):
        pc, ap, adf, bp, adp = sim_alternate_funded(pnl_pool, mae_pool, day_counts, mnq, rng_f2)
        print(f"  {mnq:>4} | {pc:>7.1f}% | {ap:>11.1f} | {adf:>11.0f} | ${adp:>8,.0f} | {bp:>5.1f}%")
        md.append(f"| {mnq} | {pc:.1f}% | {ap:.1f} | {adf:.0f} | ${adp:,.0f} | {bp:.1f}% |")
        if best_f2 is None or ap > best_f2[1]:
            best_f2 = (mnq, ap, adp)

    md.append("")

    # ── PART 2c: Funded — 2-account copytrade ────────────────────────
    print("\n" + "=" * 100)
    print("  PART 2c: FUNDED — 2-ACCOUNT COPYTRADE (both take every trade)")
    print("=" * 100)
    print(f"  {'MNQ':>4} | {'Payout%':>8} | {'Avg Payouts':>11} | {'Days to 1st':>11} | {'Avg $/PO':>10} | {'Blow%':>6}")
    print("  " + "-" * 70)

    md.append("## Part 2c: Funded Payouts -- 2-Account Copytrade (1 Year)")
    md.append("")
    md.append("Both accounts take every trade simultaneously. Independent equity/DD per account.")
    md.append("")
    md.append("| MNQ | Payout Chance | Avg Payouts | Days to 1st | Avg $/Payout | Blow % |")
    md.append("|----:|--------------:|------------:|------------:|-------------:|-------:|")

    rng_f3 = np.random.default_rng(SEED + 3)
    best_f3 = None
    for mnq in range(1, 11):
        pc, ap, adf, bp, adp = sim_copytrade_funded(pnl_pool, mae_pool, day_counts, mnq, rng_f3)
        print(f"  {mnq:>4} | {pc:>7.1f}% | {ap:>11.1f} | {adf:>11.0f} | ${adp:>8,.0f} | {bp:>5.1f}%")
        md.append(f"| {mnq} | {pc:.1f}% | {ap:.1f} | {adf:.0f} | ${adp:,.0f} | {bp:.1f}% |")
        if best_f3 is None or ap > best_f3[1]:
            best_f3 = (mnq, ap, adp)

    md.append("")

    # ── PART 3: Slippage sensitivity — full MNQ x Slip grids ────────
    print("\n" + "=" * 100)
    print("  PART 3: SLIPPAGE SENSITIVITY (MNQ 1-10 x Slip 1-10 ticks)")
    print("=" * 100)

    slip_range = range(1, 11)
    mnq_range = range(1, 11)
    slip_header = " | ".join(f"{s}t" for s in slip_range)

    # ── 3a: Challenge pass % grid
    md.append("## Part 3a: Challenge Pass % by MNQ and Slippage")
    md.append("")
    md.append("2-account alternating. Values are pass % at each MNQ/slippage combination.")
    md.append("")
    md.append("| MNQ | " + " | ".join(f"{s}t" for s in slip_range) + " |")
    md.append("|----:|" + "|".join("---:" for _ in slip_range) + "|")

    print("\n  3a: Challenge Pass %")
    print(f"  {'MNQ':>4} | {slip_header}")
    print("  " + "-" * (7 + 7 * len(list(slip_range))))

    for mnq in mnq_range:
        vals = []
        for slip_ticks in slip_range:
            slip_pts = slip_ticks * TICK
            rng_s = np.random.default_rng(SEED + 1000 + mnq * 100 + slip_ticks)
            pr, _, _ = sim_alternate_challenge(pnl_pool, mae_pool, day_counts, mnq, rng_s, slip_pts=slip_pts)
            vals.append(pr * 100)
        print(f"  {mnq:>4} | " + " | ".join(f"{v:5.1f}%" for v in vals))
        md.append(f"| {mnq} | " + " | ".join(f"{v:.1f}%" for v in vals) + " |")

    md.append("")

    # ── 3b: Funded single-account avg payouts grid
    md.append("## Part 3b: Funded Avg Payouts -- Single Account by MNQ and Slippage")
    md.append("")
    md.append("1 account, 1 year. Values are average payouts per year.")
    md.append("")
    md.append("| MNQ | " + " | ".join(f"{s}t" for s in slip_range) + " |")
    md.append("|----:|" + "|".join("---:" for _ in slip_range) + "|")

    print("\n  3b: Funded Single-Account Avg Payouts")
    print(f"  {'MNQ':>4} | {slip_header}")
    print("  " + "-" * (7 + 7 * len(list(slip_range))))

    for mnq in mnq_range:
        vals = []
        for slip_ticks in slip_range:
            slip_pts = slip_ticks * TICK
            rng_s = np.random.default_rng(SEED + 2000 + mnq * 100 + slip_ticks)
            _, ap, _, _, _ = sim_single_funded(pnl_pool, mae_pool, day_counts, mnq, rng_s, slip_pts=slip_pts)
            vals.append(ap)
        print(f"  {mnq:>4} | " + " | ".join(f"{v:5.1f} " for v in vals))
        md.append(f"| {mnq} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")

    md.append("")

    # ── 3c: Funded 2-acct alternating avg payouts grid
    md.append("## Part 3c: Funded Avg Payouts -- 2-Account Alternating by MNQ and Slippage")
    md.append("")
    md.append("2 accounts alternating, 1 year. Values are average total payouts (both accounts combined).")
    md.append("")
    md.append("| MNQ | " + " | ".join(f"{s}t" for s in slip_range) + " |")
    md.append("|----:|" + "|".join("---:" for _ in slip_range) + "|")

    print("\n  3c: Funded 2-Acct Alternating Avg Payouts")
    print(f"  {'MNQ':>4} | {slip_header}")
    print("  " + "-" * (7 + 7 * len(list(slip_range))))

    for mnq in mnq_range:
        vals = []
        for slip_ticks in slip_range:
            slip_pts = slip_ticks * TICK
            rng_s = np.random.default_rng(SEED + 3000 + mnq * 100 + slip_ticks)
            _, ap, _, _, _ = sim_alternate_funded(pnl_pool, mae_pool, day_counts, mnq, rng_s, slip_pts=slip_pts)
            vals.append(ap)
        print(f"  {mnq:>4} | " + " | ".join(f"{v:5.1f} " for v in vals))
        md.append(f"| {mnq} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")

    md.append("")

    # ── 3d: Funded 2-acct copytrade avg payouts grid
    md.append("## Part 3d: Funded Avg Payouts -- 2-Account Copytrade by MNQ and Slippage")
    md.append("")
    md.append("Both accounts take every trade. Values are average total payouts (both accounts combined).")
    md.append("")
    md.append("| MNQ | " + " | ".join(f"{s}t" for s in slip_range) + " |")
    md.append("|----:|" + "|".join("---:" for _ in slip_range) + "|")

    print("\n  3d: Funded 2-Acct Copytrade Avg Payouts")
    print(f"  {'MNQ':>4} | {slip_header}")
    print("  " + "-" * (7 + 7 * len(list(slip_range))))

    for mnq in mnq_range:
        vals = []
        for slip_ticks in slip_range:
            slip_pts = slip_ticks * TICK
            rng_s = np.random.default_rng(SEED + 4000 + mnq * 100 + slip_ticks)
            _, ap, _, _, _ = sim_copytrade_funded(pnl_pool, mae_pool, day_counts, mnq, rng_s, slip_pts=slip_pts)
            vals.append(ap)
        print(f"  {mnq:>4} | " + " | ".join(f"{v:5.1f} " for v in vals))
        md.append(f"| {mnq} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")

    md.append("")

    # ── Recommendations ──────────────────────────────────────────────
    md.append("## Recommendations")
    md.append("")
    md.append(f"- **Challenge (2-acct alternating):** {best_ch[0]} MNQ ({best_ch[1]*100:.1f}% pass, ~{best_ch[2]:.0f} days avg)")
    md.append(f"- **Funded single account:** {best_f1[0]} MNQ ({best_f1[1]:.1f} payouts/yr, ~${best_f1[2]:,.0f}/payout)")
    md.append(f"- **Funded 2-acct alternating:** {best_f2[0]} MNQ ({best_f2[1]:.1f} payouts/yr, ~${best_f2[2]:,.0f}/payout)")
    md.append(f"- **Funded 2-acct copytrade:** {best_f3[0]} MNQ ({best_f3[1]:.1f} payouts/yr, ~${best_f3[2]:,.0f}/payout)")
    md.append("")
    md.append("---")
    md.append(f"*Generated by `montecarlo_lucid_2acct_alternate.py` | {N:,} sims per cell*")

    # Write markdown
    DOCS_DIR.mkdir(exist_ok=True)
    out_path = DOCS_DIR / "MONTECARLO_REPORT.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nReport written to {out_path}")

    print("\n" + "=" * 100)
    print("  RECOMMENDATIONS")
    print("=" * 100)
    print(f"  Challenge (2-acct):     {best_ch[0]} MNQ  (pass {best_ch[1]*100:.1f}%, avg {best_ch[2]:.1f} days)")
    print(f"  Funded single:          {best_f1[0]} MNQ  ({best_f1[1]:.1f} payouts/yr, ~${best_f1[2]:,.0f}/payout)")
    print(f"  Funded 2-acct alt:      {best_f2[0]} MNQ  ({best_f2[1]:.1f} payouts/yr, ~${best_f2[2]:,.0f}/payout)")
    print(f"  Funded 2-acct copytrade:{best_f3[0]} MNQ  ({best_f3[1]:.1f} payouts/yr, ~${best_f3[2]:,.0f}/payout)")


if __name__ == "__main__":
    main()
