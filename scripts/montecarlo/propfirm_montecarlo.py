"""
Monte Carlo: prop-firm payout probability for the 3-strategy combined portfolio.

Rules:
  - Account starts at equity $0
  - Trailing drawdown: $2,000
    * Floor = peak - $2,000, capped at $0 (once peak >= $2,000, DD floor LOCKS at $0/break-even)
  - Payout: when equity >= +$4,000, withdraw $2,000 -> equity drops to $2,000
  - Risk: MNQ contracts 1-10 (base size multiplier; per-strategy martingale already baked into trade PnL)
  - Multi-account: round-robin trade assignment across N accounts; skip blown accounts

Bootstrap: sample whole trading days (with replacement) from the combined trade log.
Stop simulation at first payout (success) or all accounts blown (failure) or max-horizon reached.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
PLAN_DIR = HERE.parent.parent / "live" / "combined deployment plan"

# Prop firm spec
TRAILING_DD = 2000.0           # account blows at peak - $2k (capped at break-even after peak >= $2k)
PEAK_LOCK   = 2000.0           # peak threshold at which DD floor locks at $0
PAYOUT_AT   = 4000.0           # equity threshold to take payout
PAYOUT_AMT  = 2000.0           # withdraw $2k each time payout fires

# Simulation parameters
N_SIMS = 5000
HORIZONS = [30, 60, 90, 180, 365]  # trading-day horizons to report
MAX_HORIZON = max(HORIZONS)
BASE_CONTRACTS_GRID = list(range(1, 11))
ACCOUNTS_GRID = [1, 2, 3]


def load_daily_trades():
    """Returns list-of-lists: each entry = one trading day's list of NQ-1-contract PnL ($)."""
    df = pd.read_csv(RESULTS_DIR / "combined_3way_trades.csv")
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = df["exit_ts"].dt.date
    grouped = df.groupby("date")["pnl_$"].apply(list).tolist()
    n_trades_per_day = [len(d) for d in grouped]
    print(f"Loaded combined trade log: {len(df)} trades across {len(grouped)} trading days")
    print(f"  Trades/day: median={int(np.median(n_trades_per_day))}, "
          f"mean={np.mean(n_trades_per_day):.2f}, max={max(n_trades_per_day)}")
    return grouped


def simulate_once(daily_lists, n_accounts, base_contracts, rng):
    """Run one MC trial. Returns (success, day_of_first_payout, horizon_reached_days)."""
    n_data = len(daily_lists)
    eq = np.zeros(n_accounts)
    peak = np.zeros(n_accounts)
    blown = np.zeros(n_accounts, dtype=bool)
    payout_day = np.full(n_accounts, -1, dtype=np.int32)
    next_acc = 0
    scale = base_contracts * 0.1  # NQ$ -> MNQ$ at base_contracts

    for day_i in range(MAX_HORIZON):
        # Sample one historical trading day's trades
        idx = rng.integers(0, n_data)
        day_trades = daily_lists[idx]
        for tpnl_nq in day_trades:
            if blown.all():
                return (False, -1, day_i + 1)
            # rotate to next alive account
            tries = 0
            while blown[next_acc] and tries < n_accounts:
                next_acc = (next_acc + 1) % n_accounts
                tries += 1
            if blown.all():
                return (False, -1, day_i + 1)
            # apply MNQ-scaled trade
            mnq_pnl = tpnl_nq * scale
            eq[next_acc] += mnq_pnl
            if eq[next_acc] > peak[next_acc]:
                peak[next_acc] = eq[next_acc]
            # DD floor: peak - 2k, but capped at $0 (locked once peak >= 2k)
            if peak[next_acc] >= PEAK_LOCK:
                dd_floor = 0.0
            else:
                dd_floor = peak[next_acc] - TRAILING_DD
            # blowup check
            if eq[next_acc] <= dd_floor:
                blown[next_acc] = True
                next_acc = (next_acc + 1) % n_accounts
                continue
            # payout check (FIRST payout this account)
            if eq[next_acc] >= PAYOUT_AT and payout_day[next_acc] < 0:
                payout_day[next_acc] = day_i + 1
                eq[next_acc] -= PAYOUT_AMT
                # equity now = $2k, peak unchanged (so DD floor stays at $0)
                # account stays alive — we still count "first payout day" as success
            next_acc = (next_acc + 1) % n_accounts

        # End-of-day checks
        if (payout_day >= 0).any():
            first = int(payout_day[payout_day >= 0].min())
            return (True, first, day_i + 1)
        if blown.all():
            return (False, -1, day_i + 1)
    return (False, -1, MAX_HORIZON)


def run_grid(daily_lists):
    rng = np.random.default_rng(2026)
    rows = []
    for n_acc in ACCOUNTS_GRID:
        for bc in BASE_CONTRACTS_GRID:
            outcomes = []
            for _ in range(N_SIMS):
                success, day, _ = simulate_once(daily_lists, n_acc, bc, rng)
                outcomes.append((success, day))
            successes = [d for s, d in outcomes if s]
            n_success = len(successes)
            row = dict(
                n_accounts=n_acc, base_mnq=bc,
                payout_prob=n_success / N_SIMS,
                p_payout_by_30  = sum(1 for d in successes if d <= 30)  / N_SIMS,
                p_payout_by_60  = sum(1 for d in successes if d <= 60)  / N_SIMS,
                p_payout_by_90  = sum(1 for d in successes if d <= 90)  / N_SIMS,
                p_payout_by_180 = sum(1 for d in successes if d <= 180) / N_SIMS,
                p_payout_by_365 = sum(1 for d in successes if d <= 365) / N_SIMS,
                p_blowup        = sum(1 for s, d in outcomes if (not s) and d == -1) / N_SIMS,
                median_days=int(np.median(successes)) if successes else -1,
                p25_days=int(np.percentile(successes, 25)) if successes else -1,
                p75_days=int(np.percentile(successes, 75)) if successes else -1,
            )
            rows.append(row)
            print(f"  acc={n_acc} mnq={bc:>2}  "
                  f"payout_prob={row['payout_prob']:.3f}  "
                  f"median_days={row['median_days']:>3}  "
                  f"by_30={row['p_payout_by_30']:.2f} by_90={row['p_payout_by_90']:.2f} by_365={row['p_payout_by_365']:.2f}  "
                  f"P(blowup)={row['p_blowup']:.3f}")
    return rows


def report(rows, daily_lists):
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "propfirm_montecarlo.csv", index=False)
    df.to_csv(PLAN_DIR / "propfirm_montecarlo.csv", index=False)

    lines = []
    lines.append("=" * 120)
    lines.append("Prop-Firm Monte Carlo — 3-Strategy Combined Portfolio")
    lines.append("=" * 120)
    lines.append(f"Rules: $2,000 trailing drawdown (locks at break-even once peak >= $2,000)")
    lines.append(f"       Payout = withdraw $2,000 when equity >= $4,000")
    lines.append(f"       Risk: MNQ ($2/pt) at base size 1-10 (per-strategy martingale already baked in)")
    lines.append(f"       Multi-account: round-robin trade assignment, skip blown accounts")
    lines.append("")
    lines.append(f"Bootstrap: sample whole trading days with replacement from {len(daily_lists)} historical days")
    lines.append(f"  Sims per cell: {N_SIMS}   Max horizon: {MAX_HORIZON} trading days")
    lines.append("")
    for n_acc in ACCOUNTS_GRID:
        sub = df[df["n_accounts"] == n_acc]
        lines.append(f"--- {n_acc} ACCOUNT{'S' if n_acc>1 else ''} ---")
        lines.append(f"{'MNQ':>4} {'P(payout)':>10} {'P(blowup)':>10}  "
                      f"{'by_30d':>7} {'by_60d':>7} {'by_90d':>7} {'by_180d':>8} {'by_365d':>8}  "
                      f"{'days_p25':>9} {'days_med':>9} {'days_p75':>9}")
        for _, r in sub.iterrows():
            lines.append(f"{int(r['base_mnq']):>4} "
                          f"{r['payout_prob']*100:>9.1f}% "
                          f"{r['p_blowup']*100:>9.1f}%  "
                          f"{r['p_payout_by_30']*100:>6.1f}% "
                          f"{r['p_payout_by_60']*100:>6.1f}% "
                          f"{r['p_payout_by_90']*100:>6.1f}% "
                          f"{r['p_payout_by_180']*100:>7.1f}% "
                          f"{r['p_payout_by_365']*100:>7.1f}%  "
                          f"{int(r['p25_days']):>9} {int(r['median_days']):>9} {int(r['p75_days']):>9}")
        lines.append("")

    # Compare across accounts at each MNQ size
    lines.append("=" * 120)
    lines.append("Effect of adding accounts (payout probability within horizon)")
    lines.append("=" * 120)
    lines.append(f"{'MNQ':>4}   {'1 acc P':>9} {'2 acc P':>9} {'3 acc P':>9}   {'1 acc med':>10} {'2 acc med':>10} {'3 acc med':>10}")
    for bc in BASE_CONTRACTS_GRID:
        p1 = df[(df['n_accounts']==1)&(df['base_mnq']==bc)].iloc[0]
        p2 = df[(df['n_accounts']==2)&(df['base_mnq']==bc)].iloc[0]
        p3 = df[(df['n_accounts']==3)&(df['base_mnq']==bc)].iloc[0]
        lines.append(f"{bc:>4}   "
                      f"{p1['payout_prob']*100:>8.1f}% "
                      f"{p2['payout_prob']*100:>8.1f}% "
                      f"{p3['payout_prob']*100:>8.1f}%   "
                      f"{int(p1['median_days']):>10} "
                      f"{int(p2['median_days']):>10} "
                      f"{int(p3['median_days']):>10}")
    out_txt = RESULTS_DIR / "propfirm_montecarlo.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    (PLAN_DIR / "propfirm_montecarlo.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines[-120:]))
    print(f"\nWrote {out_txt}")


def main():
    daily = load_daily_trades()
    rows = run_grid(daily)
    report(rows, daily)


if __name__ == "__main__":
    main()
