"""
Hola Prime 2-step Prime $100k — Monte Carlo with N=2 accounts alternating trades.
Goal: find MNQ size that balances pass probability vs time-to-pass.

Multi-account: trades round-robin to next account; skip accounts that already
failed or are between phases. Pass = at least 1 account completes BOTH phases.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
PLAN_DIR = HERE.parent.parent / "live" / "combined deployment plan"

BALANCE_INIT = 100_000.0
P1_TARGET_PCT = 0.08
P2_TARGET_PCT = 0.05
DAILY_LOSS_PCT = 0.05
MAX_DD_PCT = 0.08
MAX_HORIZON = 365
N_SIMS = 2000
MNQ_SIZES = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
N_ACCOUNTS_LIST = [1, 2, 3]


def load_daily():
    df = pd.read_csv(RESULTS_DIR / "combined_3way_trades.csv")
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = df["exit_ts"].dt.date
    return df.groupby("date")["pnl_$"].apply(list).tolist()


def simulate_multi(daily_lists, n_accounts, mnq_size, rng):
    """Round-robin trades across N accounts each running 2-step Prime.
    Returns (any_passed, day_first_pass, n_passed, per_account_status)."""
    scale = mnq_size * 0.1
    p1_target = BALANCE_INIT * (1 + P1_TARGET_PCT)
    p2_target = BALANCE_INIT * (1 + P2_TARGET_PCT)
    dd_floor = BALANCE_INIT * (1 - MAX_DD_PCT)

    # per-account state
    phase = np.ones(n_accounts, dtype=np.int8)            # 1 or 2
    balance = np.full(n_accounts, BALANCE_INIT, dtype=np.float64)
    prev_eod = np.full(n_accounts, BALANCE_INIT, dtype=np.float64)
    failed = np.zeros(n_accounts, dtype=bool)
    passed = np.zeros(n_accounts, dtype=bool)
    pass_day = np.full(n_accounts, -1, dtype=np.int32)

    next_acc = 0
    n_data = len(daily_lists)

    for day_i in range(MAX_HORIZON):
        idx = rng.integers(0, n_data)
        # accumulate per-account intraday equity-change
        day_pnl = np.zeros(n_accounts)
        for tpnl in daily_lists[idx]:
            if (failed | passed).all():
                break
            tries = 0
            while (failed[next_acc] or passed[next_acc]) and tries < n_accounts:
                next_acc = (next_acc + 1) % n_accounts
                tries += 1
            if (failed | passed).all():
                break
            # apply this trade
            pnl = tpnl * scale
            day_pnl[next_acc] += pnl
            cur_eq = balance[next_acc] + day_pnl[next_acc]
            # max-DD (static from initial each phase, since balance resets between phases)
            if cur_eq <= dd_floor:
                failed[next_acc] = True
                next_acc = (next_acc + 1) % n_accounts
                continue
            # daily loss check (intraday)
            if (prev_eod[next_acc] - cur_eq) >= prev_eod[next_acc] * DAILY_LOSS_PCT:
                failed[next_acc] = True
                next_acc = (next_acc + 1) % n_accounts
                continue
            # phase target hit
            target = p1_target if phase[next_acc] == 1 else p2_target
            if cur_eq >= target:
                if phase[next_acc] == 1:
                    # advance to phase 2 — reset balance + EOD baseline
                    phase[next_acc] = 2
                    balance[next_acc] = BALANCE_INIT
                    prev_eod[next_acc] = BALANCE_INIT
                    day_pnl[next_acc] = 0.0  # remaining day P&L doesn't carry into phase 2
                else:
                    passed[next_acc] = True
                    pass_day[next_acc] = day_i + 1
            next_acc = (next_acc + 1) % n_accounts
        # end-of-day update for non-passed/non-failed accounts
        for a in range(n_accounts):
            if not failed[a] and not passed[a]:
                balance[a] += day_pnl[a]
                prev_eod[a] = balance[a]
        # check stop conditions
        if passed.any():
            first = int(pass_day[passed].min())
            return (True, first, int(passed.sum()), phase.copy())
        if (failed | passed).all():
            return (False, -1, 0, phase.copy())
    return (False, -1, 0, phase.copy())


def run_grid(daily_lists):
    import time
    rng = np.random.default_rng(2026)
    rows = []
    total_cells = len(N_ACCOUNTS_LIST) * len(MNQ_SIZES)
    cell_i = 0
    t0 = time.time()
    for n_acc in N_ACCOUNTS_LIST:
        for mnq in MNQ_SIZES:
            cell_i += 1
            ct0 = time.time()
            results = []
            for _ in range(N_SIMS):
                results.append(simulate_multi(daily_lists, n_acc, mnq, rng))
            passes = [d for ok, d, _, _ in results if ok]
            pass_rate = len(passes) / N_SIMS
            row = dict(
                n_accounts=n_acc, mnq=mnq, nas_lots=mnq*2,
                pass_rate=pass_rate,
                median_days=int(np.median(passes)) if passes else -1,
                p25=int(np.percentile(passes, 25)) if passes else -1,
                p75=int(np.percentile(passes, 75)) if passes else -1,
                p10=int(np.percentile(passes, 10)) if passes else -1,
                p90=int(np.percentile(passes, 90)) if passes else -1,
            )
            rows.append(row)
            dt = time.time() - ct0
            elapsed = time.time() - t0
            eta = elapsed / cell_i * (total_cells - cell_i)
            print(f"  [{cell_i}/{total_cells}] acc={n_acc} mnq={mnq:>4}  "
                  f"pass={row['pass_rate']*100:>5.1f}%  median={row['median_days']:>3}d  "
                  f"({dt:.1f}s, ETA {eta:.0f}s)", flush=True)
    return rows


def main():
    daily = load_daily()
    rows = run_grid(daily)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "holaprime_100k_2step_multiacc.csv", index=False)
    df.to_csv(PLAN_DIR / "holaprime_100k_2step_multiacc.csv", index=False)

    # Score: balance of pass rate and speed
    # MAR-like: pass_rate / (median_days / 100) — higher better
    # Goal: high pass AND low median days
    df["speed_score"] = df["pass_rate"] / (df["median_days"].clip(lower=1) / 100.0)
    df["balanced_score"] = (df["pass_rate"] * 0.6) + (1 - df["median_days"].clip(lower=1)/MAX_HORIZON) * 0.4

    lines = []
    lines.append("=" * 130)
    lines.append("Hola Prime $100k 2-step Prime — Multi-account Monte Carlo")
    lines.append("=" * 130)
    lines.append("Rules: Phase 1 8% target, Phase 2 5% target, 5% daily loss, 8% static DD per phase")
    lines.append("Account: $100,000.  N accounts alternate trades round-robin (skip already-passed/failed).")
    lines.append("PASS = at least 1 account completes BOTH phases.  Days = until first pass.")
    lines.append("1 MNQ = 2 lots NAS100 (assuming $1/pt per lot, MT5 retail standard).")
    lines.append(f"N_SIMS={N_SIMS}  max_horizon={MAX_HORIZON} trading days")
    lines.append("")

    for n_acc in N_ACCOUNTS_LIST:
        lines.append(f"--- {n_acc} ACCOUNT{'S' if n_acc>1 else ''} ---")
        sub = df[df["n_accounts"] == n_acc].sort_values("mnq")
        lines.append(f"{'MNQ':>4} {'NAS lots':>8}  {'Pass %':>7}  "
                      f"{'p10':>4} {'p25':>4} {'median':>6} {'p75':>4} {'p90':>4}  "
                      f"{'speed':>6} {'balanced':>8}")
        for _, r in sub.iterrows():
            lines.append(f"{r['mnq']:>4.1f} {r['nas_lots']:>8.1f}  {r['pass_rate']*100:>6.1f}%  "
                          f"{int(r['p10']):>4} {int(r['p25']):>4} {int(r['median_days']):>6} "
                          f"{int(r['p75']):>4} {int(r['p90']):>4}  "
                          f"{r['speed_score']:>6.2f} {r['balanced_score']:>8.3f}")
        lines.append("")

    # Compare 1 vs 2 vs 3 at each MNQ
    lines.append("=" * 130)
    lines.append("Side-by-side at each MNQ: 1 acc | 2 acc | 3 acc")
    lines.append("=" * 130)
    lines.append(f"{'MNQ':>4}   {'1-acc pass':>10} {'1-acc med':>9}  |  "
                  f"{'2-acc pass':>10} {'2-acc med':>9}  |  "
                  f"{'3-acc pass':>10} {'3-acc med':>9}")
    for mnq in MNQ_SIZES:
        a1 = df[(df["n_accounts"]==1) & (df["mnq"]==mnq)].iloc[0]
        a2 = df[(df["n_accounts"]==2) & (df["mnq"]==mnq)].iloc[0]
        a3 = df[(df["n_accounts"]==3) & (df["mnq"]==mnq)].iloc[0]
        lines.append(f"{mnq:>4.1f}   "
                      f"{a1['pass_rate']*100:>9.1f}% {int(a1['median_days']):>9}d  |  "
                      f"{a2['pass_rate']*100:>9.1f}% {int(a2['median_days']):>9}d  |  "
                      f"{a3['pass_rate']*100:>9.1f}% {int(a3['median_days']):>9}d")
    lines.append("")

    # Top by balanced score per n_acc
    lines.append("=" * 130)
    lines.append("Top balanced setups (pass-rate × speed weighted)")
    lines.append("=" * 130)
    for n_acc in N_ACCOUNTS_LIST:
        sub = df[df["n_accounts"] == n_acc].copy()
        # Filter to plausible (pass rate >= 70%, median <= 90 days)
        plausible = sub[(sub["pass_rate"] >= 0.70) & (sub["median_days"] <= 120) & (sub["median_days"] > 0)]
        if len(plausible) == 0:
            plausible = sub
        plausible = plausible.sort_values("balanced_score", ascending=False)
        lines.append(f"\n{n_acc} account(s) — top 3 balanced:")
        for _, r in plausible.head(3).iterrows():
            lines.append(f"  MNQ={r['mnq']:>4.1f} (NAS100 {r['nas_lots']:>4.1f} lots): "
                          f"pass {r['pass_rate']*100:>5.1f}%   median {int(r['median_days'])}d   "
                          f"p25/p75 {int(r['p25'])}/{int(r['p75'])}d")

    out = RESULTS_DIR / "holaprime_100k_2step_multiacc.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    (PLAN_DIR / "holaprime_100k_2step_multiacc.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
