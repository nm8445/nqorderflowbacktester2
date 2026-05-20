"""
Hola Prime Monte Carlo — pass rate for 1-step / 2-step Prime forex accounts
trading NAS100 with the 3-strategy combined portfolio.

Rules (per Hola Prime Prime forex):
  1-Step Prime:
    - Profit target: 10%
    - Daily loss limit: 3% of prev EOD balance (account fails if intraday loss exceeds)
    - Max overall drawdown: 6% trailing from peak; locks at INITIAL balance once peak >= +5%
    - No min/max trading days

  2-Step Prime:
    - Phase 1 target: 8%
    - Phase 2 target: 5%  (account resets to starting balance after passing P1)
    - Daily loss limit: 5%
    - Max overall drawdown: 8% static from initial balance (each phase independently)
    - Sequential: must pass P1, then P2

NAS100 contract assumption: 1 lot NAS100 = $1/pt (MT5 retail standard).
  1 MNQ = $2/pt => 1 MNQ-equivalent = 2 NAS100 lots
  Combined trade log stores pnl_$ at NQ basis ($20/pt). For MNQ_size N:
    pnl_per_trade = pnl_$_NQ * (N / 10)   # since 1 NQ = 10 MNQ

Account sizes: $50k, $100k.
Risk slider: MNQ_size 0.5, 1, 2, 3, ..., 10.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
PLAN_DIR = HERE.parent.parent / "live" / "combined deployment plan"

ACCOUNT_SIZES = [50_000, 100_000]
MNQ_SIZES = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
N_SIMS = 5000
MAX_HORIZON = 365  # trading days


def load_daily():
    df = pd.read_csv(RESULTS_DIR / "combined_3way_trades.csv")
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = df["exit_ts"].dt.date
    # Build daily lists of NQ-basis pnl (at 1 NQ contract)
    return df.groupby("date")["pnl_$"].apply(list).tolist()


def simulate_1step(daily_lists, balance_init, mnq_size, rng,
                   target_pct=0.10, daily_loss_pct=0.03,
                   max_dd_pct=0.06, lock_at_profit_pct=0.05):
    """Returns (passed, days_used, fail_reason)."""
    scale = mnq_size * 0.1  # NQ$ -> MNQ$ at mnq_size contracts
    target = balance_init * (1 + target_pct)
    max_dd_abs = balance_init * max_dd_pct
    lock_threshold = balance_init * (1 + lock_at_profit_pct)
    daily_loss_pct_v = daily_loss_pct

    balance = balance_init
    peak = balance_init
    prev_eod = balance_init
    n_data = len(daily_lists)

    for day_i in range(MAX_HORIZON):
        idx = rng.integers(0, n_data)
        day_pnl_sum = 0.0
        intraday_min = balance  # track lowest equity during day for daily-loss check
        for tpnl in daily_lists[idx]:
            day_pnl_sum += tpnl * scale
            cur_eq = balance + day_pnl_sum
            if cur_eq < intraday_min:
                intraday_min = cur_eq
            # check max DD intraday
            if cur_eq > peak:
                peak = cur_eq
            if peak >= lock_threshold:
                dd_floor = balance_init
            else:
                dd_floor = peak - max_dd_abs
            if cur_eq <= dd_floor:
                return (False, day_i + 1, "max_dd")
            # check daily loss intraday
            if (prev_eod - cur_eq) >= prev_eod * daily_loss_pct_v:
                return (False, day_i + 1, "daily_loss")
            # check target hit (still apply daily and DD first)
            if cur_eq >= target:
                return (True, day_i + 1, "target")
        balance += day_pnl_sum
        prev_eod = balance
    return (False, MAX_HORIZON, "horizon")


def simulate_2step(daily_lists, balance_init, mnq_size, rng,
                    p1_target_pct=0.08, p2_target_pct=0.05,
                    daily_loss_pct=0.05, max_dd_pct=0.08):
    """Returns (passed_both, total_days, fail_reason, phase_failed)."""
    scale = mnq_size * 0.1
    n_data = len(daily_lists)
    total_days = 0
    fail_phase = 0

    for phase, target_pct in [(1, p1_target_pct), (2, p2_target_pct)]:
        balance = balance_init
        target = balance_init * (1 + target_pct)
        max_dd_abs = balance_init * max_dd_pct
        prev_eod = balance_init
        # static DD from initial balance
        dd_floor = balance_init - max_dd_abs
        passed_phase = False
        for day_i in range(MAX_HORIZON - total_days):
            idx = rng.integers(0, n_data)
            day_pnl_sum = 0.0
            for tpnl in daily_lists[idx]:
                day_pnl_sum += tpnl * scale
                cur_eq = balance + day_pnl_sum
                if cur_eq <= dd_floor:
                    return (False, total_days + day_i + 1, "max_dd", phase)
                if (prev_eod - cur_eq) >= prev_eod * daily_loss_pct:
                    return (False, total_days + day_i + 1, "daily_loss", phase)
                if cur_eq >= target:
                    total_days += day_i + 1
                    passed_phase = True
                    break
            if passed_phase:
                break
            balance += day_pnl_sum
            prev_eod = balance
        if not passed_phase:
            return (False, total_days + (MAX_HORIZON - total_days), "horizon", phase)
    return (True, total_days, "target", 0)


def run_grid(daily_lists):
    rng = np.random.default_rng(2026)
    rows = []
    for size in ACCOUNT_SIZES:
        for mnq in MNQ_SIZES:
            # 1-step
            results = [simulate_1step(daily_lists, size, mnq, rng) for _ in range(N_SIMS)]
            passes = [d for ok, d, _ in results if ok]
            fail_reasons = pd.Series([r for ok, _, r in results if not ok]).value_counts(normalize=True).to_dict()
            pass_rate = len(passes) / N_SIMS
            row1 = dict(
                account=size, mnq=mnq, type="1-step",
                pass_rate=pass_rate,
                median_days=int(np.median(passes)) if passes else -1,
                p25=int(np.percentile(passes, 25)) if passes else -1,
                p75=int(np.percentile(passes, 75)) if passes else -1,
                fail_max_dd=fail_reasons.get("max_dd", 0.0),
                fail_daily=fail_reasons.get("daily_loss", 0.0),
                fail_horizon=fail_reasons.get("horizon", 0.0),
            )
            rows.append(row1)
            # 2-step
            results = [simulate_2step(daily_lists, size, mnq, rng) for _ in range(N_SIMS)]
            passes = [d for ok, d, _, _ in results if ok]
            fail_reasons = pd.Series([r for ok, _, r, _ in results if not ok]).value_counts(normalize=True).to_dict()
            pass_rate = len(passes) / N_SIMS
            row2 = dict(
                account=size, mnq=mnq, type="2-step",
                pass_rate=pass_rate,
                median_days=int(np.median(passes)) if passes else -1,
                p25=int(np.percentile(passes, 25)) if passes else -1,
                p75=int(np.percentile(passes, 75)) if passes else -1,
                fail_max_dd=fail_reasons.get("max_dd", 0.0),
                fail_daily=fail_reasons.get("daily_loss", 0.0),
                fail_horizon=fail_reasons.get("horizon", 0.0),
            )
            rows.append(row2)
            print(f"  ${size//1000}k {('1-step ' if row1['type']=='1-step' else '2-step')} MNQ={mnq:>4}  "
                  f"pass={row1['pass_rate']*100:>5.1f}%  median_days={row1['median_days']:>3}  | "
                  f"2-step pass={row2['pass_rate']*100:>5.1f}%  median_days={row2['median_days']:>3}")
    return rows


def report(rows):
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "holaprime_montecarlo.csv", index=False)
    df.to_csv(PLAN_DIR / "holaprime_montecarlo.csv", index=False)

    lines = []
    lines.append("=" * 140)
    lines.append("Hola Prime Pass-Rate Monte Carlo — 3-strategy combined portfolio trading NAS100")
    lines.append("=" * 140)
    lines.append("Rules: 1-step Prime — 10% target, 3% daily, 6% trailing DD (locks at initial after +5%)")
    lines.append("       2-step Prime — Phase 1 8% target + Phase 2 5% target, 5% daily, 8% static DD per phase")
    lines.append("Lot conversion: 1 MNQ = 2 lots NAS100 (assuming Hola Prime NAS100 = $1/pt per lot, MT5 standard)")
    lines.append(f"  Bootstrap: 5,000 sims per cell from {1354} historical days  |  Max horizon: 365 trading days")
    lines.append("")
    for size in ACCOUNT_SIZES:
        for typ in ("1-step", "2-step"):
            lines.append(f"--- ${size:,} {typ.upper()} Prime ---")
            sub = df[(df["account"] == size) & (df["type"] == typ)]
            lines.append(f"{'MNQ':>4} {'NAS100 lots':>11}  {'Pass %':>7}  {'days p25':>9} {'days med':>9} {'days p75':>9}  "
                          f"{'fail_DD':>8} {'fail_daily':>11} {'fail_horizon':>13}")
            for _, r in sub.iterrows():
                lines.append(f"{r['mnq']:>4} {r['mnq']*2:>11.1f}  {r['pass_rate']*100:>6.1f}%  "
                              f"{int(r['p25']):>9} {int(r['median_days']):>9} {int(r['p75']):>9}  "
                              f"{r['fail_max_dd']*100:>7.1f}% {r['fail_daily']*100:>10.1f}% "
                              f"{r['fail_horizon']*100:>12.1f}%")
            lines.append("")

    # Best-of summary
    lines.append("=" * 140)
    lines.append("Best risk levels per account x type")
    lines.append("=" * 140)
    for size in ACCOUNT_SIZES:
        for typ in ("1-step", "2-step"):
            sub = df[(df["account"] == size) & (df["type"] == typ)].sort_values("pass_rate", ascending=False)
            top = sub.head(3)
            lines.append(f"\n${size:,} {typ.upper()} — top by pass rate:")
            for _, r in top.iterrows():
                lines.append(f"  MNQ={r['mnq']:>4} (NAS100 {r['mnq']*2:>4.1f} lots): "
                              f"pass {r['pass_rate']*100:>5.1f}%   median {int(r['median_days']):>3}d   "
                              f"p25/p75 {int(r['p25'])}/{int(r['p75'])}d")
            # also fastest
            valid = sub[sub["pass_rate"] >= 0.30]
            if len(valid) > 0:
                fastest = valid.sort_values("median_days").head(1).iloc[0]
                lines.append(f"  Fastest with pass-rate >= 30%:  MNQ={fastest['mnq']} "
                              f"({fastest['mnq']*2:.1f} NAS100 lots): "
                              f"pass {fastest['pass_rate']*100:.1f}%, median {int(fastest['median_days'])}d")

    out_txt = RESULTS_DIR / "holaprime_montecarlo.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    (PLAN_DIR / "holaprime_montecarlo.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nWrote {out_txt}")


def main():
    daily = load_daily()
    print(f"Loaded {len(daily)} trading days")
    rows = run_grid(daily)
    report(rows)


if __name__ == "__main__":
    main()
