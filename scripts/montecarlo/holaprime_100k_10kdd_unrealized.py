"""
Hola Prime Monte Carlo with UNREALIZED-DD tracking for 100k account / $10k max DD.

Same daily-bootstrap structure as holaprime_montecarlo.py, but now during each
day we apply each trade's MAE (worst unrealized loss in $) BEFORE applying its
realized PnL. If the equity at MAE dip breaches the DD floor, the account fails
even if the trade ultimately closed green.

Rules (parameterized for the user's account):
  Account size:    $100k
  Max DD:          $10k (10%) — trailing from peak with lock at initial after +5%
                   (we also report the static-from-initial variant for comparison)
  Daily loss:      $5k (5%) — applied to intraday equity vs prior EOD balance

Risk slider: MNQ contracts 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
  combined log is at NQ basis ($20/pt). MNQ multiplier = mnq_size * 0.1.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"

COMBINED_MAE = RESULTS_DIR / "combined_3way_trades_with_mae.csv"

ACCOUNT_SIZE = 100_000
MAX_DD       = 10_000     # $10k
LOCK_PROFIT  = 5_000      # $5k (lock DD floor at initial balance after this profit)
DAILY_LOSS   = 5_000      # $5k (5%)
TARGET_PCT   = 0.10       # 10% profit target (typical 1-step)

MNQ_SIZES = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
N_SIMS = 5000
MAX_HORIZON = 365

# We compare three DD modes:
DD_MODES = ["trailing_to_initial", "static_from_initial"]


def load_daily():
    df = pd.read_csv(COMBINED_MAE)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = df["exit_ts"].dt.date
    # Group: list of (pnl_$, mae_$) tuples per day
    out = []
    for d, g in df.groupby("date"):
        items = list(zip(g["pnl_$"].astype(float), g["mae_$"].astype(float)))
        out.append(items)
    return out


def simulate_one(daily_lists, balance_init, mnq_size, rng,
                  target_pct, max_dd_abs, lock_profit, daily_loss_abs,
                  dd_mode):
    """Single Monte Carlo run. Returns (passed, days_used, fail_reason)."""
    scale = mnq_size * 0.1  # NQ$ -> MNQ$
    balance = balance_init
    peak = balance_init
    prev_eod = balance_init
    target = balance_init * (1 + target_pct)
    n_data = len(daily_lists)

    for day_i in range(MAX_HORIZON):
        idx = rng.integers(0, n_data)
        day_pnl_so_far = 0.0
        for pnl_nq, mae_nq in daily_lists[idx]:
            mae_d = mae_nq * scale       # MAE in MNQ $
            pnl_d = pnl_nq * scale       # realized PnL in MNQ $

            # 1) Apply MAE first — unrealized dip during the trade
            eq_dip = balance + day_pnl_so_far + mae_d
            if eq_dip > peak:
                peak = eq_dip
            if dd_mode == "trailing_to_initial":
                dd_floor = max(balance_init, peak - max_dd_abs) if peak >= balance_init + lock_profit \
                            else (peak - max_dd_abs)
            else:  # static_from_initial
                dd_floor = balance_init - max_dd_abs
            if eq_dip <= dd_floor:
                return (False, day_i + 1, "max_dd_unrealized")
            # daily loss check at MAE dip
            if (prev_eod - eq_dip) >= daily_loss_abs:
                return (False, day_i + 1, "daily_loss_unrealized")

            # 2) Apply realized PnL
            day_pnl_so_far += pnl_d
            cur_eq = balance + day_pnl_so_far
            if cur_eq > peak:
                peak = cur_eq
            if dd_mode == "trailing_to_initial":
                dd_floor = max(balance_init, peak - max_dd_abs) if peak >= balance_init + lock_profit \
                            else (peak - max_dd_abs)
            else:
                dd_floor = balance_init - max_dd_abs
            if cur_eq <= dd_floor:
                return (False, day_i + 1, "max_dd_realized")
            if (prev_eod - cur_eq) >= daily_loss_abs:
                return (False, day_i + 1, "daily_loss_realized")
            if cur_eq >= target:
                return (True, day_i + 1, "target")

        balance += day_pnl_so_far
        prev_eod = balance

    return (False, MAX_HORIZON, "horizon")


def run_grid(daily_lists):
    rng = np.random.default_rng(2026)
    rows = []
    for dd_mode in DD_MODES:
        for mnq in MNQ_SIZES:
            results = [simulate_one(daily_lists, ACCOUNT_SIZE, mnq, rng,
                                     TARGET_PCT, MAX_DD, LOCK_PROFIT, DAILY_LOSS,
                                     dd_mode) for _ in range(N_SIMS)]
            passes = [d for ok, d, _ in results if ok]
            fails = [r for ok, _, r in results if not ok]
            fr = pd.Series(fails).value_counts(normalize=True).to_dict() if fails else {}
            row = dict(
                mnq=mnq, dd_mode=dd_mode,
                pass_rate=len(passes) / N_SIMS,
                median_days=int(np.median(passes)) if passes else -1,
                p25=int(np.percentile(passes, 25)) if passes else -1,
                p75=int(np.percentile(passes, 75)) if passes else -1,
                fail_dd_unreal=fr.get("max_dd_unrealized", 0.0),
                fail_dd_real=fr.get("max_dd_realized", 0.0),
                fail_daily_unreal=fr.get("daily_loss_unrealized", 0.0),
                fail_daily_real=fr.get("daily_loss_realized", 0.0),
                fail_horizon=fr.get("horizon", 0.0),
            )
            rows.append(row)
            print(f"  {dd_mode:<22}  MNQ={mnq:>4}  pass={row['pass_rate']*100:>5.1f}%  "
                   f"med_days={row['median_days']:>3}  "
                   f"DDfail unreal={row['fail_dd_unreal']*100:>4.1f}% real={row['fail_dd_real']*100:>4.1f}%  "
                   f"dailyfail unreal={row['fail_daily_unreal']*100:>4.1f}% real={row['fail_daily_real']*100:>4.1f}%")
    return rows


def report(rows):
    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "holaprime_100k_10kdd_unrealized.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")

    lines = []
    lines.append("=" * 120)
    lines.append(f"Hola Prime Monte Carlo — ${ACCOUNT_SIZE:,} account / ${MAX_DD:,} max DD")
    lines.append("with UNREALIZED-DD tracking (MAE applied per trade before realized PnL)")
    lines.append("=" * 120)
    lines.append(f"  3-way combined: OD + RV + B2  (1357+806+619 = 2782 trades over 5+ yr)")
    lines.append(f"  Daily loss: ${DAILY_LOSS:,} (5%) | Target: {TARGET_PCT*100:.0f}% | Sims: {N_SIMS} per cell")
    lines.append("")
    for dd_mode in DD_MODES:
        lines.append(f"\n=== DD mode: {dd_mode} ===")
        if dd_mode == "trailing_to_initial":
            lines.append(f"  (trails {MAX_DD/ACCOUNT_SIZE*100:.0f}% from peak; floor locks at initial balance after +${LOCK_PROFIT:,} profit)")
        else:
            lines.append(f"  (static {MAX_DD/ACCOUNT_SIZE*100:.0f}% from initial balance, no trailing relief)")
        lines.append("")
        lines.append(f"{'MNQ':>4} {'NAS lots':>9}  {'Pass %':>7}  {'days p25':>9} {'days med':>9} {'days p75':>9}  "
                      f"{'DDunreal':>9} {'DDreal':>9} {'DAILYunreal':>11} {'DAILYreal':>10} {'horizon':>8}")
        sub = df[df["dd_mode"] == dd_mode]
        for _, r in sub.iterrows():
            lines.append(f"{r['mnq']:>4} {r['mnq']*2:>9.1f}  {r['pass_rate']*100:>6.1f}%  "
                          f"{int(r['p25']):>9} {int(r['median_days']):>9} {int(r['p75']):>9}  "
                          f"{r['fail_dd_unreal']*100:>8.1f}% {r['fail_dd_real']*100:>8.1f}% "
                          f"{r['fail_daily_unreal']*100:>10.1f}% {r['fail_daily_real']*100:>9.1f}% "
                          f"{r['fail_horizon']*100:>7.1f}%")

    # Best size summary
    lines.append("\n" + "=" * 120)
    lines.append("Top sizes per DD mode (by pass rate)")
    lines.append("=" * 120)
    for dd_mode in DD_MODES:
        sub = df[df["dd_mode"] == dd_mode].sort_values("pass_rate", ascending=False)
        top = sub.head(3)
        lines.append(f"\n{dd_mode}:")
        for _, r in top.iterrows():
            lines.append(f"  MNQ={r['mnq']:>4} ({r['mnq']*2:>4.1f} NAS lots): "
                          f"pass {r['pass_rate']*100:>5.1f}%   median {int(r['median_days']):>3}d   "
                          f"p25/p75 {int(r['p25'])}/{int(r['p75'])}d")

    txt = "\n".join(lines)
    out_txt = RESULTS_DIR / "holaprime_100k_10kdd_unrealized.txt"
    out_txt.write_text(txt, encoding="utf-8")
    plan_path = HERE.parent.parent / "live" / "combined deployment plan" / "holaprime_100k_10kdd_unrealized.txt"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(txt, encoding="utf-8")
    print("\n" + txt)
    print(f"\nWrote {out_txt}")


def main():
    daily = load_daily()
    print(f"Loaded {len(daily)} trading days\n")
    rows = run_grid(daily)
    report(rows)


if __name__ == "__main__":
    main()
