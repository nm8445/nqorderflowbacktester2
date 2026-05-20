"""
Hola Prime $100k / $10k DD Monte Carlo WITH optional OD per-trade cutoff.

For each OD trade, if MAE_$ < -cutoff_NQ, the trade is force-closed at -cutoff_NQ.
This caps both MAE and realized PnL at -cutoff_NQ for those trades.

Compares: no_cutoff, $15k, $12.5k, $10k cutoffs (all in NQ basis $).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
PLAN_DIR = HERE.parent.parent / "live" / "combined deployment plan"
PLAN_DIR.mkdir(parents=True, exist_ok=True)

COMBINED_MAE = RESULTS_DIR / "combined_3way_trades_with_mae.csv"

ACCOUNT_SIZE = 100_000
MAX_DD       = 10_000
LOCK_PROFIT  = 5_000
DAILY_LOSS   = 5_000
TARGET_PCT   = 0.10

CUTOFFS = [None, 15_000, 12_500, 10_000]   # NQ basis $
MNQ_SIZES = [1, 2, 3, 4]
DD_MODES = ["trailing_to_initial", "static_from_initial"]
N_SIMS = 5000
MAX_HORIZON = 365


def load_daily_with_cutoff(cutoff_nq):
    df = pd.read_csv(COMBINED_MAE)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = df["exit_ts"].dt.date
    if cutoff_nq is not None:
        od = df["strat"] == "OD"
        capped = od & (df["mae_$"] < -cutoff_nq)
        df.loc[capped, "pnl_$"] = -cutoff_nq
        df.loc[capped, "mae_$"] = -cutoff_nq
    df = df.sort_values(["date", "entry_ts"])
    out = []
    for d, g in df.groupby("date", sort=True):
        items = list(zip(g["pnl_$"].astype(float), g["mae_$"].astype(float)))
        out.append(items)
    return out


def simulate_one(daily_lists, mnq, rng, dd_mode):
    scale = mnq * 0.1
    balance = ACCOUNT_SIZE; peak = ACCOUNT_SIZE; prev_eod = ACCOUNT_SIZE
    target = ACCOUNT_SIZE * (1 + TARGET_PCT)
    n_data = len(daily_lists)
    for day_i in range(MAX_HORIZON):
        idx = rng.integers(0, n_data)
        day_pnl = 0.0
        for pnl_nq, mae_nq in daily_lists[idx]:
            mae_d = mae_nq * scale
            pnl_d = pnl_nq * scale
            # MAE dip first
            eq_dip = balance + day_pnl + mae_d
            if eq_dip > peak: peak = eq_dip
            if dd_mode == "trailing_to_initial":
                dd_floor = max(ACCOUNT_SIZE, peak - MAX_DD) if peak >= ACCOUNT_SIZE + LOCK_PROFIT \
                            else (peak - MAX_DD)
            else:
                dd_floor = ACCOUNT_SIZE - MAX_DD
            if eq_dip <= dd_floor:
                return (False, day_i + 1, "max_dd_unrealized")
            if (prev_eod - eq_dip) >= DAILY_LOSS:
                return (False, day_i + 1, "daily_loss_unrealized")
            # Realized PnL
            day_pnl += pnl_d
            cur = balance + day_pnl
            if cur > peak: peak = cur
            if dd_mode == "trailing_to_initial":
                dd_floor = max(ACCOUNT_SIZE, peak - MAX_DD) if peak >= ACCOUNT_SIZE + LOCK_PROFIT \
                            else (peak - MAX_DD)
            else:
                dd_floor = ACCOUNT_SIZE - MAX_DD
            if cur <= dd_floor:
                return (False, day_i + 1, "max_dd_realized")
            if (prev_eod - cur) >= DAILY_LOSS:
                return (False, day_i + 1, "daily_loss_realized")
            if cur >= target:
                return (True, day_i + 1, "target")
        balance += day_pnl
        prev_eod = balance
    return (False, MAX_HORIZON, "horizon")


def run_cell(cutoff, mnq, dd_mode):
    daily_lists = load_daily_with_cutoff(cutoff)
    rng = np.random.default_rng(2026 + (cutoff or 0) + int(mnq * 10) + (0 if dd_mode == "trailing_to_initial" else 1))
    results = [simulate_one(daily_lists, mnq, rng, dd_mode) for _ in range(N_SIMS)]
    passes = [d for ok, d, _ in results if ok]
    fails = [r for ok, _, r in results if not ok]
    fr = pd.Series(fails).value_counts(normalize=True).to_dict() if fails else {}
    return dict(
        cutoff=cutoff if cutoff else "none",
        mnq=mnq, dd_mode=dd_mode,
        pass_rate=len(passes) / N_SIMS,
        median_days=int(np.median(passes)) if passes else -1,
        p25=int(np.percentile(passes, 25)) if passes else -1,
        p75=int(np.percentile(passes, 75)) if passes else -1,
        bust_dd_unreal=fr.get("max_dd_unrealized", 0.0) * (1 - len(passes) / N_SIMS),
        bust_dd_real=fr.get("max_dd_realized", 0.0) * (1 - len(passes) / N_SIMS),
        bust_daily_unreal=fr.get("daily_loss_unrealized", 0.0) * (1 - len(passes) / N_SIMS),
        bust_daily_real=fr.get("daily_loss_realized", 0.0) * (1 - len(passes) / N_SIMS),
        bust_horizon=fr.get("horizon", 0.0) * (1 - len(passes) / N_SIMS),
    )


def main():
    rows = []
    print(f"{'cutoff_NQ':>10} {'MNQ':>4} {'mode':<22}  {'pass%':>6}  {'med_d':>5}  "
          f"{'DDunreal%':>10} {'DDreal%':>8} {'DLunreal%':>10} {'DLreal%':>8} {'horiz%':>7}")
    for cutoff in CUTOFFS:
        for mnq in MNQ_SIZES:
            for dd_mode in DD_MODES:
                r = run_cell(cutoff, mnq, dd_mode)
                rows.append(r)
                print(f"{str(r['cutoff']):>10} {r['mnq']:>4} {r['dd_mode']:<22}  "
                      f"{r['pass_rate']*100:>5.1f}%  {r['median_days']:>5}  "
                      f"{r['bust_dd_unreal']*100:>9.2f}% {r['bust_dd_real']*100:>7.2f}% "
                      f"{r['bust_daily_unreal']*100:>9.2f}% {r['bust_daily_real']*100:>7.2f}% "
                      f"{r['bust_horizon']*100:>6.2f}%")
    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "holaprime_100k_with_od_cutoff.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")

    # Pivot to make easy reading
    print("\n\n=== PASS RATE COMPARISON (rows=cutoff, cols=MNQ+mode) ===\n")
    piv = df.pivot_table(values="pass_rate", index="cutoff", columns=["dd_mode", "mnq"], aggfunc="first")
    print((piv * 100).round(1))
    txt_path = PLAN_DIR / "holaprime_100k_with_od_cutoff.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Hola Prime $100k / $10k DD pass-rate MC with optional OD per-trade cutoff\n")
        f.write("=" * 90 + "\n\n")
        f.write("PASS RATE (% of 5000 sims passing 10% target before bust):\n\n")
        f.write((piv * 100).round(1).to_string() + "\n\n")
        f.write("\nDetail per cell:\n")
        f.write(df.to_string(index=False))
    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
