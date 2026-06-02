"""Personal (non-prop) account scaling for the live 4-way combined.

Monte-Carlo bootstraps the 4-way daily P&L (combined_4way_trades.csv, OD/RV/B2/FB at 1 NQ = 10 MNQ)
to get, per MNQ size: annual P&L distribution + the distribution of intra-year max drawdown.
From the DD distribution we recommend a safe account size (survive a p99 bad year with margin)
and the resulting annual return.

No prop rules here — just risk-of-ruin / position sizing for your own capital.

Run:  python scripts/montecarlo/personal_account_scaling.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"
COST_RT_PER_MNQ = 4.0
OVERNIGHT_MARGIN_PER_MNQ = 1700.0   # ~broker MNQ overnight margin (adjust to yours)
N_SIMS = 20_000
YEAR_TD = 252
MNQ_GRID = [1, 2, 3, 5, 10, 15, 20, 30]


def load_daily():
    df = pd.read_csv(CSV)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True, format="mixed").dt.tz_convert("America/New_York")
    df["d"] = df["exit_ts"].dt.date
    g = df.groupby("d").agg(pnl=("pnl_$", "sum"), n=("pnl_$", "size"))
    return g["pnl"].values.astype(float), g["n"].values.astype(float)


def max_dd(equity_path):
    peak = np.maximum.accumulate(equity_path)
    return float((peak - equity_path).max())


def main():
    pnl1nq, ntr = load_daily()
    n = len(pnl1nq)
    # historical single-path stats at 1 NQ
    cum = np.cumsum(pnl1nq)
    hist_dd_1nq = max_dd(cum)
    print(f"4-way combined: {n} trading days @1NQ | mean ${pnl1nq.mean():.0f}/day "
          f"| worst day ${pnl1nq.min():.0f} | historical max DD ${hist_dd_1nq:,.0f} (1 NQ)")
    print(f"Costs ${COST_RT_PER_MNQ}/MNQ RT; overnight margin ${OVERNIGHT_MARGIN_PER_MNQ:.0f}/MNQ.\n")

    rng = np.random.default_rng(7)
    rows = []
    for mnq in MNQ_GRID:
        scale = mnq / 10.0
        ann_pnl = np.empty(N_SIMS)
        ann_dd = np.empty(N_SIMS)
        for i in range(N_SIMS):
            idx = rng.integers(0, n, YEAR_TD)
            day = pnl1nq[idx] * scale - ntr[idx] * COST_RT_PER_MNQ * mnq
            eq = np.cumsum(day)
            ann_pnl[i] = eq[-1]
            ann_dd[i] = max_dd(eq)
        dd99 = np.percentile(ann_dd, 99)
        # safe capital: cover a p99 bad-year drawdown at 2x (so worst year ~ -50% equity), or margin
        safe_cap = max(2.0 * dd99, OVERNIGHT_MARGIN_PER_MNQ * mnq * 1.5)
        rows.append({
            "mnq": mnq,
            "E[annual $]": round(ann_pnl.mean(), 0),
            "median annual $": round(np.median(ann_pnl), 0),
            "p10 annual $": round(np.percentile(ann_pnl, 10), 0),
            "yr maxDD p50": round(np.percentile(ann_dd, 50), 0),
            "yr maxDD p99": round(dd99, 0),
            "SAFE capital": round(safe_cap, 0),
            "annual return %": round(100 * ann_pnl.mean() / safe_cap, 0),
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    print(df.to_string(index=False))
    print("\nSAFE capital = max(2 x p99 annual maxDD, 1.5 x overnight margin). A p99 bad year then")
    print("draws ~50% of equity (survivable, keep trading). Scale to your risk tolerance:")
    print("  aggressive = 1x p99 DD (full-send), conservative = 3x p99 DD (shallow drawdowns).")
    out = ROOT / "scripts" / "montecarlo" / "results" / "personal_account_scaling.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
