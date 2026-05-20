"""
Combine:
  - Rough vol v3 (N=400) with martingale streak=1 mult=1.5 maxd=2 any_loss
  - Overnight drift locked (martingale already baked into trade log)
And plot a combined equity curve.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
ROOT = HERE.parent.parent

# ---------- Rough vol v3 with martingale streak=1 mult=1.5 maxd=2 any_loss ----------
rv = pd.read_csv(RESULTS_DIR / "inspect_v3_FULL_log.csv")
rv["entry_ts"] = pd.to_datetime(rv["entry_ts"], utc=True).dt.tz_convert("America/New_York")
rv["exit_ts"] = pd.to_datetime(rv["exit_ts"], utc=True).dt.tz_convert("America/New_York")
rv = rv.sort_values("entry_ts").reset_index(drop=True)

def apply_mart(df, streak=1, mult=1.5, max_doubles=2):
    qty = 1
    loss_streak = 0
    sized = np.zeros(len(df))
    qtys = np.zeros(len(df), dtype=np.int32)
    for i, r in enumerate(df.itertuples()):
        sized[i] = r.pnl_dollars * qty
        qtys[i] = qty
        if r.pnl_dollars < 0:
            loss_streak += 1
        else:
            loss_streak = 0
        if loss_streak >= streak:
            steps = min(loss_streak - streak + 1, max_doubles)
            qty = max(1, int(round(mult ** steps)))
        else:
            qty = 1
    return sized, qtys

rv_sized, rv_qty = apply_mart(rv, streak=1, mult=1.5, max_doubles=2)
rv["sized_pnl"] = rv_sized
rv["qty"] = rv_qty
print(f"Rough vol (v3 + s1m1.5d2 mart): {len(rv)} trades  PnL ${rv['sized_pnl'].sum():+,.0f}  "
      f"max_qty={rv_qty.max()} avg_qty={rv_qty.mean():.2f}")

# ---------- Overnight drift (martingale already applied) ----------
od = pd.read_csv(ROOT / "live" / "overnight drift" / "trades.csv")
od["entry_ts"] = pd.to_datetime(od["entry_time"], utc=True).dt.tz_convert("America/New_York")
od["exit_ts"] = pd.to_datetime(od["exit_time"], utc=True).dt.tz_convert("America/New_York")
od = od.sort_values("entry_ts").reset_index(drop=True)
od["sized_pnl"] = od["pnl_dollars"]
print(f"Overnight drift: {len(od)} trades  PnL ${od['sized_pnl'].sum():+,.0f}")

# ---------- Combined ----------
rv_c = rv[["entry_ts", "exit_ts", "sized_pnl"]].copy()
rv_c["strat"] = "RoughVol_v3"
od_c = od[["entry_ts", "exit_ts", "sized_pnl"]].copy()
od_c["strat"] = "OvernightDrift"

combined = pd.concat([rv_c, od_c], ignore_index=True).sort_values("exit_ts").reset_index(drop=True)
print(f"Combined: {len(combined)} trades")

combined["cum_total"] = combined["sized_pnl"].cumsum()

# Per-strat cumulative for chart
rv_cs = rv_c.sort_values("exit_ts").reset_index(drop=True)
rv_cs["cum"] = rv_cs["sized_pnl"].cumsum()
od_cs = od_c.sort_values("exit_ts").reset_index(drop=True)
od_cs["cum"] = od_cs["sized_pnl"].cumsum()

# Combined DD
combined["peak"] = combined["cum_total"].cummax()
combined["dd"] = combined["cum_total"] - combined["peak"]

# ---------- Stats ----------
def stats(p, label):
    w = p[p > 0]; l = p[p < 0]
    pf = w.sum() / abs(l.sum()) if len(l) else 99
    wr = 100 * len(w) / len(p) if len(p) else 0
    cum = p.cumsum()
    mdd = (cum - np.maximum.accumulate(cum)).min()
    print(f"{label:>30}: {len(p):>5}t  PF {pf:.2f}  WR {wr:.1f}%  PnL ${p.sum():+,.0f}  MDD ${mdd:+,.0f}  MAR {p.sum()/abs(mdd):.2f}")

print()
stats(rv["sized_pnl"].to_numpy(), "Rough vol v3 + mart")
stats(od["sized_pnl"].to_numpy(), "Overnight drift")
stats(combined["sized_pnl"].to_numpy(), "COMBINED")

# Year-by-year combined
combined["year"] = combined["exit_ts"].dt.year
print("\nYear-by-year COMBINED:")
print(f"{'year':>4} {'tr':>5} {'PF':>5} {'WR':>5} {'PnL$':>11} {'MDD$':>11}")
for y, g in combined.groupby("year"):
    p = g["sized_pnl"].to_numpy()
    w = p[p>0]; l = p[p<0]
    pf = w.sum()/abs(l.sum()) if len(l) else 99
    cum = p.cumsum()
    mdd = (cum - np.maximum.accumulate(cum)).min()
    print(f"{y:>4} {len(p):>5} {pf:>5.2f} {100*len(w)/len(p):>4.1f}% {p.sum():>+11,.0f} {mdd:>+11,.0f}")

# ---------- Plot ----------
is_end = pd.Timestamp("2024-12-31").tz_localize("America/New_York")

fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1]})

axes[0].plot(rv_cs["exit_ts"], rv_cs["cum"], lw=1.2, color="steelblue", label=f"Rough vol v3 + mart   ${rv['sized_pnl'].sum():+,.0f}")
axes[0].plot(od_cs["exit_ts"], od_cs["cum"], lw=1.2, color="darkorange", label=f"Overnight drift       ${od['sized_pnl'].sum():+,.0f}")
axes[0].plot(combined["exit_ts"], combined["cum_total"], lw=1.6, color="black", label=f"COMBINED              ${combined['sized_pnl'].sum():+,.0f}")
axes[0].axvline(is_end, color="red", ls="--", lw=1, alpha=0.7, label="rough-vol IS/OOS")
axes[0].axhline(0, color="black", lw=0.5)
axes[0].set_ylabel("Cumulative PnL ($)")
mdd_combined = combined["dd"].min()
axes[0].set_title(f"Combined: Rough Vol v3 (+ s1m1.5d2 mart) + Overnight Drift (locked)\n"
                   f"Total ${combined['sized_pnl'].sum():+,.0f}   MDD ${mdd_combined:+,.0f}   "
                   f"{len(combined)} trades")
axes[0].legend(loc="upper left", fontsize=10)
axes[0].grid(alpha=0.3)

axes[1].fill_between(combined["exit_ts"], combined["dd"], 0, color="red", alpha=0.4)
axes[1].set_ylabel("Combined DD ($)")
axes[1].set_xlabel("Date")
axes[1].grid(alpha=0.3)

plt.tight_layout()
out_png = RESULTS_DIR / "combined_curve.png"
plt.savefig(out_png, dpi=110)
print(f"\nCurve -> {out_png}")
