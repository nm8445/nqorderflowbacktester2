"""
Combined deployment: Rough Vol v3 (no martingale) + B2 OHI/OLO (FC-only mart, locked)
                     + Overnight Drift (locked).

Conflict resolution:
  - Rough Vol + B2 overlap intraday (both 09:00-15:00 ET range).
    Same direction: both fire.
    Opposite directions: SECOND entry blocked (no hedging).
  - Overnight Drift trades 19:00 -> 08:00 ET: never overlaps with intraday.
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

NQ_PT = 20.0  # NQ dollar per point

# ---------- Load trades ----------
# Rough vol — no martingale (1 contract)
rv = pd.read_csv(RESULTS_DIR / "inspect_v3_FULL_log.csv")
rv["entry_ts"] = pd.to_datetime(rv["entry_ts"], utc=True).dt.tz_convert("America/New_York")
rv["exit_ts"] = pd.to_datetime(rv["exit_ts"], utc=True).dt.tz_convert("America/New_York")
rv["direction"] = rv["side"]  # already "LONG"/"SHORT"
rv["pnl_$"] = rv["pnl_dollars"]
rv["strat"] = "RV"
rv = rv[["entry_ts", "exit_ts", "direction", "pnl_$", "strat"]].copy()

# B2 — FC-only mart applied (scaled_pnl is points; multiply by $20)
b2 = pd.read_csv(ROOT / "scripts" / "overnight range strat" / "tradelogs" / "robust_configs"
                  / "locked_v2_k08_lock045_mart_fc_filtered_trades.csv")
b2["entry_ts"] = pd.to_datetime(b2["entry_ts"], utc=True).dt.tz_convert("America/New_York")
b2["exit_ts"] = pd.to_datetime(b2["exit_ts"], utc=True).dt.tz_convert("America/New_York")
b2["pnl_$"] = b2["scaled_pnl"] * NQ_PT
b2["strat"] = "B2"
b2 = b2[["entry_ts", "exit_ts", "direction", "pnl_$", "strat"]].copy()

# Overnight drift — locked mart already applied, pnl_dollars in $
od = pd.read_csv(ROOT / "live" / "overnight drift" / "trades.csv")
od["entry_ts"] = pd.to_datetime(od["entry_time"], utc=True).dt.tz_convert("America/New_York")
od["exit_ts"] = pd.to_datetime(od["exit_time"], utc=True).dt.tz_convert("America/New_York")
od["direction"] = "LONG"  # overnight drift is long-only
od["pnl_$"] = od["pnl_dollars"]
od["strat"] = "OD"
od = od[["entry_ts", "exit_ts", "direction", "pnl_$", "strat"]].copy()

print(f"RV trades:  {len(rv):>5}  PnL ${rv['pnl_$'].sum():+,.0f}")
print(f"B2 trades:  {len(b2):>5}  PnL ${b2['pnl_$'].sum():+,.0f}")
print(f"OD trades:  {len(od):>5}  PnL ${od['pnl_$'].sum():+,.0f}")


# ---------- Conflict resolution between RV and B2 ----------
intraday = pd.concat([rv, b2], ignore_index=True).sort_values("entry_ts").reset_index(drop=True)

kept = []
open_positions = []  # list of (exit_ts, direction)
n_blocked = 0
for _, r in intraday.iterrows():
    # Drop closed positions
    open_positions = [p for p in open_positions if p[0] > r["entry_ts"]]
    # Conflict if any open position has opposite direction
    if any(p[1] != r["direction"] for p in open_positions):
        n_blocked += 1
        continue
    kept.append(r)
    open_positions.append((r["exit_ts"], r["direction"]))
kept_df = pd.DataFrame(kept)
print(f"\nIntraday conflict resolution: {len(intraday)} -> {len(kept_df)} ({n_blocked} blocked by opposing position)")

# Breakdown blocked by source
blocked_rv = blocked_b2 = 0
open_positions = []
for _, r in intraday.iterrows():
    open_positions = [p for p in open_positions if p[0] > r["entry_ts"]]
    if any(p[1] != r["direction"] for p in open_positions):
        if r["strat"] == "RV": blocked_rv += 1
        else: blocked_b2 += 1
        continue
    open_positions.append((r["exit_ts"], r["direction"]))
print(f"  RV trades blocked: {blocked_rv} / {len(rv)} ({100*blocked_rv/len(rv):.1f}%)")
print(f"  B2 trades blocked: {blocked_b2} / {len(b2)} ({100*blocked_b2/len(b2):.1f}%)")


# ---------- Combine everything ----------
combined = pd.concat([kept_df, od], ignore_index=True).sort_values("exit_ts").reset_index(drop=True)
combined["cum"] = combined["pnl_$"].cumsum()
combined["peak"] = combined["cum"].cummax()
combined["dd"] = combined["cum"] - combined["peak"]


# ---------- Stats ----------
def stats(p, label):
    p = np.asarray(p)
    w = p[p > 0]; l = p[p < 0]
    pf = w.sum() / abs(l.sum()) if len(l) else 99.0
    wr = 100 * len(w) / len(p) if len(p) else 0
    cum = p.cumsum()
    mdd = (cum - np.maximum.accumulate(cum)).min()
    mar = p.sum() / abs(mdd) if mdd < 0 else 99.0
    print(f"{label:>40}: {len(p):>5}t  PF {pf:.2f}  WR {wr:.1f}%  "
          f"PnL ${p.sum():+,.0f}  MDD ${mdd:+,.0f}  MAR {mar:.2f}")


print()
stats(rv["pnl_$"].to_numpy(), "Rough vol v3 (no mart)")
stats(b2["pnl_$"].to_numpy(), "B2 (FC-only mart)")
stats(od["pnl_$"].to_numpy(), "Overnight drift (locked)")
print()
stats(kept_df["pnl_$"].to_numpy(), "RV+B2 (after conflict resolution)")
stats(combined["pnl_$"].to_numpy(), "COMBINED (RV+B2+OD)")


# Daily Sharpe
combined["date"] = combined["exit_ts"].dt.date
daily = combined.groupby("date")["pnl_$"].sum()
sh = daily.mean() / daily.std(ddof=1) * np.sqrt(252)
down = daily[daily < 0]
so = daily.mean() / down.std(ddof=1) * np.sqrt(252) if len(down) > 1 else 99.0
print(f"\nCombined daily Sharpe (annualized): {sh:.2f}")
print(f"Combined daily Sortino (annualized): {so:.2f}")
print(f"Combined Calmar (annual PnL / |MDD|): {(combined['pnl_$'].sum()/(len(daily)/252)) / abs(combined['dd'].min()):.2f}")

# Year-by-year
combined["year"] = combined["exit_ts"].dt.year
print("\nYear-by-year COMBINED (RV+B2+OD):")
print(f"{'year':>4} {'tr':>5} {'PF':>5} {'WR':>5} {'PnL$':>11} {'MDD$':>11} {'Sharpe':>7}")
for y, g in combined.groupby("year"):
    p = g["pnl_$"].to_numpy()
    w = p[p>0]; l = p[p<0]
    pf = w.sum()/abs(l.sum()) if len(l) else 99
    wr = 100*len(w)/len(p)
    cum = p.cumsum(); mdd = (cum - np.maximum.accumulate(cum)).min()
    g["date"] = g["exit_ts"].dt.date
    dly = g.groupby("date")["pnl_$"].sum()
    sh_y = dly.mean() / dly.std(ddof=1) * np.sqrt(252) if dly.std(ddof=1) > 0 else 0
    print(f"{y:>4} {len(p):>5} {pf:>5.2f} {wr:>4.1f}% {p.sum():>+11,.0f} {mdd:>+11,.0f} {sh_y:>7.2f}")


# Per-strategy contribution (combined timeframe)
print("\nPer-strategy contribution in COMBINED:")
combined_with_strat = pd.concat([kept_df.assign(_kept_strat=kept_df["strat"]),
                                  od.assign(_kept_strat="OD")], ignore_index=True).sort_values("exit_ts")
for s in ("RV", "B2", "OD"):
    sub = combined_with_strat[combined_with_strat["_kept_strat"] == s]
    print(f"  {s}: {len(sub)} trades  PnL ${sub['pnl_$'].sum():+,.0f}")


# ---------- Plot ----------
is_end = pd.Timestamp("2024-12-31").tz_localize("America/New_York")
rv_only = kept_df[kept_df["strat"] == "RV"].sort_values("exit_ts").reset_index(drop=True)
b2_only = kept_df[kept_df["strat"] == "B2"].sort_values("exit_ts").reset_index(drop=True)
od_only = od.sort_values("exit_ts").reset_index(drop=True)
rv_only["cum"] = rv_only["pnl_$"].cumsum()
b2_only["cum"] = b2_only["pnl_$"].cumsum()
od_only["cum"] = od_only["pnl_$"].cumsum()

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1]})
axes[0].plot(rv_only["exit_ts"], rv_only["cum"], lw=1.1, color="steelblue",
             label=f"Rough Vol v3 (kept, no mart)    ${rv_only['pnl_$'].sum():+,.0f}")
axes[0].plot(b2_only["exit_ts"], b2_only["cum"], lw=1.1, color="seagreen",
             label=f"B2 OHI/OLO (kept, FC-mart)       ${b2_only['pnl_$'].sum():+,.0f}")
axes[0].plot(od_only["exit_ts"], od_only["cum"], lw=1.1, color="darkorange",
             label=f"Overnight Drift (locked)          ${od_only['pnl_$'].sum():+,.0f}")
axes[0].plot(combined["exit_ts"], combined["cum"], lw=1.8, color="black",
             label=f"COMBINED                          ${combined['pnl_$'].sum():+,.0f}")
axes[0].axvline(is_end, color="red", ls="--", lw=1, alpha=0.7, label="RV IS/OOS split")
axes[0].axhline(0, color="black", lw=0.5)
mdd = combined["dd"].min()
axes[0].set_ylabel("Cumulative PnL ($)")
axes[0].set_title("3-strategy combined: Rough Vol v3 + B2 OHI/OLO + Overnight Drift\n"
                   f"Total ${combined['pnl_$'].sum():+,.0f}   MDD ${mdd:+,.0f}   "
                   f"{len(combined)} trades   Sharpe {sh:.2f}   "
                   f"(RV+B2 conflict-blocked: {blocked_rv+blocked_b2})")
axes[0].legend(loc="upper left", fontsize=10)
axes[0].grid(alpha=0.3)
axes[1].fill_between(combined["exit_ts"], combined["dd"], 0, color="red", alpha=0.4)
axes[1].set_ylabel("Combined DD ($)")
axes[1].set_xlabel("Date")
axes[1].grid(alpha=0.3)
plt.tight_layout()
out_png = RESULTS_DIR / "combined_3way_curve.png"
plt.savefig(out_png, dpi=110)
print(f"\nCurve -> {out_png}")

# Save merged trade log
combined.to_csv(RESULTS_DIR / "combined_3way_trades.csv", index=False)
print(f"Trades CSV -> {RESULTS_DIR/'combined_3way_trades.csv'}")
