"""Plot equity curve for one-shot any_loss continuous martingale vs baseline."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"

# Load the martingale-applied trade log (already has sized_pnl, qty)
df = pd.read_csv(RESULTS_DIR / "martingale_one_shot_log.csv")
df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True).dt.tz_convert("America/New_York")
df = df.sort_values("entry_ts").reset_index(drop=True)

df["base_cum"] = df["pnl_dollars"].cumsum()
df["mart_cum"] = df["sized_pnl"].cumsum()
df["base_peak"] = df["base_cum"].cummax()
df["mart_peak"] = df["mart_cum"].cummax()
df["base_dd"] = df["base_cum"] - df["base_peak"]
df["mart_dd"] = df["mart_cum"] - df["mart_peak"]

is_end = pd.Timestamp("2024-12-31").tz_localize("America/New_York")

fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1.2, 1]})

# Equity
axes[0].plot(df["entry_ts"], df["base_cum"], lw=1.4, color="steelblue", label="Baseline (1 contract)")
axes[0].plot(df["entry_ts"], df["mart_cum"], lw=1.4, color="darkorange", label="One-shot any_loss continuous")
axes[0].fill_between(df["entry_ts"], df["base_cum"], df["mart_cum"],
                      where=df["mart_cum"] >= df["base_cum"], alpha=0.15, color="green", label="mart > baseline")
axes[0].fill_between(df["entry_ts"], df["base_cum"], df["mart_cum"],
                      where=df["mart_cum"] < df["base_cum"], alpha=0.20, color="red", label="mart < baseline")
axes[0].axvline(is_end, color="red", ls="--", lw=1, alpha=0.7, label="IS/OOS split")
axes[0].axhline(0, color="black", lw=0.5)
axes[0].set_ylabel("Cumulative PnL ($)")
axes[0].set_title("20m N=400 v3 — Martingale: one-shot any_loss continuous  vs  baseline (1 contract)\n"
                  f"baseline ${df['base_cum'].iloc[-1]:+,.0f} (MDD ${df['base_dd'].min():+,.0f})   "
                  f"mart ${df['mart_cum'].iloc[-1]:+,.0f} (MDD ${df['mart_dd'].min():+,.0f})")
axes[0].legend(loc="upper left", fontsize=9)
axes[0].grid(alpha=0.3)

# Drawdowns
axes[1].fill_between(df["entry_ts"], df["base_dd"], 0, color="steelblue", alpha=0.4, label="baseline DD")
axes[1].fill_between(df["entry_ts"], df["mart_dd"], 0, color="darkorange", alpha=0.4, label="mart DD")
axes[1].set_ylabel("Drawdown ($)")
axes[1].legend(loc="lower left", fontsize=9)
axes[1].grid(alpha=0.3)

# Position size used (1 or 2)
axes[2].plot(df["entry_ts"], df["qty"], lw=0.5, color="darkred", alpha=0.6, drawstyle="steps-post")
axes[2].fill_between(df["entry_ts"], 1, df["qty"], step="post", alpha=0.3, color="darkred")
axes[2].set_ylabel("Contracts")
axes[2].set_xlabel("Date")
axes[2].set_ylim(0.5, 2.5)
axes[2].set_yticks([1, 2])
axes[2].grid(alpha=0.3)

plt.tight_layout()
out_png = RESULTS_DIR / "martingale_one_shot_curve.png"
plt.savefig(out_png, dpi=110)
print(f"Curve -> {out_png}")

# Key stats summary
print(f"\nBaseline: trades={len(df)}, final ${df['base_cum'].iloc[-1]:+,.0f}, MDD ${df['base_dd'].min():+,.0f}")
print(f"Mart:     trades={len(df)}, final ${df['mart_cum'].iloc[-1]:+,.0f}, MDD ${df['mart_dd'].min():+,.0f}")
print(f"Mart - Baseline: ${df['mart_cum'].iloc[-1] - df['base_cum'].iloc[-1]:+,.0f}")
print(f"Mart used qty=2 on {(df['qty']==2).sum()} / {len(df)} trades ({100*(df['qty']==2).mean():.1f}%)")
