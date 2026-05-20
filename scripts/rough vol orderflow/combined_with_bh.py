"""
Render combined 3-strategy PnL curve with an NQ buy-and-hold overlay.

NQ buy-and-hold price series is taken from the overnight drift trade log
(entry_price column @ ~19:00 ET each trading day) — it's the densest
clean NQ price stream in the repo over the full 2020-12 -> 2026-05 span.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
ROOT = HERE.parent.parent
NQ_PT = 20.0


def main():
    df = pd.read_csv(RESULTS_DIR / "combined_3way_trades.csv")
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True).dt.tz_convert("America/New_York")
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    df = df.sort_values("exit_ts").reset_index(drop=True)
    df["cum"] = df["pnl_$"].cumsum()
    df["peak"] = df["cum"].cummax()
    df["dd"] = df["cum"] - df["peak"]

    rv = df[df["strat"] == "RV"].sort_values("exit_ts")
    b2 = df[df["strat"] == "B2"].sort_values("exit_ts")
    od = df[df["strat"] == "OD"].sort_values("exit_ts")
    rv = rv.assign(c=rv["pnl_$"].cumsum())
    b2 = b2.assign(c=b2["pnl_$"].cumsum())
    od = od.assign(c=od["pnl_$"].cumsum())

    # ---------- Buy-and-hold price series from OD log ----------
    od_raw = pd.read_csv(ROOT / "live" / "overnight drift" / "trades.csv")
    od_raw["t"] = pd.to_datetime(od_raw["entry_time"], utc=True).dt.tz_convert("America/New_York")
    od_raw = od_raw.sort_values("t").reset_index(drop=True)
    # also append the final exit point so the curve ends where the combined ends
    last_exit_t = pd.to_datetime(od_raw["exit_time"].iloc[-1], utc=True).tz_convert("America/New_York")
    last_exit_px = float(od_raw["exit_price"].iloc[-1])
    bh = pd.DataFrame({
        "t": list(od_raw["t"]) + [last_exit_t],
        "px": list(od_raw["entry_price"]) + [last_exit_px],
    }).sort_values("t").reset_index(drop=True)

    px0 = float(bh["px"].iloc[0])
    bh["bh_$"] = (bh["px"] - px0) * NQ_PT
    bh_peak = bh["bh_$"].cummax()
    bh_dd = bh["bh_$"] - bh_peak

    is_end = pd.Timestamp("2024-12-31").tz_localize("America/New_York")

    fig, axes = plt.subplots(2, 1, figsize=(22, 13), sharex=True,
                              gridspec_kw={"height_ratios": [3.2, 1]})

    axes[0].plot(rv["exit_ts"], rv["c"], lw=1.7, color="steelblue",
                 label=f"Rough Vol v3 (no mart)         ${rv['pnl_$'].sum():+,.0f}")
    axes[0].plot(b2["exit_ts"], b2["c"], lw=1.7, color="seagreen",
                 label=f"B2 OHI/OLO (FC-only mart)     ${b2['pnl_$'].sum():+,.0f}")
    axes[0].plot(od["exit_ts"], od["c"], lw=1.7, color="darkorange",
                 label=f"Overnight Drift (locked)        ${od['pnl_$'].sum():+,.0f}")
    axes[0].plot(df["exit_ts"], df["cum"], lw=3.0, color="black",
                 label=f"COMBINED                        ${df['pnl_$'].sum():+,.0f}")
    axes[0].plot(bh["t"], bh["bh_$"], lw=2.0, color="purple", ls="--",
                 label=f"NQ buy & hold (1 contract)     ${bh['bh_$'].iloc[-1]:+,.0f}")
    axes[0].axvline(is_end, color="red", ls="--", lw=1.5, alpha=0.7, label="RV IS/OOS split")
    axes[0].axhline(0, color="black", lw=0.8)

    mdd = df["dd"].min()
    bh_mdd = bh_dd.min()
    combined_total = df["pnl_$"].sum()
    bh_total = bh["bh_$"].iloc[-1]
    axes[0].set_ylabel("Cumulative PnL ($, NQ basis)", fontsize=14)
    axes[0].set_title(
        "3-strategy combined vs NQ buy & hold (1 contract basis)\n"
        f"Combined ${combined_total:+,.0f}  MDD ${mdd:+,.0f}  MAR {combined_total/abs(mdd):.1f}    "
        f"|    B&H ${bh_total:+,.0f}  MDD ${bh_mdd:+,.0f}  MAR {bh_total/abs(bh_mdd):.2f}",
        fontsize=15,
    )
    axes[0].legend(loc="upper left", fontsize=13, framealpha=0.9)
    axes[0].grid(alpha=0.3)
    axes[0].tick_params(labelsize=12)

    # Drawdown panel: both
    axes[1].fill_between(df["exit_ts"], df["dd"], 0, color="red", alpha=0.45,
                          label=f"Combined DD (MDD ${mdd:+,.0f})")
    axes[1].fill_between(bh["t"], bh_dd, 0, color="purple", alpha=0.18,
                          label=f"B&H DD (MDD ${bh_mdd:+,.0f})")
    axes[1].set_ylabel("Drawdown ($)", fontsize=14)
    axes[1].set_xlabel("Date", fontsize=14)
    axes[1].legend(loc="lower left", fontsize=12, framealpha=0.9)
    axes[1].grid(alpha=0.3)
    axes[1].tick_params(labelsize=12)

    plt.tight_layout()
    out = RESULTS_DIR / "combined_3way_curve_BIG_with_BH.png"
    plt.savefig(out, dpi=120)
    print(f"Curve -> {out}")
    print(f"Combined total ${combined_total:+,.0f}  MDD ${mdd:+,.0f}")
    print(f"B&H total      ${bh_total:+,.0f}  MDD ${bh_mdd:+,.0f}")
    print(f"B&H start px {px0:.2f}  end px {bh['px'].iloc[-1]:.2f}  "
          f"delta {bh['px'].iloc[-1]-px0:+.2f} pts")


if __name__ == "__main__":
    main()
