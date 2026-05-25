"""
Generate the equity curve + key stats for the locked live config.

Saves outputs to live/overnight drift/:
  - equity_curve.html      (interactive plot)
  - equity_curve.png       (static image)
  - trades.csv             (full trade log)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overnight_drift_strategy import (  # noqa: E402
    StrategyParams,
    build_full_20min_series,
    run_backtest,
    trades_to_df,
)

PARQUET = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
PICKLES = "D:/trading_pythonbacktest_data/timebars_5min"
OUT_DIR = Path("C:/trading/nqorderflowbacktester/live/overnight drift")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TZ = "America/New_York"


def main() -> None:
    print("Loading bars...", flush=True)
    bars = build_full_20min_series(PARQUET, PICKLES)
    print(f"  bars: {len(bars):,}  range: {bars.index.min()} -> {bars.index.max()}\n", flush=True)

    # REVERTED 2026-05-25: back to ORIGINAL LIVE config after 5/14 OOS disaster.
    # See live/od_green_sweep_top_configs.md for full sweep + revert rationale.
    params = StrategyParams(
        yellow_atr_len=14,
        yellow_atr_mult=1.30,           # original live
        yellow_drift=0.0,
        yellow_mode="pure_ratchet",
        green_atr_len=14,
        green_atr_mult=1.00,            # original live
        green_base=82.5,                # original live
        green_decay=1.50,               # original live
        red_intercept=0.0,
        red_drift=0.45,
        use_be=False,
        use_martingale=True,
        base_qty=1,
        loss_qty=2,
        yellow_suppress_bars=0,         # original live (no suppress)
    )

    trades = trades_to_df(run_backtest(bars, params))
    trades["entry_time"] = pd.to_datetime(trades["entry_time"]).dt.tz_convert(TZ)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"]).dt.tz_convert(TZ)
    trades = trades.sort_values("entry_time").reset_index(drop=True)
    trades["cum_pnl_$"] = trades["pnl_dollars"].cumsum()
    trades["peak"] = trades["cum_pnl_$"].cummax()
    trades["drawdown"] = trades["cum_pnl_$"] - trades["peak"]

    # Stats
    pnl = trades["pnl_dollars"].values
    wins = (pnl > 0).sum()
    gw = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    pf = gw / gl if gl > 0 else float("inf")
    mdd = float(trades["drawdown"].min())
    mdd_at = trades.loc[trades["drawdown"].idxmin(), "entry_time"]

    print(f"Trades: {len(trades)}")
    print(f"Win rate: {wins/len(trades)*100:.2f}%")
    print(f"PF: {pf:.3f}")
    print(f"Gross $: ${pnl.sum():,.0f}")
    print(f"Max DD: ${mdd:,.0f}  at {mdd_at}")
    print(f"Best trade: ${pnl.max():,.0f}")
    print(f"Worst trade: ${pnl.min():,.0f}")
    print(f"Avg win: ${pnl[pnl>0].mean():.0f}")
    print(f"Avg loss: ${pnl[pnl<0].mean():.0f}")

    # Save full trade log
    trades.to_csv(OUT_DIR / "trades.csv", index=False)
    print(f"\nSaved trades -> {OUT_DIR / 'trades.csv'}")

    # ---- Matplotlib equity curve (PNG) ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(trades["entry_time"], trades["cum_pnl_$"], color="#0a7", linewidth=1.5,
             label="Equity (cumulative $)")
    ax1.plot(trades["entry_time"], trades["peak"], color="#888", linewidth=0.8,
             linestyle="--", label="Running peak", alpha=0.6)
    ax1.fill_between(trades["entry_time"], trades["cum_pnl_$"], trades["peak"],
                     color="#f44", alpha=0.18, label="Drawdown zone")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Cumulative P&L ($)")
    ax1.set_title("Overnight Drift — Locked Live Config\n"
                  "y=1.30 / g=1.00 / g_base=82.5 / g_decay=1.5 | pure_ratchet | BE off | marti 1/2 (s1-L2)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

    ax2.fill_between(trades["entry_time"], trades["drawdown"], 0,
                     color="#f44", alpha=0.5)
    ax2.set_ylabel("Drawdown ($)")
    ax2.set_xlabel("Trade entry date")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.axhline(mdd, color="darkred", linewidth=0.8, linestyle=":",
                label=f"Max DD: ${mdd:,.0f}")
    ax2.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "equity_curve.png", dpi=120, bbox_inches="tight")
    print(f"Saved chart -> {OUT_DIR / 'equity_curve.png'}")

    # ---- HTML version (Plotly) ----
    try:
        import plotly.graph_objs as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.72, 0.28],
                            vertical_spacing=0.04,
                            subplot_titles=("Equity (cumulative $)", "Drawdown ($)"))
        fig.add_trace(go.Scatter(x=trades["entry_time"], y=trades["cum_pnl_$"],
                                 mode="lines", name="Equity",
                                 line=dict(color="#0a7", width=1.6)), row=1, col=1)
        fig.add_trace(go.Scatter(x=trades["entry_time"], y=trades["peak"],
                                 mode="lines", name="Running peak",
                                 line=dict(color="#888", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=trades["entry_time"], y=trades["drawdown"],
                                 mode="lines", fill="tozeroy", name="Drawdown",
                                 line=dict(color="#f44", width=1)), row=2, col=1)
        fig.update_layout(
            title=("Overnight Drift — Locked Live Config<br>"
                   "<sub>y=1.30 / g=1.00 / g_base=82.5 / g_decay=1.5 | pure_ratchet | BE off | marti 1/2</sub>"),
            template="plotly_white", height=720, hovermode="x unified")
        fig.update_yaxes(tickformat="$,.0f", row=1, col=1)
        fig.update_yaxes(tickformat="$,.0f", row=2, col=1)
        fig.write_html(OUT_DIR / "equity_curve.html", include_plotlyjs="cdn")
        print(f"Saved HTML  -> {OUT_DIR / 'equity_curve.html'}")
    except ImportError:
        print("plotly not installed — skipping HTML version")

    # Year x metric breakdown for the markdown
    trades["year"] = trades["entry_time"].dt.year
    yearly = trades.groupby("year").apply(
        lambda g: pd.Series({
            "trades": len(g),
            "win%": (g["pnl_dollars"] > 0).mean() * 100,
            "PF": (g.loc[g["pnl_dollars"] > 0, "pnl_dollars"].sum() /
                   abs(g.loc[g["pnl_dollars"] < 0, "pnl_dollars"].sum()))
                  if (g["pnl_dollars"] < 0).any() else float("inf"),
            "gross_$": g["pnl_dollars"].sum(),
            "max_dd": (g["pnl_dollars"].cumsum() - g["pnl_dollars"].cumsum().cummax()).min(),
        }), include_groups=False)
    yearly.to_csv(OUT_DIR / "yearly_stats.csv")
    print(f"Saved yearly -> {OUT_DIR / 'yearly_stats.csv'}")


if __name__ == "__main__":
    main()
