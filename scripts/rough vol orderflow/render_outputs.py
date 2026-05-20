"""
1) Render combined PnL curve bigger (PNG)
2) Build interactive monthly PnL calendar HTML with NQ/MNQ slider
3) Format 2026 trade log CSV as readable fixed-width txt
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
PLAN_DIR = HERE.parent.parent / "live" / "combined deployment plan"


# ---------- 1) Bigger combined curve ----------
def render_big_curve():
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

    is_end = pd.Timestamp("2024-12-31").tz_localize("America/New_York")

    fig, axes = plt.subplots(2, 1, figsize=(22, 13), sharex=True,
                              gridspec_kw={"height_ratios": [3.2, 1]})
    axes[0].plot(rv["exit_ts"], rv["c"], lw=2.0, color="steelblue",
                 label=f"Rough Vol v3 (no mart)         ${rv['pnl_$'].sum():+,.0f}")
    axes[0].plot(b2["exit_ts"], b2["c"], lw=2.0, color="seagreen",
                 label=f"B2 OHI/OLO (FC-only mart)     ${b2['pnl_$'].sum():+,.0f}")
    axes[0].plot(od["exit_ts"], od["c"], lw=2.0, color="darkorange",
                 label=f"Overnight Drift (locked)        ${od['pnl_$'].sum():+,.0f}")
    axes[0].plot(df["exit_ts"], df["cum"], lw=3.0, color="black",
                 label=f"COMBINED                        ${df['pnl_$'].sum():+,.0f}")
    axes[0].axvline(is_end, color="red", ls="--", lw=1.5, alpha=0.8, label="RV IS/OOS split")
    axes[0].axhline(0, color="black", lw=0.8)
    mdd = df["dd"].min()
    axes[0].set_ylabel("Cumulative PnL ($, NQ basis)", fontsize=14)
    axes[0].set_title("3-strategy combined: Rough Vol v3 + B2 OHI/OLO + Overnight Drift\n"
                       f"Total ${df['pnl_$'].sum():+,.0f}   MDD ${mdd:+,.0f}   "
                       f"{len(df)} trades   Sharpe 2.40   MAR 21.68",
                       fontsize=15)
    axes[0].legend(loc="upper left", fontsize=13, framealpha=0.9)
    axes[0].grid(alpha=0.3)
    axes[0].tick_params(labelsize=12)

    axes[1].fill_between(df["exit_ts"], df["dd"], 0, color="red", alpha=0.45)
    axes[1].plot(df["exit_ts"], df["dd"], lw=0.8, color="darkred", alpha=0.5)
    axes[1].set_ylabel("Drawdown ($)", fontsize=14)
    axes[1].set_xlabel("Date", fontsize=14)
    axes[1].grid(alpha=0.3)
    axes[1].tick_params(labelsize=12)

    plt.tight_layout()
    out = RESULTS_DIR / "combined_3way_curve_BIG.png"
    plt.savefig(out, dpi=120)
    print(f"Big curve -> {out}")
    out2 = PLAN_DIR / "combined_equity_curve_big.png"
    plt.savefig(out2, dpi=120)
    print(f"Also -> {out2}")


# ---------- 2) Interactive PnL calendar with NQ/MNQ slider ----------
def render_calendar_html():
    df = pd.read_csv(RESULTS_DIR / "combined_3way_trades.csv")
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = df["exit_ts"].dt.date
    df["month"] = df["exit_ts"].dt.to_period("M").astype(str)

    daily = df.groupby("date")["pnl_$"].sum().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["day"] = daily["date"].dt.day
    daily["dow"] = daily["date"].dt.dayofweek  # 0=Mon
    daily["week_of_month"] = (daily["day"] - 1 + daily["date"].apply(
        lambda d: pd.Timestamp(d.year, d.month, 1).dayofweek)) // 7

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Build a calendar-style heatmap: one subplot per (year, month)
    months = sorted(daily.groupby(["year", "month"]).size().index.tolist())

    # Layout: rows = years, cols = 12 months
    years = sorted(daily["year"].unique())
    n_years = len(years)

    fig = make_subplots(
        rows=n_years, cols=12,
        subplot_titles=[f"{y}-{m:02d}" for y in years for m in range(1, 13)],
        horizontal_spacing=0.005, vertical_spacing=0.04,
        shared_yaxes=False, shared_xaxes=False,
    )

    # We'll create two sets of traces — NQ and MNQ — and the slider toggles visibility
    for scale_label, scale_factor in [("NQ ($20/pt)", 1.0), ("MNQ ($2/pt)", 0.1)]:
        visible = (scale_label == "NQ ($20/pt)")
        for yi, y in enumerate(years, 1):
            for m in range(1, 13):
                sub = daily[(daily["year"] == y) & (daily["month"] == m)].copy()
                if len(sub) == 0:
                    # add a blank trace so columns align
                    fig.add_trace(go.Heatmap(z=[[None]], visible=visible, showscale=False,
                                              hoverinfo="skip"),
                                   row=yi, col=m)
                    continue
                # Build a 6-row × 7-col grid (rows = week of month, cols = day of week Mon-Sun)
                grid = np.full((6, 7), np.nan)
                text = np.full((6, 7), "", dtype=object)
                first_dow = pd.Timestamp(y, m, 1).dayofweek  # Mon=0
                for _, r in sub.iterrows():
                    dom = r["day"]
                    pos = first_dow + dom - 1
                    wk = pos // 7
                    dow = pos % 7
                    val = r["pnl_$"] * scale_factor
                    grid[wk, dow] = val
                    text[wk, dow] = f"{int(dom):>2}<br>${val:+,.0f}"
                fig.add_trace(go.Heatmap(
                    z=grid,
                    text=text, texttemplate="%{text}", textfont={"size": 9},
                    colorscale=[[0, "darkred"], [0.5, "white"], [1, "darkgreen"]],
                    zmid=0,
                    showscale=False,
                    hoverinfo="text",
                    visible=visible,
                    name=f"{scale_label}",
                ), row=yi, col=m)
                fig.update_yaxes(autorange="reversed", showgrid=False, showticklabels=False,
                                 row=yi, col=m)
                fig.update_xaxes(showgrid=False, showticklabels=False, row=yi, col=m)

    # The slider toggles visibility between NQ and MNQ traces
    n_traces_per_set = n_years * 12
    fig.update_layout(
        title="Combined daily PnL calendar — NQ vs MNQ toggle below<br>"
              "<sub>Green = winning day, Red = losing day, intensity = $ size. Calendar layout: rows = weeks of month, cols = Mon..Sun</sub>",
        height=max(2200, 220 * n_years),
        width=2400,
        margin=dict(l=20, r=20, t=120, b=80),
        updatemenus=[{
            "type": "buttons",
            "direction": "right",
            "showactive": True,
            "x": 0.5, "xanchor": "center",
            "y": 1.02, "yanchor": "bottom",
            "buttons": [
                {"label": "NQ ($20/pt)", "method": "update",
                 "args": [{"visible": [True] * n_traces_per_set + [False] * n_traces_per_set}]},
                {"label": "MNQ ($2/pt)", "method": "update",
                 "args": [{"visible": [False] * n_traces_per_set + [True] * n_traces_per_set}]},
            ],
        }],
    )
    # Smaller subplot titles
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=10)

    out_html = RESULTS_DIR / "combined_pnl_calendar.html"
    fig.write_html(str(out_html))
    out_html_plan = PLAN_DIR / "combined_pnl_calendar.html"
    fig.write_html(str(out_html_plan))
    print(f"Calendar HTML -> {out_html}")
    print(f"Also -> {out_html_plan}")


# ---------- 3) Formatted 2026 trade log txt ----------
def render_2026_txt():
    df = pd.read_csv(RESULTS_DIR / "inspect_v3_2026_log.csv")
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True).dt.tz_convert("America/New_York")
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    df = df.sort_values("entry_ts").reset_index(drop=True)

    lines = []
    lines.append("=" * 145)
    lines.append("Rough Vol v3 — 2026 trade log (32 trades)")
    lines.append("Config: 20m N=400 Z=75 EMA=80 HZ=2.00 SL=2.0 TP=2.0 | window_N8_D150 | gamma=none | session 09:00-12:59 + 14:00-14:44 ET")
    lines.append("=" * 145)
    lines.append("")
    hdr = (f"{'#':>3} {'entry_time':<19} {'side':>5}  "
           f"{'sig_O':>8} {'sig_H':>8} {'sig_L':>8} {'sig_C':>8}  "
           f"{'entry':>8} {'SL':>8} {'TP':>8}  "
           f"{'exit_time':<19} {'exit_px':>8}  {'reason':>11}  {'PnL_$':>9}  {'cum_$':>10}")
    lines.append(hdr)
    lines.append("-" * 145)
    cum = 0
    for i, r in enumerate(df.itertuples(), 1):
        cum += r.pnl_dollars
        lines.append(
            f"{i:>3} {str(r.entry_ts)[:19]:<19} {r.side:>5}  "
            f"{r.sig_open:>8.2f} {r.sig_high:>8.2f} {r.sig_low:>8.2f} {r.sig_close:>8.2f}  "
            f"{r.entry_price:>8.2f} {r.sl_price:>8.2f} {r.tp_price:>8.2f}  "
            f"{str(r.exit_ts)[:19]:<19} {r.exit_price:>8.2f}  {r.reason:>11}  "
            f"${r.pnl_dollars:>+8,.0f}  ${cum:>+9,.0f}"
        )
    lines.append("-" * 145)
    p = df["pnl_dollars"].to_numpy()
    w = p[p > 0]; l = p[p < 0]
    pf = w.sum() / abs(l.sum()) if len(l) else 99
    wr = 100 * len(w) / len(p)
    cum_arr = p.cumsum()
    mdd = (cum_arr - np.maximum.accumulate(cum_arr)).min()
    lines.append(f"TOTAL: {len(p)} trades   PF {pf:.2f}   WR {wr:.1f}%   PnL ${p.sum():+,.0f}   MDD ${mdd:+,.0f}   "
                  f"avg win ${w.mean():+,.0f}   avg loss ${l.mean():+,.0f}")
    lines.append("")
    lines.append("Reason breakdown:")
    for reason, g in df.groupby("reason"):
        pp = g["pnl_dollars"].to_numpy()
        lines.append(f"  {reason:>11}: {len(pp):>2}t   PnL ${pp.sum():>+8,.0f}   "
                      f"avg ${pp.mean():>+7,.0f}")

    out = RESULTS_DIR / "inspect_v3_2026_log.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"2026 txt -> {out}")
    print("\nFirst 10 lines preview:")
    for L in lines[:14]:
        print(L)


def main():
    render_big_curve()
    print()
    render_calendar_html()
    print()
    render_2026_txt()


if __name__ == "__main__":
    main()
