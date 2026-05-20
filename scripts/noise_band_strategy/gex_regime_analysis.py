"""
GEX regime analysis for the noise-band strategy.

Tags each trading day as positive-GEX or negative-GEX using the squeezemetrics
DIX.csv data, then compares strategy performance across regimes.

IMPORTANT: GEX is shifted forward by 1 day. The value for date X was computed
after X's close, so it's used to classify trades on date X+1.

Usage:
    python -u scripts/noise_band_strategy/gex_regime_analysis.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

DIX_CSV = Path("D:/trading_pythonbacktest_data/NQ L2 data/DIX.csv")
TRADES_CSV = Path("C:/trading/nqorderflowbacktester/results/noise_band/trades_quantitativo.csv")

NQ_POINT_VALUE = 20.0


def load_gex() -> pd.DataFrame:
    """Load DIX.csv and shift GEX forward by 1 business day."""
    df = pd.read_csv(DIX_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # Shift GEX forward: date X's GEX classifies X+1's trades
    df["gex_for_next_day"] = df["gex"].values
    df["trade_date"] = df["date"].shift(-1)  # this GEX applies to the next row's date
    df = df.dropna(subset=["trade_date"])
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df[["trade_date", "gex_for_next_day", "dix"]].copy()


def load_trades() -> pd.DataFrame:
    """Load trade CSV and extract trade date."""
    df = pd.read_csv(TRADES_CSV)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df["trade_date"] = df["entry_time"].dt.tz_convert("America/New_York").dt.date
    return df


def print_regime_stats(trades: pd.DataFrame, label: str):
    """Print summary stats for a subset of trades."""
    if len(trades) == 0:
        print(f"  {label}: no trades")
        return

    pnls = trades["net_pnl"].values
    w = pnls[pnls > 0]
    l = pnls[pnls < 0]
    pf = w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else 99
    wr = 100 * len(w) / len(pnls)
    avg_pnl = pnls.mean()

    # Daily aggregation for Sharpe
    daily = trades.groupby("trade_date")["net_pnl"].sum()
    daily_arr = daily.values
    sharpe = (daily_arr.mean() / daily_arr.std() * np.sqrt(252)) if daily_arr.std() > 0 else 0

    # Drawdown
    cum = pnls.cumsum()
    dd = (cum - np.maximum.accumulate(cum)).min()

    # Trading days
    n_days = trades["trade_date"].nunique()

    print(f"  {label}:")
    print(f"    Trades: {len(pnls):,}  ({n_days} trading days)")
    print(f"    PF: {pf:.2f}  WR: {wr:.1f}%  Sharpe: {sharpe:.2f}")
    print(f"    Total PnL: ${pnls.sum():+,.0f}  Avg: ${avg_pnl:+,.0f}  Max DD: ${dd:,.0f}")
    if len(w):
        print(f"    Avg Win: ${w.mean():+,.0f}  Avg Loss: ${l.mean():+,.0f}")

    # By direction
    for d in ["long", "short"]:
        sub = trades[trades["direction"] == d]
        if len(sub) == 0:
            continue
        sp = sub["net_pnl"].values
        sw = sp[sp > 0]
        sl_arr = sp[sp < 0]
        spf = sw.sum() / abs(sl_arr.sum()) if len(sl_arr) and sl_arr.sum() != 0 else 99
        swr = 100 * len(sw) / len(sp)
        print(f"      {d:>5}: {len(sp)} trades  PF={spf:.2f}  WR={swr:.1f}%  PnL=${sp.sum():+,.0f}")


def main():
    print("Loading GEX data...", flush=True)
    gex_df = load_gex()
    print(f"  GEX records: {len(gex_df):,}  ({gex_df['trade_date'].iloc[0]} to {gex_df['trade_date'].iloc[-1]})")

    print("Loading trades...", flush=True)
    trades = load_trades()
    print(f"  Trades: {len(trades):,}  ({trades['trade_date'].iloc[0]} to {trades['trade_date'].iloc[-1]})")

    # Merge trades with GEX
    merged = trades.merge(gex_df, on="trade_date", how="left")
    matched = merged.dropna(subset=["gex_for_next_day"])
    print(f"  Matched with GEX: {len(matched):,} / {len(trades):,}")

    gex_vals = matched["gex_for_next_day"]
    print(f"\n  GEX stats: median={gex_vals.median():,.0f}  mean={gex_vals.mean():,.0f}")
    print(f"  GEX > 0: {(gex_vals > 0).sum():,} trades  ({100*(gex_vals > 0).mean():.1f}%)")
    print(f"  GEX < 0: {(gex_vals < 0).sum():,} trades  ({100*(gex_vals < 0).mean():.1f}%)")

    # ── Binary regime: positive vs negative GEX ─────────────────────────────
    pos_gex = matched[matched["gex_for_next_day"] > 0]
    neg_gex = matched[matched["gex_for_next_day"] <= 0]

    print(f"\n{'='*70}")
    print("REGIME COMPARISON: Positive GEX vs Negative GEX")
    print(f"{'='*70}")
    print_regime_stats(pos_gex, "POSITIVE GEX (dealers long gamma -> dampening)")
    print()
    print_regime_stats(neg_gex, "NEGATIVE GEX (dealers short gamma -> amplifying)")

    # ── Quartile analysis ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("GEX QUARTILE BREAKDOWN")
    print(f"{'='*70}")
    matched_sorted = matched.copy()
    matched_sorted["gex_quartile"] = pd.qcut(
        matched_sorted["gex_for_next_day"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    )
    for q in ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]:
        sub = matched_sorted[matched_sorted["gex_quartile"] == q]
        gex_range = sub["gex_for_next_day"]
        print(f"\n  {q}: GEX range [{gex_range.min():+,.0f} to {gex_range.max():+,.0f}]")
        pnls = sub["net_pnl"].values
        w = pnls[pnls > 0]; l_arr = pnls[pnls < 0]
        pf = w.sum() / abs(l_arr.sum()) if len(l_arr) and l_arr.sum() != 0 else 99
        wr = 100 * len(w) / len(pnls) if len(pnls) else 0
        print(f"    Trades: {len(pnls)}  PF: {pf:.2f}  WR: {wr:.1f}%  "
              f"PnL: ${pnls.sum():+,.0f}  Avg: ${pnls.mean():+,.0f}")

    # ── Yearly regime breakdown ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("YEARLY BREAKDOWN BY GEX REGIME")
    print(f"{'='*70}")
    matched["year"] = pd.to_datetime(matched["entry_time"]).dt.year
    matched["regime"] = np.where(matched["gex_for_next_day"] > 0, "pos_gex", "neg_gex")

    print(f"\n{'Year':<6} {'Regime':<10} {'Trades':>7} {'PF':>6} {'WR':>6} {'PnL':>12} {'Avg':>8}")
    print("-" * 60)
    for year in sorted(matched["year"].unique()):
        for regime in ["pos_gex", "neg_gex"]:
            sub = matched[(matched["year"] == year) & (matched["regime"] == regime)]
            if len(sub) == 0:
                continue
            pnls = sub["net_pnl"].values
            w = pnls[pnls > 0]; l_arr = pnls[pnls < 0]
            pf = w.sum() / abs(l_arr.sum()) if len(l_arr) and l_arr.sum() != 0 else 99
            wr = 100 * len(w) / len(pnls)
            tag = "+" if regime == "pos_gex" else "-"
            print(f"{year:<6} {tag+'GEX':<10} {len(pnls):>7} {pf:>6.2f} {wr:>5.1f}% ${pnls.sum():>+10,.0f} ${pnls.mean():>+7,.0f}")

    # ── Exit reason by regime ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("EXIT REASON BY REGIME")
    print(f"{'='*70}")
    for regime_name, sub_df in [("POSITIVE GEX", pos_gex), ("NEGATIVE GEX", neg_gex)]:
        print(f"\n  {regime_name}:")
        for reason in sorted(sub_df["exit_reason"].unique()):
            rsub = sub_df[sub_df["exit_reason"] == reason]
            pnls = rsub["net_pnl"].values
            wr = 100 * (pnls > 0).sum() / len(pnls) if len(pnls) else 0
            print(f"    {reason:<15} {len(pnls):>5} trades  WR={wr:.1f}%  Avg=${pnls.mean():+,.0f}  Total=${pnls.sum():+,.0f}")


if __name__ == "__main__":
    main()
