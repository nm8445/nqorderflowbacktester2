"""
In-sample / out-of-sample test for VWAP reaction strategy with filters:
  - ADX 20-30 only
  - Time: 9:30am - 4pm ET, skip 12pm-1pm lunch
  - 60% in-sample / 40% out-of-sample (chronological split)
"""

import pandas as pd
import numpy as np

DATA_DIR = "D:/trading_pythonbacktest_data"
TRADES_CSV = f"{DATA_DIR}/vwap_reaction_adx_trades.csv"


def load_and_filter():
    df = pd.read_csv(TRADES_CSV)
    df["entry_dt"] = pd.to_datetime(df["entry_time"], utc=True).dt.tz_convert("America/New_York")

    # ADX filter: 20-30
    df = df[(df["adx"] >= 20) & (df["adx"] < 30)].copy()

    # Time filter: 9:30am - 4pm ET, skip 12-1pm
    hour = df["entry_dt"].dt.hour
    minute = df["entry_dt"].dt.minute
    entry_min = hour * 60 + minute

    rth = (entry_min >= 570) & (entry_min < 960)  # 9:30am - 4pm
    no_lunch = ~((entry_min >= 720) & (entry_min < 780))  # skip 12-1pm
    df = df[rth & no_lunch].copy()

    df = df.sort_values("entry_dt").reset_index(drop=True)
    return df


def report(df, label):
    n = len(df)
    if n == 0:
        print(f"\n{label}: 0 trades")
        return

    winners = df[df["pnl_dollars"] > 0]
    losers = df[df["pnl_dollars"] <= 0]

    wr = len(winners) / n * 100
    gp = winners["pnl_dollars"].sum() if len(winners) else 0
    gl = abs(losers["pnl_dollars"].sum()) if len(losers) else 1
    pf = gp / gl if gl > 0 else 999

    avg_pnl = df["pnl_dollars"].mean()
    total_pnl = df["pnl_dollars"].sum()

    cum = df["pnl_dollars"].cumsum()
    max_dd = float((cum - cum.cummax()).min())

    avg_win = winners["pnl_dollars"].mean() if len(winners) else 0
    avg_loss = losers["pnl_dollars"].mean() if len(losers) else 0

    date_range = f"{df['date'].iloc[0]} to {df['date'].iloc[-1]}"

    print()
    print("=" * 70)
    print(f"  {label}")
    print(f"  Period: {date_range}  |  {n} trades")
    print("=" * 70)
    print(f"  Win rate:       {wr:.1f}%")
    print(f"  Profit factor:  {pf:.2f}")
    print(f"  Expectancy:     ${avg_pnl:+,.0f}/trade")
    print(f"  Avg winner:     ${avg_win:+,.0f}")
    print(f"  Avg loser:      ${avg_loss:+,.0f}")
    print(f"  Total P&L:      ${total_pnl:+,.0f}")
    print(f"  Max drawdown:   ${max_dd:+,.0f}")
    print(f"  Return/DD:      {abs(total_pnl / max_dd):.2f}" if max_dd != 0 else "  Return/DD:      N/A")

    # By direction
    for d in ["long", "short"]:
        sub = df[df["direction"] == d]
        if len(sub) == 0:
            continue
        sn = len(sub)
        swr = len(sub[sub["pnl_dollars"] > 0]) / sn * 100
        spnl = sub["pnl_dollars"].sum()
        print(f"  {d.capitalize()+'s':<7s}:        {sn} trades | WR {swr:.1f}% | ${spnl:+,.0f}")

    # Monthly
    df = df.copy()
    df["month"] = df["entry_dt"].dt.to_period("M")
    monthly = df.groupby("month")["pnl_dollars"].agg(["sum", "count"])
    print(f"\n  {'Month':<10s} | {'Trades':>6s} | {'P&L':>10s}")
    print(f"  {'-'*35}")
    for m, row in monthly.iterrows():
        print(f"  {str(m):<10s} | {int(row['count']):6d} | ${row['sum']:>+9,.0f}")
    print(f"  {'-'*35}")
    print(f"  {'TOTAL':<10s} | {n:6d} | ${total_pnl:>+9,.0f}")


def main():
    df = load_and_filter()
    n = len(df)

    print(f"Total filtered trades: {n}")
    print(f"Filters: ADX 20-30 | 9:30am-4pm ET | no lunch (12-1pm)")

    # 60/40 chronological split
    split_idx = int(n * 0.60)
    is_df = df.iloc[:split_idx].copy()
    oos_df = df.iloc[split_idx:].copy()

    print(f"\nSplit: {split_idx} in-sample / {n - split_idx} out-of-sample")

    report(is_df, "IN-SAMPLE (60%)")
    report(oos_df, "OUT-OF-SAMPLE (40%)")
    report(df, "FULL DATASET (reference)")


if __name__ == "__main__":
    main()
