"""Analyze the 54 entries blocked by the 5%ers news filter.

Are we leaving money on the table, or is the rule SAVING us from bad trades?
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import bisect
import calendar
from datetime import datetime, timedelta
import zoneinfo

ROOT = Path(__file__).resolve().parents[2]
TRADES_CSV = ROOT / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"
ET = zoneinfo.ZoneInfo("America/New_York")
START_DATE = datetime(2020, 11, 1, tzinfo=ET)
END_DATE = datetime(2026, 6, 1, tzinfo=ET)
BLACKOUT_MIN = 2


def nth_business_day(year, month, n):
    cal = calendar.Calendar()
    bdays = [d for d in cal.itermonthdates(year, month) if d.month == month and d.weekday() < 5]
    return bdays[n - 1] if 0 < n <= len(bdays) else None


def nth_weekday(year, month, weekday, n):
    cal = calendar.Calendar()
    days = [d for d in cal.itermonthdates(year, month) if d.month == month and d.weekday() == weekday]
    return days[n - 1] if 0 < n <= len(days) else None


def generate_calendar():
    events = []
    fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
    for year in range(2020, 2027):
        for m in fomc_months:
            d = nth_weekday(year, m, 2, 4)
            if d:
                events.append(("FOMC", datetime(d.year, d.month, d.day, 14, 0, tzinfo=ET)))
                minutes = d + timedelta(days=21)
                events.append(("FOMC_Minutes", datetime(minutes.year, minutes.month, minutes.day, 14, 0, tzinfo=ET)))
        for m in range(1, 13):
            for bd, name in [(1, "ISM_Mfg"), (3, "ISM_Svc"), (8, "JOLTS"),
                             (17, "NewHomeSales"), (18, "ConsConf"), (19, "PendingHomes")]:
                d = nth_business_day(year, m, bd)
                if d:
                    events.append((name, datetime(d.year, d.month, d.day, 10, 0, tzinfo=ET)))
    events = [(n, t) for n, t in events if START_DATE <= t <= END_DATE]
    events.sort(key=lambda x: x[1])
    return events


def main():
    events = generate_calendar()
    event_ns = np.array([pd.Timestamp(ev[1]).value for ev in events], dtype=np.int64)
    event_names = [ev[0] for ev in events]
    blackout_ns = BLACKOUT_MIN * 60 * 1_000_000_000

    df = pd.read_csv(TRADES_CSV)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True).dt.tz_convert(ET)
    df["pnl_$_1mnq"] = df["pnl_$"] * 0.1   # convert from 1 NQ basis to 1 MNQ basis

    # Filter
    trade_ns = np.array([pd.Timestamp(t).value for t in df["entry_ts"]], dtype=np.int64)
    blocked_idx = []
    blocked_events = []
    for i, ts_n in enumerate(trade_ns):
        idx = np.searchsorted(event_ns, ts_n)
        for j in [idx - 1, idx]:
            if 0 <= j < len(event_ns) and abs(ts_n - event_ns[j]) <= blackout_ns:
                blocked_idx.append(i)
                blocked_events.append(event_names[j])
                break

    blocked = df.iloc[blocked_idx].copy()
    blocked["blocked_by"] = blocked_events
    kept = df.drop(df.index[blocked_idx]).copy()

    print(f"\n{'='*80}")
    print(f"BLOCKED ENTRIES ANALYSIS — {len(blocked)} trades")
    print(f"{'='*80}")
    print(f"At 1 MNQ basis (multiply by 4 for $40K equiv, by 10 for full NQ):\n")

    # Overall stats
    n_wins = (blocked["pnl_$_1mnq"] > 0).sum()
    n_losses = (blocked["pnl_$_1mnq"] < 0).sum()
    n_zero = (blocked["pnl_$_1mnq"] == 0).sum()
    total_pnl = blocked["pnl_$_1mnq"].sum()
    avg_pnl = blocked["pnl_$_1mnq"].mean()
    win_sum = blocked.loc[blocked["pnl_$_1mnq"] > 0, "pnl_$_1mnq"].sum()
    loss_sum = blocked.loc[blocked["pnl_$_1mnq"] < 0, "pnl_$_1mnq"].sum()

    print(f"  WINS:    {n_wins:3d}  total ${win_sum:>8.0f}   avg ${win_sum/max(n_wins,1):>6.0f}")
    print(f"  LOSSES:  {n_losses:3d}  total ${loss_sum:>8.0f}   avg ${loss_sum/max(n_losses,1):>6.0f}")
    print(f"  ZERO:    {n_zero:3d}")
    print(f"  WR:      {n_wins/len(blocked)*100:.1f}%")
    print(f"  NET:     ${total_pnl:>8.0f}   avg ${avg_pnl:>6.0f}")

    # Compare to overall kept trades
    print(f"\n{'='*80}")
    print(f"vs KEPT (non-blocked) — {len(kept)} trades")
    print(f"{'='*80}")
    n_wins_k = (kept["pnl_$_1mnq"] > 0).sum()
    print(f"  WR:      {n_wins_k/len(kept)*100:.1f}%")
    print(f"  Total:   ${kept['pnl_$_1mnq'].sum():>10,.0f}")
    print(f"  Avg/tr:  ${kept['pnl_$_1mnq'].mean():>6.0f}")

    # Per-strategy breakdown of blocked
    print(f"\n{'='*80}")
    print(f"BLOCKED BY STRATEGY")
    print(f"{'='*80}")
    for s in ["OD", "B2", "RV", "FB"]:
        sub = blocked[blocked["strat"] == s]
        if len(sub) == 0:
            print(f"  {s}: 0 blocked")
            continue
        n_w = (sub["pnl_$_1mnq"] > 0).sum()
        n_l = (sub["pnl_$_1mnq"] < 0).sum()
        wr = n_w / len(sub) * 100 if len(sub) > 0 else 0
        net = sub["pnl_$_1mnq"].sum()
        avg = sub["pnl_$_1mnq"].mean()
        # Compare to that strat's overall WR
        all_s = df[df["strat"] == s]
        all_wr = (all_s["pnl_$_1mnq"] > 0).mean() * 100
        all_avg = all_s["pnl_$_1mnq"].mean()
        print(f"  {s}: {len(sub)} trades  WR {wr:.1f}% (vs {all_wr:.1f}% baseline)  "
              f"NET ${net:>+7.0f}  avg ${avg:>+6.0f} (vs ${all_avg:>+6.0f} baseline)")

    # Top 10 wins and losses among blocked
    print(f"\n{'='*80}")
    print(f"TOP 5 BLOCKED WINS (money left on table)")
    print(f"{'='*80}")
    top_wins = blocked.nlargest(5, "pnl_$_1mnq")[["entry_ts", "strat", "direction", "pnl_$_1mnq", "blocked_by"]]
    print(top_wins.to_string(index=False))

    print(f"\n{'='*80}")
    print(f"TOP 5 BLOCKED LOSSES (saved from disasters)")
    print(f"{'='*80}")
    top_losses = blocked.nsmallest(5, "pnl_$_1mnq")[["entry_ts", "strat", "direction", "pnl_$_1mnq", "blocked_by"]]
    print(top_losses.to_string(index=False))

    # By event type
    print(f"\n{'='*80}")
    print(f"BLOCKED BY EVENT TYPE")
    print(f"{'='*80}")
    by_event = blocked.groupby("blocked_by").agg(
        n=("pnl_$_1mnq", "count"),
        wins=("pnl_$_1mnq", lambda x: (x > 0).sum()),
        net=("pnl_$_1mnq", "sum"),
        avg=("pnl_$_1mnq", "mean"),
    ).round(0).astype({"n": int, "wins": int})
    by_event["wr%"] = (by_event["wins"] / by_event["n"] * 100).round(1)
    print(by_event.to_string())

    # Verdict
    print(f"\n{'='*80}")
    print(f"VERDICT")
    print(f"{'='*80}")
    if total_pnl > 0:
        print(f"  NET on blocked trades: ${total_pnl:+,.0f} at 1 MNQ basis")
        print(f"  -> The rule COSTS you ${total_pnl:,.0f} over {len(blocked)} trades")
        print(f"     Scale by your live MNQ: at scale 1.25 (your funded rec),")
        print(f"     average across 4 strats: 2.5 MNQ * 0.1 = 0.25x, so ~${total_pnl * 0.25:,.0f} actual")
    else:
        print(f"  NET on blocked trades: ${total_pnl:+,.0f} at 1 MNQ basis")
        print(f"  -> The rule SAVES you ${-total_pnl:,.0f} over {len(blocked)} trades")
        print(f"     You're NOT leaving money on the table — you're avoiding losers.")


if __name__ == "__main__":
    main()
