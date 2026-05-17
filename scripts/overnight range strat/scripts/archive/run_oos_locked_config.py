"""Out-of-sample test of the locked B2 config.

In-sample window:  2020-12-04 -> 2024-12-31  (existing trades parquet)
OOS window:        2025-01-01 -> latest available with MQ levels coverage

Generates OOS trades parquet using the same study logic, then applies the
locked config (B2 X=1.25 N=5 D=70 strict=True BAND_K=0.25 TP=SL=1.0 chained
Mode 1) and reports stats vs the in-sample baseline.

Locked config (from in-sample optimization):
  trades=657  total=+3,471  PF=1.30  Sharpe=2.09  WR=54.3%
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from range_break_entry_signal_study import (  # noqa: E402
    load_range_per_day, load_mq_levels, load_5min_features,
    levels_for_date, process_one_day,
)
from range_break_entry_summary import (  # noqa: E402
    apply_filters, mode1_chained_dedupe, trade_pnls_vectorized,
)

PARQUET_DIR = Path(__file__).parent / "parquets"
PARQUET_DIR.mkdir(exist_ok=True)
TRADES_OOS  = PARQUET_DIR / "entry_signal_trades_oos.parquet"
TRADES_IS   = PARQUET_DIR / "entry_signal_trades.parquet"

OOS_START = dt.date(2025, 1, 1)
OOS_END   = dt.date(2026, 5, 7)    # exclusive — covers Jan 2025 to 2026-05-06 inclusive

# Locked config
VARIANT, X, N, D, STRICT, BAND_K = "B2", 1.25, 5, 70, True, 0.25
TP_M, SL_M = 1.0, 1.0


def build_oos_trades():
    """Generate OOS trades parquet using the existing study logic."""
    print("=" * 100)
    print(f"BUILDING OOS TRADES  ({OOS_START} -> {OOS_END})")
    print("=" * 100)

    print("loading per-day range data...")
    rng = load_range_per_day()
    print(f"  range parquet: {len(rng)} days")

    print("loading MenthorQ levels...")
    mq = load_mq_levels()
    print(f"  MQ levels: {len(mq)} days  (latest: {max(mq.index)})")

    print(f"loading volumetric 5-min for OOS window...")
    bars_all, levels_all = load_5min_features((OOS_START, OOS_END))

    bars_by_day = dict(list(bars_all.groupby("session_date", sort=True)))
    levels_by_day = dict(list(levels_all.groupby("session_date", sort=True)))
    print(f"  {len(bars_by_day)} session days to process")

    mq_dates_sorted = sorted(mq.index.tolist())
    def prior_mq_day(session_date):
        prev = None
        for md in mq_dates_sorted:
            if md < session_date: prev = md
            else: break
        return prev

    all_trades = []
    skipped_no_mq = 0
    t0 = time.time()
    for i, (d, bars_day) in enumerate(bars_by_day.items(), 1):
        if d not in rng.index:
            continue
        rng_row = rng.loc[d]
        levels_day = levels_by_day.get(d, pd.DataFrame())
        prior_d = prior_mq_day(d)
        if prior_d is None:
            skipped_no_mq += 1
            continue
        mq_levels = levels_for_date(mq, prior_d)
        if mq_levels.size == 0:
            skipped_no_mq += 1
            continue
        try:
            day_trades = process_one_day(d, bars_day, levels_day, rng_row, mq_levels)
            all_trades.extend(day_trades)
        except Exception as e:
            print(f"  ! {d}: {type(e).__name__}: {e}")
        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i}/{len(bars_by_day)}  elapsed={elapsed:.0f}s  trades={len(all_trades)}")

    print(f"  total: {len(all_trades)} OOS trade candidates  ({skipped_no_mq} days skipped no-MQ)")
    if not all_trades:
        return None

    df = pd.DataFrame(all_trades)
    df.to_parquet(TRADES_OOS, compression="zstd", index=False)
    print(f"\nwrote {TRADES_OOS}  ({len(df):,} trades, {len(df.columns)} cols)")
    return df


def apply_locked_and_stats(label: str, df: pd.DataFrame) -> dict:
    print()
    print("=" * 100)
    print(f"APPLYING LOCKED CONFIG TO {label}")
    print(f"  B2 X={X} N={N} D={D} strict={STRICT} BAND_K={BAND_K} TP=SL={TP_M} chained-Mode-1")
    print("=" * 100)

    filtered = apply_filters(df, VARIANT, X, N, D, STRICT, BAND_K)
    print(f"  after filters: {len(filtered):,} trades")
    deduped  = mode1_chained_dedupe(filtered, TP_M, SL_M)
    print(f"  after chained Mode 1 dedupe: {len(deduped):,} trades")
    if len(deduped) == 0:
        return {}
    deduped = deduped.copy()
    deduped["pnl"] = trade_pnls_vectorized(deduped, TP_M, SL_M)
    deduped["date"] = pd.to_datetime(deduped["date"]).dt.date

    pnl = deduped["pnl"].values
    long_mask = (deduped["direction"] == "LONG").values
    short_mask = (deduped["direction"] == "SHORT").values
    wins = pnl > 0
    pos = pnl[pnl > 0].sum(); neg = -pnl[pnl < 0].sum()
    pf = pos / neg if neg > 0 else (np.inf if pos > 0 else 0.0)
    daily = pd.Series(pnl, index=deduped["date"].values).groupby(level=0).sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0
    eq = np.cumsum(pnl); peak = np.maximum.accumulate(eq); max_dd = (eq - peak).min()

    long_pnl  = pnl[long_mask]
    short_pnl = pnl[short_mask]

    summary = {
        "label": label,
        "n": len(deduped),
        "n_long": int(long_mask.sum()),
        "n_short": int(short_mask.sum()),
        "total": pnl.sum(),
        "mean": pnl.mean(),
        "wr": wins.mean(),
        "wr_long":  (long_pnl  > 0).mean() if len(long_pnl)  else float("nan"),
        "wr_short": (short_pnl > 0).mean() if len(short_pnl) else float("nan"),
        "pf": pf,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "long_total":  long_pnl.sum(),
        "short_total": short_pnl.sum(),
        "date_min": deduped["date"].min(),
        "date_max": deduped["date"].max(),
    }
    return summary


def print_summary(s: dict):
    print(f"  {s['label']:<14}  n={s['n']:>4}  L={s['n_long']:>4}/S={s['n_short']:>4}  "
          f"total={s['total']:>+8.1f}  mean={s['mean']:>+5.2f}  "
          f"WR={s['wr']:>5.1%}  WR_L={s['wr_long']:>5.1%}  WR_S={s['wr_short']:>5.1%}  "
          f"PF={s['pf']:>5.2f}  Sharpe={s['sharpe']:>+5.2f}  MDD={s['max_dd']:>+8.0f}")
    print(f"  {'  longs':<14}  total={s['long_total']:>+8.1f}    "
          f"{'  shorts':<14}  total={s['short_total']:>+8.1f}")


def main():
    # 1) Build OOS trades parquet
    if TRADES_OOS.exists():
        print(f"OOS trades parquet exists: {TRADES_OOS}")
        oos = pd.read_parquet(TRADES_OOS)
        print(f"  {len(oos):,} trades, date range {pd.to_datetime(oos['date']).dt.date.min()} -> {pd.to_datetime(oos['date']).dt.date.max()}")
        rebuild = input("Rebuild from scratch? [y/N]: ").strip().lower() == "y"
        if rebuild:
            oos = build_oos_trades()
    else:
        oos = build_oos_trades()

    if oos is None or oos.empty:
        print("no OOS trades produced; aborting comparison")
        return

    # 2) Apply locked config to OOS
    s_oos = apply_locked_and_stats("OOS 2025", oos)

    # 3) Compare against in-sample
    print("\nloading in-sample trades for comparison...")
    is_df = pd.read_parquet(TRADES_IS)
    print(f"  {len(is_df):,} IS trades")
    s_is = apply_locked_and_stats("IS 2020-2024", is_df)

    # 4) Side-by-side
    print()
    print("=" * 100)
    print("SIDE-BY-SIDE — IN-SAMPLE vs OUT-OF-SAMPLE")
    print("=" * 100)
    print_summary(s_is)
    print_summary(s_oos)

    # 5) Verdict
    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    deg = []
    def chk(name, is_v, oos_v, lower_better=False):
        diff = oos_v - is_v
        d = (oos_v / is_v - 1) * 100 if is_v != 0 else float("nan")
        emoji = "OK" if (lower_better and diff <= 0) or (not lower_better and diff >= 0) else "DOWN"
        deg.append((name, is_v, oos_v, d))
        print(f"  {name:<10}  IS={is_v:>+8.3f}  OOS={oos_v:>+8.3f}  delta={d:>+6.1f}%   {emoji}")
    chk("PF",     s_is["pf"], s_oos["pf"])
    chk("Sharpe", s_is["sharpe"], s_oos["sharpe"])
    chk("WR",     s_is["wr"], s_oos["wr"])
    chk("Mean",   s_is["mean"], s_oos["mean"])
    chk("MDD",    abs(s_is["max_dd"]), abs(s_oos["max_dd"]), lower_better=True)
    print()


if __name__ == "__main__":
    sys.exit(main())
