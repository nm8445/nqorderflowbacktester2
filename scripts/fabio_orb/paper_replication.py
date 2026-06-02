"""Fabio IVB Model — paper-exact replication.

Source: "Fabio IVB Model, The Institutional Protocol" (Matteo Conti, 2026 v1.0)
Located: D:/trading_pythonbacktest_data/fabio orb/Fabio IVB Model, The Institutional Protocol.pdf

PDF parameters (verified with author — "NY time" means "NY SESSION", so shift +1 hour vs PDF text):
  ORB_Start_H_NY = 8 -> 09:30 ET (NY session open)
  ORB_Start_M_NY = 30
  ORB_Dur_Min    = 30 -> ORB window 09:30-10:00 ET
  Trade_End_H_NY = 14 -> 15:00 ET (NY session, 1 hr before regular close)
  Trade_End_M_NY = 0
  TP_RR_Ratio    = 1.0  (paper uses 1R, NOT 4R like our locked variant)
  Num_Contracts  = 1
  DeltaThreshold = 200 (paper's value — we SWEEP this)
  UseCumulativeDelta = false (single 5-min bar delta, not cumulative)
  CumDeltaThreshold = 500 (irrelevant since UseCumulativeDelta=false)

Other paper rules (from PDF text):
  - "On a 5-min close above the range high, with a delta reading above the threshold,
     a single long position is opened at the close. Only one long entry per session."
  - N_ConfirmCloses = 1 (one close past ORB_High, not 4 like locked variant)
  - SL = ORB_Low (static, set at entry)
  - TP = entry + 1.0 × (entry - ORB_Low)
  - "Any open position is flattened on the bar that crosses 15:00 ET"
  - No SkipBucket (no 09:30 skip rule)
  - Long-only
  - "TP and SL are re-armed every bar while the position is open" — both fixed at entry,
    just means broker keeps them resting

DELTA: Paper's MultiCharts implementation likely used uptick/downtick approximation.
We use REAL aggressor-classified buy_vol-sell_vol (in contracts) from MBP-1 data.
This is the MAIN DIFFERENCE from the paper — better delta accounting may shift the
optimal threshold.

Paper-reported results to match:
  823 trades, 58.3% WR, +$201/trade avg, PF 1.31, MaxDD $22,215
  Avg winner $1,474, avg loser $1,580, ratio 0.93
  Sample 2021-01-01 -> 2026-04-16
"""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pandas as pd

VOL_PARQUET = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ET = "America/New_York"

# Paper-exact times (with "NY session" interpretation: +1 hour vs PDF text)
ORB_START_HHMM = 930    # ORB open  09:30 ET
ORB_END_HHMM   = 1000   # ORB close 10:00 ET
TRADE_END_HHMM = 1500   # Force-flat 15:00 ET
TP_RR_RATIO = 1.0       # Paper-exact
N_CONFIRM = 1           # Paper-exact (single close above ORB_High)

TICK_SIZE = 0.25
TICK_VALUE = 5.0
DOLLARS_PER_PT = TICK_VALUE / TICK_SIZE   # 20
SLIP_TICKS_PER_SIDE = 1
COMM_RT = 5.0

# DELTA THRESHOLD SWEEP — paper used 200, but with REAL delta the optimum may differ
DELTA_THRESHOLDS = [0, 50, 100, 150, 200, 250, 300, 400, 500, 600, 750, 1000, 1500]


def load_bars():
    """Load 5-min volumetric parquet, aggregate to bar-level OHLC + total delta."""
    print(f"Loading {VOL_PARQUET.name}...")
    df = pd.read_parquet(VOL_PARQUET)
    df["bar_open_time"] = pd.to_datetime(df["bar_open_time"]).dt.tz_convert(ET)
    agg = df.groupby("bar_open_time", as_index=False).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"),
    )
    agg["close_et"] = agg["bar_open_time"] + pd.Timedelta(minutes=5)
    agg["hhmm"] = agg["close_et"].dt.hour * 100 + agg["close_et"].dt.minute
    agg["session_date"] = agg["close_et"].dt.normalize().dt.tz_localize(None)
    agg["delta"] = agg["buy_vol"] - agg["sell_vol"]
    # Drop NaN-OHLC placeholder rows (build_volumetric inserts them for empty 5-min buckets)
    agg = agg.dropna(subset=["open", "high", "low", "close"]).copy()
    # Filter to dates >= 2021-01-01 to match paper's sample range
    start = pd.Timestamp("2021-01-01")
    agg = agg[agg["session_date"] >= start].copy()
    print(f"  {len(agg):,} 5-min bars (NaN-filtered), range {agg['close_et'].min()} -> {agg['close_et'].max()}")
    return agg


def run_strategy(bars: pd.DataFrame, delta_threshold: float):
    """Paper-exact engine. Returns list of trade dicts."""
    trades = []
    for sd, day in bars.groupby("session_date"):
        # ORB: bars whose close is in (09:30, 10:00] ET
        orb_bars = day[(day["hhmm"] > ORB_START_HHMM) & (day["hhmm"] <= ORB_END_HHMM)]
        if len(orb_bars) == 0: continue
        orb_high = float(orb_bars["high"].max())
        orb_low  = float(orb_bars["low"].min())

        # Post-ORB bars (close > 10:00 and <= 15:00)
        post = day[(day["hhmm"] > ORB_END_HHMM) & (day["hhmm"] <= TRADE_END_HHMM)].reset_index(drop=True)
        if len(post) == 0: continue

        # Find entry: first bar with close > ORB_High AND delta >= threshold
        entry_idx = -1
        for i in range(len(post)):
            bar = post.iloc[i]
            if bar["hhmm"] > TRADE_END_HHMM: break
            if bar["close"] <= orb_high: continue
            if bar["delta"] < delta_threshold: continue
            if orb_low >= bar["close"]: continue   # sanity
            entry_idx = i
            break
        if entry_idx < 0: continue

        eb = post.iloc[entry_idx]
        entry_price = float(eb["close"])
        sl = orb_low
        risk = entry_price - sl
        tp = entry_price + TP_RR_RATIO * risk

        # Walk forward to exit
        exit_price = None; exit_reason = None
        for j in range(entry_idx + 1, len(post)):
            bar = post.iloc[j]
            hit_sl = bar["low"]  <= sl
            hit_tp = bar["high"] >= tp
            # Paper: if both touched same bar, SL fills first (conservative)
            if hit_sl and hit_tp:
                exit_price = sl; exit_reason = "SL_TP"
                break
            if hit_sl:
                exit_price = sl; exit_reason = "SL"
                break
            if hit_tp:
                exit_price = tp; exit_reason = "TP"
                break
            if int(bar["hhmm"]) >= TRADE_END_HHMM:
                exit_price = float(bar["close"]); exit_reason = "EOD"
                break
        if exit_price is None:
            last = post.iloc[-1]
            exit_price = float(last["close"]); exit_reason = "EOD_LAST"

        gross_pts = exit_price - entry_price
        net_pts   = gross_pts - 2 * SLIP_TICKS_PER_SIDE * TICK_SIZE
        net_dollars = net_pts * DOLLARS_PER_PT - COMM_RT
        trades.append({
            "session_date": sd, "entry_time": eb["close_et"],
            "entry_price": entry_price, "exit_price": exit_price,
            "sl": sl, "tp": tp, "risk_pts": risk,
            "delta_at_entry": float(eb["delta"]),
            "net_pts": net_pts, "net_dollars": net_dollars, "reason": exit_reason,
        })
    return pd.DataFrame(trades)


def stats(df: pd.DataFrame):
    n = len(df)
    if n == 0: return None
    p = df["net_dollars"].values
    w = p[p > 0]; l = p[p < 0]
    pf = w.sum() / abs(l.sum()) if len(l) > 0 else 99.0
    cum = p.cumsum()
    mdd = float((cum - np.maximum.accumulate(cum)).min())
    return dict(
        n=n,
        wr=round(len(w) / n * 100, 1),
        net=round(float(p.sum()), 0),
        avg_trade=round(float(p.mean()), 1),
        avg_win=round(float(w.mean()), 0) if len(w) > 0 else 0,
        avg_loss=round(float(l.mean()), 0) if len(l) > 0 else 0,
        pf=round(pf, 3),
        mdd=round(mdd, 0),
        tp_pct=round((df["reason"] == "TP").mean() * 100, 1),
        sl_pct=round((df["reason"].isin(["SL", "SL_TP"])).mean() * 100, 1),
        eod_pct=round((df["reason"].isin(["EOD", "EOD_LAST"])).mean() * 100, 1),
    )


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Starting paper replication sweep...")
    bars = load_bars()

    print(f"\nPaper reference (PDF):")
    print(f"  823 trades, 58.3% WR, +$201/trade, PF 1.31, MaxDD $22,215")
    print(f"  Sample: 2021-01-01 -> 2026-04-16")
    print(f"  DeltaThreshold = 200 in paper (we sweep with REAL delta)")

    print(f"\n{'='*100}")
    print(f"  DELTA THRESHOLD SWEEP — paper-exact engine, REAL aggressor delta")
    print(f"{'='*100}")
    print(f"  ORB 09:30-10:00 ET | N_confirm=1 | TP=1R | Force-close 15:00 ET")
    print(f"  Long-only, 1 contract, slip 1 tick/side, $5 commission RT")
    print()
    print(f"  {'delta':>5} {'n':>4} {'WR':>5} {'avg/tr':>8} {'avg_win':>9} {'avg_loss':>10} "
          f"{'net':>11} {'PF':>6} {'MDD':>11} {'TP%':>5} {'SL%':>5} {'EOD%':>5}")
    print(f"  {'-'*5:>5} {'-'*4:>4} {'-'*5:>5} {'-'*8:>8} {'-'*9:>9} {'-'*10:>10} "
          f"{'-'*11:>11} {'-'*6:>6} {'-'*11:>11} {'-'*5:>5} {'-'*5:>5} {'-'*5:>5}")

    all_rows = []
    for dt_thr in DELTA_THRESHOLDS:
        trades = run_strategy(bars, dt_thr)
        s = stats(trades)
        if s is None:
            print(f"  {dt_thr:>5} no trades")
            continue
        all_rows.append({"delta_threshold": dt_thr, **s})
        print(f"  {dt_thr:>5} {s['n']:>4} {s['wr']:>4.1f}% ${s['avg_trade']:>+6,.0f} "
              f"${s['avg_win']:>+7,.0f} ${s['avg_loss']:>+8,.0f} ${s['net']:>+9,.0f} "
              f"{s['pf']:>6.3f} ${s['mdd']:>+9,.0f} {s['tp_pct']:>4.1f}% {s['sl_pct']:>4.1f}% {s['eod_pct']:>4.1f}%")

        # Save trade log for delta=200 (paper-direct comparison) and delta=300 (alt)
        if dt_thr in (200, 300):
            trades.to_csv(OUT_DIR / f"paper_replication_delta{dt_thr}_trades.csv", index=False)

    df_summary = pd.DataFrame(all_rows)
    df_summary.to_csv(OUT_DIR / "paper_replication_delta_sweep.csv", index=False)
    print(f"\nSaved sweep: {OUT_DIR / 'paper_replication_delta_sweep.csv'}")

    # Paper comparison
    print(f"\n=== COMPARISON: PAPER vs OUR REAL-DELTA RESULT at delta=200 ===")
    paper = {"n":823, "wr":58.3, "avg_trade":201, "avg_win":1474, "avg_loss":-1580,
             "pf":1.31, "mdd":-22215}
    ours = df_summary[df_summary["delta_threshold"] == 200].iloc[0].to_dict() if 200 in df_summary["delta_threshold"].values else None
    if ours:
        print(f"  {'metric':<14} {'paper':>15} {'ours @ 200':>15} {'delta':>15}")
        for k in ["n", "wr", "avg_trade", "avg_win", "avg_loss", "pf", "mdd"]:
            pv = paper[k]; ov = ours[k]
            delta = ov - pv
            sign = "+" if delta > 0 else ""
            print(f"  {k:<14} {pv:>15} {ov:>15} {sign}{delta:>14}")


if __name__ == "__main__":
    main()
