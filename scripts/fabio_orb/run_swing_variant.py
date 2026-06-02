"""Fabio ORB — SWING variant (CFD): hold until SL or TP, NO intraday force-close.

Same entry as locked config:
  - 8:30-9:00 ET ORB
  - N=4 consecutive 5-min closes above ORB_high
  - delta >= 300 on entry bar
  - skip 9:30 bucket
  - entries allowed for bars closing 9:00-14:00 ET (same entry window as locked)
  - SL = ORB_Low (static), TP = entry + 4R

DIFFERENCE vs locked:
  - NO 14:00 force-close. Position held across overnight/multi-day until SL or TP.
  - One position at a time (no new entry while holding — clean swing model).
  - Gap-aware fills: if a bar gaps THROUGH the level, fill at the bar open
    (worse for SL, better for TP).

Compares against the intraday locked version.
"""
from __future__ import annotations
import warnings; warnings.simplefilter("ignore")
from pathlib import Path
import numpy as np, pandas as pd

VOL_PARQUET = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
OUT_DIR = Path(__file__).parent / "results"
ET = "America/New_York"

ORB_START_HHMM, ORB_END_HHMM = 830, 900
ENTRY_END_HHMM = 1400          # entries allowed up to 14:00 (same as locked)
SKIP_BUCKET_HHMM = 930
N_CONFIRM = 4
DELTA_THR = 300.0
TP_RR = 4.0

TICK_SIZE, TICK_VALUE = 0.25, 5.0
DPP = TICK_VALUE / TICK_SIZE        # $20/pt
SLIP, COMM = 1, 5.0

MAX_HOLD_DAYS = 30                   # safety cap (a trade can't hold forever)


def load_continuous():
    """Full 24h continuous 5-min bar series + per-bar session/ORB context."""
    df = pd.read_parquet(VOL_PARQUET)
    agg = df.groupby("bar_open_time", as_index=False).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"),
    ).dropna(subset=["open", "high", "low", "close"])
    agg["bar_open_time"] = pd.to_datetime(agg["bar_open_time"])
    if agg["bar_open_time"].dt.tz is None:
        agg["bar_open_time"] = agg["bar_open_time"].dt.tz_localize("UTC")
    agg["close_et"] = agg["bar_open_time"].dt.tz_convert(ET) + pd.Timedelta(minutes=5)
    agg["hhmm"] = agg["close_et"].dt.hour * 100 + agg["close_et"].dt.minute
    agg["delta"] = agg["buy_vol"] - agg["sell_vol"]
    agg["session_date"] = agg["close_et"].dt.normalize().dt.tz_localize(None)
    agg = agg.sort_values("close_et").reset_index(drop=True)
    agg = agg[agg["session_date"] >= pd.Timestamp("2021-01-01")].copy().reset_index(drop=True)
    return agg


def compute_orb_per_session(agg):
    """ORB high/low for each session (8:30-9:00 ET bars)."""
    orb = {}
    in_orb = agg[(agg["hhmm"] > ORB_START_HHMM) & (agg["hhmm"] <= ORB_END_HHMM)]
    for sd, sub in in_orb.groupby("session_date"):
        orb[sd] = (float(sub["high"].max()), float(sub["low"].min()))
    return orb


def run_swing(agg, orb):
    close = agg["close"].to_numpy(np.float64)
    high  = agg["high"].to_numpy(np.float64)
    low   = agg["low"].to_numpy(np.float64)
    openp = agg["open"].to_numpy(np.float64)
    delta = agg["delta"].to_numpy(np.float64)
    hhmm  = agg["hhmm"].to_numpy()
    sdate = agg["session_date"].to_numpy()
    etime = agg["close_et"].to_numpy()
    n = len(agg)

    trades = []
    i = 0
    while i < n:
        sd = sdate[i]
        if sd not in orb:
            i += 1; continue
        oh, ol = orb[sd]

        # Entry conditions on this bar (must be in entry window, not ORB, not skip)
        if not (ORB_END_HHMM < hhmm[i] <= ENTRY_END_HHMM):
            i += 1; continue
        if hhmm[i] == SKIP_BUCKET_HHMM:
            i += 1; continue
        if i < N_CONFIRM - 1:
            i += 1; continue
        # N consecutive closes above ORB_high (all within same session)
        if not all(close[i-k] > oh for k in range(N_CONFIRM)):
            i += 1; continue
        # all confirmation bars must be same session (don't span overnight for entry)
        if not all(sdate[i-k] == sd for k in range(N_CONFIRM)):
            i += 1; continue
        if delta[i] < DELTA_THR:
            i += 1; continue
        ep = close[i]
        if ol >= ep:
            i += 1; continue

        # ----- ENTER -----
        entry_price = ep
        sl = ol
        tp = ep + TP_RR * (ep - ol)
        entry_time = etime[i]
        entry_sd = sd

        # ----- WALK FORWARD until SL or TP (no force-close) -----
        min_low = entry_price; max_high = entry_price
        xp = None; reason = None; xt = None
        for j in range(i + 1, n):
            # Safety: max hold days
            days_held = (pd.Timestamp(sdate[j]) - pd.Timestamp(entry_sd)).days
            if days_held > MAX_HOLD_DAYS:
                xp, reason, xt = close[j], "MAX_HOLD", etime[j]; break
            if low[j]  < min_low:  min_low  = low[j]
            if high[j] > max_high: max_high = high[j]
            hit_sl = low[j] <= sl
            hit_tp = high[j] >= tp
            if hit_sl and hit_tp:
                # Both in same bar — assume SL first (conservative)
                fill = min(openp[j], sl) if openp[j] < sl else sl   # gap-through worse
                xp, reason, xt = fill, "SL_TP", etime[j]; break
            if hit_sl:
                fill = openp[j] if openp[j] < sl else sl            # gap-down fills worse
                xp, reason, xt = fill, "SL", etime[j]; break
            if hit_tp:
                fill = openp[j] if openp[j] > tp else tp            # gap-up fills better
                xp, reason, xt = fill, "TP", etime[j]; break
        else:
            xp, reason, xt = close[-1], "EOD_LAST", etime[-1]; j = n - 1

        raw = xp - entry_price
        gross = raw * DPP
        net = gross - SLIP * TICK_SIZE * 2 * DPP - COMM
        hold_bars = j - i
        trades.append({
            "entry_time": pd.Timestamp(entry_time), "exit_time": pd.Timestamp(xt),
            "entry": entry_price, "exit": xp, "sl": sl, "tp": tp,
            "risk_pts": entry_price - ol, "raw_pts": raw,
            "gross_dollars": gross, "net_dollars": net,
            "mae_pts": entry_price - min_low, "mfe_pts": max_high - entry_price,
            "hold_bars": hold_bars, "hold_hours": hold_bars * 5 / 60.0,
            "reason": reason,
        })

        # One-position-at-a-time: resume scanning AFTER the exit bar
        i = j + 1

    return pd.DataFrame(trades)


def summarize(label, df):
    if df.empty:
        print(f"{label}: empty"); return
    n = len(df); wins = int((df["net_dollars"] > 0).sum())
    wd = df.loc[df["net_dollars"] > 0, "net_dollars"].sum()
    ld = -df.loc[df["net_dollars"] < 0, "net_dollars"].sum()
    pf = wd / ld if ld > 0 else float("inf")
    eq = df.sort_values("entry_time")["net_dollars"].cumsum()
    maxdd = (eq - eq.cummax()).min()
    print(f"\n=== {label} ===")
    print(f"  Trades:        {n}")
    print(f"  Win rate:      {100*wins/n:.1f}% ({wins}/{n})")
    print(f"  Net P&L:       ${df['net_dollars'].sum():,.0f}")
    print(f"  Profit factor: {pf:.3f}")
    print(f"  Max DD:        ${maxdd:,.0f}")
    print(f"  Net/|MDD|:     {df['net_dollars'].sum()/abs(maxdd):.2f}x")
    print(f"  Avg trade:     ${df['net_dollars'].mean():,.0f}")
    print(f"  Avg hold:      {df['hold_hours'].mean():.1f} hrs ({df['hold_bars'].mean():.0f} bars)")
    print(f"  Median hold:   {df['hold_hours'].median():.1f} hrs")
    print(f"  Max hold:      {df['hold_hours'].max():.1f} hrs ({df['hold_hours'].max()/24:.1f} days)")
    print(f"  Avg MAE:       {df['mae_pts'].mean():.1f} pts (${df['mae_pts'].mean()*DPP:,.0f})")
    print(f"  p95 MAE:       {df['mae_pts'].quantile(0.95):.1f} pts (${df['mae_pts'].quantile(0.95)*DPP:,.0f})")
    print(f"  Worst MAE:     {df['mae_pts'].max():.1f} pts (${df['mae_pts'].max()*DPP:,.0f})")
    print(f"  Exits: " + ", ".join(f"{r}={c}" for r, c in df['reason'].value_counts().items()))
    # IS/OOS split
    df2 = df.sort_values("entry_time").reset_index(drop=True)
    cut = int(len(df2) * 0.6)
    for ph, sub in [("IS", df2.iloc[:cut]), ("OOS", df2.iloc[cut:])]:
        w = (sub["net_dollars"] > 0).sum()
        wdd = sub.loc[sub["net_dollars"]>0,"net_dollars"].sum()
        ldd = -sub.loc[sub["net_dollars"]<0,"net_dollars"].sum()
        p = wdd/ldd if ldd>0 else 99
        print(f"  {ph}: n={len(sub)} WR={w/len(sub)*100:.1f}% net=${sub['net_dollars'].sum():,.0f} PF={p:.2f}")
    return df


def main():
    print("Loading continuous 24h bars...", flush=True)
    agg = load_continuous()
    print(f"  {len(agg):,} bars  {agg['session_date'].min().date()} -> {agg['session_date'].max().date()}")
    orb = compute_orb_per_session(agg)
    print(f"  {len(orb)} sessions with ORB")

    print("\nRunning SWING variant (hold to SL/TP, no force-close)...")
    df_swing = run_swing(agg, orb)
    summarize("SWING: hold until SL or TP (one position at a time)", df_swing)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_swing.to_csv(OUT_DIR / "fabio_swing_trades.csv", index=False)

    # Compare against intraday locked (load existing trades file)
    locked_path = Path("D:/trading_pythonbacktest_data/fabio orb/trades_final_modeA.csv")
    if locked_path.exists():
        locked = pd.read_csv(locked_path)
        print(f"\n=== INTRADAY LOCKED (force-close 14:00) for comparison ===")
        n = len(locked); wins = (locked['net_dollars']>0).sum()
        wd = locked.loc[locked['net_dollars']>0,'net_dollars'].sum()
        ld = -locked.loc[locked['net_dollars']<0,'net_dollars'].sum()
        eq = locked['net_dollars'].cumsum(); mdd = (eq-eq.cummax()).min()
        print(f"  Trades: {n}  WR: {wins/n*100:.1f}%  Net: ${locked['net_dollars'].sum():,.0f}  "
              f"PF: {wd/ld:.3f}  MDD: ${mdd:,.0f}  Net/MDD: {locked['net_dollars'].sum()/abs(mdd):.2f}x")

    print(f"\nSaved: {OUT_DIR / 'fabio_swing_trades.csv'}")


if __name__ == "__main__":
    main()
