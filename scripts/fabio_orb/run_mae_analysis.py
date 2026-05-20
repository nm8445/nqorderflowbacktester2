"""
MAE / MFE analysis for the Fabio ORB strategy.

For TP=4.0R, delta=300 contracts, compare N=1 vs N=4 confirmation:
  - per-trade MAE (max adverse excursion in $ from entry to exit)
  - per-trade MFE (max favorable excursion)
  - distribution of MAE for winning vs losing trades
"""
from __future__ import annotations
import warnings; warnings.simplefilter("ignore")
from pathlib import Path
import numpy as np, pandas as pd

VOL_PARQUET = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
ET = "America/New_York"
ORB_START_HHMM, ORB_END_HHMM, TRADE_END_HHMM = 830, 900, 1400
TICK_SIZE, TICK_VALUE = 0.25, 5.0
DPP = TICK_VALUE / TICK_SIZE
SLIP, COMM = 1, 5.0


def load_days():
    df = pd.read_parquet(VOL_PARQUET)
    agg = df.groupby("bar_open_time", as_index=False).agg(
        open=("open","first"), high=("high","max"), low=("low","min"),
        close=("close","last"), buy_vol=("buy_vol","sum"), sell_vol=("sell_vol","sum"),
    ).dropna(subset=["open","high","low","close"])
    agg["close_et"] = agg["bar_open_time"] + pd.Timedelta(minutes=5)
    agg["hhmm"] = agg["close_et"].dt.hour*100 + agg["close_et"].dt.minute
    agg = agg[(agg["hhmm"] > ORB_START_HHMM) & (agg["hhmm"] <= TRADE_END_HHMM)].copy()
    agg["delta"] = agg["buy_vol"] - agg["sell_vol"]
    agg["session_date"] = agg["close_et"].dt.normalize()
    days = {}
    for sd, sub in agg.groupby("session_date"):
        sub = sub.sort_values("close_et").reset_index(drop=True)
        in_orb = sub[(sub["hhmm"] > ORB_START_HHMM) & (sub["hhmm"] <= ORB_END_HHMM)]
        if in_orb.empty: continue
        post = sub[(sub["hhmm"] > ORB_END_HHMM) & (sub["hhmm"] <= TRADE_END_HHMM)]
        if post.empty: continue
        days[pd.Timestamp(sd).tz_localize(None)] = {
            "orb_high": float(in_orb["high"].max()),
            "orb_low":  float(in_orb["low"].min()),
            "hhmm":  post["hhmm"].to_numpy(),
            "high":  post["high"].to_numpy(dtype=np.float64),
            "low":   post["low"].to_numpy(dtype=np.float64),
            "close": post["close"].to_numpy(dtype=np.float64),
            "delta": post["delta"].to_numpy(dtype=np.float64),
            "close_et": post["close_et"].to_numpy(),
        }
    return days


def run_day(day, n_confirm, tp_rr=4.0, delta_thr=300.0):
    oh = day["orb_high"]; ol = day["orb_low"]
    hhmm = day["hhmm"]; high = day["high"]; low = day["low"]
    close = day["close"]; delta = day["delta"]; etime = day["close_et"]
    n = len(hhmm)

    entry_idx = -1
    for i in range(n_confirm - 1, n):
        if hhmm[i] > TRADE_END_HHMM: break
        ok = all(close[i-k] > oh for k in range(n_confirm))
        if not ok: continue
        if delta[i] < delta_thr: continue
        ep = close[i]
        if ol >= ep: continue
        entry_idx = i; entry_price = ep
        sl = ol; tp = ep + tp_rr * (ep - ol)
        entry_time = etime[i]
        break

    if entry_idx < 0: return None

    # Track MAE/MFE on every bar from entry+1 to exit
    min_low = low[entry_idx]   # include entry bar low to capture any wick
    max_high = high[entry_idx]
    # But entry was at the CLOSE of entry_idx — so price-action AFTER entry is what counts.
    # Restart MAE/MFE counters from entry close.
    min_low = entry_price
    max_high = entry_price

    for j in range(entry_idx + 1, n):
        if low[j]  < min_low:  min_low  = low[j]
        if high[j] > max_high: max_high = high[j]
        if hhmm[j] >= TRADE_END_HHMM:
            xp, reason = close[j], "EOD"; xt = etime[j]; break
        hit_sl = low[j] <= sl; hit_tp = high[j] >= tp
        if hit_sl and hit_tp:
            xp, reason = sl, "SL_TP"; xt = etime[j]; break
        if hit_sl:
            xp, reason = sl, "SL"; xt = etime[j]; break
        if hit_tp:
            xp, reason = tp, "TP"; xt = etime[j]; break
    else:
        xp, reason = close[-1], "EOD_LAST"; xt = etime[-1]

    raw = xp - entry_price
    gross = raw * DPP
    net = gross - SLIP * TICK_SIZE * 2 * DPP - COMM
    mae_pts = entry_price - min_low      # how far below entry (positive = bad)
    mfe_pts = max_high - entry_price     # how far above (positive = good)
    return {
        "entry_time": pd.Timestamp(entry_time), "exit_time": pd.Timestamp(xt),
        "entry": entry_price, "exit": xp, "sl": sl, "tp": tp,
        "risk_pts": entry_price - ol,
        "raw_pts": raw, "gross_dollars": gross, "net_dollars": net,
        "mae_pts": mae_pts, "mfe_pts": mfe_pts,
        "mae_dollars": mae_pts * DPP, "mfe_dollars": mfe_pts * DPP,
        "reason": reason,
    }


def summarize(label, trades):
    df = pd.DataFrame(trades)
    if df.empty:
        print(f"{label}: empty"); return
    n = len(df); wins = int((df["net_dollars"]>0).sum())
    wd = df.loc[df["net_dollars"]>0, "net_dollars"]; ld = -df.loc[df["net_dollars"]<0, "net_dollars"]
    pf = wd.sum()/ld.sum() if ld.sum()>0 else float("inf")
    eq = df["net_dollars"].cumsum(); maxdd = (eq - eq.cummax()).min()

    print(f"\n=== {label} ===")
    print(f"  Trades: {n}   Wins: {wins} ({100*wins/n:.1f}%)   PF: {pf:.3f}   Net: ${df['net_dollars'].sum():,.0f}   MaxDD: ${maxdd:,.0f}")
    print(f"  Avg risk (SL distance): {df['risk_pts'].mean():.1f} pts (${df['risk_pts'].mean()*DPP:,.0f})")
    print()
    # MAE distribution
    print(f"  MAE (max unrealized DD per trade, points/dollars):")
    print(f"    mean:    {df['mae_pts'].mean():>6.1f} pts   ${df['mae_dollars'].mean():>7,.0f}")
    print(f"    median:  {df['mae_pts'].median():>6.1f} pts   ${df['mae_dollars'].median():>7,.0f}")
    print(f"    p75:     {df['mae_pts'].quantile(0.75):>6.1f} pts   ${df['mae_dollars'].quantile(0.75)*1:>7,.0f}")
    print(f"    p90:     {df['mae_pts'].quantile(0.90):>6.1f} pts   ${df['mae_dollars'].quantile(0.90)*1:>7,.0f}")
    print(f"    p95:     {df['mae_pts'].quantile(0.95):>6.1f} pts   ${df['mae_dollars'].quantile(0.95)*1:>7,.0f}")
    print(f"    p99:     {df['mae_pts'].quantile(0.99):>6.1f} pts   ${df['mae_dollars'].quantile(0.99)*1:>7,.0f}")
    print(f"    max:     {df['mae_pts'].max():>6.1f} pts   ${df['mae_dollars'].max():>7,.0f}")
    print()
    # Split MAE by outcome
    w = df[df["net_dollars"]>0]
    l = df[df["net_dollars"]<=0]
    print(f"  MAE on WINNING trades (n={len(w)}):")
    print(f"    mean:  {w['mae_pts'].mean():>6.1f} pts ${w['mae_dollars'].mean():>7,.0f}     p75: {w['mae_pts'].quantile(0.75):>6.1f} pts     p90: {w['mae_pts'].quantile(0.90):>6.1f} pts")
    print(f"  MAE on LOSING trades  (n={len(l)}):")
    print(f"    mean:  {l['mae_pts'].mean():>6.1f} pts ${l['mae_dollars'].mean():>7,.0f}     p75: {l['mae_pts'].quantile(0.75):>6.1f} pts     p90: {l['mae_pts'].quantile(0.90):>6.1f} pts")
    print()
    # MAE as % of SL distance (for winners): how close did winners get to being stopped out?
    nz = df[df["risk_pts"]>0].copy()
    nz["mae_pct_risk"] = nz["mae_pts"] / nz["risk_pts"]
    w2 = nz[nz["net_dollars"]>0]
    print(f"  WINNERS: MAE as fraction of SL distance:")
    print(f"    mean:    {w2['mae_pct_risk'].mean()*100:>5.1f}%   median: {w2['mae_pct_risk'].median()*100:>5.1f}%   p75: {w2['mae_pct_risk'].quantile(0.75)*100:>5.1f}%   p90: {w2['mae_pct_risk'].quantile(0.90)*100:>5.1f}%")


def main():
    print("Loading bars...", flush=True)
    days = load_days()
    keys = sorted(days.keys())
    print(f"  {len(keys)} days loaded\n")

    for n_c in [1, 4]:
        trades = []
        for d in keys:
            t = run_day(days[d], n_c)
            if t is not None: trades.append(t)
        summarize(f"N_confirm = {n_c}", trades)


if __name__ == "__main__":
    main()
