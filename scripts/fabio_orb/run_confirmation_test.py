"""
Confirmation-bars filter for the Fabio ORB.

At TP=4.0R, DeltaThreshold=300 contracts, scan N = 1..6 (number of bars that
must each close above ORB_high before entry). N=1 is the baseline.

Two modes for the delta gate:
  - delta required on each of the N confirming bars (stricter)
  - delta required only on the entry bar (the Nth)

Reports per-N: total stats + 9:30 bucket carve-out.
"""
from __future__ import annotations
import warnings; warnings.simplefilter("ignore")
from pathlib import Path
import numpy as np, pandas as pd

VOL_PARQUET = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
ET = "America/New_York"
ORB_START_HHMM, ORB_END_HHMM, TRADE_END_HHMM = 830, 900, 1400

TICK_SIZE, TICK_VALUE = 0.25, 5.0
DOLLARS_PER_POINT = TICK_VALUE / TICK_SIZE
SLIP_TICKS_PER_SIDE = 1
COMMISSION_RT = 5.0


def load_days():
    df = pd.read_parquet(VOL_PARQUET)
    agg = df.groupby("bar_open_time", as_index=False).agg(
        open=("open","first"), high=("high","max"), low=("low","min"),
        close=("close","last"),
        buy_vol=("buy_vol","sum"), sell_vol=("sell_vol","sum"),
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


def run_day(day, tp_rr, delta_thr, n_confirm, delta_each_bar=False):
    oh = day["orb_high"]; ol = day["orb_low"]
    hhmm = day["hhmm"]; high = day["high"]; low = day["low"]
    close = day["close"]; delta = day["delta"]; etime = day["close_et"]
    n = len(hhmm)

    entry_idx = -1
    for i in range(n_confirm - 1, n):
        if hhmm[i] > TRADE_END_HHMM:
            break
        # require N consecutive closes above ORB_high
        ok = True
        for k in range(n_confirm):
            if close[i-k] <= oh:
                ok = False; break
            if delta_each_bar and delta[i-k] < delta_thr:
                ok = False; break
        if not ok: continue
        if not delta_each_bar and delta[i] < delta_thr:
            continue
        ep = close[i]
        if ol >= ep: continue
        entry_idx = i; entry_price = ep
        sl = ol; tp = ep + tp_rr * (ep - ol)
        entry_time = etime[i]
        break

    if entry_idx < 0: return None

    for j in range(entry_idx + 1, n):
        if hhmm[j] >= TRADE_END_HHMM:
            return _close(entry_price, close[j], entry_time, etime[j], "EOD", sl, tp, oh, ol)
        hit_sl = low[j] <= sl; hit_tp = high[j] >= tp
        if hit_sl and hit_tp:
            return _close(entry_price, sl, entry_time, etime[j], "SL_TP_same_bar", sl, tp, oh, ol)
        if hit_sl:
            return _close(entry_price, sl, entry_time, etime[j], "SL", sl, tp, oh, ol)
        if hit_tp:
            return _close(entry_price, tp, entry_time, etime[j], "TP", sl, tp, oh, ol)
    return _close(entry_price, close[-1], entry_time, etime[-1], "EOD_LAST", sl, tp, oh, ol)


def _close(ep, xp, etime, xtime, reason, sl, tp, oh, ol):
    raw_pts = xp - ep
    slip = SLIP_TICKS_PER_SIDE * TICK_SIZE * 2
    gross = raw_pts * DOLLARS_PER_POINT
    net = gross - slip * DOLLARS_PER_POINT - COMMISSION_RT
    return {
        "entry_time": pd.Timestamp(etime), "exit_time": pd.Timestamp(xtime),
        "entry": ep, "exit": xp, "sl": sl, "tp": tp,
        "risk_pts": ep - ol, "raw_pts": raw_pts,
        "gross_dollars": gross, "net_dollars": net, "reason": reason,
    }


def stats_block(label, trades):
    df = pd.DataFrame(trades)
    if df.empty:
        print(f"  {label:25} n=0"); return
    df["entry_et"] = pd.to_datetime(df["entry_time"]).dt.tz_convert(ET) if df["entry_time"].dt.tz is None or str(df["entry_time"].dt.tz)!="America/New_York" else df["entry_time"]
    n = len(df); wins = int((df["net_dollars"]>0).sum())
    wd = df.loc[df["net_dollars"]>0, "net_dollars"]
    ld = -df.loc[df["net_dollars"]<0, "net_dollars"]
    pf = wd.sum()/ld.sum() if ld.sum()>0 else float("inf")
    eq = df["net_dollars"].cumsum(); dd = (eq - eq.cummax()).min()
    avg = df["net_dollars"].mean()
    avg_risk = df["risk_pts"].mean()
    avg_winner = df["net_dollars"][df["net_dollars"]>0].mean() if wins else 0
    avg_loser = df["net_dollars"][df["net_dollars"]<0].mean() if (n-wins) else 0
    print(f"  {label:25} n={n:>4} win={wins:>3} ({100*wins/n:>5.1f}%) net=${df['net_dollars'].sum():>9,.0f} PF={pf:.3f} avg_risk={avg_risk:>5.1f}pts avg=${avg:>6.0f} wWin=${avg_winner:>6.0f} wLoss=${avg_loser:>6.0f} MaxDD=${dd:>9,.0f}")


def main():
    print("Loading bars...", flush=True)
    days = load_days()
    keys = sorted(days.keys())
    print(f"  {len(keys)} days from {keys[0].date()} to {keys[-1].date()}")
    print()

    TP_RR, DTHR = 4.0, 300.0
    print(f"TP=4.0R, DeltaThreshold=300 contracts. Vary N consec closes above ORB_high.\n")

    # Two modes
    for delta_each in [False, True]:
        mode = "delta@entry-only" if not delta_each else "delta@each-bar"
        print(f"\n========== Mode: {mode} ==========")
        for n_c in range(1, 7):
            trades = []
            for d in keys:
                t = run_day(days[d], TP_RR, DTHR, n_c, delta_each_bar=delta_each)
                if t is not None: trades.append(t)
            df = pd.DataFrame(trades)
            df["entry_et"] = pd.to_datetime(df["entry_time"]).dt.tz_convert(ET)
            df["hhmm"] = df["entry_et"].dt.hour*100 + df["entry_et"].dt.minute

            print(f"\n--- N_confirm = {n_c} ---")
            stats_block("ALL entries", trades)
            sub = df[df["hhmm"]==930].to_dict("records")
            stats_block("9:30 bucket only", sub)
            sub = df[(df["hhmm"]>=900) & (df["hhmm"]<=959)].to_dict("records")
            stats_block("9:00-9:59 bucket", sub)


if __name__ == "__main__":
    main()
