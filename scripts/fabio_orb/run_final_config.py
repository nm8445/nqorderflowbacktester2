"""
Final Fabio ORB variant:
  - TP=4.0R, DeltaThreshold=300 contracts
  - N=4 consecutive closes above ORB_high required
  - Skip entries with hhmm==930 (entry bar closes at 9:30 ET)
  - Test two delta-confirmation modes:
      MODE_A: delta only on entry bar (>= 300)
      MODE_B: delta on entry bar (>= 300) AND positive delta on each of the 3 prior bars

Outputs:
  - per-trade CSV
  - equity curve PNG (replaces old one)
  - stats
"""
from __future__ import annotations
import warnings; warnings.simplefilter("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VOL_PARQUET = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
OUT_DIR = Path("D:/trading_pythonbacktest_data/fabio orb")
ET = "America/New_York"
ORB_START_HHMM, ORB_END_HHMM, TRADE_END_HHMM = 830, 900, 1400
SKIP_BUCKET_HHMM = 930

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


def run_day(day, mode):
    """
    mode: 'A' = delta on entry bar only (>=300)
          'B' = delta on entry bar (>=300) AND prior 3 bars all delta > 0
    """
    oh = day["orb_high"]; ol = day["orb_low"]
    hhmm = day["hhmm"]; high = day["high"]; low = day["low"]
    close = day["close"]; delta = day["delta"]; etime = day["close_et"]
    N = 4; n = len(hhmm)
    DTHR = 300.0

    entry_idx = -1
    for i in range(N - 1, n):
        if hhmm[i] > TRADE_END_HHMM: break
        if hhmm[i] == SKIP_BUCKET_HHMM: continue       # skip 9:30 bucket
        # 4 consec closes above ORB_high
        if not all(close[i-k] > oh for k in range(N)): continue
        # delta gate on entry bar
        if delta[i] < DTHR: continue
        # mode B: prior 3 bars all need positive delta
        if mode == "B":
            if not all(delta[i-k] > 0 for k in range(1, N)): continue
        ep = close[i]
        if ol >= ep: continue
        entry_idx = i; entry_price = ep
        sl = ol; tp = ep + 4.0 * (ep - ol)
        entry_time = etime[i]
        break

    if entry_idx < 0: return None

    min_low = entry_price; max_high = entry_price
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
    return {
        "entry_time": pd.Timestamp(entry_time), "exit_time": pd.Timestamp(xt),
        "entry": entry_price, "exit": xp, "sl": sl, "tp": tp,
        "risk_pts": entry_price - ol, "raw_pts": raw,
        "gross_dollars": gross, "net_dollars": net,
        "mae_pts": entry_price - min_low,
        "mfe_pts": max_high - entry_price,
        "reason": reason, "mode": mode,
    }


def summarize(label, trades):
    if not trades: print(f"{label}: empty"); return None
    df = pd.DataFrame(trades)
    n = len(df); wins = int((df["net_dollars"]>0).sum())
    wd = df.loc[df["net_dollars"]>0, "net_dollars"]; ld = -df.loc[df["net_dollars"]<0, "net_dollars"]
    pf = wd.sum()/ld.sum() if ld.sum()>0 else float("inf")
    eq = df.sort_values("entry_time")["net_dollars"].cumsum()
    maxdd = (eq - eq.cummax()).min()
    print(f"\n=== {label} ===")
    print(f"  Trades:        {n}")
    print(f"  Win rate:      {100*wins/n:.1f}% ({wins}/{n})")
    print(f"  Net P&L:       ${df['net_dollars'].sum():,.0f}")
    print(f"  Profit factor: {pf:.3f}")
    print(f"  Max DD:        ${maxdd:,.0f}")
    print(f"  Avg trade:     ${df['net_dollars'].mean():,.0f}")
    print(f"  Avg risk:      {df['risk_pts'].mean():.1f} pts (${df['risk_pts'].mean()*DPP:,.0f})")
    print(f"  Avg MAE:       {df['mae_pts'].mean():.1f} pts (${df['mae_pts'].mean()*DPP:,.0f})")
    print(f"  p95 MAE:       {df['mae_pts'].quantile(0.95):.1f} pts (${df['mae_pts'].quantile(0.95)*DPP:,.0f})")
    print(f"  Exits: " + ", ".join(f"{r}={c}" for r,c in df['reason'].value_counts().items()))
    return df


def plot_equity(df, title, outpath):
    df = df.sort_values("entry_time").reset_index(drop=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["equity"] = df["net_dollars"].cumsum()
    peak = df["equity"].cummax()
    dd = df["equity"] - peak

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                    gridspec_kw={"height_ratios":[3,1]})
    ax1.plot(df["entry_time"], df["equity"], color="steelblue", lw=1.2)
    ax1.fill_between(df["entry_time"], df["equity"], peak, color="red", alpha=0.18, label="Drawdown")
    ax1.plot(df["entry_time"], peak, color="darkgreen", lw=0.6, ls="--", alpha=0.5, label="Equity peak")
    ax1.set_title(title)
    ax1.set_ylabel("Cumulative net P&L ($)"); ax1.grid(True, alpha=0.3); ax1.legend(loc="upper left")
    ax2.fill_between(df["entry_time"], dd, 0, color="red", alpha=0.5)
    ax2.set_ylabel("Drawdown ($)"); ax2.set_xlabel("Date"); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()
    print(f"  Saved plot: {outpath}")


def main():
    print("Loading bars...", flush=True)
    days = load_days()
    keys = sorted(days.keys())
    print(f"  {len(keys)} days from {keys[0].date()} to {keys[-1].date()}\n")

    # Run mode A
    trades_A = [t for d in keys if (t := run_day(days[d], "A")) is not None]
    df_A = summarize("Mode A: N=4 + skip 9:30 + delta>=300 on entry bar", trades_A)
    # Run mode B
    trades_B = [t for d in keys if (t := run_day(days[d], "B")) is not None]
    df_B = summarize("Mode B: N=4 + skip 9:30 + delta>=300 entry + delta>0 on 3 prior bars", trades_B)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Replace old equity curve with Mode A (the primary recommended config)
    plot_equity(df_A,
                "Fabio ORB — N=4 confirmation + skip 9:30 + delta>=300 (entry bar)",
                OUT_DIR / "equity_tp4_delta300.png")
    # Mode B as separate PNG
    plot_equity(df_B,
                "Fabio ORB — N=4 + skip 9:30 + delta>=300 entry + delta>0 priors",
                OUT_DIR / "equity_tp4_delta300_priors_pos.png")

    # Save trades
    df_A.to_csv(OUT_DIR / "trades_final_modeA.csv", index=False)
    df_B.to_csv(OUT_DIR / "trades_final_modeB.csv", index=False)


if __name__ == "__main__":
    main()
