"""Risk-normalized (dollar-risk) sizing: lot = $R / (entry-to-native-stop distance), so a stop-out
loses ~$R. NOT a hard stop — the native exit (yellow/ATR/ORB/EOD) still governs. Tests whether this
keeps the floating loss under the $2k RPTI far better than fixed-MNQ.

Stop distance per strat (points):
  OD : 1.30 * ATR20(14)   (yellow)      B2 : 2.50 * ATR20(14)  (yellow)
  RV : 2.00 * ATR20(14)   (ATR stop)    FB : entry - ORB_low   (ORB)
floating at risk $R = (MAE_pts / stop_pts) * R  ->  RPTI breach if MAE >= 2 * stop (ratio >= 2).

Run:  python "scripts/cfd prop firms/risk_normalized_sizing.py"
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd

ET = "America/New_York"; NQ_PT = 20.0
ONE_MIN = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
TRADES = Path(__file__).resolve().parents[2] / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"
MULT = {"OD": 1.30, "B2": 2.50, "RV": 2.00}   # x ATR20(14)


def bars(df1, mins):
    b = df1.resample(f"{mins}min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna(subset=["open"])
    return b


def wilder(b, n=14):
    pc = b["close"].shift(1)
    tr = pd.concat([b["high"]-b["low"], (b["high"]-pc).abs(), (b["low"]-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def main():
    d1 = pd.read_parquet(ONE_MIN)
    if d1.index.tz is None: d1.index = d1.index.tz_localize("UTC")
    d1.index = d1.index.tz_convert(ET); d1 = d1.sort_index()
    idx = d1.index.values.astype("int64"); hi=d1["high"].values; lo=d1["low"].values; cl=d1["close"].values
    b20 = bars(d1, 20); atr20 = wilder(b20); a_idx = b20.index.values.astype("int64"); a_val = atr20.values
    b5 = bars(d1, 5)
    # ORB per session date: high/low of 5-min bars with close in (08:30, 09:00]
    b5h = b5.index.hour*100 + b5.index.minute
    orb = b5[(b5h > 830) & (b5h <= 900)]
    orb_low = orb.groupby(orb.index.date)["low"].min()

    t = pd.read_csv(TRADES)
    t["entry_ts"] = pd.to_datetime(t["entry_ts"], utc=True, format="mixed").dt.tz_convert(ET)
    t["exit_ts"] = pd.to_datetime(t["exit_ts"], utc=True, format="mixed").dt.tz_convert(ET)
    t = t.sort_values("entry_ts").reset_index(drop=True)

    rows = []
    for _, r in t.iterrows():
        strat = r["strat"]; lng = r["direction"] == "LONG"
        fill = r["entry_ts"] + (pd.Timedelta(minutes=20) if strat == "OD" else pd.Timedelta(0))
        f = np.int64(fill.value); x = np.int64(r["exit_ts"].value)
        ep_i = int(np.searchsorted(idx, f, side="right"))-1
        s = int(np.searchsorted(idx, f, side="right")); e = int(np.searchsorted(idx, x, side="right"))
        if ep_i < 0 or e <= s: continue
        entry = cl[ep_i]
        mae_pts = (entry - lo[s:e].min()) if lng else (hi[s:e].max() - entry)
        mae_pts = max(mae_pts, 0.0)
        if strat == "FB":
            ol = orb_low.get(fill.date())
            if ol is None or ol >= entry: continue
            stop = entry - ol
        else:
            ai = int(np.searchsorted(a_idx, f, side="right"))-1
            if ai < 0 or not np.isfinite(a_val[ai]) or a_val[ai] <= 0: continue
            stop = MULT[strat] * a_val[ai]
        if stop <= 0: continue
        rows.append({"date": fill.date(), "strat": strat,
                     "pnl_pts": (r["pnl_$"]/NQ_PT),   # 1-contract points (qty handled below not needed)
                     "mae_pts": mae_pts, "stop_pts": stop,
                     "ratio": mae_pts/stop})
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).resolve().parent / "_risknorm_trades.csv", index=False)
    print("Per-strat MAE/stop ratio (floating at $1k risk = ratio*$1k; RPTI breach if ratio>=2):")
    print("%-4s %5s %8s %8s %8s %12s"%("","n","median","p95","p99","P(ratio>=2)"))
    for s in ["OD","RV","B2","FB"]:
        rr = df[df.strat==s]["ratio"]
        print("%-4s %5d %8.2f %8.2f %8.2f %11.1f%%"%(s,len(rr),rr.median(),rr.quantile(.95),rr.quantile(.99),100*(rr>=2).mean()))

    # $1k-risk normalized pnl/mae (1 contract sized so stop=$R); pnl/mae in $ scale linearly with R
    for R in [1000, 1500]:
        df["pnl_$"] = df["pnl_pts"]/df["stop_pts"]*R
        df["mae_$"] = -(df["mae_pts"]/df["stop_pts"]*R)   # negative adverse
        # FundingPips funded RPTI sim, uniform $R risk all strats, marti off
        by={}
        for _,r in df.iterrows(): by.setdefault(r["date"],[]).append((r["pnl_$"], r["mae_$"]))
        packs=[by[k] for k in sorted(by)]; n=len(packs)
        def funded(rng):
            bal=100000.;dic=0;cash=0.
            for d in range(252):
                base=bal;dll=0.05*base;real=0.;bust=0
                for p,m in packs[rng.integers(0,n)]:
                    flo=-m
                    if flo>=2000. or bal+real-flo<=90000. or bal+real-flo<=base-dll: bust=1;break
                    real+=p
                if bust: return cash,1
                bal+=real;dic+=1
                if dic>=10:
                    pr=bal-100000.
                    if pr>=200.: cash+=pr*0.8;bal=100000.
                    dic=0
            return cash,0
        rng=np.random.default_rng(7)
        res=[funded(rng) for _ in range(15000)]
        print(f"\nFundingPips funded @ ${R}/trade risk-normalized (all 4 strats, marti off): "
              f"E[$wd]=${np.mean([r[0] for r in res]):.0f}/yr  blow={100*np.mean([r[1] for r in res]):.1f}%")


if __name__ == "__main__":
    main()
