"""Paper-faithful engine — 2D sweep TP × Delta threshold.

Engine: paper-exact (single-bar break, no consecutive closes, no skip bucket,
ORB 09:30-10:00 ET, force-close 15:00 ET, long-only, 1 contract).
Delta: REAL aggressor-classified buy_vol - sell_vol from MBP-1 data.

Sweep:
  TP_RR: 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0   (10 values)
  DeltaThreshold: 0, 50, 100, 150, 200, 250, 300, 400, 500, 750   (10 values)
  = 100 cells.

IS/OOS 60/40 chronological split. Pass = beat paper baseline (TP=1, delta=200)
on both IS+OOS net $.
"""
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
import pandas as pd

VOL_PARQUET = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ET = "America/New_York"

ORB_START_HHMM = 930
ORB_END_HHMM   = 1000
TRADE_END_HHMM = 1500

TICK_SIZE = 0.25
DOLLARS_PER_PT = 5.0 / TICK_SIZE   # 20
SLIP_PTS_RT = 1 * TICK_SIZE * 2     # 0.5
COMM_RT = 5.0

TP_GRID    = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
DELTA_GRID = [0, 50, 100, 150, 200, 250, 300, 400, 500, 750]

N_WORKERS = 6
_BARS = None


def load_bars():
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
    agg = agg.dropna(subset=["open", "high", "low", "close"]).copy()
    agg = agg[agg["session_date"] >= pd.Timestamp("2021-01-01")].copy()
    return agg


def run_strategy(bars, delta_thr: float, tp_rr: float):
    trades = []
    for sd, day in bars.groupby("session_date"):
        orb = day[(day["hhmm"] > ORB_START_HHMM) & (day["hhmm"] <= ORB_END_HHMM)]
        if len(orb) == 0: continue
        orb_high = float(orb["high"].max())
        orb_low  = float(orb["low"].min())
        post = day[(day["hhmm"] > ORB_END_HHMM) & (day["hhmm"] <= TRADE_END_HHMM)].reset_index(drop=True)
        if len(post) == 0: continue

        entry_idx = -1
        for i in range(len(post)):
            bar = post.iloc[i]
            if bar["hhmm"] > TRADE_END_HHMM: break
            if bar["close"] <= orb_high: continue
            if bar["delta"] < delta_thr: continue
            if orb_low >= bar["close"]: continue
            entry_idx = i
            break
        if entry_idx < 0: continue

        eb = post.iloc[entry_idx]
        ep = float(eb["close"])
        sl = orb_low
        risk = ep - sl
        tp = ep + tp_rr * risk

        exit_price = None; reason = None
        for j in range(entry_idx + 1, len(post)):
            bar = post.iloc[j]
            hit_sl = bar["low"]  <= sl
            hit_tp = bar["high"] >= tp
            if hit_sl and hit_tp: exit_price = sl; reason = "SL_TP"; break
            if hit_sl: exit_price = sl; reason = "SL"; break
            if hit_tp: exit_price = tp; reason = "TP"; break
            if int(bar["hhmm"]) >= TRADE_END_HHMM:
                exit_price = float(bar["close"]); reason = "EOD"; break
        if exit_price is None:
            last = post.iloc[-1]
            exit_price = float(last["close"]); reason = "EOD_LAST"

        net_pts = (exit_price - ep) - SLIP_PTS_RT
        net_dollars = net_pts * DOLLARS_PER_PT - COMM_RT
        trades.append({
            "session_date": sd, "entry_time": eb["close_et"],
            "entry": ep, "exit": exit_price, "tp": tp, "sl": sl, "risk": risk,
            "delta_at_entry": float(eb["delta"]),
            "net_pts": net_pts, "net_dollars": net_dollars, "reason": reason,
        })
    return pd.DataFrame(trades)


def stats_block(pnls):
    n = len(pnls)
    if n == 0: return dict(n=0, wr=0, net=0, pf=0, mdd=0, avg=0)
    w = pnls[pnls > 0]; l = pnls[pnls < 0]
    pf = w.sum() / abs(l.sum()) if len(l) > 0 else 99.0
    cum = pnls.cumsum()
    mdd = float((cum - np.maximum.accumulate(cum)).min())
    return dict(n=n, wr=round(len(w)/n*100, 1), net=round(float(pnls.sum()), 0),
                pf=round(pf, 3), mdd=round(mdd, 0), avg=round(float(pnls.mean()), 1))


def _init_worker(bars):
    global _BARS
    _BARS = bars


def _run_cell(args):
    delta_thr, tp_rr = args
    trades = run_strategy(_BARS, delta_thr, tp_rr)
    if len(trades) == 0:
        return {"delta": delta_thr, "tp_rr": tp_rr, "n_all": 0}
    trades = trades.sort_values("session_date").reset_index(drop=True)
    pnls = trades["net_dollars"].values
    dates = sorted(trades["session_date"].unique())
    cutoff = dates[int(len(dates) * 0.6)] if len(dates) > 1 else dates[-1]
    is_mask  = trades["session_date"] <  cutoff
    oos_mask = trades["session_date"] >= cutoff
    s_all = stats_block(pnls)
    s_is  = stats_block(trades.loc[is_mask,  "net_dollars"].values)
    s_oos = stats_block(trades.loc[oos_mask, "net_dollars"].values)
    return {
        "delta": delta_thr, "tp_rr": tp_rr,
        "n_all": s_all["n"], "wr_all": s_all["wr"], "all_net": s_all["net"],
        "all_PF": s_all["pf"], "all_mdd": s_all["mdd"], "all_avg": s_all["avg"],
        "is_n": s_is["n"], "is_net": s_is["net"], "is_PF": s_is["pf"],
        "oos_n": s_oos["n"], "oos_net": s_oos["net"], "oos_PF": s_oos["pf"],
    }


def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading bars...")
    bars = load_bars()
    print(f"  {len(bars):,} bars ({time.time()-t0:.1f}s)")

    configs = [(d, t) for d in DELTA_GRID for t in TP_GRID]
    print(f"\n[{time.strftime('%H:%M:%S')}] Sweeping {len(configs)} cells on {N_WORKERS} workers...")
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker, initargs=(bars,)) as ex:
        rows = list(ex.map(_run_cell, configs))
    print(f"[{time.strftime('%H:%M:%S')}] Done in {time.time()-t1:.1f}s")

    df = pd.DataFrame(rows)
    df["beat_both"] = (df["is_net"] > 0) & (df["oos_net"] > 0)
    df["net_mdd_ratio"] = df["all_net"] / df["all_mdd"].abs().replace(0, 1)
    df.to_csv(OUT_DIR / "paper_2d_tp_delta_sweep.csv", index=False)

    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)

    # Paper baseline
    paper_cell = df[(df["delta"] == 200) & (df["tp_rr"] == 1.0)].iloc[0]
    print(f"\n=== PAPER BASELINE (delta=200, TP=1.0) ===")
    print(f"  n={paper_cell['n_all']}  WR={paper_cell['wr_all']}%  net=${paper_cell['all_net']:,.0f}  "
          f"PF={paper_cell['all_PF']}  MDD=${paper_cell['all_mdd']:,.0f}")

    print(f"\n=== TOP 15 by ALL NET $ (must beat both IS+OOS) ===")
    cols = ["delta", "tp_rr", "n_all", "wr_all", "all_net", "all_PF", "all_mdd",
            "is_net", "is_PF", "oos_net", "oos_PF"]
    print(df[df["beat_both"]].sort_values("all_net", ascending=False).head(15)[cols].to_string(index=False))

    print(f"\n=== HEATMAP: ALL NET $ by (delta x tp_rr) ===")
    pivot = df.pivot(index="delta", columns="tp_rr", values="all_net")
    print(pivot.fillna(0).astype(int).to_string())

    print(f"\n=== HEATMAP: PF by (delta x tp_rr) ===")
    pivot_pf = df.pivot(index="delta", columns="tp_rr", values="all_PF")
    print(pivot_pf.fillna(0).round(2).to_string())

    print(f"\n=== HEATMAP: MDD by (delta x tp_rr) ===")
    pivot_mdd = df.pivot(index="delta", columns="tp_rr", values="all_mdd")
    print(pivot_mdd.fillna(0).astype(int).to_string())

    print(f"\n=== TOP 5 by NET/|MDD| RATIO ===")
    ra = df[df["beat_both"]].sort_values("net_mdd_ratio", ascending=False).head(5)
    print(ra[cols + ["net_mdd_ratio"]].to_string(index=False))

    print(f"\n=== COUNT positive on both phases per delta ===")
    for d in DELTA_GRID:
        sub = df[df["delta"] == d]
        print(f"  delta={d}: {sub['beat_both'].sum()}/{len(sub)} cells positive IS+OOS")


if __name__ == "__main__":
    main()
