"""Stage 1: core rough-vol lookback sweep NORM_LEN x Z_LOOKBACK.

Objective = IS net-of-cost EV/trade (drives pass rate). Reports gross win rate (edge) + OOS net EV
(robustness). All other params held neutral; orderflow flags are config-independent (cached).
Cost = COST_PER_MNQ round-trip; size = $1500/(k*ATR*$2_per_MNQ_pt).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from live.combined.rv_features import RVConfig
from features import compute_features
from data import load_bars
from backtest import run, BTParams
import eval_mc

NORM_GRID = [100, 150, 200, 300, 400, 600, 900]
ZLOOK_GRID = [50, 75, 100, 150, 200]
K_NEUTRAL = 8.0
COST_PER_MNQ = 2.0
HIGH_Z = 2.0
SPLIT = pd.Timestamp("2024-04-05")
DOLLARS_PER_MNQ_PT = 2.0


def net_trades(tr: pd.DataFrame) -> pd.DataFrame:
    """Add commission (size from k*ATR) -> net pnl/mae columns."""
    sl_pts = K_NEUTRAL * tr["atr"].to_numpy()
    mnq = 1500.0 / (sl_pts * DOLLARS_PER_MNQ_PT)
    comm = mnq * COST_PER_MNQ
    tr = tr.copy()
    tr["comm"] = comm
    tr["net_$"] = tr["pnl_$"].to_numpy() - comm
    tr["net_mae_$"] = tr["mae_$"].to_numpy() - comm
    return tr


def main():
    t0 = time.time()
    bars = load_bars()
    c = bars["close"].to_numpy(); h = bars["high"].to_numpy(); l = bars["low"].to_numpy()
    flags = pd.read_parquet(HERE / "cache" / "orderflow_flags_n8_d40.parquet")
    lo = flags["long_ok"].reindex(bars.index, fill_value=False).to_numpy()
    so = flags["short_ok"].reindex(bars.index, fill_value=False).to_numpy()
    ent_et_all = bars["et"].dt.tz_localize(None)
    print(f"loaded bars+flags ({time.time()-t0:.0f}s); sweeping {len(NORM_GRID)*len(ZLOOK_GRID)} cells "
          f"(k={K_NEUTRAL}, cost=${COST_PER_MNQ}/MNQ, HIGH_Z={HIGH_Z})\n")

    rows = []
    for nl in NORM_GRID:
        for zl in ZLOOK_GRID:
            cfg = RVConfig(NORM_LEN=nl, Z_LOOKBACK=zl)
            feat = compute_features(c, h, l, cfg)
            tr = run(bars, feat, BTParams(high_z=HIGH_Z, k=K_NEUTRAL, atr_max=150.0,
                                          use_orderflow=True), lo, so)
            if tr.empty:
                continue
            tr = net_trades(tr)
            ent = ent_et_all.reindex(range(len(bars)))  # not used directly
            ent_ts = pd.to_datetime(tr["entry_ts"]).dt.tz_convert("America/New_York").dt.tz_localize(None)
            is_m = (ent_ts < SPLIT).to_numpy()
            isn = tr["net_$"].to_numpy()[is_m]; oosn = tr["net_$"].to_numpy()[~is_m]
            rows.append({
                "NORM": nl, "ZLOOK": zl,
                "is_n": int(is_m.sum()), "oos_n": int((~is_m).sum()),
                "is_grosswin": float((tr["pnl_$"].to_numpy()[is_m] > 0).mean()),
                "is_netEV": float(isn.mean()), "oos_netEV": float(oosn.mean()) if oosn.size else float("nan"),
                "is_netwin": float((isn > 0).mean()),
            })
    df = pd.DataFrame(rows).sort_values("is_netEV", ascending=False).reset_index(drop=True)
    pd.set_option("display.width", 200)
    print("TOP 12 by IS net EV/trade:")
    print(df.head(12).to_string(index=False,
          formatters={"is_grosswin": "{:.3f}".format, "is_netEV": "{:.1f}".format,
                      "oos_netEV": "{:.1f}".format, "is_netwin": "{:.3f}".format}))

    # pass-rate peek on the top 3 IS cells (net, with cost)
    print("\neval-MC pass% (IS pool, net of cost, 1 trade/day, 20k sims) for top-3 IS cells:")
    for _, r in df.head(3).iterrows():
        cfg = RVConfig(NORM_LEN=int(r["NORM"]), Z_LOOKBACK=int(r["ZLOOK"]))
        feat = compute_features(c, h, l, cfg)
        tr = net_trades(run(bars, feat, BTParams(high_z=HIGH_Z, k=K_NEUTRAL, atr_max=150.0,
                                                 use_orderflow=True), lo, so))
        ent_ts = pd.to_datetime(tr["entry_ts"]).dt.tz_convert("America/New_York").dt.tz_localize(None)
        is_m = (ent_ts < SPLIT).to_numpy()
        pool = tr[is_m]
        res = eval_mc.run_mc(pool["net_$"].to_numpy(), pool["net_mae_$"].to_numpy(),
                             n_sims=20000, trades_per_day=1)
        print(f"  NORM={int(r['NORM'])} ZLOOK={int(r['ZLOOK'])}: grosswin={r['is_grosswin']*100:.1f}% "
              f"netEV=${r['is_netEV']:.0f}  PASS={res['pass']*100:.1f}% BUST={res['bust']*100:.1f}% "
              f"median {res['median_days']:.0f}d")

    (HERE / "results").mkdir(exist_ok=True)
    df.to_csv(HERE / "results" / "results_stage1.csv", index=False)
    print(f"\nwrote results/results_stage1.csv  | TOTAL {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
