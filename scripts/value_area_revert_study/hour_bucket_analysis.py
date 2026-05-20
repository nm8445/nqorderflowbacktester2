"""Entry-hour-bucket analysis for the largest-sample VA-revert config:
   N=5, sig_D=50, conf_D=50, sl_mult=2.0, overshoot_cap=75%.
"""
from __future__ import annotations

import datetime as dt
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_SCRIPTS = Path(__file__).parent.parent / "overnight range strat" / "scripts"
sys.path.insert(0, str(PROJECT_SCRIPTS))
from range_break_entry_signal_study import load_5min_features

CACHE_SIGS = Path(__file__).parent / "va_revert_signals.parquet"
ET         = "America/New_York"
FORCE_CLOSE = dt.time(16, 0)

N        = 5
SIG_D    = 100
CONF_D   = 50
SL_MULT  = 0.50
CAP      = 75.0


def simulate_exit(direction, entry_price, entry_idx, bars_day, sl_pts, tp_price):
    sign = 1 if direction == "LONG" else -1
    sl_price = entry_price - sign * sl_pts
    n = len(bars_day)
    for k in range(entry_idx + 1, n):
        bar = bars_day.iloc[k]
        bt = pd.Timestamp(bar["bar_open_time"])
        if bt.time() >= FORCE_CLOSE:
            close_px = float(bars_day.iloc[k - 1]["close"]) if k > 0 else float(bar["open"])
            return ("held->close", sign * (close_px - entry_price), bt)
        hi, lo = float(bar["high"]), float(bar["low"])
        if sign > 0:
            tp_hit = hi >= tp_price; sl_hit = lo <= sl_price
        else:
            tp_hit = lo <= tp_price; sl_hit = hi >= sl_price
        if tp_hit and sl_hit:
            return ("SL", sign * (sl_price - entry_price), bt)
        if tp_hit:
            return ("TP", sign * (tp_price - entry_price), bt)
        if sl_hit:
            return ("SL", sign * (sl_price - entry_price), bt)
    last = bars_day.iloc[-1]
    return ("held->close", sign * (float(last["close"]) - entry_price),
            pd.Timestamp(last["bar_open_time"]))


def main():
    print("loading signals + bars ...")
    sigs = pd.read_parquet(CACHE_SIGS)
    sigs["date"] = pd.to_datetime(sigs["date"]).dt.date
    sigs["entry_time"] = pd.to_datetime(sigs["entry_time"])

    # Filter to config
    sig_col  = f"sig_abs_w{N}"
    conf_col = f"conf_abs_w{N}"
    sub = sigs[(sigs["distance_pct"] < CAP)
               & (sigs[sig_col].fillna(0)  >= SIG_D)
               & (sigs[conf_col].fillna(0) >= CONF_D)].copy()
    print(f"  filtered signals: {len(sub)}")

    bars_all, _ = load_5min_features((dt.date(2020,12,1), dt.date(2026,5,7)))
    bars_by_day = {d: g.sort_values("bar_open_time").reset_index(drop=True)
                   for d, g in bars_all.groupby("session_date")}

    print("running chained simulation ...")
    rows = []
    last_exit = pd.Timestamp(0, tz=ET)
    for _, t in sub.sort_values("entry_time").iterrows():
        ent_t = pd.Timestamp(t["entry_time"])
        if ent_t <= last_exit: continue
        d = pd.to_datetime(t["date"]).date()
        bars_day = bars_by_day.get(d)
        if bars_day is None: continue
        entry_idx = int(t["entry_idx"])
        if entry_idx >= len(bars_day): continue
        entry_price = float(t["entry_price"])
        atr = float(t["atr_at_entry"])
        sl_pts = SL_MULT * atr
        outcome, pnl, exit_t = simulate_exit(t["direction"], entry_price, entry_idx,
                                              bars_day, sl_pts, float(t["target"]))
        rows.append({"entry_time": ent_t, "exit_time": exit_t,
                     "direction": t["direction"], "outcome": outcome, "pnl": pnl,
                     "date": d, "atr": atr,
                     "tp_dist": float(t["target"]) - entry_price if t["direction"]=="LONG"
                                else entry_price - float(t["target"]),
                     "sl_dist": sl_pts,
                     "rr": (float(t["target"]) - entry_price) / sl_pts if t["direction"]=="LONG"
                            else (entry_price - float(t["target"])) / sl_pts})
        last_exit = pd.Timestamp(exit_t)
    trades = pd.DataFrame(rows)
    print(f"  {len(trades)} trades after chained dedupe")

    trades["entry_hour"] = trades["entry_time"].dt.hour
    trades["period"] = np.where(trades["date"] < dt.date(2025,1,1), "IS", "OOS")

    L = []
    L.append("=" * 220)
    L.append(f"VA-REVERT  CONFIG: N={N}, sig_D={SIG_D}, conf_D={CONF_D}, sl_mult={SL_MULT}, overshoot_cap={CAP}%")
    L.append("=" * 220)
    L.append(f"Total trades: {len(trades)}  (IS={len(trades[trades['period']=='IS'])}  OOS={len(trades[trades['period']=='OOS'])})")
    L.append("")

    def hour_table(df, label):
        out = (df.groupby("entry_hour", observed=True)
                 .agg(n=("pnl","count"),
                      tp=("outcome", lambda s: (s=="TP").sum()),
                      sl=("outcome", lambda s: (s=="SL").sum()),
                      hc=("outcome", lambda s: (s=="held->close").sum()),
                      wr=("pnl", lambda s: (s>0).mean()*100),
                      pf=("pnl", lambda s: (s[s>0].sum()/abs(s[s<0].sum())) if (s<0).any() else float("inf")),
                      total_pts=("pnl","sum"),
                      avg_pts=("pnl","mean"),
                      median_rr=("rr","median"))
                 .round(2))
        out["dollars_nq"] = (out["total_pts"] * 20).round(0)
        L.append("")
        L.append(f"--- {label} ---")
        L.append(out.to_string())
        return out

    L.append("=" * 220)
    L.append("ENTRY-HOUR BREAKDOWN")
    L.append("=" * 220)
    hour_table(trades, "ALL (combined IS+OOS)")
    hour_table(trades[trades["period"]=="IS"],  "IS only")
    hour_table(trades[trades["period"]=="OOS"], "OOS only")

    # Direction split per hour
    L.append("")
    L.append("=" * 220)
    L.append("ENTRY-HOUR x DIRECTION")
    L.append("=" * 220)
    pivot = trades.pivot_table(index="entry_hour", columns="direction",
                                values="pnl", aggfunc=["count","sum","mean"],
                                observed=True).round(2)
    L.append(pivot.to_string())

    OUT = (Path(__file__).parent.parent / "overnight range strat" / "tradelogs" / "robust_configs"
           / f"va_revert_hour_bucket_N{N}_sigD{SIG_D}_confD{CONF_D}_sl{SL_MULT}_cap{int(CAP)}.txt")
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {OUT}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
