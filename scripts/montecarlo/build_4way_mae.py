"""Build a UNIFIED 4-way trade log with consistent MAE for all 4 strats (OD/RV/B2/FB).

Why: MAE was previously stitched from inconsistent sources (mae_$ col of unknown resolution for
OD/RV/B2; risk_pts / 5-min for FB). Here we recompute MAE the same way for everyone:

  MAE methodology — for EACH trade, over (fill_time, exit_time] using 1-MIN bars:
    LONG : mae_pts = entry_price - min(1-min low)   (deepest float below entry)
    SHORT: mae_pts = max(1-min high) - entry_price
  mae_$ (1 NQ, 1 contract) = mae_pts * $20.  (qty-independent; this is per-contract floating DD)

  entry_price = 1-min close at fill_time (OD fills 20 min after its logged 19:00 bar-open).
  Martingale OFF: pnl divided by reconstructed qty (OD loss-marti, B2 FC-marti via ~16:00 exit).

Output: results/combined_4way_with_mae_1min.csv  (date, strat, dir, pnl_1c, mae_1c)
+ prints per-strat MAE distribution (1 NQ) and the per-MNQ RPTI-breach view.

Run:  python scripts/montecarlo/build_4way_mae.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRADES = ROOT / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"
ONE_MIN = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
OUT = ROOT / "scripts" / "montecarlo" / "results" / "combined_4way_with_mae_1min.csv"
ET = "America/New_York"
NQ_PT = 20.0


def main():
    df1 = pd.read_parquet(ONE_MIN)
    if df1.index.tz is None: df1.index = df1.index.tz_localize("UTC")
    df1.index = df1.index.tz_convert(ET)
    df1 = df1.sort_index()
    idx = df1.index.values.astype("int64")
    hi = df1["high"].values; lo = df1["low"].values; cl = df1["close"].values

    t = pd.read_csv(TRADES)
    t["entry_ts"] = pd.to_datetime(t["entry_ts"], utc=True, format="mixed").dt.tz_convert(ET)
    t["exit_ts"] = pd.to_datetime(t["exit_ts"], utc=True, format="mixed").dt.tz_convert(ET)
    t = t.sort_values("entry_ts").reset_index(drop=True)

    # reconstruct marti qty per strat (OD loss-marti, B2 FC-marti via ~16:00 exit)
    t["qty"] = 1
    for strat in ["OD", "B2"]:
        sub = t[t.strat == strat]
        st = 0; ns = 1; qs = []
        for _, r in sub.iterrows():
            if strat == "OD":
                qs.append(2 if st == 1 else 1); loss = r["pnl_$"] < 0
                st = (1 if loss else 0) if st == 0 else (2 if st == 1 else (1 if loss else 0))
            else:  # B2 FC-marti
                qs.append(ns); fc = (r["exit_ts"].hour == 16)
                ns = 1 if ns == 2 else (2 if (r["pnl_$"] < 0 and fc) else 1)
        t.loc[sub.index, "qty"] = qs

    rows = []
    skipped = 0
    for _, r in t.iterrows():
        strat = r["strat"]; longish = (r["direction"] == "LONG")
        fill = r["entry_ts"] + (pd.Timedelta(minutes=20) if strat == "OD" else pd.Timedelta(0))
        f_ns = np.int64(fill.value); x_ns = np.int64(r["exit_ts"].value)
        ep_pos = int(np.searchsorted(idx, f_ns, side="right")) - 1   # 1-min close asof fill
        s = int(np.searchsorted(idx, f_ns, side="right"))            # bars after fill
        e = int(np.searchsorted(idx, x_ns, side="right"))            # through exit
        if ep_pos < 0 or e <= s:
            skipped += 1; continue
        entry_price = cl[ep_pos]
        if longish:
            mae_pts = entry_price - lo[s:e].min()
        else:
            mae_pts = hi[s:e].max() - entry_price
        mae_pts = max(mae_pts, 0.0)
        rows.append({
            "date": fill.date(),
            "ts": fill.isoformat(),                   # fill (open) time, for ordering + news flag
            "exit_ts": r["exit_ts"].isoformat(),      # exit time, for news flag
            "strat": strat, "dir": r["direction"],
            "pnl_1c": r["pnl_$"] / r["qty"],          # marti OFF (per contract)
            "mae_1c": -(mae_pts * NQ_PT),             # negative = adverse, 1 NQ
        })
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Built {len(out)} trades (skipped {skipped}). MAE = worst 1-min low/high over hold.\n")

    print("Per-strat MAE (floating DD, 1 NQ) + RPTI($2k) breach by MNQ:")
    print("%-4s %5s %9s %9s %9s | %s" % ("", "n", "median$", "p99$", "max$", "MNQ where worst trade >= $2k"))
    for s in ["OD", "RV", "B2", "FB"]:
        m = -out[out.strat == s]["mae_1c"]   # positive magnitude
        worst = m.max()
        breach_mnq = 2000.0 / (worst / 10.0) if worst > 0 else 999   # mnq at which max float hits 2k
        print("%-4s %5d %9.0f %9.0f %9.0f | worst hits $2k at %.1f MNQ"
              % (s, len(m), m.median(), m.quantile(.99), worst, breach_mnq))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
