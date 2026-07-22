"""WR of the 4-way combined in the LIVE FARM 1:1 config (risk $1500 to make $1500).

Each strat's LIVE stop distance, mirrored 1:1 (TP = SL distance), first-touch on the 1-min forward
path (SL-first tie-break), per-strat native force-close. Exact live stops:
  OD : 1.30 x ATR(14, 20-min)   [od_engine YELLOW_MULT]
  B2 : 2.50 x ATR(14, 20-min)   [b2_engine YMULT]
  RV : 2.00 x ATR(14, 20-min)   [rv_engine ATR_SL_MULT]  (== 4-way combined config)
  FB : entry -> ORB-low (08:30-09:00 ET range low)        [fabio_orb_engine, static ORB_Low]
Entries are the LOCKED 4-way log signals. Reuses propfirm_milking common/entries machinery.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from scripts.propfirm_milking.common import (
    eval_bracket, first_touch, load_1min_bars, is_oos_cutoff, ET, MNQ_POINT_VALUE, trade_cost, mnq_size,
)
from scripts.propfirm_milking.entries import build_entry_packs

LIVE_STOP = {"OD": ("atr", 14, 1.30), "B2": ("atr", 14, 2.50), "RV": ("atr", 14, 2.00),
             "FB": ("orb", None, None)}
ORB_START, ORB_END = 830, 900   # ET HHMM window for FB ORB-low


def orb_low_for_date(df1m: pd.DataFrame, d) -> float:
    """ORB low = min 1-min low over [08:30, 09:00) ET on date d."""
    day = df1m[df1m.index.normalize() == pd.Timestamp(d, tz=ET)]
    hhmm = day.index.hour * 100 + day.index.minute
    win = day[(hhmm >= ORB_START) & (hhmm < ORB_END)]
    return float(win["low"].min()) if len(win) else np.nan


def eval_fb_orb(packs, df1m) -> pd.DataFrame:
    """FB 1:1 off ORB-low: SL dist = entry - ORB_low, TP dist = same (mirrored)."""
    rows = []
    for p in packs:
        orb = orb_low_for_date(df1m, p.date)
        if not np.isfinite(orb):
            continue
        sl_dist = p.sign * (p.entry_price - orb)   # long: entry-orb (>0); sign keeps it positive dist
        if sl_dist <= 0:
            continue
        exit_price, reason, is_win = first_touch(p, sl_dist, sl_dist)
        n = mnq_size(sl_dist)
        pnl_pts = p.sign * (exit_price - p.entry_price)
        rows.append((p.date, reason, is_win, n, sl_dist, pnl_pts,
                     pnl_pts * MNQ_POINT_VALUE * n - trade_cost(n)))
    return pd.DataFrame(rows, columns=["date", "reason", "win", "mnq", "sl_dist_pts", "pnl_pts", "pnl_$"])


def block(tr, label, cutoff):
    rc = tr["reason"].value_counts(normalize=True)
    ism = tr["date"] < cutoff
    wr = tr["win"].mean() * 100
    iswr = tr.loc[ism, "win"].mean() * 100 if ism.any() else float("nan")
    ooswr = tr.loc[~ism, "win"].mean() * 100 if (~ism).any() else float("nan")
    print(f"  {label:>4} n={len(tr):>5} WR={wr:>5.1f}%  IS={iswr:>5.1f}%  OOS={ooswr:>5.1f}%  "
          f"TP={rc.get('TP',0)*100:>4.0f}% SL={rc.get('SL',0)*100:>4.0f}% FC={rc.get('FC',0)*100:>4.0f}%")
    return tr[["date", "win"]].assign(strat=label)


def main():
    df1m = load_1min_bars()
    all_dates = []
    print("4-WAY COMBINED — LIVE FARM 1:1 CONFIG win rate (entries=locked 4-way log):\n")
    parts = []
    # global IS/OOS cutoff across all strats' dates for a consistent split
    tmp = {}
    for s in ["OD", "B2", "RV", "FB"]:
        packs = build_entry_packs(s, verbose=False)
        if s == "FB":
            tr = eval_fb_orb(packs, df1m)
        else:
            _, alen, mult = LIVE_STOP[s]
            tr = eval_bracket(packs, atr_len=alen, sl_mult=mult, tp_mult=mult)
        tmp[s] = tr
        all_dates += list(tr["date"])
    cutoff = is_oos_cutoff(all_dates)
    print(f"  (IS/OOS cutoff @ {cutoff})\n")
    for s in ["OD", "B2", "RV", "FB"]:
        parts.append(block(tmp[s], s, cutoff))
    comb = pd.concat(parts, ignore_index=True)
    ism = comb["date"] < cutoff
    print(f"\n  {'COMBINED':>8} n={len(comb):>5} WR={comb['win'].mean()*100:>5.1f}%  "
          f"IS={comb.loc[ism,'win'].mean()*100:>5.1f}%  OOS={comb.loc[~ism,'win'].mean()*100:>5.1f}%  "
          f"(trade-weighted)")

    # dump the per-trade R-multiple distribution (R = pnl_pts / sl_dist) for the eval gamble sim.
    # R = +1 for a TP, -1 for an SL, partial for a force-close -> NOT clean +-1500.
    rrows = []
    for s in ["OD", "B2", "RV", "FB"]:
        t = tmp[s]
        r = (t["pnl_pts"] / t["sl_dist_pts"]).to_numpy()
        for dt, rr, rsn in zip(t["date"], r, t["reason"]):
            rrows.append((str(dt), s, rr, rsn))
    out = pd.DataFrame(rrows, columns=["date", "strat", "R", "reason"])
    out_path = Path(__file__).resolve().parents[1] / "farm_income" / "combined_eval_R.csv"
    out.to_csv(out_path, index=False)
    print(f"\n  wrote {out_path.name}: {len(out)} trades, R mean={out['R'].mean():.3f} "
          f"(TP=+1/SL=-1/FC=partial); win%={(out['R']>0).mean()*100:.1f}")


if __name__ == "__main__":
    main()
