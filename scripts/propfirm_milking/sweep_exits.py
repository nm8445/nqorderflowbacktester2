"""Phase 1 — fixed-ATR-bracket exit sweep, per strategy, parallel.

For each strategy, replaces the native exit with a pure ATR bracket and sweeps
    atr_len x sl_mult x tp_mult
over the LOCKED entries (from the 4-way log). Reports per-config all/IS/OOS
stats and applies robustness gates:
  - tagged_frac >= TAGGED_MIN on BOTH IS and OOS  (SL/TP actually get hit)
  - net > 0 on BOTH IS and OOS                     (>breakeven WR for the RR)
  - n_trades >= MIN_TRADES
Writes results/<strat>_bracket_sweep.csv and prints top configs.

Run:  python -m scripts.propfirm_milking.sweep_exits   (from repo root)
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from scripts.propfirm_milking.common import (
    ATR_LEN_GRID, SL_MULT_GRID, TP_MULT_GRID, RESULTS_DIR,
    eval_bracket, stats_block, is_oos_cutoff,
)
from scripts.propfirm_milking.entries import build_entry_packs

STRATS = ["RV", "B2", "FB", "MEC"]   # challenge set: OD dropped, MEC added
TAGGED_MIN = 0.70
MIN_TRADES = 100
MAX_WORKERS = 6

# Worker globals
_PACKS = None
_CUTOFF = None


def _init_worker(packs, cutoff):
    global _PACKS, _CUTOFF
    _PACKS = packs
    _CUTOFF = cutoff


def _eval_cell(args):
    atr_len, sl_mult, tp_mult = args
    tr = eval_bracket(_PACKS, atr_len, sl_mult, tp_mult)
    if len(tr) == 0:
        return None
    is_tr = tr[tr["date"] < _CUTOFF]
    oos_tr = tr[tr["date"] >= _CUTOFF]
    a = stats_block(tr)
    i = stats_block(is_tr)
    o = stats_block(oos_tr)
    rr = round(tp_mult / sl_mult, 3)
    qualifies = (
        a["n"] >= MIN_TRADES
        and i["tagged_frac"] >= TAGGED_MIN and o["tagged_frac"] >= TAGGED_MIN
        and i["net"] > 0 and o["net"] > 0
    )
    return {
        "atr_len": atr_len, "sl_mult": sl_mult, "tp_mult": tp_mult, "rr": rr,
        "n": a["n"], "wr": a["wr"], "net": a["net"], "pf": a["pf"], "mdd": a["mdd"],
        "tagged": a["tagged_frac"], "mean_mnq": a["mean_mnq"], "mean_risk": a["mean_risk"],
        "is_n": i["n"], "is_wr": i["wr"], "is_net": i["net"], "is_tagged": i["tagged_frac"],
        "oos_n": o["n"], "oos_wr": o["wr"], "oos_net": o["net"], "oos_tagged": o["tagged_frac"],
        "min_wr": min(i["wr"], o["wr"]), "qualifies": qualifies,
    }


def sweep_strat(strat: str) -> pd.DataFrame:
    t0 = time.time()
    print(f"\n=== {strat}: building entry packs ===")
    packs = build_entry_packs(strat)
    if not packs:
        print(f"  {strat}: no packs, skipping")
        return pd.DataFrame()
    cutoff = is_oos_cutoff([p.date for p in packs])
    grid = [(a, s, t) for a in ATR_LEN_GRID for s in SL_MULT_GRID for t in TP_MULT_GRID]
    print(f"  {len(packs)} entries | IS/OOS cutoff={cutoff} | {len(grid)} cells "
          f"on {MAX_WORKERS} workers")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS,
                             initializer=_init_worker, initargs=(packs, cutoff)) as ex:
        rows = [r for r in ex.map(_eval_cell, grid, chunksize=8) if r is not None]
    df = pd.DataFrame(rows)
    df["strat"] = strat
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{strat}_bracket_sweep.csv"
    df.to_csv(out, index=False)
    print(f"  done in {time.time()-t0:.1f}s -> {out}")

    q = df[df["qualifies"]].copy()
    print(f"  qualifying configs (tagged>={TAGGED_MIN} & IS+OOS net>0): {len(q)}/{len(df)}")
    if len(q):
        show = ["atr_len", "sl_mult", "tp_mult", "rr", "n", "wr",
                "is_wr", "oos_wr", "min_wr", "net", "is_net", "oos_net", "tagged", "mean_mnq"]
        print("  --- top 8 by min(IS_wr, OOS_wr) [pass-rate proxy] ---")
        print(q.sort_values("min_wr", ascending=False).head(8)[show].to_string(index=False))
        print("  --- top 5 by ALL net $ ---")
        print(q.sort_values("net", ascending=False).head(5)[show].to_string(index=False))
    return df


def main():
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    all_df = []
    for s in STRATS:
        d = sweep_strat(s)
        if len(d):
            all_df.append(d)
    if all_df:
        combo = pd.concat(all_df, ignore_index=True)
        combo.to_csv(RESULTS_DIR / "all_bracket_sweep.csv", index=False)
        print("\n=== SUMMARY: best qualifying config per strat (by min_wr) ===")
        for s in STRATS:
            sub = combo[(combo["strat"] == s) & (combo["qualifies"])]
            if not len(sub):
                print(f"  {s}: NO qualifying config")
                continue
            b = sub.sort_values("min_wr", ascending=False).iloc[0]
            print(f"  {s}: atr={int(b['atr_len'])} sl={b['sl_mult']} tp={b['tp_mult']} "
                  f"(RR {b['rr']}) | wr {b['wr']}% (IS {b['is_wr']} / OOS {b['oos_wr']}) "
                  f"| net ${b['net']:,.0f} | tagged {b['tagged']} | {b['mean_mnq']} MNQ")


if __name__ == "__main__":
    main()
