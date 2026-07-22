"""Do EMA-cross and the 9:30-10:00 orderflow delta stack? Trade only when they AGREE.

If the orderflow adds ORTHOGONAL info, agreement WR > EMA-alone (52.4%) and disagreement WR < 50%.
Same exec as before (10:00 entry, 1.5x ATR(10) 1:1 bracket, FC 14:30, 60/40 split).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scripts.propfirm_milking.common import eval_bracket, is_oos_cutoff
from scripts.propfirm_milking.entries import build_entry_packs
from delta_cross import daily_delta

ATR_LEN, SL, TP = 10, 1.5, 1.5


def stats(packs, label):
    if not packs:
        print(f"  {label:>28}  (no trades)"); return
    tr = eval_bracket(packs, atr_len=ATR_LEN, sl_mult=SL, tp_mult=TP)
    cut = is_oos_cutoff(list(tr["date"])); ism = tr["date"] < cut
    wr = lambda s: s["win"].mean()*100 if len(s) else float("nan")
    print(f"  {label:>28}  n={len(tr):>4}  WR={wr(tr):>5.1f}%  IS={wr(tr[ism]):>5.1f}%  "
          f"OOS={wr(tr[~ism]):>5.1f}%  EV=${tr['pnl_$'].mean():>5.0f}")


def main():
    g = daily_delta()
    dsign = {d: (1 if x > 0 else -1) for d, x in zip(g.index, g["delta"]) if x != 0}
    dpct = {d: abs(p) for d, p in zip(g.index, g["pct"])}
    packs = build_entry_packs("MEC", atr_lens=[ATR_LEN], verbose=False)
    packs = [p for p in packs if p.date in dsign]   # keep days with a delta signal

    agree = [p for p in packs if dsign[p.date] == p.sign]
    disagree = [p for p in packs if dsign[p.date] != p.sign]
    # agreement + stronger imbalance (Q4+: |pct| >= 0.019)
    agree_strong = [p for p in agree if dpct.get(p.date, 0) >= 0.019]
    agree_mod = [p for p in agree if 0.019 <= dpct.get(p.date, 0) < 0.028]   # Q4 sweet spot

    print(f"EMA-cross x 9:30-10:00 delta AGREEMENT (1.5x ATR(10), FC 14:30):  {len(packs)} aligned days\n")
    stats(packs, "EMA all (baseline)")
    stats(agree, "EMA == delta (agree)")
    stats(disagree, "EMA != delta (disagree)")
    stats(agree_strong, "agree + |pct|>=1.9% (Q4+)")
    stats(agree_mod, "agree + Q4 only (1.9-2.8%)")


if __name__ == "__main__":
    main()
