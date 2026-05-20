"""
Phase 2 variant - same enrichment but on top RR>=1.0 base configs from phase 1.
"""
import sys
import time
import pickle
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
RESULTS_DIR = HERE / "results"
CACHE_DIR = HERE / ".cache"

import core  # noqa
from phase2_orderflow_gamma import (
    WINDOW_N_LIST, WINDOW_D_15M, WINDOW_D_20M, PER_LEVEL_D_LIST, K_LIST,
    GAMMA_MODES, MIN_TRADES_TOTAL, COLS,
    _worker_init, worker_one_base, build_orderflow_cache, build_gamma_cache,
)


def main():
    # Caches already built by phase2_orderflow_gamma
    for bm in (15, 20):
        build_orderflow_cache(bm)
        build_gamma_cache(bm)

    # Pull top 50 RR>=1 robust per bar-size from phase 1
    df1 = pd.read_csv(RESULTS_DIR / "phase1_all_configs.csv")
    df1["rr"] = df1["tp"] / df1["sl"]
    rob = df1[(df1["is_pf"] > 1.0) & (df1["oos_pf"] > 1.0) & (df1["tot_t"] >= 500)
              & (df1["rr"] >= 1.0)].copy()
    rob["robust_pf"] = rob[["is_pf", "oos_pf"]].min(axis=1)
    rob = rob.sort_values("robust_pf", ascending=False)

    tasks = []
    for bm in (15, 20):
        top = rob[rob["bm"] == bm].head(50)
        for _, r in top.iterrows():
            tasks.append(dict(
                bm=int(r["bm"]), norm=int(r["norm"]), zlook=int(r["zlook"]),
                ema_len=int(r["ema"]), hz=float(r["hz"]),
                sl=float(r["sl"]), tp=float(r["tp"]),
            ))
    print(f"{len(tasks)} top RR>=1 base configs to enrich")

    t0 = time.time()
    all_rows = []
    with mp.Pool(6, initializer=_worker_init) as p:
        for i, sub in enumerate(p.imap_unordered(worker_one_base, tasks), 1):
            all_rows.extend(sub)
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} done ({time.time()-t0:.0f}s) - {len(all_rows)} rows")
    print(f"Done in {time.time()-t0:.0f}s")

    df = pd.DataFrame(all_rows, columns=COLS)
    df = df[df["tot_t"] >= MIN_TRADES_TOTAL].copy()
    df.to_csv(RESULTS_DIR / "phase2_highrr_all.csv", index=False)

    rob2 = df[(df["is_pf"] > 1.0) & (df["oos_pf"] > 1.0)].copy()
    rob2["robust_pf"] = rob2[["is_pf", "oos_pf"]].min(axis=1)
    rob2 = rob2.sort_values("robust_pf", ascending=False)
    print(f"\nRobust enriched (RR>=1): {len(rob2)}")

    txt = RESULTS_DIR / "phase2_highrr_top.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("Phase 2 - top robust enriched configs (RR >= 1.0, IS+OOS PF > 1)\n")
        f.write(f"Min total trades = {MIN_TRADES_TOTAL}\n\n")
        for bm in (15, 20):
            sub = rob2[rob2["bm"] == bm].head(30)
            f.write(f"=== {bm}m bars - top {len(sub)} robust (RR>=1) ===\n")
            for _, r in sub.iterrows():
                rr = r["tp"] / r["sl"]
                f.write(f"{int(r['bm']):>2}m N={int(r['norm']):>3} Z={int(r['zlook']):>3} "
                        f"EMA={int(r['ema']):>2} HZ={r['hz']:.2f} SL={r['sl']:.1f} TP={r['tp']:.2f} RR={rr:.2f} "
                        f"| {r['variant']:>18} | g={r['gamma']:>8} "
                        f"| IS {int(r['is_t']):>4}t PF{r['is_pf']:>5.2f} WR{r['is_wr']:>4.1f}% "
                        f"${r['is_pnl']:>+9,.0f} DD${r['is_mdd']:>+8,.0f} "
                        f"| OOS {int(r['oos_t']):>3}t PF{r['oos_pf']:>5.2f} WR{r['oos_wr']:>4.1f}% "
                        f"${r['oos_pnl']:>+8,.0f} DD${r['oos_mdd']:>+8,.0f} "
                        f"| TOT {int(r['tot_t']):>4}t PF{r['tot_pf']:>5.2f} ${r['tot_pnl']:>+9,.0f}\n")
            f.write("\n")

        # Lift vs baseline per base config
        f.write("=== Lift table (baseline -> best enriched) ===\n")
        for bm in (15, 20):
            f.write(f"\n--- {bm}m RR>=1 ---\n")
            sub = rob2[rob2["bm"] == bm]
            base_keys = sub[["norm", "zlook", "ema", "hz", "sl", "tp"]].drop_duplicates()
            lifts = []
            for _, bk in base_keys.iterrows():
                grp = sub[(sub["norm"] == bk["norm"]) & (sub["zlook"] == bk["zlook"]) &
                          (sub["ema"] == bk["ema"]) & (sub["hz"] == bk["hz"]) &
                          (sub["sl"] == bk["sl"]) & (sub["tp"] == bk["tp"])]
                bl = grp[grp["variant"] == "baseline"]
                if len(bl) == 0: continue
                bl_pf = bl["robust_pf"].iloc[0]
                best = grp.sort_values("robust_pf", ascending=False).iloc[0]
                rr = bk["tp"] / bk["sl"]
                lifts.append((bk["norm"], bk["zlook"], bk["ema"], bk["hz"], bk["sl"], bk["tp"], rr,
                              bl_pf, best["robust_pf"], best["variant"], best["gamma"], int(best["tot_t"])))
            lifts.sort(key=lambda x: x[8] - x[7], reverse=True)
            for L in lifts[:15]:
                f.write(f"  N={int(L[0]):>3} Z={int(L[1]):>3} EMA={int(L[2]):>2} HZ={L[3]:.2f} "
                        f"SL={L[4]:.1f} TP={L[5]:.2f} RR={L[6]:.2f} | "
                        f"baseline PF{L[7]:.2f} -> best PF{L[8]:.2f} "
                        f"({L[9]}, g={L[10]}, {L[11]}t)\n")
    print(f"Wrote {txt}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
