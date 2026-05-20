"""
Phase 1 - base param sweep for rough vol + orderflow strategy.
Sweeps: NORM_LEN, Z_LOOKBACK, HIGH_Z, EMA_LEN, ATR_SL, ATR_TP
Bar sizes: 15-min and 20-min
No orderflow / no gamma / no martingale (base PnL).
Session 05:45-14:45 ET. IS through 2024-12-31, OOS 2025+.
Filter: total trades >= 500. 6 workers.
"""
import sys
import time
import pickle
import multiprocessing as mp
from pathlib import Path
from itertools import product

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
RESULTS_DIR = HERE / "results"
CACHE_DIR = HERE / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

import core  # noqa: E402


# Grids
NORM_GRID = list(range(100, 401, 25))           # 100, 125, ..., 400  (13 vals)
ZLOOK_GRID = list(range(50, 201, 25))           # 50, 75, ..., 200    (7 vals)
HIGH_Z_GRID = [round(1.0 + 0.05 * k, 2) for k in range(21)]  # 1.00..2.00 (21)
EMA_GRID = [20, 40, 60, 80]                      # 4
ATR_SL_GRID = [1.5, 2.0, 2.5, 3.0]              # 4
ATR_TP_GRID = [0.75, 1.0, 1.25, 1.5, 2.0]       # 5

MIN_TRADES_TOTAL = 500


def cache_bars_to_disk(bar_minutes):
    """Build bars + ATR; pickle to disk for fast worker load."""
    cache_path = CACHE_DIR / f"bars_{bar_minutes}m.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    print(f"  building {bar_minutes}-min bars (one-time)...")
    df = core.build_bars(bar_minutes)
    arr = core.extract_arrays(df)
    arr["atr"] = core.compute_atr(arr["highs"], arr["lows"], arr["closes"])
    # we don't need idx_et for the sweep
    pack = {k: v for k, v in arr.items() if k != "idx_et"}
    with open(cache_path, "wb") as f:
        pickle.dump(pack, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  cached -> {cache_path} ({len(df)} bars)")
    return pack


# Worker-side cache: each process loads bars once.
_WORKER_BARS = {}


def _get_worker_bars(bar_minutes):
    if bar_minutes not in _WORKER_BARS:
        with open(CACHE_DIR / f"bars_{bar_minutes}m.pkl", "rb") as f:
            _WORKER_BARS[bar_minutes] = pickle.load(f)
    return _WORKER_BARS[bar_minutes]


def worker(task):
    """task = (bar_minutes, norm, zlook, ema_len) → list of result tuples."""
    bar_minutes, norm, zlook, ema_len = task
    b = _get_worker_bars(bar_minutes)
    n = len(b["closes"])
    of_mask = np.ones(n, dtype=np.int8)
    g_sign = np.zeros(n, dtype=np.int8)

    z_vol = core.compute_zvol(b["closes"], norm, zlook)
    ema = core.compute_ema(b["closes"], ema_len)

    out = []
    for high_z in HIGH_Z_GRID:
        for atr_sl in ATR_SL_GRID:
            for atr_tp in ATR_TP_GRID:
                pnls, in_is = core.backtest_jit(
                    b["highs"], b["lows"], b["closes"],
                    z_vol, ema, b["atr"],
                    b["minutes_of_day"], b["day_idx"],
                    of_mask, of_mask, g_sign,
                    high_z, atr_sl, atr_tp,
                    core.SESSION_START_MIN, core.SESSION_END_MIN,
                    core.MAX_TRADES_PER_DAY, 0, b["is_end_ord"],
                )
                if len(pnls) < MIN_TRADES_TOTAL:
                    continue
                m = core.calc_metrics(pnls, in_is)
                if m["is"]["trades"] == 0 or m["oos"]["trades"] == 0:
                    continue
                out.append((
                    bar_minutes, norm, zlook, ema_len, high_z, atr_sl, atr_tp,
                    m["is"]["trades"], m["is"]["pf"], m["is"]["wr"], m["is"]["pnl"], m["is"]["mdd"],
                    m["oos"]["trades"], m["oos"]["pf"], m["oos"]["wr"], m["oos"]["pnl"], m["oos"]["mdd"],
                    m["total"]["trades"], m["total"]["pf"], m["total"]["wr"], m["total"]["pnl"], m["total"]["mdd"],
                ))
    return out


def _worker_init():
    # Force the numba JIT compile once per worker
    core.warmup_jit()


def format_row(r):
    (bm, norm, zlook, ema, hz, sl, tp,
     ist, ipf, iwr, ipnl, imdd,
     ot, opf, owr, opnl, omdd,
     tt, tpf, twr, tpnl, tmdd) = r
    return (f"{bm}m  N={norm:>3} Z={zlook:>3} EMA={ema:>2} HZ={hz:.2f} "
            f"SL={sl:.1f} TP={tp:.2f} | "
            f"IS {ist:>4}t PF{ipf:>4.2f} WR{iwr:>4.1f}% ${ipnl:>+8,.0f} DD${imdd:>+9,.0f} | "
            f"OOS {ot:>3}t PF{opf:>4.2f} WR{owr:>4.1f}% ${opnl:>+8,.0f} DD${omdd:>+9,.0f} | "
            f"TOT {tt:>4}t PF{tpf:>4.2f} ${tpnl:>+9,.0f}")


def main():
    print("=" * 80)
    print("Phase 1 - base param sweep (rough vol, no orderflow/gamma/martingale)")
    print(f"Grid: norm({len(NORM_GRID)}) x zlook({len(ZLOOK_GRID)}) x hz({len(HIGH_Z_GRID)}) "
          f"x ema({len(EMA_GRID)}) x sl({len(ATR_SL_GRID)}) x tp({len(ATR_TP_GRID)}) "
          f"= {len(NORM_GRID)*len(ZLOOK_GRID)*len(HIGH_Z_GRID)*len(EMA_GRID)*len(ATR_SL_GRID)*len(ATR_TP_GRID)} per bar-size")
    print(f"Session 05:45-14:45 ET, IS <= {core.IS_END}, min total trades = {MIN_TRADES_TOTAL}")
    print("=" * 80)

    # Build & cache bars
    for bm in (15, 20):
        cache_bars_to_disk(bm)

    # Build task list (outer combos)
    tasks = []
    for bm in (15, 20):
        for norm, zlook, ema in product(NORM_GRID, ZLOOK_GRID, EMA_GRID):
            tasks.append((bm, norm, zlook, ema))
    print(f"\n{len(tasks)} outer tasks, {len(HIGH_Z_GRID)*len(ATR_SL_GRID)*len(ATR_TP_GRID)} inner per outer")

    t0 = time.time()
    all_results = []
    with mp.Pool(6, initializer=_worker_init) as p:
        for i, sub in enumerate(p.imap_unordered(worker, tasks), 1):
            all_results.extend(sub)
            if i % 50 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                eta = elapsed / i * (len(tasks) - i)
                print(f"  {i}/{len(tasks)} outer tasks done ({elapsed:.0f}s elapsed, ETA {eta:.0f}s) "
                      f"- {len(all_results)} configs passing trade threshold")
    print(f"\nSweep complete in {time.time()-t0:.0f}s. {len(all_results)} configs passed filter.")

    if not all_results:
        print("No configs met threshold.")
        return

    # Persist raw results
    cols = ["bm", "norm", "zlook", "ema", "hz", "sl", "tp",
            "is_t", "is_pf", "is_wr", "is_pnl", "is_mdd",
            "oos_t", "oos_pf", "oos_wr", "oos_pnl", "oos_mdd",
            "tot_t", "tot_pf", "tot_wr", "tot_pnl", "tot_mdd"]
    import pandas as pd
    df = pd.DataFrame(all_results, columns=cols)
    out_csv = RESULTS_DIR / "phase1_all_configs.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    # Top configs (per bar size) - robust = IS PF > 1, OOS PF > 1, ranked by min(IS PF, OOS PF)
    robust = df[(df["is_pf"] > 1.0) & (df["oos_pf"] > 1.0)].copy()
    robust["robust_pf"] = robust[["is_pf", "oos_pf"]].min(axis=1)
    robust = robust.sort_values("robust_pf", ascending=False)
    print(f"\nRobust configs (IS PF > 1 AND OOS PF > 1): {len(robust)}")

    # Per-bar-size top 50
    top_per_bm = {}
    for bm in (15, 20):
        sub = robust[robust["bm"] == bm].head(50)
        top_per_bm[bm] = sub
        out_csv2 = RESULTS_DIR / f"phase1_top50_{bm}m.csv"
        sub.to_csv(out_csv2, index=False)
        print(f"  {bm}m: {len(sub)} top configs -> {out_csv2}")

    # Text report
    txt = RESULTS_DIR / "phase1_top_configs.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("Phase 1 - top robust configs (IS PF > 1 AND OOS PF > 1)\n")
        f.write(f"Session 05:45-14:45 ET, IS <= {core.IS_END}, min total trades = {MIN_TRADES_TOTAL}\n")
        f.write(f"Total robust configs: {len(robust)}\n\n")
        for bm in (15, 20):
            sub = top_per_bm[bm]
            f.write(f"\n=== {bm}-min bars - top {len(sub)} ===\n")
            for _, r in sub.iterrows():
                f.write(format_row(tuple(r[c] for c in cols)) + "\n")
    print(f"Wrote {txt}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
