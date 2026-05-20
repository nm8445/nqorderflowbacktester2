"""
Phase 2 - take top-50 base configs from phase 1, enrich with orderflow filters
(windowed absorption + multi-level absorption) and gamma regime filter (3 modes).

Outputs: per base config x orderflow params x gamma mode = enriched configs.
Reports top robust IS+OOS configs and best orderflow lifts vs baseline.
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
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

import core  # noqa: E402


# Orderflow grids
WINDOW_N_LIST = [3, 5, 8, 10]                                    # ticks
WINDOW_D_15M = [100, 250, 500, 1000, 2500]
WINDOW_D_20M = [150, 350, 750, 1500, 3500]
PER_LEVEL_D_LIST = list(range(25, 301, 25))                      # 25, 50, ..., 300 (12 vals)
K_LIST = [2, 3, 4, 5]
GAMMA_MODES = [(0, "none"), (1, "drop_neg"), (2, "drop_pos")]

MIN_TRADES_TOTAL = 500


def build_orderflow_cache(bar_minutes):
    """Build orderflow masks for a bar-size; pickle to disk."""
    cache_path = CACHE_DIR / f"orderflow_{bar_minutes}m.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"  building orderflow features for {bar_minutes}m ...")
    t0 = time.time()
    bars = core.build_bars(bar_minutes)
    window_d = WINDOW_D_15M if bar_minutes == 15 else WINDOW_D_20M
    feats = core.build_orderflow_features(
        bars, bar_minutes,
        window_n_ticks_list=WINDOW_N_LIST,
        window_d_list=window_d,
        per_level_d_list=PER_LEVEL_D_LIST,
        k_list=K_LIST,
    )
    with open(cache_path, "wb") as f:
        pickle.dump(feats, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  done in {time.time()-t0:.0f}s -> {cache_path}")
    return feats


def build_gamma_cache(bar_minutes):
    cache_path = CACHE_DIR / f"gamma_{bar_minutes}m.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    bars = core.build_bars(bar_minutes)
    gs = core.load_gamma_signs(bars.index)
    with open(cache_path, "wb") as f:
        pickle.dump(gs, f, protocol=pickle.HIGHEST_PROTOCOL)
    return gs


# Worker-side caches
_BARS = {}
_OF = {}
_GS = {}


def _worker_init():
    core.warmup_jit()


def _get_bars(bm):
    if bm not in _BARS:
        with open(CACHE_DIR / f"bars_{bm}m.pkl", "rb") as f:
            _BARS[bm] = pickle.load(f)
    return _BARS[bm]


def _get_of(bm):
    if bm not in _OF:
        with open(CACHE_DIR / f"orderflow_{bm}m.pkl", "rb") as f:
            _OF[bm] = pickle.load(f)
    return _OF[bm]


def _get_gs(bm):
    if bm not in _GS:
        with open(CACHE_DIR / f"gamma_{bm}m.pkl", "rb") as f:
            _GS[bm] = pickle.load(f)
    return _GS[bm]


def evaluate_base(b, z_vol, ema, long_mask, short_mask, g_sign,
                  hz, sl, tp, gamma_mode):
    pnls, in_is = core.backtest_jit(
        b["highs"], b["lows"], b["closes"],
        z_vol, ema, b["atr"],
        b["minutes_of_day"], b["day_idx"],
        long_mask, short_mask, g_sign,
        hz, sl, tp,
        core.SESSION_START_MIN, core.SESSION_END_MIN,
        core.MAX_TRADES_PER_DAY, gamma_mode, b["is_end_ord"],
    )
    if len(pnls) == 0:
        return None
    return core.calc_metrics(pnls, in_is)


def worker_one_base(task):
    """task = dict(bm, norm, zlook, ema_len, hz, sl, tp).
    Sweeps orderflow variants + gamma modes; returns list of rows.
    """
    bm = task["bm"]
    b = _get_bars(bm)
    of = _get_of(bm)
    gs = _get_gs(bm)
    n = len(b["closes"])
    z_vol = core.compute_zvol(b["closes"], task["norm"], task["zlook"])
    ema = core.compute_ema(b["closes"], task["ema_len"])
    ones = np.ones(n, dtype=np.int8)

    window_d = WINDOW_D_15M if bm == 15 else WINDOW_D_20M

    rows = []
    base_key = (task["bm"], task["norm"], task["zlook"], task["ema_len"],
                task["hz"], task["sl"], task["tp"])

    # Baseline (no orderflow, no gamma)
    m = evaluate_base(b, z_vol, ema, ones, ones, gs,
                       task["hz"], task["sl"], task["tp"], 0)
    if m is None:
        return rows
    rows.append(base_key + ("baseline", 0, 0, "none",
                              m["is"]["trades"], m["is"]["pf"], m["is"]["wr"], m["is"]["pnl"], m["is"]["mdd"],
                              m["oos"]["trades"], m["oos"]["pf"], m["oos"]["wr"], m["oos"]["pnl"], m["oos"]["mdd"],
                              m["total"]["trades"], m["total"]["pf"], m["total"]["wr"], m["total"]["pnl"], m["total"]["mdd"]))

    # Gamma-only (no orderflow)
    for gm, gname in GAMMA_MODES:
        if gm == 0:
            continue
        m = evaluate_base(b, z_vol, ema, ones, ones, gs,
                           task["hz"], task["sl"], task["tp"], gm)
        if m is None: continue
        rows.append(base_key + ("gamma_only", 0, 0, gname,
                                 m["is"]["trades"], m["is"]["pf"], m["is"]["wr"], m["is"]["pnl"], m["is"]["mdd"],
                                 m["oos"]["trades"], m["oos"]["pf"], m["oos"]["wr"], m["oos"]["pnl"], m["oos"]["mdd"],
                                 m["total"]["trades"], m["total"]["pf"], m["total"]["wr"], m["total"]["pnl"], m["total"]["mdd"]))

    # Windowed orderflow
    for N in WINDOW_N_LIST:
        for D in window_d:
            lmask = of["window_long"][(N, D)]
            smask = of["window_short"][(N, D)]
            for gm, gname in GAMMA_MODES:
                m = evaluate_base(b, z_vol, ema, lmask, smask, gs,
                                   task["hz"], task["sl"], task["tp"], gm)
                if m is None: continue
                rows.append(base_key + (f"window_N{N}_D{D}", N, D, gname,
                                         m["is"]["trades"], m["is"]["pf"], m["is"]["wr"], m["is"]["pnl"], m["is"]["mdd"],
                                         m["oos"]["trades"], m["oos"]["pf"], m["oos"]["wr"], m["oos"]["pnl"], m["oos"]["mdd"],
                                         m["total"]["trades"], m["total"]["pf"], m["total"]["wr"], m["total"]["pnl"], m["total"]["mdd"]))

    # Multi-level orderflow
    for pd_v in PER_LEVEL_D_LIST:
        for K in K_LIST:
            lmask = of["multi_long"][(pd_v, K)]
            smask = of["multi_short"][(pd_v, K)]
            for gm, gname in GAMMA_MODES:
                m = evaluate_base(b, z_vol, ema, lmask, smask, gs,
                                   task["hz"], task["sl"], task["tp"], gm)
                if m is None: continue
                rows.append(base_key + (f"multi_pd{pd_v}_K{K}", pd_v, K, gname,
                                         m["is"]["trades"], m["is"]["pf"], m["is"]["wr"], m["is"]["pnl"], m["is"]["mdd"],
                                         m["oos"]["trades"], m["oos"]["pf"], m["oos"]["wr"], m["oos"]["pnl"], m["oos"]["mdd"],
                                         m["total"]["trades"], m["total"]["pf"], m["total"]["wr"], m["total"]["pnl"], m["total"]["mdd"]))

    return rows


COLS = ["bm", "norm", "zlook", "ema", "hz", "sl", "tp",
        "variant", "of_p1", "of_p2", "gamma",
        "is_t", "is_pf", "is_wr", "is_pnl", "is_mdd",
        "oos_t", "oos_pf", "oos_wr", "oos_pnl", "oos_mdd",
        "tot_t", "tot_pf", "tot_wr", "tot_pnl", "tot_mdd"]


def main():
    print("=" * 80)
    print("Phase 2 - orderflow + gamma enrichment")
    print(f"Variants: baseline, gamma_only, windowed ({len(WINDOW_N_LIST)} N x {len(WINDOW_D_15M)} D),")
    print(f"           multi-level ({len(PER_LEVEL_D_LIST)} pd x {len(K_LIST)} K)")
    print(f"Gamma modes: {len(GAMMA_MODES)} (none, drop_neg, drop_pos)")
    print(f"Min total trades = {MIN_TRADES_TOTAL}")
    print("=" * 80)

    # Build caches
    for bm in (15, 20):
        build_orderflow_cache(bm)
        build_gamma_cache(bm)

    # Load top-50 base configs per bar-size
    tasks = []
    for bm in (15, 20):
        top = pd.read_csv(RESULTS_DIR / f"phase1_top50_{bm}m.csv")
        for _, r in top.iterrows():
            tasks.append(dict(
                bm=int(r["bm"]), norm=int(r["norm"]), zlook=int(r["zlook"]),
                ema_len=int(r["ema"]), hz=float(r["hz"]),
                sl=float(r["sl"]), tp=float(r["tp"]),
            ))
    print(f"\n{len(tasks)} base configs to enrich")

    t0 = time.time()
    all_rows = []
    with mp.Pool(6, initializer=_worker_init) as p:
        for i, sub in enumerate(p.imap_unordered(worker_one_base, tasks), 1):
            all_rows.extend(sub)
            if i % 10 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                eta = elapsed / i * (len(tasks) - i)
                print(f"  {i}/{len(tasks)} base configs done ({elapsed:.0f}s, ETA {eta:.0f}s) "
                      f"- {len(all_rows)} enriched rows")
    print(f"\nEnrichment complete in {time.time()-t0:.0f}s.")

    df = pd.DataFrame(all_rows, columns=COLS)
    # Filter total trades
    df = df[df["tot_t"] >= MIN_TRADES_TOTAL].copy()
    df.to_csv(RESULTS_DIR / "phase2_all_enriched.csv", index=False)
    print(f"Wrote phase2_all_enriched.csv ({len(df)} rows after trade filter)")

    # Robust: IS PF > 1 AND OOS PF > 1
    rob = df[(df["is_pf"] > 1.0) & (df["oos_pf"] > 1.0)].copy()
    rob["robust_pf"] = rob[["is_pf", "oos_pf"]].min(axis=1)
    rob = rob.sort_values("robust_pf", ascending=False)
    print(f"Robust enriched: {len(rob)}")

    # Top 30 overall per bar-size
    txt = RESULTS_DIR / "phase2_top_configs.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("Phase 2 - top robust enriched configs (IS PF > 1 AND OOS PF > 1)\n")
        f.write(f"Min total trades = {MIN_TRADES_TOTAL}\n\n")
        for bm in (15, 20):
            sub = rob[rob["bm"] == bm].head(30)
            f.write(f"=== {bm}m bars - top {len(sub)} robust ===\n")
            for _, r in sub.iterrows():
                f.write(f"{int(r['bm']):>2}m N={int(r['norm']):>3} Z={int(r['zlook']):>3} "
                        f"EMA={int(r['ema']):>2} HZ={r['hz']:.2f} SL={r['sl']:.1f} TP={r['tp']:.2f} "
                        f"| {r['variant']:>18} | gamma={r['gamma']:>8} "
                        f"| IS {int(r['is_t']):>4}t PF{r['is_pf']:>5.2f} ${r['is_pnl']:>+9,.0f} DD${r['is_mdd']:>+9,.0f} "
                        f"| OOS {int(r['oos_t']):>3}t PF{r['oos_pf']:>5.2f} ${r['oos_pnl']:>+8,.0f} DD${r['oos_mdd']:>+9,.0f} "
                        f"| TOT {int(r['tot_t']):>4}t PF{r['tot_pf']:>5.2f} ${r['tot_pnl']:>+9,.0f}\n")
            f.write("\n")

        # Best-of-baseline vs best-of-enriched, per base
        f.write("=== Lift comparison (baseline vs best enriched per base config) ===\n")
        for bm in (15, 20):
            f.write(f"\n--- {bm}m ---\n")
            sub = rob[rob["bm"] == bm]
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
                lifts.append((bk["norm"], bk["zlook"], bk["ema"], bk["hz"], bk["sl"], bk["tp"],
                              bl_pf, best["robust_pf"], best["variant"], best["gamma"], int(best["tot_t"])))
            lifts.sort(key=lambda x: x[7] - x[6], reverse=True)
            f.write("  base config -> baseline_pf -> best_pf (variant/gamma, trades)\n")
            for L in lifts[:20]:
                f.write(f"  N={int(L[0]):>3} Z={int(L[1]):>3} EMA={int(L[2]):>2} HZ={L[3]:.2f} "
                        f"SL={L[4]:.1f} TP={L[5]:.2f} | "
                        f"baseline PF{L[6]:.2f} -> best PF{L[7]:.2f} "
                        f"({L[8]}, gamma={L[9]}, {L[10]}t)\n")
    print(f"Wrote {txt}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
