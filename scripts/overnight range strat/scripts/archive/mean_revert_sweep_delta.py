"""Sweep conf_D and conf_N on the mean-revert signals + 4 exit variants.

Uses the cached mean_revert_signals.parquet from mean_revert_strategy.py.
"""
from __future__ import annotations

import sys
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mean_revert_strategy import (
    MR_OUT, IS_START, IS_END, OOS_END,
    simulate_variant_A, simulate_variant_B, simulate_variant_C, simulate_variant_D,
    chained_simulate, stats, load_5min_features, load_mq_levels, levels_for_date,
)

TRADELOG_DIR = Path(__file__).parent.parent / "tradelogs"
OUT_TXT = TRADELOG_DIR / "robust_configs" / "mean_revert_sweep_delta.txt"

CONF_N_GRID = [5, 10, 15, 20]
CONF_D_GRID = [0, 25, 50, 75, 100]   # 0 = effectively no filter (signal body filter still applied)
SYM_PTS_GRID = [20, 30, 40, 50]


def main():
    if not MR_OUT.exists():
        print("Signal parquet missing -- run mean_revert_strategy.py first.")
        return
    sigs = pd.read_parquet(MR_OUT)
    sigs["entry_time"] = pd.to_datetime(sigs["entry_time"])
    sigs["date"] = pd.to_datetime(sigs["date"]).dt.date
    is_sigs  = sigs[sigs["date"] <  IS_END].copy()
    oos_sigs = sigs[sigs["date"] >= IS_END].copy()
    print(f"signals: IS={len(is_sigs)}  OOS={len(oos_sigs)}")

    print("loading 5-min bars...")
    bars_all, _ = load_5min_features((IS_START, OOS_END))
    bars_by_day = {d: g.sort_values("bar_open_time").reset_index(drop=True)
                   for d, g in bars_all.groupby("session_date")}
    mq = load_mq_levels()
    def mq_lookup_for_date(d):
        return levels_for_date(mq, d)

    rows = []
    print("sweeping conf_N x conf_D x exit_variant ...")
    n_combos = len(CONF_N_GRID) * len(CONF_D_GRID) * (len(SYM_PTS_GRID) + 3)
    print(f"  total combos: {n_combos}")
    done = 0

    for N in CONF_N_GRID:
        for D in CONF_D_GRID:
            # Variant A: sym pts grid
            for pts in SYM_PTS_GRID:
                kw = dict(variant_fn=simulate_variant_A, sym_pts=pts)
                is_t  = chained_simulate(is_sigs,  bars_by_day, conf_N=N, conf_D_min=D, **kw)
                oos_t = chained_simulate(oos_sigs, bars_by_day, conf_N=N, conf_D_min=D, **kw)
                all_t = pd.concat([is_t, oos_t], ignore_index=True)
                rows.append({"variant": f"A pts={pts}", "conf_N": N, "conf_D": D,
                             **{f"is_{k}": v for k, v in stats(is_t).items()},
                             **{f"oos_{k}": v for k, v in stats(oos_t).items()},
                             **{f"all_{k}": v for k, v in stats(all_t).items()}})
                done += 1
            # Variant B
            kw = dict(variant_fn=simulate_variant_B, mq_lookup_for_date=mq_lookup_for_date)
            is_t  = chained_simulate(is_sigs,  bars_by_day, conf_N=N, conf_D_min=D, **kw)
            oos_t = chained_simulate(oos_sigs, bars_by_day, conf_N=N, conf_D_min=D, **kw)
            all_t = pd.concat([is_t, oos_t], ignore_index=True)
            rows.append({"variant": "B lvl-to-lvl", "conf_N": N, "conf_D": D,
                         **{f"is_{k}": v for k, v in stats(is_t).items()},
                         **{f"oos_{k}": v for k, v in stats(oos_t).items()},
                         **{f"all_{k}": v for k, v in stats(all_t).items()}})
            done += 1
            # Variant C
            kw = dict(variant_fn=simulate_variant_C, sl_m=1.0, tp_m=2.0)
            is_t  = chained_simulate(is_sigs,  bars_by_day, conf_N=N, conf_D_min=D, **kw)
            oos_t = chained_simulate(oos_sigs, bars_by_day, conf_N=N, conf_D_min=D, **kw)
            all_t = pd.concat([is_t, oos_t], ignore_index=True)
            rows.append({"variant": "C 1xSL/2xTP", "conf_N": N, "conf_D": D,
                         **{f"is_{k}": v for k, v in stats(is_t).items()},
                         **{f"oos_{k}": v for k, v in stats(oos_t).items()},
                         **{f"all_{k}": v for k, v in stats(all_t).items()}})
            done += 1
            # Variant D
            kw = dict(variant_fn=simulate_variant_D)
            is_t  = chained_simulate(is_sigs,  bars_by_day, conf_N=N, conf_D_min=D, **kw)
            oos_t = chained_simulate(oos_sigs, bars_by_day, conf_N=N, conf_D_min=D, **kw)
            all_t = pd.concat([is_t, oos_t], ignore_index=True)
            rows.append({"variant": "D mfe-hybrid", "conf_N": N, "conf_D": D,
                         **{f"is_{k}": v for k, v in stats(is_t).items()},
                         **{f"oos_{k}": v for k, v in stats(oos_t).items()},
                         **{f"all_{k}": v for k, v in stats(all_t).items()}})
            done += 1
            if done % 20 == 0:
                print(f"  done {done}/{n_combos}")

    df = pd.DataFrame(rows)
    df["min_pf"] = df[["is_pf", "oos_pf"]].min(axis=1)
    df["min_wr"] = df[["is_wr", "oos_wr"]].min(axis=1)
    df["worst_mdd"] = df[["is_mdd", "oos_mdd"]].abs().max(axis=1)

    L = []
    L.append("=" * 220)
    L.append("MEAN-REVERT — conf_N x conf_D x EXIT_VARIANT sweep  (IS+OOS, chained Mode 1, FC 16:00 ET)")
    L.append("=" * 220)
    L.append(f"Signal candidates pre-conf-D filter: IS={len(is_sigs)}  OOS={len(oos_sigs)}")
    L.append("")
    L.append("All candidates already passed: pinbar X>=0.75, GEX-level proximity (BAND_K=0.25*ATR),")
    L.append("approach filter, bias NEUTRAL, hours 9-14 ET, conf direction match, conf_body > sig_body.")
    L.append("conf_D = 0 means: no extra orderflow-delta filter on conf bar (body filter still applied).")
    L.append("")
    L.append("=" * 220)
    L.append("TOP 15 BY combined-IS+OOS TOTAL (filtered to both IS and OOS positive AND both WR>=50%)")
    L.append("=" * 220)
    qual = df[(df["is_total"] > 0) & (df["oos_total"] > 0)
              & (df["is_wr"] >= 50) & (df["oos_wr"] >= 50)
              & (df["is_n"] >= 30) & (df["oos_n"] >= 5)]
    L.append(f"  {len(qual)} configs qualify  (out of {len(df)} total)")
    if not qual.empty:
        top = qual.sort_values("all_total", ascending=False).head(15)
        cols = ["variant","conf_N","conf_D","is_n","is_wr","is_pf","is_total","is_mdd",
                "oos_n","oos_wr","oos_pf","oos_total","oos_mdd",
                "all_n","all_wr","all_pf","all_total","min_pf","min_wr","worst_mdd"]
        L.append(top[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    L.append("")
    L.append("=" * 220)
    L.append("TOP 15 BY min(IS_PF, OOS_PF)  (relaxed:  both totals positive)")
    L.append("=" * 220)
    qual2 = df[(df["is_total"] > 0) & (df["oos_total"] > 0)
                & (df["is_n"] >= 30) & (df["oos_n"] >= 5)]
    L.append(f"  {len(qual2)} configs qualify")
    if not qual2.empty:
        top = qual2.sort_values("min_pf", ascending=False).head(15)
        cols = ["variant","conf_N","conf_D","is_n","is_wr","is_pf","is_total",
                "oos_n","oos_wr","oos_pf","oos_total",
                "all_n","all_wr","all_pf","all_total","min_pf","worst_mdd"]
        L.append(top[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    L.append("")
    L.append("=" * 220)
    L.append("BY-VARIANT BEST (largest combined total per variant)")
    L.append("=" * 220)
    for v in df["variant"].unique():
        sub = df[df["variant"] == v].sort_values("all_total", ascending=False).head(3)
        L.append(f"\n  {v}:")
        cols = ["conf_N","conf_D","is_n","is_wr","is_pf","is_total","is_mdd",
                "oos_n","oos_wr","oos_pf","oos_total","oos_mdd","all_total"]
        L.append(sub[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    L.append("")
    L.append("=" * 220)
    L.append("AT conf_N=5, conf_D=0  (loosest natural strategy)")
    L.append("=" * 220)
    sub = df[(df["conf_N"] == 5) & (df["conf_D"] == 0)].sort_values("all_total", ascending=False)
    cols = ["variant","is_n","is_wr","is_pf","is_total","is_mdd",
            "oos_n","oos_wr","oos_pf","oos_total","oos_mdd","all_total","all_wr","all_pf"]
    L.append(sub[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
