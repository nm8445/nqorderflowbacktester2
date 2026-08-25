"""Walk-forward validation of the B2 exit parameters.

Answers the fair objection to sweep_live_b2_exit_params.py: that sweep reported
IS/OOS but the winning cell was PICKED while looking at the whole period, so its
OOS number is no longer out-of-sample.

Design follows the house convention in `scripts/overnight drift strategy/walk_forward.py`:
  - Run every config once over the FULL history (chained dedupe intact).
  - Per fold, slice trades into train / test windows by entry date.
  - Rank configs by TRAIN metric only (PF, with a min-trade floor), pick the top,
    record that config's TEST metrics.
  - Stitch all test windows end to end => honest out-of-sample equity.

Two fold schemes (both 24-month rolling train):
  A  24mo train / 12mo test / 12mo step   (4 folds — matches the OD script)
  B  24mo train /  6mo test /  6mo step   (7 folds — more folds, shorter tests)
Plus an ANCHORED (expanding-window) variant of scheme B.

Baselines over the identical stitched test span:
  LIVE   y2.50 tp2.00 mfe0.8/0.45   — the config actually running
  FIXED  y2.50 tp4.00 mfe0.6/0.45   — the full-sample pick (i.e. fitted on
                                       everything, shown to quantify the fit bonus)

Also dumps, per train window, which TPMULT wins and where TPMULT=2.0 ranks — the
real test of whether "the live TP is too tight" is a stable property of the
strategy or an artifact of one period.

Output -> tradelogs/naked_break/walk_forward_b2_exit.txt
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import sweep_live_b2_exit_params as S              # noqa: E402
import lock_v2_k08_lock045_mart_fc_filtered as LOCK  # noqa: E402
from sweep_ratchet_sl_fixed_tp import filter_pre_dedupe  # noqa: E402

OUT = HERE.parent / "tradelogs" / "naked_break" / "walk_forward_b2_exit.txt"

LIVE = (2.50, 2.00, 0.80, 0.45)
FIXED = (2.50, 4.00, 0.60, 0.45)

YMULTS = [2.0, 2.5, 3.0]
TPMULTS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
MFE_KS = [0.4, 0.6, 0.8, 1.0]
MFE_LOCK = 0.45

MIN_TRAIN_TRADES = 60


def month(y, m):
    return pd.Timestamp(year=y, month=m, day=1).date()


def add_months(d, n):
    return (pd.Timestamp(d) + pd.DateOffset(months=n)).date()


def make_folds(test_months: int, step_months: int, train_months: int,
               anchored: bool, first_test=(2022, 12), last=(2026, 6, 20)):
    """[(train_start, train_end, test_start, test_end), ...] — half-open ends."""
    folds = []
    ts = month(*first_test)
    hard_end = pd.Timestamp(*last).date()
    while ts < hard_end:
        te = min(add_months(ts, test_months), hard_end)
        tr_end = ts
        tr_start = month(2020, 12) if anchored else add_months(ts, -train_months)
        if tr_start < month(2020, 12):
            tr_start = month(2020, 12)
        if te <= ts:
            break
        folds.append((tr_start, tr_end, ts, te))
        ts = add_months(ts, step_months)
    return folds


def slice_stats(df, start, end):
    sub = df[(df["date"] >= start) & (df["date"] < end)]
    return sub, S.stats(sub)


def run_scheme(all_trades: dict, cells: list, folds: list, criterion: str):
    """Pick per fold on TRAIN `criterion`, return (per-fold rows, stitched test df)."""
    rows, stitched = [], []
    for tr_s, tr_e, te_s, te_e in folds:
        best, best_key = None, None
        for cell in cells:
            _, tr = slice_stats(all_trades[cell], tr_s, tr_e)
            if tr["n"] < MIN_TRAIN_TRADES:
                continue
            key = tr[criterion]
            if best_key is None or key > best_key:
                best_key, best = key, cell
        if best is None:
            continue
        te_df, te = slice_stats(all_trades[best], te_s, te_e)
        _, lv = slice_stats(all_trades[LIVE], te_s, te_e)
        _, fx = slice_stats(all_trades[FIXED], te_s, te_e)
        stitched.append(te_df)
        rows.append(dict(train=f"{tr_s}..{tr_e}", test=f"{te_s}..{te_e}",
                         pick=f"y{best[0]:g} tp{best[1]:g} mfe{best[2]:g}",
                         picked_tp=best[1], picked_y=best[0], picked_mk=best[2],
                         te_n=te["n"], te_net=te["net"], te_pf=te["pf"], te_wr=te["wr"],
                         live_n=lv["n"], live_net=lv["net"], live_pf=lv["pf"],
                         fixed_net=fx["net"]))
    st = pd.concat(stitched, ignore_index=True) if stitched else pd.DataFrame()
    return rows, st


def main():
    t0 = _time.time()
    bars = S.build_20min_bars()
    parts = []
    for f in ("entry_signal_trades.parquet", "entry_signal_trades_oos.parquet"):
        c = filter_pre_dedupe(pd.read_parquet(S.PARQUETS / f))
        c = LOCK.attach_gamma_to_candidates(c)
        c, _ = LOCK.filter_candidates(c)
        parts.append(c)
    cands = pd.concat(parts, ignore_index=True).sort_values("entry_time_et").reset_index(drop=True)

    cells = [(y, t, mk, MFE_LOCK) for y in YMULTS for t in TPMULTS for mk in MFE_KS]
    print(f"{len(cands)} candidates | {len(cells)} configs over full history...")
    all_trades = {}
    for i, (y, t, mk, ml) in enumerate(cells, 1):
        all_trades[(y, t, mk, ml)] = S.run(cands, bars, y, t, mk, ml)
        if i % 24 == 0:
            print(f"  {i}/{len(cells)}  ({_time.time()-t0:.0f}s)")

    L = ["=" * 165,
         "B2 EXIT PARAMS — WALK-FORWARD VALIDATION",
         "=" * 165, "",
         f"  {len(cells)} configs (YMULT x TPMULT x MFE_K, MFE_LOCK fixed {MFE_LOCK})",
         f"  train selection metric: PF (min {MIN_TRAIN_TRADES} train trades); "
         f"secondary runs use net and Sharpe",
         "  LIVE  = y2.5 tp2.0 mfe0.8/0.45     FIXED = y2.5 tp4.0 mfe0.6/0.45 "
         "(full-sample pick — carries fit bias, shown for reference)", ""]

    schemes = [
        ("A  24mo train / 12mo test / 12mo step (rolling)", make_folds(12, 12, 24, False)),
        ("B  24mo train /  6mo test /  6mo step (rolling)", make_folds(6, 6, 24, False)),
        ("C  anchored train / 6mo test / 6mo step", make_folds(6, 6, 24, True)),
    ]
    summary = []
    for name, folds in schemes:
        L += ["=" * 165, name, "=" * 165, ""]
        for crit in ("pf", "net", "sharpe"):
            rows, st = run_scheme(all_trades, cells, folds, crit)
            if not rows:
                continue
            s = S.stats(st)
            lv_all = pd.concat([slice_stats(all_trades[LIVE], a, b)[0]
                                for *_, a, b in folds], ignore_index=True)
            fx_all = pd.concat([slice_stats(all_trades[FIXED], a, b)[0]
                                for *_, a, b in folds], ignore_index=True)
            ls, fs = S.stats(lv_all), S.stats(fx_all)
            L.append(f"  --- train-selection metric: {crit.upper()} ---")
            L.append(f"  {'train window':<24} {'test window':<24} {'picked':<22} "
                     f"{'te_n':>5} {'te_net':>8} {'te_pf':>6} {'live_net':>9} "
                     f"{'live_pf':>7} {'WF-live':>8} {'fixed_net':>9}")
            for r in rows:
                L.append(f"  {r['train']:<24} {r['test']:<24} {r['pick']:<22} "
                         f"{r['te_n']:>5} {r['te_net']:>+8.1f} {r['te_pf']:>6.2f} "
                         f"{r['live_net']:>+9.1f} {r['live_pf']:>7.2f} "
                         f"{r['te_net']-r['live_net']:>+8.1f} {r['fixed_net']:>+9.1f}")
            wins = sum(1 for r in rows if r["te_net"] > r["live_net"])
            L.append(f"  STITCHED OOS   walk-forward  n={s['n']:<4} net={s['net']:+8.1f} "
                     f"(${s['net']*2:+,.0f} MNQ) pf={s['pf']:.3f} wr={s['wr']:.1f}% "
                     f"sharpe={s['sharpe']:.2f} mdd={s['mdd']:+.1f}")
            L.append(f"                 LIVE config   n={ls['n']:<4} net={ls['net']:+8.1f} "
                     f"(${ls['net']*2:+,.0f} MNQ) pf={ls['pf']:.3f} wr={ls['wr']:.1f}% "
                     f"sharpe={ls['sharpe']:.2f} mdd={ls['mdd']:+.1f}")
            L.append(f"                 FIXED (fitted) n={fs['n']:<3} net={fs['net']:+8.1f} "
                     f"(${fs['net']*2:+,.0f} MNQ) pf={fs['pf']:.3f} wr={fs['wr']:.1f}% "
                     f"sharpe={fs['sharpe']:.2f} mdd={fs['mdd']:+.1f}")
            L.append(f"                 folds where WF beat LIVE: {wins}/{len(rows)}   "
                     f"WF captures {s['net']/fs['net']*100:.0f}% of the fitted config's net")
            L.append(f"                 picked TPMULT per fold: "
                     f"{[r['picked_tp'] for r in rows]}")
            L.append("")
            summary.append(dict(scheme=name.split()[0], crit=crit, wf=s["net"],
                                live=ls["net"], fixed=fs["net"], wins=wins,
                                folds=len(rows), wf_sharpe=s["sharpe"],
                                live_sharpe=ls["sharpe"]))

    # --- is "TP too tight" stable across train windows, independent of selection? ---
    L += ["=" * 165,
          "TRAIN-WINDOW STABILITY — for each 24mo train window, the best TPMULT "
          "(YMULT/MFE_K free) and where the live TPMULT=2.0 ranks",
          "=" * 165, "",
          f"  {'train window':<24} {'best cell':<24} {'best_net':>9} "
          f"{'best tp by PF':>14} {'live tp2.0 best_net':>20} {'rank of tp2.0 cells':>21}"]
    for tr_s, tr_e, _, _ in make_folds(6, 6, 24, False):
        scored = []
        for cell in cells:
            _, tr = slice_stats(all_trades[cell], tr_s, tr_e)
            if tr["n"] >= MIN_TRAIN_TRADES:
                scored.append((tr["net"], tr["pf"], cell))
        if not scored:
            continue
        scored.sort(key=lambda x: -x[0])
        best = scored[0]
        by_pf = max(scored, key=lambda x: x[1])
        tp2 = [s_ for s_ in scored if s_[2][1] == 2.0]
        best_tp2 = max(tp2, key=lambda x: x[0]) if tp2 else None
        rank_tp2 = min(i for i, s_ in enumerate(scored, 1) if s_[2][1] == 2.0) if tp2 else -1
        L.append(f"  {str(tr_s)+'..'+str(tr_e):<24} "
                 f"{f'y{best[2][0]:g} tp{best[2][1]:g} mfe{best[2][2]:g}':<24} "
                 f"{best[0]:>+9.1f} {f'tp{by_pf[2][1]:g}':>14} "
                 f"{best_tp2[0] if best_tp2 else 0:>+20.1f} "
                 f"{f'{rank_tp2}/{len(scored)}':>21}")
    L.append("")

    L += ["=" * 165, "VERDICT", "=" * 165, ""]
    for r in summary:
        v = ("WF > LIVE" if r["wf"] > r["live"] else "WF <= LIVE")
        L.append(f"  scheme {r['scheme']} / select-by-{r['crit']:<7} "
                 f"WF {r['wf']:+8.1f} vs LIVE {r['live']:+8.1f} "
                 f"(Sharpe {r['wf_sharpe']:.2f} vs {r['live_sharpe']:.2f})  "
                 f"folds won {r['wins']}/{r['folds']}   -> {v}")
    L.append("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {OUT}\ntotal {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
