"""
Test 1 — Parameter Stability (Neighborhood Sensitivity).

For each strategy and each key parameter, run a 1D sweep of ±2 steps around the
locked value. Pass criterion: all 5 neighboring values profitable (PnL > 0, PF > 1).
Strong pass: every neighbor within 25% of locked PnL.
"""
from __future__ import annotations
import sys
import time
import pickle
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# ROUGH VOL — sweep NORM, HZ, SL_ATR, TP_ATR
# ===========================================================================
RV_DIR = Path("C:/trading/nqorderflowbacktester/scripts/rough vol orderflow")
sys.path.insert(0, str(RV_DIR))
import core as rv_core

RV_LOCKED = dict(norm=400, zlook=75, ema_len=80, hz=2.00, sl_atr=2.0, tp_atr=2.0)

RV_SWEEPS = {
    "norm":   [300, 350, 400, 450, 500],
    "hz":     [1.80, 1.90, 2.00, 2.10, 2.20],
    "sl_atr": [1.50, 1.75, 2.00, 2.25, 2.50],
    "tp_atr": [1.50, 1.75, 2.00, 2.25, 2.50],
}

ENTRY_START_MIN = 9 * 60
SKIP_START_MIN  = 13 * 60
SKIP_END_MIN    = 14 * 60
SESSION_END_MIN = 14 * 60 + 45


def rv_run(cfg):
    cache = RV_DIR / ".cache"
    with open(cache / "bars_20m.pkl", "rb") as f:
        b = pickle.load(f)
    with open(cache / "orderflow_20m.pkl", "rb") as f:
        of = pickle.load(f)
    z_vol = rv_core.compute_zvol(b["closes"], cfg["norm"], cfg["zlook"])
    ema = rv_core.compute_ema(b["closes"], cfg["ema_len"])
    atr = b["atr"]
    lmask = of["window_long"][(8, 150)]
    smask = of["window_short"][(8, 150)]

    highs = b["highs"]; lows = b["lows"]; closes = b["closes"]
    mod = b["minutes_of_day"]; di = b["day_idx"]
    n = len(closes)
    pos = 0; ep = sl_p = tp_p = 0.0
    pnls = []
    for i in range(n):
        m = mod[i]
        in_session_full = ENTRY_START_MIN <= m < SESSION_END_MIN
        if pos != 0 and m >= SESSION_END_MIN:
            pnls.append((closes[i] - ep) * pos)
            pos = 0
        if not in_session_full:
            continue
        if pos != 0:
            exited = False; xp = 0.0
            if pos == 1:
                if lows[i] <= sl_p: xp = sl_p; exited = True
                elif highs[i] >= tp_p: xp = tp_p; exited = True
            else:
                if highs[i] >= sl_p: xp = sl_p; exited = True
                elif lows[i] <= tp_p: xp = tp_p; exited = True
            if exited:
                pnls.append((xp - ep) * pos); pos = 0; continue
        in_skip = SKIP_START_MIN <= m < SKIP_END_MIN
        if pos == 0 and not in_skip:
            atr_v = atr[i]
            if atr_v <= 0: continue
            z = z_vol[i]; cl = closes[i]; em = ema[i]
            if z > cfg["hz"]:
                new_dir = 0
                if cl > em: new_dir = 1
                elif cl < em: new_dir = -1
                if new_dir != 0:
                    if new_dir == 1 and lmask[i] == 0: continue
                    if new_dir == -1 and smask[i] == 0: continue
                    pos = new_dir; ep = cl
                    if new_dir == 1:
                        sl_p = cl - cfg["sl_atr"] * atr_v; tp_p = cl + cfg["tp_atr"] * atr_v
                    else:
                        sl_p = cl + cfg["sl_atr"] * atr_v; tp_p = cl - cfg["tp_atr"] * atr_v
    p = np.array(pnls) * rv_core.POINT_VALUE
    return _stats(p)


# ===========================================================================
# OVERNIGHT DRIFT — sweep yellow_atr_mult, green_base, green_decay, green_atr_mult
# ===========================================================================
OD_DIR = Path("C:/trading/nqorderflowbacktester/scripts/overnight drift strategy")
sys.path.insert(0, str(OD_DIR))

OD_LOCKED = dict(
    yellow_atr_len=14, yellow_atr_mult=1.30, yellow_drift=0.0,
    yellow_mode="pure_ratchet",
    green_atr_len=14, green_atr_mult=1.00, green_base=82.5, green_decay=1.5,
    red_intercept=0.0, red_drift=0.45,
    use_be=False, use_martingale=True, base_qty=1, loss_qty=2,
    tp_intrabar_fill=False,
)

OD_SWEEPS = {
    "yellow_atr_mult": [1.10, 1.20, 1.30, 1.40, 1.50],
    "green_base":      [65.0, 75.0, 82.5, 90.0, 100.0],
    "green_decay":     [1.00, 1.25, 1.50, 1.75, 2.00],
    "green_atr_mult":  [0.50, 0.75, 1.00, 1.25, 1.50],
}

_OD_BARS = None


def _od_init():
    global _OD_BARS
    import overnight_drift_strategy as ods
    PARQUET = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
    PICKLES = "D:/trading_pythonbacktest_data/timebars_5min"
    _OD_BARS = ods.build_full_20min_series(PARQUET, PICKLES)


def od_run(cfg):
    import overnight_drift_strategy as ods
    params = ods.StrategyParams(**cfg)
    trades = ods.run_backtest(_OD_BARS, params)
    pnls = np.array([t.pnl_dollars for t in trades])
    return _stats(pnls)


# ===========================================================================
# B2 OHI/OLO — sweep YMULT, TPMULT, MFE_K, MFE_LOCK
# ===========================================================================
B2_DIR = Path("C:/trading/nqorderflowbacktester/scripts/overnight range strat/scripts")
sys.path.insert(0, str(B2_DIR))

B2_LOCKED = dict(YMULT=2.50, TPMULT=2.00, MFE_K=0.8, MFE_LOCK=0.45)

B2_SWEEPS = {
    "YMULT":    [2.10, 2.30, 2.50, 2.70, 2.90],
    "TPMULT":   [1.50, 1.75, 2.00, 2.25, 2.50],
    "MFE_K":    [0.60, 0.70, 0.80, 0.90, 1.00],
    "MFE_LOCK": [0.30, 0.40, 0.45, 0.55, 0.65],
}

_B2_BARS = None
_B2_CANDS = None


def _b2_init():
    global _B2_BARS, _B2_CANDS
    from test_pure_ratchet_exits import build_20min_bars
    from sweep_ratchet_sl_fixed_tp import filter_pre_dedupe
    import lock_v2_k08_lock045_mart_fc_filtered as L
    _B2_BARS = build_20min_bars()
    pdir = B2_DIR / "parquets"
    is_df = pd.read_parquet(pdir / "entry_signal_trades.parquet")
    oos_df = pd.read_parquet(pdir / "entry_signal_trades_oos.parquet")
    is_c = filter_pre_dedupe(is_df)
    oos_c = filter_pre_dedupe(oos_df)
    is_c = L.attach_gamma_to_candidates(is_c)
    oos_c = L.attach_gamma_to_candidates(oos_c)
    is_k, _ = L.filter_candidates(is_c)
    oos_k, _ = L.filter_candidates(oos_c)
    _B2_CANDS = pd.concat([is_k, oos_k], ignore_index=True).sort_values("entry_time_et").reset_index(drop=True)


def b2_run(cfg):
    """Monkey-patch the locked-config module's YMULT/TPMULT/MFE_K/MFE_LOCK constants."""
    import lock_v2_k08_lock045_mart_fc_filtered as L
    # Save originals
    orig = (L.YMULT, L.TPMULT, L.MFE_K, L.MFE_LOCK)
    try:
        L.YMULT = cfg["YMULT"]; L.TPMULT = cfg["TPMULT"]
        L.MFE_K = cfg["MFE_K"]; L.MFE_LOCK = cfg["MFE_LOCK"]
        out = L.run(_B2_CANDS, _B2_BARS, period_label="ALL")
    finally:
        L.YMULT, L.TPMULT, L.MFE_K, L.MFE_LOCK = orig
    if len(out) == 0:
        return dict(trades=0, pnl=0, pf=0, mdd=0, wr=0)
    pnls = out["pnl"].values
    return _stats(pnls)


def _stats(pnls):
    n = len(pnls)
    if n == 0: return dict(trades=0, pnl=0.0, pf=0.0, mdd=0.0, wr=0.0)
    w = pnls[pnls > 0]; l = pnls[pnls < 0]
    pf = float(w.sum() / abs(l.sum())) if len(l) else 99.0
    cum = np.cumsum(pnls)
    mdd = float((cum - np.maximum.accumulate(cum)).min())
    return dict(trades=n, pnl=float(pnls.sum()), pf=pf, mdd=mdd,
                wr=100.0 * len(w) / n)


# ===========================================================================
# Sweep runners
# ===========================================================================
def sweep_strategy(name, locked, sweeps, runner, init_fn=None):
    print(f"\n>>> {name} parameter stability")
    if init_fn:
        init_fn()
    # First, run the locked baseline
    base = runner(locked)
    print(f"  LOCKED: trades={base['trades']}  PnL=${base['pnl']:+,.0f}  PF={base['pf']:.3f}  MDD=${base['mdd']:+,.0f}")

    sweep_results = {}
    for param, values in sweeps.items():
        print(f"  -- sweeping {param}: {values}")
        rows = []
        for v in values:
            cfg = dict(locked); cfg[param] = v
            t0 = time.time()
            m = runner(cfg)
            dt = time.time() - t0
            tag = "  *LOCKED*" if v == locked[param] else ""
            print(f"    {param}={v}  trades={m['trades']:>5}  PnL=${m['pnl']:>+10,.0f}  PF={m['pf']:.3f}  MDD=${m['mdd']:>+10,.0f}{tag} ({dt:.1f}s)")
            rows.append(dict(param=param, value=v, **m))
        sweep_results[param] = rows
    return base, sweep_results


def analyze(name, base, sweep_results):
    lines = []
    lines.append(f"\n### {name}")
    lines.append("=" * 80)
    lines.append(f"Locked: PnL=${base['pnl']:+,.0f}  PF={base['pf']:.3f}  MDD=${base['mdd']:+,.0f}  trades={base['trades']}")

    overall_pass = True
    for param, rows in sweep_results.items():
        # all 5 neighbors profitable & PF > 1?
        all_profitable = all(r["pnl"] > 0 for r in rows)
        all_pf_above_1 = all(r["pf"] > 1.0 for r in rows)
        verdict = "PASS" if all_profitable and all_pf_above_1 else "FAIL"
        if not (all_profitable and all_pf_above_1):
            overall_pass = False
        # variation (max/min PnL relative to locked)
        pnls = [r["pnl"] for r in rows]
        min_rel = min(pnls) / base["pnl"] if base["pnl"] > 0 else 0
        max_rel = max(pnls) / base["pnl"] if base["pnl"] > 0 else 0
        lines.append(f"  {param:<18}  all_profitable={all_profitable}  all_PF>1={all_pf_above_1}  "
                     f"min/max PnL rel locked: {min_rel:.2f}x..{max_rel:.2f}x  [{verdict}]")
    lines.append(f"  OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    return "\n".join(lines)


def main():
    summary = ["=" * 80, "TEST 1 — PARAMETER STABILITY", "=" * 80]

    # Rough Vol (no init needed — uses cache files lazily)
    rv_base, rv_res = sweep_strategy("Rough Vol", RV_LOCKED, RV_SWEEPS, rv_run)
    summary.append(analyze("Rough Vol", rv_base, rv_res))

    # Overnight Drift
    od_base, od_res = sweep_strategy("Overnight Drift", OD_LOCKED, OD_SWEEPS, od_run, _od_init)
    summary.append(analyze("Overnight Drift", od_base, od_res))

    # B2
    b2_base, b2_res = sweep_strategy("OHI/OLO B2", B2_LOCKED, B2_SWEEPS, b2_run, _b2_init)
    summary.append(analyze("OHI/OLO B2", b2_base, b2_res))

    text = "\n".join(summary)
    print("\n" + text)
    with open(RESULTS / "test1_summary.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nSaved to {RESULTS / 'test1_summary.txt'}")


if __name__ == "__main__":
    main()
