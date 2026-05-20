"""
Test 5 — Direction Permutation (the strongest overfit test).

For each strategy:
  1. Identify all entry-signal bars (using the strategy's gating logic, but stripping
     the direction decision).
  2. For N permutations, assign random long/short to each signal.
  3. Re-run the strategy's exit logic with the permuted directions.
  4. Compare real PnL to the permutation distribution.

Real PnL should sit in the top 1% of the distribution. Median permutation PnL ~ $0
if the strategy has no inherent long-bias from being in the market.

Uses multiprocessing.Pool for parallel permutations.
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

N_PERMS = 1000

# ====================================================================
# ROUGH VOL ORDERFLOW
# ====================================================================
RV_DIR = Path("C:/trading/nqorderflowbacktester/scripts/rough vol orderflow")
sys.path.insert(0, str(RV_DIR))
import core as rv_core

RV_NORM, RV_ZLOOK, RV_EMA_LEN, RV_HZ = 400, 75, 80, 2.00
RV_SL_ATR, RV_TP_ATR = 2.0, 2.0
ENTRY_START_MIN = 9 * 60
SKIP_START_MIN  = 13 * 60
SKIP_END_MIN    = 14 * 60
SESSION_END_MIN = 14 * 60 + 45


def rv_extract_signals():
    """Run the strategy WITHOUT the EMA direction filter, collect signal bar indices."""
    cache = RV_DIR / ".cache"
    with open(cache / "bars_20m.pkl", "rb") as f:
        b = pickle.load(f)
    with open(cache / "orderflow_20m.pkl", "rb") as f:
        of = pickle.load(f)
    z_vol = rv_core.compute_zvol(b["closes"], RV_NORM, RV_ZLOOK)
    ema = rv_core.compute_ema(b["closes"], RV_EMA_LEN)
    atr = b["atr"]
    lmask = of["window_long"][(8, 150)]
    smask = of["window_short"][(8, 150)]

    highs = b["highs"]; lows = b["lows"]; closes = b["closes"]
    mod = b["minutes_of_day"]; di = b["day_idx"]
    n = len(closes)

    # Track open position (to skip new signals while in trade — same as real strategy)
    pos = 0; ep = sl_p = tp_p = 0.0; entry_dir = 0
    signals = []   # list of (signal_idx, allowed_directions)

    for i in range(n):
        m = mod[i]
        in_session_full = ENTRY_START_MIN <= m < SESSION_END_MIN
        if pos != 0 and m >= SESSION_END_MIN:
            pos = 0
        if not in_session_full:
            continue
        if pos != 0:
            if pos == 1:
                if lows[i] <= sl_p or highs[i] >= tp_p:
                    pos = 0
            else:
                if highs[i] >= sl_p or lows[i] <= tp_p:
                    pos = 0
            if pos != 0:
                continue
        in_skip = SKIP_START_MIN <= m < SKIP_END_MIN
        if in_skip:
            continue
        atr_v = atr[i]
        if atr_v <= 0:
            continue
        if z_vol[i] > RV_HZ:
            # Both long and short directions are eligible iff orderflow mask passes for that side
            allow_long = lmask[i] == 1
            allow_short = smask[i] == 1
            if not allow_long and not allow_short:
                continue
            signals.append((i, allow_long, allow_short))
            # For session-tracking, simulate that we DO take this signal (use real direction
            # via EMA to track when next entry can fire). This mirrors real backtest pacing.
            cl = closes[i]; em = ema[i]
            if cl > em and allow_long: new_dir = 1
            elif cl < em and allow_short: new_dir = -1
            else: new_dir = 0
            if new_dir != 0:
                pos = new_dir
                ep = cl
                if new_dir == 1:
                    sl_p = cl - RV_SL_ATR * atr_v; tp_p = cl + RV_TP_ATR * atr_v
                else:
                    sl_p = cl + RV_SL_ATR * atr_v; tp_p = cl - RV_TP_ATR * atr_v

    return signals, b


def rv_simulate_with_directions(signals, b, directions):
    """Replay strategy with given directions (1=long, -1=short, 0=skip).
       Returns total PnL in $."""
    highs = b["highs"]; lows = b["lows"]; closes = b["closes"]
    mod = b["minutes_of_day"]
    atr = b["atr"]
    n = len(closes)

    # Build a quick lookup: signal_idx -> direction
    sig_map = {sig[0]: directions[k] for k, sig in enumerate(signals)}

    pos = 0; ep = sl_p = tp_p = 0.0
    pnls = []
    cur_sig_idx = -1
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
                pnls.append((xp - ep) * pos)
                pos = 0
                continue
        in_skip = SKIP_START_MIN <= m < SKIP_END_MIN
        if pos == 0 and not in_skip and i in sig_map:
            d = sig_map[i]
            if d == 0:
                continue
            atr_v = atr[i]
            if atr_v <= 0:
                continue
            cl = closes[i]
            pos = d
            ep = cl
            if d == 1:
                sl_p = cl - RV_SL_ATR * atr_v; tp_p = cl + RV_TP_ATR * atr_v
            else:
                sl_p = cl + RV_SL_ATR * atr_v; tp_p = cl - RV_TP_ATR * atr_v
    return sum(pnls) * rv_core.POINT_VALUE


_RV_SIGNALS = None
_RV_BARS = None


def _rv_worker_init():
    global _RV_SIGNALS, _RV_BARS
    _RV_SIGNALS, _RV_BARS = rv_extract_signals()


def _rv_perm_run(seed):
    rng = np.random.default_rng(seed)
    n = len(_RV_SIGNALS)
    dirs = np.zeros(n, dtype=np.int8)
    for k, (idx, allow_long, allow_short) in enumerate(_RV_SIGNALS):
        # Coin flip among allowed directions
        if allow_long and allow_short:
            dirs[k] = 1 if rng.random() < 0.5 else -1
        elif allow_long:
            dirs[k] = 1 if rng.random() < 0.5 else 0  # 50% chance "skip" if only one side allowed
        elif allow_short:
            dirs[k] = -1 if rng.random() < 0.5 else 0
    return rv_simulate_with_directions(_RV_SIGNALS, _RV_BARS, dirs)


def rv_run_permutation_test(n_perms=N_PERMS, n_workers=6):
    print(f"\n>>> Rough Vol direction permutation ({n_perms} perms)")
    t0 = time.time()
    # Real (with EMA filter)
    signals, b = rv_extract_signals()
    real_dirs = np.zeros(len(signals), dtype=np.int8)
    closes = b["closes"]
    ema = rv_core.compute_ema(b["closes"], RV_EMA_LEN)
    for k, (idx, al, ash_) in enumerate(signals):
        cl = closes[idx]; em = ema[idx]
        if cl > em and al: real_dirs[k] = 1
        elif cl < em and ash_: real_dirs[k] = -1
        else: real_dirs[k] = 0
    real_pnl = rv_simulate_with_directions(signals, b, real_dirs)
    print(f"  n_signals={len(signals)}  real_pnl=${real_pnl:+,.0f}  (took {time.time()-t0:.1f}s)")

    # Permutations
    t1 = time.time()
    seeds = list(range(n_perms))
    with mp.Pool(n_workers, initializer=_rv_worker_init) as pool:
        pnls = list(pool.imap_unordered(_rv_perm_run, seeds, chunksize=10))
    pnls = np.array(pnls)
    print(f"  perms done in {time.time()-t1:.0f}s")

    return dict(name="Rough Vol", real_pnl=real_pnl, perm_pnls=pnls, n_signals=len(signals))


# ====================================================================
# OVERNIGHT DRIFT — long-only at 19:00 ET, flip to short with 50% probability
# ====================================================================
OD_DIR = Path("C:/trading/nqorderflowbacktester/scripts/overnight drift strategy")
sys.path.insert(0, str(OD_DIR))


def od_run_one_perm(args):
    """Build OD bars+atr once at module level via initializer; here just run with random dirs."""
    seed, params_dict = args
    rng = np.random.default_rng(seed)
    return _od_simulate_with_random_directions(rng, params_dict)


_OD_BARS = None
_OD_ATR_Y = None
_OD_ATR_G = None


def _od_worker_init():
    global _OD_BARS, _OD_ATR_Y, _OD_ATR_G
    import overnight_drift_strategy as ods
    PARQUET = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
    PICKLES = "D:/trading_pythonbacktest_data/timebars_5min"
    _OD_BARS = ods.build_full_20min_series(PARQUET, PICKLES)
    _OD_ATR_Y = ods.rma_atr(_OD_BARS["high"], _OD_BARS["low"], _OD_BARS["close"], 14)
    _OD_ATR_G = ods.rma_atr(_OD_BARS["high"], _OD_BARS["low"], _OD_BARS["close"], 14)


def _od_simulate_with_random_directions(rng, params_dict):
    """Long entries at 19:00 ET; flip each entry to short with 50% prob.
       For SHORT: yellow above entry (ratchets DOWN), green below entry."""
    from datetime import time as dtime
    bars = _OD_BARS
    atr_y = _OD_ATR_Y; atr_g = _OD_ATR_G
    entry_t = dtime(19, 0)
    force_t = dtime(8, 0)
    YM = params_dict["YM"]; GM = params_dict["GM"]
    GB = params_dict["GB"]; GD = params_dict["GD"]
    RI = params_dict["RI"]; RD = params_dict["RD"]
    BASE_QTY = 1

    idx = bars.index
    o = bars["open"].values; h = bars["high"].values
    l = bars["low"].values; c = bars["close"].values
    n = len(bars)

    pos = 0  # 1 = long, -1 = short
    entry_price = 0.0; entry_idx = 0; entry_qty = BASE_QTY
    prev_yellow = np.nan
    bars_in_trade = 0
    pnls = []

    for i in range(n):
        ts = idx[i]
        local_t = ts.time()
        ay = atr_y.iloc[i]; ag = atr_g.iloc[i]
        if pos != 0:
            bars_in_trade += 1
            sign = pos
            # Yellow ratchet (sign-aware)
            raw_yellow = c[i] - sign * YM * ay if not np.isnan(ay) else np.nan
            if np.isnan(prev_yellow):
                yellow_val = raw_yellow
            else:
                yellow_val = (max(prev_yellow, raw_yellow) if sign > 0
                              else min(prev_yellow, raw_yellow))
            red_val = entry_price + sign * (RI + RD * bars_in_trade)
            green_val = red_val + sign * (GB - GD * bars_in_trade) + sign * (GM * ag if not np.isnan(ag) else 0.0)

            exit_price = np.nan; exited = False
            # TP green: long uses high; short uses low
            if not np.isnan(green_val):
                if sign == 1 and h[i] >= green_val:
                    exit_price = c[i]; exited = True
                elif sign == -1 and l[i] <= green_val:
                    exit_price = c[i]; exited = True
            # SL yellow: bearish close at/below yellow (long); bullish close at/above (short)
            if not exited and not np.isnan(yellow_val):
                if sign == 1 and c[i] <= yellow_val and c[i] < o[i]:
                    exit_price = c[i]; exited = True
                elif sign == -1 and c[i] >= yellow_val and c[i] > o[i]:
                    exit_price = c[i]; exited = True
            # Force close
            if not exited and local_t == force_t:
                exit_price = c[i]; exited = True
            if exited:
                pnl = (exit_price - entry_price) * sign * entry_qty * 20.0  # NQ $20/pt
                pnls.append(pnl)
                pos = 0; prev_yellow = np.nan; bars_in_trade = 0
                continue
            prev_yellow = yellow_val
        # Entry
        if pos == 0 and local_t == entry_t:
            pos = 1 if rng.random() < 0.5 else -1
            entry_price = c[i]; entry_idx = i; entry_qty = BASE_QTY
            prev_yellow = np.nan; bars_in_trade = 0
    return float(sum(pnls))


def od_run_permutation_test(n_perms=N_PERMS, n_workers=6):
    print(f"\n>>> Overnight Drift direction permutation ({n_perms} perms)")
    params_dict = dict(YM=1.30, GM=1.00, GB=82.5, GD=1.5, RI=0.0, RD=0.45)
    seeds = list(range(n_perms))
    args = [(s, params_dict) for s in seeds]

    # Real run = all long
    t0 = time.time()
    _od_worker_init()  # init in main proc to compute real
    real_rng = np.random.default_rng(0)
    # For "real" we want all long (deterministic)
    real_pnl = _od_simulate_with_random_directions(
        np.random.default_rng(123_456),  # dummy
        params_dict
    )
    # Actually we want deterministic all-long. Override with biased RNG:
    class AllLong:
        def random(self): return 0.0  # always < 0.5, so always long
    real_pnl = _od_simulate_with_random_directions(AllLong(), params_dict)
    print(f"  real_pnl (all long, locked params)=${real_pnl:+,.0f}  (took {time.time()-t0:.0f}s)")

    t1 = time.time()
    with mp.Pool(n_workers, initializer=_od_worker_init) as pool:
        pnls = list(pool.imap_unordered(od_run_one_perm, args, chunksize=4))
    pnls = np.array(pnls)
    print(f"  perms done in {time.time()-t1:.0f}s")
    return dict(name="Overnight Drift", real_pnl=real_pnl, perm_pnls=pnls,
                n_signals=None)


# ====================================================================
# OHI/OLO B2 — flip direction in precomputed candidates
# ====================================================================
B2_DIR = Path("C:/trading/nqorderflowbacktester/scripts/overnight range strat/scripts")
sys.path.insert(0, str(B2_DIR))

_B2_BARS = None
_B2_CANDS = None


def _b2_worker_init():
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
    cands = pd.concat([is_k, oos_k], ignore_index=True).sort_values("entry_time_et").reset_index(drop=True)
    _B2_CANDS = cands


def _b2_perm_run(seed):
    import lock_v2_k08_lock045_mart_fc_filtered as L
    rng = np.random.default_rng(seed)
    cands = _B2_CANDS.copy()
    # Random direction per candidate
    flip = rng.random(len(cands)) < 0.5
    new_dir = np.where(flip, "LONG", "SHORT")
    cands["direction"] = new_dir
    out = L.run(cands, _B2_BARS, period_label="PERM")
    if len(out) == 0:
        return 0.0
    return float(out["pnl"].sum())


def b2_run_permutation_test(n_perms=N_PERMS, n_workers=6):
    print(f"\n>>> B2 direction permutation ({n_perms} perms)")
    t0 = time.time()
    # Real = use baseline trade log
    real_df = pd.read_csv("C:/trading/nqorderflowbacktester/scripts/overnight range strat/tradelogs/robust_configs/locked_v2_k08_lock045_mart_fc_filtered_trades.csv")
    real_pnl = float(real_df["pnl"].sum())  # in NQ pts. Convert by *20 for $ NQ; pnl col is raw NQ pts
    print(f"  real_pnl (NQ pts)={real_pnl:+,.2f}  (= ${real_pnl*20:+,.0f} NQ)")

    t1 = time.time()
    seeds = list(range(n_perms))
    with mp.Pool(n_workers, initializer=_b2_worker_init) as pool:
        pnls = list(pool.imap_unordered(_b2_perm_run, seeds, chunksize=4))
    pnls = np.array(pnls)
    print(f"  perms done in {time.time()-t1:.0f}s")
    return dict(name="OHI/OLO B2", real_pnl=real_pnl, perm_pnls=pnls,
                n_signals=len(_B2_CANDS) if _B2_CANDS is not None else None)


def report(res):
    name = res["name"]; real = res["real_pnl"]; perms = res["perm_pnls"]
    median = float(np.median(perms))
    q05, q95 = float(np.quantile(perms, 0.05)), float(np.quantile(perms, 0.95))
    q99 = float(np.quantile(perms, 0.99))
    pct_real_higher = float((perms <= real).mean() * 100)
    p_value = float((perms >= real).mean())

    lines = [
        f"\n--- {name} ---",
        f"  Real PnL:               {real:+,.2f}",
        f"  Permutation median:     {median:+,.2f}",
        f"  Permutation 5/95%:      {q05:+,.2f} ... {q95:+,.2f}",
        f"  Permutation 99%:        {q99:+,.2f}",
        f"  Real beats {pct_real_higher:.2f}% of permutations",
        f"  p-value (real >= perm): {p_value:.4f}",
        f"  Verdict (need p<0.01): {'PASS' if p_value < 0.01 else 'FAIL'}",
    ]
    return "\n".join(lines)


def main():
    mp.freeze_support()
    rv = rv_run_permutation_test(N_PERMS, 6)
    od = od_run_permutation_test(N_PERMS, 6)
    b2 = b2_run_permutation_test(N_PERMS, 6)
    text = "\n".join([report(rv), report(od), report(b2)])
    # Save numeric distributions
    np.savez(RESULTS / "test5_perm_distributions.npz",
             rv_real=rv["real_pnl"], rv_perms=rv["perm_pnls"],
             od_real=od["real_pnl"], od_perms=od["perm_pnls"],
             b2_real=b2["real_pnl"], b2_perms=b2["perm_pnls"])
    with open(RESULTS / "test5_summary.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\nSaved to {RESULTS / 'test5_summary.txt'}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
