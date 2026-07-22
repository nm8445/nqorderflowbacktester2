"""FB yellow-trail TIMEFRAME sweep: enter on 5-min (FB ORB break, unchanged), but drive the giveback
trailing yellow off N-min candles (N in 5/10/15/20/25/30). Re-fit the yellow params separately at each
timeframe and rank by robust min(IS_PF, OOS_PF) -> answers "is 20-min the right yellow timeframe, or
does 5-min (or another) win?".

Design (confirmed with user 2026-07-22):
  - Entry: FB 5-min ORB break (find_entry from run_giveback_variant) -- identical across timeframes.
  - Yellow: the LIVE giveback mechanism (drift_floor/pure_ratchet base, bearish giveback, scale_body,
    max_gb, min_gap) but computed on N-min candles. Updated once per N-min candle close; exit = a
    BEARISH N-min CLOSE <= yellow. Hard floor = ORB_Low (never looser).
  - N-min candles: clock-aligned to 09:00 ET (session-relative buckets, so 25-min tiles 09:00-14:00
    cleanly). ATR(14) is CONTINUOUS across days (no session warmup) -> fair across timeframes.
  - TP (4R) stays on 5-min intrabar high (resting limit fills anytime). EOD 14:00.
  - Params re-fit per timeframe over the run_giveback grid; ranked on robust min(IS,OOS) PF.

Run:  python scripts/fabio_orb/sweep_yellow_timeframe.py
"""
from __future__ import annotations
import warnings; warnings.simplefilter("ignore")
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

# reuse the validated loader / entry / cost model / metrics from the 5-min giveback sweep
from run_giveback_variant import (
    load_days, find_entry, metrics, run_static, _close, CUTOFF, DPP, SLIP_PTS, COMM,
)

OUT_DIR = Path(__file__).parent / "results"; OUT_DIR.mkdir(exist_ok=True)
TIMEFRAMES = [5, 10, 15, 20, 25, 30]
ATR_LEN_N = 14

# yellow param grid (same shape as run_giveback_variant)
KS      = [1.0, 1.5, 2.0, 3.0, 4.0]
MODES   = ["pure_ratchet", "drift_floor"]
DRIFTS  = [0.0, 1.0]
GBS     = [0.0, 0.3, 0.5, 0.7]
SCALES  = [True, False]
MAXGB   = [0.5, 1.0]
MINGAP  = [0.0, 0.3]

# current LIVE 5-min config (fabio_orb_engine.py) -- flagged in the output for reference
LIVE_CFG = dict(k=1.5, mode="drift_floor", drift=0.0, giveback=0.3,
                scale_body=True, max_gb=0.5, min_gap=0.3)

_DAYS: dict = {}
_KEYS: list = []


# ------------------------- N-min candle construction -------------------------

def build_candles(days, keys, N):
    """Per day: N-min candles (clock-aligned to 09:00) + a CONTINUOUS rolling ATR(14) across days.
    Returns per-day dict with comp_idx (post-array index where each candle completes), candle
    o/h/l/c/atr/bearish, and comp_of_index (post-index -> candle position, else -1)."""
    per_day = {}
    all_h, all_l, all_c = [], [], []
    order = []                                   # (day_key, n_candles) to slice ATR back
    for d in keys:
        day = days[d]
        hhmm = day["hhmm"]
        mins = (hhmm // 100) * 60 + (hhmm % 100) - 540      # minutes since 09:00 (close-based)
        bucket = np.floor((mins - 1) / N).astype(int)        # candle index within the day
        o, h, l, c = day["open"], day["high"], day["low"], day["close"]
        comp_idx, co, ch, cl, cc = [], [], [], [], []
        for bb in np.unique(bucket):
            idx = np.where(bucket == bb)[0]
            comp_idx.append(int(idx.max()))
            co.append(o[idx[0]]); ch.append(h[idx].max()); cl.append(l[idx].min()); cc.append(c[idx[-1]])
        comp_idx = np.array(comp_idx)
        co = np.array(co); ch = np.array(ch); cl = np.array(cl); cc = np.array(cc)
        comp_of_index = np.full(len(hhmm), -1, dtype=int)
        comp_of_index[comp_idx] = np.arange(len(comp_idx))
        per_day[d] = dict(comp_idx=comp_idx, co=co, ch=ch, cl=cl, cc=cc,
                          comp_of_index=comp_of_index)
        all_h.append(ch); all_l.append(cl); all_c.append(cc)
        order.append((d, len(cc)))
    # continuous ATR over all candles chronologically
    H = np.concatenate(all_h); L = np.concatenate(all_l); C = np.concatenate(all_c)
    pc = np.empty_like(C); pc[0] = np.nan; pc[1:] = C[:-1]
    tr = np.maximum(H - L, np.maximum(np.abs(H - pc), np.abs(L - pc)))
    atr = pd.Series(tr).rolling(ATR_LEN_N, min_periods=3).mean().to_numpy()
    pos = 0
    for d, n in order:
        per_day[d]["atr"] = atr[pos:pos + n]
        per_day[d]["bearish"] = per_day[d]["cc"] < per_day[d]["co"]
        pos += n
    return per_day


# ------------------------- per-day simulation -------------------------

def run_giveback_nmin(day, cand, k, mode, drift, gb, scale_body, max_gb, min_gap):
    """FB 5-min entry; giveback yellow driven by the day's N-min candles. TP on 5-min high,
    yellow on bearish N-min close, EOD 14:00. Mirrors run_giveback_variant.run_giveback at N-min."""
    ent = day["entry"]
    if ent is None: return None
    i, ep, ol, tp, t0 = ent
    hhmm, high, low, close, etime = day["hhmm"], day["high"], day["low"], day["close"], day["close_et"]
    risk = ep - ol

    comp_of_index = cand["comp_of_index"]
    cc, co, catr, cbear = cand["cc"], cand["co"], cand["atr"], cand["bearish"]
    yellow = ol
    prev_yellow = np.nan
    gb_on = gb > 0.0

    for j in range(i + 1, len(hhmm)):
        # 1. TP -- 5-min intrabar high (resting limit)
        if high[j] >= tp:
            return _close(ep, tp, t0, etime[j], "TP", risk)
        # 2. N-min candle completed at this 5-min bar?  -> update yellow, test bearish-close exit
        m = comp_of_index[j]
        if m > 0:                                        # need a prior candle for giveback/ratchet
            a = catr[m]
            if np.isfinite(a) and a > 0:
                raw_yellow = cc[m] - k * a
                prev_bearish = bool(cbear[m - 1])
                if gb_on and prev_bearish and not np.isnan(prev_yellow):
                    gap = max(0.0, prev_yellow - raw_yellow)
                    frac = gb
                    if scale_body:
                        frac *= min(1.0, (co[m - 1] - cc[m - 1]) / a)
                    cptr = prev_yellow - min(gap * frac, max_gb * a)
                elif mode == "pure_ratchet":
                    cptr = max(prev_yellow, raw_yellow) if not np.isnan(prev_yellow) else raw_yellow
                else:  # drift_floor
                    base = (prev_yellow + drift) if not np.isnan(prev_yellow) else raw_yellow
                    cptr = max(base, raw_yellow)
                if gb_on:
                    cptr = max(cptr, raw_yellow + min_gap * a)
                yellow = max(cptr, ol)
                if cbear[m] and cc[m] <= yellow:
                    return _close(ep, cc[m], t0, etime[j], "YELLOW", risk)
                prev_yellow = yellow
        # 3. EOD 14:00
        if hhmm[j] >= 1400:
            return _close(ep, close[j], t0, etime[j], "EOD", risk)
    return _close(ep, close[-1], t0, etime[-1], "EOD_LAST", risk)


def _combos():
    out = []
    for k, mode, drift, gb, scale, maxgb, mingap in itertools.product(
            KS, MODES, DRIFTS, GBS, SCALES, MAXGB, MINGAP):
        if mode == "pure_ratchet" and drift != 0.0: continue
        if gb == 0.0 and (scale, maxgb, mingap) != (SCALES[0], MAXGB[0], MINGAP[0]): continue
        out.append((k, mode, drift, gb, scale, maxgb, mingap))
    return out


def _worker(N):
    days, keys = _DAYS, _KEYS
    cands = build_candles(days, keys, N)
    rows = []
    for k, mode, drift, gb, scale, maxgb, mingap in _combos():
        trades = [r for d in keys
                  if (r := run_giveback_nmin(days[d], cands[d], k, mode, drift, gb, scale, maxgb, mingap)) is not None]
        m = metrics(trades)
        if m is None: continue
        rows.append(dict(N=N, k=k, mode=mode, drift=drift, giveback=gb, scale_body=scale,
                         max_gb=maxgb, min_gap=mingap,
                         IS_pf=m["IS"]["pf"], OOS_pf=m["OOS"]["pf"], FULL_pf=m["FULL"]["pf"],
                         FULL_net=m["FULL"]["net"], FULL_dd=m["FULL"]["dd"], n=m["FULL"]["n"]))
    return rows


def _init(days, keys):
    global _DAYS, _KEYS
    _DAYS, _KEYS = days, keys


def main():
    print("Loading 5-min bars...", flush=True)
    days = load_days(); keys = sorted(days.keys())
    for d in keys:                                   # precompute FB entry once (TF/param-independent)
        days[d]["entry"] = find_entry(days[d])
    n_traded = sum(days[d]["entry"] is not None for d in keys)
    print(f"  {len(keys)} session days, {n_traded} with an FB entry\n", flush=True)

    base = metrics([r for d in keys if (r := run_static(days[d])) is not None])
    print(f"V0 STATIC ORB_Low (no trail):        IS PF {base['IS']['pf']:.3f} | OOS PF {base['OOS']['pf']:.3f} "
          f"| FULL PF {base['FULL']['pf']:.3f} net ${base['FULL']['net']:,.0f} (n={base['FULL']['n']})\n", flush=True)

    print(f"Sweeping {len(_combos())} yellow configs x {len(TIMEFRAMES)} timeframes "
          f"(parallel across timeframes)...\n", flush=True)
    with ProcessPoolExecutor(max_workers=len(TIMEFRAMES), initializer=_init, initargs=(days, keys)) as ex:
        results = list(ex.map(_worker, TIMEFRAMES))
    df = pd.concat([pd.DataFrame(r) for r in results], ignore_index=True)
    df["robust_pf"] = df[["IS_pf", "OOS_pf"]].min(axis=1).replace(np.inf, 9.9)
    df.to_csv(OUT_DIR / "yellow_timeframe_sweep.csv", index=False)

    def is_live(r):
        return (r.k == LIVE_CFG["k"] and r.mode == LIVE_CFG["mode"] and r.drift == LIVE_CFG["drift"]
                and r.giveback == LIVE_CFG["giveback"] and r.scale_body == LIVE_CFG["scale_body"]
                and r.max_gb == LIVE_CFG["max_gb"] and r.min_gap == LIVE_CFG["min_gap"])

    cols = ["k", "mode", "drift", "giveback", "scale_body", "max_gb", "min_gap",
            "IS_pf", "OOS_pf", "robust_pf", "FULL_pf", "FULL_net", "FULL_dd", "n"]
    print("=" * 96)
    print("BEST yellow config per timeframe (ranked robust min(IS,OOS) PF):")
    print("=" * 96)
    best_rows = []
    for N in TIMEFRAMES:
        sub = df[df.N == N].sort_values("robust_pf", ascending=False)
        b = sub.iloc[0]
        best_rows.append(b)
        live = sub[sub.apply(is_live, axis=1)]
        lref = f"   [live-cfg here: robust {live.iloc[0].robust_pf:.3f}]" if len(live) else ""
        print(f"\n--- N = {N}-min ---{lref}")
        print(sub.head(3)[cols].round(3).to_string(index=False))

    print("\n" + "=" * 96)
    print("TIMEFRAME LEADERBOARD (each timeframe's best robust config):")
    print("=" * 96)
    bl = pd.DataFrame(best_rows).sort_values("robust_pf", ascending=False)
    print(bl[["N"] + cols].round(3).to_string(index=False))
    win = bl.iloc[0]
    print(f"\n  WINNER: N={int(win.N)}-min  robust PF {win.robust_pf:.3f} "
          f"(IS {win.IS_pf:.3f} / OOS {win.OOS_pf:.3f}) net ${win.FULL_net:,.0f}")
    print(f"  -> Full table: {OUT_DIR / 'yellow_timeframe_sweep.csv'}")


if __name__ == "__main__":
    main()
