"""FB (Fabio ORB) bearish-giveback stop sweep — STAGE 1: FB standalone equity PF (IS/OOS).

Adapts the OD yellow-giveback stop to FB: the stop is a trailing "yellow" that STARTS at ORB_Low
(its hard floor — never looser than the original static stop), ratchets UP toward raw_yellow =
close - k*ATR as price rises, and on a bar FOLLOWING a bearish 5-min candle RETREATS (gives back) a
body-scaled fraction of the (prev_yellow - raw_yellow) gap instead of tightening. Exit is OD-style:
a BEARISH 5-min CLOSE <= yellow (not intrabar touch). TP fixed at 4R; EOD 14:00.

Three tiers to isolate what does the work:
  V0            static ORB_Low (current FB)                        -> the PF 1.347 baseline
  trailing off  yellow trails, giveback=0                          -> does a trailing stop help at all?
  trailing+gb   yellow trails + bearish giveback                   -> does giveback add?

Reuses the loader/entry/reporting shape of run_exit_variants.py. Ranks by robust min(IS_PF, OOS_PF).
Writes the winning configs to results/ for STAGE 2 (1-MNQ 4-way pass rate).

Run:  python scripts/fabio_orb/run_giveback_variant.py
"""
from __future__ import annotations
import warnings; warnings.simplefilter("ignore")
import itertools
from pathlib import Path
from datetime import date
import numpy as np, pandas as pd

VOL_PARQUET = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
OUT_DIR = Path(__file__).parent / "results"; OUT_DIR.mkdir(exist_ok=True)
ET = "America/New_York"
ORB_START_HHMM, ORB_END_HHMM, TRADE_END_HHMM, SKIP_BUCKET = 830, 900, 1400, 930
TICK_SIZE, TICK_VALUE = 0.25, 5.0
DPP = TICK_VALUE / TICK_SIZE
SLIP_PTS = 1 * TICK_SIZE
COMM = 5.0
CUTOFF = date(2024, 3, 12)     # IS/OOS split (same as run_exit_variants)
ATR_LEN = 14                   # 5-min ATR length (fixed; k scales the trail)


def load_days():
    df = pd.read_parquet(VOL_PARQUET)
    agg = df.groupby("bar_open_time", as_index=False).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"),
    ).dropna(subset=["open", "high", "low", "close"])
    agg["close_et"] = agg["bar_open_time"] + pd.Timedelta(minutes=5)
    agg["hhmm"] = agg["close_et"].dt.hour * 100 + agg["close_et"].dt.minute
    agg = agg[(agg["hhmm"] > ORB_START_HHMM) & (agg["hhmm"] <= TRADE_END_HHMM)].copy()
    agg["delta"] = agg["buy_vol"] - agg["sell_vol"]
    agg["session_date"] = agg["close_et"].dt.normalize()
    days = {}
    for sd, sub in agg.groupby("session_date"):
        sub = sub.sort_values("close_et").reset_index(drop=True)
        pc = sub["close"].shift()
        tr = np.maximum(sub["high"] - sub["low"],
                        np.maximum((sub["high"] - pc).abs(), (sub["low"] - pc).abs()))
        # strictly backward-looking; NO bfill. Bars 0-2 stay NaN (need >=3 TR) but FB entries are >=10
        # bars in, so no trade ever reads a NaN/filled ATR. Identical to the validated run minus the bfill.
        sub["atr"] = tr.rolling(ATR_LEN, min_periods=3).mean()
        in_orb = sub[(sub["hhmm"] > ORB_START_HHMM) & (sub["hhmm"] <= ORB_END_HHMM)]
        if in_orb.empty: continue
        post = sub[(sub["hhmm"] > ORB_END_HHMM) & (sub["hhmm"] <= TRADE_END_HHMM)].reset_index(drop=True)
        if post.empty: continue
        days[pd.Timestamp(sd).tz_localize(None)] = {
            "orb_high": float(in_orb["high"].max()), "orb_low": float(in_orb["low"].min()),
            "hhmm": post["hhmm"].to_numpy(),
            "open": post["open"].to_numpy(np.float64), "high": post["high"].to_numpy(np.float64),
            "low": post["low"].to_numpy(np.float64), "close": post["close"].to_numpy(np.float64),
            "delta": post["delta"].to_numpy(np.float64), "atr": post["atr"].to_numpy(np.float64),
            "close_et": post["close_et"].to_numpy(),
        }
    return days


def find_entry(day):
    oh, ol = day["orb_high"], day["orb_low"]
    hhmm, close, delta, etime = day["hhmm"], day["close"], day["delta"], day["close_et"]
    N, DTHR, n = 4, 300.0, len(hhmm)
    for i in range(N - 1, n):
        if hhmm[i] > TRADE_END_HHMM: break
        if hhmm[i] == SKIP_BUCKET: continue
        if not all(close[i - k] > oh for k in range(N)): continue
        if delta[i] < DTHR: continue
        ep = close[i]
        if ol >= ep: continue
        return i, ep, ol, ep + 4.0 * (ep - ol), etime[i]
    return None


def _close(ep, xp, t0, t1, reason, risk):
    raw = xp - ep
    net = raw * DPP - 2 * SLIP_PTS * DPP - COMM
    return {"entry_time": pd.Timestamp(t0), "exit_time": pd.Timestamp(t1), "entry": ep, "exit": xp,
            "risk_pts": risk, "raw_pts": raw, "net_dollars": net, "reason": reason}


def run_static(day):
    """V0 baseline: static ORB_Low SL (intrabar touch), TP 4R, EOD 14:00 — current FB."""
    ent = find_entry(day)
    if ent is None: return None
    i, ep, ol, tp, t0 = ent
    hhmm, high, low, close, etime = day["hhmm"], day["high"], day["low"], day["close"], day["close_et"]
    risk = ep - ol
    for j in range(i + 1, len(hhmm)):
        if hhmm[j] >= TRADE_END_HHMM: return _close(ep, close[j], t0, etime[j], "EOD", risk)
        if low[j] <= ol: return _close(ep, ol, t0, etime[j], "SL", risk)
        if high[j] >= tp: return _close(ep, tp, t0, etime[j], "TP", risk)
    return _close(ep, close[-1], t0, etime[-1], "EOD_LAST", risk)


def run_giveback(day, k, mode, drift, gb, scale_body, max_gb, min_gap):
    """Trailing yellow from ORB_Low with bearish giveback, OD-style bearish-close exit."""
    ent = find_entry(day)
    if ent is None: return None
    i, ep, ol, tp, t0 = ent
    op, hi, lo, cl = day["open"], day["high"], day["low"], day["close"]
    hhmm, atr, etime = day["hhmm"], day["atr"], day["close_et"]
    risk = ep - ol
    yellow = ol
    prev_yellow = np.nan
    gb_on = gb > 0.0
    for j in range(i + 1, len(hhmm)):
        a = atr[j] if atr[j] > 0 else max(atr[i], 1e-6)
        raw_yellow = cl[j] - k * a
        prev_bearish = cl[j - 1] < op[j - 1]
        if gb_on and prev_bearish and not np.isnan(prev_yellow):
            pyi = prev_yellow
            gap = max(0.0, pyi - raw_yellow)
            frac = gb
            if scale_body and a > 0:
                frac *= min(1.0, (op[j - 1] - cl[j - 1]) / a)
            cand = pyi - min(gap * frac, max_gb * a)
        elif mode == "pure_ratchet":
            cand = max(prev_yellow, raw_yellow) if not np.isnan(prev_yellow) else raw_yellow
        else:  # drift_floor
            base = (prev_yellow + drift) if not np.isnan(prev_yellow) else raw_yellow
            cand = max(base, raw_yellow)
        if gb_on:
            cand = max(cand, raw_yellow + min_gap * a)
        yellow = max(cand, ol)                     # hard floor: never looser than ORB_Low
        if hi[j] >= tp: return _close(ep, tp, t0, etime[j], "TP", risk)
        if cl[j] <= yellow and cl[j] < op[j]: return _close(ep, cl[j], t0, etime[j], "YELLOW", risk)
        if hhmm[j] >= TRADE_END_HHMM: return _close(ep, cl[j], t0, etime[j], "EOD", risk)
        prev_yellow = yellow
    return _close(ep, cl[-1], t0, etime[-1], "EOD_LAST", risk)


def metrics(trades):
    df = pd.DataFrame(trades)
    if df.empty: return None
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["date"] = df["entry_time"].dt.tz_convert(ET).dt.date
    out = {}
    for name, s in (("IS", df[df.date < CUTOFF]), ("OOS", df[df.date >= CUTOFF]), ("FULL", df)):
        if len(s) == 0:
            out[name] = {"n": 0, "pf": 0.0, "net": 0.0, "win": 0.0, "dd": 0.0}; continue
        wd = s.loc[s.net_dollars > 0, "net_dollars"].sum()
        ld = -s.loc[s.net_dollars < 0, "net_dollars"].sum()
        eq = s.sort_values("entry_time")["net_dollars"].cumsum()
        out[name] = {"n": len(s), "pf": (wd / ld if ld > 0 else float("inf")),
                     "net": s.net_dollars.sum(), "win": 100 * (s.net_dollars > 0).mean(),
                     "dd": (eq - eq.cummax()).min()}
    return out


def main():
    print("Loading 5-min bars...", flush=True)
    days = load_days(); keys = sorted(days.keys())
    print(f"  {len(keys)} session days\n", flush=True)

    base = metrics([r for d in keys if (r := run_static(days[d])) is not None])
    print(f"V0 STATIC ORB_Low (current FB): IS PF {base['IS']['pf']:.3f} | OOS PF {base['OOS']['pf']:.3f} | "
          f"FULL PF {base['FULL']['pf']:.3f} net ${base['FULL']['net']:,.0f} DD ${base['FULL']['dd']:,.0f} "
          f"(n={base['FULL']['n']})\n", flush=True)

    KS = [1.0, 1.5, 2.0, 3.0, 4.0]
    MODES = ["pure_ratchet", "drift_floor"]
    DRIFTS = [0.0, 1.0]
    GBS = [0.0, 0.3, 0.5, 0.7]
    SCALES = [True, False]
    MAXGB = [0.5, 1.0]
    MINGAP = [0.0, 0.3]
    rows = []
    combos = list(itertools.product(KS, MODES, DRIFTS, GBS, SCALES, MAXGB, MINGAP))
    for k, mode, drift, gb, scale, maxgb, mingap in combos:
        if mode == "pure_ratchet" and drift != 0.0: continue          # drift only matters for drift_floor
        if gb == 0.0 and (scale, maxgb, mingap) != (SCALES[0], MAXGB[0], MINGAP[0]): continue  # gb off -> knobs inert
        m = metrics([r for d in keys if (r := run_giveback(days[d], k, mode, drift, gb, scale, maxgb, mingap)) is not None])
        if m is None: continue
        rows.append({"k": k, "mode": mode, "drift": drift, "giveback": gb, "scale_body": scale,
                     "max_gb": maxgb, "min_gap": mingap,
                     "IS_pf": m["IS"]["pf"], "OOS_pf": m["OOS"]["pf"], "FULL_pf": m["FULL"]["pf"],
                     "FULL_net": m["FULL"]["net"], "FULL_dd": m["FULL"]["dd"], "n": m["FULL"]["n"]})
    df = pd.DataFrame(rows)
    df["robust_pf"] = df[["IS_pf", "OOS_pf"]].min(axis=1).replace(np.inf, 9.9)
    df.to_csv(OUT_DIR / "giveback_fb_sweep.csv", index=False)

    cols = ["k", "mode", "drift", "giveback", "scale_body", "max_gb", "min_gap",
            "IS_pf", "OOS_pf", "FULL_pf", "FULL_net", "FULL_dd", "n"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("=== Trailing, giveback OFF (does a trailing stop help at all?) ===")
        print(df[df.giveback == 0].sort_values("robust_pf", ascending=False).head(6)[cols].round(3).to_string(index=False))
        print("\n=== Top 12 by robust min(IS,OOS) PF (giveback ON) ===")
        print(df[df.giveback > 0].sort_values("robust_pf", ascending=False).head(12)[cols].round(3).to_string(index=False))

    lb, lo = base["IS"]["pf"], base["OOS"]["pf"]
    beat = df[(df.IS_pf > lb) & (df.OOS_pf > lo)]
    print(f"\nBaseline (static) IS {lb:.3f} / OOS {lo:.3f}. Configs beating it on BOTH: {len(beat)}/{len(df)}")
    print(f"Full table -> {OUT_DIR / 'giveback_fb_sweep.csv'}")


if __name__ == "__main__":
    main()
