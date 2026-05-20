"""Backtest: VA mean-reversion with reaction-level entries.

Entry conditions:
  - Today's premarket-last-print is outside prev-day RTH VA
  - |overshoot| as % of prev-day VA width < 40%
  - During RTH (9:30-16:00 ET), a 5-min bar wicks into a 'reaction level':
      SHORT (open > prev VAH): pool = today's OHI + GEX levels > open
      LONG  (open < prev VAL): pool = today's OLO + GEX levels < open
  - Next bar (confirmation) closes in trade direction
  - Optional orderflow delta thresholds on top-half (SHORT) / bottom-half (LONG)
    of BOTH signal and conf bars (windowed absorption scan).

Exit:
  - TP: prev VAH (short) / prev VAL (long)
  - SL: entry  sl_mult * ATR_14_5min
  - Force-close at 16:00 ET
  - Chained Mode 1 dedupe (one trade in flight at a time)

Sweep:
  N        in {5, 10, 15, 20}   (window size in ticks for absorption scan)
  sig_D    in {0, 50, 100, 150} (signal-bar absorbed |delta| threshold)
  conf_D   in {0, 50, 100, 150} (conf-bar absorbed  |delta| threshold)
  sl_mult  in {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0}

Output: scripts/overnight range strat/tradelogs/robust_configs/va_revert_backtest.txt
"""
from __future__ import annotations

import datetime as dt
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Reuse infrastructure from the break strategy
PROJECT_SCRIPTS = Path(__file__).parent.parent / "overnight range strat" / "scripts"
sys.path.insert(0, str(PROJECT_SCRIPTS))
from range_break_entry_signal_study import (
    compute_windowed_absorption, load_range_per_day, load_mq_levels,
    levels_for_date, load_5min_features,
)

VOL5M       = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
M1_BARS     = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
VA_PARQUET  = Path(__file__).parent / "prev_day_rth_va.parquet"
CACHE_SIGS  = Path(__file__).parent / "va_revert_signals.parquet"
OUT_TXT     = Path(__file__).parent.parent / "overnight range strat" / "tradelogs" / "robust_configs" / "va_revert_backtest.txt"
ET          = "America/New_York"

RTH_START   = dt.time(9, 30)
RTH_END     = dt.time(16, 0)
FORCE_CLOSE = dt.time(16, 0)
TICK        = 0.25
ATR_PERIOD  = 14

OVERSHOOT_LIMIT_PCT = 75.0   # most permissive cap; post-filter to {40, 50, 75} in sweep
MIN_VA_WIDTH        = 20.0
MIN_RTH_BARS        = 30

N_GRID      = [5, 10, 15, 20]
SIG_D_GRID  = [0, 50, 100, 150]
CONF_D_GRID = [0, 50, 100, 150]
SL_MULT_GRID = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
MIN_TRADES  = 30


# ============================== load + prep ==============================

def load_va_table():
    if not VA_PARQUET.exists():
        raise FileNotFoundError(f"{VA_PARQUET} -- run value_area_revert_study.py first to build it.")
    va = pd.read_parquet(VA_PARQUET)
    va["date"] = pd.to_datetime(va["date"]).dt.date
    return va.sort_values("date").reset_index(drop=True)


def load_premarket_open():
    df = pd.read_parquet(M1_BARS, columns=["close"])
    idx = pd.to_datetime(df.index, utc=True).tz_convert(ET)
    df = df.set_index(idx).sort_index()
    df["date"] = df.index.date
    df["t"]    = df.index.time
    pre = df[df["t"] < RTH_START].copy()
    last = pre.groupby("date").tail(1)
    return last[["date", "close"]].rename(columns={"close": "open_price"}).set_index("date")


def build_qualifying_days():
    """For each date with valid prev-day VA + outside-VA open with <40% overshoot."""
    va = load_va_table()
    pre = load_premarket_open()
    # build prev_day lookup
    dates = va.dropna(subset=["vah","val"]).sort_values("date")["date"].tolist()
    prev_lookup = {dates[i]: dates[i-1] for i in range(1, len(dates))}
    va_idx = va.set_index("date")
    rows = []
    for today in sorted(set(va["date"]).intersection(pre.index)):
        prev_d = prev_lookup.get(today)
        if prev_d is None or prev_d not in va_idx.index:
            continue
        vah_p = va_idx.loc[prev_d, "vah"]; val_p = va_idx.loc[prev_d, "val"]
        if not (np.isfinite(vah_p) and np.isfinite(val_p)):
            continue
        width = vah_p - val_p
        if width < MIN_VA_WIDTH:
            continue
        op = float(pre.loc[today, "open_price"])
        if not np.isfinite(op):
            continue
        if op > vah_p:
            direction = "SHORT"
            distance  = op - vah_p
        elif op < val_p:
            direction = "LONG"
            distance  = val_p - op
        else:
            continue
        pct = distance / width * 100
        if pct >= OVERSHOOT_LIMIT_PCT:
            continue
        rows.append({"date": today, "direction": direction,
                     "open_price": op, "vah_prev": float(vah_p),
                     "val_prev": float(val_p), "width_prev": float(width),
                     "distance_pct": pct, "distance_pts": distance,
                     "target": float(vah_p) if direction == "SHORT" else float(val_p)})
    print(f"qualifying days (<{OVERSHOOT_LIMIT_PCT}% overshoot, outside VA): {len(rows)}")
    return pd.DataFrame(rows)


# ============================== signal build ==============================

def adaptive_atr(bars_day, atr_period=ATR_PERIOD):
    bars_day = bars_day.copy()
    bars_day["prev_close"] = bars_day["close"].shift(1)
    bars_day["tr"] = np.maximum.reduce([
        (bars_day["high"] - bars_day["low"]).values,
        (bars_day["high"] - bars_day["prev_close"]).abs().values,
        (bars_day["low"]  - bars_day["prev_close"]).abs().values,
    ])
    bars_day["atr14"] = bars_day["tr"].ewm(alpha=1/atr_period, adjust=False, min_periods=1).mean()
    return bars_day


def build_signals(qual_df, bars_by_day, levels_by_day, ohi_olo_lookup, mq_levels_by_day):
    """For each qualifying day, walk RTH bars, detect signal+conf candidates."""
    rows = []
    for _, row in qual_df.iterrows():
        d = row["date"]
        direction = row["direction"]
        open_price = row["open_price"]
        target = row["target"]
        bars = bars_by_day.get(d)
        if bars is None or len(bars) < MIN_RTH_BARS:
            continue
        # Reaction levels pool
        if direction == "SHORT":
            ohi_today = ohi_olo_lookup.get(d, {}).get("ohi", np.nan)
            gex_today = mq_levels_by_day.get(d, np.array([]))
            pool = []
            if np.isfinite(ohi_today) and ohi_today > open_price:
                pool.append(ohi_today)
            pool.extend([lv for lv in gex_today if lv > open_price])
            pool = sorted(set(pool))
        else:  # LONG
            olo_today = ohi_olo_lookup.get(d, {}).get("olo", np.nan)
            gex_today = mq_levels_by_day.get(d, np.array([]))
            pool = []
            if np.isfinite(olo_today) and olo_today < open_price:
                pool.append(olo_today)
            pool.extend([lv for lv in gex_today if lv < open_price])
            pool = sorted(set(pool), reverse=True)
        if not pool:
            continue
        pool_arr = np.array(pool, dtype=float)

        # ATR per bar (5-min)
        bars = adaptive_atr(bars).reset_index(drop=True)
        bars["body_low"]  = bars[["open","close"]].min(axis=1)
        bars["body_high"] = bars[["open","close"]].max(axis=1)

        levels_day = levels_by_day.get(d, pd.DataFrame())
        levels_by_bar = ({t: g for t, g in levels_day.groupby("bar_open_time")}
                          if not levels_day.empty else {})

        for idx in range(len(bars) - 2):
            bar = bars.iloc[idx]
            # Touch condition
            if direction == "SHORT":
                touched = pool_arr[(pool_arr <= float(bar["high"])) & (pool_arr >= float(bar["low"]))]
                if len(touched) == 0:
                    continue
                lvl = float(touched[np.argmax(np.abs(touched - float(bar["high"])))])  # furthest into wick
            else:  # LONG
                touched = pool_arr[(pool_arr >= float(bar["low"])) & (pool_arr <= float(bar["high"]))]
                if len(touched) == 0:
                    continue
                lvl = float(touched[np.argmax(np.abs(touched - float(bar["low"])))])

            conf_bar = bars.iloc[idx + 1]
            entry_bar = bars.iloc[idx + 2]
            if direction == "SHORT" and not (float(conf_bar["close"]) < float(conf_bar["open"])):
                continue
            if direction == "LONG" and not (float(conf_bar["close"]) > float(conf_bar["open"])):
                continue
            atr_entry = float(conf_bar.get("atr14", np.nan))
            if not np.isfinite(atr_entry) or atr_entry <= 0:
                continue

            # Delta scan on signal + conf bars (top half for SHORT, bottom half for LONG)
            scan_dir = "SHORT" if direction == "SHORT" else "LONG"
            sig_lvls = levels_by_bar.get(bar["bar_open_time"], pd.DataFrame())
            cf_lvls  = levels_by_bar.get(conf_bar["bar_open_time"], pd.DataFrame())
            sig_win = compute_windowed_absorption(
                sig_lvls,
                float(bar["body_low"]), float(bar["body_high"]),
                scan_dir, N_GRID,
            ) if not sig_lvls.empty else {N: (None, None, 0, 0, 0) for N in N_GRID}
            cf_win = compute_windowed_absorption(
                cf_lvls,
                float(conf_bar["body_low"]), float(conf_bar["body_high"]),
                scan_dir, N_GRID,
            ) if not cf_lvls.empty else {N: (None, None, 0, 0, 0) for N in N_GRID}

            rec = {
                "date": d, "direction": direction,
                "signal_time": pd.Timestamp(bar["bar_open_time"]),
                "conf_time":   pd.Timestamp(conf_bar["bar_open_time"]),
                "entry_time":  pd.Timestamp(entry_bar["bar_open_time"]),
                "entry_idx":   int(idx + 2),
                "entry_price": float(entry_bar["open"]),
                "atr_at_entry": atr_entry,
                "level": lvl,
                "target": target,
                "vah_prev": row["vah_prev"], "val_prev": row["val_prev"],
                "distance_pct": row["distance_pct"],
            }
            for N in N_GRID:
                rec[f"sig_abs_w{N}"]  = abs(sig_win.get(N, (None, None, 0, 0, 0))[3])
                rec[f"conf_abs_w{N}"] = abs(cf_win.get(N, (None, None, 0, 0, 0))[3])
            rows.append(rec)
    return pd.DataFrame(rows)


# ============================== exit simulation ==============================

def simulate_exit(direction, entry_price, entry_idx, bars_day, sl_pts, tp_price):
    sign = 1 if direction == "LONG" else -1
    sl_price = entry_price - sign * sl_pts
    n = len(bars_day)
    for k in range(entry_idx + 1, n):
        bar = bars_day.iloc[k]
        bt = pd.Timestamp(bar["bar_open_time"])
        if bt.time() >= FORCE_CLOSE:
            close_px = float(bars_day.iloc[k - 1]["close"]) if k > 0 else float(bar["open"])
            return ("held->close", sign * (close_px - entry_price), bt)
        hi, lo = float(bar["high"]), float(bar["low"])
        if sign > 0:
            tp_hit = hi >= tp_price
            sl_hit = lo <= sl_price
        else:
            tp_hit = lo <= tp_price
            sl_hit = hi >= sl_price
        if tp_hit and sl_hit:
            return ("SL", sign * (sl_price - entry_price), bt)
        if tp_hit:
            return ("TP", sign * (tp_price - entry_price), bt)
        if sl_hit:
            return ("SL", sign * (sl_price - entry_price), bt)
    last = bars_day.iloc[-1]
    return ("held->close", sign * (float(last["close"]) - entry_price),
            pd.Timestamp(last["bar_open_time"]))


def run_chained(sigs, bars_by_day, sl_mult):
    rows = []
    last_exit = pd.Timestamp(0, tz=ET)
    for _, t in sigs.sort_values("entry_time").iterrows():
        ent_t = pd.Timestamp(t["entry_time"])
        if ent_t <= last_exit: continue
        d = pd.to_datetime(t["date"]).date()
        bars_day = bars_by_day.get(d)
        if bars_day is None: continue
        entry_idx = int(t["entry_idx"])
        if entry_idx >= len(bars_day): continue
        entry_price = float(t["entry_price"])
        atr = float(t["atr_at_entry"])
        sl_pts = sl_mult * atr
        tp_price = float(t["target"])
        outcome, pnl, exit_t = simulate_exit(
            t["direction"], entry_price, entry_idx, bars_day, sl_pts, tp_price)
        rows.append({"pnl": pnl, "outcome": outcome, "direction": t["direction"]})
        last_exit = pd.Timestamp(exit_t)
    return pd.DataFrame(rows)


def stats(df):
    n = len(df)
    if n == 0:
        return {"n":0,"wr":0,"pf":0,"total":0,"mdd":0,"tp":0,"sl":0,"hc":0}
    pnl = df["pnl"].values
    wins = pnl[pnl > 0]; losses = pnl[pnl < 0]
    wr = (pnl > 0).mean() * 100
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    eq = np.cumsum(pnl); peak = np.maximum.accumulate(eq); mdd = (eq - peak).min()
    return {"n":n,"wr":wr,"pf":pf,"total":pnl.sum(),"mdd":mdd,
            "tp":int((df["outcome"]=="TP").sum()),
            "sl":int((df["outcome"]=="SL").sum()),
            "hc":int((df["outcome"]=="held->close").sum())}


# ============================== main ==============================

def main():
    print("loading 5-min features (bars + level-resolved vol) ...")
    bars_all, levels_all = load_5min_features((dt.date(2020,12,1), dt.date(2026,5,7)))
    print(f"  bars={len(bars_all):,}  levels={len(levels_all):,}")

    print("loading overnight range + MenthorQ levels ...")
    rng = load_range_per_day()
    mq  = load_mq_levels()

    bars_by_day   = {d: g.sort_values("bar_open_time").reset_index(drop=True)
                     for d, g in bars_all.groupby("session_date")}
    levels_by_day = {d: g for d, g in levels_all.groupby("session_date")}

    # OHI/OLO + MQ per day
    ohi_olo = {}
    for d, r in rng.iterrows():
        ohi_olo[d] = {"ohi": float(r.get("overnight_high", np.nan)),
                      "olo": float(r.get("overnight_low",  np.nan))}
    mq_by_day = {d: levels_for_date(mq, d) for d in bars_by_day.keys()}

    print("building qualifying days ...")
    qual = build_qualifying_days()

    if CACHE_SIGS.exists():
        sigs = pd.read_parquet(CACHE_SIGS)
        print(f"loaded cached signals: {len(sigs)}")
    else:
        print("building signals ...")
        sigs = build_signals(qual, bars_by_day, levels_by_day, ohi_olo, mq_by_day)
        sigs.to_parquet(CACHE_SIGS)
        print(f"  saved {len(sigs)} signals to {CACHE_SIGS}")
    sigs["date"] = pd.to_datetime(sigs["date"]).dt.date
    sigs["entry_time"] = pd.to_datetime(sigs["entry_time"])
    print(f"  by direction:")
    print(sigs["direction"].value_counts().to_string())

    is_sigs  = sigs[sigs["date"] <  dt.date(2025,1,1)].copy()
    oos_sigs = sigs[sigs["date"] >= dt.date(2025,1,1)].copy()
    print(f"  IS={len(is_sigs)}  OOS={len(oos_sigs)}")

    OVERSHOOT_CAPS = [40.0, 50.0, 75.0]
    print(f"\nsweeping overshoot_cap x N x sig_D x conf_D x sl_mult ...")
    rows = []
    total = len(OVERSHOOT_CAPS) * len(N_GRID) * len(SIG_D_GRID) * len(CONF_D_GRID) * len(SL_MULT_GRID)
    done = 0
    for cap in OVERSHOOT_CAPS:
        is_sigs_c  = is_sigs[is_sigs["distance_pct"] < cap]
        oos_sigs_c = oos_sigs[oos_sigs["distance_pct"] < cap]
        for N in N_GRID:
            sig_c  = f"sig_abs_w{N}"
            conf_c = f"conf_abs_w{N}"
            for sig_D in SIG_D_GRID:
                for conf_D in CONF_D_GRID:
                    sub_is  = is_sigs_c[(is_sigs_c[sig_c].fillna(0)  >= sig_D) &
                                         (is_sigs_c[conf_c].fillna(0) >= conf_D)]
                    sub_oos = oos_sigs_c[(oos_sigs_c[sig_c].fillna(0)  >= sig_D) &
                                          (oos_sigs_c[conf_c].fillna(0) >= conf_D)]
                    if len(sub_is) + len(sub_oos) < MIN_TRADES // 2:
                        done += len(SL_MULT_GRID)
                        continue
                    for sl in SL_MULT_GRID:
                        it = run_chained(sub_is,  bars_by_day, sl)
                        ot = run_chained(sub_oos, bars_by_day, sl)
                        at = pd.concat([it, ot], ignore_index=True)
                        s_is, s_oos, s_all = stats(it), stats(ot), stats(at)
                        rows.append({"cap":cap,"N":N,"sig_D":sig_D,"conf_D":conf_D,"sl_mult":sl,
                                     **{f"is_{k}":v for k,v in s_is.items()},
                                     **{f"oos_{k}":v for k,v in s_oos.items()},
                                     **{f"all_{k}":v for k,v in s_all.items()}})
                        done += 1
                        if done % 200 == 0:
                            print(f"  {done}/{total}")
    df = pd.DataFrame(rows)
    df["min_pf"] = df[["is_pf","oos_pf"]].min(axis=1)
    df["min_wr"] = df[["is_wr","oos_wr"]].min(axis=1)
    print(f"completed {len(df)} combos")

    cols = ["cap","N","sig_D","conf_D","sl_mult",
            "is_n","is_wr","is_pf","is_total","is_mdd",
            "oos_n","oos_wr","oos_pf","oos_total","oos_mdd",
            "all_n","all_wr","all_pf","all_total","all_mdd","min_pf","min_wr"]

    L = []
    L.append("=" * 220)
    L.append("VA-REVERT BACKTEST  (overshoot_cap x delta-N-SL sweep)")
    L.append("=" * 220)
    L.append(f"Qualifying days = open outside prev VA AND overshoot < cap of VA width")
    L.append(f"Signal pool (pre-cap, at {OVERSHOOT_LIMIT_PCT}% max): total={len(sigs)}  IS={len(is_sigs)}  OOS={len(oos_sigs)}")
    L.append(f"   by direction: {sigs['direction'].value_counts().to_dict()}")
    L.append(f"Robust = IS_n >= {MIN_TRADES}, both IS+OOS profitable.")
    L.append("")

    for cap in [40.0, 50.0, 75.0]:
        sub = df[df["cap"] == cap]
        qual_cap = sub[(sub["is_n"] >= MIN_TRADES) & (sub["is_total"] > 0) & (sub["oos_total"] > 0)]
        L.append("=" * 220)
        L.append(f"OVERSHOOT CAP = {cap:.0f}%  ->  {len(qual_cap)} robust configs (of {len(sub)} tested)")
        L.append("=" * 220)
        if not qual_cap.empty:
            qual_cap = qual_cap.sort_values("min_pf", ascending=False).reset_index(drop=True)
            L.append("TOP 15 BY min(IS_PF, OOS_PF):")
            L.append(qual_cap.head(15)[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
            L.append("")
            L.append("TOP 10 BY combined total pts:")
            L.append(qual_cap.sort_values("all_total", ascending=False).head(10)[cols].to_string(
                index=False, float_format=lambda x: f"{x:.2f}"))
        L.append("")

    qual_df = df[(df["is_n"] >= MIN_TRADES) & (df["is_total"] > 0) & (df["oos_total"] > 0)]

    L.append("")
    L.append("=" * 220)
    L.append("RELAXED (only IS profitable required, n>=30):")
    L.append("=" * 220)
    relax = df[(df["is_n"] >= MIN_TRADES) & (df["is_total"] > 0)]
    if not relax.empty:
        L.append(f"  {len(relax)} configs")
        L.append(relax.sort_values("is_pf", ascending=False).head(15)[cols].to_string(
            index=False, float_format=lambda x: f"{x:.2f}"))

    # Diagnostic: which N values appear most in top robust?
    if not qual_df.empty:
        L.append("")
        L.append("=" * 220)
        L.append("N VALUE FREQUENCY in top-30 robust configs:")
        L.append("=" * 220)
        L.append(qual_df.head(30)["N"].value_counts().to_string())
        L.append("")
        L.append("SIG_D FREQUENCY in top-30 robust:")
        L.append(qual_df.head(30)["sig_D"].value_counts().to_string())
        L.append("")
        L.append("CONF_D FREQUENCY in top-30 robust:")
        L.append(qual_df.head(30)["conf_D"].value_counts().to_string())
        L.append("")
        L.append("SL_MULT FREQUENCY in top-30 robust:")
        L.append(qual_df.head(30)["sl_mult"].value_counts().to_string())

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
