"""Optimize pure_ratchet yellow + green-band exit parameters.
Gate: must produce positive total PnL in BOTH IS and OOS.

Sweep:
  yellow_atr_mult : stop distance multiplier
  green_atr_mult  : target ATR multiplier
  green_base      : target intercept
  green_decay     : per-bar target tightening

Holds entry config constant (977-trade variant). 20-min bars for management.
Reuses the building blocks from test_pure_ratchet_exits.py.
"""

from __future__ import annotations

import datetime as dt
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from range_break_entry_summary import apply_filters, mode1_chained_dedupe
from test_pure_ratchet_exits import (
    rma_atr, build_20min_bars, filter_to_entries,
    YELLOW_ATR_LEN, GREEN_ATR_LEN, FORCE_CLOSE_TIME,
    RED_INTERCEPT, RED_DRIFT,
)

PARQUET_DIR  = Path(__file__).parent / "parquets"
TRADELOG_DIR = Path(__file__).parent.parent / "tradelogs"
OUT_TXT      = TRADELOG_DIR / "robust_configs" / "pure_ratchet_optimization.txt"

# Sweep
YELLOW_MULTS = [1.0, 1.25, 1.42, 1.75, 2.0, 2.5, 3.0]
GREEN_MULTS  = [1.0, 1.5, 2.0, 2.5, 2.6, 3.5]
GREEN_BASES  = [25, 50, 75, 100, 150, 200]
GREEN_DECAYS = [0.0, 0.5, 1.0, 1.31, 2.0]


def simulate_exit_arrays(sign: int, entry_price: float, init_atr_y: float,
                          o_arr, h_arr, l_arr, c_arr, ay_arr, ag_arr,
                          force_idx: int,
                          ymult, gmult, gbase, gdecay) -> tuple[float, str, int]:
    """Pure-numpy variant. force_idx = index into arrays where bar.time >= force_close_time."""
    yellow_val = entry_price - sign * ymult * init_atr_y
    prev_yellow = yellow_val
    n = len(c_arr)
    for i in range(n):
        bars_in_trade = i + 1
        o, h, l, c = o_arr[i], h_arr[i], l_arr[i], c_arr[i]
        ay = ay_arr[i]; ag = ag_arr[i]

        if not np.isnan(ay):
            raw_yellow = c - sign * ymult * ay
            yellow_val = max(prev_yellow, raw_yellow) if sign > 0 \
                          else min(prev_yellow, raw_yellow)

        red_val = entry_price + sign * (RED_INTERCEPT + RED_DRIFT * bars_in_trade)
        green_offset = (gbase - gdecay * bars_in_trade
                        + (gmult * ag if not np.isnan(ag) else 0.0))
        green_val = red_val + sign * green_offset

        if sign > 0 and h >= green_val:
            return (c - entry_price, "TP_GREEN", bars_in_trade)
        if sign < 0 and l <= green_val:
            return (entry_price - c, "TP_GREEN", bars_in_trade)
        if sign > 0 and c <= yellow_val and c < o:
            return (c - entry_price, "SL_YELLOW", bars_in_trade)
        if sign < 0 and c >= yellow_val and c > o:
            return (entry_price - c, "SL_YELLOW", bars_in_trade)
        if i >= force_idx:
            return (sign * (c - entry_price), "FORCE_CLOSE", bars_in_trade)

        prev_yellow = yellow_val

    return (sign * (c_arr[-1] - entry_price), "EOD", n)


def stats_combo(ded: pd.DataFrame, pnls: np.ndarray) -> dict:
    if len(ded) == 0:
        return {"n":0, "total":0.0, "pf":0.0, "sharpe":0.0, "wr":0.0, "max_dd":0.0,
                "long_total":0.0, "short_total":0.0}
    pos = pnls[pnls > 0].sum(); neg = -pnls[pnls < 0].sum()
    pf = pos / neg if neg > 0 else (np.inf if pos > 0 else 0.0)
    daily = pd.Series(pnls, index=ded["date"].values).groupby(level=0).sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0
    eq = np.cumsum(pnls); peak = np.maximum.accumulate(eq); max_dd = (eq - peak).min()
    long_mask = (ded["direction"]=="LONG").values
    return {"n":len(ded), "total":pnls.sum(), "pf":pf, "sharpe":sharpe,
            "wr":(pnls > 0).mean(), "max_dd":max_dd,
            "long_total":pnls[long_mask].sum(), "short_total":pnls[~long_mask].sum()}


def precache_trade_bars(ded_with_ts, bars20):
    """For each trade pre-extract its in-day post-entry 20-min bars + entry-bar ATR.
    Returns list of dicts with arrays.
    """
    cache = []
    bars20_idx = bars20.index
    bars20_dates = bars20_idx.date
    bars20_times = bars20_idx.time
    o_arr_full = bars20["open"].values
    h_arr_full = bars20["high"].values
    l_arr_full = bars20["low"].values
    c_arr_full = bars20["close"].values
    ay_arr_full = bars20["atr_y"].values
    ag_arr_full = bars20["atr_g"].values

    directions = list(ded_with_ts["direction"])
    entry_tss  = list(ded_with_ts["entry_ts"])
    entry_prices = list(ded_with_ts["entry_price"])

    # Use searchsorted on the (timezone-aware) DatetimeIndex
    for i in range(len(ded_with_ts)):
        ts = entry_tss[i]
        # First idx STRICTLY AFTER entry_ts
        start = bars20_idx.searchsorted(ts, side="right")
        if start >= len(bars20_idx):
            cache.append(None); continue
        # Find day boundary
        ent_date = ts.date()
        end = start
        while end < len(bars20_idx) and bars20_dates[end] == ent_date:
            end += 1
        if end == start:
            cache.append(None); continue
        # Initial ATR_y from last bar at-or-before entry_ts
        init_idx = start - 1
        if init_idx < 0 or np.isnan(ay_arr_full[init_idx]):
            cache.append(None); continue
        # force_idx: first bar where time >= 16:00
        force_idx_local = -1
        for j in range(start, end):
            if bars20_times[j] >= FORCE_CLOSE_TIME:
                force_idx_local = j - start
                break
        if force_idx_local == -1:
            force_idx_local = end - start - 1   # use last bar as force-close

        cache.append({
            "sign": 1 if directions[i] == "LONG" else -1,
            "entry_price": entry_prices[i],
            "init_atr_y": float(ay_arr_full[init_idx]),
            "o": o_arr_full[start:end],
            "h": h_arr_full[start:end],
            "l": l_arr_full[start:end],
            "c": c_arr_full[start:end],
            "ay": ay_arr_full[start:end],
            "ag": ag_arr_full[start:end],
            "force_idx": force_idx_local,
        })
    return cache


def evaluate_combo(cache, ymult, gmult, gbase, gdecay):
    n = len(cache)
    pnls = np.zeros(n)
    for i in range(n):
        c = cache[i]
        if c is None:
            continue
        pnl, _, _ = simulate_exit_arrays(
            c["sign"], c["entry_price"], c["init_atr_y"],
            c["o"], c["h"], c["l"], c["c"], c["ay"], c["ag"],
            c["force_idx"], ymult, gmult, gbase, gdecay)
        pnls[i] = pnl
    return pnls


def prep_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Filter + add tz-aware entry timestamp + date."""
    ded = filter_to_entries(df)
    if ded.empty:
        return ded
    ts = pd.to_datetime(ded["entry_time"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    else:
        ts = ts.dt.tz_convert("America/New_York")
    ded = ded.copy()
    ded["entry_ts"] = ts
    ded["date"] = pd.to_datetime(ded["date"]).dt.date
    return ded


def main():
    print("loading 20-min bars + filtering trades...")
    bars20 = build_20min_bars()
    is_df  = pd.read_parquet(PARQUET_DIR / "entry_signal_trades.parquet")
    oos_df = pd.read_parquet(PARQUET_DIR / "entry_signal_trades_oos.parquet")
    is_ded  = prep_trades(is_df)
    oos_ded = prep_trades(oos_df)
    print(f"  IS:  {len(is_ded):,}    OOS: {len(oos_ded):,}")

    print("pre-caching per-trade bar windows...")
    is_cache  = precache_trade_bars(is_ded,  bars20)
    oos_cache = precache_trade_bars(oos_ded, bars20)
    print(f"  IS cache: {sum(c is not None for c in is_cache)}/{len(is_cache)} valid")
    print(f"  OOS cache: {sum(c is not None for c in oos_cache)}/{len(oos_cache)} valid")

    combos = list(product(YELLOW_MULTS, GREEN_MULTS, GREEN_BASES, GREEN_DECAYS))
    print(f"sweeping {len(combos):,} combos (yellow x green_mult x green_base x green_decay)")

    rows = []
    import time as _time
    t0 = _time.time()
    for i, (ym, gm, gb, gd) in enumerate(combos, 1):
        is_pnls  = evaluate_combo(is_cache,  ym, gm, gb, gd)
        oos_pnls = evaluate_combo(oos_cache, ym, gm, gb, gd)
        is_s  = stats_combo(is_ded,  is_pnls)
        oos_s = stats_combo(oos_ded, oos_pnls)
        rows.append({
            "ymult":ym, "gmult":gm, "gbase":gb, "gdecay":gd,
            "is_total":is_s["total"],  "is_pf":is_s["pf"],   "is_sharpe":is_s["sharpe"],   "is_wr":is_s["wr"],   "is_mdd":is_s["max_dd"],
            "oos_total":oos_s["total"],"oos_pf":oos_s["pf"], "oos_sharpe":oos_s["sharpe"], "oos_wr":oos_s["wr"], "oos_mdd":oos_s["max_dd"],
            "is_long_total":is_s["long_total"], "is_short_total":is_s["short_total"],
            "oos_long_total":oos_s["long_total"], "oos_short_total":oos_s["short_total"],
        })
        if i % 100 == 0 or i == len(combos):
            elapsed = _time.time() - t0
            rate = i / max(elapsed, 0.01)
            eta = (len(combos) - i) / max(rate, 0.01)
            print(f"  {i}/{len(combos)}  elapsed={elapsed:.0f}s  rate={rate:.1f}/s  eta={eta:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    # Cap PFs for readability
    df["is_pf_capped"]  = df["is_pf"].clip(upper=99.0)
    df["oos_pf_capped"] = df["oos_pf"].clip(upper=99.0)
    df["min_pf"]     = df[["is_pf_capped","oos_pf_capped"]].min(axis=1)
    df["min_sharpe"] = df[["is_sharpe","oos_sharpe"]].min(axis=1)
    df["combined_total"] = df["is_total"] + df["oos_total"]

    robust = df[(df["is_total"] > 0) & (df["oos_total"] > 0)].sort_values("min_sharpe", ascending=False)
    print(f"\n{len(robust)}/{len(df)} configs positive on both IS and OOS")

    lines = []
    lines.append("=" * 220)
    lines.append("PURE_RATCHET EXIT OPTIMIZATION — 977-trade entry config")
    lines.append("=" * 220)
    lines.append("")
    lines.append("Entry config: B2 X=0.75 N=15 D=70 strict BAND_K=0.25 + conf_N=5 conf_D=75 HALF, chained Mode 1")
    lines.append("Exit:         pure_ratchet yellow + green-decay target on 20-min bars")
    lines.append("              red_intercept=0.0 red_drift=0.45 ATR_yellow_len=14 ATR_green_len=13")
    lines.append("              force_close at 16:00 ET")
    lines.append("")
    lines.append(f"Sweep: {len(YELLOW_MULTS)}x{len(GREEN_MULTS)}x{len(GREEN_BASES)}x{len(GREEN_DECAYS)} = {len(combos)} combos")
    lines.append(f"Robustness gate: positive total in BOTH IS and OOS")
    lines.append("")
    lines.append(f"Robust configs: {len(robust)}/{len(df)}")
    lines.append("")
    lines.append("=" * 220)
    lines.append("TOP 25 ROBUST (sorted by min Sharpe descending)")
    lines.append("=" * 220)
    cols = ["ymult","gmult","gbase","gdecay",
            "is_total","is_pf_capped","is_sharpe","is_wr","is_mdd",
            "oos_total","oos_pf_capped","oos_sharpe","oos_wr","oos_mdd",
            "min_pf","min_sharpe","combined_total"]
    head = robust.head(25)
    lines.append(head[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    lines.append("")
    lines.append("=" * 220)
    lines.append("LONG/SHORT split for top 25:")
    lines.append("=" * 220)
    cols2 = ["ymult","gmult","gbase","gdecay",
             "is_long_total","is_short_total","oos_long_total","oos_short_total"]
    lines.append(head[cols2].to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    lines.append("")
    lines.append("=" * 220)
    lines.append("DEFAULT BENCHMARK (drift original): ymult=1.42 gmult=2.6 gbase=107.6 gdecay=1.31")
    lines.append("=" * 220)
    # Approximate to nearest sweep point (107.6 -> 100; 1.31 -> 1.31)
    bench = df[(df["ymult"]==1.42) & (df["gmult"]==2.6) & (df["gbase"]==100) & (df["gdecay"]==1.31)]
    if not bench.empty:
        lines.append(bench[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    else:
        # Closest match
        df["dist"] = (df["ymult"]-1.42).abs() + (df["gmult"]-2.6).abs() + (df["gbase"]-107.6).abs()/100 + (df["gdecay"]-1.31).abs()
        lines.append("(closest match to defaults)")
        lines.append(df.sort_values("dist").head(1)[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}")
    print()
    # Print top 10 to console
    print("TOP 10 by min Sharpe:")
    print(head[cols].head(10).to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    sys.exit(main())
