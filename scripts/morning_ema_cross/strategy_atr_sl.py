"""Morning EMA Cross — ATR-fixed SL variant (no yellow ratchet).

Same entry rules as strategy.py:
  - Entry at 10:00 ET (30-min bar close)
  - LONG if close > EMA, SHORT if close < EMA

Exit rules (DIFFERENT from yellow ratchet):
  - Fixed SL set at entry: entry_price - sign * sl_mult * ATR(entry)
  - SL fires INTRABAR (low <= sl for LONG, high >= sl for SHORT)
  - Force close at 14:30 ET

Sweep:
  - sl_mult: 0.25 -> 3.00 step 0.10 (28 values)
  - atr_period: 7, 10, 14, 21, 28 (5 values)
  - EMA fixed at 60
Total: 28 × 5 = 140 cells. IS/OOS 60/40 split.
Martingale: OD-style (any loss → 2c next, then back to 1c).
"""
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
import pandas as pd

PARQUET = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
ET = "America/New_York"
NQ_PT = 20.0

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMA_FIXED = 60  # winner-zone from prior sweep
SL_GRID = [round(x, 2) for x in np.arange(0.25, 3.001, 0.10).tolist()]
ATR_GRID = [7, 10, 14, 21, 28]


def build_30min_bars():
    df = pd.read_parquet(PARQUET)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    bars = df.resample("30min", label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum",
    }).dropna(subset=["open"])
    return bars


def compute_indicators(bars: pd.DataFrame, ema_n: int, atr_n: int) -> pd.DataFrame:
    b = bars.copy()
    b["ema"] = b["close"].ewm(span=ema_n, adjust=False).mean()
    prev_close = b["close"].shift(1)
    tr = pd.concat([
        b["high"] - b["low"],
        (b["high"] - prev_close).abs(),
        (b["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    b["atr"] = tr.ewm(alpha=1.0/atr_n, adjust=False).mean()
    b["hhmm"] = b.index.hour * 100 + b.index.minute
    b["dow"] = b.index.dayofweek
    b["date"] = b.index.date
    return b


def run_strategy(b: pd.DataFrame, sl_mult: float,
                 use_marti: bool = True, base_qty: int = 1, loss_qty: int = 2) -> pd.DataFrame:
    trades = []
    marti_state = 0
    for date, day in b.groupby("date"):
        if day["dow"].iloc[0] >= 5:
            continue
        entry_bar = day[day["hhmm"] == 1000]
        if len(entry_bar) == 0: continue
        eb = entry_bar.iloc[0]
        if pd.isna(eb["ema"]) or pd.isna(eb["atr"]) or eb["atr"] <= 0:
            continue
        entry_price = float(eb["close"])
        entry_atr   = float(eb["atr"])
        if entry_price > eb["ema"]:   sign = 1
        elif entry_price < eb["ema"]: sign = -1
        else: continue

        qty = loss_qty if (use_marti and marti_state == 1) else base_qty
        sl_price = entry_price - sign * sl_mult * entry_atr

        post = day[(day["hhmm"] > 1000) & (day["hhmm"] <= 1430)]
        if len(post) == 0: continue

        exit_price = None
        exit_reason = None
        exit_time = None
        bars_held = 0
        for ts, bar in post.iterrows():
            bars_held += 1
            high = float(bar["high"])
            low  = float(bar["low"])
            close = float(bar["close"])

            # Force close at 14:30 (priority — clean bookkeeping)
            if int(bar["hhmm"]) == 1430:
                exit_price = close; exit_reason = "FC"; exit_time = ts; break

            # SL intrabar
            if sign > 0 and low <= sl_price:
                exit_price = sl_price; exit_reason = "SL"; exit_time = ts; break
            if sign < 0 and high >= sl_price:
                exit_price = sl_price; exit_reason = "SL"; exit_time = ts; break

        if exit_price is None: continue
        pnl_pts = sign * (exit_price - entry_price)
        pnl_usd = pnl_pts * NQ_PT * qty
        trades.append({
            "date": date, "entry_time": eb.name, "exit_time": exit_time,
            "direction": "LONG" if sign > 0 else "SHORT",
            "entry_price": entry_price, "exit_price": exit_price,
            "entry_atr": entry_atr, "sl_price": sl_price,
            "qty": qty, "bars_held": bars_held,
            "pnl_pts": pnl_pts, "pnl_$": pnl_usd, "reason": exit_reason,
        })
        if use_marti:
            last_was_loss = exit_price * sign < entry_price * sign
            if marti_state == 0:
                marti_state = 1 if last_was_loss else 0
            elif marti_state == 1:
                marti_state = 2
            else:
                marti_state = 1 if last_was_loss else 0
    return pd.DataFrame(trades)


def stats_block(pnls):
    n = len(pnls)
    if n == 0: return dict(n=0, wr=0, net=0, pf=0, mdd=0)
    w = pnls[pnls > 0]; l = pnls[pnls < 0]
    pf = w.sum() / abs(l.sum()) if len(l) > 0 else 99.0
    cum = pnls.cumsum()
    mdd = float((cum - np.maximum.accumulate(cum)).min())
    return dict(n=n, wr=round(len(w)/n*100, 1), net=round(float(pnls.sum()), 0),
                pf=round(pf, 3), mdd=round(mdd, 0))


_BARS = None


def _init_worker(bars):
    global _BARS
    _BARS = bars


def _run_cell(args):
    atr_n, sl_mult = args
    b = compute_indicators(_BARS, EMA_FIXED, atr_n)
    trades = run_strategy(b, sl_mult, use_marti=True)
    if len(trades) == 0:
        return {"atr_n": atr_n, "sl_mult": sl_mult, "n_all": 0}
    trades = trades.sort_values("entry_time").reset_index(drop=True)
    pnls = trades["pnl_$"].values
    dates = sorted(trades["date"].unique())
    cutoff = dates[int(len(dates) * 0.6)] if len(dates) > 1 else dates[-1]
    is_mask  = trades["date"] <  cutoff
    oos_mask = trades["date"] >= cutoff
    s_all = stats_block(pnls)
    s_is  = stats_block(trades.loc[is_mask,  "pnl_$"].values)
    s_oos = stats_block(trades.loc[oos_mask, "pnl_$"].values)
    return {
        "atr_n": atr_n, "sl_mult": sl_mult,
        "n_all": s_all["n"], "wr_all": s_all["wr"], "all_net": s_all["net"],
        "all_PF": s_all["pf"], "all_mdd": s_all["mdd"],
        "is_n": s_is["n"], "is_net": s_is["net"], "is_PF": s_is["pf"],
        "oos_n": s_oos["n"], "oos_net": s_oos["net"], "oos_PF": s_oos["pf"],
    }


def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading bars + resampling to 30-min...")
    bars = build_30min_bars()
    print(f"  {len(bars):,} bars  ({time.time()-t0:.1f}s)\n  EMA fixed at {EMA_FIXED}")

    configs = [(atr_n, sl) for atr_n in ATR_GRID for sl in SL_GRID]
    print(f"\n[{time.strftime('%H:%M:%S')}] Sweeping {len(configs)} cells on 6 workers...")
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=6, initializer=_init_worker, initargs=(bars,)) as ex:
        rows = list(ex.map(_run_cell, configs))
    print(f"[{time.strftime('%H:%M:%S')}] Done in {time.time()-t1:.1f}s")

    df = pd.DataFrame(rows)
    df["beat_both"] = (df["is_net"] > 0) & (df["oos_net"] > 0)
    df["net_mdd_ratio"] = df["all_net"] / df["all_mdd"].abs()
    df.to_csv(OUT_DIR / "morning_ema_cross_atr_sl_sweep.csv", index=False)

    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)

    show = ["atr_n", "sl_mult", "n_all", "wr_all", "all_net", "all_PF",
            "all_mdd", "is_net", "is_PF", "oos_net", "oos_PF", "beat_both"]

    print(f"\n=== TOP 15 by ALL NET $ (must beat both IS+OOS) ===")
    winners = df[df["beat_both"]].sort_values("all_net", ascending=False).head(15)
    if len(winners) == 0:
        print("  NONE")
    else:
        print(winners[show].to_string(index=False))

    print(f"\n=== HEATMAP: ALL NET $ by (atr_n × sl_mult) [positive cells only] ===")
    # Pivot
    pivot = df.pivot(index="atr_n", columns="sl_mult", values="all_net")
    print(pivot.fillna(0).astype(int).to_string())

    print(f"\n=== HEATMAP: OOS NET $ (must beat both) ===")
    df_oos = df[df["beat_both"]].copy()
    pivot_oos = df_oos.pivot(index="atr_n", columns="sl_mult", values="oos_net")
    print(pivot_oos.fillna("").to_string())

    print(f"\n=== Positive IS+OOS count by atr_n ===")
    for atr_n in ATR_GRID:
        sub = df[df["atr_n"] == atr_n]
        n_pos = sub["beat_both"].sum()
        best = sub[sub["beat_both"]]["all_net"].max() if n_pos > 0 else 0
        print(f"  atr_n={atr_n}: {n_pos}/{len(sub)} pass both, best NET ${best:,.0f}")

    print(f"\n=== TOP 5 by NET/|MDD| RATIO ===")
    ra = df[df["beat_both"]].sort_values("net_mdd_ratio", ascending=False).head(5)
    print(ra[show + ["net_mdd_ratio"]].to_string(index=False))

    # Save best trades
    if df["beat_both"].any():
        best = df[df["beat_both"]].sort_values("all_net", ascending=False).iloc[0]
        print(f"\n=== BEST CONFIG: ATR={int(best['atr_n'])}, SL_MULT={best['sl_mult']} ===")
        b = compute_indicators(bars, EMA_FIXED, int(best["atr_n"]))
        best_trades = run_strategy(b, best["sl_mult"], use_marti=True)
        best_trades.to_csv(OUT_DIR / "morning_ema_atr_sl_best_trades.csv", index=False)
        print(f"  Saved {len(best_trades)} trades")


if __name__ == "__main__":
    main()
