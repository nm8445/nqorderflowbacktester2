"""Morning EMA Cross strategy on 30-min NQ bars.

Rules:
  - Entry at 10:00 ET (close of the 09:30-10:00 30-min bar).
    - If close > EMA(N) -> LONG
    - If close < EMA(N) -> SHORT
  - SL: yellow ratchet. For LONG, yellow = close - YMULT*ATR (ratchets UP only).
    For SHORT, yellow = close + YMULT*ATR (ratchets DOWN only).
    Exit: bar close past yellow AND reversal candle.
  - No TP.
  - Force-close at 14:30 ET if SL hasn't fired.
  - Martingale: OD-style (any loss → 2c next, then back to 1c regardless).

Optimization: sweep EMA period and yellow ATR mult, IS/OOS 60/40 split.
"""
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
import pandas as pd

PARQUET = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
ET = "America/New_York"
NQ_PT = 20.0  # $/point/contract
ATR_LEN = 14

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_30min_bars(parquet_path: str = PARQUET) -> pd.DataFrame:
    """Resample 1-min UTC bars to 30-min ET bars (all-hours continuous)."""
    df = pd.read_parquet(parquet_path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    bars = df.resample("30min", label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum",
    }).dropna(subset=["open"])
    return bars


def compute_indicators(bars: pd.DataFrame, ema_n: int, atr_n: int = ATR_LEN) -> pd.DataFrame:
    """Add EMA and ATR (Wilder/RMA) columns."""
    b = bars.copy()
    b["ema"] = b["close"].ewm(span=ema_n, adjust=False).mean()
    prev_close = b["close"].shift(1)
    tr = pd.concat([
        b["high"] - b["low"],
        (b["high"] - prev_close).abs(),
        (b["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Wilder/RMA ATR (matches Pine ta.atr)
    b["atr"] = tr.ewm(alpha=1.0/atr_n, adjust=False).mean()
    b["hhmm"] = b.index.hour * 100 + b.index.minute
    b["dow"] = b.index.dayofweek
    b["date"] = b.index.date
    return b


def run_strategy(b: pd.DataFrame, ymult: float, use_marti: bool = True,
                 base_qty: int = 1, loss_qty: int = 2) -> pd.DataFrame:
    """Run the strategy on indicator-ready 30-min bars. Returns trades DataFrame.

    Martingale (OD-style state machine):
      state 0 (1c base): if loss -> state 1, if win -> stay 0
      state 1 (2c)     : always -> state 2 next
      state 2 (1c)     : if loss -> state 1, if win -> stay 0
    """
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
        if entry_price > eb["ema"]:
            sign = 1   # LONG
        elif entry_price < eb["ema"]:
            sign = -1  # SHORT
        else:
            continue

        # Position size from martingale state
        qty = loss_qty if (use_marti and marti_state == 1) else base_qty

        # Initialize yellow at entry bar
        yellow = entry_price - sign * ymult * entry_atr

        post = day[(day["hhmm"] > 1000) & (day["hhmm"] <= 1430)]
        if len(post) == 0: continue

        exit_price = None
        exit_reason = None
        exit_time = None
        bars_held = 0

        for ts, bar in post.iterrows():
            bars_held += 1
            close = float(bar["close"])
            open_  = float(bar["open"])
            atr_b = float(bar["atr"])

            # Update yellow (ratchets only in favorable direction)
            raw_yellow = close - sign * ymult * atr_b
            if sign > 0:
                yellow = max(yellow, raw_yellow)
            else:
                yellow = min(yellow, raw_yellow)

            # Force close at 14:30 (priority over yellow for clean bookkeeping)
            if int(bar["hhmm"]) == 1430:
                exit_price = close; exit_reason = "FC"; exit_time = ts; break

            # Exit: close past yellow + reversal candle
            if sign > 0 and close <= yellow and close < open_:
                exit_price = close; exit_reason = "SL"; exit_time = ts; break
            if sign < 0 and close >= yellow and close > open_:
                exit_price = close; exit_reason = "SL"; exit_time = ts; break

        if exit_price is None:
            continue   # shouldn't happen if 14:30 bar exists

        pnl_pts = sign * (exit_price - entry_price)
        pnl_usd = pnl_pts * NQ_PT * qty
        trades.append({
            "date": date,
            "entry_time": eb.name,
            "exit_time": exit_time,
            "direction": "LONG" if sign > 0 else "SHORT",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_atr": entry_atr,
            "yellow_final": yellow,
            "qty": qty,
            "bars_held": bars_held,
            "pnl_pts": pnl_pts,
            "pnl_$": pnl_usd,
            "reason": exit_reason,
        })

        # Update martingale state
        if use_marti:
            last_was_loss = exit_price * sign < entry_price * sign  # signed loss
            if marti_state == 0:
                marti_state = 1 if last_was_loss else 0
            elif marti_state == 1:
                marti_state = 2  # forced cooldown
            else:
                marti_state = 1 if last_was_loss else 0

    return pd.DataFrame(trades)


def stats_block(pnls: np.ndarray) -> dict:
    n = len(pnls)
    if n == 0: return dict(n=0, wr=0, net=0, pf=0, mdd=0, avg=0)
    w = pnls[pnls > 0]; l = pnls[pnls < 0]
    pf = w.sum() / abs(l.sum()) if len(l) > 0 else 99.0
    cum = pnls.cumsum()
    mdd = float((cum - np.maximum.accumulate(cum)).min())
    return dict(
        n=n, wr=round(len(w)/n*100, 1), net=round(float(pnls.sum()), 0),
        pf=round(pf, 3), mdd=round(mdd, 0), avg=round(float(pnls.mean()), 1),
    )


# === parallel sweep ===

_BARS = None


def _init_worker(bars):
    global _BARS
    _BARS = bars


def _run_cell(args):
    ema_n, ymult = args
    b = compute_indicators(_BARS, ema_n)
    trades = run_strategy(b, ymult, use_marti=True)
    if len(trades) == 0:
        return {"ema_n": ema_n, "ymult": ymult, "n_all": 0}

    trades = trades.sort_values("entry_time").reset_index(drop=True)
    pnls = trades["pnl_$"].values

    # 60/40 chronological split on dates
    dates = sorted(trades["date"].unique())
    cutoff_idx = int(len(dates) * 0.6)
    cutoff = dates[cutoff_idx] if cutoff_idx < len(dates) else dates[-1]
    is_mask  = trades["date"] <  cutoff
    oos_mask = trades["date"] >= cutoff

    s_all = stats_block(pnls)
    s_is  = stats_block(trades.loc[is_mask,  "pnl_$"].values)
    s_oos = stats_block(trades.loc[oos_mask, "pnl_$"].values)

    return {
        "ema_n": ema_n, "ymult": ymult,
        "n_all": s_all["n"], "wr_all": s_all["wr"], "all_net": s_all["net"],
        "all_PF": s_all["pf"], "all_mdd": s_all["mdd"], "all_avg": s_all["avg"],
        "is_n": s_is["n"], "is_net": s_is["net"], "is_PF": s_is["pf"], "is_wr": s_is["wr"],
        "oos_n": s_oos["n"], "oos_net": s_oos["net"], "oos_PF": s_oos["pf"], "oos_wr": s_oos["wr"],
        "oos_mdd": s_oos["mdd"],
    }


def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading 1-min bars and resampling to 30-min...")
    bars = build_30min_bars()
    print(f"[{time.strftime('%H:%M:%S')}]   {len(bars):,} 30-min bars, "
          f"{bars.index.min()} -> {bars.index.max()}  ({time.time()-t0:.1f}s)")

    EMA_GRID = [50, 60, 70, 80, 90, 100, 110, 120]
    YMULT_GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    configs = [(n, y) for n in EMA_GRID for y in YMULT_GRID]
    print(f"\n[{time.strftime('%H:%M:%S')}] Sweeping {len(configs)} cells on 6 workers...")

    t1 = time.time()
    with ProcessPoolExecutor(max_workers=6, initializer=_init_worker, initargs=(bars,)) as ex:
        rows = list(ex.map(_run_cell, configs))
    print(f"[{time.strftime('%H:%M:%S')}] Done in {time.time()-t1:.1f}s")

    df = pd.DataFrame(rows)
    df["net_mdd_ratio"] = df["all_net"] / df["all_mdd"].abs()
    df["beat_both"] = (df["is_net"] > 0) & (df["oos_net"] > 0)
    df.to_csv(OUT_DIR / "morning_ema_cross_sweep.csv", index=False)

    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)

    print(f"\n=== TOP 15 by ALL NET $ (any sign) ===")
    show_cols = ["ema_n", "ymult", "n_all", "wr_all", "all_net", "all_PF",
                 "all_mdd", "is_net", "is_PF", "oos_net", "oos_PF", "beat_both"]
    print(df.sort_values("all_net", ascending=False).head(15)[show_cols].to_string(index=False))

    print(f"\n=== CONFIGS POSITIVE in BOTH IS and OOS (n={df['beat_both'].sum()}/{len(df)}) ===")
    winners = df[df["beat_both"]].sort_values("all_net", ascending=False)
    if len(winners) == 0:
        print("  NONE.")
    else:
        print(winners[show_cols].to_string(index=False))

    print(f"\n=== TOP 5 by NET/|MDD| RATIO (risk-adjusted, must beat both) ===")
    ra = df[df["beat_both"]].sort_values("net_mdd_ratio", ascending=False).head(5)
    print(ra[show_cols + ["net_mdd_ratio"]].to_string(index=False))

    # Pick best config and save its full trade log
    if df["beat_both"].any():
        best = df[df["beat_both"]].sort_values("all_net", ascending=False).iloc[0]
        print(f"\n=== BEST CONFIG: EMA={int(best['ema_n'])}, YMULT={best['ymult']} ===")
        print(f"  ALL: ${best['all_net']:,.0f}  PF {best['all_PF']}  MDD ${best['all_mdd']:,.0f}  WR {best['wr_all']}%")
        print(f"  IS:  ${best['is_net']:,.0f}  PF {best['is_PF']}")
        print(f"  OOS: ${best['oos_net']:,.0f}  PF {best['oos_PF']}")
        b = compute_indicators(bars, int(best["ema_n"]))
        best_trades = run_strategy(b, best["ymult"], use_marti=True)
        best_trades.to_csv(OUT_DIR / "morning_ema_cross_best_trades.csv", index=False)
        print(f"  Saved trades: {OUT_DIR / 'morning_ema_cross_best_trades.csv'} ({len(best_trades)} trades)")


if __name__ == "__main__":
    main()
