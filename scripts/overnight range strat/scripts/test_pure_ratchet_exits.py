"""Apply the overnight drift strategy's pure_ratchet yellow + green-band exits
to the entries from our 5-min B2 strategy.

Pipeline:
  - Entries / confirmations: existing 5-min bar logic (B2 X=0.75 N=15 D=70 strict
    BAND_K=0.25, conf_N=5 conf_D=75 HALF, chained Mode 1)
  - Stop management: 20-min bars
  - Yellow stop: close - 1.42 * ATR_14 (LONG) or close + 1.42 * ATR_14 (SHORT)
                 pure_ratchet — never moves against position
  - Green target: red + green_base - green_decay * bars_in_trade + 2.60 * ATR_13
                  red = entry + 0.45 * bars_in_trade (LONG)  [intercept=0, drift=0.45]
                  green decays toward spot every 20-min bar
  - Exit priority (per 20-min bar close):
      1) TP Green: high >= green_val (LONG) or low <= green_val (SHORT) -> exit at close
      2) SL Yellow: close beyond yellow AND bar is adverse-direction -> exit at close
      3) Force close at RTH end (16:00 ET) -> exit at close
  - Mirroring for SHORT: signs flipped
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from range_break_entry_summary import apply_filters, mode1_chained_dedupe

PARQUET_DIR  = Path(__file__).parent / "parquets"
TRADELOG_DIR = Path(__file__).parent.parent / "tradelogs"
OUT_TXT      = TRADELOG_DIR / "robust_configs" / "pure_ratchet_exits.txt"

NQ_1MIN = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")

# ---- locked entry config (977-trade variant) ----
VARIANT, X, N, D, STRICT, BAND_K = "B2", 0.75, 15, 70, True, 0.25
CONF_N, CONF_D = 5, 75   # 977-trade config
TP_M, SL_M = 1.0, 1.0    # only used to look up tp/sl idx for chained-mode dedupe

# ---- ratchet exit params (drift strategy defaults) ----
YELLOW_ATR_LEN  = 14
YELLOW_ATR_MULT = 1.42
GREEN_ATR_LEN   = 13
GREEN_ATR_MULT  = 2.60
GREEN_BASE      = 107.6
GREEN_DECAY     = 1.31
RED_INTERCEPT   = 0.0
RED_DRIFT       = 0.45
FORCE_CLOSE_TIME = dt.time(16, 0)   # RTH end (force-close at 16:00 ET)


# --------- ATR (Pine RMA) ---------
def rma_atr(high, low, close, length):
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(),
         (high - prev_close).abs(),
         (low  - prev_close).abs()], axis=1).max(axis=1)
    atr = pd.Series(np.nan, index=close.index, dtype=float)
    if len(tr) < length:
        return atr
    seed = tr.iloc[:length].mean()
    atr.iloc[length - 1] = seed
    alpha = 1.0 / length
    prev = seed
    out = atr.values
    for i in range(length, len(tr)):
        cur = tr.iloc[i]
        if np.isnan(cur):
            out[i] = prev; continue
        prev = (1 - alpha) * prev + alpha * cur
        out[i] = prev
    return atr


def build_20min_bars():
    """Build 20-min OHLC bars from 1-min markettick, ETH session range."""
    print("loading 1-min bars + resampling to 20-min...")
    df = pd.read_parquet(NQ_1MIN)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")
    df = df.sort_index()
    # 1-min timestamps are bar CLOSE times, so a 1-min bar at 09:31:00 covers 09:30:00-09:31:00.
    # For 20-min bars labeled by bar OPEN, anchor to 09:30 ET so first RTH 20-min = 09:30-09:50.
    # Build 20-min bars over the full ETH session (any timezone-aware ts).
    bars = (df["close"].resample("20min", label="left", closed="left").last().rename("close").to_frame()
            .assign(open  = df["open"] .resample("20min", label="left", closed="left").first(),
                    high  = df["high"] .resample("20min", label="left", closed="left").max(),
                    low   = df["low"]  .resample("20min", label="left", closed="left").min(),
                    volume= df["volume"].resample("20min", label="left", closed="left").sum())
            .dropna(subset=["open","high","low","close"]))
    bars = bars[["open","high","low","close","volume"]]
    print(f"  built {len(bars):,} 20-min bars  range {bars.index.min()} -> {bars.index.max()}")

    # Compute ATRs on 20-min bars
    bars["atr_y"] = rma_atr(bars["high"], bars["low"], bars["close"], YELLOW_ATR_LEN)
    bars["atr_g"] = rma_atr(bars["high"], bars["low"], bars["close"], GREEN_ATR_LEN)
    return bars


def filter_to_entries(df: pd.DataFrame) -> pd.DataFrame:
    f = apply_filters(df, VARIANT, X, N, D, STRICT, BAND_K)
    col = f"conf_delta_half_w{CONF_N}"
    cf = f[((f["direction"]=="LONG")  & (f[col].notna()) & (f[col] >=  CONF_D)) |
           ((f["direction"]=="SHORT") & (f[col].notna()) & (f[col] <= -CONF_D))]
    ded = mode1_chained_dedupe(cf, TP_M, SL_M)
    return ded.copy() if not ded.empty else ded


def simulate_ratchet_exit(direction: str, entry_ts: pd.Timestamp,
                           entry_price: float, bars20: pd.DataFrame
                           ) -> tuple[float, float, str, int]:
    """Simulate one trade's exit with pure_ratchet yellow + green target on 20-min bars.
    Returns (exit_price, exit_pnl_pts, exit_reason, bars_held).
    LONG: pnl = exit - entry  ;  SHORT: pnl = entry - exit
    """
    sign = 1 if direction == "LONG" else -1
    # Find first 20-min bar STRICTLY AFTER entry_ts (bar where ratchet starts)
    bars_after = bars20[bars20.index > entry_ts]
    if bars_after.empty:
        return (entry_price, 0.0, "NO_DATA", 0)

    # Day boundary: only manage within same day (RTH)
    entry_day = entry_ts.date()
    bars_today = bars_after[bars_after.index.date == entry_day]
    if bars_today.empty:
        return (entry_price, 0.0, "NO_DATA", 0)

    # Initial yellow stop using entry-bar's ATR (last bar AT or BEFORE entry_ts)
    init_bars = bars20[bars20.index <= entry_ts]
    if init_bars.empty or pd.isna(init_bars.iloc[-1]["atr_y"]):
        return (entry_price, 0.0, "NO_DATA", 0)
    init_atr_y = float(init_bars.iloc[-1]["atr_y"])
    yellow_val = entry_price - sign * YELLOW_ATR_MULT * init_atr_y
    prev_yellow = yellow_val
    prev_close  = entry_price

    for bars_in_trade, (ts, bar) in enumerate(bars_today.iterrows(), 1):
        o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
        ay, ag = bar["atr_y"], bar["atr_g"]

        # Yellow ratchet (pure_ratchet): never moves against position
        if not np.isnan(ay):
            raw_yellow = c - sign * YELLOW_ATR_MULT * ay
            if sign > 0:  # LONG: yellow ratchets up
                yellow_val = max(prev_yellow, raw_yellow)
            else:         # SHORT: yellow ratchets down
                yellow_val = min(prev_yellow, raw_yellow)

        # Green target with decay
        red_val = entry_price + sign * (RED_INTERCEPT + RED_DRIFT * bars_in_trade)
        green_offset = (GREEN_BASE - GREEN_DECAY * bars_in_trade
                        + (GREEN_ATR_MULT * ag if not np.isnan(ag) else 0.0))
        green_val = red_val + sign * green_offset

        # 1) TP Green: did the bar reach the target?
        if sign > 0 and h >= green_val:
            return (c, c - entry_price, "TP_GREEN", bars_in_trade)
        if sign < 0 and l <= green_val:
            return (c, entry_price - c, "TP_GREEN", bars_in_trade)

        # 2) SL Yellow on close (adverse-direction bar)
        if sign > 0:
            if c <= yellow_val and c < o:
                return (c, c - entry_price, "SL_YELLOW", bars_in_trade)
        else:
            if c >= yellow_val and c > o:
                return (c, entry_price - c, "SL_YELLOW", bars_in_trade)

        # 3) Force close at RTH end
        if ts.time() >= FORCE_CLOSE_TIME:
            return (c, sign * (c - entry_price), "FORCE_CLOSE", bars_in_trade)

        prev_yellow = yellow_val
        prev_close  = c

    # Ran out of bars in day with no force close — use last close
    last_bar = bars_today.iloc[-1]
    return (last_bar["close"], sign * (last_bar["close"] - entry_price),
            "EOD", len(bars_today))


def stats(df: pd.DataFrame, pnl_col: str) -> dict:
    if df.empty:
        return {"n":0, "total":0.0, "wr":float("nan"), "pf":float("nan"),
                "sharpe":float("nan"), "max_dd":0.0, "n_long":0, "n_short":0,
                "long_total":0.0, "short_total":0.0}
    pnl = df[pnl_col].values
    pos = pnl[pnl > 0].sum(); neg = -pnl[pnl < 0].sum()
    pf = pos / neg if neg > 0 else (np.inf if pos > 0 else 0.0)
    daily = pd.Series(pnl, index=df["date"].values).groupby(level=0).sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0
    eq = np.cumsum(pnl); peak = np.maximum.accumulate(eq); max_dd = (eq - peak).min()
    long_mask = (df["direction"]=="LONG").values
    short_mask = (df["direction"]=="SHORT").values
    return {"n":len(df), "total":pnl.sum(), "wr":(pnl > 0).mean(),
            "pf":pf, "sharpe":sharpe, "max_dd":max_dd,
            "n_long":int(long_mask.sum()), "n_short":int(short_mask.sum()),
            "long_total":pnl[long_mask].sum(), "short_total":pnl[short_mask].sum()}


def fmt(label, s):
    pf = f"{s['pf']:>5.2f}" if np.isfinite(s["pf"]) else "  inf"
    return (f"  {label:<22}  n={s['n']:>4}  L={s['n_long']:>3}/S={s['n_short']:>3}  "
            f"total={s['total']:>+8.1f}  L_t={s['long_total']:>+7.1f}  S_t={s['short_total']:>+7.1f}  "
            f"WR={s['wr']:>5.1%}  PF={pf}  Sh={s['sharpe']:>+5.2f}  MDD={s['max_dd']:>+7.0f}")


def main():
    print("=" * 100)
    print("PURE_RATCHET STOP MANAGEMENT TEST  (977-trade entry config)")
    print("=" * 100)

    bars20 = build_20min_bars()

    print("\nfiltering trades (locked entry config)...")
    is_df  = pd.read_parquet(PARQUET_DIR / "entry_signal_trades.parquet")
    oos_df = pd.read_parquet(PARQUET_DIR / "entry_signal_trades_oos.parquet")
    is_ded  = filter_to_entries(is_df)
    oos_ded = filter_to_entries(oos_df)
    print(f"  IS:  {len(is_ded):,} trades")
    print(f"  OOS: {len(oos_ded):,} trades")

    print("\nsimulating pure_ratchet exits per trade...")
    for label, ded in [("IS", is_ded), ("OOS", oos_ded)]:
        ded["pnl_ratchet"] = np.nan
        ded["exit_reason"] = ""
        ded["bars_held_20m"] = 0
        for ix, t in ded.iterrows():
            ts = pd.to_datetime(t["entry_time"])
            if ts.tz is None:
                ts = ts.tz_localize("America/New_York")
            else:
                ts = ts.tz_convert("America/New_York")
            ep = float(t["entry_price"])
            exit_p, pnl, reason, bh = simulate_ratchet_exit(
                t["direction"], ts, ep, bars20)
            ded.at[ix, "pnl_ratchet"] = pnl
            ded.at[ix, "exit_reason"] = reason
            ded.at[ix, "bars_held_20m"] = bh
        ded["date"] = pd.to_datetime(ded["date"]).dt.date

    # Compute baseline (locked symmetric 1xATR)
    from range_break_entry_summary import trade_pnls_vectorized
    is_ded["pnl_baseline"]  = trade_pnls_vectorized(is_ded,  TP_M, SL_M)
    oos_ded["pnl_baseline"] = trade_pnls_vectorized(oos_ded, TP_M, SL_M)
    is_ded["date"]  = pd.to_datetime(is_ded["date"]).dt.date
    oos_ded["date"] = pd.to_datetime(oos_ded["date"]).dt.date

    lines = []
    lines.append("=" * 200)
    lines.append("PURE_RATCHET EXIT MANAGEMENT — vs symmetric 1xATR baseline")
    lines.append("=" * 200)
    lines.append("")
    lines.append("Entry config (5-min logic):  B2 X=0.75 N=15 D=70 strict BAND_K=0.25")
    lines.append(f"                              + conf_N={CONF_N} conf_D={CONF_D} HALF, chained Mode 1")
    lines.append("Exit management (20-min):    pure_ratchet yellow + green decay target")
    lines.append(f"  Yellow: close - {YELLOW_ATR_MULT}*ATR_{YELLOW_ATR_LEN} (signed); pure ratchet (never moves against)")
    lines.append(f"  Green:  entry + {RED_DRIFT}*bars + {GREEN_BASE} - {GREEN_DECAY}*bars + {GREEN_ATR_MULT}*ATR_{GREEN_ATR_LEN}")
    lines.append(f"  Force close at {FORCE_CLOSE_TIME} ET")
    lines.append("")

    for label, ded in [("IN-SAMPLE",  is_ded), ("OUT-OF-SAMPLE", oos_ded)]:
        lines.append("=" * 200)
        lines.append(f"{label}  (n={len(ded):,})")
        lines.append("=" * 200)
        s_base = stats(ded, "pnl_baseline")
        s_ratch = stats(ded, "pnl_ratchet")
        lines.append(fmt("BASELINE 1xATR",     s_base))
        lines.append(fmt("PURE_RATCHET",       s_ratch))
        lines.append("")
        # exit reason breakdown for ratchet
        lines.append("  Exit reason breakdown (ratchet):")
        rc = ded.groupby("exit_reason").agg(n=("pnl_ratchet","size"),
                                             total=("pnl_ratchet","sum"),
                                             wr=("pnl_ratchet", lambda x: (x>0).mean()),
                                             mean=("pnl_ratchet","mean"),
                                             median_bars=("bars_held_20m","median"))
        for reason, row in rc.iterrows():
            lines.append(f"    {reason:<14}  n={int(row['n']):>4}  total={row['total']:>+8.1f}  "
                         f"mean={row['mean']:>+5.2f}  WR={row['wr']:>5.1%}  med_bars={int(row['median_bars']):>2}")
        lines.append("")

    # Combined
    combined = pd.concat([is_ded, oos_ded], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.date
    s_base_c = stats(combined, "pnl_baseline")
    s_ratch_c = stats(combined, "pnl_ratchet")
    lines.append("=" * 200)
    lines.append(f"COMBINED IS+OOS  (n={len(combined):,})")
    lines.append("=" * 200)
    lines.append(fmt("BASELINE 1xATR",     s_base_c))
    lines.append(fmt("PURE_RATCHET",       s_ratch_c))

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
