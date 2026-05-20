"""
Backtest: RTH zone-break entries.

BUY break  = bullish candle closing above an active RESISTANCE zone's zone_high
SELL break = bearish candle closing below an active SUPPORT zone's zone_low

Filters:
  body_ratio = |close - open| / (high - low) >= 0.70
  |bar_total_delta| >= STRONG_DELTA  (sweep)
  session = RTH (09:30-16:59 ET)

Entry = break candle close, 0.125 pt adverse slippage
BUY:  SL = break.bot_wick_low - 1*ATR14(5min);   TP = entry + (entry-SL) + 1*ATR
SELL: SL = break.top_wick_high + 1*ATR14(5min);  TP = entry - (SL-entry) - 1*ATR
Costs: $4.50 fixed RT + 0.125 pt slippage both sides.
Exit: SL / TP first hit intraday, else force-close at 16:59 close.

Outputs:
  results_zone_break/
    sweep_summary.csv       - trade count + headline by STRONG_DELTA per direction
    trades_buy_t200.csv     - full trade rows at default threshold
    trades_sell_t200.csv
    breakdowns.txt          - entry-hour, dev_loc, noise_loc, week_loc,
                              bot_wick sign, top_wick sign (at threshold=200)
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("D:/trading_pythonbacktest_data")
OUT_DIR = Path(__file__).resolve().parent / "results_zone_break"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ET = "America/New_York"

BODY_RATIO_MIN = 0.70
ATR_LEN = 14
ATR_MULT_SL = 1.0
ATR_MULT_TP = 1.0

POINT_VALUE = 20.0
SLIPPAGE_PTS = 0.125
FIXED_COST_RT = 2 * (0.85 + 1.40)   # $4.50

STRONG_DELTA_SWEEP = list(range(200, 1001, 100))
DETAIL_THRESHOLD = 200


# ── helpers ─────────────────────────────────────────────────────────────────
def localize_to_et(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    if s.dt.tz is None: return s.dt.tz_localize("UTC").dt.tz_convert(ET)
    return s.dt.tz_convert(ET)


def metrics(df):
    if df.empty: return {"n": 0, "wr": 0, "pf": float("nan"), "mean": 0, "total": 0}
    w = df[df["pnl_net"] > 0]; l = df[df["pnl_net"] < 0]
    pf = w["pnl_net"].sum() / abs(l["pnl_net"].sum()) if len(l) and l["pnl_net"].sum() != 0 else float("inf")
    return {"n": len(df), "wr": 100*len(w)/len(df), "pf": pf,
            "mean": df["pnl_net"].mean(), "total": df["pnl_net"].sum(),
            "avg_win": w["pnl_net"].mean() if len(w) else 0,
            "avg_loss": l["pnl_net"].mean() if len(l) else 0}


# ── build break-candidate table ─────────────────────────────────────────────
def build_breaks():
    print("Loading caches...", flush=True)
    idx = pd.read_parquet(DATA_DIR / "wick_delta_index_5min.parquet")
    idx["bar_open_time"] = localize_to_et(idx["bar_open_time"])
    idx = idx.sort_values("bar_open_time").reset_index(drop=True)

    # ATR
    prev_close = idx["close"].shift(1)
    tr = pd.concat([
        idx["high"] - idx["low"],
        (idx["high"] - prev_close).abs(),
        (idx["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    idx["atr14"] = tr.rolling(ATR_LEN, min_periods=ATR_LEN).mean()
    # body ratio
    rng = idx["high"] - idx["low"]
    idx["body_ratio"] = (idx["close"] - idx["open"]).abs() / rng.where(rng > 0, np.nan)

    zones = pd.read_parquet(DATA_DIR / "wick_zones_5min.parquet")
    zones["invalidation_time"] = localize_to_et(zones["invalidation_time"])

    rth_breaks = zones[zones["invalidation_reason"] == "rth_break"].copy()
    print(f"  RTH-broken zones: {len(rth_breaks):,}")

    # Break events keyed by (bar_open_time, direction)
    buy_b = rth_breaks[rth_breaks["zone_type"] == "resistance"][
        ["zone_id", "zone_low", "zone_high", "invalidation_time"]].rename(columns={"invalidation_time": "break_time"})
    sell_b = rth_breaks[rth_breaks["zone_type"] == "support"][
        ["zone_id", "zone_low", "zone_high", "invalidation_time"]].rename(columns={"invalidation_time": "break_time"})

    # Dedupe to one signal per (break_time, direction) — keep first zone encountered
    buy_b = buy_b.drop_duplicates(subset=["break_time"]).reset_index(drop=True)
    sell_b = sell_b.drop_duplicates(subset=["break_time"]).reset_index(drop=True)
    print(f"  unique buy-break bars: {len(buy_b):,}")
    print(f"  unique sell-break bars: {len(sell_b):,}")

    # Join to index for break candle stats
    idx_keyed = idx.set_index("bar_open_time")
    def join(b, dirc):
        b = b.copy()
        b = b.rename(columns={"break_time": "bar_open_time"})
        b = b.merge(idx_keyed.reset_index(), on="bar_open_time", how="left")
        b["direction"] = dirc
        return b
    buy = join(buy_b, "long")
    sell = join(sell_b, "short")
    return idx, buy, sell


def eligible(df, direction, strong_delta):
    """Apply the break-candle filters (direction, body ratio, strong delta, RTH)."""
    hm = df["bar_open_time"].dt.strftime("%H:%M")
    in_rth = (hm >= "09:30") & (hm <= "16:59")
    if direction == "long":
        mask = df["is_bullish"].fillna(False) & (df["body_ratio"] >= BODY_RATIO_MIN) \
             & (df["bar_total_delta"] >= strong_delta) & in_rth
    else:
        mask = df["is_bearish"].fillna(False) & (df["body_ratio"] >= BODY_RATIO_MIN) \
             & (df["bar_total_delta"] <= -strong_delta) & in_rth
    mask &= df["atr14"].notna() & df["bot_wick_low"].notna() & df["top_wick_high"].notna()
    return df[mask].reset_index(drop=True)


# ── trade simulation ────────────────────────────────────────────────────────
def simulate(signals_df, idx, direction):
    """Walk forward 5-min bars, emit trade rows."""
    idx_sorted = idx.sort_values("bar_open_time").reset_index(drop=True)
    idx_sorted["session_date"] = idx_sorted["session_date"].astype(str)
    bars_by_session = {s: g.reset_index(drop=True) for s, g in idx_sorted.groupby("session_date")}

    trades = []
    for _, r in signals_df.iterrows():
        entry_raw = float(r["close"])
        atr = float(r["atr14"])
        if direction == "long":
            entry = entry_raw + SLIPPAGE_PTS
            sl_raw = float(r["bot_wick_low"]) - ATR_MULT_SL * atr
            sl_dist = entry_raw - sl_raw
            if sl_dist <= 0: continue
            tp_raw = entry_raw + sl_dist + ATR_MULT_TP * atr
            sl_fill = sl_raw - SLIPPAGE_PTS
            tp_fill = tp_raw - SLIPPAGE_PTS
        else:
            entry = entry_raw - SLIPPAGE_PTS
            sl_raw = float(r["top_wick_high"]) + ATR_MULT_SL * atr
            sl_dist = sl_raw - entry_raw
            if sl_dist <= 0: continue
            tp_raw = entry_raw - sl_dist - ATR_MULT_TP * atr
            sl_fill = sl_raw + SLIPPAGE_PTS
            tp_fill = tp_raw + SLIPPAGE_PTS

        sess = str(r["session_date"])
        day = bars_by_session.get(sess)
        if day is None: continue
        post = day[day["bar_open_time"] > r["bar_open_time"]]
        post = post[post["bar_open_time"].dt.strftime("%H:%M").between("09:30", "16:59")]
        if post.empty: continue

        exit_price = None; exit_time = None; exit_reason = None
        for _, b in post.iterrows():
            h = b["high"]; l = b["low"]; c = b["close"]
            if np.isnan(h) or np.isnan(l): continue
            if direction == "long":
                hit_sl = l <= sl_raw
                hit_tp = h >= tp_raw
                if hit_sl and hit_tp:
                    exit_price = sl_fill; exit_reason = "sl_conflict"
                elif hit_sl:
                    exit_price = sl_fill; exit_reason = "sl"
                elif hit_tp:
                    exit_price = tp_fill; exit_reason = "tp"
            else:
                hit_sl = h >= sl_raw
                hit_tp = l <= tp_raw
                if hit_sl and hit_tp:
                    exit_price = sl_fill; exit_reason = "sl_conflict"
                elif hit_sl:
                    exit_price = sl_fill; exit_reason = "sl"
                elif hit_tp:
                    exit_price = tp_fill; exit_reason = "tp"
            if exit_price is not None:
                exit_time = b["bar_open_time"]; break
        if exit_price is None:
            last = post.iloc[-1]
            if direction == "long":
                exit_price = last["close"] - SLIPPAGE_PTS
            else:
                exit_price = last["close"] + SLIPPAGE_PTS
            exit_time = last["bar_open_time"]; exit_reason = "eod"

        if direction == "long":
            pts = exit_price - entry
        else:
            pts = entry - exit_price
        pnl_gross = pts * POINT_VALUE
        pnl_net = pnl_gross - FIXED_COST_RT

        trades.append({
            "direction": direction,
            "signal_time": r["bar_open_time"],
            "session_date": r["session_date"],
            "exit_time": exit_time, "exit_reason": exit_reason,
            "entry": entry, "exit": exit_price,
            "entry_raw": entry_raw, "sl_raw": sl_raw, "tp_raw": tp_raw,
            "atr": atr, "body_ratio": r["body_ratio"],
            "bar_total_delta": r["bar_total_delta"],
            "bot_wick_delta": r["bot_wick_delta"],
            "top_wick_delta": r["top_wick_delta"],
            "zone_low": r["zone_low"], "zone_high": r["zone_high"],
            "pts": pts, "pnl_gross": pnl_gross, "pnl_net": pnl_net,
        })
    return pd.DataFrame(trades)


# ── context attach + breakdowns ─────────────────────────────────────────────
def attach_context(tr):
    ctx = pd.read_parquet(DATA_DIR / "bar_context_1min.parquet")
    ctx["bar_open_time"] = localize_to_et(ctx["bar_open_time"])
    ctx_keyed = ctx.set_index("bar_open_time")
    cols = ["dev_poc","dev_vah","dev_val",
            "noise_upper_14","noise_lower_14","noise_upper_90","noise_lower_90",
            "prev_week_iso_poc","prev_week_iso_vah","prev_week_iso_val"]
    for c in cols:
        tr[c] = ctx_keyed.reindex(tr["signal_time"])[c].values

    tr["entry_hour"] = tr["signal_time"].dt.strftime("%H:00")

    def dev_loc(row):
        p, vah, val = row["entry_raw"], row["dev_vah"], row["dev_val"]
        if pd.isna(vah) or pd.isna(val): return "unknown"
        if p > vah: return "above_vah"
        if p < val: return "below_val"
        return "in_va"
    def noise_loc(row):
        p = row["entry_raw"]
        u14,l14 = row["noise_upper_14"], row["noise_lower_14"]
        u90,l90 = row["noise_upper_90"], row["noise_lower_90"]
        in_14 = (not pd.isna(u14)) and (l14 <= p <= u14)
        in_90 = (not pd.isna(u90)) and (l90 <= p <= u90)
        if pd.isna(u14) and pd.isna(u90): return "no_bands"
        if in_14 and in_90: return "inside_both"
        if in_14: return "inside_14_only"
        if in_90: return "inside_90_only"
        return "outside_both"
    def wk_loc(row):
        p, vah, val = row["entry_raw"], row["prev_week_iso_vah"], row["prev_week_iso_val"]
        if pd.isna(vah) or pd.isna(val): return "unknown"
        if p > vah: return "above_week_vah"
        if p < val: return "below_week_val"
        return "in_week_va"
    def sign_bucket(x):
        if pd.isna(x) or x == 0: return "zero"
        return "neg" if x < 0 else "pos"

    tr["dev_loc"] = tr.apply(dev_loc, axis=1)
    tr["noise_loc"] = tr.apply(noise_loc, axis=1)
    tr["week_loc"] = tr.apply(wk_loc, axis=1)
    tr["bot_wick_sign"] = tr["bot_wick_delta"].apply(sign_bucket)
    tr["top_wick_sign"] = tr["top_wick_delta"].apply(sign_bucket)
    return tr


def breakdown(df, col):
    rows = []
    for val, grp in df.groupby(col):
        m = metrics(grp); m[col] = val
        rows.append(m)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


# ── main ────────────────────────────────────────────────────────────────────
def main():
    idx, buy_pool, sell_pool = build_breaks()

    # Sweep summary
    print(f"\n=== SWEEP over strong_delta thresholds ({STRONG_DELTA_SWEEP}) ===")
    sweep_rows = []
    for sd in STRONG_DELTA_SWEEP:
        for direction, pool in [("long", buy_pool), ("short", sell_pool)]:
            el = eligible(pool, direction, sd)
            if el.empty:
                sweep_rows.append({"direction": direction, "strong_delta": sd, "n": 0,
                                    "wr": 0, "pf": float("nan"), "mean": 0, "total": 0})
                continue
            tr = simulate(el, idx, direction)
            m = metrics(tr)
            m["direction"] = direction; m["strong_delta"] = sd
            sweep_rows.append(m)
            print(f"  {direction:5s}  delta>={sd:>4}   n={m['n']:>5}  WR={m['wr']:5.1f}%  "
                  f"PF={m['pf']:.2f}  mean=${m['mean']:+,.1f}  total=${m['total']:+,.0f}",
                  flush=True)
    sweep_df = pd.DataFrame(sweep_rows)[["direction","strong_delta","n","wr","pf","mean","total","avg_win","avg_loss"]]
    sweep_df.to_csv(OUT_DIR / "sweep_summary.csv", index=False)

    # Detail run at default threshold
    print(f"\n=== DETAIL breakdowns (strong_delta >= {DETAIL_THRESHOLD}) ===")
    for direction, pool in [("long", buy_pool), ("short", sell_pool)]:
        el = eligible(pool, direction, DETAIL_THRESHOLD)
        tr = simulate(el, idx, direction)
        if tr.empty:
            print(f"\n{direction.upper()}: no trades")
            continue
        tr = attach_context(tr)
        tr.to_csv(OUT_DIR / f"trades_{direction}_t{DETAIL_THRESHOLD}.csv", index=False)

        print(f"\n{direction.upper()} ({len(tr)} trades)")
        o = metrics(tr)
        print(f"  overall   n={o['n']}  WR={o['wr']:.1f}%  PF={o['pf']:.2f}  "
              f"mean=${o['mean']:+,.1f}  total=${o['total']:+,.0f}  "
              f"avg_win=${o['avg_win']:+,.0f}  avg_loss=${o['avg_loss']:+,.0f}")
        for col in ["entry_hour", "dev_loc", "noise_loc", "week_loc",
                     "bot_wick_sign", "top_wick_sign", "exit_reason"]:
            bt = breakdown(tr, col)
            if bt.empty: continue
            print(f"\n  ── by {col} ──")
            for _, row in bt.iterrows():
                print(f"    {str(row[col]):18s}  n={int(row['n']):>4}  WR={row['wr']:5.1f}%  "
                      f"PF={row['pf']:.2f}  mean=${row['mean']:+,.0f}  total=${row['total']:+,.0f}")
    print(f"\nArtifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
