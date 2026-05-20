"""
Backtest: RTH pullback-into-support long signal.

Setup:
  - long_2bar_bothwicks pattern + bar N bar_total_delta >= STRONG_DELTA
    (both wicks have aggressive sellers trapped, bar N closes bullish with strong total delta)
  - signal bar's low lies inside the [zone_low, zone_high] of an active support zone
    at signal time (strict zone filter — zone formed before signal, not yet invalidated)
  - signal_time in RTH (09:30-16:59 ET)

Entry: bar N close, with 0.125 pt adverse slippage.
SL    = signal_bar.bot_wick_low - 1 * ATR14(5min)     (adverse slippage on fill)
TP    = entry + SL_distance + 1 * ATR14(5min)         (adverse slippage on fill)
Exit: walk forward 5-min bars intraday. SL or TP fills on first hit. If neither,
      force-close at last RTH bar (16:59) on that day.

Costs: $4.50 fixed round-trip + slippage baked into fill prices ($5 RT slippage).
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path("D:/trading_pythonbacktest_data")
OUT_DIR = Path(__file__).resolve().parent / "results_pullback_support"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ET = "America/New_York"
MIN_WICK_DELTA = 80
STRONG_DELTA = 200
ATR_MULT_SL = 1.0
ATR_MULT_TP = 1.0
ATR_LEN = 14

POINT_VALUE = 20.0      # $ per NQ pt
TICK_SIZE = 0.25
SLIPPAGE_PTS = 0.125    # per side
FIXED_COST_RT = 2 * (0.85 + 1.40)   # $4.50


def main():
    # ── 1. Load inputs ───────────────────────────────────────────────────────
    print("Loading...", flush=True)
    idx = pd.read_parquet(DATA_DIR / "wick_delta_index_5min.parquet")
    idx["bar_open_time"] = pd.to_datetime(idx["bar_open_time"])
    if idx["bar_open_time"].dt.tz is None:
        idx["bar_open_time"] = idx["bar_open_time"].dt.tz_localize("UTC").dt.tz_convert(ET)
    else:
        idx["bar_open_time"] = idx["bar_open_time"].dt.tz_convert(ET)
    idx = idx.sort_values("bar_open_time").reset_index(drop=True)

    # Compute 14-period ATR on 5-min bars (SMA of true range)
    prev_close = idx["close"].shift(1)
    tr = pd.concat([
        idx["high"] - idx["low"],
        (idx["high"] - prev_close).abs(),
        (idx["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    idx["atr14"] = tr.rolling(ATR_LEN, min_periods=ATR_LEN).mean()

    signals = pd.read_parquet(DATA_DIR / "signals_5min.parquet")
    signals["signal_time"] = pd.to_datetime(signals["signal_time"])
    if signals["signal_time"].dt.tz is None:
        signals["signal_time"] = signals["signal_time"].dt.tz_localize("UTC").dt.tz_convert(ET)
    else:
        signals["signal_time"] = signals["signal_time"].dt.tz_convert(ET)

    zones = pd.read_parquet(DATA_DIR / "wick_zones_5min.parquet")
    zones["formed_at"] = pd.to_datetime(zones["formed_at"])
    if zones["formed_at"].dt.tz is None:
        zones["formed_at"] = zones["formed_at"].dt.tz_localize("UTC").dt.tz_convert(ET)
    else:
        zones["formed_at"] = zones["formed_at"].dt.tz_convert(ET)
    zones["invalidation_time"] = pd.to_datetime(zones["invalidation_time"])
    tz = zones["invalidation_time"].dt
    if tz.tz is None:
        zones["invalidation_time"] = zones["invalidation_time"].dt.tz_localize("UTC").dt.tz_convert(ET)
    else:
        zones["invalidation_time"] = zones["invalidation_time"].dt.tz_convert(ET)
    support_zones = zones[zones["zone_type"] == "support"].reset_index(drop=True)

    ctx = pd.read_parquet(DATA_DIR / "bar_context_1min.parquet")
    ctx["bar_open_time"] = pd.to_datetime(ctx["bar_open_time"])
    if ctx["bar_open_time"].dt.tz is None:
        ctx["bar_open_time"] = ctx["bar_open_time"].dt.tz_localize("UTC").dt.tz_convert(ET)
    else:
        ctx["bar_open_time"] = ctx["bar_open_time"].dt.tz_convert(ET)

    # ── 2. Filter signals to our hybrid setup ───────────────────────────────
    sig = signals[(signals["signal_type"] == "long_2bar_bothwicks")
                  & (signals["signal_bar_total_delta"] >= STRONG_DELTA)].copy()
    print(f"  long_2bar_bothwicks with bar_total_delta>={STRONG_DELTA}: {len(sig):,}")

    # RTH only
    hm = sig["signal_time"].dt.strftime("%H:%M")
    sig = sig[(hm >= "09:30") & (hm <= "16:59")].reset_index(drop=True)
    print(f"  RTH only: {len(sig):,}")

    # Join signal bar's low, bot_wick_low, atr
    idx_keyed = idx.set_index("bar_open_time")
    sig["signal_low"]         = idx_keyed.reindex(sig["signal_time"])["low"].values
    sig["signal_bot_wick_low"] = idx_keyed.reindex(sig["signal_time"])["bot_wick_low"].values
    sig["signal_close"]       = idx_keyed.reindex(sig["signal_time"])["close"].values
    sig["atr"]                = idx_keyed.reindex(sig["signal_time"])["atr14"].values

    sig = sig[sig["atr"].notna() & sig["signal_bot_wick_low"].notna()].reset_index(drop=True)
    print(f"  with ATR and bot_wick_low available: {len(sig):,}")

    # ── 3. Strict zone filter ────────────────────────────────────────────────
    # For each signal: find any support zone where formed_at < signal_time AND
    # (invalidation_time is NaT OR invalidation_time > signal_time) AND
    # zone_low <= signal_low <= zone_high
    print("\nApplying strict zone filter...", flush=True)
    # Pre-index zones by formation time for fast search. Work in UTC-ns ints
    # to avoid tz comparison headaches with numpy datetime64.
    sz = support_zones.sort_values("formed_at").reset_index(drop=True)
    zone_formed_ns = sz["formed_at"].astype("int64").to_numpy()
    zone_inv_raw = sz["invalidation_time"]
    zone_inv_ns = zone_inv_raw.astype("int64").to_numpy()
    zone_inv_isnat = zone_inv_raw.isna().to_numpy()
    zone_lows = sz["zone_low"].to_numpy()
    zone_highs = sz["zone_high"].to_numpy()
    zone_ids = sz["zone_id"].to_numpy()

    sig_times_ns = sig["signal_time"].astype("int64").to_numpy()

    keep_idx = []
    zone_id_for_sig = []
    for i in range(len(sig)):
        st_ns = sig_times_ns[i]
        sl_low = sig.iloc[i]["signal_low"]
        if pd.isna(sl_low):
            continue
        mask = (zone_formed_ns < st_ns)
        mask &= (zone_inv_isnat | (zone_inv_ns > st_ns))
        mask &= (zone_lows <= sl_low) & (sl_low <= zone_highs)
        if mask.any():
            picked = np.where(mask)[0][-1]
            keep_idx.append(i)
            zone_id_for_sig.append(int(zone_ids[picked]))
    sig = sig.iloc[keep_idx].reset_index(drop=True)
    sig["matched_zone_id"] = zone_id_for_sig
    print(f"  signals inside active support zones: {len(sig):,}")

    # ── 4. Simulate trades ───────────────────────────────────────────────────
    print("\nSimulating trades...", flush=True)
    # Build a per-session lookup of future 5-min bars for exit simulation
    idx["session_date"] = idx["session_date"].astype(str)
    bars_by_session = {s: g for s, g in idx.groupby("session_date")}

    trades = []
    for _, r in sig.iterrows():
        entry_raw = float(r["signal_close"])
        entry = entry_raw + SLIPPAGE_PTS  # adverse entry
        sl_raw = float(r["signal_bot_wick_low"]) - ATR_MULT_SL * float(r["atr"])
        sl_dist_raw = entry_raw - sl_raw
        if sl_dist_raw <= 0:
            continue
        tp_raw = entry_raw + sl_dist_raw + ATR_MULT_TP * float(r["atr"])
        sl_fill = sl_raw - SLIPPAGE_PTS  # adverse fill on SL (worse for long)
        tp_fill = tp_raw - SLIPPAGE_PTS  # adverse fill on TP

        # Walk forward until SL/TP or end of RTH
        sess = str(r["session_date"])
        day_bars = bars_by_session.get(sess)
        if day_bars is None:
            continue
        st = r["signal_time"]
        post = day_bars[day_bars["bar_open_time"] > st]
        if post.empty:
            continue
        # Force close at 16:59 close of that day
        rth_mask = post["bar_open_time"].dt.strftime("%H:%M").between("09:30", "16:59")
        post = post[rth_mask]

        exit_price = None
        exit_time = None
        exit_reason = None
        for _, b in post.iterrows():
            h = b["high"]; l = b["low"]; c = b["close"]
            if np.isnan(h) or np.isnan(l):
                continue
            if l <= sl_raw and h >= tp_raw:
                # both hit same bar — conservative: SL first
                exit_price = sl_fill; exit_time = b["bar_open_time"]; exit_reason = "sl_conflict"; break
            if l <= sl_raw:
                exit_price = sl_fill; exit_time = b["bar_open_time"]; exit_reason = "sl"; break
            if h >= tp_raw:
                exit_price = tp_fill; exit_time = b["bar_open_time"]; exit_reason = "tp"; break
        if exit_price is None:
            last = post.iloc[-1]
            exit_price = last["close"] - SLIPPAGE_PTS
            exit_time = last["bar_open_time"]
            exit_reason = "eod"

        pts = exit_price - entry
        pnl_gross = pts * POINT_VALUE
        pnl_net = pnl_gross - FIXED_COST_RT
        trades.append({
            "signal_time": r["signal_time"],
            "exit_time": exit_time,
            "exit_reason": exit_reason,
            "entry": entry, "exit": exit_price,
            "entry_raw": entry_raw, "sl_raw": sl_raw, "tp_raw": tp_raw,
            "atr": r["atr"],
            "signal_low": r["signal_low"],
            "bot_wick_low": r["signal_bot_wick_low"],
            "signal_bar_total_delta": r["signal_bar_total_delta"],
            "zone_id": r["matched_zone_id"],
            "pts": pts,
            "pnl_gross": pnl_gross,
            "pnl_net": pnl_net,
        })

    tr = pd.DataFrame(trades)
    if tr.empty:
        print("No trades generated.")
        return
    print(f"  trades: {len(tr):,}", flush=True)

    # ── 5. Attach context (for breakdowns) ───────────────────────────────────
    print("\nAttaching context columns...", flush=True)
    ctx_keyed = ctx.set_index("bar_open_time")
    ctx_cols = ["dev_poc", "dev_vah", "dev_val",
                "noise_upper_14", "noise_lower_14",
                "noise_upper_90", "noise_lower_90",
                "prev_week_iso_poc", "prev_week_iso_vah", "prev_week_iso_val"]
    for c in ctx_cols:
        tr[c] = ctx_keyed.reindex(tr["signal_time"])[c].values

    tr["entry_hour"] = tr["signal_time"].dt.strftime("%H:00")

    def dev_loc(row):
        p = row["entry_raw"]; vah = row["dev_vah"]; val = row["dev_val"]
        if pd.isna(vah) or pd.isna(val): return "unknown"
        if p > vah: return "above_vah"
        if p < val: return "below_val"
        return "in_va"
    tr["dev_loc"] = tr.apply(dev_loc, axis=1)

    def noise_loc(row):
        p = row["entry_raw"]
        u14, l14 = row["noise_upper_14"], row["noise_lower_14"]
        u90, l90 = row["noise_upper_90"], row["noise_lower_90"]
        in_14 = (not pd.isna(u14)) and (l14 <= p <= u14)
        in_90 = (not pd.isna(u90)) and (l90 <= p <= u90)
        if pd.isna(u14) and pd.isna(u90): return "no_bands"
        if in_14 and in_90: return "inside_both"
        if in_14 and not in_90: return "inside_14_only"
        if in_90 and not in_14: return "inside_90_only"
        return "outside_both"
    tr["noise_loc"] = tr.apply(noise_loc, axis=1)

    def wk_loc(row):
        p = row["entry_raw"]; vah = row["prev_week_iso_vah"]; val = row["prev_week_iso_val"]
        if pd.isna(vah) or pd.isna(val): return "unknown"
        if p > vah: return "above_week_vah"
        if p < val: return "below_week_val"
        return "in_week_va"
    tr["week_loc"] = tr.apply(wk_loc, axis=1)

    # ── 6. Metrics ───────────────────────────────────────────────────────────
    def metrics(df):
        if df.empty: return {}
        w = df[df["pnl_net"] > 0]; l = df[df["pnl_net"] < 0]
        pf = w["pnl_net"].sum() / abs(l["pnl_net"].sum()) if len(l) and l["pnl_net"].sum() != 0 else float("nan")
        return {
            "n": len(df),
            "wr": 100 * len(w) / len(df),
            "mean_pnl": df["pnl_net"].mean(),
            "sum_pnl": df["pnl_net"].sum(),
            "pf": pf,
            "avg_win": w["pnl_net"].mean() if len(w) else 0,
            "avg_loss": l["pnl_net"].mean() if len(l) else 0,
        }

    overall = metrics(tr)
    print(f"\n=== OVERALL ===")
    print(f"  trades={overall['n']:,}  WR={overall['wr']:.1f}%  PF={overall['pf']:.2f}  "
          f"mean_pnl=${overall['mean_pnl']:+,.1f}  total=${overall['sum_pnl']:+,.0f}")
    print(f"  avg_win=${overall['avg_win']:+,.0f}  avg_loss=${overall['avg_loss']:+,.0f}")

    def breakdown_table(group_col, df=tr):
        rows = []
        for val, grp in df.groupby(group_col):
            m = metrics(grp)
            m[group_col] = val
            rows.append(m)
        return pd.DataFrame(rows).sort_values("n", ascending=False)

    print(f"\n=== BY ENTRY HOUR ===")
    print(breakdown_table("entry_hour").to_string(index=False))

    print(f"\n=== BY DEV PROFILE LOCATION ===")
    print(breakdown_table("dev_loc").to_string(index=False))

    print(f"\n=== BY NOISE BAND LOCATION ===")
    print(breakdown_table("noise_loc").to_string(index=False))

    print(f"\n=== BY PREV-WEEK ISO PROFILE LOCATION ===")
    print(breakdown_table("week_loc").to_string(index=False))

    # Exit reason breakdown
    print(f"\n=== BY EXIT REASON ===")
    print(breakdown_table("exit_reason").to_string(index=False))

    # Save artifacts
    tr.to_csv(OUT_DIR / "trades.csv", index=False)
    breakdown_table("entry_hour").to_csv(OUT_DIR / "by_entry_hour.csv", index=False)
    breakdown_table("dev_loc").to_csv(OUT_DIR / "by_dev_loc.csv", index=False)
    breakdown_table("noise_loc").to_csv(OUT_DIR / "by_noise_loc.csv", index=False)
    breakdown_table("week_loc").to_csv(OUT_DIR / "by_week_loc.csv", index=False)
    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
