"""V4 (historical conflict analysis) + V2 (combined strategy backtest):
  Combine locked filtered B2 break strategy with VA-revert mean-revert strategy.

  Locked B2 = V2 K=0.8 lock=0.45 + mart-fc-only + hours 9-14 + drop POS+SHORT
  VA-revert = N=20, sig_D=0, conf_D=150, sl_mult=1.0, hours={9,11,12}, overshoot<75%

Daily bucket categorization:
  AGREE-LONG       : VA-revert qualifies LONG  AND B2 first-trade is LONG
  AGREE-SHORT      : VA-revert qualifies SHORT AND B2 first-trade is SHORT
  CONFLICT-VAL-B2S : VA says LONG,  B2 says SHORT
  CONFLICT-VAS-B2L : VA says SHORT, B2 says LONG
  VA-ONLY          : VA-revert qualifies, B2 doesn't fire that day
  B2-ONLY          : B2 fires, VA-revert day not qualifying
  NEITHER          : nothing fires

V4 (analysis): outcome on each bucket — which strategy won, by how much
V2 (combined): on CONFLICT days take VA-revert (stop after first TP); on AGREE
               take B2 only; on single-strategy buckets take that one.
"""
from __future__ import annotations

import datetime as dt
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_SCRIPTS = Path(__file__).parent.parent / "overnight range strat" / "scripts"
sys.path.insert(0, str(PROJECT_SCRIPTS))
from sweep_ratchet_sl_fixed_tp import filter_pre_dedupe
from test_pure_ratchet_exits import build_20min_bars, FORCE_CLOSE_TIME

PARQUET_DIR_B2 = PROJECT_SCRIPTS / "parquets"
EOD_MQ         = Path("D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet")
SIGS_VA        = Path(__file__).parent / "va_revert_signals.parquet"
VA_PARQUET     = Path(__file__).parent / "prev_day_rth_va.parquet"
M1_BARS        = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
OUT_TXT        = Path(__file__).parent.parent / "overnight range strat" / "tradelogs" / "robust_configs" / "b2_va_revert_combine.txt"
ET             = "America/New_York"

# Locked B2 params
YMULT, TPMULT   = 2.50, 2.00
MFE_K, MFE_LOCK = 0.8, 0.45
ALLOWED_HOURS_B2 = {9, 10, 11, 12, 13, 14}
DROP_POS_SHORT   = True

# VA-revert chosen config
VA_N        = 20
VA_SIG_D    = 0
VA_CONF_D   = 150
VA_SL_MULT  = 1.00
VA_HOURS    = {9, 11, 12}
OVERSHOOT_LIMIT_PCT = 75.0
MIN_VA_WIDTH = 20.0

RTH_START   = dt.time(9, 30)
FORCE_CLOSE = dt.time(16, 0)


# ====================== B2 simulation (locked filtered V2 + mart-fc) ======================

def simulate_b2_exit(direction, entry_ts, entry_price, bars20):
    sign = 1 if direction == "LONG" else -1
    bars_idx = bars20.index
    start = bars_idx.searchsorted(entry_ts, side="right")
    if start >= len(bars_idx): return None
    ent_date = entry_ts.date()
    end = start
    while end < len(bars_idx) and bars_idx[end].date() == ent_date: end += 1
    if end == start: return None
    init_idx = start - 1
    if init_idx < 0 or np.isnan(bars20["atr_y"].iloc[init_idx]): return None
    init_atr_y = float(bars20["atr_y"].iloc[init_idx])
    yellow_val = entry_price - sign * YMULT * init_atr_y
    prev_yellow = yellow_val
    o = bars20["open"].values[start:end]; h = bars20["high"].values[start:end]
    l = bars20["low"].values[start:end];  c = bars20["close"].values[start:end]
    ay = bars20["atr_y"].values[start:end]; ts_arr = bars_idx[start:end]
    n = end - start
    green_val = entry_price + sign * TPMULT * init_atr_y
    tp_dist = abs(green_val - entry_price)
    mfe_so_far = 0.0
    for i in range(n):
        bar_close_ts = ts_arr[i] + pd.Timedelta(minutes=20)
        cur_mfe = (h[i] - entry_price) if sign > 0 else (entry_price - l[i])
        if cur_mfe > mfe_so_far: mfe_so_far = cur_mfe
        if not np.isnan(ay[i]):
            raw_yellow = c[i] - sign * YMULT * ay[i]
            yellow_val = max(prev_yellow, raw_yellow) if sign > 0 else min(prev_yellow, raw_yellow)
        if mfe_so_far >= MFE_K * tp_dist:
            mfe_stop = entry_price + sign * MFE_LOCK * mfe_so_far
            stop_level = max(yellow_val, mfe_stop) if sign > 0 else min(yellow_val, mfe_stop)
        else:
            stop_level = yellow_val
        if sign > 0 and h[i] >= green_val:
            return (sign * (green_val - entry_price), "TP_FIXED", bar_close_ts)
        if sign < 0 and l[i] <= green_val:
            return (sign * (green_val - entry_price), "TP_FIXED", bar_close_ts)
        if sign > 0 and c[i] <= stop_level and c[i] < o[i]:
            return (c[i] - entry_price, "SL_TRAIL", bar_close_ts)
        if sign < 0 and c[i] >= stop_level and c[i] > o[i]:
            return (entry_price - c[i], "SL_TRAIL", bar_close_ts)
        if ts_arr[i].time() >= FORCE_CLOSE_TIME:
            return (sign * (c[i] - entry_price), "FORCE_CLOSE", bar_close_ts)
        prev_yellow = yellow_val
    return (sign * (c[-1] - entry_price), "EOD",
            ts_arr[-1] + pd.Timedelta(minutes=20))


def attach_gamma_b2(cands):
    if not EOD_MQ.exists():
        cands = cands.copy(); cands["gamma_sign"] = np.nan; return cands
    eod = pd.read_parquet(EOD_MQ)
    eod["date"] = pd.to_datetime(eod["date"]).dt.date
    eod = eod.set_index("date").sort_index()
    dates_sorted = sorted(eod.index.tolist())
    def prior(d):
        prev = None
        for md in dates_sorted:
            if md < d: prev = md
            else: break
        return prev
    cands = cands.copy()
    cands["entry_date"] = pd.to_datetime(cands["entry_time_et"]).dt.date
    col = "qqq_gamma_sign" if "qqq_gamma_sign" in eod.columns else None
    if col is None:
        cands["gamma_sign"] = np.nan
    else:
        g = {d: (eod.loc[prior(d), col] if prior(d) in eod.index else np.nan)
             for d in cands["entry_date"].unique()}
        cands["gamma_sign"] = cands["entry_date"].map(g)
    return cands


def filter_b2(cands):
    cands = cands.copy()
    cands["entry_hour"] = pd.to_datetime(cands["entry_time_et"]).dt.hour
    keep = cands["entry_hour"].isin(ALLOWED_HOURS_B2)
    if DROP_POS_SHORT:
        keep &= ~((cands["gamma_sign"] == 1) & (cands["direction"] == "SHORT"))
    return cands[keep].reset_index(drop=True)


def run_b2_chained(cands, bars20):
    rows = []
    last_exit = pd.Timestamp(0, tz=ET)
    for _, t in cands.sort_values("entry_time_et").iterrows():
        ent_t = pd.Timestamp(t["entry_time_et"])
        if ent_t <= last_exit: continue
        ex = simulate_b2_exit(t["direction"], ent_t, float(t["entry_price"]), bars20)
        if ex is None: continue
        pnl, reason, exit_ts = ex
        rows.append({"entry_ts": ent_t, "exit_ts": exit_ts,
                     "direction": t["direction"], "reason": reason, "pnl": pnl})
        last_exit = pd.Timestamp(exit_ts)
    return pd.DataFrame(rows)


def apply_mart_fc(df):
    sizes = []; cur = 1
    for pnl, reason in zip(df["pnl"].values, df["reason"].values):
        sizes.append(cur)
        if cur == 2: cur = 1
        else: cur = 2 if (pnl < 0 and reason == "FORCE_CLOSE") else 1
    return np.array(sizes)


# ====================== VA-revert simulation (chosen config) ======================

def simulate_va_exit(direction, entry_price, entry_idx, bars_day, sl_pts, tp_price):
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
            tp_hit = hi >= tp_price; sl_hit = lo <= sl_price
        else:
            tp_hit = lo <= tp_price; sl_hit = hi >= sl_price
        if tp_hit and sl_hit: return ("SL", sign * (sl_price - entry_price), bt)
        if tp_hit: return ("TP", sign * (tp_price - entry_price), bt)
        if sl_hit: return ("SL", sign * (sl_price - entry_price), bt)
    last = bars_day.iloc[-1]
    return ("held->close", sign * (float(last["close"]) - entry_price),
            pd.Timestamp(last["bar_open_time"]))


def run_va_chained(sigs, bars_by_day, stop_after_tp=False):
    """Chained Mode 1 VA-revert sim. If stop_after_tp=True, after a TP hit on
    a date, no further entries on that same date."""
    rows = []
    last_exit = pd.Timestamp(0, tz=ET)
    days_done = set()
    for _, t in sigs.sort_values("entry_time").iterrows():
        ent_t = pd.Timestamp(t["entry_time"])
        d = pd.to_datetime(t["date"]).date()
        if stop_after_tp and d in days_done: continue
        if ent_t <= last_exit: continue
        bars_day = bars_by_day.get(d)
        if bars_day is None: continue
        entry_idx = int(t["entry_idx"])
        if entry_idx >= len(bars_day): continue
        entry_price = float(t["entry_price"]); atr = float(t["atr_at_entry"])
        sl_pts = VA_SL_MULT * atr; tp_price = float(t["target"])
        outcome, pnl, exit_t = simulate_va_exit(
            t["direction"], entry_price, entry_idx, bars_day, sl_pts, tp_price)
        rows.append({"entry_ts": ent_t, "exit_ts": exit_t, "date": d,
                     "direction": t["direction"], "outcome": outcome, "pnl": pnl})
        last_exit = pd.Timestamp(exit_t)
        if stop_after_tp and outcome == "TP":
            days_done.add(d)
    return pd.DataFrame(rows)


# ====================== loaders ======================

def load_premarket_open():
    df = pd.read_parquet(M1_BARS, columns=["close"])
    idx = pd.to_datetime(df.index, utc=True).tz_convert(ET)
    df = df.set_index(idx).sort_index()
    df["date"] = df.index.date
    df["t"]    = df.index.time
    pre = df[df["t"] < RTH_START].copy()
    last = pre.groupby("date").tail(1)
    return last[["date", "close"]].rename(columns={"close": "open_price"}).set_index("date")


def qualifying_days_va():
    va = pd.read_parquet(VA_PARQUET)
    va["date"] = pd.to_datetime(va["date"]).dt.date
    pre = load_premarket_open()
    dates = va.dropna(subset=["vah","val"]).sort_values("date")["date"].tolist()
    prev_lookup = {dates[i]: dates[i-1] for i in range(1, len(dates))}
    va_idx = va.set_index("date")
    rows = []
    for today in sorted(set(va["date"]).intersection(pre.index)):
        prev_d = prev_lookup.get(today)
        if prev_d is None or prev_d not in va_idx.index: continue
        vah_p = va_idx.loc[prev_d, "vah"]; val_p = va_idx.loc[prev_d, "val"]
        if not (np.isfinite(vah_p) and np.isfinite(val_p)): continue
        width = vah_p - val_p
        if width < MIN_VA_WIDTH: continue
        op = float(pre.loc[today, "open_price"])
        if not np.isfinite(op): continue
        if op > vah_p:
            direction = "SHORT"; distance = op - vah_p
        elif op < val_p:
            direction = "LONG"; distance = val_p - op
        else:
            continue
        pct = distance / width * 100
        if pct >= OVERSHOOT_LIMIT_PCT: continue
        rows.append({"date": today, "va_dir": direction,
                     "open_price": op, "vah_prev": float(vah_p),
                     "val_prev": float(val_p), "distance_pct": pct})
    return pd.DataFrame(rows).set_index("date")


# ====================== main ======================

def main():
    print("=== B2 SIDE ===")
    print("loading 20-min bars + B2 candidates ...")
    bars20 = build_20min_bars()
    is_df  = pd.read_parquet(PARQUET_DIR_B2 / "entry_signal_trades.parquet")
    oos_df = pd.read_parquet(PARQUET_DIR_B2 / "entry_signal_trades_oos.parquet")
    is_cands  = filter_b2(attach_gamma_b2(filter_pre_dedupe(is_df)))
    oos_cands = filter_b2(attach_gamma_b2(filter_pre_dedupe(oos_df)))

    print("simulating B2 chained ...")
    is_t  = run_b2_chained(is_cands,  bars20)
    oos_t = run_b2_chained(oos_cands, bars20)
    b2_all = pd.concat([is_t, oos_t], ignore_index=True).sort_values("entry_ts").reset_index(drop=True)
    sizes = apply_mart_fc(b2_all)
    b2_all["size"] = sizes
    b2_all["pnl_scaled"] = b2_all["pnl"] * sizes
    b2_all["date"] = b2_all["entry_ts"].dt.date
    print(f"  B2 trades: {len(b2_all)}  (IS={len(is_t)} + OOS={len(oos_t)})")
    print(f"  B2 total pnl (mart-fc):  {b2_all['pnl_scaled'].sum():+.1f} pts")

    print("\n=== VA-REVERT SIDE ===")
    print(f"loading VA-revert signals from {SIGS_VA} ...")
    va_sigs = pd.read_parquet(SIGS_VA)
    va_sigs["date"] = pd.to_datetime(va_sigs["date"]).dt.date
    va_sigs["entry_time"] = pd.to_datetime(va_sigs["entry_time"])
    # Filter to chosen config
    sig_col = f"sig_abs_w{VA_N}"; conf_col = f"conf_abs_w{VA_N}"
    va_sigs["entry_hour"] = va_sigs["entry_time"].dt.hour
    va_sub = va_sigs[(va_sigs["distance_pct"] < OVERSHOOT_LIMIT_PCT) &
                      (va_sigs[sig_col].fillna(0)  >= VA_SIG_D) &
                      (va_sigs[conf_col].fillna(0) >= VA_CONF_D) &
                      (va_sigs["entry_hour"].isin(VA_HOURS))].copy()
    print(f"  VA-revert signals after filter: {len(va_sub)}")

    # Need bars_by_day for VA sim
    from range_break_entry_signal_study import load_5min_features
    print("loading 5-min bars ...")
    bars_all, _ = load_5min_features((dt.date(2020,12,1), dt.date(2026,5,7)))
    bars_by_day = {d: g.sort_values("bar_open_time").reset_index(drop=True)
                   for d, g in bars_all.groupby("session_date")}

    print("running VA-revert chained (V4 — no day-stop) ...")
    va_t_v4 = run_va_chained(va_sub, bars_by_day, stop_after_tp=False)
    print(f"  VA-revert trades (V4): {len(va_t_v4)}, total pnl: {va_t_v4['pnl'].sum():+.1f}")

    print("running VA-revert chained (V2 — stop-after-TP per day) ...")
    va_t_v2 = run_va_chained(va_sub, bars_by_day, stop_after_tp=True)
    print(f"  VA-revert trades (V2): {len(va_t_v2)}, total pnl: {va_t_v2['pnl'].sum():+.1f}")

    print("\n=== DAILY CROSS-TAB ===")
    qual = qualifying_days_va()
    print(f"  VA-revert qualifying days (open outside VA, <{OVERSHOOT_LIMIT_PCT}% overshoot): {len(qual)}")

    # B2 first-trade direction per day
    b2_daily = b2_all.groupby("date").agg(
        b2_first_dir=("direction", "first"),
        b2_n=("direction", "count"),
        b2_pnl_scaled=("pnl_scaled", "sum"),
    )
    print(f"  Days with B2 trades: {len(b2_daily)}")

    # Build per-day buckets
    all_days = sorted(set(b2_daily.index) | set(qual.index))
    daily = []
    for d in all_days:
        va_qual = d in qual.index
        va_dir  = qual.loc[d, "va_dir"] if va_qual else None
        b2_fired = d in b2_daily.index
        b2_dir   = b2_daily.loc[d, "b2_first_dir"] if b2_fired else None
        if va_qual and b2_fired:
            if va_dir == b2_dir:
                bucket = f"AGREE-{va_dir}"
            else:
                bucket = f"CONFLICT-VA{va_dir[:1]}-B2{b2_dir[:1]}"
        elif va_qual:
            bucket = "VA-ONLY"
        elif b2_fired:
            bucket = "B2-ONLY"
        else:
            bucket = "NEITHER"
        daily.append({"date": d, "bucket": bucket, "va_dir": va_dir, "b2_dir": b2_dir,
                      "b2_n": int(b2_daily.loc[d, "b2_n"]) if b2_fired else 0,
                      "b2_pnl_scaled": float(b2_daily.loc[d, "b2_pnl_scaled"]) if b2_fired else 0.0})
    daily_df = pd.DataFrame(daily)

    # VA-revert pnl per day (V4 chaining — no stop-after-tp)
    va_daily_v4 = va_t_v4.groupby("date").agg(va_n=("pnl","count"), va_pnl=("pnl","sum"),
                                                va_hit_tp=("outcome", lambda s: (s=="TP").any()))
    va_daily_v2 = va_t_v2.groupby("date").agg(va_n_v2=("pnl","count"), va_pnl_v2=("pnl","sum"))
    daily_df = daily_df.merge(va_daily_v4, left_on="date", right_index=True, how="left").merge(
        va_daily_v2, left_on="date", right_index=True, how="left")
    daily_df[["va_n","va_pnl","va_pnl_v2","va_n_v2"]] = daily_df[
        ["va_n","va_pnl","va_pnl_v2","va_n_v2"]].fillna(0)
    daily_df["va_hit_tp"] = daily_df["va_hit_tp"].fillna(False)

    L = []
    L.append("=" * 200)
    L.append("V4 — DAILY BUCKET CROSS-TAB")
    L.append("=" * 200)
    L.append(f"B2: locked filtered V2 + mart-fc-only")
    L.append(f"VA-revert: N={VA_N}, sig_D={VA_SIG_D}, conf_D={VA_CONF_D}, sl={VA_SL_MULT}, hours={sorted(VA_HOURS)}, cap={OVERSHOOT_LIMIT_PCT}%")
    L.append("")
    L.append(f"Total qualifying VA-revert days: {len(qual)}")
    L.append(f"Total B2 trade-days:             {len(b2_daily)}")
    L.append("")

    bucket_stats = (daily_df.groupby("bucket")
                    .agg(n_days=("date","count"),
                         b2_pnl=("b2_pnl_scaled","sum"),
                         va_pnl_v4=("va_pnl","sum"),
                         va_pnl_v2=("va_pnl_v2","sum"),
                         va_trades_v4=("va_n","sum"),
                         va_trades_v2=("va_n_v2","sum")))
    L.append("Bucket counts and aggregate $ pts (B2 mart-scaled; VA 1-contract):")
    L.append(bucket_stats.round(1).to_string())

    # Conflict-specific analysis
    L.append("")
    L.append("=" * 200)
    L.append("CONFLICT DAYS — who wins?")
    L.append("=" * 200)
    conflicts = daily_df[daily_df["bucket"].str.startswith("CONFLICT")].copy()
    L.append(f"Total conflict days: {len(conflicts)}")
    if not conflicts.empty:
        # WR perspectives
        conflicts["b2_won"] = conflicts["b2_pnl_scaled"] > 0
        conflicts["va_won_v4"] = conflicts["va_pnl"] > 0
        conflicts["va_better_than_b2"] = conflicts["va_pnl"] > conflicts["b2_pnl_scaled"]
        L.append(f"  Days B2 net positive:        {conflicts['b2_won'].sum()} ({conflicts['b2_won'].mean()*100:.1f}%)")
        L.append(f"  Days VA-revert net positive: {conflicts['va_won_v4'].sum()} ({conflicts['va_won_v4'].mean()*100:.1f}%)")
        L.append(f"  Days VA-revert > B2 on day:  {conflicts['va_better_than_b2'].sum()} ({conflicts['va_better_than_b2'].mean()*100:.1f}%)")
        L.append("")
        L.append("Aggregate $ on conflict days:")
        L.append(f"  B2 total:        {conflicts['b2_pnl_scaled'].sum():+.1f} pts (${conflicts['b2_pnl_scaled'].sum()*20:+,.0f} NQ)")
        L.append(f"  VA-revert (V4):  {conflicts['va_pnl'].sum():+.1f} pts (${conflicts['va_pnl'].sum()*20:+,.0f} NQ)")
        L.append(f"  VA-revert (V2 stop-after-tp): {conflicts['va_pnl_v2'].sum():+.1f} pts")
        L.append("")
        # Split by direction
        for b in conflicts["bucket"].unique():
            sub = conflicts[conflicts["bucket"] == b]
            L.append(f"  {b}: n={len(sub)}  B2 ${sub['b2_pnl_scaled'].sum()*20:+,.0f}  VA-revert ${sub['va_pnl'].sum()*20:+,.0f}")
        L.append("")
        L.append("Per-day rows (conflict only):")
        L.append(conflicts[["date","bucket","b2_n","b2_pnl_scaled","va_n","va_pnl","va_pnl_v2","va_hit_tp"]]
                 .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    L.append("")
    L.append("=" * 200)
    L.append("V2 — COMBINED STRATEGY: B2 on AGREE/B2-ONLY, VA-revert on CONFLICT/VA-ONLY (stop-after-TP)")
    L.append("=" * 200)
    # V2 rules per day:
    def v2_pnl(row):
        b = row["bucket"]
        if b.startswith("AGREE") or b == "B2-ONLY":
            return row["b2_pnl_scaled"]
        if b.startswith("CONFLICT") or b == "VA-ONLY":
            return row["va_pnl_v2"]
        return 0.0
    daily_df["v2_pnl"] = daily_df.apply(v2_pnl, axis=1)
    daily_df["b2_only_pnl"] = daily_df["b2_pnl_scaled"]
    daily_df["va_only_pnl_v2"] = daily_df["va_pnl_v2"]

    daily_df["year"] = pd.to_datetime(daily_df["date"]).dt.year
    daily_df["period"] = np.where(pd.to_datetime(daily_df["date"]).dt.date < dt.date(2025,1,1), "IS", "OOS")

    def agg(df_, lab):
        total = df_["v2_pnl"].sum()
        b2_only = df_["b2_only_pnl"].sum()
        va_only = df_["va_only_pnl_v2"].sum()
        return f"{lab}  combined_v2 = {total:>+8.1f} pts  (${total*20:>+9,.0f})   |   B2-only = {b2_only:>+8.1f}  |  VA-revert-only = {va_only:>+8.1f}"

    L.append(agg(daily_df, "ALL"))
    L.append(agg(daily_df[daily_df["period"]=="IS"],  "IS "))
    L.append(agg(daily_df[daily_df["period"]=="OOS"], "OOS"))

    L.append("")
    L.append("V2 per-year breakdown:")
    yr = daily_df.groupby("year").agg(
        v2_pts=("v2_pnl","sum"),
        b2_only_pts=("b2_only_pnl","sum"),
        va_only_pts=("va_only_pnl_v2","sum"),
        n_days=("date","count")).round(1)
    yr["v2_dollars_nq"] = (yr["v2_pts"]*20).round(0)
    yr["b2_dollars_nq"] = (yr["b2_only_pts"]*20).round(0)
    L.append(yr.to_string())

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
