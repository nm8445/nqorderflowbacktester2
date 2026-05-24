"""20-min break entry — COMBINED filter (DELTA AND ABSORPTION both required).

Sweeps (delta_threshold, absorption_threshold, tp_rr) jointly because the
combined filter is stricter than either alone and the sweet spots may shift.

Reuses the same data prep as compare_20min_break_filters.py.

Outputs:
  - Specific user-requested config: DELTA 600 + ABSORPTION 60, all TPs
  - Best TP per (delta, absorption) combo (mini heatmap)
  - Top 10 overall (n_trades >= 100)
  - Comparison to: locked baseline, delta-only best, absorption-only best
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

VOL_PARQUET = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
OUT_CSV = Path("C:/trading/nqorderflowbacktester/scripts/fabio_orb/20min_combined_filter_sweep.csv")
ET = "America/New_York"

ORB_START_HHMM = 830
ORB_END_HHMM   = 900
TRADE_END_HHMM = 1400
EOD_HHMM       = 1400

DELTA_GRID = [0, 100, 200, 300, 400, 500, 600, 800]
ABS_GRID   = [0, 30, 40, 50, 60, 70, 80, 100]
TP_GRID    = [round(x, 2) for x in np.arange(1.0, 4.01, 0.25)]
ABS_MIN_LEVELS = 2

TICK = 0.25
TICK_VAL = 5.0
DOLLARS_PER_PT = TICK_VAL / TICK
SLIP_PTS_RT = 1 * TICK * 2
COMM_RT = 5.0


def load_5min_bars():
    print(f"Loading {VOL_PARQUET.name}...")
    df = pd.read_parquet(VOL_PARQUET)
    df["bar_open_time"] = pd.to_datetime(df["bar_open_time"]).dt.tz_convert(ET)
    df["bar_close_time"] = df["bar_open_time"] + pd.Timedelta(minutes=5)
    df["hhmm"] = df["bar_close_time"].dt.hour * 100 + df["bar_close_time"].dt.minute
    df["date"] = df["bar_close_time"].dt.date
    ohlc = df.groupby("bar_open_time", as_index=False).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    )
    ohlc["bar_close_time"] = ohlc["bar_open_time"] + pd.Timedelta(minutes=5)
    ohlc["hhmm"] = ohlc["bar_close_time"].dt.hour * 100 + ohlc["bar_close_time"].dt.minute
    ohlc["date"] = ohlc["bar_close_time"].dt.date
    print(f"  {len(ohlc):,} 5-min bars, {len(df):,} per-level rows")
    return df, ohlc


def aggregate_20min(per_level: pd.DataFrame, abs_thresholds: list[int]) -> pd.DataFrame:
    df = per_level.copy()
    df["bar_open_20min"] = df["bar_open_time"].dt.floor("20min")
    five_min = df.drop_duplicates("bar_open_time")[
        ["bar_open_time", "bar_open_20min", "open", "high", "low", "close"]
    ].sort_values("bar_open_time")
    ohlc20 = five_min.groupby("bar_open_20min", as_index=False).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    )
    ohlc20["bar_close_20min"] = ohlc20["bar_open_20min"] + pd.Timedelta(minutes=20)
    ohlc20["hhmm"] = ohlc20["bar_close_20min"].dt.hour * 100 + ohlc20["bar_close_20min"].dt.minute
    ohlc20["date"] = ohlc20["bar_close_20min"].dt.date

    lvl20 = (df.groupby(["bar_open_20min", "level_price"], as_index=False)
               .agg(buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum")))
    tot = lvl20.groupby("bar_open_20min", as_index=False).agg(
        total_buy=("buy_vol", "sum"), total_sell=("sell_vol", "sum"))
    tot["total_delta"] = tot["total_buy"] - tot["total_sell"]
    ohlc20 = ohlc20.merge(tot[["bar_open_20min", "total_delta"]], on="bar_open_20min", how="left")

    lvl20 = lvl20.merge(ohlc20[["bar_open_20min", "high", "low"]], on="bar_open_20min", how="left")
    lvl20["mid"] = (lvl20["high"] + lvl20["low"]) / 2.0
    lvl20["is_lower_half"] = lvl20["level_price"] <= lvl20["mid"]
    lvl20["seller_pressure"] = lvl20["sell_vol"] - lvl20["buy_vol"]
    lower = lvl20[lvl20["is_lower_half"]].copy()

    print(f"  Computing absorption counts at {len(abs_thresholds)} thresholds...")
    for T in abs_thresholds:
        if T == 0:
            ohlc20[f"abs_count_{T}"] = 999   # always passes
            continue
        col = f"abs_count_{T}"
        passes = lower[lower["seller_pressure"] >= T].groupby("bar_open_20min").size()
        ohlc20 = ohlc20.merge(passes.rename(col), left_on="bar_open_20min", right_index=True, how="left")
        ohlc20[col] = ohlc20[col].fillna(0).astype(int)
    print(f"  {len(ohlc20):,} 20-min bars")
    return ohlc20


def find_entry_combo(bars20_day, orb_high, orb_low, T_delta, T_abs):
    abs_col = f"abs_count_{T_abs}"
    for _, bar in bars20_day.iterrows():
        if bar["hhmm"] > TRADE_END_HHMM: break
        if bar["close"] <= orb_high: continue
        if orb_low >= bar["close"]: continue
        if bar["total_delta"] < T_delta: continue
        if bar[abs_col] < ABS_MIN_LEVELS: continue
        return {"entry_time": bar["bar_close_20min"],
                "entry_price": float(bar["close"]),
                "entry_hhmm": int(bar["hhmm"])}
    return None


def simulate_exit_5min(bars5_after, entry_price, orb_low, tp_rr):
    sl = orb_low; risk = entry_price - sl; tp = entry_price + tp_rr * risk
    for _, bar in bars5_after.iterrows():
        if bar["low"] <= sl: return sl, "SL"
        if bar["high"] >= tp: return tp, "TP"
        if bar["hhmm"] >= EOD_HHMM: return float(bar["close"]), "EOD"
    last = bars5_after.iloc[-1] if len(bars5_after) > 0 else None
    return (float(last["close"]) if last is not None else entry_price), "EOD"


def run_combo(bars20, bars5_by_day, T_delta, T_abs, tp_rr):
    bars20_by_day = {d: g for d, g in bars20.groupby("date")}
    trades = []
    for d, bars20_d in bars20_by_day.items():
        bars5_d = bars5_by_day.get(d)
        if bars5_d is None: continue
        orb_bars = bars5_d[(bars5_d["hhmm"] > ORB_START_HHMM) & (bars5_d["hhmm"] <= ORB_END_HHMM)]
        if len(orb_bars) == 0: continue
        orb_high = float(orb_bars["high"].max())
        orb_low  = float(orb_bars["low"].min())
        post_orb_20 = bars20_d[(bars20_d["hhmm"] > ORB_END_HHMM) & (bars20_d["hhmm"] <= TRADE_END_HHMM)]
        if len(post_orb_20) == 0: continue
        entry = find_entry_combo(post_orb_20, orb_high, orb_low, T_delta, T_abs)
        if entry is None: continue
        post_entry_5 = bars5_d[bars5_d["bar_open_time"] >= entry["entry_time"]]
        if len(post_entry_5) == 0: continue
        exit_price, reason = simulate_exit_5min(post_entry_5, entry["entry_price"], orb_low, tp_rr)
        gross_pts = exit_price - entry["entry_price"]
        net_pts = gross_pts - SLIP_PTS_RT
        pnl = net_pts * DOLLARS_PER_PT - COMM_RT
        trades.append({"date": d, "entry_price": entry["entry_price"],
                       "exit_price": exit_price, "reason": reason, "pnl_$": pnl,
                       "entry_hhmm": entry["entry_hhmm"]})

    if not trades:
        return {"T_delta": T_delta, "T_abs": T_abs, "tp_rr": tp_rr,
                "n_trades": 0, "wr%": 0, "net_$": 0, "PF": 0, "MaxDD_$": 0,
                "avg_$": 0, "TP%": 0, "SL%": 0, "EOD%": 0}
    df = pd.DataFrame(trades).sort_values("date").reset_index(drop=True)
    df["cum"] = df["pnl_$"].cumsum(); df["peak"] = df["cum"].cummax(); df["dd"] = df["cum"] - df["peak"]
    wins = df[df["pnl_$"] > 0]; losses = df[df["pnl_$"] < 0]
    pf = wins["pnl_$"].sum() / abs(losses["pnl_$"].sum()) if len(losses) > 0 else float("inf")
    return {
        "T_delta": T_delta, "T_abs": T_abs, "tp_rr": tp_rr,
        "n_trades": len(df), "wr%": round(len(wins) / len(df) * 100, 1),
        "net_$": round(df["pnl_$"].sum(), 0), "PF": round(pf, 2),
        "MaxDD_$": round(df["dd"].min(), 0),
        "avg_$": round(df["pnl_$"].mean(), 1),
        "TP%": round((df["reason"] == "TP").mean() * 100, 1),
        "SL%": round((df["reason"] == "SL").mean() * 100, 1),
        "EOD%": round((df["reason"] == "EOD").mean() * 100, 1),
    }


def main():
    per_level, bars5 = load_5min_bars()
    bars20 = aggregate_20min(per_level, ABS_GRID)
    bars5_by_day = {d: g for d, g in bars5.groupby("date")}

    print(f"\nSweeping {len(DELTA_GRID) * len(ABS_GRID) * len(TP_GRID)} configs...")
    rows = []
    for d_th in DELTA_GRID:
        for a_th in ABS_GRID:
            for tp in TP_GRID:
                r = run_combo(bars20, bars5_by_day, d_th, a_th, tp)
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV} ({len(df)} rows)")
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)

    # 1) User-requested specific combo
    print("\n=== USER-REQUESTED: DELTA 600 + ABSORPTION 60 (sweep TP) ===")
    spec = df[(df["T_delta"] == 600) & (df["T_abs"] == 60)].sort_values("net_$", ascending=False)
    print(spec.to_string(index=False))

    # 2) Heatmap: best TP per (delta, absorption) — net $
    print("\n=== HEATMAP: best NET $ per (delta, absorption) combo (best TP picked) ===")
    best_per_combo = (df.loc[df.groupby(["T_delta", "T_abs"])["net_$"].idxmax()]
                       .pivot(index="T_delta", columns="T_abs", values="net_$"))
    print(best_per_combo.fillna(0).astype(int).to_string())

    print("\n=== HEATMAP: # trades per (delta, absorption) at best-TP cell ===")
    best_per_combo_n = (df.loc[df.groupby(["T_delta", "T_abs"])["net_$"].idxmax()]
                         .pivot(index="T_delta", columns="T_abs", values="n_trades"))
    print(best_per_combo_n.fillna(0).astype(int).to_string())

    print("\n=== HEATMAP: PF per (delta, absorption) at best-TP cell ===")
    best_per_combo_pf = (df.loc[df.groupby(["T_delta", "T_abs"])["net_$"].idxmax()]
                          .pivot(index="T_delta", columns="T_abs", values="PF"))
    print(best_per_combo_pf.fillna(0).round(2).to_string())

    # 3) Top 10 with reasonable trade count
    print("\n=== TOP 10 (n_trades >= 100) ===")
    top = df[df["n_trades"] >= 100].sort_values("net_$", ascending=False).head(10)
    print(top.to_string(index=False))

    # 4) Comparison row
    print("\n=== COMPARISON ===")
    print(f"  Locked 5-min Fabio:      709 trades, 53.2% WR, $151,265 net, PF 1.33, MaxDD -$20,850")
    print(f"  DELTA 600 only (best):   866 trades, 43.0% WR, $134,471 net, PF 1.25, MaxDD -$23,362")
    print(f"  ABSORPTION 60 only:      751 trades, 49.1% WR, $129,710 net, PF 1.28, MaxDD -$29,085")
    best_combo = df[df["n_trades"] >= 100].sort_values("net_$", ascending=False).iloc[0]
    print(f"  BEST COMBO:              {best_combo['n_trades']} trades, {best_combo['wr%']}% WR, "
          f"${best_combo['net_$']:,.0f}, PF {best_combo['PF']}, MaxDD ${best_combo['MaxDD_$']:,.0f}"
          f"  (D={best_combo['T_delta']}, A={best_combo['T_abs']}, TP={best_combo['tp_rr']})")


if __name__ == "__main__":
    main()
