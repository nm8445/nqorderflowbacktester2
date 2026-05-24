"""20-min ORB-break entry, sweep delta vs lower-half-absorption filters.

Setup:
  - ORB: 5-min bars, closes 08:35-09:00 ET (same as locked Fabio)
  - Entry trigger: first 20-min bar after 09:00 ET whose close > ORB_High
  - Entry price = that 20-min bar's close
  - Trade window: 20-min closes between 09:20 and 14:00 ET
  - One trade per day max
  - SL = ORB_Low (static)
  - TP = entry + R * (entry - ORB_Low) -- sweep R in [1.00, 1.25, ..., 4.00]
  - EOD: first 5-min bar close >= 14:00 (walk forward on 5-min for exit precision)

Filters compared (all sweep against the same trade-window/exit logic):
  A) NO_FILTER  -- baseline: any 20-min break enters
  B) DELTA      -- 20-min bar total (buy - sell) >= threshold T_delta
                   sweep T_delta in [0, 100, 200, 300, 400, 500, 600, 800, 1000]
  C) ABSORPTION -- count of LOWER-HALF levels where (sell_vol - buy_vol) >= T_lvl
                   pass if count >= MIN_LEVELS (fixed at 2)
                   sweep T_lvl in [30, 40, ..., 200]
                   "lower half" = level_price <= (bar_high + bar_low) / 2

Costs:
  - 1 tick slip per side (round-trip = 0.5 pt = $10)
  - $5 RT commission
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

VOL_PARQUET = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
OUT_CSV     = Path("C:/trading/nqorderflowbacktester/scripts/fabio_orb/20min_break_filter_sweep.csv")
ET = "America/New_York"

ORB_START_HHMM = 830
ORB_END_HHMM   = 900
TRADE_END_HHMM = 1400
EOD_HHMM       = 1400

DELTA_THRESHOLDS  = [0, 100, 200, 300, 400, 500, 600, 800, 1000]
ABS_THRESHOLDS    = list(range(30, 201, 10))
ABS_MIN_LEVELS    = 2
TP_RR_VALUES      = [round(x, 2) for x in np.arange(1.0, 4.01, 0.25)]

TICK = 0.25
TICK_VAL = 5.0
DOLLARS_PER_PT = TICK_VAL / TICK    # 20
SLIP_PTS_RT = SLIP_TICKS = 1 * TICK * 2   # 0.5 pt round trip
COMM_RT = 5.0


# ============================== data prep ==============================

def load_5min_bars() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per_level_df, ohlc_5min_df)."""
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
    print(f"  {len(ohlc):,} 5-min bars, {len(df):,} per-level rows, "
          f"{ohlc['date'].min()} -> {ohlc['date'].max()}")
    return df, ohlc


def aggregate_20min(per_level: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-level 5-min data into 20-min bars.

    Returns a frame indexed by 20-min bar_open_time with columns:
      open, high, low, close, total_delta,
      abs_count_<T> for T in ABS_THRESHOLDS
    """
    # 20-min bucket = floor(bar_open_time to 20 minutes)
    df = per_level.copy()
    df["bar_open_20min"] = df["bar_open_time"].dt.floor("20min")

    # OHLC roll-up: use the 5-min bar properties (one row per 5-min bar suffices).
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

    # Per-level aggregate within 20-min bucket
    lvl20 = (df.groupby(["bar_open_20min", "level_price"], as_index=False)
               .agg(buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum")))

    # total delta per 20-min bar
    tot = lvl20.groupby("bar_open_20min", as_index=False).agg(
        total_buy=("buy_vol", "sum"), total_sell=("sell_vol", "sum"))
    tot["total_delta"] = tot["total_buy"] - tot["total_sell"]
    ohlc20 = ohlc20.merge(tot[["bar_open_20min", "total_delta"]], on="bar_open_20min", how="left")

    # absorption counts: for each 20-min bar, count lower-half levels with (sell-buy) >= T
    # merge bar_high/bar_low into level frame
    lvl20 = lvl20.merge(ohlc20[["bar_open_20min", "high", "low"]], on="bar_open_20min", how="left")
    lvl20["mid"] = (lvl20["high"] + lvl20["low"]) / 2.0
    lvl20["is_lower_half"] = lvl20["level_price"] <= lvl20["mid"]
    lvl20["seller_pressure"] = lvl20["sell_vol"] - lvl20["buy_vol"]
    lower = lvl20[lvl20["is_lower_half"]].copy()

    print(f"  Computing absorption counts at {len(ABS_THRESHOLDS)} thresholds...")
    for T in ABS_THRESHOLDS:
        col = f"abs_count_{T}"
        # count of qualifying levels per 20-min bar
        passes = lower[lower["seller_pressure"] >= T].groupby("bar_open_20min").size()
        ohlc20 = ohlc20.merge(passes.rename(col), left_on="bar_open_20min", right_index=True, how="left")
        ohlc20[col] = ohlc20[col].fillna(0).astype(int)

    print(f"  Built {len(ohlc20):,} 20-min bars")
    return ohlc20


# ============================== backtest ==============================

def find_entry_20min(bars20_day: pd.DataFrame, orb_high: float, orb_low: float,
                     filter_col: str | None, filter_threshold: float, filter_type: str) -> dict | None:
    """Find first 20-min bar that:
      - closes above ORB_High
      - passes filter (NO_FILTER/DELTA/ABSORPTION)
      - ORB_Low < close sanity
      - hhmm <= TRADE_END_HHMM

    Returns dict with entry details or None.
    """
    for _, bar in bars20_day.iterrows():
        if bar["hhmm"] > TRADE_END_HHMM: break
        if bar["close"] <= orb_high: continue
        if orb_low >= bar["close"]: continue
        # Filter
        if filter_type == "NO_FILTER":
            pass
        elif filter_type == "DELTA":
            if bar["total_delta"] < filter_threshold: continue
        elif filter_type == "ABSORPTION":
            col = f"abs_count_{int(filter_threshold)}"
            if bar[col] < ABS_MIN_LEVELS: continue
        return {
            "entry_time": bar["bar_close_20min"], "entry_price": float(bar["close"]),
            "entry_hhmm": int(bar["hhmm"]),
        }
    return None


def simulate_exit_5min(bars5_after: pd.DataFrame, entry_price: float, orb_low: float, tp_rr: float) -> tuple[float, str]:
    """Walk forward on 5-min bars after entry. Return (exit_price, reason)."""
    sl = orb_low
    risk = entry_price - sl
    tp = entry_price + tp_rr * risk
    for _, bar in bars5_after.iterrows():
        if bar["low"] <= sl: return sl, "SL"
        if bar["high"] >= tp: return tp, "TP"
        if bar["hhmm"] >= EOD_HHMM:
            return float(bar["close"]), "EOD"
    # Out of bars
    last = bars5_after.iloc[-1] if len(bars5_after) > 0 else None
    return (float(last["close"]) if last is not None else entry_price), "EOD"


def run_backtest(bars20: pd.DataFrame, bars5: pd.DataFrame,
                 filter_type: str, filter_threshold: float, tp_rr: float) -> dict:
    bars20_by_day = {d: g for d, g in bars20.groupby("date")}
    bars5_by_day  = {d: g for d, g in bars5.groupby("date")}

    trades = []
    for d, bars20_d in bars20_by_day.items():
        bars5_d = bars5_by_day.get(d)
        if bars5_d is None: continue

        # Build ORB from 5-min bars (closes 08:35-09:00)
        orb_bars = bars5_d[(bars5_d["hhmm"] > ORB_START_HHMM) & (bars5_d["hhmm"] <= ORB_END_HHMM)]
        if len(orb_bars) == 0: continue
        orb_high = float(orb_bars["high"].max())
        orb_low  = float(orb_bars["low"].min())

        # Post-ORB 20-min bars only
        post_orb_20 = bars20_d[(bars20_d["hhmm"] > ORB_END_HHMM) & (bars20_d["hhmm"] <= TRADE_END_HHMM)]
        if len(post_orb_20) == 0: continue

        entry = find_entry_20min(post_orb_20, orb_high, orb_low,
                                  None, filter_threshold, filter_type)
        if entry is None: continue

        # Walk forward on 5-min for exit, starting AFTER the 20-min bar's close
        post_entry_5 = bars5_d[bars5_d["bar_open_time"] >= entry["entry_time"]]
        if len(post_entry_5) == 0: continue

        exit_price, reason = simulate_exit_5min(post_entry_5, entry["entry_price"], orb_low, tp_rr)
        gross_pts = exit_price - entry["entry_price"]
        net_pts = gross_pts - SLIP_PTS_RT
        pnl = net_pts * DOLLARS_PER_PT - COMM_RT
        trades.append({
            "date": d, "entry_price": entry["entry_price"], "exit_price": exit_price,
            "reason": reason, "pnl_$": pnl, "risk_pts": entry["entry_price"] - orb_low,
            "entry_hhmm": entry["entry_hhmm"],
        })

    if not trades:
        return {"filter": filter_type, "threshold": filter_threshold, "tp_rr": tp_rr,
                "n_trades": 0, "wr%": 0, "net_$": 0, "PF": 0, "MaxDD_$": 0}

    df = pd.DataFrame(trades).sort_values("date").reset_index(drop=True)
    df["cum"] = df["pnl_$"].cumsum()
    df["peak"] = df["cum"].cummax()
    df["dd"] = df["cum"] - df["peak"]
    wins = df[df["pnl_$"] > 0]; losses = df[df["pnl_$"] < 0]
    pf = wins["pnl_$"].sum() / abs(losses["pnl_$"].sum()) if len(losses) > 0 else float("inf")
    return {
        "filter": filter_type, "threshold": filter_threshold, "tp_rr": tp_rr,
        "n_trades": len(df), "wr%": round(len(wins) / len(df) * 100, 1),
        "net_$": round(df["pnl_$"].sum(), 0),
        "PF": round(pf, 2),
        "MaxDD_$": round(df["dd"].min(), 0),
        "avg_$": round(df["pnl_$"].mean(), 1),
        "TP%": round((df["reason"] == "TP").mean() * 100, 1),
        "SL%": round((df["reason"] == "SL").mean() * 100, 1),
        "EOD%": round((df["reason"] == "EOD").mean() * 100, 1),
    }


def main():
    per_level, bars5 = load_5min_bars()
    bars20 = aggregate_20min(per_level)

    print(f"\nSweeping configs...")
    rows = []
    # No-filter baseline
    for tp in TP_RR_VALUES:
        r = run_backtest(bars20, bars5, "NO_FILTER", 0, tp)
        rows.append(r)
    # Delta sweep
    for T in DELTA_THRESHOLDS:
        for tp in TP_RR_VALUES:
            r = run_backtest(bars20, bars5, "DELTA", T, tp)
            rows.append(r)
    # Absorption sweep
    for T in ABS_THRESHOLDS:
        for tp in TP_RR_VALUES:
            r = run_backtest(bars20, bars5, "ABSORPTION", T, tp)
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved full sweep: {OUT_CSV} ({len(df)} rows)")

    # Top picks
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
    print("\n=== NO-FILTER (raw 20-min break) ===")
    nofilt = df[df["filter"] == "NO_FILTER"].sort_values("net_$", ascending=False)
    print(nofilt.head(6).to_string(index=False))

    print("\n=== BEST DELTA per threshold ===")
    delta = df[df["filter"] == "DELTA"]
    best_delta = delta.loc[delta.groupby("threshold")["net_$"].idxmax()].sort_values("threshold")
    print(best_delta.to_string(index=False))

    print("\n=== BEST ABSORPTION per threshold ===")
    abso = df[df["filter"] == "ABSORPTION"]
    best_abs = abso.loc[abso.groupby("threshold")["net_$"].idxmax()].sort_values("threshold")
    print(best_abs.to_string(index=False))

    print("\n=== TOP 10 OVERALL (any filter) ===")
    top = df[df["n_trades"] >= 100].sort_values("net_$", ascending=False).head(10)
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
