"""
Sweep all robust SL/TP params through two filter configurations:
  1) ADX 20-30, 9:30-4pm, no lunch
  2) Choppy+Moderate regime, 9:30-4pm, no lunch

50/50 chronological IS/OOS split.
Compare filtered vs unfiltered to check if filters improve broadly (not overfit).

Optimization: pre-build minute-resolution regime + ADX lookup per date,
then use pd.merge_asof for fast tagging.
"""

import pickle
import sys
import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import run_backtest

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
TIMEBAR_DIR = DATA_DIR / "timebars_5min"
ROBUST_CSV = Path("C:/trading/nqorderflowbacktester/results/csv/monte_carlo_robust_params_results.csv")

LOOKBACK = 14
R2_TRENDING = 0.6
R2_MODERATE = 0.3
SLOPE_THRESH = 0.5
ADX_PERIOD = 14


def load_5min_bars(date_str):
    fmt = date_str.replace("-", "_")
    path = TIMEBAR_DIR / f"timebars_5min_{fmt}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        bars = pickle.load(f)
    rows = [{"timestamp": b["open_time"], "open": b["open"], "high": b["high"],
             "low": b["low"], "close": b["close"]} for b in bars]
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    return df


def compute_adx(df, period=ADX_PERIOD):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    up = high - high.shift()
    dn = low.shift() - low
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def rolling_regime(df_5m, lookback=LOOKBACK):
    """Compute regime label at each 5-min bar using rolling linreg."""
    closes = df_5m["close"].values
    n = len(closes)
    regimes = ["unknown"] * n

    for i in range(lookback, n):
        y = closes[i - lookback:i].astype(float)
        x = np.arange(lookback, dtype=float)
        xm, ym = x.mean(), y.mean()
        ss_xy = ((x - xm) * (y - ym)).sum()
        ss_xx = ((x - xm) ** 2).sum()
        ss_yy = ((y - ym) ** 2).sum()
        if ss_xx == 0 or ss_yy == 0:
            regimes[i] = "choppy"
            continue
        r2 = (ss_xy ** 2) / (ss_xx * ss_yy)
        slope = ss_xy / ss_xx
        ns = slope / (ym if ym != 0 else 1) * 10000
        if r2 > R2_TRENDING and ns > SLOPE_THRESH:
            regimes[i] = "trending_up"
        elif r2 > R2_TRENDING and ns < -SLOPE_THRESH:
            regimes[i] = "trending_down"
        elif r2 < R2_MODERATE or abs(ns) <= SLOPE_THRESH:
            regimes[i] = "choppy"
        else:
            regimes[i] = "moderate"

    return pd.Series(regimes, index=df_5m.index)


def build_lookup_tables():
    """Build per-date ADX and regime Series indexed by 5-min bar timestamps."""
    from run_backtest import VWAP_REACTION_CACHE_DIR
    dates = sorted([f.stem for f in VWAP_REACTION_CACHE_DIR.glob("*.pkl")])
    print(f"Building lookup tables for {len(dates)} dates...", flush=True)

    # Build a single combined DataFrame with regime + adx at each 5-min bar
    all_rows = []
    for date_str in dates:
        df_5m = load_5min_bars(date_str)
        if df_5m is None or len(df_5m) <= max(LOOKBACK, ADX_PERIOD * 2):
            continue
        adx_s = compute_adx(df_5m)
        regime_s = rolling_regime(df_5m)
        tmp = pd.DataFrame({"adx": adx_s, "regime": regime_s}, index=df_5m.index)
        all_rows.append(tmp)

    lookup = pd.concat(all_rows).sort_index()
    lookup.index.name = "timestamp"
    print(f"  {len(lookup)} lookup rows built.", flush=True)
    return lookup


def tag_trades_fast(trades_df, lookup):
    """Tag trades using merge_asof against pre-computed lookup."""
    trades = trades_df.copy()
    trades["entry_dt"] = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_convert(ET)

    # merge_asof needs sorted keys
    trades = trades.sort_values("entry_dt").reset_index(drop=True)
    lk = lookup.reset_index()
    lk = lk.rename(columns={"timestamp": "entry_dt"})

    merged = pd.merge_asof(
        trades[["entry_dt"]],
        lk,
        on="entry_dt",
        direction="backward",
    )

    trades["regime"] = merged["regime"].values
    trades["adx"] = merged["adx"].values
    return trades


def apply_time_filter(df):
    h = df["entry_dt"].dt.hour
    m = df["entry_dt"].dt.minute
    em = h * 60 + m
    return df[(em >= 570) & (em < 960) & ~((em >= 720) & (em < 780))].copy()


def calc(df):
    n = len(df)
    if n < 5:
        return {"trades": n, "wr": 0, "pf": 0, "exp": 0, "total": 0, "dd": 0}
    w = df[df["pnl_dollars"] > 0]
    l = df[df["pnl_dollars"] <= 0]
    gp = w["pnl_dollars"].sum() if len(w) else 0
    gl = abs(l["pnl_dollars"].sum()) if len(l) else 1
    cum = df["pnl_dollars"].cumsum()
    dd = float((cum - cum.cummax()).min())
    return {
        "trades": n, "wr": len(w)/n*100,
        "pf": gp/gl if gl > 0 else 999,
        "exp": df["pnl_dollars"].mean(),
        "total": df["pnl_dollars"].sum(), "dd": dd,
    }


def main():
    robust = pd.read_csv(ROBUST_CSV)
    combos = robust[["stop_mult", "tp_mult"]].drop_duplicates().values.tolist()
    combos.sort()
    print(f"Testing {len(combos)} SL/TP combos × 2 filters + baseline")
    print(f"50/50 IS/OOS chronological split\n")

    lookup = build_lookup_tables()

    results = []
    for i, (sl, tp) in enumerate(combos):
        if (i + 1) % 40 == 0 or i == 0:
            print(f"  {i+1}/{len(combos)}: SL={sl} TP={tp}...", flush=True)

        # Suppress backtest print output
        with redirect_stdout(io.StringIO()):
            trades_df = run_backtest(sl_mult=sl, tp_mult=tp)

        if len(trades_df) < 10:
            continue

        tagged = tag_trades_fast(trades_df, lookup)
        tf = apply_time_filter(tagged)
        if len(tf) < 10:
            continue

        tf = tf.sort_values("entry_dt").reset_index(drop=True)
        sp = len(tf) // 2
        b_is, b_oos = calc(tf.iloc[:sp]), calc(tf.iloc[sp:])

        af = tf[(tf["adx"] >= 20) & (tf["adx"] < 30)].reset_index(drop=True)
        asp = len(af) // 2
        a_is = calc(af.iloc[:asp]) if asp >= 5 else None
        a_oos = calc(af.iloc[asp:]) if asp >= 5 else None

        rf = tf[tf["regime"].isin(["choppy", "moderate"])].reset_index(drop=True)
        rsp = len(rf) // 2
        r_is = calc(rf.iloc[:rsp]) if rsp >= 5 else None
        r_oos = calc(rf.iloc[rsp:]) if rsp >= 5 else None

        row = {"sl": sl, "tp": tp}
        for pfx, s in [("base_is", b_is), ("base_oos", b_oos)]:
            for k, v in s.items(): row[f"{pfx}_{k}"] = v
        if a_is and a_oos:
            for pfx, s in [("adx_is", a_is), ("adx_oos", a_oos)]:
                for k, v in s.items(): row[f"{pfx}_{k}"] = v
        if r_is and r_oos:
            for pfx, s in [("reg_is", r_is), ("reg_oos", r_oos)]:
                for k, v in s.items(): row[f"{pfx}_{k}"] = v

        results.append(row)

    df = pd.DataFrame(results)

    # ── TOP 25 ADX ────────────────────────────────────────────────────────────
    hdr = f"{'SL':>4s} {'TP':>4s} | {'ISTrd':>5s} {'ISPF':>5s} {'ISExp':>7s} {'ISPnL':>9s} {'ISDD':>8s} | {'OOSTrd':>6s} {'OOSPF':>5s} {'OOSExp':>7s} {'OOSPnL':>9s} {'OOSDD':>8s} | {'BasPF':>5s}"

    print("\n" + "=" * 120)
    print("TOP 25 — ADX 20-30 FILTER (by OOS Profit Factor)")
    print("=" * 120)
    adx_ok = df.dropna(subset=["adx_oos_pf"])
    adx_top = adx_ok.nlargest(25, "adx_oos_pf")
    print(hdr); print("-" * 120)
    for _, r in adx_top.iterrows():
        print(f"{r['sl']:4.1f} {r['tp']:4.1f} | "
              f"{r['adx_is_trades']:5.0f} {r['adx_is_pf']:5.2f} ${r['adx_is_exp']:>+6,.0f} ${r['adx_is_total']:>+8,.0f} ${r['adx_is_dd']:>+7,.0f} | "
              f"{r['adx_oos_trades']:6.0f} {r['adx_oos_pf']:5.2f} ${r['adx_oos_exp']:>+6,.0f} ${r['adx_oos_total']:>+8,.0f} ${r['adx_oos_dd']:>+7,.0f} | "
              f"{r['base_oos_pf']:5.2f}")

    print("\n" + "=" * 120)
    print("TOP 25 — CHOPPY+MODERATE REGIME FILTER (by OOS Profit Factor)")
    print("=" * 120)
    reg_ok = df.dropna(subset=["reg_oos_pf"])
    reg_top = reg_ok.nlargest(25, "reg_oos_pf")
    print(hdr); print("-" * 120)
    for _, r in reg_top.iterrows():
        print(f"{r['sl']:4.1f} {r['tp']:4.1f} | "
              f"{r['reg_is_trades']:5.0f} {r['reg_is_pf']:5.2f} ${r['reg_is_exp']:>+6,.0f} ${r['reg_is_total']:>+8,.0f} ${r['reg_is_dd']:>+7,.0f} | "
              f"{r['reg_oos_trades']:6.0f} {r['reg_oos_pf']:5.2f} ${r['reg_oos_exp']:>+6,.0f} ${r['reg_oos_total']:>+8,.0f} ${r['reg_oos_dd']:>+7,.0f} | "
              f"{r['base_oos_pf']:5.2f}")

    # ── OVERFIT CHECK ─────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("OVERFIT CHECK — Does the filter improve OOS PF vs unfiltered baseline?")
    print("=" * 100)

    for label, oos_col, is_col in [
        ("ADX 20-30", "adx_oos_pf", "adx_is_pf"),
        ("Choppy+Moderate", "reg_oos_pf", "reg_is_pf"),
    ]:
        valid = df.dropna(subset=[oos_col])
        n = len(valid)
        if n == 0:
            continue

        improved = (valid[oos_col] > valid["base_oos_pf"]).sum()
        worse = (valid[oos_col] < valid["base_oos_pf"]).sum()
        pct = improved / n * 100

        avg_is = valid[is_col].mean()
        avg_oos = valid[oos_col].mean()
        avg_base = valid["base_oos_pf"].mean()
        deg = (avg_is - avg_oos) / avg_is * 100 if avg_is > 0 else 0

        print(f"\n{label} ({n} combos):")
        print(f"  Improved OOS PF:     {improved}/{n} ({pct:.1f}%)")
        print(f"  Worse OOS PF:        {worse}/{n} ({worse/n*100:.1f}%)")
        print(f"  Avg filter IS PF:    {avg_is:.3f}")
        print(f"  Avg filter OOS PF:   {avg_oos:.3f}  (IS->OOS degradation: {deg:.1f}%)")
        print(f"  Avg baseline OOS PF: {avg_base:.3f}")
        print(f"  Avg OOS PF lift:     {avg_oos - avg_base:+.3f}")

        print(f"\n  By baseline OOS PF bucket:")
        for lo, hi, lbl in [(0, 1.0, "Losers (<1.0)"), (1.0, 1.3, "Weak (1.0-1.3)"),
                             (1.3, 1.6, "Decent (1.3-1.6)"), (1.6, 999, "Strong (1.6+)")]:
            bk = valid[(valid["base_oos_pf"] >= lo) & (valid["base_oos_pf"] < hi)]
            if len(bk) == 0:
                continue
            bi = (bk[oos_col] > bk["base_oos_pf"]).sum()
            print(f"    {lbl:<20s}: {bi}/{len(bk)} improved ({bi/len(bk)*100:.0f}%)")

    print()
    out = DATA_DIR / "vwap_filter_sweep_results.csv"
    df.to_csv(out, index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
