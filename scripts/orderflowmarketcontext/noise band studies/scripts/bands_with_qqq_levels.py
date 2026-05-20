"""Band-break trades with QQQ 0-1 DTE level interaction analysis.

For each band-break day (above_only or below_only) from the 14-day overnight
band study:
  1. Compute QQQ 0-1 DTE levels on the prior-EOD chain:
     - Call Resistance (max calls-only gamma above spot)
     - Put Support (max puts-only gamma below spot)
     - GEX 1..5 (top combined |GEX|+|DEX|, excluding CR/PS, in 1D EM window)
  2. Convert each level to NQ space via prior-day settle ratio
  3. Filter to levels OUTSIDE the bands in trade direction:
     - Above-only entries: only levels above upper_band
     - Below-only entries: only levels below lower_band
  4. Scan 5-min bars from break time to 17:00 ET; detect first 5-min close
     that crosses each level
  5. Measure PnL from level-touch time to 17:00 ET close
  6. Aggregate: per level, P(continue), mean PnL, t-stat, p-value, with
     gamma-regime sub-splits

Output:
  - per-trade-level parquet at .../bands_with_levels.parquet
  - printed cohort tables
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

QQQ_ROOT      = Path("D:/trading_pythonbacktest_data/QQQ_thetadata")
NQ_1MIN       = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
BAND_PARQUET  = Path(__file__).parent / "overnight_band_per_day.parquet"
OUT_DIR       = Path(__file__).parent
OUT_PARQUET   = OUT_DIR / "bands_with_levels.parquet"

EM_BAND = 0.01  # 1% safety band for ATM IV-based EM window default
EM_LOOKBACK_FALLBACK = 0.005  # if IV missing


# ------------------------------ Level computation (mirrors menthorq_style) ------------------------------

def atm_iv_qqq(g: pd.DataFrame, spot: float) -> float:
    """Mean implied_vol of 4 strikes nearest spot at the shortest non-zero DTE."""
    sub = g[g["dte"] > 0].copy()
    if sub.empty: return float("nan")
    min_dte = int(sub["dte"].min())
    sub = sub[sub["dte"] == min_dte].copy()
    sub["dist"] = (sub["strike"] - spot).abs()
    near = sub.nsmallest(4, "dist")
    return float(near["implied_vol"].mean())


def compute_qqq_levels(prev_date: dt.date, today: dt.date) -> dict | None:
    """Compute 0-1 DTE QQQ levels for trading day `today` using `prev_date`'s
    settled greeks + OI. Returns dict with strike values:
        {cr, ps, gex_1..gex_5, qqq_spot}
    All in QQQ strike units."""
    g_path = QQQ_ROOT / prev_date.isoformat() / "greeks_eod.parquet"
    o_path = QQQ_ROOT / prev_date.isoformat() / "open_interest.parquet"
    if not g_path.exists() or not o_path.exists(): return None
    g = pd.read_parquet(g_path); o = pd.read_parquet(o_path)
    g["expiration"] = pd.to_datetime(g["expiration"])
    o["expiration"] = pd.to_datetime(o["expiration"])
    end = today + dt.timedelta(days=2)
    g = g[(g["expiration"].dt.date >= today) & (g["expiration"].dt.date <= end)]
    if g.empty: return None
    g["dte"] = (g["expiration"] - pd.Timestamp(today)).dt.days
    g["dte"] = g["dte"].clip(lower=0)

    spot = float(g["underlying_price"].iloc[0])
    iv = atm_iv_qqq(g, spot)
    if not np.isfinite(iv) or iv <= 0:
        em = spot * EM_LOOKBACK_FALLBACK
    else:
        em = spot * iv * (1.0 / 252.0) ** 0.5

    # Per-strike net GEX, net DEX, plus call-only and put-only gamma exposure
    chain = g.merge(o[["strike","right","expiration","open_interest"]],
                    on=["strike","right","expiration"], how="left")
    chain["gex_abs"] = chain["gamma"] * chain["open_interest"].fillna(0) * 100 * spot**2
    chain["signed_gex"] = chain["gex_abs"]
    chain.loc[chain["right"].str.upper()=="PUT", "signed_gex"] *= -1
    chain["signed_dex"] = chain["delta"] * chain["open_interest"].fillna(0) * 100 * spot
    is_c = (chain["right"].str.upper() == "CALL")
    chain["call_only_gex"] = np.where(is_c, chain["gex_abs"], 0.0)
    chain["put_only_gex"]  = np.where(is_c, 0.0, chain["gex_abs"])

    by = (chain.groupby("strike").agg(net_gex=("signed_gex","sum"),
                                       net_dex=("signed_dex","sum"),
                                       call_only_gex=("call_only_gex","sum"),
                                       put_only_gex=("put_only_gex","sum"))
          .reset_index().sort_values("strike"))

    # 1D EM window
    win = by[(by["strike"] >= spot - em) & (by["strike"] <= spot + em)].copy()
    if win.empty: return None

    above = win[win["strike"] >= spot]
    below = win[win["strike"] <  spot]

    cr_strike = float(above.loc[above["call_only_gex"].idxmax(), "strike"]) if not above.empty and above["call_only_gex"].max() > 0 else None
    ps_strike = float(below.loc[below["put_only_gex"].idxmax(),  "strike"]) if not below.empty and below["put_only_gex"].max() > 0 else None

    excluded = {cr_strike, ps_strike}
    excluded.discard(None)
    rest = win[~win["strike"].isin(excluded)].copy()
    rest["abs_gex"] = rest["net_gex"].abs()
    rest["abs_dex"] = rest["net_dex"].abs()
    max_g = rest["abs_gex"].max() or 1.0
    max_d = rest["abs_dex"].max() or 1.0
    rest["score"] = rest["abs_gex"]/max_g + rest["abs_dex"]/max_d
    top = rest.sort_values("score", ascending=False).head(5)
    gex_strikes = list(top["strike"].values.astype(float))
    while len(gex_strikes) < 5:
        gex_strikes.append(np.nan)

    return {
        "qqq_spot": spot,
        "cr":   cr_strike,
        "ps":   ps_strike,
        "gex_1": gex_strikes[0],
        "gex_2": gex_strikes[1],
        "gex_3": gex_strikes[2],
        "gex_4": gex_strikes[3],
        "gex_5": gex_strikes[4],
    }


# ------------------------------ NQ price loader and 5-min RTH bars ------------------------------

def load_nq_5min() -> dict:
    """Returns {date: pd.Series of 5-min closes during RTH (9:30-17:00 ET)}."""
    df = pd.read_parquet(NQ_1MIN)
    idx = df.index
    if idx.tz is None: idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert("America/New_York")
    nq = df["close"].sort_index()
    rth_mask = (nq.index.time >= dt.time(9, 30)) & (nq.index.time <= dt.time(17, 0))
    rth = nq[rth_mask].resample("5min", origin="epoch").last().dropna()
    rth = rth[(rth.index.time >= dt.time(9, 30)) & (rth.index.time <= dt.time(17, 0))]
    return {d: g for d, g in rth.groupby(rth.index.date)}


# ------------------------------ Driver ------------------------------

LEVEL_NAMES = ["cr", "ps", "gex_1", "gex_2", "gex_3", "gex_4", "gex_5"]


def main():
    print("loading band parquet + 5-min RTH bars...")
    band = pd.read_parquet(BAND_PARQUET)
    band["date"] = pd.to_datetime(band["date"]).dt.date
    band = band.sort_values("date").reset_index(drop=True)
    print(f"  band rows: {len(band)}")
    bars5 = load_nq_5min()
    print(f"  RTH days with 5-min bars: {len(bars5)}")

    # Map each date to its prior trading date (using band table order)
    date_to_prev = {}
    dates_sorted = band["date"].tolist()
    for i in range(1, len(dates_sorted)):
        date_to_prev[dates_sorted[i]] = dates_sorted[i-1]

    out_rows = []
    t0 = time.time()
    valid_band = band[band["break_class_simple_14d"].isin(["above_only","below_only"])].copy()
    print(f"\nprocessing {len(valid_band)} above/below break days (14d lookback)...\n")

    for i, r in enumerate(valid_band.itertuples(index=False), 1):
        d = r.date
        prev = date_to_prev.get(d)
        if prev is None: continue
        levels_qqq = compute_qqq_levels(prev, d)
        if levels_qqq is None: continue

        # Convert QQQ levels to NQ space using prior-day settle ratio
        # ratio = NQ price at 17:00 prev day / QQQ spot at prev day
        # band parquet anchor IS NQ at 17:00 prev day (close_1659 shifted)
        # We use band.anchor (= prior-day NQ close) and levels_qqq["qqq_spot"] (= prior-day QQQ EOD)
        if not np.isfinite(r.anchor) or levels_qqq["qqq_spot"] <= 0:
            continue
        ratio = float(r.anchor) / float(levels_qqq["qqq_spot"])

        # Build NQ-equivalent levels
        nq_levels = {}
        for name in LEVEL_NAMES:
            v = levels_qqq.get(name)
            nq_levels[name] = (float(v) * ratio) if (v is not None and np.isfinite(v)) else np.nan

        # Pull the 5-min bars for this day starting at the entry bar
        bars = bars5.get(d)
        if bars is None or len(bars) < 5: continue
        first_break_time_str = r.first_break_time_14d
        if first_break_time_str is None: continue
        # Find the entry bar timestamp matching first_break_time_14d
        entry_bars = [t for t in bars.index if t.strftime("%H:%M") == first_break_time_str]
        if not entry_bars: continue
        entry_t = entry_bars[0]
        post_entry = bars.loc[entry_t:]
        if post_entry.empty: continue
        entry_price = float(post_entry.iloc[0])
        close_1700 = float(r.close_1700)

        is_long = (r.break_class_simple_14d == "above_only")
        # Bands
        upper_band = float(r.upper_band_14d)
        lower_band = float(r.lower_band_14d)

        # For each level: filter to OUTSIDE the band in trade direction
        for name in LEVEL_NAMES:
            level_nq = nq_levels[name]
            if not np.isfinite(level_nq): continue

            # Filter direction:
            # Long entry: level must be ABOVE upper_band (forward target)
            # Short entry: level must be BELOW lower_band (forward target)
            if is_long:
                if level_nq <= upper_band: continue
            else:
                if level_nq >= lower_band: continue

            # Scan 5-min closes from entry bar onward; first cross
            if is_long:
                cross_mask = post_entry >= level_nq
            else:
                cross_mask = post_entry <= level_nq
            if not cross_mask.any(): continue
            touch_t = post_entry.index[cross_mask][0]
            touch_price = float(post_entry.loc[touch_t])

            # PnL from touch to 17:00 close
            if is_long:
                pnl = close_1700 - touch_price
            else:
                pnl = touch_price - close_1700

            # P(continue past level) at 17:00:
            if is_long:
                continued = close_1700 > level_nq
            else:
                continued = close_1700 < level_nq

            out_rows.append({
                "date": d,
                "direction":   "long" if is_long else "short",
                "level_name":  name,
                "level_nq":    level_nq,
                "entry_price": entry_price,
                "touch_time":  touch_t.strftime("%H:%M"),
                "touch_price": touch_price,
                "close_1700":  close_1700,
                "pnl_from_touch": pnl,
                "continued":   bool(continued),
                "qqq_regime_open": r.qqq_regime_open,
                "ndx_regime_open": r.ndx_regime_open,
                "upper_band":  upper_band,
                "lower_band":  lower_band,
            })

        if i % 100 == 0:
            print(f"  {i}/{len(valid_band)}  elapsed={time.time()-t0:.0f}s  rows={len(out_rows)}")

    out = pd.DataFrame(out_rows)
    out.to_parquet(OUT_PARQUET, compression="zstd", index=False)
    print(f"\nwrote {OUT_PARQUET}  ({len(out)} level-touch rows)")

    # ---------------- Reporting ----------------

    def cohort(label, sub: pd.DataFrame, direction: str | None = None):
        n = len(sub)
        if n < 2:
            return f"  {label:<60}  n={n:>4}  insufficient"
        pnl = sub["pnl_from_touch"].dropna().values
        n = len(pnl)
        p_continue = float(sub["continued"].mean())
        p_profit = float((pnl > 0).mean())
        m = float(pnl.mean())
        t, p = ss.ttest_1samp(pnl, 0)
        sig = "  ***" if p<0.001 else ("   **" if p<0.01 else ("    *" if p<0.05 else "     "))
        return (f"  {label:<60}  n={n:>4}  P(cont)={p_continue:.1%}  P(profit)={p_profit:.1%}  "
                f"mean={m:+6.2f} pts  t={t:+5.2f}  p={p:.4f}{sig}")

    print("\n" + "=" * 100)
    print("LEVEL-INTERACTION RESULTS")
    print("=" * 100)
    print("Sig markers: * p<0.05, ** p<0.01, *** p<0.001")
    print("P(cont) = price closed at 17:00 past the level (further in trade direction)")
    print("P(profit) = pnl from level-touch to 17:00 was positive (in trade direction)")
    print()

    for direction in ["long", "short"]:
        df_d = out[out["direction"] == direction]
        print(f"\n--- {direction.upper()} entries (above_only / below_only) ---")
        print(cohort(f"  ALL levels {direction}", df_d))
        for name in LEVEL_NAMES:
            sub = df_d[df_d["level_name"] == name]
            print(cohort(f"  {name:<6}  ALL gamma", sub))
            for source, regime_col in [("QQQ", "qqq_regime_open"),
                                        ("NDX", "ndx_regime_open")]:
                for regime in ["pos", "neg"]:
                    sub_r = sub[sub[regime_col] == regime]
                    print(cohort(f"    {name:<6}  + {source}_{regime}-gamma", sub_r))


if __name__ == "__main__":
    sys.exit(main())
