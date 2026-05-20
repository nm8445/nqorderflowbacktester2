"""One-off: compute QQQ EOD 0-1 DTE levels from 2026-03-26 settle data,
showing what would have been available at the 2026-03-27 open."""

import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import databento as db
from scipy.stats import norm

DATE = dt.date(2026, 3, 26)
QQQ_ROOT = Path("D:/trading_pythonbacktest_data/QQQ_thetadata")
NDX_ROOT = Path("D:/trading_pythonbacktest_data/NDX_thetadata")
DBN_DIR  = Path("D:/trading_pythonbacktest_data/dbn")
R = 0.05; Q_QQQ = 0.005

# ---------- NQ at 17:00 ET on 3/26 to compute ratio ----------
def nq_at_17(date):
    fname = f"NQ_c_0_mbp-1_{date.isoformat()}_{(date + dt.timedelta(days=2)).isoformat()}.dbn"
    p = DBN_DIR / fname
    if not p.exists():
        # Try simple naming
        p2 = DBN_DIR / f"{date.isoformat()}.dbn"
        if p2.exists(): p = p2
        else:
            print(f"DBN not found"); return None
    store = db.DBNStore.from_file(str(p))
    df = store.to_df()
    target = pd.Timestamp(f"{date.isoformat()} 16:59:30",
                          tz="America/New_York").tz_convert("UTC")
    near = df.loc[target - pd.Timedelta(seconds=60): target + pd.Timedelta(seconds=30)]
    trades = near[near["action"] == "T"]
    return float(trades.iloc[-1]["price"]) if not trades.empty else None


nq_17 = nq_at_17(DATE)
print(f"NQ at ~17:00 ET on {DATE}: {nq_17}")

# ---------- QQQ EOD greeks for 3/26 ----------
g = pd.read_parquet(QQQ_ROOT / DATE.isoformat() / "greeks_eod.parquet")
o = pd.read_parquet(QQQ_ROOT / DATE.isoformat() / "open_interest.parquet")
g["expiration"] = pd.to_datetime(g["expiration"])
o["expiration"] = pd.to_datetime(o["expiration"])
g["dte"] = (g["expiration"] - pd.Timestamp(DATE)).dt.days
g_01 = g[(g["dte"] >= 0) & (g["dte"] <= 1)].copy()
print(f"QQQ chain rows in 0-1 DTE: {len(g_01)}")

spot = float(g_01["underlying_price"].iloc[0])
print(f"QQQ spot at 3/26 EOD: {spot:.2f}")
ratio = nq_17 / spot if nq_17 else 41.30
print(f"QQQ-NQ ratio: {ratio:.3f}")

# ATM IV
sub_atm = g_01[g_01["dte"] > 0].copy()
if not sub_atm.empty:
    min_dte = int(sub_atm["dte"].min())
    sub_atm = sub_atm[sub_atm["dte"] == min_dte]
    sub_atm["dist"] = (sub_atm["strike"] - spot).abs()
    iv = float(sub_atm.nsmallest(4, "dist")["implied_vol"].mean())
    em = spot * iv * np.sqrt(1/252)
else:
    iv = 0.20; em = spot * 0.005
print(f"ATM IV: {iv:.4f}    1D EM: ±{em:.2f}")

# Compute net GEX, net DEX, call_only_gex, put_only_gex per strike
chain = g_01.merge(o[["strike","right","expiration","open_interest"]],
                   on=["strike","right","expiration"], how="left")
chain["gex_abs"] = chain["gamma"] * chain["open_interest"].fillna(0) * 100 * spot**2
chain["signed_gex"] = chain["gex_abs"]
chain.loc[chain["right"].str.upper()=="PUT", "signed_gex"] *= -1
chain["signed_dex"] = chain["delta"] * chain["open_interest"].fillna(0) * 100 * spot
is_c = (chain["right"].str.upper() == "CALL")
chain["call_only_gex"] = np.where(is_c, chain["gex_abs"], 0.0)
chain["put_only_gex"]  = np.where(is_c, 0.0, chain["gex_abs"])
chain["call_only_dex"] = np.where(is_c, chain["signed_dex"], 0.0)
chain["put_only_dex"]  = np.where(is_c, 0.0, chain["signed_dex"])

by = (chain.groupby("strike").agg(
    net_gex=("signed_gex","sum"),
    net_dex=("signed_dex","sum"),
    call_only_gex=("call_only_gex","sum"),
    put_only_gex=("put_only_gex","sum"),
    call_only_dex=("call_only_dex","sum"),
    put_only_dex=("put_only_dex","sum"),
).reset_index().sort_values("strike"))

# 1D EM window
win = by[(by["strike"] >= spot - em) & (by["strike"] <= spot + em)].copy()
above = win[win["strike"] >= spot]
below = win[win["strike"] <  spot]

# Call Resistance, Put Support
cr_strike = float(above.loc[above["call_only_gex"].idxmax(), "strike"]) if not above.empty and above["call_only_gex"].max() > 0 else None
ps_strike = float(below.loc[below["put_only_gex"].idxmax(),  "strike"]) if not below.empty and below["put_only_gex"].max() > 0 else None

# HVL: cumulative-GEX flip in window
def hvl_calc(by, spot, max_pct=0.05):
    s = by.sort_values("strike").reset_index(drop=True).copy()
    s["cum"] = s["net_gex"].cumsum()
    band = spot * max_pct
    flips = []
    for i in range(1, len(s)):
        K = float(s.iloc[i]["strike"])
        if abs(K - spot) > band: continue
        if (s.iloc[i-1]["cum"] > 0) != (s.iloc[i]["cum"] > 0):
            flips.append(K)
    return min(flips, key=lambda k: abs(k-spot)) if flips else None
hvl_strike = hvl_calc(by, spot)

# GEX 1-5: combined |GEX|+|DEX| normalized, excluding CR/PS/HVL
excluded = {cr_strike, ps_strike, hvl_strike}; excluded.discard(None)
rest = win[~win["strike"].isin(excluded)].copy()
rest["abs_gex"] = rest["net_gex"].abs()
rest["abs_dex"] = rest["net_dex"].abs()
max_g = rest["abs_gex"].max() or 1.0
max_d = rest["abs_dex"].max() or 1.0
rest["score"] = rest["abs_gex"]/max_g + rest["abs_dex"]/max_d
top5 = rest.sort_values("score", ascending=False).head(5)

# ---------- Output ----------
print()
print("=" * 80)
print(f"QQQ 0-1 DTE LEVELS from 2026-03-26 EOD (usable at 3/27 open)")
print("=" * 80)
print(f"  QQQ spot:    {spot:.2f}    (NQ-equiv: {spot*ratio:.0f})")
print(f"  ATM IV:      {iv:.4f}      1D EM window: ±{em:.2f}")
print(f"  Window:      [{spot-em:.2f}, {spot+em:.2f}]")
print()
def line(label, K):
    if K is None: return f"  {label:<20} --"
    return f"  {label:<20} QQQ {K:>7.2f}    NQ {K*ratio:>9.2f}"

print(line("Call Resistance", cr_strike) + (f"   (calls-only gex +{above.loc[above['strike']==cr_strike,'call_only_gex'].iloc[0]:.2e})" if cr_strike else ""))
print(line("Put Support", ps_strike) + (f"   (puts-only gex {-below.loc[below['strike']==ps_strike,'put_only_gex'].iloc[0]:+.2e})" if ps_strike else ""))
print(line("HVL (gamma flip)", hvl_strike))
print()
print("  GEX 1..5 (combined |Net GEX| + |Net DEX|, excl CR/PS/HVL):")
for i, r in enumerate(top5.itertuples(index=False), 1):
    sign = "+" if r.net_gex >= 0 else ""
    print(f"    GEX_{i}            QQQ {r.strike:>7.2f}    NQ {r.strike*ratio:>9.2f}   "
          f"net_gex={sign}{r.net_gex:.2e}  net_dex={r.net_dex:+.2e}")
