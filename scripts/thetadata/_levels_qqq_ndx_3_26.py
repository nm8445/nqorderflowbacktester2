"""QQQ + NDX 0-1 DTE EOD levels from 2026-03-26, showing top 10 GEX strikes
with NQ-equivalent prices. Used to check what levels were nearest 3/27 9:30
NQ open of 23,636."""

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
R = 0.05; Q_NDX = 0.006; Q_QQQ = 0.005

NQ_OPEN_3_27 = 23636.25  # for highlighting nearest levels


# ---------- Helpers ----------
def nq_at_17(date):
    fname = f"NQ_c_0_mbp-1_{date.isoformat()}_{(date + dt.timedelta(days=2)).isoformat()}.dbn"
    p = DBN_DIR / fname
    if not p.exists():
        p = DBN_DIR / f"{date.isoformat()}.dbn"
    store = db.DBNStore.from_file(str(p))
    df = store.to_df()
    target = pd.Timestamp(f"{date.isoformat()} 16:59:30",
                          tz="America/New_York").tz_convert("UTC")
    near = df.loc[target - pd.Timedelta(seconds=60): target + pd.Timedelta(seconds=30)]
    trades = near[near["action"] == "T"]
    return float(trades.iloc[-1]["price"]) if not trades.empty else None


def _d1(S, K, T, sigma, q):
    return (np.log(S/K) + (R - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
def bs_call(S,K,T,sigma,q):
    d1=_d1(S,K,T,sigma,q); d2=d1-sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1)-K*np.exp(-R*T)*norm.cdf(d2)
def bs_put(S,K,T,sigma,q):
    d1=_d1(S,K,T,sigma,q); d2=d1-sigma*np.sqrt(T)
    return K*np.exp(-R*T)*norm.cdf(-d2)-S*np.exp(-q*T)*norm.cdf(-d1)
def bs_vega(S,K,T,sigma,q):
    return S*np.exp(-q*T)*norm.pdf(_d1(S,K,T,sigma,q))*np.sqrt(T)
def bs_gamma(S,K,T,sigma,q):
    return np.exp(-q*T)*norm.pdf(_d1(S,K,T,sigma,q))/(S*sigma*np.sqrt(T))
def bs_call_delta(S,K,T,sigma,q): return np.exp(-q*T)*norm.cdf(_d1(S,K,T,sigma,q))
def bs_put_delta(S,K,T,sigma,q):  return -np.exp(-q*T)*norm.cdf(-_d1(S,K,T,sigma,q))
def iv_vec(price,S,K,T,is_call,q):
    sigma = np.full_like(price, 0.30, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(15):
            bp = np.where(is_call, bs_call(S,K,T,sigma,q), bs_put(S,K,T,sigma,q))
            d = bp-price; v = bs_vega(S,K,T,sigma,q)
            step = np.divide(d, v, out=np.zeros_like(d), where=(v>1e-8))
            sigma = np.clip(sigma-step, 1e-4, 5.0)
        bp = np.where(is_call, bs_call(S,K,T,sigma,q), bs_put(S,K,T,sigma,q))
        bad = (np.abs(bp-price) > np.maximum(0.10, 0.01*price))
    sigma[bad] = np.nan
    return sigma


def fmt_level(strike, nq, gex, dex, scale_label, scale_fn, nq_target):
    nq_val = scale_fn(strike) if scale_fn else nq
    diff = nq_val - nq_target
    flag = " *" if abs(diff) < 50 else ("    +" if diff > 0 else "    -")
    return (f"  K={strike:>8.1f}  NQ={nq_val:>9.2f}  diff_to_open={diff:+7.0f}"
            f"  net_gex={gex:+.2e}  net_dex={dex:+.2e}{flag}")


# ===================== QQQ =====================
print("=" * 100)
print(f"QQQ 0-1 DTE EOD LEVELS from {DATE}  (usable at 3/27 open)")
print("=" * 100)

g = pd.read_parquet(QQQ_ROOT / DATE.isoformat() / "greeks_eod.parquet")
o = pd.read_parquet(QQQ_ROOT / DATE.isoformat() / "open_interest.parquet")
g["expiration"] = pd.to_datetime(g["expiration"])
o["expiration"] = pd.to_datetime(o["expiration"])
g["dte"] = (g["expiration"] - pd.Timestamp(DATE)).dt.days
g_01 = g[(g["dte"] >= 0) & (g["dte"] <= 1)].copy()
spot_q = float(g_01["underlying_price"].iloc[0])
nq_17 = nq_at_17(DATE)
ratio = nq_17 / spot_q
sub_atm = g_01[g_01["dte"] > 0]
if not sub_atm.empty:
    md = int(sub_atm["dte"].min())
    sub_atm2 = sub_atm[sub_atm["dte"] == md]
    iv = float(sub_atm2.assign(d=(sub_atm2["strike"]-spot_q).abs()).nsmallest(4,"d")["implied_vol"].mean())
    em_q = spot_q * iv * np.sqrt(1/252)
else:
    iv = 0.20; em_q = spot_q * 0.005
print(f"  QQQ spot 3/26 EOD: {spot_q:.2f}    NQ-equiv: {spot_q*ratio:.2f}    Ratio: {ratio:.3f}")
print(f"  ATM IV: {iv:.4f}    1D EM window: ±{em_q:.2f}    Window: [{spot_q-em_q:.2f}, {spot_q+em_q:.2f}]")
print(f"  3/27 NQ open: {NQ_OPEN_3_27:.2f}")
print()

chain = g_01.merge(o[["strike","right","expiration","open_interest"]],
                   on=["strike","right","expiration"], how="left")
chain["gex_abs"] = chain["gamma"]*chain["open_interest"].fillna(0)*100*spot_q**2
chain["signed_gex"] = chain["gex_abs"]
chain.loc[chain["right"].str.upper()=="PUT","signed_gex"] *= -1
chain["signed_dex"] = chain["delta"]*chain["open_interest"].fillna(0)*100*spot_q
is_c = (chain["right"].str.upper()=="CALL")
chain["call_only_gex"] = np.where(is_c, chain["gex_abs"], 0.0)
chain["put_only_gex"]  = np.where(is_c, 0.0, chain["gex_abs"])
by_q = (chain.groupby("strike").agg(net_gex=("signed_gex","sum"),
                                     net_dex=("signed_dex","sum"),
                                     call_only_gex=("call_only_gex","sum"),
                                     put_only_gex=("put_only_gex","sum"))
        .reset_index().sort_values("strike"))
win_q = by_q[(by_q["strike"]>=spot_q-em_q) & (by_q["strike"]<=spot_q+em_q)].copy()
above = win_q[win_q["strike"]>=spot_q]; below = win_q[win_q["strike"]<spot_q]
cr_q = float(above.loc[above["call_only_gex"].idxmax(),"strike"]) if not above.empty else None
ps_q = float(below.loc[below["put_only_gex"].idxmax(),"strike"]) if not below.empty else None
print(f"  Call Resistance:  QQQ {cr_q:.2f}    NQ {cr_q*ratio:.2f}    diff_to_open: {cr_q*ratio - NQ_OPEN_3_27:+.0f}")
print(f"  Put Support:      QQQ {ps_q:.2f}    NQ {ps_q*ratio:.2f}    diff_to_open: {ps_q*ratio - NQ_OPEN_3_27:+.0f}")
print()
exc = {cr_q, ps_q}
rest = win_q[~win_q["strike"].isin(exc)].copy()
rest["abs_gex"] = rest["net_gex"].abs(); rest["abs_dex"] = rest["net_dex"].abs()
mg = rest["abs_gex"].max() or 1; md = rest["abs_dex"].max() or 1
rest["score"] = rest["abs_gex"]/mg + rest["abs_dex"]/md
top10_q = rest.sort_values("score", ascending=False).head(10)
print("  GEX 1..10 (combined |Net GEX| + |Net DEX|, excl CR/PS):")
for i, r in enumerate(top10_q.itertuples(index=False), 1):
    print(f"    GEX_{i:<2} " + fmt_level(r.strike, r.strike*ratio, r.net_gex, r.net_dex,
                                          None, lambda x: x*ratio, NQ_OPEN_3_27))


# ===================== NDX =====================
print()
print("=" * 100)
print(f"NDX 0-1 DTE EOD LEVELS from {DATE}  (usable at 3/27 open)")
print("=" * 100)

eod = pd.read_parquet(NDX_ROOT / DATE.isoformat() / "eod.parquet")
oi  = pd.read_parquet(NDX_ROOT / DATE.isoformat() / "oi.parquet")
eod["expiration"] = pd.to_datetime(eod["expiration"])
oi["expiration"]  = pd.to_datetime(oi["expiration"])

# Parity spot from full chain
df = eod[(eod["bid"]>0) & (eod["ask"]>0)].copy()
df["mid"] = (df["bid"]+df["ask"])/2
df["dte"] = (df["expiration"] - pd.Timestamp(DATE)).dt.days
df_pairs = df[(df["dte"]>0) & (df["dte"]<=36)]
calls = df_pairs[df_pairs["right"].str.upper()=="CALL"][["root","strike","expiration","dte","mid"]].rename(columns={"mid":"C"})
puts  = df_pairs[df_pairs["right"].str.upper()=="PUT"][["root","strike","expiration","dte","mid"]].rename(columns={"mid":"P"})
pairs = calls.merge(puts, on=["root","strike","expiration","dte"])
T = pairs["dte"].values/365.25
S_arr = ((pairs["C"]-pairs["P"])*np.exp(R*T)+pairs["strike"])/np.exp(-Q_NDX*T)
spot_n = float(np.median(S_arr))
basis = nq_17 - spot_n
print(f"  NDX parity spot 3/26 EOD: {spot_n:.2f}    NQ-equiv: {spot_n+basis:.2f}    Basis: +{basis:.1f}")
print(f"  3/27 NQ open: {NQ_OPEN_3_27:.2f}")
print()

# 0-1 DTE NDX chain
eod_01 = eod[(eod["expiration"].dt.date >= DATE) & (eod["expiration"].dt.date <= DATE + dt.timedelta(days=1))].copy()
ch = eod_01.merge(oi[["root","strike","right","expiration","open_interest"]],
                  on=["root","strike","right","expiration"], how="left")
ch["mid"] = (ch["bid"]+ch["ask"])/2
ch = ch[(ch["bid"]>0) & (ch["ask"]>0) & (ch["mid"]>0)]
ch["dte"] = (ch["expiration"] - pd.Timestamp(DATE)).dt.days
T = ch["dte"].values/365.25
K = ch["strike"].values.astype(float); P = ch["mid"].values.astype(float)
is_call_n = (ch["right"].str.upper()=="CALL").values
S2 = np.full_like(K, spot_n)
ch["iv"] = iv_vec(P, S2, K, T, is_call_n, Q_NDX)
ch = ch.dropna(subset=["iv"])
ch["gamma"] = bs_gamma(spot_n, ch["strike"].values, ch["dte"].values/365.25, ch["iv"].values, Q_NDX)
ch["delta"] = np.where(is_call_n[ch.index < len(is_call_n) if False else slice(None)] if False else (ch["right"].str.upper()=="CALL").values,
    bs_call_delta(spot_n, ch["strike"].values, ch["dte"].values/365.25, ch["iv"].values, Q_NDX),
    bs_put_delta(spot_n, ch["strike"].values, ch["dte"].values/365.25, ch["iv"].values, Q_NDX))
ch["gex_abs"] = ch["gamma"]*ch["open_interest"].fillna(0)*100*spot_n**2
ch["signed_gex"] = ch["gex_abs"]
ch.loc[ch["right"].str.upper()=="PUT","signed_gex"] *= -1
ch["signed_dex"] = ch["delta"]*ch["open_interest"].fillna(0)*100*spot_n
is_c2 = (ch["right"].str.upper()=="CALL")
ch["call_only_gex"] = np.where(is_c2, ch["gex_abs"], 0.0)
ch["put_only_gex"]  = np.where(is_c2, 0.0, ch["gex_abs"])
by_n = (ch.groupby("strike").agg(net_gex=("signed_gex","sum"),
                                  net_dex=("signed_dex","sum"),
                                  call_only_gex=("call_only_gex","sum"),
                                  put_only_gex=("put_only_gex","sum"))
        .reset_index().sort_values("strike"))

# ATM IV for 1D EM
sub_atm_n = ch[ch["dte"] > 0]
if not sub_atm_n.empty:
    md = int(sub_atm_n["dte"].min())
    sub_atm_n2 = sub_atm_n[sub_atm_n["dte"] == md]
    iv_n = float(sub_atm_n2.assign(d=(sub_atm_n2["strike"]-spot_n).abs()).nsmallest(4,"d")["iv"].mean())
    em_n = spot_n * iv_n * np.sqrt(1/252)
else:
    iv_n = 0.20; em_n = spot_n * 0.005
print(f"  ATM IV: {iv_n:.4f}    1D EM window: ±{em_n:.2f}    Window: [{spot_n-em_n:.2f}, {spot_n+em_n:.2f}]")
print()
win_n = by_n[(by_n["strike"]>=spot_n-em_n) & (by_n["strike"]<=spot_n+em_n)].copy()
abv = win_n[win_n["strike"]>=spot_n]; blw = win_n[win_n["strike"]<spot_n]
cr_n = float(abv.loc[abv["call_only_gex"].idxmax(),"strike"]) if not abv.empty else None
ps_n = float(blw.loc[blw["put_only_gex"].idxmax(),"strike"]) if not blw.empty else None
print(f"  Call Resistance:  NDX {cr_n:.0f}    NQ {cr_n+basis:.2f}    diff_to_open: {cr_n+basis - NQ_OPEN_3_27:+.0f}")
print(f"  Put Support:      NDX {ps_n:.0f}    NQ {ps_n+basis:.2f}    diff_to_open: {ps_n+basis - NQ_OPEN_3_27:+.0f}")
print()
exc_n = {cr_n, ps_n}
rest_n = win_n[~win_n["strike"].isin(exc_n)].copy()
rest_n["abs_gex"] = rest_n["net_gex"].abs(); rest_n["abs_dex"] = rest_n["net_dex"].abs()
mg = rest_n["abs_gex"].max() or 1; md = rest_n["abs_dex"].max() or 1
rest_n["score"] = rest_n["abs_gex"]/mg + rest_n["abs_dex"]/md
top10_n = rest_n.sort_values("score", ascending=False).head(10)
print("  GEX 1..10 (combined |Net GEX| + |Net DEX|, excl CR/PS):")
for i, r in enumerate(top10_n.itertuples(index=False), 1):
    nq_val = r.strike + basis
    diff = nq_val - NQ_OPEN_3_27
    flag = " *" if abs(diff) < 50 else ""
    print(f"    GEX_{i:<2}  K={r.strike:>8.0f}  NQ={nq_val:>9.2f}  diff_to_open={diff:+7.0f}  "
          f"net_gex={r.net_gex:+.2e}  net_dex={r.net_dex:+.2e}{flag}")
