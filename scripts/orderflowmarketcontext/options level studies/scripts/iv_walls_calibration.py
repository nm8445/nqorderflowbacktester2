"""IV WALLS — calibrate option-implied expected-move bands as levels NQ rarely CLOSES outside,
and test whether touching them is a reversal signal.

Construction
  anchor(D)   = NQ close at 17:00 ET on day D           (prior session settle)
  EM(D)       = ATM_IV(D) * spot(D) * sqrt(1/252)       (MenthorQ 1D expected move, from the
                17:15 ET settle chain -> known before D+1 opens, no lookahead)
  wall_k      = anchor(D) +/- k * EM(D)                 (evaluated on day D+1)

Questions answered
  1. EXCEEDANCE  P(close(D+1) outside +/-k*EM) vs k. This is the "wall" calibration: pick k for a
     target close-outside rate. Compared against a realized-vol (HV) band of matched construction,
     because the variance risk premium (IV > RV) is what makes IV bands hold better than HV bands.
  2. TOUCH->REJECT  given price TOUCHES a wall intraday, how often does it close back inside?
     A wall is only tradeable if touch-then-close-inside beats the base rate.
  3. REVERSAL EDGE  fade the touch (short upper / long lower), exit 17:00 close. Reported against
     the day's unconditional drift, and split by gamma regime + IV/HV, because prior work
     (overnight_band_gamma, band_break_with_qqq_levels) found band breaks CONTINUE in neg-gamma.

Run:  python "scripts/orderflowmarketcontext/options level studies/scripts/iv_walls_calibration.py"
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

ET = "America/New_York"
LEVELS = Path("D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet")
BARS = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
KS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]


def load_sessions() -> pd.DataFrame:
    """Per-day NQ: 17:00 ET anchor, and next session's RTH high/low/close."""
    b = pd.read_parquet(BARS)
    if b.index.tz is None:
        b.index = b.index.tz_localize("UTC")
    b = b.tz_convert(ET).sort_index()
    # markettick 1-min bars are RIGHT-labeled (index = bar close); shift back so a timestamp
    # names the minute it OPENS, which is what the session-window filters below assume.
    b.index = b.index - pd.Timedelta(minutes=1)
    b["d"] = b.index.date
    hhmm = b.index.hour * 100 + b.index.minute

    # anchor = last print at/before 17:00 ET
    anch = b[hhmm <= 1700].groupby("d")["close"].last().rename("anchor")
    # RTH session 09:30 -> 17:00 ET
    rth = b[(hhmm >= 930) & (hhmm <= 1700)]
    g = rth.groupby("d")
    s = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                      "low": g["low"].min(), "close": g["close"].last()})
    s["anchor_prev"] = anch.reindex(s.index).shift(1)
    s.index = pd.to_datetime(s.index)
    return s.dropna()


def main():
    lv = pd.read_parquet(LEVELS)
    lv["date"] = pd.to_datetime(lv["date"])
    lv = lv.sort_values("date").reset_index(drop=True)

    s = load_sessions()
    # EM/IV/gamma from day D are applied to day D+1 -> shift the levels frame forward one session.
    lv["apply_date"] = lv["date"].shift(-1)
    L = lv.dropna(subset=["apply_date"]).set_index("apply_date")

    df = s.join(L[["ndx_em", "qqq_em", "ndx_iv", "qqq_iv", "qqq_ratio", "qqq_gamma_sign"]], how="inner")
    df = df.dropna(subset=["ndx_em", "anchor_prev"])
    df["em"] = df["ndx_em"]                      # NDX points ~ NQ points (native, no ratio scaling)

    # realized-vol comparator: same formula, IV replaced by trailing 20d close-to-close vol
    r = np.log(df["close"] / df["close"].shift(1))
    df["hv"] = r.rolling(20).std() * np.sqrt(252)
    df["em_hv"] = df["anchor_prev"] * df["hv"] * np.sqrt(1 / 252)
    df["ivhv"] = df["ndx_iv"] / df["hv"]
    df = df.dropna(subset=["em_hv"])

    print(f"sample: {len(df)} sessions  {df.index.min().date()} .. {df.index.max().date()}")
    print(f"mean ATM IV {df['ndx_iv'].mean():.3f}   mean HV20 {df['hv'].mean():.3f}   "
          f"IV/HV {df['ivhv'].median():.2f} (median)   mean 1d EM {df['em'].mean():.0f} NQ pts")

    # ---------- 1. exceedance calibration ----------
    print("\n=== 1. P(RTH close outside the wall) — IV-EM band vs HV band ===")
    print(f"{'k':>6}{'wall +/- pts':>14}{'IV close-out':>14}{'HV close-out':>14}{'IV touch':>11}")
    print("-" * 59)
    for k in KS:
        up, dn = df["anchor_prev"] + k * df["em"], df["anchor_prev"] - k * df["em"]
        uh, dh = df["anchor_prev"] + k * df["em_hv"], df["anchor_prev"] - k * df["em_hv"]
        out_iv = ((df["close"] > up) | (df["close"] < dn)).mean()
        out_hv = ((df["close"] > uh) | (df["close"] < dh)).mean()
        touch = ((df["high"] >= up) | (df["low"] <= dn)).mean()
        print(f"{k:>6.2f}{(k*df['em']).mean():>14.0f}{out_iv*100:>13.1f}%"
              f"{out_hv*100:>13.1f}%{touch*100:>10.1f}%")

    # ---------- 2. touch -> reject ----------
    print("\n=== 2. Given an intraday TOUCH, does it close back inside? ===")
    print(f"{'k':>6}{'n touch':>9}{'closed back in':>16}{'n up-touch':>12}{'up rejected':>13}"
          f"{'n dn-touch':>12}{'dn rejected':>13}")
    print("-" * 81)
    for k in KS:
        up, dn = df["anchor_prev"] + k * df["em"], df["anchor_prev"] - k * df["em"]
        tu, td = df["high"] >= up, df["low"] <= dn
        t = tu | td
        back = t & (df["close"] <= up) & (df["close"] >= dn)
        print(f"{k:>6.2f}{t.sum():>9}{100*back.sum()/max(t.sum(),1):>15.1f}%"
              f"{tu.sum():>12}{100*(tu&(df['close']<=up)).sum()/max(tu.sum(),1):>12.1f}%"
              f"{td.sum():>12}{100*(td&(df['close']>=dn)).sum()/max(td.sum(),1):>12.1f}%")

    # ---------- 3. fade edge ----------
    print("\n=== 3. FADE the touch (short upper / long lower), exit 17:00 close ===")
    print("    baseline drift, anchor->close: %+.1f pts/day\n" % (df["close"] - df["anchor_prev"]).mean())
    print(f"{'k':>6}{'side':>6}{'n':>6}{'P(win)':>9}{'mean pts':>10}{'t-stat':>9}")
    print("-" * 46)
    for k in [1.0, 1.25, 1.5]:
        up, dn = df["anchor_prev"] + k * df["em"], df["anchor_prev"] - k * df["em"]
        for side, m, pnl in [("short", df["high"] >= up, up - df["close"]),
                             ("long", df["low"] <= dn, df["close"] - dn)]:
            x = pnl[m].dropna()
            t = x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 1 else np.nan
            print(f"{k:>6.2f}{side:>6}{len(x):>6}{(x>0).mean()*100:>8.1f}%{x.mean():>10.1f}{t:>9.2f}")

    # ---------- 4. regime conditioning ----------
    print("\n=== 4. Fade at k=1.0, split by gamma regime and IV/HV ===")
    k = 1.0
    up, dn = df["anchor_prev"] + k * df["em"], df["anchor_prev"] - k * df["em"]
    df["_up"], df["_dn"] = up, dn
    hi_ivhv = df["ivhv"] > df["ivhv"].median()
    cuts = [("pos-gamma", df["qqq_gamma_sign"] > 0), ("neg-gamma", df["qqq_gamma_sign"] < 0),
            ("IV/HV high", hi_ivhv), ("IV/HV low", ~hi_ivhv),
            ("pos-gamma & IV/HV high", (df["qqq_gamma_sign"] > 0) & hi_ivhv)]
    print(f"{'cohort':<24}{'side':>6}{'n':>6}{'P(win)':>9}{'mean pts':>10}{'t-stat':>9}")
    print("-" * 64)
    for name, c in cuts:
        for side, m, pnl in [("short", (df["high"] >= up) & c, up - df["close"]),
                             ("long", (df["low"] <= dn) & c, df["close"] - dn)]:
            x = pnl[m].dropna()
            if len(x) < 25:
                continue
            t = x.mean() / (x.std() / np.sqrt(len(x)))
            print(f"{name:<24}{side:>6}{len(x):>6}{(x>0).mean()*100:>8.1f}%{x.mean():>10.1f}{t:>9.2f}")


if __name__ == "__main__":
    main()
