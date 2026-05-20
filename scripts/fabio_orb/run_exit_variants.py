"""
Test exit-rule variants on the Mode A config
(N=4 confirmation + skip 9:30 + delta>=300, TP=4R nominal).

Variants:
  V0  baseline (current Mode A): SL at ORB_low, EOD at 14:00
  V1  trail-to-BE @ MFE >= 0.5R  (move SL to entry once peak hits 0.5R)
  V2  trail-to-BE @ MFE >= 1.0R
  V3  partial exit @ 1R: exit half at 1R limit, ride other half to EOD/SL
  V4  exit time sweep:  exit at 11:00, 12:00, 13:00, 14:00 (baseline SL)

Reports IS / OOS / FULL stats for each variant.
"""
from __future__ import annotations
import warnings; warnings.simplefilter("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from datetime import date

VOL_PARQUET = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
OUT_DIR = Path("D:/trading_pythonbacktest_data/fabio orb")
ET = "America/New_York"
ORB_START_HHMM, ORB_END_HHMM, TRADE_END_HHMM = 830, 900, 1400
SKIP_BUCKET = 930

TICK_SIZE, TICK_VALUE = 0.25, 5.0
DPP = TICK_VALUE / TICK_SIZE
SLIP_PTS = 1 * TICK_SIZE     # one tick per side
COMM = 5.0                    # $ round trip
CUTOFF = date(2024, 3, 12)    # IS/OOS split

# ============================== bar loading (same as before) ==============================

def load_days():
    df = pd.read_parquet(VOL_PARQUET)
    agg = df.groupby("bar_open_time", as_index=False).agg(
        open=("open","first"), high=("high","max"), low=("low","min"),
        close=("close","last"), buy_vol=("buy_vol","sum"), sell_vol=("sell_vol","sum"),
    ).dropna(subset=["open","high","low","close"])
    agg["close_et"] = agg["bar_open_time"] + pd.Timedelta(minutes=5)
    agg["hhmm"] = agg["close_et"].dt.hour*100 + agg["close_et"].dt.minute
    agg = agg[(agg["hhmm"] > ORB_START_HHMM) & (agg["hhmm"] <= TRADE_END_HHMM)].copy()
    agg["delta"] = agg["buy_vol"] - agg["sell_vol"]
    agg["session_date"] = agg["close_et"].dt.normalize()
    days = {}
    for sd, sub in agg.groupby("session_date"):
        sub = sub.sort_values("close_et").reset_index(drop=True)
        in_orb = sub[(sub["hhmm"] > ORB_START_HHMM) & (sub["hhmm"] <= ORB_END_HHMM)]
        if in_orb.empty: continue
        post = sub[(sub["hhmm"] > ORB_END_HHMM) & (sub["hhmm"] <= TRADE_END_HHMM)]
        if post.empty: continue
        days[pd.Timestamp(sd).tz_localize(None)] = {
            "orb_high": float(in_orb["high"].max()),
            "orb_low":  float(in_orb["low"].min()),
            "hhmm":  post["hhmm"].to_numpy(),
            "high":  post["high"].to_numpy(dtype=np.float64),
            "low":   post["low"].to_numpy(dtype=np.float64),
            "close": post["close"].to_numpy(dtype=np.float64),
            "delta": post["delta"].to_numpy(dtype=np.float64),
            "close_et": post["close_et"].to_numpy(),
        }
    return days


# ============================== shared entry rules (Mode A) ==============================

def find_entry(day):
    """Return (entry_idx, entry_price, sl, tp, entry_time) or None."""
    oh = day["orb_high"]; ol = day["orb_low"]
    hhmm = day["hhmm"]; close = day["close"]; delta = day["delta"]; etime = day["close_et"]
    N = 4; DTHR = 300.0; n = len(hhmm)
    for i in range(N - 1, n):
        if hhmm[i] > TRADE_END_HHMM: break
        if hhmm[i] == SKIP_BUCKET: continue
        if not all(close[i-k] > oh for k in range(N)): continue
        if delta[i] < DTHR: continue
        ep = close[i]
        if ol >= ep: continue
        return i, ep, ol, ep + 4.0*(ep-ol), etime[i]
    return None


# ============================== exit logic (per variant) ==============================

def run_v_baseline(day, end_hhmm=TRADE_END_HHMM, trail_be_at_R=None):
    """Baseline + optional trail-to-BE at MFE >= R-multiple threshold."""
    ent = find_entry(day)
    if ent is None: return None
    i, ep, sl, tp, etime0 = ent
    hhmm, high, low, close, etime = day["hhmm"], day["high"], day["low"], day["close"], day["close_et"]
    risk = ep - sl
    cur_sl = sl
    max_high = ep
    for j in range(i+1, len(hhmm)):
        if high[j] > max_high: max_high = high[j]
        # update trailing SL if condition met
        if trail_be_at_R is not None and cur_sl < ep:
            if (max_high - ep) >= trail_be_at_R * risk:
                cur_sl = ep   # move to breakeven
        # check exits
        if hhmm[j] >= end_hhmm:
            return _close(ep, close[j], etime0, etime[j], "EOD", risk)
        hit_sl = low[j] <= cur_sl; hit_tp = high[j] >= tp
        if hit_sl and hit_tp:
            return _close(ep, cur_sl, etime0, etime[j], "SL_TP", risk)
        if hit_sl:
            reason = "BE" if cur_sl == ep else "SL"
            return _close(ep, cur_sl, etime0, etime[j], reason, risk)
        if hit_tp:
            return _close(ep, tp, etime0, etime[j], "TP", risk)
    return _close(ep, close[-1], etime0, etime[-1], "EOD_LAST", risk)


def run_v_partial(day, partial_R=1.0):
    """Exit half at +partial_R limit; ride other half to EOD/SL."""
    ent = find_entry(day)
    if ent is None: return None
    i, ep, sl, tp, etime0 = ent
    hhmm, high, low, close, etime = day["hhmm"], day["high"], day["low"], day["close"], day["close_et"]
    risk = ep - sl
    partial_target = ep + partial_R * risk
    half_taken = False; half_exit_price = None; half_exit_time = None
    final_exit_price = None; final_exit_time = None; final_reason = None
    for j in range(i+1, len(hhmm)):
        # half-exit at limit
        if not half_taken and high[j] >= partial_target:
            half_taken = True
            half_exit_price = partial_target
            half_exit_time = etime[j]
        if hhmm[j] >= TRADE_END_HHMM:
            final_exit_price = close[j]; final_exit_time = etime[j]; final_reason = "EOD"; break
        if low[j] <= sl:
            final_exit_price = sl; final_exit_time = etime[j]; final_reason = "SL"; break
        if high[j] >= tp:
            final_exit_price = tp; final_exit_time = etime[j]; final_reason = "TP"; break
    else:
        final_exit_price = close[-1]; final_exit_time = etime[-1]; final_reason = "EOD_LAST"
    # Two halves -- model as 0.5 contracts each, total 1 contract risk
    def half_pnl(xp):
        raw = xp - ep
        gross = 0.5 * raw * DPP
        cost = 0.5 * (2 * SLIP_PTS * DPP + COMM)
        return gross - cost
    pnl1 = half_pnl(half_exit_price) if half_taken else 0.0
    pnl2 = half_pnl(final_exit_price)
    if not half_taken:
        # full risk is on the 1.0-contract; but our half logic only pays half costs.
        # To keep apples-apples, pay the missing half costs (if we never partial-exited, we're
        # effectively 1-contract risk that fully exited at final_exit_price): add the missing half.
        pnl2 += 0.5 * (final_exit_price - ep) * DPP - 0.5 * (2 * SLIP_PTS * DPP + COMM)
    net = pnl1 + pnl2
    raw_pts = (0.5*(half_exit_price - ep) if half_taken else 0) + 0.5*(final_exit_price - ep) \
              if half_taken else (final_exit_price - ep)
    reason = ("PARTIAL+" + final_reason) if half_taken else final_reason
    return {
        "entry_time": pd.Timestamp(etime0), "exit_time": pd.Timestamp(final_exit_time),
        "entry": ep, "exit": final_exit_price, "sl": sl, "tp": tp,
        "risk_pts": risk, "raw_pts": raw_pts,
        "gross_dollars": net + (0.5*2*SLIP_PTS*DPP + 0.5*COMM)*(2 if half_taken else 1),  # approx
        "net_dollars": net, "reason": reason,
        "half_taken": half_taken,
    }


def _close(ep, xp, etime0, etime, reason, risk):
    raw = xp - ep
    gross = raw * DPP
    net = gross - 2 * SLIP_PTS * DPP - COMM
    return {
        "entry_time": pd.Timestamp(etime0), "exit_time": pd.Timestamp(etime),
        "entry": ep, "exit": xp, "sl": ep - risk, "tp": ep + 4*risk,
        "risk_pts": risk, "raw_pts": raw,
        "gross_dollars": gross, "net_dollars": net, "reason": reason,
    }


# ============================== reporting ==============================

def split_is_oos(trades):
    df = pd.DataFrame(trades)
    if df.empty: return df, df
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["date"] = df["entry_time"].dt.tz_convert(ET).dt.date
    return df[df["date"] < CUTOFF], df[df["date"] >= CUTOFF]


def line(label, s):
    n = len(s)
    if n == 0:
        print(f"  {label:<10} n=0"); return
    wins = int((s["net_dollars"]>0).sum())
    wd = s.loc[s["net_dollars"]>0, "net_dollars"]
    ld = -s.loc[s["net_dollars"]<0, "net_dollars"]
    pf = wd.sum()/ld.sum() if ld.sum()>0 else float("inf")
    eq = s.sort_values("entry_time")["net_dollars"].cumsum()
    dd = (eq - eq.cummax()).min()
    print(f"  {label:<10} n={n:>4}  win={100*wins/n:>5.1f}%  net=${s['net_dollars'].sum():>9,.0f}  PF={pf:.3f}  avg=${s['net_dollars'].mean():>5,.0f}  MaxDD=${dd:>9,.0f}")


def report_variant(name, trades):
    IS, OOS = split_is_oos(trades)
    FULL = pd.concat([IS, OOS])
    print(f"\n=== {name} ===")
    line("IS",   IS)
    line("OOS",  OOS)
    line("FULL", FULL)
    # exit reason counts
    if not FULL.empty:
        rc = FULL["reason"].value_counts()
        rc_str = ", ".join(f"{r}={c}" for r,c in rc.head(6).items())
        print(f"  Exits: {rc_str}")


# ============================== main ==============================

def main():
    print("Loading bars...", flush=True)
    days = load_days(); keys = sorted(days.keys())
    print(f"  {len(keys)} days\n")

    # V0 baseline (recompute for reference)
    trades = [r for d in keys if (r := run_v_baseline(days[d])) is not None]
    report_variant("V0  BASELINE  (Mode A: SL=ORB_low, EOD@14:00, TP=4R)", trades)

    # V1 trail-to-BE at 0.5R
    trades = [r for d in keys if (r := run_v_baseline(days[d], trail_be_at_R=0.5)) is not None]
    report_variant("V1  Trail-to-BE @ MFE>=0.5R", trades)

    # V2 trail-to-BE at 1R
    trades = [r for d in keys if (r := run_v_baseline(days[d], trail_be_at_R=1.0)) is not None]
    report_variant("V2  Trail-to-BE @ MFE>=1.0R", trades)

    # V3 partial exit at 1R
    trades = [r for d in keys if (r := run_v_partial(days[d], partial_R=1.0)) is not None]
    report_variant("V3  Partial exit (50% @ 1R limit, rest to EOD/SL)", trades)

    # V4 exit-time sweep
    print("\n=== V4  Exit-time sweep (no trail, no partial — just earlier EOD cutoff) ===")
    for end in [1100, 1200, 1300, 1400]:
        trades = [r for d in keys if (r := run_v_baseline(days[d], end_hhmm=end)) is not None]
        IS, OOS = split_is_oos(trades); FULL = pd.concat([IS, OOS])
        n = len(FULL)
        wins = int((FULL["net_dollars"]>0).sum()) if n else 0
        net = FULL["net_dollars"].sum() if n else 0
        eq = FULL.sort_values("entry_time")["net_dollars"].cumsum() if n else pd.Series()
        dd = (eq - eq.cummax()).min() if n else 0
        wd = FULL.loc[FULL["net_dollars"]>0, "net_dollars"].sum() if n else 0
        ld = -FULL.loc[FULL["net_dollars"]<0, "net_dollars"].sum() if n else 0
        pf = wd/ld if ld>0 else float("inf")
        end_label = f"{end//100:02d}:{end%100:02d}"
        print(f"  exit@{end_label}  n={n:>4}  win={100*wins/n:>5.1f}%  net=${net:>9,.0f}  PF={pf:.3f}  MaxDD=${dd:>9,.0f}")


if __name__ == "__main__":
    main()
