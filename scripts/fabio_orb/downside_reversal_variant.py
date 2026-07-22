"""Fabio downside-reversal LONG variant test (long-only, thesis-OPPOSITE of the breakout).

Adds a 2nd LONG entry to Fabio: when price breaks BELOW the ORB low (N=4 consecutive 5-min closes
< ORB_Low) AND the entry bar has BULLISH delta >= 300 (buyers absorbing the breakdown) -> go LONG.
  SL = entry - ORB_range (risk = ORB_High - ORB_Low);  TP = entry + 4 * ORB_range.
Same window (09:00-14:00 closes), skip the 09:30 bucket, 14:00 EOD. Fabio stays LONG-only.

Three books over the locked period:
  UP   = original upside breakout long only (reproduces trades_final_modeA)
  DOWN = downside reversal long only (the new entry, standalone)
  BOTH = one trade/day, whichever setup (up-breakout / down-reversal) fires FIRST

Run: python scripts/fabio_orb/downside_reversal_variant.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np, pandas as pd
from run_final_config import (load_days, summarize, DPP, SLIP, COMM, TICK_SIZE,
                              SKIP_BUCKET_HHMM, TRADE_END_HHMM)

N = 4
DTHR = 300.0


def _simulate_exit(day, entry_idx, entry_price, sl, tp):
    hhmm = day["hhmm"]; high = day["high"]; low = day["low"]; close = day["close"]; etime = day["close_et"]
    n = len(hhmm); min_low = entry_price; max_high = entry_price
    for j in range(entry_idx + 1, n):
        if low[j] < min_low: min_low = low[j]
        if high[j] > max_high: max_high = high[j]
        if hhmm[j] >= TRADE_END_HHMM:
            xp, reason, xt = close[j], "EOD", etime[j]; break
        hit_sl = low[j] <= sl; hit_tp = high[j] >= tp
        if hit_sl and hit_tp: xp, reason, xt = sl, "SL_TP", etime[j]; break    # tie -> SL (conservative)
        if hit_sl: xp, reason, xt = sl, "SL", etime[j]; break
        if hit_tp: xp, reason, xt = tp, "TP", etime[j]; break
    else:
        xp, reason, xt = close[-1], "EOD_LAST", etime[-1]
    raw = xp - entry_price; gross = raw * DPP; net = gross - SLIP * TICK_SIZE * 2 * DPP - COMM
    return {"entry_time": pd.Timestamp(etime[entry_idx]), "exit_time": pd.Timestamp(xt),
            "entry": entry_price, "exit": xp, "sl": sl, "tp": tp, "risk_pts": entry_price - sl,
            "raw_pts": raw, "gross_dollars": gross, "net_dollars": net,
            "mae_pts": entry_price - min_low, "mfe_pts": max_high - entry_price, "reason": reason}


def find_up(day):
    """Original: 4 closes ABOVE ORB_high + delta>=300. SL=ORB_low, TP=entry+4*(entry-ORB_low)."""
    oh = day["orb_high"]; ol = day["orb_low"]; hhmm = day["hhmm"]; close = day["close"]; delta = day["delta"]
    for i in range(N - 1, len(hhmm)):
        if hhmm[i] > TRADE_END_HHMM: break
        if hhmm[i] == SKIP_BUCKET_HHMM: continue
        if not all(close[i - k] > oh for k in range(N)): continue
        if delta[i] < DTHR: continue
        ep = close[i]
        if ol >= ep: continue
        return (i, ep, ol, ep + 4.0 * (ep - ol))
    return None


def find_down(day):
    """New: 4 closes BELOW ORB_low + BULLISH delta>=300 (absorption). SL=entry-range, TP=entry+4*range."""
    oh = day["orb_high"]; ol = day["orb_low"]; hhmm = day["hhmm"]; close = day["close"]; delta = day["delta"]
    rng = oh - ol
    for i in range(N - 1, len(hhmm)):
        if hhmm[i] > TRADE_END_HHMM: break
        if hhmm[i] == SKIP_BUCKET_HHMM: continue
        if not all(close[i - k] < ol for k in range(N)): continue
        if delta[i] < DTHR: continue            # bullish delta on the breakdown bar = absorption
        ep = close[i]
        return (i, ep, ep - rng, ep + 4.0 * rng)
    return None


def run(days, which):
    trades = []
    for d in sorted(days):
        day = days[d]
        up = find_up(day); dn = find_down(day)
        if which == "up": pick, tag = up, "up"
        elif which == "down": pick, tag = dn, "down"
        else:  # both: take whichever fires first (lower bar index)
            if up and dn: pick, tag = (up, "up") if up[0] <= dn[0] else (dn, "down")
            elif up: pick, tag = up, "up"
            else: pick, tag = dn, "down"
        if pick is None: continue
        i, ep, sl, tp = pick
        t = _simulate_exit(day, i, ep, sl, tp); t["which"] = tag
        trades.append(t)
    return trades


def main():
    print("Loading bars...", flush=True)
    days = load_days()
    keys = sorted(days)
    print(f"  {len(keys)} days {keys[0].date()}..{keys[-1].date()}")

    for which in ("up", "down", "both"):
        tr = run(days, which)
        df = summarize(f"Fabio {which.upper()}", tr)
        if which == "both" and df is not None:
            vc = df["which"].value_counts().to_dict()
            print(f"  BOTH composition: {vc}")
        if which == "down" and tr:
            pd.DataFrame(tr).to_csv(Path(__file__).resolve().parent / "results" /
                                    "fabio_downside_reversal_trades.csv", index=False)


if __name__ == "__main__":
    main()
