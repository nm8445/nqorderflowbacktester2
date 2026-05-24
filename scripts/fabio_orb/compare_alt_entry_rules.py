"""Compare Fabio ORB entry rules:

  A) BASELINE (current locked rule):
     - 4 consecutive 5-min closes above ORB_High
     - Delta check (>=300) on the ENTRY bar (the 4th confirming bar)
     - Skip 09:30 entry bucket

  B) PROPOSED v3:
     - First bar to close above ORB_High gets delta check (>=300)
     - If passes: next 3 bars all close above ORB_High
     - Enter at the 3rd confirming bar's close (4 bars total above ORB)
     - Skip 09:30 entry bucket

  C) PROPOSED v4:
     - Same as v3 but next 4 bars must close above ORB_High
     - Enter at the 4th confirming bar's close (5 bars total above ORB)
     - Skip 09:30 entry bucket

Shared:
  ORB = 08:30-09:00 ET (bar closes)
  Trade window = 09:00-14:00 ET
  SL = ORB_Low (static)
  TP = entry + 4.0 * (entry - ORB_Low)
  EOD = first bar close >= 14:00 ET
  ORB_Low < close sanity check
  Costs: 1 tick slip/side, $5 RT commission per contract
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

VOL_PARQUET = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
ET = "America/New_York"

ORB_START_HHMM = 830
ORB_END_HHMM = 900
TRADE_END_HHMM = 1400
SKIP_BUCKET_HHMM = 930
DELTA_THRESHOLD = 300
TP_RR = 4.0

TICK = 0.25
TICK_VAL = 5.0
DOLLARS_PER_PT = TICK_VAL / TICK   # = 20
SLIP_TICKS_PER_SIDE = 1
COMM_RT = 5.0


def load_bars() -> dict[pd.Timestamp.date, pd.DataFrame]:
    df = pd.read_parquet(VOL_PARQUET)
    agg = df.groupby("bar_open_time", as_index=False).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"),
    )
    agg["bar_open_time"] = pd.to_datetime(agg["bar_open_time"]).dt.tz_convert(ET)
    # close_time = open + 5 min, hhmm = close hh*100 + mm
    agg["close_time"] = agg["bar_open_time"] + pd.Timedelta(minutes=5)
    agg["hhmm"] = agg["close_time"].dt.hour * 100 + agg["close_time"].dt.minute
    agg["delta"] = agg["buy_vol"] - agg["sell_vol"]
    agg["date"] = agg["close_time"].dt.date
    return {d: g.reset_index(drop=True) for d, g in agg.groupby("date")}


def simulate_exit(post_orb: pd.DataFrame, entry_idx: int, entry_price: float,
                  orb_low: float) -> tuple[float, str]:
    """Walk forward from entry bar. Return (exit_price, reason)."""
    sl_price = orb_low
    risk = entry_price - sl_price
    tp_price = entry_price + TP_RR * risk
    for i in range(entry_idx + 1, len(post_orb)):
        bar = post_orb.iloc[i]
        # SL first (intrabar)
        if bar["low"] <= sl_price:
            return sl_price, "SL"
        if bar["high"] >= tp_price:
            return tp_price, "TP"
        if bar["hhmm"] >= TRADE_END_HHMM:
            return float(bar["close"]), "EOD"
    # Ran out of bars — assume close at last bar
    last = post_orb.iloc[-1]
    return float(last["close"]), "EOD"


def find_entry_baseline(post_orb: pd.DataFrame, orb_high: float, orb_low: float) -> int | None:
    """Current rule: 4 consec closes above ORB_High, delta>=300 on the 4th, skip 09:30, ORB_Low<close.
    Returns the index in post_orb of the entry bar, or None."""
    closes = post_orb["close"].values
    deltas = post_orb["delta"].values
    hhmms = post_orb["hhmm"].values
    for i in range(3, len(post_orb)):
        if hhmms[i] == SKIP_BUCKET_HHMM: continue
        if not all(closes[i - k] > orb_high for k in range(4)): continue
        if deltas[i] < DELTA_THRESHOLD: continue
        if orb_low >= closes[i]: continue
        return i
    return None


def find_entry_proposed(post_orb: pd.DataFrame, orb_high: float, orb_low: float,
                         n_confirm_after: int) -> int | None:
    """Proposed: first close above ORB_High has delta>=300, then next n_confirm_after
    bars all close above ORB_High. Enter on the n_confirm_after-th post-break bar.
    Skip 09:30 ENTRY bar."""
    closes = post_orb["close"].values
    deltas = post_orb["delta"].values
    hhmms = post_orb["hhmm"].values
    n = len(post_orb)
    for i in range(n):
        # First bar that closes above ORB_High
        if closes[i] <= orb_high: continue
        # Delta check on this break bar
        if deltas[i] < DELTA_THRESHOLD:
            # Failed delta — this opportunity is dead. But the rule should allow
            # a LATER bar to be "the first break" if price came back below and
            # broke again. We model by waiting for price to dip below ORB_High,
            # then a new close above counts as a fresh break.
            # For simplicity here: scan forward for the next close >= ORB_High
            # AFTER a close < ORB_High.
            continue
        # Need i + n_confirm_after to exist
        entry_idx = i + n_confirm_after
        if entry_idx >= n: return None
        # Check all bars from i+1 to entry_idx close above ORB_High
        if not all(closes[k] > orb_high for k in range(i + 1, entry_idx + 1)): continue
        # Skip 09:30 entry
        if hhmms[entry_idx] == SKIP_BUCKET_HHMM: continue
        if orb_low >= closes[entry_idx]: continue
        return entry_idx
    return None


def find_entry_proposed_v2(post_orb: pd.DataFrame, orb_high: float, orb_low: float,
                            n_confirm_after: int) -> int | None:
    """Alternative interpretation: the break bar's index restarts each time price
    crosses back below ORB_High. So the 'first break' is the first close > ORB_High
    where the prior bar closed <= ORB_High (or it's the first post-ORB bar).
    Delta on that break bar, then n_confirm_after bars after must close above ORB_High."""
    closes = post_orb["close"].values
    deltas = post_orb["delta"].values
    hhmms = post_orb["hhmm"].values
    n = len(post_orb)
    for i in range(n):
        if closes[i] <= orb_high: continue
        # Is this a "fresh break"? Yes if i==0 or prior close <= ORB_High
        is_break = (i == 0) or (closes[i - 1] <= orb_high)
        if not is_break: continue
        # Delta check on break bar
        if deltas[i] < DELTA_THRESHOLD: continue
        entry_idx = i + n_confirm_after
        if entry_idx >= n: return None
        # All bars after the break must close above ORB_High
        if not all(closes[k] > orb_high for k in range(i + 1, entry_idx + 1)): continue
        if hhmms[entry_idx] == SKIP_BUCKET_HHMM: continue
        if orb_low >= closes[entry_idx]: continue
        return entry_idx
    return None


def run_rule(bars_by_day: dict, rule_name: str, finder) -> dict:
    trades = []
    for d, bars in bars_by_day.items():
        # Build ORB from bars whose close hhmm in (08:30, 09:00]
        orb_bars = bars[(bars["hhmm"] > ORB_START_HHMM) & (bars["hhmm"] <= ORB_END_HHMM)]
        if len(orb_bars) == 0: continue
        orb_high = float(orb_bars["high"].max())
        orb_low  = float(orb_bars["low"].min())
        post_orb = bars[(bars["hhmm"] > ORB_END_HHMM) & (bars["hhmm"] <= TRADE_END_HHMM)].reset_index(drop=True)
        if len(post_orb) == 0: continue
        entry_idx = finder(post_orb, orb_high, orb_low)
        if entry_idx is None: continue
        entry_bar = post_orb.iloc[entry_idx]
        entry_price = float(entry_bar["close"])
        exit_price, reason = simulate_exit(post_orb, entry_idx, entry_price, orb_low)
        # Apply costs
        slip = SLIP_TICKS_PER_SIDE * TICK * 2  # round-trip slippage in points
        gross_pts = exit_price - entry_price
        net_pts = gross_pts - slip
        pnl = net_pts * DOLLARS_PER_PT - COMM_RT
        trades.append({
            "date": d, "entry_hhmm": int(entry_bar["hhmm"]),
            "entry_price": entry_price, "exit_price": exit_price, "reason": reason,
            "pnl_$": pnl, "risk_pts": entry_price - orb_low,
        })
    if not trades:
        return {"rule": rule_name, "n_trades": 0}
    df = pd.DataFrame(trades)
    wins = df[df["pnl_$"] > 0]
    losses = df[df["pnl_$"] < 0]
    win_sum = wins["pnl_$"].sum()
    loss_sum = abs(losses["pnl_$"].sum())
    pf = win_sum / loss_sum if loss_sum > 0 else float("inf")
    df = df.sort_values("date").reset_index(drop=True)
    df["cum"] = df["pnl_$"].cumsum()
    df["peak"] = df["cum"].cummax()
    df["dd"] = df["cum"] - df["peak"]
    return {
        "rule": rule_name,
        "n_trades": len(df),
        "wr%": round(len(wins) / len(df) * 100, 1),
        "net_$": round(df["pnl_$"].sum(), 0),
        "avg_$": round(df["pnl_$"].mean(), 1),
        "PF": round(pf, 2),
        "MaxDD_$": round(df["dd"].min(), 0),
        "TP%": round((df["reason"] == "TP").mean() * 100, 1),
        "SL%": round((df["reason"] == "SL").mean() * 100, 1),
        "EOD%": round((df["reason"] == "EOD").mean() * 100, 1),
        "avg_entry_hhmm": round(df["entry_hhmm"].mean(), 0),
    }


def main():
    print("Loading 5-min volumetric bars...")
    bars_by_day = load_bars()
    print(f"  {len(bars_by_day)} session days  ({min(bars_by_day)} -> {max(bars_by_day)})")

    rows = []
    rows.append(run_rule(bars_by_day, "BASELINE (4 consec, delta on entry)", find_entry_baseline))

    # Both interpretations of "first break with delta":
    # _proposed = any close > ORB_High with delta passes; future bars stay above
    # _proposed_v2 = "fresh break" (came from <= ORB_High); future bars stay above
    rows.append(run_rule(bars_by_day, "PROPOSED v3 (delta-break + 3 after, loose)",
                          lambda po, oh, ol: find_entry_proposed(po, oh, ol, 3)))
    rows.append(run_rule(bars_by_day, "PROPOSED v4 (delta-break + 4 after, loose)",
                          lambda po, oh, ol: find_entry_proposed(po, oh, ol, 4)))
    rows.append(run_rule(bars_by_day, "PROPOSED v3 (delta on fresh break + 3 after)",
                          lambda po, oh, ol: find_entry_proposed_v2(po, oh, ol, 3)))
    rows.append(run_rule(bars_by_day, "PROPOSED v4 (delta on fresh break + 4 after)",
                          lambda po, oh, ol: find_entry_proposed_v2(po, oh, ol, 4)))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print()
    print(df.to_string(index=False))
    out = Path("C:/trading/nqorderflowbacktester/scripts/fabio_orb/alt_entry_rules_compare.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
