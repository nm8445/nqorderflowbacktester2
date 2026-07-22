"""Per-trade audit log for the FB giveback stop (k=1.5/gb0.3): entry/exit ts+price, TP, initial SL
(ORB_Low), and the SL (yellow) trailing EVERY 5-min bar — ratcheting up or giving back down.

Also serves as a lookahead check: every value on a bar's line is computable AT that bar's close
(OHLC of that bar, ATR through that bar, prev bar for the bearish flag). The SL update and the exit
decision both happen at the bar CLOSE (process_orders_on_close) — no future bar is referenced.

Writes the FULL trace to results/giveback_trade_trace.txt and prints 3 representative trades.

Run:  python scripts/fabio_orb/giveback_trade_log.py
"""
from __future__ import annotations
import warnings; warnings.simplefilter("ignore")
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_giveback_variant import load_days, find_entry   # noqa: E402

OUT = Path(__file__).parent / "results" / "giveback_trade_trace.txt"
TRADE_END_HHMM = 1400
DPP, SLIP_PTS, COMM = 20.0, 0.25, 5.0
COST = 2 * SLIP_PTS * DPP + COMM
GB = dict(k=1.5, mode="drift_floor", drift=0.0, gb=0.3, scale_body=True, max_gb=0.5, min_gap=0.3)


def trace_trade(day, k, mode, drift, gb, scale_body, max_gb, min_gap):
    ent = find_entry(day)
    if ent is None: return None
    i, ep, ol, tp, t0 = ent
    op, hi, lo, cl = day["open"], day["high"], day["low"], day["close"]
    hhmm, atr, etime = day["hhmm"], day["atr"], day["close_et"]
    yellow = ol; prev_yellow = np.nan; gb_on = gb > 0
    rows = []
    exit_px = exit_ts = reason = None
    for j in range(i + 1, len(hhmm)):
        a = atr[j] if atr[j] > 0 else max(atr[i], 1e-6)
        raw_yellow = cl[j] - k * a
        prev_bearish = cl[j - 1] < op[j - 1]
        gaveback = False
        if gb_on and prev_bearish and not np.isnan(prev_yellow):
            gap = max(0.0, prev_yellow - raw_yellow); frac = gb
            if scale_body:
                frac *= min(1.0, (op[j - 1] - cl[j - 1]) / a)
            cand = prev_yellow - min(gap * frac, max_gb * a); gaveback = True
        elif mode == "pure_ratchet":
            cand = max(prev_yellow, raw_yellow) if not np.isnan(prev_yellow) else raw_yellow
        else:
            base = (prev_yellow + drift) if not np.isnan(prev_yellow) else raw_yellow
            cand = max(base, raw_yellow)
        if gb_on:
            cand = max(cand, raw_yellow + min_gap * a)
        new_yellow = max(cand, ol)
        prev_disp = prev_yellow if not np.isnan(prev_yellow) else ol
        move = "ratchet UP  " if new_yellow > prev_disp + 1e-9 else \
               ("giveback DN " if new_yellow < prev_disp - 1e-9 else "flat        ")
        yellow = new_yellow
        rows.append((pd.Timestamp(etime[j]), op[j], hi[j], lo[j], cl[j], a, raw_yellow, yellow,
                     move, "gb" if gaveback else ("floor@ORB" if abs(yellow - ol) < 1e-9 else "")))
        if hi[j] >= tp:
            exit_px, exit_ts, reason = tp, etime[j], "TP"; break
        if cl[j] <= yellow and cl[j] < op[j]:
            exit_px, exit_ts, reason = cl[j], etime[j], "YELLOW"; break
        if hhmm[j] >= TRADE_END_HHMM:
            exit_px, exit_ts, reason = cl[j], etime[j], "EOD"; break
        prev_yellow = yellow
    if exit_px is None:
        exit_px, exit_ts, reason = cl[-1], etime[-1], "EOD_LAST"
    net = (exit_px - ep) * DPP - COST
    return dict(entry_ts=pd.Timestamp(t0), entry=ep, orb_low=ol, tp=tp, atr_entry=atr[i],
                exit_ts=pd.Timestamp(exit_ts), exit=exit_px, reason=reason, net=net, rows=rows)


def fmt(t, n):
    L = []
    L.append("=" * 96)
    L.append(f"TRADE {n}   {pd.Timestamp(t['entry_ts']).date()}   LONG   -> {t['reason']}  pnl ${t['net']:+,.0f}")
    L.append(f"  ENTRY {pd.Timestamp(t['entry_ts'])}  entry={t['entry']:.2f}   "
             f"init SL (ORB_Low)={t['orb_low']:.2f}   TP(4R)={t['tp']:.2f}   ATR(entry)={t['atr_entry']:.2f}")
    L.append(f"  {'time':<20} {'open':>9} {'high':>9} {'low':>9} {'close':>9} {'atr':>6} {'rawSL':>9} {'SL(yellow)':>10}  move")
    for (ts, o, h, l, c, a, ry, y, mv, tag) in t["rows"]:
        L.append(f"  {str(ts):<20} {o:>9.2f} {h:>9.2f} {l:>9.2f} {c:>9.2f} {a:>6.2f} {ry:>9.2f} {y:>10.2f}  {mv} {tag}")
    L.append(f"  EXIT  {pd.Timestamp(t['exit_ts'])}  exit={t['exit']:.2f}  reason={t['reason']}  pnl=${t['net']:+,.0f}")
    return "\n".join(L)


def main():
    print("Loading 5-min bars...", flush=True)
    days = load_days(); keys = sorted(days.keys())
    trades = [(d, tr) for d in keys if (tr := trace_trade(days[d], **GB)) is not None]
    with open(OUT, "w") as f:
        for n, (d, t) in enumerate(trades, 1):
            f.write(fmt(t, n) + "\n\n")
    print(f"Wrote {len(trades)} trades -> {OUT}\n")

    # print 3 representative: a giveback-heavy one, a long ratchet-to-EOD winner, a yellow stop
    gbheavy = max(trades, key=lambda x: sum(1 for r in x[1]["rows"] if r[9] == "gb"))
    winner = max((x for x in trades if x[1]["reason"] in ("EOD", "EOD_LAST")),
                 key=lambda x: x[1]["net"], default=trades[0])
    stop = next((x for x in trades if x[1]["reason"] == "YELLOW" and x[1]["net"] > 0), trades[0])
    seen = set()
    for label, (d, t) in [("MOST GIVEBACKS", gbheavy), ("BIG EOD WINNER (ratchet up)", winner),
                          ("YELLOW STOP (locked profit)", stop)]:
        if id(t) in seen: continue
        seen.add(id(t))
        print(f"\n### {label} ###")
        print(fmt(t, trades.index((d, t)) + 1))


if __name__ == "__main__":
    main()
