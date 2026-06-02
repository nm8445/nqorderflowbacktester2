"""Dump detailed REVERSION trades (combined mode, single-account serial) for 2026 — entry/exit
time, entry reason, entry price, SL price, TP price, exit reason, pnl. Run:
    python scripts/fair_price_theory/inspect_reversion_trades.py
"""
from __future__ import annotations
import pandas as pd
from fair_price_strategy import load, day_signals, eval_trade

YEAR = 2026


def main():
    df = load()
    df["date"] = df.index.date
    rows = []
    for date, g in df.groupby("date", sort=True):
        if date.year != YEAR:
            continue
        sigs, arrs = day_signals(g)
        if not arrs:
            continue
        o, h, l, c, t, mins = arrs
        # combined mode, single position at a time
        sel = [s for s in sigs if s["kind"] == "CONT" or s["cand"] or s["bos"]]
        last_exit = -1
        for s in sorted(sel, key=lambda x: x["fill"]):
            if s["fill"] <= last_exit:
                continue
            pnl_pts, jx, reason = eval_trade(arrs, s["fill"], s["dir"], s["sl"], s["tp"])
            last_exit = jx
            if s["kind"] != "REV":
                continue
            entry = o[s["fill"]]
            d = s["dir"]
            sl_price = entry + s["sl"] if d == -1 else entry - s["sl"]
            tp_price = entry - s["tp"] if d == -1 else entry + s["tp"]
            rows.append(dict(
                date=str(date),
                entry_time=t[s["fill"]].strftime("%H:%M"),
                exit_time=t[jx].strftime("%H:%M"),
                side="SHORT" if d == -1 else "LONG",
                reason=f"{'above' if s['above'] else 'below'}-fair {'BOS' if s['bos'] else 'candle'}",
                fair=round(s["fair"], 2),
                cndl_rng=round(s["rng"], 1),
                entry=round(entry, 2),
                sl_price=round(sl_price, 2),
                tp_price=round(tp_price, 2),
                exit_reason=reason,
                pnl_pts=round(pnl_pts, 2),
            ))
    out = pd.DataFrame(rows)
    if out.empty:
        print(f"No reversion trades in {YEAR}."); return
    pd.set_option("display.width", 240); pd.set_option("display.max_rows", 200)
    print(f"{len(out)} reversion trades in {YEAR} (single-account serial, combined mode)\n")
    # later in the year first
    print("=== latest 25 ===")
    print(out.tail(25).to_string(index=False))
    out.to_csv("results/fpt_reversion_2026.csv", index=False)
    print("\nfull list -> results/fpt_reversion_2026.csv")


if __name__ == "__main__":
    main()
