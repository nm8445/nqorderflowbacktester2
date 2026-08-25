"""Does the 4-way drift into PROFIT before LOSS? First-passage +50pt vs -100pt at 1 NQ.

User's farm idea (2026-07-30): 50k account, $2k EOD trailing-lock DD, 1 NQ per trade.
Cut at +$1,000 (= +50 pts). The account itself is the stop: room is always $2,000 (= 100 pts),
because the floor trails to peak-2000 and locks at start. Pass = +$3,000 = 3 winning cuts.
So every trade is a race: +50 pts (bank $1k) vs -100 pts (account dead).

NULL: a driftless random walk hits +50 before -100 with p = 100/(50+100) = 66.67%.
Anything above that is genuine "profit comes first" ordering edge.

Horizon = the strategy's own exit (with the exit-bar offset, per reference_mae_exit_bar_bug).
If neither level is touched, the trade closes at the strategy's exit P&L.

Run: python scripts/futurespropmc/first_passage_50_100.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRADES = ROOT / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"
ONE_MIN = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
ET = "America/New_York"
BAR_MIN = {"OD": 20, "RV": 20, "B2": 20, "FB": 5}
TP_PTS, SL_PTS, NQ_PT = 50.0, 100.0, 20.0
NULL_P = SL_PTS / (TP_PTS + SL_PTS)


def build():
    d1 = pd.read_parquet(ONE_MIN)
    if d1.index.tz is None:
        d1.index = d1.index.tz_localize("UTC")
    d1.index = d1.index.tz_convert(ET); d1 = d1.sort_index()
    idx = d1.index.values.astype("int64")
    hi, lo, cl = d1["high"].values, d1["low"].values, d1["close"].values

    t = pd.read_csv(TRADES)
    for c in ("entry_ts", "exit_ts"):
        t[c] = pd.to_datetime(t[c], utc=True, format="mixed").dt.tz_convert(ET)
    t = t.sort_values("entry_ts").reset_index(drop=True)

    rows = []
    for _, r in t.iterrows():
        s = r["strat"]; lng = r["direction"] == "LONG"
        fill = r["entry_ts"] + (pd.Timedelta(minutes=20) if s == "OD" else pd.Timedelta(0))
        xend = r["exit_ts"] + pd.Timedelta(minutes=BAR_MIN[s])
        f = np.int64(fill.value); x = np.int64(xend.value)
        ep_i = int(np.searchsorted(idx, f, "right")) - 1
        a = int(np.searchsorted(idx, f, "right")); b = int(np.searchsorted(idx, x, "right"))
        if ep_i < 0 or b <= a:
            continue
        ep = cl[ep_i]
        if lng:
            tp_hits = np.nonzero(hi[a:b] >= ep + TP_PTS)[0]
            sl_hits = np.nonzero(lo[a:b] <= ep - SL_PTS)[0]
        else:
            tp_hits = np.nonzero(lo[a:b] <= ep - TP_PTS)[0]
            sl_hits = np.nonzero(hi[a:b] >= ep + SL_PTS)[0]
        it = tp_hits[0] if tp_hits.size else np.inf
        isl = sl_hits[0] if sl_hits.size else np.inf
        if it == np.inf and isl == np.inf:
            out, pnl = "neither", (cl[b - 1] - ep) * (1 if lng else -1) * NQ_PT
        elif isl <= it:                      # tie -> loss first (conservative)
            out, pnl = "sl", -SL_PTS * NQ_PT
        else:
            out, pnl = "tp", TP_PTS * NQ_PT
        rows.append({"date": str(fill.date()), "ts": fill, "strat": s, "outcome": out, "pnl": pnl})
    return pd.DataFrame(rows)


def main():
    d = build()
    print(f"First passage +{TP_PTS:.0f}pt vs -{SL_PTS:.0f}pt at 1 NQ (TP=+$1,000 / blow=-$2,000)")
    print(f"NULL (driftless random walk): P(TP first) = {NULL_P:.1%}\n")
    print(f"  {'leg':>6} {'n':>6} {'TP first':>9} {'SL first':>9} {'neither':>9} "
          f"{'P(TP|resolved)':>16} {'vs null':>9}")
    for s in ["RV", "B2", "OD", "FB", "ALL"]:
        g = d if s == "ALL" else d[d.strat == s]
        n = len(g); tp = (g.outcome == "tp").sum(); sl = (g.outcome == "sl").sum()
        ne = (g.outcome == "neither").sum()
        p = tp / (tp + sl) if tp + sl else np.nan
        z = (p - NULL_P) / np.sqrt(NULL_P * (1 - NULL_P) / (tp + sl)) if tp + sl else np.nan
        print(f"  {s:>6} {n:>6} {tp:>9} {sl:>9} {ne:>9} {p:>15.1%} {z:>+8.1f} sd")
    print(f"\n  (sd = z-score of P(TP|resolved) against the {NULL_P:.1%} null)")
    d.to_csv(ROOT / "scripts" / "futurespropmc" / "results" / "first_passage_50_100.csv", index=False)


if __name__ == "__main__":
    main()
