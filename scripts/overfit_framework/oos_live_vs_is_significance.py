"""Is the OOS + live-paper record distinguishable from noise, given the in-sample result?

Three windows, SAME live engine throughout (baselines in live/combined/state/live_*_trades.csv
are the full-history replays of the engines that are actually running):
  IS    : start        -> 2024-01-01
  OOS   : 2024-01-01   -> end of backtest (~2026-04/05)
  LIVE  : 2026-05-19   -> now   (live/combined/state/paper/live_*_signals.csv, forward paper)

Unit is POINTS PER CONTRACT so the three windows are directly comparable (martingale removed;
live qty ignored).

Reported per strat and for the pooled combined:
  - n, mean, sd, t vs zero, p  for each window
  - 95% CI on the OOS+LIVE mean, and whether it contains 0 and contains the IS mean
  - Welch two-sample test OOS+LIVE vs IS (did the edge CHANGE?)
  - POWER: given the observed per-trade sd, the n needed to detect the IS edge at 80% power,
    and the minimum edge the actual OOS+LIVE n could have detected.

Run:  python scripts/overfit_framework/oos_live_vs_is_significance.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "live" / "combined" / "state"
PAPER = STATE / "paper"
IS_END = pd.Timestamp("2024-01-01", tz="America/New_York")
ET = "America/New_York"

BASE = {"RV": "live_rv_trades.csv", "B2": "live_b2_trades.csv",
        "OD": "live_od_trades.csv", "FB": "live_fabio_trades.csv"}


def backtest(strat: str) -> pd.DataFrame:
    d = pd.read_csv(STATE / BASE[strat])
    ts = pd.to_datetime(d["entry_ts"], utc=True, format="mixed").dt.tz_convert(ET)
    return pd.DataFrame({"ts": ts, "pts": d["pnl_points"].astype(float)})


def paper(strat: str) -> pd.DataFrame:
    """Pair ENTRY->EXIT from the forward paper log; points per contract."""
    p = PAPER / f"live_{strat.lower()}_signals.csv"
    d = pd.read_csv(p)
    d["ts"] = pd.to_datetime(d["ts_et"].astype(str).str.replace(r"\s*(EDT|EST)$", "", regex=True))
    rows, open_e = [], None
    for _, r in d.iterrows():
        if r["event"] == "ENTRY":
            open_e = r
        elif r["event"] == "EXIT" and open_e is not None:
            sgn = 1 if open_e["direction"] == "LONG" else -1
            rows.append({"ts": open_e["ts"].tz_localize(ET),
                         "pts": (r["price"] - open_e["price"]) * sgn})
            open_e = None
    return pd.DataFrame(rows)


def desc(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return {"n": n, "mean": np.nan, "sd": np.nan, "t": np.nan, "p": np.nan}
    t, p = stats.ttest_1samp(x, 0.0)
    return {"n": n, "mean": x.mean(), "sd": x.std(ddof=1), "t": t, "p": p}


def main():
    print(f"IS/OOS split at {IS_END.date()}; LIVE = forward paper log\n")
    pool = {"IS": [], "OOS": [], "LIVE": []}
    rows = []

    for s in ["OD", "RV", "B2", "FB"]:
        bt, pa = backtest(s), paper(s)
        live_start = pa["ts"].min() if len(pa) else None
        # guard against the baseline overlapping the paper window
        bt_oos = bt[(bt.ts >= IS_END) & ((live_start is None) | (bt.ts < live_start))]
        w = {"IS": bt[bt.ts < IS_END]["pts"].values,
             "OOS": bt_oos["pts"].values,
             "LIVE": pa["pts"].values if len(pa) else np.array([])}
        for k in pool:
            pool[k].append(w[k])
        rows.append((s, w))

    rows.append(("COMBO", {k: np.concatenate(v) for k, v in pool.items()}))

    hdr = (f"{'':<7}{'window':<7}{'n':>6}{'mean':>9}{'sd':>8}{'t':>8}{'p':>10}")
    for s, w in rows:
        print("=" * 74)
        print(f"{s}")
        print(hdr); print("-" * 74)
        for k in ["IS", "OOS", "LIVE"]:
            d = desc(w[k])
            print(f"{'':<7}{k:<7}{d['n']:>6}{d['mean']:>9.2f}{d['sd']:>8.1f}"
                  f"{d['t']:>8.2f}{d['p']:>10.4f}")
        comb = np.concatenate([w["OOS"], w["LIVE"]])
        d = desc(comb); i = desc(w["IS"])
        print(f"{'':<7}{'OOS+LIVE':<7}{d['n']:>6}{d['mean']:>9.2f}{d['sd']:>8.1f}"
              f"{d['t']:>8.2f}{d['p']:>10.4f}")

        if d["n"] > 2 and i["n"] > 2:
            se = d["sd"] / np.sqrt(d["n"])
            lo, hi = d["mean"] - 1.96 * se, d["mean"] + 1.96 * se
            tt, pp = stats.ttest_ind(comb, w["IS"], equal_var=False)
            # power: n needed to detect the IS mean at 80% power, two-sided a=0.05
            need = int(np.ceil((2.802 * d["sd"] / i["mean"]) ** 2)) if i["mean"] > 0 else -1
            mde = 2.802 * se     # smallest mean OOS+LIVE could detect at 80% power
            print(f"\n       OOS+LIVE 95% CI on mean pts/trade: [{lo:+.2f}, {hi:+.2f}]"
                  f"   contains 0: {'YES' if lo <= 0 <= hi else 'no'}"
                  f"   contains IS mean ({i['mean']:+.2f}): {'YES' if lo <= i['mean'] <= hi else 'no'}")
            print(f"       Welch OOS+LIVE vs IS: t={tt:+.2f}  p={pp:.3f}"
                  f"  -> {'edge CHANGED' if pp < 0.05 else 'no detectable change'}")
            print(f"       power: need n={need} to detect the IS edge ({i['mean']:+.2f} pts) at 80%;"
                  f" have n={d['n']}  -> {'ADEQUATE' if 0 < need <= d['n'] else 'UNDERPOWERED'}")
            print(f"       smallest edge this n could detect: {mde:+.2f} pts/trade")
        print()


if __name__ == "__main__":
    main()
