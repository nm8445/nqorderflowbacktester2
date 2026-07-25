"""Paper-vs-backtest tracker for the live 4-way combined (RV / B2 / OD / FB).

Compares the FORWARD live/paper signals the running system fires
  live/combined/state/paper/live_<strat>_signals.csv   (ENTRY/EXIT events, since go-live)
against the SAME live engine's full-history replay baseline
  live/combined/state/live_<name>_trades.csv           (2020-12 -> 2026, backtest parity)

For each strat + combined it prints n, win-rate, profit-factor, avg points/trade (per
contract), realized $, and an n-aware "consistent / diverging" verdict: does the backtest
win-rate fall inside the live win-rate's 95% binomial CI? Small samples are flagged so you
don't over-read a hot/cold streak.

Also splits FB at the 2026-07-09 giveback-SL deploy so the new stop is tracked separately.

Usage:
    python live/combined/paper_tracker.py            # print table
    python live/combined/paper_tracker.py --snapshot # also append a dated row to state/paper/tracker_history.csv
"""
from __future__ import annotations
import argparse
from datetime import date
from pathlib import Path
import math
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
PAPER = HERE / "state" / "paper"
STATE = HERE / "state"
NQ_PT = 20.0
FB_GIVEBACK_DEPLOY = pd.Timestamp("2026-07-09")

PAPER_FILES = {s: PAPER / f"live_{s}_signals.csv" for s in ["rv", "b2", "od", "fb"]}
BASELINE_FILES = {
    "rv": STATE / "live_rv_trades.csv",
    "b2": STATE / "live_b2_trades.csv",
    "od": STATE / "live_od_trades.csv",
    "fb": STATE / "live_fabio_trades.csv",
}


def pair_paper(path: Path) -> pd.DataFrame:
    """Pair ENTRY->EXIT events into trades. per-contract points = (exit-entry)*dir (qty-free)."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    clean = df["ts_et"].str.replace(r" ED?ST| ES?T", "", regex=True)
    df["ts"] = pd.to_datetime(clean, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    rows, open_e = [], None
    for _, r in df.iterrows():
        if r["event"] == "ENTRY":
            open_e = r
        elif r["event"] == "EXIT" and open_e is not None:
            d = 1 if open_e["direction"] == "LONG" else -1
            pts = (r["price"] - open_e["price"]) * d          # per contract
            qty = int(r.get("qty", 1) or 1)
            rows.append({"entry_ts": open_e["ts"], "exit_ts": r["ts"], "pts": pts,
                         "usd": pts * NQ_PT * qty, "reason": r["reason"]})
            open_e = None
    return pd.DataFrame(rows)


def load_baseline(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    col = "pnl_points" if "pnl_points" in df.columns else None
    return pd.DataFrame({"pts": df[col].astype(float)}) if col else pd.DataFrame()


def stats(pts: np.ndarray):
    pts = np.asarray(pts, float)
    n = len(pts)
    if n == 0:
        return None
    wins = pts > 0
    gp = pts[pts > 0].sum(); gl = -pts[pts < 0].sum()
    return {"n": n, "wr": wins.mean() * 100,
            "pf": (gp / gl) if gl > 0 else float("inf"),
            "avg": pts.mean()}


def wr_ci(k: int, n: int):
    """95% Wilson interval for win-rate proportion (%)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n; z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return ((c - h) * 100, (c + h) * 100)


def conf_flag(n: int) -> str:
    return "noise" if n < 15 else ("thin" if n < 40 else ("ok" if n < 100 else "solid"))


def verdict(live: dict, bt: dict, live_wins: int) -> str:
    if live is None or bt is None:
        return "-"
    lo, hi = wr_ci(live_wins, live["n"])
    inside = lo <= bt["wr"] <= hi
    if live["n"] < 15:
        return "too few (n<15)"
    return "consistent" if inside else ("AHEAD" if live["wr"] > bt["wr"] else "BEHIND")


def main(snapshot: bool):
    print(f"Paper-vs-backtest tracker  ({date.today()})")
    print("live paper signals  vs  same-engine full-history replay baseline\n")
    header = f"{'strat':<6}{'n':>4}{'WR%':>6}{'PF':>6}{'avg_pt':>8}{'$real':>9}  |  {'bt_WR%':>7}{'bt_PF':>7}{'bt_avg':>8}  {'conf':>6}  verdict"
    print(header); print("-" * len(header))

    combo_pts, combo_usd = [], []
    for s in ["rv", "b2", "od", "fb"]:
        pp = pair_paper(PAPER_FILES[s])
        bl = load_baseline(BASELINE_FILES[s])
        lv = stats(pp["pts"].values) if len(pp) else None
        bt = stats(bl["pts"].values) if len(bl) else None
        if len(pp):
            combo_pts += list(pp["pts"].values); combo_usd += list(pp["usd"].values)
        if lv is None:
            print(f"{s.upper():<6}  (no paper trades yet)"); continue
        lw = int((pp["pts"] > 0).sum())
        vd = verdict(lv, bt, lw)
        pf = "inf" if lv["pf"] == float("inf") else f"{lv['pf']:.2f}"
        btcell = (f"{bt['wr']:>7.1f}{bt['pf']:>7.2f}{bt['avg']:>8.1f}" if bt else f"{'—':>7}{'—':>7}{'—':>8}")
        print(f"{s.upper():<6}{lv['n']:>4}{lv['wr']:>6.1f}{pf:>6}{lv['avg']:>8.1f}{pp['usd'].sum():>9,.0f}  |  "
              f"{btcell}  {conf_flag(lv['n']):>6}  {vd}")

    cs = stats(combo_pts)
    if cs:
        pf = "inf" if cs["pf"] == float("inf") else f"{cs['pf']:.2f}"
        print("-" * len(header))
        print(f"{'COMBO':<6}{cs['n']:>4}{cs['wr']:>6.1f}{pf:>6}{cs['avg']:>8.1f}{sum(combo_usd):>9,.0f}  |  "
              f"{'':>22}  {conf_flag(cs['n']):>6}")

    # FB giveback split
    fb = pair_paper(PAPER_FILES["fb"])
    if len(fb):
        pre = fb[fb.entry_ts < FB_GIVEBACK_DEPLOY]; post = fb[fb.entry_ts >= FB_GIVEBACK_DEPLOY]
        print("\nFB new-SL (giveback, deployed 2026-07-09):")
        for lbl, x in (("pre  static ", pre), ("post giveback", post)):
            st = stats(x["pts"].values)
            if st:
                pf = "inf" if st["pf"] == float("inf") else f"{st['pf']:.2f}"
                print(f"  {lbl}: n={st['n']:>2}  WR {st['wr']:>5.1f}%  PF {pf:>5}  "
                      f"avg {st['avg']:>6.1f}pt  ${x['usd'].sum():>7,.0f}  [{conf_flag(st['n'])}]")

    print("\nnotes: avg_pt & PF are PER-CONTRACT (martingale removed) so they compare to backtest;")
    print("       $real includes live qty (OD/B2 martingale). verdict = does bt WR sit in live WR's 95% CI.")

    if snapshot:
        hist = PAPER / "tracker_history.csv"
        row = {"date": date.today().isoformat()}
        for s in ["rv", "b2", "od", "fb"]:
            pp = pair_paper(PAPER_FILES[s]); st = stats(pp["pts"].values) if len(pp) else None
            if st:
                row[f"{s}_n"] = st["n"]; row[f"{s}_wr"] = round(st["wr"], 1)
                row[f"{s}_pf"] = round(st["pf"], 3) if st["pf"] != float("inf") else 99
                row[f"{s}_usd"] = round(pp["usd"].sum())
        if cs:
            row["combo_n"] = cs["n"]; row["combo_pf"] = round(cs["pf"], 3); row["combo_usd"] = round(sum(combo_usd))
        pd.DataFrame([row]).to_csv(hist, mode="a", header=not hist.exists(), index=False)
        print(f"\nappended snapshot -> {hist}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true", help="append a dated row to tracker_history.csv")
    main(ap.parse_args().snapshot)
