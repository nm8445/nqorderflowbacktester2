"""Append realized live paper-trade signals to combined_4way_trades.csv.

The historical combined log (built by combined_4way.py from 4 backtest logs)
ends ~2026-05-22. This script extends it forward using the actual live paper
signals recorded in live/combined/state/paper/live_*_signals.csv.

For each strat it pairs ENTRY->EXIT chronologically, computes pnl_$ on the
NQ basis (points * $20 * qty, matching combined_4way's NQ_PT), keeps trades
whose entry is on/after the cutoff, appends them to combined_4way_trades.csv,
re-sorts by exit_ts and recomputes cum/peak/dd.

Open (unfilled) trailing entries are skipped.

Usage:
  python "scripts/rough vol orderflow/append_live_to_combined.py" --cutoff 2026-05-22
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
ROOT = HERE.parent.parent
PAPER_DIR = ROOT / "live" / "combined" / "state" / "paper"
ET_TZ = "America/New_York"
NQ_PT = 20.0

LIVE_FILES = {
    "RV": PAPER_DIR / "live_rv_signals.csv",
    "B2": PAPER_DIR / "live_b2_signals.csv",
    "OD": PAPER_DIR / "live_od_signals.csv",
    "FB": PAPER_DIR / "live_fb_signals.csv",
}


def _parse_ts(s: pd.Series) -> pd.Series:
    # strip trailing EDT/EST then localize to ET
    cleaned = s.astype(str).str.replace(r"\s*(EDT|EST)$", "", regex=True)
    return pd.to_datetime(cleaned).dt.tz_localize(ET_TZ)


def pair_strat(strat: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"  {strat}: no file at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["ts"] = _parse_ts(df["ts_et"])
    df = df.sort_values("ts").reset_index(drop=True)

    rows = []
    open_entry = None
    for _, r in df.iterrows():
        if r["event"] == "ENTRY":
            open_entry = {"entry_ts": r["ts"], "direction": str(r["direction"]).upper(),
                          "entry": float(r["price"]), "qty": float(r.get("qty", 1) or 1)}
        elif r["event"] == "EXIT" and open_entry is not None:
            exit_px = float(r["price"])
            pts = (exit_px - open_entry["entry"]) if open_entry["direction"] == "LONG" \
                else (open_entry["entry"] - exit_px)
            rows.append({
                "entry_ts": open_entry["entry_ts"],
                "exit_ts": r["ts"],
                "direction": open_entry["direction"],
                "pnl_$": pts * NQ_PT * open_entry["qty"],
                "strat": strat,
            })
            open_entry = None
    if open_entry is not None:
        print(f"  {strat}: trailing OPEN entry {open_entry['entry_ts']} skipped (no exit yet)")
    out = pd.DataFrame(rows)
    print(f"  {strat}: paired {len(out)} closed trades  (live signals)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2026-05-22",
                    help="Keep live trades with entry_ts >= this date (ET).")
    args = ap.parse_args()
    cutoff = pd.Timestamp(args.cutoff, tz=ET_TZ)

    print(f"Pairing live signals (entry >= {cutoff.date()}):")
    new = pd.concat([pair_strat(s, p) for s, p in LIVE_FILES.items()], ignore_index=True)
    new = new[new["entry_ts"] >= cutoff].copy()
    print(f"\nNew live trades after cutoff: {len(new)}  "
          f"(${new['pnl_$'].sum():+,.0f} NQ basis)")
    for s in ["RV", "B2", "OD", "FB"]:
        sub = new[new["strat"] == s]
        if len(sub):
            print(f"  {s}: {len(sub):>3} trades  ${sub['pnl_$'].sum():+,.0f}  "
                  f"{sub['exit_ts'].min().date()} -> {sub['exit_ts'].max().date()}")

    # Load existing combined log, drop any rows at/after cutoff (idempotent re-runs)
    comb_path = RESULTS_DIR / "combined_4way_trades.csv"
    comb = pd.read_csv(comb_path)
    comb["entry_ts"] = pd.to_datetime(comb["entry_ts"], utc=True).dt.tz_convert(ET_TZ)
    comb["exit_ts"] = pd.to_datetime(comb["exit_ts"], utc=True).dt.tz_convert(ET_TZ)
    before = len(comb)
    comb = comb[comb["entry_ts"] < cutoff].copy()
    if before != len(comb):
        print(f"\nDropped {before - len(comb)} existing rows with entry >= cutoff (re-run safe)")

    merged = pd.concat([comb[["entry_ts", "exit_ts", "direction", "pnl_$", "strat"]], new],
                       ignore_index=True)
    merged = merged.sort_values("exit_ts").reset_index(drop=True)
    merged["cum"] = merged["pnl_$"].cumsum()
    merged["peak"] = merged["cum"].cummax()
    merged["dd"] = merged["cum"] - merged["peak"]

    merged.to_csv(comb_path, index=False)
    print(f"\nWrote {comb_path}  ({len(merged)} trades, "
          f"{merged['exit_ts'].min().date()} -> {merged['exit_ts'].max().date()})")
    print(f"Total PnL: ${merged['pnl_$'].sum():+,.0f} NQ basis  "
          f"(= ${merged['pnl_$'].sum()/10:+,.0f} MNQ basis)")


if __name__ == "__main__":
    main()
