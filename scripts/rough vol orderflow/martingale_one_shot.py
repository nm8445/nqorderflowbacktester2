"""
Test specific martingale rule:
  - Default size = 1
  - Loss at size=1 -> next trade is size=2 (one shot)
  - After the size=2 trade (W or L) -> back to size=1
  - If next size=1 trade is a loss -> size=2 again (single shot)
  - And so on.

Tested across:
  - "any_loss": any losing trade triggers the next doubling
  - "fc_only": only force_close losing trades trigger
  - reset at day boundary vs continuous
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"

# Load the full trade log captured by audit_and_log.py
df = pd.read_csv(RESULTS_DIR / "inspect_v3_FULL_log.csv")
df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True).dt.tz_convert("America/New_York")
df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("America/New_York")
df = df.sort_values("entry_ts").reset_index(drop=True)
df["date"] = df["entry_ts"].dt.date


def replay_one_shot(df, qual="any_loss", per_day_reset=False):
    """One-shot doubling rule.

    qual: 'any_loss' or 'fc_only' — what counts as a triggering loss
    per_day_reset: if True, reset to qty=1 at each new day
    """
    pnls = np.zeros(len(df))
    qtys = np.zeros(len(df), dtype=np.int32)
    qty = 1
    prev_date = None
    for i, r in enumerate(df.itertuples()):
        if per_day_reset and r.date != prev_date:
            qty = 1
        qtys[i] = qty
        sized_pnl = r.pnl_dollars * qty
        pnls[i] = sized_pnl
        is_loss = r.pnl_dollars < 0
        is_qualifying_loss = is_loss and (qual == "any_loss" or
                                          (qual == "fc_only" and r.reason == "force_close"))
        # Update qty for next trade
        if qty == 2:
            qty = 1  # one-shot, always reset
        elif qty == 1 and is_qualifying_loss:
            qty = 2  # trigger doubling for next trade only
        # else: qty stays 1
        prev_date = r.date
    return pnls, qtys


def metrics(pnls, qtys, label):
    p = pnls
    w = p[p > 0]; l = p[p < 0]
    pf = w.sum() / abs(l.sum()) if len(l) else 99.0
    wr = 100 * len(w) / len(p)
    cum = p.cumsum()
    mdd = (cum - np.maximum.accumulate(cum)).min()
    mar = p.sum() / abs(mdd) if mdd < 0 else 99.0
    return dict(label=label, trades=len(p), pf=pf, wr=wr, pnl=p.sum(),
                mdd=mdd, mar=mar, max_q=int(qtys.max()),
                doubled_pct=100*(qtys==2).sum()/len(qtys))


def baseline():
    p = df["pnl_dollars"].to_numpy()
    w = p[p > 0]; l = p[p < 0]
    pf = w.sum()/abs(l.sum())
    cum = p.cumsum()
    mdd = (cum - np.maximum.accumulate(cum)).min()
    print(f"{'BASELINE (no mart)':>40} {'-':>9}  {len(p):>4}t  PF {pf:>4.2f}  PnL ${p.sum():>+9,.0f}  "
          f"MDD ${mdd:>+9,.0f}  MAR {p.sum()/abs(mdd):>4.2f}")


def main():
    print("\n=== One-shot doubling test ===")
    print(f"Rule: loss at qty=1 -> next trade qty=2 (single shot) -> back to qty=1 regardless")
    print(f"{'config':>40} {'reset':>9}  {'tr':>4}  {'PF':>7}  {'PnL':>10}  {'MDD':>11}  {'MAR':>4}\n")
    baseline()
    print()

    out = []
    for qual in ("any_loss", "fc_only"):
        for reset in (False, True):
            pnls, qtys = replay_one_shot(df, qual=qual, per_day_reset=reset)
            m = metrics(pnls, qtys, f"one_shot {qual}")
            rlabel = "per_day" if reset else "continuous"
            print(f"{f'one-shot {qual}':>40} {rlabel:>9}  {m['trades']:>4}t  "
                  f"PF {m['pf']:>4.2f}  PnL ${m['pnl']:>+9,.0f}  "
                  f"MDD ${m['mdd']:>+9,.0f}  MAR {m['mar']:>4.2f}  "
                  f"(% trades at qty=2: {m['doubled_pct']:.1f}%)")
            out.append((qual, rlabel, m))

    # Now also a year-by-year breakdown for the BEST one
    print("\n\n=== Year-by-year for ONE-SHOT any_loss continuous ===")
    pnls, qtys = replay_one_shot(df, qual="any_loss", per_day_reset=False)
    df_y = df.copy()
    df_y["sized_pnl"] = pnls
    df_y["qty"] = qtys
    df_y["year"] = df_y["entry_ts"].dt.year
    print(f"{'year':>4} {'tr':>4} {'PF':>5} {'WR':>5} {'PnL$':>11} {'MDD$':>11} {'doubles':>8}")
    for y, g in df_y.groupby("year"):
        p = g["sized_pnl"].to_numpy()
        w = p[p>0]; l = p[p<0]
        pf = w.sum()/abs(l.sum()) if len(l) else 99
        wr = 100*len(w)/len(p)
        cum = p.cumsum()
        mdd = (cum - np.maximum.accumulate(cum)).min()
        dq = (g["qty"]==2).sum()
        print(f"{y:>4} {len(p):>4d} {pf:>5.2f} {wr:>4.1f}% {p.sum():>+11,.0f} {mdd:>+11,.0f} {dq:>8d}")

    # Save
    df_y.to_csv(RESULTS_DIR / "martingale_one_shot_log.csv", index=False)
    print(f"\nWrote martingale_one_shot_log.csv")


if __name__ == "__main__":
    main()
