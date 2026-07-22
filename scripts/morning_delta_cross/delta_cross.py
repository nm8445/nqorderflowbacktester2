"""Morning order-flow-imbalance cross: at 10:00 ET, trade the 9:30-10:00 buyer/seller delta.

  signal: delta = sum(buy_vol - sell_vol) over 9:30-10:00 ET.  more buyers -> LONG, more sellers -> SHORT.
  pct    = delta / (buy+sell)  (imbalance strength).
  exec   : enter 10:00 ET, 30-min ATR bracket (sl_mult/tp_mult), force-close 14:30 ET (MEC execution).

Questions: (1) does sign(delta) predict the 10:00->14:30 direction? (2) does the MAGNITUDE (|pct|) matter
-- bucket by |pct| and see if a stronger imbalance gives a better edge. IS/OOS split (60/40).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from scripts.propfirm_milking.common import (
    load_1min_bars, eval_bracket, is_oos_cutoff, ET, MNQ_POINT_VALUE,
)
from scripts.propfirm_milking.entries import packs_from_fills

VOL = Path("D:/trading_pythonbacktest_data/volumetric_1min_1tpl.parquet")
ATR_LEN, SL_MULT, TP_MULT = 10, 1.5, 1.5   # 1:1 ATR bracket (default; tunable)


def daily_delta() -> pd.DataFrame:
    v = pd.read_parquet(VOL, columns=["session_date", "bar_open_time", "buy_vol", "sell_vol"])
    bot = pd.to_datetime(v["bar_open_time"])
    if bot.dt.tz is None:
        bot = bot.dt.tz_localize(ET)
    mod = (bot.dt.hour * 60 + bot.dt.minute).to_numpy()
    m = (mod >= 570) & (mod < 600)                      # 9:30-10:00 ET
    g = v[m].groupby("session_date").agg(buy=("buy_vol", "sum"), sell=("sell_vol", "sum"))
    g = g[(g["buy"] + g["sell"]) > 0].copy()
    g["delta"] = g["buy"] - g["sell"]
    g["pct"] = g["delta"] / (g["buy"] + g["sell"])
    g.index = pd.to_datetime(g.index).date
    return g


def main():
    g = daily_delta()
    print(f"days with 9:30-10:00 flow: {len(g)}")

    # entries: 10:00 ET each day, sign = sign(delta)
    g = g[g["delta"] != 0]
    fills = [pd.Timestamp(d, tz=ET) + pd.Timedelta(hours=10) for d in g.index]
    signs = [1 if x > 0 else -1 for x in g["delta"]]
    packs = packs_from_fills("MEC", fills, signs, atr_lens=[ATR_LEN], verbose=False)  # 30-min ATR, FC 14:30
    tr = eval_bracket(packs, atr_len=ATR_LEN, sl_mult=SL_MULT, tp_mult=TP_MULT)
    if tr.empty:
        print("no trades"); return

    # join |pct| onto each trade (by date)
    pct_by_date = {d: abs(p) for d, p in zip(g.index, g["pct"])}
    tr["abs_pct"] = tr["date"].map(pct_by_date)
    cutoff = is_oos_cutoff(list(tr["date"]))
    tr["is"] = tr["date"] < cutoff

    def wr(s):  return (s["win"].mean() * 100) if len(s) else float("nan")
    def ev(s):  return s["pnl_$"].mean() if len(s) else float("nan")
    print(f"\nbracket: {SL_MULT}x/{TP_MULT}x ATR({ATR_LEN}), FC 14:30, IS/OOS cutoff {cutoff}")
    print(f"OVERALL  n={len(tr)}  WR={wr(tr):.1f}%  IS={wr(tr[tr['is']]):.1f}%  OOS={wr(tr[~tr['is']]):.1f}%  EV=${ev(tr):.0f}")

    # raw direction edge (bracket-independent): sign-correct forward move 10:00->FC
    raw = np.array([p.sign * (p.fc_close - p.entry_price) for p in packs])
    print(f"raw fwd edge (sign*move, pts): mean={raw.mean():+.2f}  >0 frac={ (raw>0).mean()*100:.1f}%")

    # bucket by imbalance strength
    print("\nby |pct| imbalance quintile (Q1=weakest .. Q5=strongest):")
    tr2 = tr.dropna(subset=["abs_pct"]).copy()
    tr2["q"] = pd.qcut(tr2["abs_pct"], 5, labels=[1, 2, 3, 4, 5])
    print(f"  {'Q':>2} {'|pct| range':>16} {'n':>5} {'WR':>6} {'IS':>6} {'OOS':>6} {'EV$':>7}")
    for q, s in tr2.groupby("q", observed=True):
        lo, hi = s["abs_pct"].min(), s["abs_pct"].max()
        print(f"  {q:>2} {lo:>7.3f}-{hi:<7.3f} {len(s):>5} {wr(s):>5.1f}% {wr(s[s['is']]):>5.1f}% "
              f"{wr(s[~s['is']]):>5.1f}% {ev(s):>7.0f}")


if __name__ == "__main__":
    main()
