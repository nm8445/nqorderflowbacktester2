"""Does adding MEC (atr28/sl1.75, 1:1) to the funded 4-way combined speed up payouts without changing
the other stats? Compares 4-way vs 5-way (combined+MEC) on: milk daily green-rate, days-to-5-wins,
gamble reach, and E[$/funded] + total funded days.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np, pandas as pd
from scripts.propfirm_milking.common import eval_bracket, load_1min_bars, ET
from scripts.propfirm_milking.entries import build_entry_packs
from scripts.propfirm_milking.farm_config_wr import LIVE_STOP, orb_low_for_date
from funded_rr_sweep import build_all_packs, combined_R
from funded_target_risk_sweep import _gamble
from farm_income_mc import _milk, SPLIT, MIN_WD, WINS_NEEDED, MAX_WD, WIN_DAY

MILK_CSV = Path(__file__).resolve().parents[1] / "futurespropmc" / "results" / "combined_4way_with_mae_1min.csv"


def fourway_daily_by_date():
    """4-way 1-MNQ daily P&L keyed by FIRM date (OD 19:00 -> next business day)."""
    df = pd.read_csv(MILK_CSV)
    ts = pd.to_datetime(df["ts"], utc=True, format="mixed").dt.tz_convert(ET)
    base = ts.dt.normalize()
    fd = base.where(ts.dt.hour < 17, base + pd.tseries.offsets.BDay(1)).dt.date
    return df.assign(fd=fd).groupby("fd")["pnl_1c"].sum() / 10.0   # Series date->$ (1 MNQ)


def mec_daily_by_date(df1m):
    """MEC 1-MNQ daily P&L (atr28/sl1.75 1:1). 10:00 entry -> firm date = calendar date."""
    packs = build_entry_packs("MEC", atr_lens=[28], verbose=False)
    tr = eval_bracket(packs, atr_len=28, sl_mult=1.75, tp_mult=1.75)
    tr = tr.copy(); tr["pnl_1mnq"] = tr["pnl_pts"] * 2.0   # 1 MNQ = $2/pt
    return tr.groupby("date")["pnl_1mnq"].sum()


def milk_days_to_5(daily, n=100000, seed=3, dd=2000.0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        c = dd; w = 0; d = 0
        while w < WINS_NEEDED:
            d += 1; pnl = daily[rng.integers(daily.size)]; c += pnl
            if c <= 0: out.append((False, d)); break
            if pnl >= WIN_DAY: w += 1
        else:
            out.append((True, d))
    ok = np.array([o[0] for o in out]); dys = np.array([o[1] for o in out])
    return ok.mean(), np.median(dys[ok])


def main():
    df1m = load_1min_bars()
    four = fourway_daily_by_date()
    mec = mec_daily_by_date(df1m)
    five = four.add(mec, fill_value=0.0)            # combine on date; MEC-only days included
    for name, s in [("4-way", four), ("5-way (+MEC)", five)]:
        a = s.to_numpy()
        print(f"{name:>13} milk daily: n={a.size}  mean=${a.mean():.0f}  green>=$150={(a>=WIN_DAY).mean()*100:.0f}%  red={(a<0).mean()*100:.0f}%")
    print()
    for name, s in [("4-way", four), ("5-way (+MEC)", five)]:
        p, md = milk_days_to_5(s.to_numpy())
        print(f"{name:>13} milk: P(5 wins @ $2k DD)={p*100:.1f}%  median days-to-5-wins={md:.0f}")

    # full funded life: gamble (combined +/- MEC R) + milk (4 or 5-way daily). Structure A, RR 1.0.
    packs = build_all_packs(df1m)
    comb_R = combined_R(packs, 1.0)
    mec_packs = build_entry_packs("MEC", atr_lens=[28], verbose=False)
    mec_tr = eval_bracket(mec_packs, atr_len=28, sl_mult=1.75, tp_mult=1.75)
    mec_R = (mec_tr["pnl_pts"] / mec_tr["sl_dist_pts"]).to_numpy()
    five_R = np.concatenate([comb_R, mec_R])

    def funded_life(poolR, daily, rng):
        pool_d = poolR * 1500.0
        p, gw = _gamble(pool_d, 3000.0, rng, locked=False); td = 0
        if p is None: return 0.0, td
        cash = 0.0; nwd = 0
        while nwd < MAX_WD:
            if nwd > 0:
                p2, gw = _gamble(pool_d, 3000.0, rng, locked=True, start=p)
                if p2 is None: return cash, td
                p = p2
            need = max(0, WINS_NEEDED - gw)
            if need > 0:
                p, md = _milk(p, need, daily, rng, 0.0); td += md
                if p is None: return cash, td
            w = p / 2.0
            if w < MIN_WD: return cash, td
            cash += w * SPLIT; p -= w; nwd += 1
        return cash, td

    print()
    for name, poolR, daily in [("4-way", comb_R, four.to_numpy()), ("5-way (+MEC)", five_R, five.to_numpy())]:
        rng = np.random.default_rng(5)
        res = [funded_life(poolR, daily, rng) for _ in range(40000)]
        cash = np.array([r[0] for r in res]); days = np.array([r[1] for r in res])
        print(f"{name:>13}: E[$/funded]=${cash.mean():.0f}  median funded-life days(among paid)={np.median(days[cash>0]):.0f}")


if __name__ == "__main__":
    main()
