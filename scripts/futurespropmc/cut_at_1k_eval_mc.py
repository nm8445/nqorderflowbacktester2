"""50k eval, 1 NQ/trade, cut at +$1,000, pass at +$3,000. How many of 10 evals pass?

Faithful model:
  room  = profit - floor, floor = min(0, max(-2000, peak_EOD - 2000))  -> room is $2,000 at the
          start of every day (the "100 point stop"), and GROWS with intraday realised profit.
  trade = walk 1-min bars from fill to the strategy's exit (exit-bar offset applied).
          * mae_pre = worst adverse excursion BEFORE the +50pt TP prints (or before exit).
            If mae_pre >= room -> the account is dead mid-trade. This is what actually kills you,
            not the nominal 100 pts, because room shrinks after a losing "neither" trade.
          * else if +50pt printed -> bank +$1,000
          * else                  -> take the strategy's own exit P&L
  1 signal per account at a time; accounts draw from the pooled 4-way signal stream in order.

Run: python scripts/futurespropmc/cut_at_1k_eval_mc.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRADES = ROOT / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"
ONE_MIN = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
ET = "America/New_York"
BAR_MIN = {"OD": 20, "RV": 20, "B2": 20, "FB": 5}
TP_PTS, NQ_PT = 50.0, 20.0
DD, TARGET = 2_000.0, 3_000.0
N_SIMS, MAX_TRADES = 40_000, 400
CACHE = ROOT / "scripts" / "futurespropmc" / "results" / "_cut1k_pool.csv"


def build():
    if CACHE.exists():
        return pd.read_csv(CACHE)
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
        ep_i = int(np.searchsorted(idx, np.int64(fill.value), "right")) - 1
        a = int(np.searchsorted(idx, np.int64(fill.value), "right"))
        b = int(np.searchsorted(idx, np.int64(xend.value), "right"))
        if ep_i < 0 or b <= a:
            continue
        ep = cl[ep_i]
        if lng:
            tph = np.nonzero(hi[a:b] >= ep + TP_PTS)[0]
            adverse = ep - lo[a:b]
        else:
            tph = np.nonzero(lo[a:b] <= ep - TP_PTS)[0]
            adverse = hi[a:b] - ep
        k = int(tph[0]) if tph.size else len(adverse) - 1
        mae_pre = max(0.0, float(np.max(adverse[:k + 1]))) * NQ_PT   # $ at 1 NQ, before TP prints
        hit = bool(tph.size)
        flat = (cl[b - 1] - ep) * (1 if lng else -1) * NQ_PT
        rows.append({"date": str(fill.date()), "strat": s, "tp": int(hit),
                     "mae_pre": mae_pre, "flat_pnl": flat})
    out = pd.DataFrame(rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE, index=False)
    return out


def sim(tp, mae, flat, rng, n_sims=N_SIMS):
    """Returns (passed, trades_taken)."""
    n = len(tp)
    res = np.zeros(n_sims, bool); nt = np.zeros(n_sims, int)
    for k in range(n_sims):
        prof = 0.0; peak = 0.0; floor = -DD
        for j in range(MAX_TRADES):
            i = rng.integers(0, n)
            room = prof - floor
            if mae[i] >= room:                      # floating dip eats the whole cushion
                nt[k] = j + 1; break
            prof += 1000.0 if tp[i] else flat[i]
            if prof >= TARGET:
                res[k] = True; nt[k] = j + 1; break
            peak = max(peak, prof)
            floor = min(0.0, max(-DD, peak - DD))
        else:
            nt[k] = MAX_TRADES
    return res, nt


def main():
    d = build()
    tp = d.tp.to_numpy(); mae = d.mae_pre.to_numpy(); flat = d.flat_pnl.to_numpy()
    print(f"pool n={len(d)}  |  TP printed on {tp.mean():.1%} of trades  |  "
          f"median mae_pre ${np.median(mae):,.0f}  p95 ${np.percentile(mae,95):,.0f}")
    print(f"  trades whose pre-TP dip alone exceeds a full $2,000 room: "
          f"{(mae>=DD).mean():.1%}\n")
    rng = np.random.default_rng(7)
    res, nt = sim(tp, mae, flat, rng)
    p = res.mean()
    print(f"50k eval — 1 NQ, cut at +$1,000, pass at +$3,000:")
    print(f"  P(pass)            {p:.1%}")
    print(f"  median trades      {int(np.median(nt))}   (passers: {int(np.median(nt[res]))})")
    print(f"\n  BUY 10 EVALS -> expected passes: {10*p:.2f}")
    for k in range(0, 7):
        from math import comb
        print(f"    P(exactly {k} of 10 pass) = {comb(10,k)*p**k*(1-p)**(10-k):.1%}")


if __name__ == "__main__":
    main()
