"""Force a HARD static bracket on every 4-way signal: TP = +X pts, SL = -Y pts. Then MC the 50k eval.

Because the bracket is static, every outcome is EXACT: +X*2*MNQ dollars or -Y*2*MNQ dollars.
Trades are held to first passage (no strategy exit), capped at HORIZON_DAYS.

Part A: P(TP first) for a grid of (TP, SL) point pairs. Null for a driftless walk = Y/(X+Y).
Part B: 50k eval MC ($2,000 trailing-lock DD, +$3,000 target) swept over MNQ size.
        At size S: TP=$2*X*S, SL=$2*Y*S. Wins needed = 3000/TP; losses tolerated ~ 2000/SL.
        Smaller S => more trades => the ordering edge compounds instead of being all-or-nothing.

Run: python scripts/futurespropmc/hard_bracket_farm_mc.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRADES = ROOT / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"
ONE_MIN = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
OUTDIR = ROOT / "scripts" / "futurespropmc" / "results"
ET = "America/New_York"
MNQ_PT = 2.0
DD, TARGET = 2_000.0, 3_000.0
HORIZON_MIN = 60 * 24 * 5          # 5 calendar days of 1-min bars
N_SIMS, MAX_TRADES = 30_000, 3_000


def load():
    d1 = pd.read_parquet(ONE_MIN)
    if d1.index.tz is None:
        d1.index = d1.index.tz_localize("UTC")
    d1.index = d1.index.tz_convert(ET); d1 = d1.sort_index()
    t = pd.read_csv(TRADES)
    t["entry_ts"] = pd.to_datetime(t["entry_ts"], utc=True, format="mixed").dt.tz_convert(ET)
    t = t.sort_values("entry_ts").reset_index(drop=True)
    return d1, t


def first_passage(d1, t, tp_pts, sl_pts):
    """1 = TP first, 0 = SL first, -1 = unresolved within horizon."""
    idx = d1.index.values.astype("int64")
    hi, lo, cl = d1["high"].values, d1["low"].values, d1["close"].values
    out = []
    for _, r in t.iterrows():
        lng = r["direction"] == "LONG"
        fill = r["entry_ts"] + (pd.Timedelta(minutes=20) if r["strat"] == "OD" else pd.Timedelta(0))
        a = int(np.searchsorted(idx, np.int64(fill.value), "right"))
        if a == 0 or a >= len(idx):
            out.append(-1); continue
        ep = cl[a - 1]
        b = min(a + HORIZON_MIN, len(idx))
        if lng:
            th = np.nonzero(hi[a:b] >= ep + tp_pts)[0]
            sh = np.nonzero(lo[a:b] <= ep - sl_pts)[0]
        else:
            th = np.nonzero(lo[a:b] <= ep - tp_pts)[0]
            sh = np.nonzero(hi[a:b] >= ep + sl_pts)[0]
        it = th[0] if th.size else np.inf
        isl = sh[0] if sh.size else np.inf
        out.append(-1 if (it == np.inf and isl == np.inf) else (0 if isl <= it else 1))
    return np.array(out), t["strat"].to_numpy()


def eval_mc(p, tp_d, sl_d, rng, n=N_SIMS):
    """50k eval: exact +tp_d / -sl_d per trade, trailing-lock $2k floor, +$3k target."""
    ok = np.zeros(n, bool); nt = np.zeros(n, int)
    for k in range(n):
        prof = 0.0; peak = 0.0; floor = -DD
        for j in range(MAX_TRADES):
            win = rng.random() < p
            prof += tp_d if win else -sl_d
            if prof <= floor + 1e-9:
                nt[k] = j + 1; break
            if prof >= TARGET:
                ok[k] = True; nt[k] = j + 1; break
            peak = max(peak, prof)
            floor = min(0.0, max(-DD, peak - DD))
        else:
            nt[k] = MAX_TRADES
    return ok.mean(), int(np.median(nt))


def main():
    d1, t = load()
    print("PART A — P(TP first) under a hard static bracket, held to first passage (5-day cap)\n")
    print(f"  {'TP':>5} {'SL':>5} {'null':>7} {'P(TP|res)':>10} {'resolved':>9} {'edge':>7}")
    grid = [(50, 100), (50, 150), (50, 200), (75, 150), (100, 100), (100, 200), (25, 50), (30, 60)]
    store = {}
    for tp_pts, sl_pts in grid:
        res, strat = first_passage(d1, t, tp_pts, sl_pts)
        m = res >= 0
        p = res[m].mean()
        null = sl_pts / (tp_pts + sl_pts)
        store[(tp_pts, sl_pts)] = p
        print(f"  {tp_pts:>5} {sl_pts:>5} {null:>6.1%} {p:>9.1%} {m.mean():>8.1%} {p-null:>+6.1%}")

    print("\nPART B — 50k eval MC at the 50/100 bracket, swept by size")
    p = store[(50, 100)]
    print(f"  (P(TP)={p:.1%} constant; only the $ per trade changes with size)")
    print(f"  {'MNQ':>4} {'TP$':>7} {'SL$':>7} {'wins req':>9} {'losses ok':>10} {'P(pass)':>9} {'med trades':>11}")
    rng = np.random.default_rng(7)
    for S in (1, 2, 3, 4, 5, 6, 8, 10):
        tp_d = 50 * MNQ_PT * S; sl_d = 100 * MNQ_PT * S
        pp, md = eval_mc(p, tp_d, sl_d, rng)
        print(f"  {S:>4} ${tp_d:>6,.0f} ${sl_d:>6,.0f} {TARGET/tp_d:>9.0f} {DD/sl_d:>10.1f} "
              f"{pp:>8.1%} {md:>11}")

    print("\nPART C — best bracket x size (P(pass), 50k eval)")
    print(f"  {'bracket':>10} {'P(TP)':>7} | " + " ".join(f"{S:>6}MNQ" for S in (1,2,3,5,10)))
    for (tp_pts, sl_pts), pv in store.items():
        row = []
        for S in (1, 2, 3, 5, 10):
            pp, _ = eval_mc(pv, tp_pts*MNQ_PT*S, sl_pts*MNQ_PT*S, np.random.default_rng(7), n=12000)
            row.append(f"{pp:>8.1%}")
        print(f"  {f'{tp_pts}/{sl_pts}':>10} {pv:>6.1%} | " + " ".join(row))


if __name__ == "__main__":
    main()
