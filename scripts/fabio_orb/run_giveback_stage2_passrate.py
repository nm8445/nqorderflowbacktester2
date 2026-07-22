"""FB giveback STAGE 2: 1-MNQ 4-way combined eval pass rate with each FB stop variant.

For each FB stop variant we regenerate FB's per-trade (pnl_1c, mae_1c) — pnl from the 5-min backtest,
MAE from the worst 1-min low over (fill, exit] (same methodology as build_4way_mae.py) — splice it into
combined_4way_with_mae_1min.csv (replacing FB, keeping OD/RV/B2), and run the faithful futures_50 eval MC
at 1 MNQ (trailing-then-lock $2k DD, MAE-aware/floating-blowable). Reports:
  P(+$3k)  = pass rate (reach the +$3k target before the floor)
  P(+$2k)  = chance of reaching +$2k before blowing (one-way flag)
  med days = median trading days to pass.

Variants: original-CSV FB (sanity), my static FB (should ~match), trailing k=3, giveback k=1.5/gb0.3.

Run:  python scripts/fabio_orb/run_giveback_stage2_passrate.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_giveback_variant import load_days, run_static, run_giveback   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ONE_MIN = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
COMBINED = ROOT / "scripts" / "montecarlo" / "results" / "combined_4way_with_mae_1min.csv"
NQ_PT = 20.0
# eval MC (futures_50, faithful to futures_4way_eval.py)
START, TARGET, DD, LOCK, COST = 50000.0, 3000.0, 2000.0, 50000.0, 2.0
N_SIMS, CAP = 20000, 504


def gen_fb_rows(days, keys, one_min, run_fn, **kw):
    """Regenerate FB rows (date, pnl_1c, mae_1c) for a stop variant. MAE = worst 1-min low over hold."""
    idx, lo = one_min
    rows = []
    for d in keys:
        tr = run_fn(days[d], **kw)
        if tr is None:
            continue
        fill_ns = pd.Timestamp(tr["entry_time"]).value
        exit_ns = pd.Timestamp(tr["exit_time"]).value
        s = int(np.searchsorted(idx, fill_ns, "right"))
        e = int(np.searchsorted(idx, exit_ns, "right"))
        mae_pts = max(0.0, tr["entry"] - lo[s:e].min()) if e > s else 0.0
        rows.append({"date": str(d.date()), "strat": "FB",
                     "pnl_1c": tr["raw_pts"] * NQ_PT, "mae_1c": -(mae_pts * NQ_PT)})
    return pd.DataFrame(rows)


def day_packs(fb_rows):
    """Combined day packs (OD+RV+B2 from the CSV + this FB), ordered within day. FB has no 'ts' so it
    lands last in a day — fine (FB is the afternoon trade; ordering within a day barely moves the MC)."""
    base = pd.read_csv(COMBINED)
    base["date"] = base["date"].astype(str)
    keep = base[base.strat != "FB"][["date", "ts", "strat", "pnl_1c", "mae_1c"]].copy()
    fb = fb_rows.copy(); fb["ts"] = fb["date"] + " 13:00"     # sort key so FB orders near the RTH close
    both = pd.concat([keep, fb], ignore_index=True).sort_values(["date", "ts"])
    return [list(zip(g["pnl_1c"].to_numpy(), g["mae_1c"].to_numpy())) for _, g in both.groupby("date", sort=True)]


def sim(packs, rng, mnq=1):
    s = mnq / 10.0
    bal = START; peak = START; floor = START - DD; reached_2k = False
    n = len(packs)
    for d in range(CAP):
        tr = packs[rng.integers(0, n)]
        realized = 0.0; bust = False
        for pnl, mae in tr:
            adj = mae * s - COST * mnq                        # floating / MAE-aware
            if bal + realized + adj < floor:
                bust = True; break
            realized += pnl * s - COST * mnq
        if bust:
            return 0, reached_2k, d + 1
        bal += realized
        if bal - START >= 2000.0:
            reached_2k = True
        if bal - START >= TARGET:
            return 1, reached_2k, d + 1
        if bal > peak:
            peak = bal
        floor = min(LOCK, max(START - DD, peak - DD))
    return 0, reached_2k, CAP


def evaluate(label, packs):
    rng = np.random.default_rng(7)
    res = [sim(packs, rng) for _ in range(N_SIMS)]
    p3 = np.mean([r[0] for r in res])
    p2 = np.mean([r[1] for r in res])
    days = np.array([r[2] for r in res]); passed = np.array([r[0] for r in res])
    md = int(np.median(days[passed == 1])) if passed.any() else 0
    print(f"  {label:34s}  P(+$3k) {p3*100:5.1f}%   P(+$2k) {p2*100:5.1f}%   med days {md:>3}")
    return p3, p2


def main():
    print("Loading 5-min FB bars + 1-min bars...", flush=True)
    days = load_days(); keys = sorted(days.keys())
    df1 = pd.read_parquet(ONE_MIN, columns=["low"])
    if df1.index.tz is None:
        df1.index = df1.index.tz_localize("UTC")
    df1 = df1.sort_index()
    one_min = (df1.index.values.astype("int64"), df1["low"].values)
    print(f"  {len(keys)} FB days, {len(df1):,} 1-min bars\n", flush=True)

    print("1-MNQ 4-way (OD+RV+B2+FB) eval — futures_50, MAE-aware, +2k tracked:\n")
    # true baseline: the untouched combined CSV (original FB)
    base = pd.read_csv(COMBINED); base["date"] = base["date"].astype(str)
    packs0 = [list(zip(g.pnl_1c.to_numpy(), g.mae_1c.to_numpy()))
              for _, g in base.sort_values(["date", "ts"]).groupby("date", sort=True)]
    evaluate("CURRENT (original FB, all 4)", packs0)

    variants = [
        ("my static FB (sanity)", run_static, {}),
        ("FB trailing k=3 (no giveback)", run_giveback,
         dict(k=3.0, mode="drift_floor", drift=0.0, gb=0.0, scale_body=True, max_gb=0.5, min_gap=0.0)),
        ("FB giveback k=1.5 gb0.3 mingap0.3", run_giveback,
         dict(k=1.5, mode="drift_floor", drift=0.0, gb=0.3, scale_body=True, max_gb=0.5, min_gap=0.3)),
    ]
    for label, fn, kw in variants:
        fb_rows = gen_fb_rows(days, keys, one_min, fn, **kw)
        evaluate(label, day_packs(fb_rows))

    print("\n(P(+$2k) = ever reached +$2k before the floor; P(+$3k) = full pass.)")


if __name__ == "__main__":
    main()
