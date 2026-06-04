"""Eval-passer ROTATION sim — runs the actual lead+copy rotation across a pool of evals.

Four states per account: Fresh (never started), Active (trading today, has buffer), Done (hit the
daily cap today), Blown (-$2k). Rotation: each signal -> the next Fresh lead(s) start, and the signal
is copied onto every Active account that still has buffer. An Active keeps copying until it hits the
daily cap (+$1,500 / +$1,200) -> Done, or -$2k -> Blown. Pass when total >= +$3,000 (over multiple
Done days), trailing-then-lock $2k floor. <=20 evals -> 1 lead/signal; >20 -> 2.

Signal outcomes = 1-min first-touch of each real trade at the eval sizing (TP at the day-cap distance,
SL at the yellow), grouped by real trading day (so per-day signal counts + firing order are real).
This captures the CORRELATION the independent per-account MC missed: the variance of how many pass.

Run: python scripts/futurespropmc/eval_rotation_sim.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from datetime import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ONE_MIN = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
TRADES = ROOT / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"
ET = "America/New_York"
MULT = {"OD": 1.30, "B2": 2.50, "RV": 2.00}
FORCE = {"OD": (8, 0), "B2": (16, 0), "RV": (14, 45), "FB": (14, 0)}
DD, TARGET = 2000., 3000.


def wilder(b, n=14):
    pc = b.close.shift(1)
    tr = pd.concat([b.high - b.low, (b.high - pc).abs(), (b.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def per_day_outcomes(risk, tp_cap):
    """Return list of per-trading-day lists of signal $-outcomes (ordered by entry time)."""
    d1 = pd.read_parquet(ONE_MIN, columns=["open", "high", "low", "close"])
    if d1.index.tz is None: d1.index = d1.index.tz_localize("UTC")
    d1.index = d1.index.tz_convert(ET); d1 = d1.sort_index()
    idx = d1.index.values.astype("int64"); hi = d1.high.values; lo = d1.low.values; cl = d1.close.values
    b20 = d1.resample("20min", label="right", closed="right").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    atr = wilder(b20, 14); a_idx = b20.index.values.astype("int64"); a_val = atr.values
    b5 = d1.resample("5min", label="right", closed="right").agg({"high": "max", "low": "min"}).dropna()
    b5h = b5.index.hour * 100 + b5.index.minute
    orb = b5[(b5h > 830) & (b5h <= 900)]; orb_low = orb.groupby(orb.index.date)["low"].min()
    t = pd.read_csv(TRADES)
    t["entry_ts"] = pd.to_datetime(t["entry_ts"], utc=True).dt.tz_convert(ET)
    t = t.sort_values("entry_ts")
    tp_frac = tp_cap / risk
    rows = []
    for _, r in t.iterrows():
        s = r["strat"]; lng = r["direction"] == "LONG"
        fill = r["entry_ts"] + (pd.Timedelta(minutes=20) if s == "OD" else pd.Timedelta(0))
        f = np.int64(fill.value); ep = int(np.searchsorted(idx, f, "right")) - 1
        if ep < 0: continue
        entry = cl[ep]
        if s == "FB":
            ol = orb_low.get(fill.date())
            if ol is None or ol >= entry: continue
            stop = entry - ol
        else:
            ai = int(np.searchsorted(a_idx, f, "right")) - 1
            if ai < 0 or not np.isfinite(a_val[ai]) or a_val[ai] <= 0: continue
            stop = MULT[s] * a_val[ai]
        if stop <= 0: continue
        tp = entry + tp_frac * stop if lng else entry - tp_frac * stop
        sl = entry - stop if lng else entry + stop
        fc_date = fill.date() + pd.Timedelta(days=1) if s == "OD" else fill.date()
        fc = pd.Timestamp(fc_date, tz=ET) + pd.Timedelta(hours=FORCE[s][0], minutes=FORCE[s][1])
        st = int(np.searchsorted(idx, f, "right")); en = int(np.searchsorted(idx, np.int64(fc.value), "right")) - 1
        if en < st: continue
        out = None
        for j in range(st, en + 1):
            hsl = (lo[j] <= sl) if lng else (hi[j] >= sl)
            htp = (hi[j] >= tp) if lng else (lo[j] <= tp)
            if hsl: out = -risk; break
            if htp: out = tp_cap; break
        if out is None:
            pr = (cl[en] - entry) * (1 if lng else -1) / stop
            out = float(np.clip(pr, -1.0, tp_frac)) * risk
        rows.append((r["entry_ts"].date(), out))
    df = pd.DataFrame(rows, columns=["date", "out"])
    return [g["out"].tolist() for _, g in df.groupby("date", sort=True)]


def run_rotation(days, rng, n_evals, tp_cap, leads, max_days=70):
    n_days = len(days)
    total = np.zeros(n_evals); dayp = np.zeros(n_evals); peak = np.full(n_evals, 50000.)
    state = np.zeros(n_evals, dtype=int)   # 0 fresh, 1 active, 2 done(today), 3 blown, 4 passed
    used = max_days
    for dd in range(max_days):
        if np.all((state == 3) | (state == 4)):   # whole cohort resolved -> cycle done
            used = dd; break
        dayp[:] = 0.
        state[state == 2] = 1              # yesterday's Done resume as Active
        for out in days[rng.integers(0, n_days)]:
            nl = 0                          # promote next `leads` Fresh -> Active (start)
            for i in range(n_evals):
                if nl >= leads: break
                if state[i] == 0: state[i] = 1; nl += 1
            act = np.nonzero(state == 1)[0]  # every Active copies the signal
            for i in act:
                add = (tp_cap - dayp[i]) if (out > 0 and dayp[i] + out > tp_cap) else out
                total[i] += add; dayp[i] += add
                bal = 50000. + total[i]
                if bal > peak[i]: peak[i] = bal
                floor = min(50000., peak[i] - DD)
                if bal <= floor: state[i] = 3
                elif total[i] >= TARGET: state[i] = 4
                elif dayp[i] >= tp_cap: state[i] = 2
    return int(np.sum(state == 4)), int(np.sum(state == 3)), used


def main():
    print("EVAL-PASSER copy-count sweep — 40 evals, copies/signal 1..5\n")
    print("(copies = fresh accounts started per signal; every Active still copies until Done/Blown)\n")
    N = 3000; n_evals = 40
    for label, risk, tp_cap in [("50% rule (stop$1500/cap$1500)", 1500., 1500.),
                                ("40% rule (stop$1200/cap$1200)", 1200., 1200.)]:
        days = per_day_outcomes(risk, tp_cap)
        print(f"=== {label} ===   ({len(days)} days, ~{np.mean([len(d) for d in days]):.1f} signals/day)")
        print(f"{'copies':>7} {'mean pass':>10} {'as %':>6} {'p10':>5} {'p50':>5} {'p90':>5} "
              f"{'p90-p10':>8} {'cycle days':>11}")
        for leads in (1, 2, 3, 4, 5):
            rng = np.random.default_rng(7)
            res = [run_rotation(days, rng, n_evals, tp_cap, leads) for _ in range(N)]
            p = np.array([r[0] for r in res]); used = np.array([r[2] for r in res])
            p10, p90 = np.percentile(p, 10), np.percentile(p, 90)
            print(f"{leads:>7} {p.mean():>10.1f} {100*p.mean()/n_evals:>5.0f}% "
                  f"{p10:>5.0f} {np.percentile(p,50):>5.0f} {p90:>5.0f} {p90-p10:>8.0f} {used.mean():>11.1f}")
        print()


if __name__ == "__main__":
    main()
