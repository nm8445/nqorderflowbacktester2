"""Challenge pass rate, high-risk '2 winning days' approach.

Per trade, sized so the stop (yellow/ATR/ORB distance) = $2k = 1R. Consistency rule caps a day at
+$1,500 -> close at TP = +0.75R. Hard stop at -1R = -$2,000. First-touch decided on 1-MIN bars
(MFE vs MAE order) within the hold window (fill -> strat force-close). Outcome per trade:
  TP (+$1,500) | SL (-$2,000) | PARTIAL (force-closed between, realized).

Account: 50k, $2k trailing-then-lock floor. Each day take ONE signal (firing-order, freq-weighted).
Pass at +$3,000; blow if balance hits the trailing floor (so any -$2k SL after a winning day blows).
ATR computed UTC-consistently (the tz bug is fixed here). Run: python scripts/montecarlo/challenge_gamble_pass.py
"""
from __future__ import annotations
import pandas as pd, numpy as np
from pathlib import Path

ET = "America/New_York"
ONE_MIN = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
TRADES = Path(__file__).resolve().parents[2] / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"
MULT = {"OD": 1.30, "B2": 2.50, "RV": 2.00}
FORCE = {"OD": (8, 0), "B2": (16, 0), "RV": (14, 45), "FB": (14, 0)}
RISK, TP_R, SL_R = 2000., 0.75, 1.0


def wilder(b, n=14):
    pc = b.close.shift(1)
    tr = pd.concat([b.high - b.low, (b.high - pc).abs(), (b.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def build_outcomes(risk, tp_cap=1500.):
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
    t["exit_ts"] = pd.to_datetime(t["exit_ts"], utc=True).dt.tz_convert(ET)
    rows = []
    for _, r in t.iterrows():
        s = r["strat"]; lng = r["direction"] == "LONG"
        fill = r["entry_ts"] + (pd.Timedelta(minutes=20) if s == "OD" else pd.Timedelta(0))
        f = np.int64(fill.value)
        ep = int(np.searchsorted(idx, f, "right")) - 1
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
        tp_frac = tp_cap / risk                    # TP capped at the consistency limit = tp_frac x stop
        tp = entry + tp_frac * stop if lng else entry - tp_frac * stop
        sl = entry - stop if lng else entry + stop   # SL at the yellow (=1R in points), $ = risk
        fc_date = fill.date() + pd.Timedelta(days=1) if s == "OD" else fill.date()
        fc = pd.Timestamp(fc_date, tz=ET) + pd.Timedelta(hours=FORCE[s][0], minutes=FORCE[s][1])
        st = int(np.searchsorted(idx, f, "right")); en = int(np.searchsorted(idx, np.int64(fc.value), "right")) - 1
        if en < st: continue
        out = None; worst_adv = 0.0                 # worst adverse excursion (R fraction, <= 0)
        for j in range(st, en + 1):
            if lng:
                adv = (lo[j] - entry) / stop
                hsl, htp = lo[j] <= sl, hi[j] >= tp
            else:
                adv = (entry - hi[j]) / stop
                hsl, htp = hi[j] >= sl, lo[j] <= tp
            if adv < worst_adv: worst_adv = adv     # track the deepest floating dip up to resolution
            if hsl: out = "SL"; break          # same-bar tie -> SL (conservative)
            if htp: out = "TP"; break
        # mae = worst floating $ over the hold (>= -risk; a winner/partial that dipped deep still
        # blows a tight-DD account mid-trade even though its REALIZED g is positive).
        mae = max(worst_adv, -1.0) * risk
        if out == "TP":
            g = tp_cap
        elif out == "SL":
            g = -risk
        else:
            pr = (cl[en] - entry) * (1 if lng else -1) / stop
            g = float(np.clip(pr, -1.0, tp_frac)) * risk; out = "PARTIAL"
        rows.append((s, out, g, mae))
    return pd.DataFrame(rows, columns=["strat", "outcome", "g", "mae"])


def sim(pool, rng, max_days=40, mae_aware=True):
    """pool = array of (realized_g, mae_$) rows. mae_aware=True checks the trade's worst floating
    dip against the trailing floor BEFORE settling (real firm rule) — so a would-be winner that
    dips through a tight remaining-DD blows the account mid-trade. mae_aware=False = realized-only
    (the optimistic model that ignores mid-trade excursion)."""
    bal = 50000.; peak = 50000.; floor = 48000.; n = len(pool)
    for d in range(max_days):
        g, mae = pool[rng.integers(0, n)]
        if mae_aware and bal + mae <= floor:        # floating dip breaches the floor -> blown mid-trade
            return "blow", d + 1
        bal += g
        if bal <= floor: return "blow", d + 1
        if bal - 50000. >= 3000.: return "pass", d + 1
        if bal > peak: peak = bal
        floor = min(50000., peak - 2000.)
    return "timeout", max_days


N_ACCT = 10   # user's cohort size (decorrelated 1-account-per-signal -> i.i.d. per account)


def main():
    for risk, tp_cap, lbl in [(2000., 1500., "50% consist"), (1500., 1500., "50% consist"),
                              (1200., 1200., "40% consist")]:
        df = build_outcomes(risk, tp_cap)
        wdays = int(np.ceil(3000. / tp_cap))      # winning days needed to reach +$3k
        wr = (df.g > 0).mean()                     # win rate = fraction of trades net-positive
        print(f"\n===== {lbl}: stop ${risk:.0f}=1R, TP cap +${tp_cap:.0f}={tp_cap/risk:.2f}R, "
              f"need {wdays} win days, DD $2k | WR {wr*100:.1f}% =====")
        print(f"{'strat':>5} {'n':>5} {'TP%':>6} {'SL%':>6} {'PART%':>6} {'avg $':>7} {'avg MAE$':>9}")
        for s in ["OD", "RV", "B2", "FB", "ALL"]:
            m = df if s == "ALL" else df[df.strat == s]
            tp = (m.outcome == "TP").mean(); sl = (m.outcome == "SL").mean(); pa = (m.outcome == "PARTIAL").mean()
            print(f"{s:>5} {len(m):>5} {tp*100:>5.0f}% {sl*100:>5.0f}% {pa*100:>5.0f}% "
                  f"{m.g.mean():>+7.0f} {m.mae.mean():>+9.0f}")
        pool = df[["g", "mae"]].values
        for label, aware in (("MAE-aware (real firm rule)", True), ("realized-only (optimistic)", False)):
            rng = np.random.default_rng(7)         # same seed both ways -> apples-to-apples
            res = [sim(pool, rng, mae_aware=aware) for _ in range(50000)]
            outs = np.array([x[0] for x in res]); dys = np.array([x[1] for x in res])
            p = (outs == "pass").mean()
            md = int(np.median(dys[outs == "pass"])) if (outs == "pass").any() else 0
            # how many of N_ACCT pass, and P(>=1) — cohort view
            exp_pass = N_ACCT * p
            print(f"  {label:28s}: PASS {p*100:5.1f}%  BLOW {np.mean(outs=='blow')*100:5.1f}%  "
                  f"med days {md:>3}  => of {N_ACCT}: {exp_pass:.1f} expected pass")


if __name__ == "__main__":
    main()
