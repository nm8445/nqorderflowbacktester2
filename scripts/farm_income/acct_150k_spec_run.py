"""150k farm — the user's exact spec (2026-07-25), FULLY MAE-AWARE end to end.

  EVAL     risk $3,600/trade, 1:1 fixed ATR, reach +$9,000 before the floor. PASS RATE IS MEASURED.
  BUILD    funded: same $3,600 risk, gamble to +$7,200 (= 2x risk).
  HARVEST  take 5 winning days at milk size S; withdraw 50% x profit x 0.90 (firm cap applied);
           repeat -> DD shrinks every time -> keep going until the account blows.
  BATCH    buy 40 evals (~$10k). Two rebuy policies: wait-for-all-dead vs rolling.

Floor: trails to -$4,500, locks at $0 (start balance) once +$4,500 banked, and after every
withdrawal the remaining profit IS the remaining room.
MAE-aware everywhere: a floating dip through the floor blows the account mid-trade.

Run: python scripts/farm_income/acct_150k_spec_run.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from scripts.futurespropmc.challenge_gamble_pass import build_outcomes  # noqa: E402
from farm_income_mc import load_daily_1mnq_mae  # noqa: E402

DD = 4_500.0
EVAL_TARGET = 9_000.0
BUILD_TARGET = 7_200.0
RISK = 3_600.0
SPLIT, WIN, WINS = 0.90, 250.0, 5
N_ACCT, BATCH_FEE, SIG, TDY = 40, 10_000.0, 2.17, 251
CACHE = Path(__file__).parent / "_pool_3600_mae.npy"


def pool_at(risk):
    if CACHE.exists():
        base = np.load(CACHE)
    else:
        base = build_outcomes(3600.0, 3600.0)[["g", "mae"]].values
        np.save(CACHE, base)
    return base * (risk / 3600.0)


def gamble(pool, target, rng, start=0.0, peak0=None, locked0=False, maxtr=300):
    """MAE-aware 1:1 gamble to target. Returns (profit|None, trades)."""
    p = start; peak = start if peak0 is None else peak0; locked = locked0; n = len(pool)
    for tr in range(1, maxtr + 1):
        g, mae = pool[rng.integers(n)]
        floor = 0.0 if locked else min(0.0, peak - DD)
        if p + mae <= floor + 1e-9:
            return None, tr
        p += g
        peak = max(peak, p)
        locked = locked or peak >= DD
        floor = 0.0 if locked else min(0.0, peak - DD)
        if p <= floor + 1e-9:
            return None, tr
        if p >= target:
            return p, tr
    return None, maxtr


def harvest(p, d1, rng, S, cap, maxwd, maxdays=400):
    """5 winning days -> withdraw 50% (cap) -> repeat until blown. Floor locked at $0.
    Milk size S: int MNQ if >0, else adaptive profit/|S|."""
    cash = 0.0; nwd = 0; days = 0; w = 0; n = len(d1)
    while days < maxdays and nwd < maxwd:
        days += 1
        net, low = d1[rng.integers(n)]
        s = S if S > 0 else int(np.clip(p / (-S), 1, 25))
        if p + s * low <= 1e-9:
            return cash, nwd, days
        p += s * net
        if p <= 1e-9:
            return cash, nwd, days
        if s * net >= WIN:
            w += 1
            if w >= WINS:
                g = min(0.50 * p, cap)
                p -= g; cash += g * SPLIT; nwd += 1; w = 0
    return cash, nwd, days


def one_account(pool, d1, rng, S, cap, maxwd):
    """Returns (cash, eval_trades, build_trades, milk_days, passed_eval, reached_build)."""
    pe, etr = gamble(pool, EVAL_TARGET, rng)
    if pe is None:
        return 0.0, etr, 0, 0, False, False
    pb, btr = gamble(pool, BUILD_TARGET, rng)          # funded starts fresh at 0 profit
    if pb is None:
        return 0.0, etr, btr, 0, True, False
    c, w, md = harvest(pb, d1, rng, S, cap, maxwd)
    return c, etr, btr, md, True, True


def main():
    pool = pool_at(RISK)
    d1 = load_daily_1mnq_mae()

    # ---- 1. MEASURED eval pass rate ----
    rng = np.random.default_rng(7)
    n = 40_000
    ok = 0; tr = np.zeros(n)
    for i in range(n):
        p, t = gamble(pool, EVAL_TARGET, rng); tr[i] = t
        if p is not None:
            ok += 1
    ev_pass, ev_tr = ok / n, tr.mean()
    print(f"MEASURED eval pass @ ${RISK:,.0f} risk, +${EVAL_TARGET:,.0f} target, MAE-aware:"
          f"  {ev_pass:.1%}  ({ev_tr:.1f} trades avg)")

    rng = np.random.default_rng(11)
    ok = 0; tr = np.zeros(n)
    for i in range(n):
        p, t = gamble(pool, BUILD_TARGET, rng); tr[i] = t
        if p is not None:
            ok += 1
    print(f"MEASURED funded build to +${BUILD_TARGET:,.0f}:                       "
          f"  {ok/n:.1%}  ({tr.mean():.1f} trades avg)\n")

    # ---- 2. milk size sweep, per-account economics ----
    for cap, capname in ((5_000.0, "$5k cap"), (3_000.0, "$3k cap")):
        print(f"=== {capname} — per-account (maxWD 5) ===")
        print(f"  {'milk':>10} {'cash/acct':>10} {'#wd':>6} {'milk d':>7} {'trades':>7}")
        for S in (3, 6, 10, -400, -600, -900):
            rng = np.random.default_rng(7); m = 12_000
            c = np.zeros(m); wd = np.zeros(m); md = np.zeros(m); tt = np.zeros(m)
            for i in range(m):
                cc, e, b, d, _, _ = one_account(pool, d1, rng, S, cap, 5)
                c[i] = cc; md[i] = d; tt[i] = e + b
                wd[i] = 0
            lbl = f"{S} MNQ" if S > 0 else f"p/{-S}"
            print(f"  {lbl:>10} {c.mean():>10,.0f} {'':>6} {md.mean():>7.1f} {tt.mean():>7.1f}")
        print()


if __name__ == "__main__":
    main()
