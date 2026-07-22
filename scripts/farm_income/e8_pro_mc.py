"""E8 Pro (CFD, 1-step) MC for 400k / 500k: challenge pass%/speed + funded extraction, by sizing.

E8 Pro rules (e8markets help center, 2026):
  - 1-STEP challenge, +8% profit target (closed profit).
  - 2.5% DAILY drawdown, 8% STATIC drawdown — both EOD / BALANCE-based (NOT intraday equity), so NO
    floating/MAE kill (unlike FundingPips RPTI). Static is fixed from initial; on the CHALLENGE it never
    moves. On FUNDED it moves UP to the initial-balance level after the first payout (then floor = start).
  - 2% DAILY PROFIT CAP on the CHALLENGE only (max 2% counts toward the 8% target / day -> >=4 green days).
    Funded has NO daily cap.
  - Daily payouts, withdraw 50% of profit (min 1%), trader keeps up to 100% (we use 90%).
  - Trades NAS100 sized in MNQ-equivalent (1 NQ = 10 MNQ). P&L bootstrapped from the 4-way combined
    daily (combined_4way_with_mae_1min.csv, 1-MNQ basis). Daily-resampled (iid) -> understates streaks.

ASSUMPTIONS to CONFIRM (swing the EV): fees ($1,998 @500k, ~$1,600 @400k — estimate); 90% split;
challenge daily-cap excess is wiped (counted = balance); static DD locks at start on the 1st payout.
Run: python scripts/farm_income/e8_pro_mc.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from farm_income_mc import load_daily_1mnq

ACCTS = {"400k": 400_000.0, "500k": 500_000.0}
FEES = {"400k": 1_600.0, "500k": 1_998.0}     # ESTIMATE — confirm E8 Pro pricing
TARGET_PCT, DAILY_DD_PCT, STATIC_DD_PCT, DAILY_CAP_PCT = 0.08, 0.025, 0.08, 0.02
SPLIT = 0.90
TD_YR = 252
N = 40_000
# sizings in MNQ-equivalent; 10/20 = 1/2 NQ ("NQ contracts")
SIZES = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]


def challenge(d, start, m, rng, maxdays=400):
    """Vectorized 1-step pass. Returns (pass mask, days array)."""
    n = d.size
    dd_daily = DAILY_DD_PCT * start; floor = -STATIC_DD_PCT * start
    target = TARGET_PCT * start; cap = DAILY_CAP_PCT * start
    bal = np.zeros(N); done = np.zeros(N, bool); passed = np.zeros(N, bool); days = np.full(N, maxdays)
    for day in range(maxdays):
        pnl = d[rng.integers(0, n, N)] * m
        act = ~done
        db = act & (pnl < -dd_daily)               # daily DD breach
        done |= db; days[db] = day + 1
        act = ~done
        credited = np.where(pnl > 0, np.minimum(pnl, cap), pnl)   # 2% daily cap on counted profit
        bal = np.where(act, bal + credited, bal)
        sb = act & (bal <= floor)                  # static DD breach
        done |= sb; days[sb] = day + 1
        act = ~done
        pa = act & (bal >= target)
        done |= pa; days[pa] = day + 1; passed[pa] = True
        if done.all():
            break
    return passed, days


def funded(d, start, m, rng, wd_x=0.08, days=TD_YR):
    """Vectorized funded year. CORRECTED mechanics (user-confirmed): no min trading-day / profit gate,
    and the static floor moves from start-8% UP to start ONLY when you first WITHDRAW. So the lever is
    `wd_x` = grind to +wd_x% on the full -8% cushion, then withdraw 50% (keeping wd_x/2 buffer) which
    locks the floor at start; thereafter milk and withdraw 50% each time profit recovers to +wd_x.
    90% split. Daily DD 2.5% throughout. Returns (extracted$ array, blew mask)."""
    n = d.size
    dd_daily = DAILY_DD_PCT * start; wd_at = wd_x * start
    bal = np.zeros(N); floor = np.full(N, -STATIC_DD_PCT * start)
    locked = np.zeros(N, bool); dead = np.zeros(N, bool); extr = np.zeros(N)
    for _ in range(days):
        pnl = d[rng.integers(0, n, N)] * m
        act = ~dead
        db = act & (pnl < -dd_daily)                 # daily DD breach (single day < -2.5%)
        dead |= db
        act = ~dead
        bal = np.where(act, bal + pnl, bal)
        sb = act & (bal <= floor)                    # static DD breach (-8% pre-lock, start post-lock)
        dead |= sb
        act = ~dead
        pay = act & (bal >= wd_at)                   # withdraw 50% at +wd_x
        w = np.where(pay, 0.5 * bal, 0.0)
        extr += w * SPLIT
        bal = bal - w
        nl = pay & ~locked                           # FIRST withdrawal locks the floor at start
        floor = np.where(nl, 0.0, floor); locked |= nl
    return extr, dead


def main():
    d = load_daily_1mnq()
    worst = d.min()
    print(f"4-way daily 1-MNQ: n={d.size} mean=${d.mean():.0f} worst day=${worst:.0f} "
          f"p95=${np.percentile(d,95):.0f}\n")
    print("E8 Pro: 1-step +8% | 2.5% daily DD | 8% static DD | 2% daily cap (challenge) | "
          "EOD/balance-based (no float kill)\n")
    for name, start in ACCTS.items():
        fee = FEES[name]
        m_safe = (DAILY_DD_PCT * start) / abs(worst)    # size where the worst historical day == daily DD
        print("=" * 92)
        print(f"  E8 Pro {name}  (start ${start:,.0f} | target ${TARGET_PCT*start:,.0f} | "
              f"daily DD ${DAILY_DD_PCT*start:,.0f} | static DD ${STATIC_DD_PCT*start:,.0f} | fee ~${fee:,.0f})")
        print(f"  single-bad-day-safe size <= {m_safe:.1f} MNQ (worst hist day ${worst:.0f}/MNQ)")
        print("=" * 92)
        print(f"  CHALLENGE (size for speed) + FUNDED @ wd_x=8% (withdraw 50% at +8%):")
        print(f"  {'MNQ':>4} {'~NQ':>4} {'pass%':>6} {'avg dys':>8} {'med dys':>8} "
              f"{'$/funded/yr':>12} {'blow%':>6} {'EV/eval':>9}")
        for m in SIZES:
            rng = np.random.default_rng(7)
            passed, cdays = challenge(d, start, m, rng)
            pp = passed.mean()
            adays = cdays[passed].mean() if passed.any() else float("nan")
            mdays = np.median(cdays[passed]) if passed.any() else float("nan")
            rng2 = np.random.default_rng(11)
            extr, blew = funded(d, start, m, rng2)
            ef = extr.mean(); ev = pp * ef - fee
            print(f"  {m:>4} {m/10:>4.1f} {pp*100:>5.0f}% {adays:>8.0f} {mdays:>8.0f} "
                  f"{ef:>12,.0f} {blew.mean()*100:>5.0f}% {ev:>9,.0f}")
        # First-withdrawal sweep at a durable funded size: how high to grind before the 1st withdrawal
        m_dur = 6
        print(f"\n  FUNDED first-withdrawal sweep @ {m_dur} MNQ (grind to +wd_x on the -8% cushion, "
              f"then withdraw 50% -> floor locks at start):")
        print(f"  {'wd_x':>5} {'$/funded/yr':>12} {'blow%':>6}")
        best = (None, -1e18)
        for wd_x in (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20):
            rng3 = np.random.default_rng(11)
            extr, blew = funded(d, start, m_dur, rng3, wd_x=wd_x)
            ef = extr.mean()
            if ef > best[1]:
                best = (wd_x, ef)
            print(f"  {wd_x*100:>4.0f}% {ef:>12,.0f} {blew.mean()*100:>5.0f}%"
                  f"{'  <- max $' if wd_x == best[0] else ''}")
        print(f"  -> best first-withdrawal at +{best[0]*100:.0f}% -> ${best[1]:,.0f}/funded\n")


if __name__ == "__main__":
    main()
