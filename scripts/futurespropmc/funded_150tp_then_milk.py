"""50k FUNDED: single-shot TP150/SL100 gamble to +$3,000, then milk 5 winning days -> withdraw.

Idea (user, 2026-07-30): the highest-P(reach $3k) structure at 1 NQ is a ONE-TRADE gamble --
TP 150pts (=$3,000) vs SL 100pts (=$2,000 = the whole room). 44.2% measured first-passage.
Beats the farm 1:1 $1,500 config, which reaches +$3k only 34.0% of the time, because a single
coin flip isn't squared by a second survival requirement.

Then: de-risk, grind 5 winning days (>= $150 realised), withdraw 50% x 0.90, floor locks at
start balance so remaining room = remaining profit. Repeat to MAX_WD payouts or death.

Compares $/funded against the calculator's structure A/B presets ($974 / $1,019, cap 2).
Run: python scripts/futurespropmc/funded_150tp_then_milk.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "farm_income"))
from farm_income_mc import load_daily_1mnq_mae  # noqa: E402

DD = 2_000.0
P_GAMBLE = 0.442          # measured P(+150pts before -100pts), 5-day horizon
TP_D, SL_D = 3_000.0, 2_000.0
SPLIT, WIN_DAY, WINS = 0.90, 150.0, 5
N_SIMS = 40_000


def milk(p, d1, rng, size, need=WINS, maxdays=120):
    """Grind `need` winning days at `size` MNQ. Floor locked at 0 => room == profit."""
    w = 0; days = 0; n = len(d1)
    while days < maxdays:
        days += 1
        net, low = d1[rng.integers(n)]
        if p + size * low <= 1e-9:
            return None, days
        p += size * net
        if p <= 1e-9:
            return None, days
        if size * net >= WIN_DAY:
            w += 1
            if w >= need:
                return p, days
    return None, days


def run(d1, size, max_wd, cap, n=N_SIMS):
    rng = np.random.default_rng(7)
    cash = np.zeros(n); nwd = np.zeros(n); dys = np.zeros(n); reach = 0
    for i in range(n):
        if rng.random() >= P_GAMBLE:          # the single-shot gamble
            dys[i] = 1; continue
        reach += 1
        prof = TP_D; d = 1; c = 0.0; k = 0
        while k < max_wd:
            r = milk(prof, d1, rng, size)
            if r[0] is None:
                d += r[1]; break
            prof, md = r; d += md
            g = min(0.50 * prof, cap)
            prof -= g; c += g * SPLIT; k += 1
        cash[i] = c; nwd[i] = k; dys[i] = d
    return cash.mean(), nwd.mean(), dys.mean(), reach / n


def main():
    d1 = load_daily_1mnq_mae()
    print(f"50k FUNDED — single-shot TP150/SL100 gamble (P={P_GAMBLE:.1%}) then milk-to-payout")
    print(f"  vs farm 1:1 $1,500 gamble, which reaches +$3k at 34.0%\n")
    for cap, cname in ((1_500.0, "$1.5k cap"), (2_000.0, "$2k cap")):
        print(f"=== withdrawal cap {cname} ===")
        print(f"  {'milk MNQ':>9} {'payouts':>8} {'$/funded':>10} {'days':>7} {'$/paying':>10}")
        for size in (1, 2, 3, 4):
            for mw in (2,):
                c, w, dd, rt = run(d1, size, mw, cap)
                print(f"  {size:>9} {w:>8.2f} ${c:>9,.0f} {dd:>7.1f} ${c/rt if rt else 0:>9,.0f}")
        print()
    print("  calculator presets for comparison: A $974 / B $1,019 per funded (cap 2 payouts)")


if __name__ == "__main__":
    main()
