# Gamble → De-risk → Milk Playbook (50k futures, 4-way combined)

Two-phase prop strategy: **pass challenges with a high-risk gamble, then on the funded account gamble
once to build a buffer, de-risk to 1 MNQ, and milk for payouts.** De-correlation comes from putting a
*different* signal on each account (firing-order) — copy-trading the same trade is what creates the
all-or-nothing risk this design avoids.

All numbers from the 4-way combined (OD/RV/B2/FB) historical log, **RV filtered at ATR>150** (removes
only the 2 crash-vol outliers — see note at bottom), ATR computed UTC-consistently. Sized so the
**initial yellow distance = the $ risk** (a touch of yellow = that loss, via the firm's floating DD).

---

## 1. CHALLENGE configs (pass fast; the consistency rule sets the per-day cap & # winning days)

50k eval, $2k DD, +$3k target. Stop = yellow (hard, on touch). One signal/account/day, firing-order.
Consistency rule sets the daily profit cap (= rule% × $3k) and thus the winning days needed to reach $3k.

| Firm rule | Stop / day-cap | RR | Win days | TP% / SL% | **Pass/acct** | Days | Of 10 |
|---|---|---|---|---|---|---|---|
| 50% consist | $2,000 / $1,500 | 0.75 | 2 | 48 / 32 | 35.4% | 2 | 3.5 |
| **50% consist** | **$1,500 / $1,500** | 1.0 | 2 | 39 / 35 | **45.2%** | 4 | 4.5 |
| **40% consist** | **$1,200 / $1,200** | 1.0 | **3** | 39 / 35 | **38.9%** | 6 | 3.9 |

**50%-rule firm, $1,500 risk = best (45%)** — a −$1,500 loss doesn't blow a $2k-DD account, so you
survive one stop and get more swings. **40%-rule firm = ~39%**: same 1:1 bracket, but the tighter rule
caps days at $1,200 so you need **3 winning days** not 2 (2×$1,200=$2,400<$3k), which costs ~6 pts and a
couple days. Per-trade hit-rates are identical for any 1:1 config — the $ scale doesn't change first-touch;
the # of required winning days is what moves the pass rate. To fund 10: ~22 challenges (50%) / ~26 (40%),
$100 each. **Tighter consistency = more winning days needed = lower pass rate.**

---

## 2. FUNDED phase — gamble once, de-risk, milk (NO re-gambling after de-risk)

- **Gamble:** one trade at a time, sized to the $2k yellow (a touch = blow). De-risk to 1 MNQ at **ANY
  profit**. Small-loss survivors keep gambling; profit-survivors de-risk.
- **61% reach the buffer & de-risk** (of 10 funded, ~6). ~4 blow gambling (cheap — fresh accounts).
- After de-risk you **only milk 1 MNQ** — the gamble is one-time.

### Gamble win size = the buffer (varies hugely by strat, $2k=1R)
| Strat | Win rate | Avg win | Median | p90 | % of wins ≥ $3k |
|---|---|---|---|---|---|
| **OD** | 44% | **+$3,354** | +$2,609 | +$7,192 | **42%** |
| **RV** | 56% | +$1,871 | +$2,216 | +$2,480 | 0% |
| B2 | 61% | +$1,422 | +$1,602 | +$3,094 | 10% |
| FB | 54% | +$1,567 | +$1,201 | +$3,273 | 11% |

OD is the big-buffer leg (often de-risks *past* +$3k → instant payout); RV is the steady ~$2k; B2/FB
~$1.4–1.6k. Firing-order naturally tilts gambles toward OD (it fires most), which helps.

### Payout rules (funded)
- Eligible after **5 winning days (each ≥ $150)**. At 1 MNQ a day averages +$31 and clears $150 only
  **29%** of days → **~17 days to the first payout** (losing days mixed in).
- **Withdraw 50% of profit, capped at $2k; keep 80%** (= 0.4×profit take-home); leave the other 50% + DD.
  - +$3k profit → withdraw $1.5k → keep **$1,200**, leave +$1.5k cushion.
  - +$4k profit → withdraw $2k (cap) → keep **$1,600**, leave +$2k (DD locked at $2k static above $50k).
- **Max 4 payouts/account**, then retire (graduates to a live/$2k account per firm). Extract aggressively
  since accounts are assumed to eventually blow.
- Milk is **copy-traded together** (all milkers run the same 4-way at 1 MNQ, same days → correlated;
  the gamble is the only de-correlated part).

### Per-account lifecycle (with these rules)
- First payout: **~day 17**, **~$953 take-home** (median). Then a payout every **~15 days**, ~$900 each.
- **~2.94 payouts / de-risked account**; **~$2,601 take-home / de-risked account**; **~$1,590 / funded
  account** (including the ~39% that blow in the gamble). 96% of de-risked accounts get ≥1 payout.

---

## 3. ECONOMICS — capped spend (keep the profit) vs aggressive reinvest

### Capped cohort: $3k → 30 evals → milk, NO reinvestment (the chosen posture)
~14 funded → ~8 reach buffer & pay → ~24 payouts.
- **GROSS take-home ~$21k** (median $19.8k); **NET ~$18k/cohort** (median $16.8k, p25 $9.3k, p10 ~$4k,
  **P(loss) 3%**). Plays out in ~3 months.
- **Annual:** one-time $3k → ~$18k/yr; **re-buy $3k each retirement (~3.5×/yr, ~$10.5k eval/yr) → ~$64k/yr net.**

### Aggressive reinvest (for comparison — NOT chosen)
Plow payouts back, grow to a 30-account cap: **~$143k+/yr** (optimistic runs hit $200k+, but those lean
on fast funding turnaround + no payout friction). Higher eval spend (~$34k/yr), higher variance from the
correlated milking, more capital tied up.

| | Capped ($3k, keep profit) | Aggressive reinvest |
|---|---|---|
| Annual net | **~$64k** | ~$143k+ |
| Eval spend/yr | ~$10.5k | ~$34k+ |
| Risk | low (3% losing cohort) | higher (correlated-milk tail) |

---

## 4. Caveats (real, apply to all numbers above)
- **Gamble is "soft":** only ~25–40% blow because OD's wide stop rarely floats to yellow — and that
  rests on the **1-min MAE model holding live**. At ~18-MNQ gamble size, real slippage blows more.
- **Milking is correlated** (copy-traded) → a genuinely bad 4-way day dents all milkers; the de-risk
  staggering softens but doesn't remove it.
- **Funding lag & payout friction** (firm processing time, payout holding periods, minimums) were modeled
  optimistically in the aggressive case; the capped cohort is far less sensitive.
- A real market crash correlates **all four** strats, so the gamble isn't perfectly independent.

## 5. RV ATR-150 filter (live in rv_engine.py)
RV skips entries with 20-min ATR(14) > 150 — removes only the 2 genuine crash-vol outliers (2025-04-07
ATR 280 / float $1,622, + 2025-04-04), cuts RV worst float $1,622 → $738/MNQ, costs ~−$445. (An earlier
ATR_MAX=70 was a tz-bug artifact that wrongly dropped 52 profitable trades — corrected.)

## Scripts
`challenge_gamble_pass.py` (challenge configs) · `funded_payout_dynamics.py` · `account_lifecycle.py`
(per-account payout speed/size) · `farm_firing_order.py` / `farm_full_flow.py` (aggressive farm) ·
`cohort_harvest.py` (capped $3k cohort) · `two_phase_gamble_milk.py` · `_risknorm_trades.csv` (per-strat
risk-normalized outcomes).
