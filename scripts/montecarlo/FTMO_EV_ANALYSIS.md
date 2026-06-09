# FTMO 100K / 200K EV — Sizing & Blow-Rate Analysis

Strategy: 4-way combined (OD overnight + FB/RV/B2 intraday), **one trade at a time** for FB/RV/B2
(OD overnight is separate). FTMO rules: **static** max-loss floor (never trails), floating-aware
daily + max DD, **biweekly 80% split**, **no RPTI / floating-profit kill** (unlike FundingPips/FundedNext).

| Account | Daily DD | Max DD (static floor) | Budget/trade |
|---|---|---|---|
| 100K | $5,000 | $10,000 (floor $90,000) | $1,000 |
| 200K | $10,000 | $20,000 (floor $180,000) | $2,000 |

> **200K = exactly 2× 100K.** Identical risk geometry (10R static floor, 5R daily, payouts as a share
> of R); only the dollar scale differs. Everything below is shown for 200K — halve for 100K.

Scripts: `scripts/montecarlo/ftmo_ev_onetrade.py` (stop-based sizing), `ftmo_mae_sized.py` (MAE-based
sizing — the recommended scheme). Data: `_risknorm_trades.csv` joined to `combined_4way_with_mae_1min.csv`
(1,594 historical daily packs), 20k–80k sims, 252-day horizon.

---

## 1. Stop-based sizing ($2,000 risk on the stop) — too hot

Sizing each strat so its **stop** risks $2,000 gives big size (OD 21 MNQ, RV 13, B2 10, FB 9) and a
**97% chance of blowing within a year** — *not* because the DD is small, but because the static floor
never trails and stripping all profit each payout means you never build a cushion. You still bank
~3 payouts (~$38k mean take-home/yr on 200K) before it goes. Leaving a working cushion helps:

| Profit left in account | Blow % | EV take-home (200K) |
|---|---|---|
| $0 (strip to $200k) | 97% | $38,211 |
| $10k | 85% | $44,388 |
| $20k | 74% | $44,127 |

**Verdict:** stop-based $2k is a milk-then-blow profile. Not the goal.

---

## 2. MAE-based sizing (cap the *float*, not the stop) — RECOMMENDED

Size each strat so its historical **MAE (worst adverse excursion)** only floats ~$2,000 against you.
Then a single trade can't threaten the account, and (being one-at-a-time) **no single day can reach
the $10k daily limit**. Per-strat contracts = `budget / (MAE_percentile_pts × $2/pt)`.

### The frontier (200K, $2,000 MAE-budget, $10k working cushion)

| Sizing | MAE ≤ $2k for… | Blow %/yr | Daily-limit blows | EV take-home | Bottom-10% |
|---|---|---|---|---|---|
| p95 | 95% of trades | 44% | 27% of blows | $44,109 | $0 |
| p97.5 | 97.5% | 18% | **0%** | $37,741 | +$1,135 |
| p98.5 | 98.5% | 10% | **0%** | $31,195 | +$3,998 |
| **p99 (recommended)** | 99% | **6%** | **0%** | $26,377 | **+$4,153** |
| max | every trade ever | 0.4% | 0% | $9,918 | — |

**Key threshold at ~p97.5:** below it, a bad cluster in one day can still stack to the $10k daily
limit; at/above it each trade caps ~$2k float and you're one-at-a-time, so **daily-limit blows go to
zero**. Every remaining blow is slow multi-week drift into the $20k static floor.

A bigger cushion ($20k) does **not** help — it trims blow a point or two but cuts EV ~30% (idle
capital, slower milk). **$10k working cushion is optimal.**

### Recommended config — 200K, p99 sizing, $10k cushion

| Strat | Contracts | (vs stop-based $2k) |
|---|---|---|
| OD | ~7 MNQ | 21 MNQ |
| B2 | ~5 MNQ | 10 MNQ |
| FB | ~4 MNQ | 9 MNQ |
| RV | ~4 MNQ | 13 MNQ |

**100K = exactly half**: $1,000 budget, $5k cushion → 6% blow, ~$13k/yr; sizes ≈ OD 3.5 / B2 2.3 /
FB 2.2 / RV 2.2 MNQ.

---

## 3. Blow rate at the recommended config (80k sims)

| Config | Blow within 1 yr | from $10k **daily** | from $20k **overall** | EV take-home |
|---|---|---|---|---|
| **p99 (safest)** | **6.0%** | **0.0%** | 6.0% | $26,468 |
| p98.5 (more income) | 10.0% | 0.0% | 10.0% | $31,280 |

**~6% chance of blowing within a year, virtually all of it the $20k overall floor, ~0% the $10k
daily** — the daily limit is effectively unreachable at this size (0 daily blows in 80,000 account-years).
Median runway for the rare blow is ~108 trading days; median ~8 payouts/yr.

---

## Bottom line

Size off **MAE, not the stop**. At **p99 / $10k cushion** the $20k floor is plenty: **~6% annual blow,
~$26k/yr take-home on a 200K (~$13k on a 100K), zero daily-limit risk, and even a bottom-decile year
still profits +$4k.** Step up to **p98** for ~$31–34k at ~10% blow if you want more income for modestly
more risk. All figures are gross of the challenge fee (~$580 / 100K, ~$1,160 / 200K).
