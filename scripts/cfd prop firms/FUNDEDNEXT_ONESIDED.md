# FundedNext — One-Sided-Bet Compliant Regime (4-way combined)

FundedNext adds two rules FundingPips doesn't: a **One-Sided-Betting** clause (concentrated/
same-direction exposure → may be forced onto a 1% risk rule or refunded+banned) and a **3% RPTI**
(Risk Per Trade Idea = $3k on 100k / $6k on 200k floating-loss kill; funded only, eval-exempt).
The 3% RPTI is more generous than FundingPips' 2%.

## The compliant regime (makes both rules un-trippable by construction)

1. **No hedging + ONE position open at a time.** When a position is live, all other signals are
   ignored until it closes (overlapping RTH signals fan out to *other accounts* in the copier farm).
   OD is overnight (19:00→08:00) so it never overlaps the RTH legs. This satisfies the one-sided/
   no-hedge rule automatically — you can't hold two positions, let alone opposing ones.
2. **Each strat sized so its WORST historical MAE ≤ the budget** (≤$2,800 to stay under the $3k RPTI).
   Because only one position is open at a time, account-level float never exceeds one position's MAE.
   → **RPTI can never trip** (0% of blows are RPTI in sim). The 5% daily also never trips.

Per-strat whole-MNQ sizing at the $2,800 budget (100k), worst MAE per 1 MNQ: OD $556 · RV $1,622 ·
B2 $615 · FB $621:

| Strat | Size | Worst MAE |
|---|---|---|
| OD | 5 MNQ | $2,782 |
| **RV** | **1 MNQ** | $1,622 |
| B2 | 4 MNQ | $2,462 |
| FB | 4 MNQ | $2,486 |

RV is the only leg its own fat tail throttles (2 MNQ = $3,244 breaches). OD scales the most (tightest MAE).

### RV max-ATR filter (ATR_MAX=70) — RV is NOT actually the fat-tail leg (live as of 2026-06)
The $1,622 "RV worst float" was ONE trade: 2025-04-07 tariff crash, when 20-min ATR(14) hit 280 pts
(10x normal ~28) so the 2xATR stop was a legit 561 pts; it filled at the stop (−$11,228) and the
1-min bar wicked to 811 pts MAE. RV's 2nd-worst is only $738, 99th-pctile $460, median **$91**/MNQ.
Fix (now in `live/combined/rv_engine.py`, `ATR_MAX=70`): **skip RV entries where 20-min ATR(14) > 70**.
RV by fine ATR bucket: 40-50 +$3,631(61%WR), 50-60 +$1,257(64%), 60-70 +$154(50%), **70+ −$980(33%)**.
So the filter drops only ~3 trades/5yr (the net-negative 70+ tail, incl the crash) and slightly RAISES
RV net. KEY: **RV worst float stays ~$738/MNQ at ANY cap 50–100** (it's a sub-50-ATR whipsaw, not high
vol) — the ATR filter trims the losing tail, it does NOT bound float (use the ~$700/MNQ hard stop for
that). Cap 70 chosen over 60 because the 60-70 band is still net-positive (+$154) and worst float is
unchanged. Filter is **RV-only** (OD/FB's big trades are wins — leave them). EV tables below were run
at cap 60; the +$154/5yr difference at cap 70 is immaterial to the $/yr figures.

**RV can now size up — but only where the account absorbs its tail:**
| Sim | RV@1 | RV@2 | RV@3 |
|---|---|---|---|
| **FundedNext 100k funded** (OD5/B2 4/FB 4) | $13,320 @40% | $14,269 @40% | **$15,205 @41%** |
| 50k futures eval | **69% / 59d** | 60% / 38d | 53% / 26d |
| 50k milking farm (median net/yr) | **$42,959** | $35,141 | $901 |

→ **Funded 100k: RV@3 (+20% EV, blow flat).  50k eval: RV@1 (max pass) or RV@2 (speed).  50k thin-
cushion milking: RV@1** (sizing up tanks the median — bigger RV just buys variance vs the $2k floor).
Principle: **size RV up only where the buffer can absorb its $738/MNQ tail.** (Scripts: rv_atr60_rerun.py)

## Funded EV — 100k vs 200k (one-sided compliant, all blows now via the 10% MaxLoss)

| Account | Budget (2.8%) | Sizes OD/RV/B2/FB | E[$/yr net] | Blow %/yr |
|---|---|---|---|---|
| 100k | $2,800 | 5/1/4/4 | $12,618 | 42.2% |
| **200k** | $5,600 | 10/3/9/9 | **$26,861** | 48.1% |

**Scale-invariant: 200k ≈ 2.1× the dollars at ~the same (slightly higher) blow — NOT lower blow.**
The bigger $20k MaxLoss buffer is offset by 2× position size. The 200k is a touch *more* aggressive
(higher EV + blow) because whole-contract rounding wastes less budget (100k RV stuck at 1 MNQ uses
only $1,622 of $2,800; 200k RV at 3 MNQ uses $4,866 of $5,600). RPTI = 0% of blows on both.

### Budget knob (100k) — trade EV for survival
| MAE budget | Sizes OD/RV/B2/FB | E[$/yr] | Blow % |
|---|---|---|---|
| $2,800 | 5/1/4/4 | $12,618 | 42.2% |
| $2,000 | 3/1/3/3 | $9,346 | 13.7% |
| $1,500 | 2/1/2/2 | $6,642 | 2.1% |
| $1,000 | 1/1/1/1 | $3,925 | 0.0% |

Full budget maximizes aggregate $ (good for a copier farm — a blow is just a re-buy after extraction).
For a single account you want to keep, **$2,000 (≈14% blow, ~$9.3k)** is the best risk-adjusted spot.
On a 200k, scale the budget ×2 (e.g. ~$4,000 to mirror the 100k's 14%-blow point).

## Challenge — 2-step (+8% then +5%), 5% daily / 10% max, RPTI-EXEMPT (size up)

Pass time/rate are **scale-invariant** — governed by risk *as a % of account*, not dollar size.
At matched %-risk a 200k passes in the same time as a 100k; a 200k only takes longer if under-sized.

### Flat MNQ (same size all 4 strats), one-at-a-time
**100k:**
| MNQ | Pass % | Median days | p90 |
|---|---|---|---|
| 2 | 75.0% | 291 | 476 |
| **3** | **80.9%** | 193 | 380 |
| **4** | 71.6% | **124** | 262 |
| 5 | 66.6% | 88 | 190 |
| 6 | 62.1% | 66 | 138 |

**200k needs exactly 2× the MNQ** (perfect scale match): 200k @6 ≡ 100k @3 (80.9%/193d); 200k @8 ≡
100k @4 (71.6%/124d); 200k @10 ≡ 100k @5 (66.6%/88d).

**Pick:** 100k → **3 MNQ = max pass (81%, slow ~193d)**, **4 MNQ = balanced (72%, 124d)**, 5 MNQ =
fast (67%, 88d). Below 3 MNQ you're too small → time out (1 MNQ only 32%). Pass% peaks at 3 MNQ and
falls above because RV's worst float ($6,490 at 4 MNQ) starts breaching the 5% daily ($5k) / 10% max
($10k). **Crank the 200k to 4–5% (8–10 MNQ) during the eval for speed, then drop to the 2.8% MAE
budget the moment it's funded** (RPTI is eval-exempt; funded is not).

## Crash safety (MT5/CFD)
Every MT5 order is sent with a **broker-side GTC stop-loss + take-profit** (`mt5_executor.py` line
286/291), held on FundedNext's server. A full PC / power / internet failure does NOT remove them —
the position closes at its stop regardless. Keep the SL distance ≤ the MAE budget so even an offline
position can't breach the RPTI. (NT8/futures instead uses a 30s heartbeat watchdog that flattens on
*Python* death, but a full-PC crash kills the watchdog too → run on a VPS for futures.)

## Scripts
- `fundednext_onesided.py` — funded EV + budget knob (100k)
- `fundednext_size_compare.py` — 100k vs 200k funded EV + challenge scale-invariance
- `fundednext_flat_mnq.py` — flat-MNQ challenge sweep 1–10 for 100k & 200k
