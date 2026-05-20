# Overfit Test Framework — Results Summary

All 5 tests applied to all 3 locked strategies. See per-test detail files in this directory.

## Headline scorecard

| Test | Rough Vol | Overnight Drift | OHI/OLO B2 |
|---|:---:|:---:|:---:|
| 1. Parameter Stability | **PASS** | **PASS** | **PASS** |
| 2. Walk-Forward (rolling 12-mo) | **PASS** (85% PF>1) | **PASS** (78% PF>1) | **PASS** (98% PF>1) |
| 3. Monte Carlo Shuffle | **PASS** (94.2%) | **PASS** (76%) | **PASS** (96.9%) |
| 4. Bootstrap CI | **PASS** (P(loss)=0.07%) | **PASS** (P(loss)=0.09%) | **PASS** (P(loss)=0.13%) |
| 5. Direction Permutation (p<0.01) | **PASS** (p<0.0001) | **FAIL** (p=0.011) | **FAIL** (p=0.013) |

## Test 1 — Parameter Stability

1D sweeps of ±2 steps around each locked parameter. Pass = all 5 neighbors profitable + PF>1.

| Strategy | Params tested | Min PnL (rel to locked) | Verdict |
|---|---|---:|:---:|
| Rough Vol | norm, hz, sl_atr, tp_atr | 0.72x | PASS |
| Overnight Drift | yellow_mult, green_base, green_decay, green_atr_mult | 0.86x | PASS |
| OHI/OLO B2 | YMULT, TPMULT, MFE_K, MFE_LOCK | 0.73x | PASS |

**Interpretation:** All locked configs sit on parameter plateaus, not spikes. Strategies tolerate ~25% parameter mis-estimation.

## Test 2 — Walk-Forward / Rolling OOS

Rolling 12-calendar-month PF on the locked config (no refit; tests temporal stability).

| Strategy | Windows | PF>1 share | Min window PF | Worst month |
|---|---:|:---:|---:|---|
| Rough Vol | 54 | 85.2% | 0.95 | 2024-04 |
| Overnight Drift | 55 | 78.2% | 0.85 | 2023-07 |
| OHI/OLO B2 | 55 | 98.2% | 0.99 | 2024-06 |

**Interpretation:** OHI/OLO B2 is the most temporally stable. ON Drift has the deepest cyclical weakness (worst window PF 0.85). All three pass the 75% threshold.

## Test 3 — Monte Carlo Order Shuffling (10k perms)

Shuffle trade order, compute MDD distribution. Real MDD percentile-rank.

| Strategy | Real MDD | Median shuffled MDD | Real better than... |
|---|---:|---:|:---:|
| Rough Vol | −$16,957 | −$23,936 | 94.2% of orderings |
| Overnight Drift | −$28,155 | −$33,318 | 76.0% of orderings |
| OHI/OLO B2 | −$845 | −$1,270 | 96.9% of orderings |

**Interpretation:** Real trade ordering produces better drawdowns than random in all 3 cases. ON Drift the weakest — its DD is structurally near the median, suggesting its sequence isn't unusually fortunate. B2 has the most-favorable real ordering (close to top 3% of all permutations).

## Test 4 — Bootstrap (10k resamples with replacement)

95% CI on total PnL and P(strategy is losing).

| Strategy | Real PnL | 95% CI lower | 95% CI upper | P(losing) | P(PF≤1) |
|---|---:|---:|---:|---:|---:|
| Rough Vol | +$153,549 | +$59,506 | +$244,831 | 0.07% | 0.07% |
| Overnight Drift | +$208,450 | +$81,028 | +$341,596 | 0.09% | 0.09% |
| OHI/OLO B2 (MNQ) | +$7,957 | +$3,032 | +$12,750 | 0.13% | 0.13% |

**Interpretation:** All three have <0.2% probability of being losing strategies. CIs are wide but solidly positive.

## Test 5 — Direction Permutation (1000 perms) — **THE KEY TEST**

For each entry signal, randomize direction; re-run trade management; compare real PnL to permutation distribution.

| Strategy | Real PnL | Perm median | 99th pctile | p-value | Verdict |
|---|---:|---:|---:|---:|:---:|
| Rough Vol | +$154,746 | +$4,936 | +$105,043 | <0.0001 | **PASS** |
| Overnight Drift (no-mart) | +$130,915 | +$22,363 | +$131,075 | 0.011 | **FAIL** strict |
| OHI/OLO B2 (NQ pts) | +6,494 | +1,961 | +6,603 | 0.013 | **FAIL** strict |

### Why ON Drift and B2 fail strict p<0.01

Both strategies trade an instrument (NQ) with a structural overnight long bias. "Random direction" with the same exit logic still partially captures that drift because:
1. NQ has a positive overnight drift (perm median is +$22k for OD, +1,961 NQ pts for B2 — both clearly positive)
2. Asymmetric exit logic (yellow ratchets, MFE guard) interacts with the drift even under random direction

For Rough Vol, the EMA filter cleanly separates real signal from noise — perm median is near zero (+$4.9k on a $154k strategy = 3% of edge). Direction call is doing 97%+ of the work.

For OD and B2, the direction call adds substantial value (real beats ~99% of perms) but is **not statistically distinguishable from a top-1% lucky random** at the strict threshold.

### Practical implication

- **Rough Vol's direction logic (EMA filter) is provably value-adding** — far beyond random by every measure.
- **ON Drift's "always long at 19:00 ET" is mostly capturing the overnight drift** plus some yellow/green management edge. It is profitable but the direction call alone is barely distinguishable from random at strict thresholds. Risk-management edge dominates direction edge.
- **B2's bias logic (3-close OHI/OLO break) adds value over random** but again is borderline at strict thresholds. The fixed-TP + ratchet exit logic does heavy lifting.

This is not a fatal finding for OD or B2 — they pass at p<0.05 — but it tells you where the edge is coming from. If NQ's overnight drift dies, OD is in trouble. If overnight range breaks become less predictive, B2 weakens.

## Overall verdict

| Strategy | Score | Risk profile |
|---|---|---|
| **Rough Vol Orderflow** | 5/5 strict pass | Most robust. Direction signal independently meaningful. Lowest dependence on instrument drift. |
| **OHI/OLO B2** | 4/5 strict pass (T5 borderline) | Strong temporal stability. Edge partly relies on overnight range structure persisting. |
| **Overnight Drift** | 4/5 strict pass (T5 borderline) | Strong absolute PnL. Edge partly relies on NQ overnight drift persisting. |

All three are deployable. Rough Vol carries the most "independent edge"; the other two are profitable strategies whose edge is partly contingent on NQ market structure (drift, OHI/OLO mean-reversion) continuing as observed in the 2020-2026 sample.
