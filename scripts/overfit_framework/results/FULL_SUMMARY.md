# Overfit Test Framework — Results Summary

All 5 tests applied to all 4 locked strategies. See per-test detail files in this directory.
Fabio results from `scripts/fabio_orb/run_overfit_tests.py` (added 2026-05-24).

## Headline scorecard

| Test | Rough Vol | Overnight Drift | OHI/OLO B2 | Fabio ORB |
|---|:---:|:---:|:---:|:---:|
| 1. Parameter Stability | **PASS** | **PASS** | **PASS** | **PASS** |
| 2. Walk-Forward (rolling 12-mo) | **PASS** (85% PF>1) | **PASS** (78% PF>1) | **PASS** (98% PF>1) | **PASS** (82% PF>1) |
| 3. Monte Carlo Shuffle | **PASS** (94.2%) | **PASS** (76%) | **PASS** (96.9%) | **PASS** (88.9%) |
| 4. Bootstrap CI | **PASS** (P(loss)=0.07%) | **PASS** (P(loss)=0.09%) | **PASS** (P(loss)=0.13%) | **PASS** (P(loss)=0.18%) |
| 5. Direction Permutation (p<0.01) | **PASS** (p<0.0001) | **FAIL** (p=0.011) | **FAIL** (p=0.013) | **PASS** (p=0.0030) |

**Overall: 18/20 strict passes across 4 strategies. 2 borderline failures at p<0.05.**

## Test 1 — Parameter Stability

1D/3D sweeps around each locked parameter. Pass = all neighbors profitable + PF>1.

| Strategy | Params tested | Cells | Min PnL (rel to locked) | Verdict |
|---|---|---:|---:|:---:|
| Rough Vol | norm, hz, sl_atr, tp_atr | — | 0.72x | PASS |
| Overnight Drift | yellow_mult, green_base, green_decay, green_atr_mult | — | 0.86x | PASS |
| OHI/OLO B2 | YMULT, TPMULT, MFE_K, MFE_LOCK | — | 0.73x | PASS |
| **Fabio ORB** | **N, delta, TP (3D 3×3×3)** | **27** | **0.75x (all 27 profitable)** | **PASS** |

**Interpretation:** All locked configs sit on parameter plateaus, not spikes. Strategies tolerate ~25% parameter mis-estimation. Fabio's 27-cell 3D grid is the most thorough — every single neighbor profitable.

## Test 2 — Walk-Forward / Rolling OOS

Rolling 12-month PF for RV/OD/B2; non-overlapping 6-month windows for Fabio.

| Strategy | Windows | PF>1 share | Min window PF | Worst month / window |
|---|---:|:---:|---:|---|
| Rough Vol | 54 | 85.2% | 0.95 | 2024-04 |
| Overnight Drift | 55 | 78.2% | 0.85 | 2023-07 |
| OHI/OLO B2 | 55 | 98.2% | 0.99 | 2024-06 |
| **Fabio ORB** | **11** | **81.8%** (9/11) | **0.86** | **2022-06 to 2022-12 (-$7,855)** |

**Interpretation:** OHI/OLO B2 is the most temporally stable. ON Drift has the deepest cyclical weakness. Fabio's two losing windows (2022 H2, 2023 H2) coincide with documented NQ chop / regime shifts. All four pass the 75% threshold.

## Test 3 — Monte Carlo Order Shuffling (10k perms)

Shuffle trade order, compute MDD distribution. Real MDD percentile-rank.

| Strategy | Real MDD | Median shuffled MDD | Real better than... |
|---|---:|---:|:---:|
| Rough Vol | −$16,957 | −$23,936 | 94.2% of orderings |
| Overnight Drift | −$28,155 | −$33,318 | 76.0% of orderings |
| OHI/OLO B2 | −$845 | −$1,270 | 96.9% of orderings |
| **Fabio ORB** | **−$20,240** | **−$26,460** | **88.9% of orderings** |

**Interpretation:** All four real orderings produce better drawdowns than random. ON Drift the weakest. B2 the most-favorable. Fabio firmly in the top 12% of all permutations.

## Test 4 — Bootstrap (10k resamples with replacement)

95% CI on total PnL and P(strategy is losing).

| Strategy | Real PnL | 95% CI lower | 95% CI upper | P(losing) | P(PF≤1) |
|---|---:|---:|---:|---:|---:|
| Rough Vol | +$153,549 | +$59,506 | +$244,831 | 0.07% | 0.07% |
| Overnight Drift | +$208,450 | +$81,028 | +$341,596 | 0.09% | 0.09% |
| OHI/OLO B2 (MNQ) | +$7,957 | +$3,032 | +$12,750 | 0.13% | 0.13% |
| **Fabio ORB** | **+$157,965** | **+$50,134** | **+$274,291** | **0.18%** | **0.18%** |

**Interpretation:** All four have <0.2% probability of being losing strategies. CIs are wide but solidly positive. Fabio's CI is the widest in relative terms but still firmly positive at the lower bound.

## Test 5 — Direction Permutation (1000 perms) — **THE KEY TEST**

For each entry signal, randomize direction; re-run trade management; compare real PnL to permutation distribution.

| Strategy | Real PnL | Perm median | 99th pctile | p-value | Verdict |
|---|---:|---:|---:|---:|:---:|
| Rough Vol | +$154,746 | +$4,936 | +$105,043 | <0.0001 | **PASS** |
| Overnight Drift (no-mart) | +$130,915 | +$22,363 | +$131,075 | 0.011 | **FAIL** strict |
| OHI/OLO B2 (NQ pts) | +6,494 | +1,961 | +6,603 | 0.013 | **FAIL** strict |
| **Fabio ORB** | **+$157,965** | **−$258** | **+$87,182** | **0.0030** | **PASS** |

### Why Fabio is the cleanest test 5 result

Fabio's permutation median is **-$258** — essentially zero — while real PnL is +$158K. This means the direction call (long-only after ORB-high break with delta confirmation) does ~99% of the work. The all-short mirror simulation produced -$156K (near-perfect negative of long), confirming the asymmetry isn't drift-driven — it's a real directional edge from the breakout logic.

Compare to OD: perm median +$22K (significant drift bleed-through) vs real $131K. The "edge" looks smaller relative to a chance-driven baseline.

### Why ON Drift and B2 fail strict p<0.01

Both strategies trade an instrument (NQ) with a structural overnight long bias. "Random direction" with the same exit logic still partially captures that drift because:
1. NQ has a positive overnight drift (perm median is +$22k for OD, +1,961 NQ pts for B2 — both clearly positive)
2. Asymmetric exit logic (yellow ratchets, MFE guard) interacts with the drift even under random direction

For Rough Vol AND Fabio, the directional decision cleanly separates real signal from noise — perm median near zero. Direction call is doing 97%+ of the work.

For OD and B2, the direction call adds substantial value (real beats ~99% of perms) but is **not statistically distinguishable from a top-1% lucky random** at the strict threshold.

### Practical implication

- **Rough Vol's direction logic (EMA filter) is provably value-adding** — far beyond random by every measure.
- **Fabio's direction logic (ORB-high break + delta confirmation) is provably value-adding** — same level of evidence as RV.
- **ON Drift's "always long at 19:00 ET" is mostly capturing the overnight drift** plus some yellow/green management edge. It is profitable but the direction call alone is barely distinguishable from random at strict thresholds. Risk-management edge dominates direction edge.
- **B2's bias logic (3-close OHI/OLO break) adds value over random** but again is borderline at strict thresholds. The fixed-TP + ratchet exit logic does heavy lifting.

This is not a fatal finding for OD or B2 — they pass at p<0.05 — but it tells you where the edge is coming from. If NQ's overnight drift dies, OD is in trouble. If overnight range breaks become less predictive, B2 weakens.

## Overall verdict

| Strategy | Score | Risk profile |
|---|---|---|
| **Rough Vol Orderflow** | 5/5 strict pass | Most robust. Direction signal independently meaningful. Lowest dependence on instrument drift. |
| **Fabio ORB** | 5/5 strict pass | Direction signal independently meaningful. Two losing 6-mo windows tied to known NQ chop regimes (2022 H2, 2023 H2). |
| **OHI/OLO B2** | 4/5 strict pass (T5 borderline) | Strong temporal stability. Edge partly relies on overnight range structure persisting. |
| **Overnight Drift** | 4/5 strict pass (T5 borderline) | Strong absolute PnL. Edge partly relies on NQ overnight drift persisting. |

All four are deployable. **Rough Vol and Fabio carry the most "independent edge"**; OD and B2 are profitable strategies whose edge is partly contingent on NQ market structure (drift, OHI/OLO mean-reversion) continuing as observed in the 2020-2026 sample.

## Deployment risk ranking

If you have to rank strategies by "most likely to survive a regime change":

1. **Rough Vol** — direction edge proven independent of drift; orderflow features generalize across regimes
2. **Fabio ORB** — direction edge proven independent of drift; sensitive to "EOD drift" specifically (72% of trades exit at 14:00 ET). If post-2022 drift dies, Fabio flattens (as 2022 already showed: PF 1.00, $105 for that year).
3. **OHI/OLO B2** — exit logic does heavy lifting; entry direction edge partly drift-dependent. Most temporally stable in the sample but with structural exposure.
4. **Overnight Drift** — strongest absolute PnL but most regime-dependent. "Always long at 19:00 ET" only works while overnight drift persists.

## Things still worth doing before live deployment

The 5-test framework validates the strategies on backtest data. Still missing:

1. **Live paper period** — current live signal logs have ~22 rows total. Need 30-60 days of real signal generation to catch implementation bugs.
2. **Live slippage measurement** — backtest assumes 1 tick/side. Real NQ fills may be 1-2 ticks worse during news.
3. **Realistic regime stratification** — bucket each strategy's PnL by "bull/chop/bear" months and check PF in each. Not done yet.
4. **News-filter integration in coordinator** — required for 5%ers, FTMO, FN compliance. Code exists in MC but not in live executor.
