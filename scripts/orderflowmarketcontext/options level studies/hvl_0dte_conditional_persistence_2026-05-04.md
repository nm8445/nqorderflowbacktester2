# HVL 0DTE Conditional Regime Persistence (2026-05-04)

## Question

Once you observe the gamma regime at 9:30 AM, how confident can you be it'll
hold through the close? And does that confidence build as the day progresses
without flipping?

This study answers both — the conditional persistence "the longer it survives,
the more likely it stays" question for live trading.

## Methodology

Re-using the intraday regime parquet from `study_hvl_intraday_regime.py`,
extended to record the regime tag at hourly checkpoints (10:00, 11:00, 12:00,
13:00, 14:00, 15:00 ET) plus open and close.

For each opening regime (pos-gamma, neg-gamma):
- At each checkpoint T, filter to days that have stayed in the opening regime
  through every checkpoint up to and including T
- Compute P(regime_close == regime_open | survived through T)

This gives a "confidence builds with each surviving checkpoint" curve.

## Sample

- **Days**: 1,210 with valid open/close NQ prices and complete intraday data
- **Date range**: 2020-12-02 → 2025-11-26
- **HVL source timing**: prior EOD (no lookahead)
- **NQ price source**: `markettick_1min_bars.parquet` (5-min bars used for
  evaluation)

## Distribution of opening regime

| Source | Open neg-gamma | Open pos-gamma |
|---|---|---|
| **QQQ** | 928 days (80%) | 239 days (20%) |
| **NDX** | 780 days (65%) | 424 days (35%) |

Negative-gamma is the default opening regime, especially in QQQ (retail
put-buying for hedging dominates).

## QQQ — opening in pos-gamma

| Time | Days still in pos | P(close in pos | here) |
|---|---|---|
| 09:30 | 100% (n=239) | **80.3%** |
| 10:00 | 91.2% (n=218) | 85.3% |
| 11:00 | 83.3% (n=199) | **91.0%** |
| 12:00 | 79.9% (n=191) | 92.7% |
| 13:00 | 78.7% (n=188) | 93.6% |
| 14:00 | 76.6% (n=183) | 95.1% |
| 15:00 | 73.2% (n=175) | 97.1% |

## QQQ — opening in neg-gamma

| Time | Days still in neg | P(close in neg | here) |
|---|---|---|
| 09:30 | 100% (n=928) | 93.3% |
| 10:00 | 97.2% (n=902) | 94.9% |
| 11:00 | 94.7% (n=879) | 96.7% |
| 12:00 | 93.4% (n=867) | 97.8% |
| 13:00 | 92.8% (n=861) | 98.4% |
| 14:00 | 92.1% (n=855) | 98.5% |
| 15:00 | 91.7% (n=851) | 98.8% |

## NDX — opening in pos-gamma

| Time | Days still in pos | P(close in pos | here) |
|---|---|---|
| 09:30 | 100% (n=424) | 78.8% |
| 10:00 | 89.9% (n=381) | 81.9% |
| 11:00 | 81.6% (n=346) | **87.3%** |
| 12:00 | 77.1% (n=327) | 91.4% |
| 13:00 | 75.5% (n=320) | 93.1% |
| 14:00 | 73.1% (n=310) | 95.2% |
| 15:00 | 70.3% (n=298) | 97.3% |

## NDX — opening in neg-gamma

| Time | Days still in neg | P(close in neg | here) |
|---|---|---|
| 09:30 | 100% (n=780) | 89.0% |
| 10:00 | 95.1% (n=742) | 92.2% |
| 11:00 | 91.5% (n=714) | **95.1%** |
| 12:00 | 88.6% (n=691) | 97.1% |
| 13:00 | 87.8% (n=685) | 97.4% |
| 14:00 | 87.2% (n=680) | 97.8% |
| 15:00 | 86.2% (n=672) | 98.7% |

## Key findings

### 1. Most regime flips happen in the first 60-90 minutes

For QQQ opening in pos-gamma:
- **8.8% flip by 10:00 ET** (first 30 min)
- Another **7.9% flip by 11:00 ET** (next hour)
- Survival rate flattens after 11:00 — late-day flips are rare

For QQQ opening in neg-gamma: only 2.8% flip by 10:00 — much stickier from
the open.

### 2. The 11:00 ET checkpoint is the sweet spot

If at 11:00 ET the opening regime hasn't flipped:

| | Confidence at 9:30 | Confidence at 11:00 | Boost |
|---|---|---|---|
| QQQ pos-gamma | 80.3% | **91.0%** | +10.7 pp |
| QQQ neg-gamma | 93.3% | 96.7% | +3.4 pp |
| NDX pos-gamma | 78.8% | **87.3%** | +8.5 pp |
| NDX neg-gamma | 89.0% | 95.1% | +6.1 pp |

The pos-gamma confidence boost from waiting until 11:00 is the most
material — for QQQ it goes from 80% to 91%.

### 3. Negative-gamma regime is much more stable than positive-gamma

- **QQQ neg-gamma at the open: already 93% confident it persists** — no need
  to wait. Trade plan can commit immediately.
- **QQQ pos-gamma at the open: 80% confident** — waiting an hour gets you
  to 91%. Worth the wait if precision matters.

Same asymmetry holds in NDX (89% vs 79% at the open).

## Live-trading playbook

| Setup | Recommended entry time | Confidence | Justification |
|---|---|---|---|
| Sell premium / pin mode (open pos-gamma) | **11:00 ET** | ~91% (QQQ) / ~87% (NDX) | Skip the morning shakeout |
| Trade vol expansion (open neg-gamma) | **9:30 ET** | 93% (QQQ) / 89% (NDX) | Already high-confidence at open |
| Highest-confidence pin entry | 12:00-13:00 ET | ~93% | Trade-off vs less time to capture move |
| Stop-out trigger | Cross HVL → regime flipped | — | Exit pin trades immediately if regime breaks |

The cleanest practical rule: **negative-gamma at the open is a "go now" signal;
positive-gamma at the open is a "wait until 11:00 to confirm" signal.**

## Caveats

- HVL 0DTE comes from prior EOD chain (no lookahead). Same-day intraday
  re-derivation could shift the boundary slightly. Effect on regime label is
  small in practice.
- The "pos-gamma → 80% confidence" baseline is conditional on having a clear
  HVL flip in the chain. Days where the entire ±5% near-spot band is one-sided
  (deep_pos / deep_neg in the static-label study) aren't separated here —
  they're rolled into the binary pos/neg classification at each bar.
- 5-min bar granularity. A finer grid (1-min) would catch transient flips that
  reverse quickly, but the structural conclusions wouldn't change.

## Output data

`D:/trading_pythonbacktest_data/QQQ_thetadata/study_hvl0dte_intraday_regime.parquet`
(extended with `regime_at_HH` checkpoint columns for 10/11/12/13/14/15 ET)

## Scripts

- `../scripts/study_hvl_intraday_regime.py` — builds the per-day intraday
  regime parquet (now includes hourly checkpoint regime tags)
- `../scripts/analyze_hvl_conditional_persistence.py` — computes the
  conditional-persistence tables shown above
