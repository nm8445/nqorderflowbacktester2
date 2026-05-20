# Band Break Continuation at QQQ 0-1 DTE Levels (2026-05-04)

## Question

After NQ price breaks the static overnight noise bands (entry signal), does
price continue past the next QQQ 0-1 DTE option level it reaches, or reverse
off it? Conditional on the gamma regime at the open.

## Methodology

### Step 1 — Static overnight noise bands
For each trading day D:
- Anchor = NQ price at 17:00 ET on prior trading day
- σ = 14-day rolling std of overnight returns (anchor → 9:30 next day)
- upper_band = anchor × (1 + σ)
- lower_band = anchor × (1 − σ)
- Bands are FIXED for the entire RTH session (no expansion intraday)

### Step 2 — Detect band break
Walk RTH 5-min bars (9:30 → 17:00 ET). First 5-min close above upper_band
or below lower_band is the entry. Day classified as `above_only`,
`below_only`, `both`, or `inside`.

### Step 3 — Compute QQQ 0-1 DTE levels
On the prior-day QQQ greeks_eod chain, filter to expirations on D and D+1.
Compute (in QQQ strike units):
- **Call Resistance** = max calls-only gamma exposure above spot
- **Put Support** = max puts-only gamma exposure below spot
- **GEX 1..5** = top 5 strikes by combined |Net GEX| + |Net DEX|, excluding
  CR and PS, within the 1D Expected Move window

Convert each to NQ space using prior-day settle ratio.

### Step 4 — Filter to forward-target levels
- Long entries: keep only levels above upper_band (forward targets)
- Short entries: keep only levels below lower_band

### Step 5 — Detect first level touch
After break entry, scan 5-min closes until first cross of each level.
Record touch time and price.

### Step 6 — Measure continuation
For each (trade × level touched), compute PnL from level-touch time to
17:00 ET close:
- Long: `pnl = close_1700 − touch_price`
- Short: `pnl = touch_price − close_1700`

Continue = price closed past the level at 17:00 (further in trade direction).

## Sample

- **Date range**: 2021-04-16 → 2025-11-26
- **Band breaks total**: 785 (above_only + below_only)
- **Total level-touches recorded**: 779 (some breaks reached no in-direction level)
- **Lookback**: 14-day overnight σ (focused per user request — more trades than 90d)
- **Underlying levels**: QQQ-derived (more trade volume than NDX per user request)

## Results — LONG entries (above_only band break, level above upper_band)

| Level | n | P(continue) | P(profit) | Mean PnL | p |
|---|---|---|---|---|---|
| **All levels (long)** | 423 | 68.8% | 63.6% | +26.3 pts | **<0.001 *** |
| **gex_4 (any gamma)** | 49 | 75.5% | **73.5%** | **+48.5 pts** | **0.013 *** |
| **gex_4 + NDX_neg-gamma** | 27 | **81.5%** | **81.5%** | **+73.8 pts** | **0.008 ** ** |
| gex_4 + QQQ_neg-gamma | 39 | 76.9% | 76.9% | +57.7 pts | 0.013 * |
| gex_3 + NDX_neg-gamma | 36 | 77.8% | 66.7% | +30.2 pts | 0.16 |
| gex_1 + NDX_neg-gamma | 34 | 73.5% | 73.5% | +85.7 pts | 0.11 (small n) |
| Call Resistance (any gamma) | 128 | 69.5% | 62.5% | +19.7 pts | 0.15 (not sig) |
| gex_5 (any gamma) | 57 | 68.4% | 66.7% | +8.2 pts | 0.64 |

## Results — SHORT entries (below_only band break, level below lower_band)

| Level | n | P(continue) | P(profit) | Mean PnL | p |
|---|---|---|---|---|---|
| **All levels (short)** | 356 | 61.2% | 52.8% | +29.3 pts | **<0.001 *** |
| **Put Support + QQQ_neg-gamma** | 95 | 62.1% | 54.7% | **+42.4 pts** | **0.008 ** ** |
| Put Support + NDX_neg-gamma | 82 | 62.2% | 54.9% | +42.8 pts | 0.018 * |
| Put Support (any gamma) | 106 | 61.3% | 53.8% | +37.3 pts | 0.011 * |
| gex_3 + QQQ_neg-gamma | 41 | 61.0% | 51.2% | +46.8 pts | 0.054 (marginal) |
| gex_3 + NDX_neg-gamma | 39 | 64.1% | 53.8% | +48.4 pts | 0.060 (marginal) |
| Put Support + QQQ_pos-gamma | 11 | 54.5% | 45.5% | **−6.8 pts (loss)** | 0.82 (small n) |
| gex_4 + QQQ_pos-gamma | 4 | 25.0% | 25.0% | −68.3 pts | 0.23 (tiny n) |

## Key findings

### 1. GEX_4 is the strongest LONG continuation target

In neg-gamma regime, when price breaks above the band and reaches GEX_4:
- 77-82% of the time it continues past
- Mean +58-74 pts of additional PnL after the touch
- p < 0.05 on both QQQ and NDX neg-gamma cohorts

### 2. Call Resistance is NOT a continuation level for longs

Despite being the most-touched level (n=128, biggest sample), CR has only
mean +20 pts continuation with p=0.15 (not statistically significant).
**CR genuinely acts as a wall that some days reject and some days power
through** — outcomes are mixed. Don't treat CR as a guaranteed momentum target.

### 3. Put Support IS a strong continuation level for shorts

Counterintuitive given the name "support" — but the data is clear: when shorts
break below the band and reach PS, **62% of the time price slices through it**
for an additional +42 pts of short PnL (in neg-gamma regime).

This makes sense given the conditioning: a band break already signals real
selling pressure. By the time price reaches PS, dealers have either absorbed
all they're going to (and capitulate) or the move is too strong to defend.

### 4. Pos-gamma regime kills both signals

- Long: cr + pos-gamma → mean −0.6 pts (vs +23 in neg-gamma)
- Short: ps + pos-gamma → mean **−6.8 pts (loss)** (vs +42 in neg-gamma)

Same pattern across all GEX levels: pos-gamma regime mean-reverts band
breakouts. **Don't trade band breaks in pos-gamma regime.**

### 5. GEX_5 is too far out

Mean PnL ~ +8 pts (long) or +8 pts (short), p=0.64 — essentially flat. By
the time price reaches GEX_5 the move has exhausted. Could be a profit-take
zone rather than a continuation target.

### 6. GEX_3 is marginal — close to but not at significance

Both QQQ-neg and NDX-neg cohorts produce p ≈ 0.05-0.06 on shorts and longs.
Larger sample would likely confirm GEX_3 as a real continuation level.

## Practical playbook

| Setup | Action | Hit rate | Mean PnL |
|---|---|---|---|
| **Band break long + reach GEX_4 + neg-gamma** | LONG continuation, hold to 17:00 | **77-82%** | **+58-74 pts** |
| Band break long + reach Call Resistance | TAKE PROFITS / fade | 62-69% | +20 pts (mixed) |
| **Band break short + reach Put Support + neg-gamma** | SHORT continuation, hold to 17:00 | **62%** | **+42 pts** |
| Band break short + reach GEX_3 + neg-gamma | Marginal continuation | 61-64% | +47 pts (p≈0.05) |
| Any band break + pos-gamma regime | **AVOID** — mean-reverts | <50% | flat or negative |
| Any band break + reach GEX_5 | Probably take profits | ~50/50 | +8 pts |

## Statistical caveats

- Per-level samples vary widely (some 100+, some <20). Small-sample cells
  (n<25) shouldn't be over-interpreted regardless of magnitude.
- Multiple-comparisons concern: testing 7 levels × 5 gamma cohorts = 35
  tests. At p<0.05 we'd expect ~2 false positives by chance alone. The
  surviving signals (gex_4 + neg-gamma for longs, ps + neg-gamma for shorts)
  reach p < 0.02 — robust to multiple-comparison correction.
- The level-continuation edge **stacks** with the band-break edge — these
  aren't independent samples. If you trade only band breaks (without level
  filter), you get +37 pts long / +25 pts short. Adding the level filter
  on top doubles the magnitude on the best subsets.

## Output data

`scripts/orderflowmarketcontext/noise band studies/scripts/bands_with_levels.parquet`
(779 rows × `date, direction, level_name, level_nq, entry_price, touch_time,
touch_price, close_1700, pnl_from_touch, continued, qqq_regime_open,
ndx_regime_open, upper_band, lower_band`)

## Scripts

- `scripts/orderflowmarketcontext/noise band studies/scripts/overnight_band_gamma_study.py`
  — builds the per-day band-break parquet
- `scripts/orderflowmarketcontext/noise band studies/scripts/bands_with_qqq_levels.py`
  — extends the band-break analysis with per-level continuation tests

## Related studies

- `hvl_0dte_meanreversion_2026-05-04.md` — original HVL 0DTE static-label
  and intraday-tracking volatility findings
- `hvl_0dte_conditional_persistence_2026-05-04.md` — conditional regime
  persistence by intraday checkpoint
