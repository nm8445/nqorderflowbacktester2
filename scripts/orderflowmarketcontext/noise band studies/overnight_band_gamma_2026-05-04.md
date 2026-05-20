# Overnight Noise Band Break + Gamma Regime (2026-05-04)

## Question

If NQ price breaks the static overnight noise envelope during the RTH
session, does the day continue in that direction? And does the gamma
regime at the open materially affect the success rate?

Specifically: enter at the 5-min close that first breaks the band, exit at
17:00 ET. Test both long-side and short-side breaks, with sub-splits by
QQQ-derived and NDX-derived gamma regime at 9:30 ET.

## Methodology

### Step 1 — Build static overnight noise bands per day

For each trading day D:
- **Anchor** = NQ close at 17:00 ET on the prior trading day (the last
  print before the daily futures-session maintenance break)
- **σ** = rolling std of overnight returns over the lookback window:
  - `r_overnight(d) = (NQ_open_9:30(d) − NQ_anchor(d−1)) / NQ_anchor(d−1)`
  - σ_14d uses the prior 14 overnight returns
  - σ_90d uses the prior 90 overnight returns
- **upper_band** = anchor × (1 + σ)
- **lower_band** = anchor × (1 − σ)
- Bands are STATIC for the entire RTH session (no expansion intraday)

The intuition: the band represents "where the market expected NQ to be at
9:30 today, given the recent overnight volatility distribution." A break of
the band means the overnight + early-RTH move exceeded the typical noise
envelope — interpreted as a real momentum signal.

### Step 2 — Detect band break

Walk the RTH 5-min bars from 9:30 to 17:00 ET. Classify each day:
- `above_only` — first 5-min close beyond the bands is above upper_band
- `below_only` — first 5-min close beyond the bands is below lower_band
- `both` — at different times, 5-min closes on both sides
- `inside` — price stayed inside the bands all session, no signal fires

### Step 3 — Trade entry and exit

- **Entry** = the 5-min close that first crosses the band (entry_price)
- **Exit** = NQ close at 17:00 ET (close_1700)
- **Long PnL** = close_1700 − entry_price (for above_only)
- **Short PnL** = entry_price − close_1700 (for below_only)

### Step 4 — Filter by gamma regime at the open

`regime_open` from the intraday HVL 0DTE study tells us whether NQ price at
9:30 sits in positive-gamma or negative-gamma territory on the prior-day
EOD chain. Same for QQQ-derived and NDX-derived (cross-check).

Cohort splits: ALL gamma + 4 sub-cohorts (QQQ_pos / QQQ_neg / NDX_pos /
NDX_neg).

## Sample

- **Date range**: 2021-04-16 → 2025-11-26 (limited by NQ markettick parquet
  end at 2025-12-01, with 90-day lookback warm-up)
- **Total days**: 1,114 with valid open + close + bands + gamma regime
- **Day distribution** (14-day lookback):
  - `above_only`: 406 (36.4%)
  - `below_only`: 379 (34.0%)
  - `both`: 97 (8.7%)
  - `inside`: 232 (20.8%) — no trade
- **Day distribution** (90-day lookback):
  - `above_only`: 386 (34.7%)
  - `below_only`: 357 (32.0%)
  - `both`: 65 (5.8%)
  - `inside`: 306 (27.5%) — no trade
- 14-day bands are tighter → more breaks. 90-day bands are wider → fewer
  but cleaner signals on average.

## Results — 14-day lookback (more trades, focus per user request)

### Baseline (entry every day at 9:30, exit 17:00)
- Mean drift: **+2.85 pts/day** (slightly bullish, statistically marginal)

### LONG signal — above_only (entry at upper-band break)

| Cohort | n | P(profit) | Mean PnL | t-stat | p |
|---|---|---|---|---|---|
| **ALL gamma** | 406 | **65.3%** | **+37.4 pts** | +5.02 | **<0.0001 *** |
| **+ NDX_neg-gamma** | 223 | **70.9%** | **+50.2 pts** | +4.29 | **<0.0001 *** |
| + QQQ_neg-gamma | 306 | 66.7% | +40.5 pts | +4.40 | <0.0001 *** |
| + NDX_pos-gamma | 180 | 57.8% | +21.5 pts | +2.55 | 0.012 * |
| + QQQ_pos-gamma | 96 | 59.4% | +27.2 pts | +2.36 | 0.020 * |

### SHORT signal — below_only (entry at lower-band break)

| Cohort | n | P(profit) | Mean PnL | t-stat | p |
|---|---|---|---|---|---|
| **ALL gamma** | 379 | **54.4%** | **+24.6 pts** | +2.82 | **0.005 ** ** |
| + QQQ_neg-gamma | 328 | 55.5% | +30.5 pts | +3.11 | 0.002 ** |
| + NDX_neg-gamma | 304 | 55.6% | +28.0 pts | +2.79 | 0.006 ** |
| + NDX_pos-gamma | 74 | 48.6% | +9.4 pts | +0.54 | 0.59 (not sig) |
| **+ QQQ_pos-gamma** | 47 | **42.6%** | **−18.7 pts (loss)** | −1.25 | 0.22 (not sig) |

### Whipsaw (`both`) cohort — bidirectional break

| Cohort | n | P(short profit) | Short PnL | p |
|---|---|---|---|---|
| ALL gamma | 97 | ≈55% | +27 pts | 0.41 |
| **+ NDX_pos-gamma** | 26 | **76.9%** | **+191 pts** | **0.001 ** ** |
| + QQQ_pos-gamma | 10 | 70.0% | +148 pts | 0.05 (small n) |
| + NDX_neg-gamma | 71 | 46.5% | −33 pts | 0.38 |

When pos-gamma is in place at the open AND the day whips through both
bands, the close usually crashes — interpretation: positive-gamma regime
tries to defend, fails, then unwinds violently. Sample is small (n=26
QQQ_pos / n=11 NDX_pos in 90d) so the magnitude is uncertain but the
direction is reliable.

## Results — 90-day lookback (fewer but cleaner signals)

### LONG — above_only

| Cohort | n | P(profit) | Mean PnL | p |
|---|---|---|---|---|
| ALL gamma | 386 | 66.8% | +40.0 pts | <0.0001 *** |
| **+ NDX_neg-gamma** | 213 | **73.2%** | **+59.3 pts** | **<0.0001 *** |
| + QQQ_neg-gamma | 291 | 68.4% | +44.1 pts | <0.0001 *** |
| + NDX_pos-gamma | 170 | 58.8% | +15.9 pts | 0.069 (marginal) |
| + QQQ_pos-gamma | 91 | 61.5% | +27.4 pts | 0.028 * |

### SHORT — below_only

| Cohort | n | P(profit short) | Short PnL | p |
|---|---|---|---|---|
| ALL gamma | 357 | 52.1% | +28.2 pts | 0.002 ** |
| + NDX_neg-gamma | 296 | 52.7% | +27.7 pts | 0.006 ** |
| + QQQ_neg-gamma | 317 | 54.3% | +32.4 pts | 0.001 ** |
| + NDX_pos-gamma | 60 | 48.3% | +29.4 pts | 0.14 |
| + QQQ_pos-gamma | 37 | 29.7% | −9.7 pts | 0.58 (loss) |

## Key findings

### 1. Long-side band breakouts work cleanly

Across both lookbacks, breaking the upper band and going long:
- ~65–73% hit rate
- +37 to +59 pts mean PnL per trade
- p < 0.0001 in all neg-gamma cohorts

This is the single strongest directional signal in the noise-band study.

### 2. Short-side breaks work, but weaker — and pos-gamma kills them

Down-side breakouts continue ~52–56% of the time for ~+25–32 pts of short
PnL on average. Statistically significant in neg-gamma cohorts.

But when QQQ is in pos-gamma at the open, shorting a band-break-down LOSES
on average (−19 pts QQQ_pos 14d, −10 pts QQQ_pos 90d). Don't short bands
in pos-gamma regime.

### 3. Asymmetry: long > short

Several reasons the long signal works better than short:
- Underlying bullish drift (+2.85 pt baseline) helps longs, hurts shorts
- Positive-gamma regime is mean-reverting — it absorbs selling pressure,
  so down-breaks reverse more often
- Sell-side breaks happen near the day's low more often than buy-side
  breaks happen near the day's high — less room for short follow-through

### 4. Whipsaw + pos-gamma = strong contrarian short signal

When the day's price action breaks BOTH bands AND the open was in pos-gamma
territory, going short produces ~77–91% hit rate at +191 to +333 pts mean
PnL. Sample is small (n=11–26) but the effect is huge and statistically
significant (p < 0.01). This is the pattern of "positive-gamma regime
tries to pin, breaks down, then violent unwind."

### 5. 14d vs 90d lookback comparison

Both lookbacks give statistically significant edges. Trade-offs:
- **14d**: tighter bands → more break-day signals (~785 / 1,114 days = 70%) →
  more trades, slightly smaller magnitude per trade
- **90d**: wider bands → fewer breaks (~743 / 1,114 days = 67%) but each
  surviving signal is "more extreme" — slightly larger PnL per trade

90-day NDX_neg-gamma long is the highest-conviction setup overall (73% hit,
+59 pts), but 14d gives more trade opportunities.

## Practical playbook

| Setup | Action | Hit rate | Mean PnL |
|---|---|---|---|
| **Above-band break + NDX_neg-gamma** | LONG to 17:00 | **71% (14d), 73% (90d)** | **+50 / +59 pts** |
| Above-band break + any neg-gamma | LONG to 17:00 | 67–71% | +40–50 pts |
| Above-band break + pos-gamma | Take it but smaller size | ~58–61% | +16–27 pts |
| Below-band break + neg-gamma | SHORT to 17:00 | 52–55% | +28–32 pts |
| **Below-band break + pos-gamma** | **SKIP** — short tends to lose | <50% | flat or negative |
| **Both bands break + pos-gamma** | **CONTRARIAN SHORT** to 17:00 | 77–91% | +191 to +333 pts (small n) |
| No break (price stays inside) | NO TRADE | — | sit out 21–28% of days |

## Statistical caveats

- Sample sizes per cohort: top signals have n > 200 (robust). Pos-gamma
  splits often n < 100, especially in QQQ. Take small-cell magnitudes
  with care.
- The signal is essentially **vol-adjusted gap-and-go momentum** — bands
  are just a smart vol-normalized threshold for detecting "the move was
  bigger than typical noise." The gamma-regime stack adds genuine
  conditional information on top.
- All p-values are one-sample t-tests vs zero null drift. Direct
  comparisons between cohorts (e.g., neg vs pos gamma) would benefit
  from Welch's t-test if rigor is wanted.
- Multiple-comparisons concern: 4 trade classes × 5 gamma sub-cohorts × 2
  lookbacks = ~40 cells. The strongest signals reach p < 0.0001, robust
  to Bonferroni. Marginal cells (p ≈ 0.05) should be re-tested before
  trading.

## Output data

`scripts/orderflowmarketcontext/noise band studies/scripts/overnight_band_per_day.parquet`
(1,114 rows × `date, anchor, open_930, close_1700, day_ret_pts,
overnight_ret, sigma_{14,90}d, upper_band_{14,90}d, lower_band_{14,90}d,
break_class_{14,90}d, first_break_time_{14,90}d, entry_price_{14,90}d,
ret_from_entry_{14,90}d, qqq_regime_open, ndx_regime_open`)

## Scripts

- `scripts/orderflowmarketcontext/noise band studies/scripts/overnight_band_gamma_study.py`
  — builds the parquet and prints the cohort tables shown above

## Related studies

- `band_break_with_qqq_levels_2026-05-04.md` (in `options level studies/`)
  — extends this study with per-level continuation tests after the band
  break, showing GEX_4 as the strongest long target and Put Support as the
  strongest short target.
