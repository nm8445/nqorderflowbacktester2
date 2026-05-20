# HVL 0DTE Mean Reversion / Regime Study (2026-05-04)

## Hypothesis

When NQ is in a **positive-gamma regime** at the open (price above HVL 0DTE
or all-positive-gamma chain), the day should exhibit pin / mean-reverting
behavior. We test:

- **Directional**: does P(close > open) increase?
- **Volatility**: does intraday `|return|` compress?

## Methodology

For each trading day `D`:

1. Look up previous trading day `P`.
2. From `P`'s settled QQQ greeks + OI, compute the cumulative net GEX of the
   0–1 DTE chain (expirations on `D` and `D+1`), restricted to ±5% of spot.
3. Same for NDX with local IV inversion (NDX greeks gated on Standard tier).
4. Classify each day into one of 4 regimes:
   - `above_flip` — flip exists, spot above HVL strike (positive gamma)
   - `below_flip` — flip exists, spot below HVL strike (negative gamma)
   - `deep_pos`   — no flip, entire ±5% band is positive cumulative GEX (deep pin)
   - `deep_neg`   — no flip, entire ±5% band is negative cumulative GEX (deep vol)
5. NQ price source: `markettick_1min_bars.parquet` (2020-12 to 2025-12) +
   `15min_bars.parquet` (to 2026-04-08). 9:30 ET open and 17:00 ET close.
6. Compute `ret = (close - open) / open` and `|ret|`. Compare cohorts.

## Sample

- **Date range**: 2020-12-02 → 2026-04-08
- **Total days with valid open/close**: 1,310
- **HVL source timing**: prior EOD (no lookahead)

## Regime distribution

### QQQ-derived (1,310 days)

| Regime | Count | % |
|---|---|---|
| `above_flip` | 208 | 16% |
| `below_flip` | 404 | 31% |
| `deep_pos`   | 19 | 1% |
| **`deep_neg`** | **642** | **49%** |
| `mixed_no_flip` | 1 | 0% |
| `no_data` | 36 | 3% |

### NDX-derived

| Regime | Count | % |
|---|---|---|
| `above_flip` | 446 | 34% |
| `below_flip` | 419 | 32% |
| `deep_pos` | 1 | 0% |
| `deep_neg` | 444 | 34% |

QQQ chain is dominated by put-side gamma in ~50% of days — consistent with
retail buying portfolio-hedge puts. NDX is more balanced (institutional
both-sides flow).

## Results

### Headline (all 1,310 days)

| Metric | Value |
|---|---|
| P(close > open) | 55.04% |
| Mean return | +0.039% |
| `|ret|` mean | 0.870% |

### Per-regime stats — QQQ-derived

| Regime | n | P(>0) | Mean ret | `|ret|` mean | `|ret|` median |
|---|---|---|---|---|---|
| above_flip | 208 | 55.29% | +0.032% | **0.661%** | 0.462% |
| below_flip | 404 | 55.94% | +0.003% | 0.760% | 0.583% |
| deep_pos | 19 | 52.63% | +0.158% | 0.620% | 0.607% |
| **deep_neg** | 642 | 54.05% | +0.053% | **1.019%** | 0.818% |

### Per-regime stats — NDX-derived

| Regime | n | P(>0) | Mean ret | `|ret|` mean | `|ret|` median |
|---|---|---|---|---|---|
| above_flip | 446 | 52.24% | +0.016% | **0.666%** | 0.499% |
| below_flip | 419 | 55.37% | −0.022% | 0.812% | 0.643% |
| **deep_neg** | 444 | 57.43% | +0.116% | **1.131%** | 0.931% |

## Statistical tests

### Volatility compression (`|ret|`, pos-gamma vs neg-gamma)

`pos_gamma = above_flip + deep_pos`
`neg_gamma = below_flip + deep_neg`

| Source | Pos `|ret|` | Neg `|ret|` | Diff | t-stat | p |
|---|---|---|---|---|---|
| QQQ | 0.658% (n=227) | 0.919% (n=1046) | **−0.261%** | **−5.38** | **<0.00005** ✓ |
| NDX | 0.666% (n=447) | 0.976% (n=863) | **−0.310%** | **−7.19** | **<0.00005** ✓ |

### Directional bias (P(close>open), pos vs neg gamma)

| Source | Pos % | Neg % | chi² | p |
|---|---|---|---|---|
| QQQ | 55.07% | 54.78% | 0.00 | 1.000 |
| NDX | 52.35% | 56.43% | 1.82 | 0.177 |

## Conclusion

### Confirmed: vol compression is real and strong

Positive-gamma regime cuts the day's intraday range by ~30 bps. Robust across
QQQ-derived and NDX-derived cohort splits. **p < 0.00005 in both.** This is
the cleanest finding in the study.

### Not confirmed: directional close > open

P(close > open) doesn't differ significantly between regimes. Mean-reversion
in price direction isn't a thing in this data.


### Most common regime is deep_neg, especially in QQQ

QQQ's 0-1 DTE chain has put-side dominance throughout ±5% of spot on ~half of
all days (642 / 1,310 = 49%). NDX is more balanced (34% deep_neg).

### `deep_neg` days are the biggest-range days

`|ret|` mean 1.02% (QQQ) / 1.13% (NDX) vs base rate 0.87%. Days with no
positive gamma anywhere in the near-spot 0-1 DTE band are 17% wider than
average — clearest "trend day" filter we have.

## Practical applications

| Strategy | Filter |
|---|---|
| Premium selling / mean-revert intraday | `above_flip` or `deep_pos` — ~30% smaller `|ret|` |
| Trend following | `deep_neg` — 70% larger `|ret|` than `above_flip` cohort |
| Avoid: random walk in directional terms | None of the cohorts give a directional bias |

## Caveats

- **HVL 0DTE comes from prior EOD**, not live morning data. Same-day intraday
  recomputation might shift cohort assignments at the margin. A follow-up
  study with same-day 9:30 IV recomputation is queued.
- **NQ price source** ends at 2026-04-08 (15-min bar parquet limit). Days
  after that not in this run.
- **ThetaData OI is EOD only.** Industry-standard limitation, not specific to
  Standard tier.

---

## FOLLOW-UP: Intraday regime tracking (2026-05-04)

Re-ran the study tracking the gamma regime continuously through the session
instead of using a static label at the open. For each day, we evaluate
`cum_gex(NQ_price)` at every 5-min bar from 9:30 to 17:00 ET against the
prior EOD chain's cumulative-GEX curve. Each bar gets a regime tag (positive
or negative gamma based on local cum_gex sign), then aggregated per day.

### Key methodology change

The cumulative-GEX curve is built once at market open from prior EOD chain.
Price moves through that fixed curve all day. At any moment:
- `cum_gex(current_price) > 0` → positive-gamma (pin) regime
- `cum_gex(current_price) < 0` → negative-gamma (vol) regime

Days are then bucketed by **% of session spent in positive-gamma**.

### Cohort breakdown — NDX-derived (n=1,208)

| Cohort | n | P(close>open) | Mean ret | `|ret|` mean |
|---|---|---|---|---|
| **Pure pin** (≥95% pos-gamma) | 301 | **73.75%** | **+0.42%** | 0.62% |
| Mostly pin (50–95%) | 118 | 58.47% | +0.14% | 0.69% |
| Mostly vol (5–50%) | 124 | **37.90%** | **−0.35%** | 0.91% |
| Pure vol (<5% pos-gamma) | 665 | 49.47% | −0.08% | 1.04% |

### Cohort breakdown — QQQ-derived (n=1,172)

| Cohort | n | P(close>open) | Mean ret | `|ret|` mean |
|---|---|---|---|---|
| **Pure pin** (≥95% pos-gamma) | 176 | **69.89%** | **+0.365%** | 0.61% |
| Mostly pin (50–95%) | 79 | 64.56% | +0.327% | 0.71% |
| Mostly vol (5–50%) | 61 | **39.34%** | **−0.159%** | 0.92% |
| Pure vol (<5% pos-gamma) | 856 | 52.10% | −0.043% | 0.96% |

### Statistical tests

| Source | Pure-pin t-stat | p | Pin vs Vol `|ret|` t-stat | p |
|---|---|---|---|---|
| QQQ | **+6.76** | <0.00005 | −6.86 | <0.00005 |
| NDX | **+10.51** | <0.00005 | −8.56 | <0.00005 |

### Regime persistence

| Source | Open == Close regime | Median flips/day | Mean flips/day |
|---|---|---|---|
| QQQ | 90.7% | 0 | 0.78 |
| NDX | 85.4% | 0 | 1.46 |

**Most days the regime persists from open to close**, so the open snapshot
is a useful signal — just don't expect it to hold 100% of the time.

### Counterintuitive flip-count finding

| QQQ flips | n | `|ret|` mean |
|---|---|---|
| 0 flips (regime steady) | 972 | 0.91% |
| **1 flip (clean transition)** | 44 | **1.19%** |
| ≥2 flips (choppy oscillation) | 156 | 0.63% |

A **single regime transition during the day** is the largest-range cohort —
clean breakouts through HVL produce the biggest moves. Multiple flips means
price is oscillating *around* HVL, which is small-range pin behavior in
disguise.

### What changed from the static-label findings

The original static-label study (above) found NO directional bias between
above_flip and below_flip cohorts. Intraday tracking reveals a **strong
directional UP bias for pure-pin days** (~70% close > open) that was masked
when we mixed pure-pin days with transition days under the same label.

### Practical filter for live trading

| If you want… | Filter |
|---|---|
| Long bias with pin-style range | "Pure pin" — currently in pos-gamma AND has been since open. ~70% hit rate, +40 bps drift |
| Short bias | "Mostly vol" cohort — pin tries to form but mostly fails. 38% hit rate (= 62% short bias) |
| Big-range trend day | After a single regime flip is observed during the day → 1.19% range vs 0.91% baseline |
| Small-range chop | Multiple flips → 0.63% range, treat as pin |

### Output data

`D:/trading_pythonbacktest_data/QQQ_thetadata/study_hvl0dte_intraday_regime.parquet`
(1,210 rows × time-in-regime stats)

### Scripts

- `../scripts/study_hvl_intraday_regime.py` — builds the per-day intraday
  regime stats parquet
- `../scripts/analyze_hvl_intraday_regime.py` — runs the cohort analysis

## Output data

`D:/trading_pythonbacktest_data/QQQ_thetadata/study_hvl0dte_meanrev.parquet`
(1,310 rows × `date, nq_open_930, nq_close_500, qqq_hvl_0dte_strike,
qqq_hvl_0dte_nq, qqq_regime, ndx_hvl_0dte_strike, ndx_hvl_0dte_nq, ndx_regime,
qqq_ratio_used, ndx_basis_used, ret, abs_ret`)

## Scripts

- `../scripts/study_hvl_0dte_meanreversion.py` — builds the per-day parquet
  with regime classification (slow path: per-day NDX IV inversion)
- `../scripts/analyze_hvl_0dte_meanrev.py` — loads parquet, runs 4-cohort
  analysis (fast)
