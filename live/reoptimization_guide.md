# Re-optimization Guide — When and How

**Purpose**: Avoid curve-fitting when refreshing strategy parameters.

The #1 mistake retail traders make is re-optimizing in response to drawdown. Most "broken" strategies aren't broken — they're in their normal variance band. The locked configs in this repo are all backed by 5+ years of data; their statistical noise is wider than intuition suggests.

**Default: do NOT re-optimize.** This doc exists to define the rare conditions when re-optimization is justified, and the methodology to use when it is.

---

## Hard triggers — re-optimize ONLY when ALL of these hold

1. At least **6 months of new OOS data** accumulated (live or fresh data not used in any prior optimization)
2. Live PF dropped **>25% from backtest's OOS PF** over a sample of **at least 100 trades**
3. **Walk-forward analysis on the new period would have chosen materially different params** (not the same locked config ± noise)
4. The original IS+OOS data is **preserved as a fully-held-out reference sample** that you do NOT touch during the new fit

If any of these fails, you're curve-fitting noise.

---

## Where each strategy stands

| Strategy | Trades to date | Backtest split | Re-opt readiness |
|---|---|---|---|
| **RV**   | 666 IS + 148 OOS (locked v3, ends ~2025-11) | Mature IS, thin OOS | Wait for ~250+ OOS trades total — ~6 more months minimum |
| **B2**   | 666 IS + 147 OOS (filtered locked, ends 2025-11) | Mature IS, thin OOS | Same — wait 6+ months for OOS to grow |
| **OD**   | 1,358 trades over 5+ years (single param set) | No formal split; `walk_forward.py` exists | More robust — only re-opt if live is outside WF p25/p75 band |
| **Fabio**| IS + OOS te1400 modeA | Has overfit test suite (`scripts/fabio_orb/run_overfit_tests.py`) | Most defended — re-opt only if overfit tests fail |

OOS samples are **thin** (~148 trades for RV, ~147 for B2). With small OOS, even a 30% live PF drop could be noise. Statistical significance of underperformance requires **100+ trades minimum**.

---

## Step 1 — Is the strategy actually broken? (do this BEFORE any re-opt)

### Bootstrap noise-band test

1. **Bootstrap resample** historical OOS trades 10,000 times. For each resample, compute the rolling 100-trade PF window.
2. Find the **p5 and p95** of these windows — that's your "noise band."
3. **Compare your live 100-trade rolling PF** to the bootstrap band.
   - **Live PF inside p5/p95 band** → it's noise. Do NOT re-optimize.
   - **Live PF outside the band** → real edge degradation possible — investigate before re-optimizing.

Implementation hint: trade-log CSVs already exist per strategy. Pull the OOS trades, resample with replacement, compute rolling PF on each resample.

### Parameter stability check

Run `scripts/overfit_framework/test_1_parameter_stability.py` on your CURRENT locked config:
- Perturbs each parameter ± a few steps
- Plots the resulting PnL/PF surface
- **Stable** = config sits in a wide plateau → real edge
- **Spiky** = config is at a sharp local maximum → likely overfit to begin with

If the locked config is unstable, the original optimization was overfit. In that case re-optimization is overdue regardless of live performance.

### Things that are NOT triggers to re-optimize

- A discovered bug (like `slow_client`). Fix the bug, give the strategy fresh runtime, then assess.
- A single bad month. Strategies have ~25-35% probability of losing months even with intact long-term edge.
- A drawdown matching backtest's historical max DD. That's **expected** variance, not failure.
- "I think I can do better." This is the curve-fit instinct talking. Resist.

---

## Step 2 — Methodology when you DO re-optimize

### Rule 1: Triple split, not double

Don't just do IS/OOS. Do **IS / Validation / Held-out OOS**:

| Split | Time window example | Purpose |
|---|---|---|
| **IS** | 2020-12 → 2023-12 | Where you FIT the params |
| **Validation** | 2024-01 → 2024-12 | Where you COMPARE candidate configs |
| **Held-out OOS** | 2025-01 → today | NEVER touched during search |

You only look at the held-out OOS ONCE, at the very end, on the single config you've already chosen on Validation.

**If the chosen config performs badly on held-out OOS** → revert to the locked config. Do NOT re-optimize again. The held-out OOS is one-shot — re-using it kills the held-out integrity.

### Rule 2: Walk-forward, not point-fit

For each fold, fit on a rolling 24-month window, test on the next 12 months. Existing implementations:
- `scripts/overnight drift strategy/walk_forward.py`
- Use as a template for RV / B2 / Fabio

Acceptance criteria:
- **Consistent across 3+ folds** = probably real edge
- **Wildly different config per fold** = noisy surface, you're overfitting → STOP

### Rule 3: Parameter stability check on the new config

After finding a "winner" on Validation, run `test_1_parameter_stability.py` on it:
- Perturb each param ± 2 steps
- Within those perturbations, **total return should stay within 10% of the optimum**
- If perturbing one param ±1 step drops PnL by 30% → it's a curve-fit. Reject.

### Rule 4: Cap the search budget

If you test 1,000 parameter combos, by random chance ~50 will look "great" on any held-out sample.

**Budget rules:**
- **Phase 1** (coarse sweep): ≤100 combos → pick top 5 by Validation PF
- **Phase 2** (fine sweep around top 5): ≤100 combos → pick top 1
- **Total trials < 200** across both phases
- **Do NOT iterate the whole pipeline more than 2-3 times** — each iteration burns held-out validity

### Rule 5: Out-of-sample MUST beat locked

For the new config to replace the locked one, it must satisfy ALL of:
- Beats locked on **held-out OOS PF** (not Validation, not IS)
- At least **50 trades in held-out OOS**
- **IS PF ≥ 1.1** (filters out cells where IS is breakeven — those are random OOS hits, not real edge)
- Passes parameter stability test (Rule 3)
- Walk-forward shows consistency across folds (Rule 2)

If ANY check fails → keep the locked config. The new one isn't proven enough to replace 5 years of data.

---

## What to allow vs lock when re-optimizing

| Parameter type | Re-opt OK? | Reason |
|---|---|---|
| Entry filter thresholds (z, delta, pinbar ratios) | ✓ Yes | Statistical thresholds drift with regime |
| ATR multipliers (SL/TP) | ✓ Cautiously | Mild changes OK; large changes = different strategy |
| Time windows / hour filters | ✗ Rarely | Market structure is more stable than these — if hour filter changes, ask why |
| Exit logic (ratchet vs fixed, BE rules) | ✗ No | Changing this = building a new strategy, not re-optimizing |
| Strategy direction logic (z > X means LONG/SHORT) | ✗ No | Same as above |
| Martingale on/off, recovery sizing | ✗ Rarely | High-impact change, requires fresh validation pipeline |

---

## Per-strategy specific notes

### RV (`scripts/rough vol orderflow/`)
- Re-opt pipeline: `phase1_base_sweep.py` → `phase2_orderflow_gamma.py` → `inspect_config_v3.py` (locked produces this)
- Held-out OOS should be the period AFTER `core.IS_END` (currently 2024-12-31)
- Locked v3 specifics in `live/rough vol orderflow/best live config/config.md`

### B2 (`scripts/overnight range strat/scripts/`)
- Re-opt pipeline: `range_break_entry_signal_study.py` → `augment_with_confirmation_absorption.py` → `sweep_ratchet_sl_fixed_tp.py` → `lock_v2_k08_lock045_mart_fc_filtered.py`
- Held-out OOS = data after 2025-01-01 (configured in scripts)
- Gamma parquet must be current (run `scripts/thetadata/daily_pipeline/run_daily.py`)

### OD (`scripts/overnight drift strategy/`)
- Re-opt pipeline: `optimize.py` → `walk_forward.py` → `run_backtest.py` → `generate_locked_pnl.py`
- Walk-forward is already implemented and locked params are at the WF efficient frontier
- Don't change `entry_hour=19` lightly — confirmed optimal in earlier sweep at this dir

### Fabio (`scripts/fabio_orb/`)
- Re-opt pipeline: `run_fabio_sweep.py` → `run_overfit_tests.py` → `run_final_config.py`
- Overfit test suite is the strictest defense in this repo — use it before deploying anything new

---

## Concrete recommendation as of 2026-05-21

**Right now (live started ~weeks ago):**
- **Do not re-optimize anything.** Sample is too small.
- **Run for 6-9 more months** with the locked configs.
- **Track live PF per strategy** in a simple CSV — compute rolling 100-trade PF.
- Once you hit 50+ live trades per strategy, run the **bootstrap noise-band test** to establish what's normal variance.

**At month 9-12 from go-live (~early 2027):**
- If live PF is INSIDE the bootstrap noise band → keep locked. The strategies are performing as expected.
- If live PF is OUTSIDE the band → investigate root cause (data quality? broker fills? regime change?). Only THEN consider re-optimization using the triple-split + walk-forward + stability pipeline above.

**At any time:**
- Bugs aren't re-opt triggers. Fix the bug, restart, give it fresh runtime.
- Drawdowns within historical norms aren't re-opt triggers.
- "I want better numbers" isn't a re-opt trigger. That's curve-fit motivation.

The locked configs you have were optimized once on a 5-year sample. That's already statistically dense. New monthly data adds 1-2% to the sample — not enough to change the picture meaningfully. **Trust the locked configs until the evidence is overwhelming.**

---

## Reference: signs you're curve-fitting

Watch for these red flags during any optimization session:

| Red flag | Diagnosis |
|---|---|
| Best config changes dramatically each fold of walk-forward | Surface is noisy → overfitting |
| Optimum is at a sharp local maximum (±1 step drops 30%+) | Curve-fit to specific data |
| New config beats locked by 50%+ on validation | Probably looking at noise variance |
| Performance only beats locked on the most recent quarter | Recency bias / regime overfit |
| You've re-run the pipeline more than 3 times | You're burning held-out validity |
| New config requires changing exit logic, not just thresholds | You're building a new strategy, not re-optimizing |
| Higher trade count "wins" → 2× trades, slightly higher total | Likely dilution: more trades, worse PF, marginal edge |

When in doubt: keep the locked config. **The cost of staying with a working strategy is much lower than the cost of breaking it via curve-fit.**
