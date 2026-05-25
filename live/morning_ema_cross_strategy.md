# Morning EMA Cross — Strategy Reference (Validated, Not Deployed)

**Status**: Validated in backtest (140/140 sweep cells pass IS+OOS), NOT deployed live.
**Created**: 2026-05-25 — keep on the bench until deployment decision.
**Author**: nm8 + research-assist
**Backtest range**: 2020-12-01 → 2026-05-22 (~5.5 years, ~1,350 trades)

---

## Strategy idea

Pure trend-following on NQ 30-min bars during regular trading hours:

- **Entry**: at the close of the 09:30-10:00 ET bar (10:00 entry)
  - If close > EMA(N): **LONG**
  - If close < EMA(N): **SHORT**
- **SL**: fixed ATR-based stop at entry, intrabar fill
  - `sl_price = entry_price - sign * sl_mult * ATR(at_entry)`
- **TP**: none
- **Force close**: 14:30 ET
- **Martingale**: OD-style (any 1c loss → 2c next, then back to 1c regardless)

Simple structure: one trade per RTH day, max 9 × 30-min bars of holding time (10:00 → 14:30).

---

## Best configs

### Risk-adjusted winner (use this)
- **EMA=60, ATR=28, SL_mult=1.25**
- Net $: $265,352 (5.5 yrs)
- PF: 1.212
- MDD: -$39,715
- Net/MDD: 6.68x
- WR: 30.4% (low — trend-following, cut losers fast)
- R:R: ~2.88 (winners much bigger than losers)
- IS: $115K (PF 1.17), OOS: $150K (PF 1.26) — OOS bigger than IS, anti-overfit

### Absolute $ winner
- **EMA=60, ATR=10, SL_mult=2.75**
- Net $: $282,664
- PF: 1.174
- MDD: -$49,799
- Net/MDD: 5.68x
- WR: 50.2%
- Slightly weaker risk-adjusted than the above; consider only if you have more DD budget

### Lowest MDD config
- **EMA=60, ATR=10, SL_mult=0.45**
- Net $: $166,080
- MDD: -$36,856 (lowest in the entire sweep)
- Tight stop, lower income
- Use if MDD is the hard constraint

---

## Sweep robustness

- 140/140 cells (5 ATR periods × 28 SL multipliers) passed strict IS+OOS positive — **wide plateau, no overfit signature**.
- Two coexisting profit pockets:
  1. **Tight SL + long ATR** (low WR ~30%, high R:R ~2.9): ATR=21-28, SL_mult=1.0-1.5
  2. **Wide SL + short ATR** (high WR ~50%, lower R:R): ATR=7-10, SL_mult=2.5-3.0

---

## Yearly performance (unfiltered, risk-adj config)

| Year | n | WR | Net | PF | MDD | Notes |
|---|---|---|---|---|---|---|
| 2020 (Dec) | 22 | 9% | -$12,768 | 0.15 | -$12K | Partial year, choppy start |
| 2021 | 251 | 33.5% | +$90,704 | 1.57 | -$13K | Best year |
| 2022 | 239 | 30.5% | +$23,230 | 1.09 | -$38K | Bear regime, recovered |
| 2023 | 246 | 25.6% | +$4,434 | 1.02 | -$27K | Near-flat |
| 2024 | 249 | 28.1% | +$54,724 | 1.26 | -$18K | Solid |
| 2025 | 248 | 33.1% | +$80,919 | 1.28 | -$35K | Strong |
| 2026 YTD | 98 | 37.8% | +$24,109 | 1.19 | -$25K | Healthy YTD |

Edge present in every year except 2020 (partial) and 2023 (near-flat). 2021/2025 best, 2022 worst-MDD.

---

## Gamma regime finding (important caveat)

**Initial analysis was wrong due to lookahead bias.** The first pass used same-day gamma_sign in a `merge_asof(direction="backward")` — gamma for date X is computed from X's 17:15 ET settle (AFTER our 10 AM entry). Matching trade date = gamma date was peeking at post-trade settled data.

**Corrected analysis** (strict prior-day gamma, `date < trade_date`):
- NEG gamma days carry ~95% of the strategy's profit ($233K of $265K)
- POS gamma days are essentially flat (+$35K out of $265K)
- Direction within a regime barely matters (NEG+LONG +$140K, NEG+SHORT +$93K; POS+LONG +$14K, POS+SHORT +$20K)

**Filter does NOT improve the strategy.** All 4 variants (NO_FILTER, BLOCK_POS_LONG, BLOCK_POS_SHORT, BLOCK_POS_BOTH) tested:
- NO_FILTER: $265K / -$39,715 MDD
- Best filter (BLOCK_POS_BOTH): $250K / -$36,274 MDD — small MDD improvement, but bigger income loss

**Lesson**: future analyses must use prior-day gamma (or proper time-of-day awareness) to avoid this trap.

---

## MDD floor

Structural floor of ~$37K at 1 NQ basis. Cannot be reduced below this without sizing down. To hit MDD ≤ $30K target:
- **Size at 75%** of 1 NQ basis (= 7.5 MNQ or 0.75 NDX100 lots)
- Net scales to ~$199K, MDD scales to ~$30K
- Net/MDD ratio unchanged

---

## Prop firm fit (NOT a good fit standalone)

At full 1 NQ basis sizing:
- $39K MDD = 39% of $100K → blows 10% max DD
- $39K MDD = 19.5% of $200K → still blows 10%
- $39K MDD = 13% of $300K → still blows 10%

Need ~25% sizing to fit a $100K prop firm 10% rule. Net drops to ~$66K/yr.

Better fit: **add as a 5th strategy in the 4-strat combined stack** — the 4-strat combined MDD is only ~$29K because of negative correlation. Adding this would increase concurrent exposure during 10:00-14:30 RTH window (overlaps with B2, RV, Fabio). Cumulative-risk coordinator would need to budget for it.

---

## TODO before deploying

1. **Full 5-test overfit framework** (`scripts/overfit_framework/`):
   - Test 1: Parameter Stability — already implicitly passed (wide plateau, 140/140 cells)
   - Test 2: Walk-Forward — needed (rolling 12-mo windows)
   - Test 3: MC Order Shuffling — needed
   - Test 4: Bootstrap CI — needed
   - Test 5: Direction Permutation — needed (this is the killer test; PF 1.21 is borderline)

2. **Test without martingale** — is the edge real at 1c constant, or does martingale carry most of it?

3. **Inspect worst-trade tail** — what's the largest single-trade loss? Important for prop firm sizing.

4. **Long-only vs short-only breakdown** — does the edge come from both sides equally, or one side carries it?

5. **Combine with 4-strat stack** — rerun the 5%ers / FN MCs with this as a 5th strategy to see portfolio-level MDD impact.

6. **News filter integration** — strategy enters at 10:00 ET. Common 10:00 ET red folder events (ISM, JOLTS, Consumer Confidence, etc.) hit right at entry. ±2 min blackout would block these entries on news days. Test if income survives.

---

## Files

- **Engine + sweep**: `scripts/morning_ema_cross/strategy.py` (yellow-ratchet variant, superseded)
- **Engine + sweep (current best)**: `scripts/morning_ema_cross/strategy_atr_sl.py` (ATR-fixed SL)
- **Gamma analysis (lookahead-fixed)**: `scripts/morning_ema_cross/gamma_split_no_lookahead.py`
- **Filter variants test**: `scripts/morning_ema_cross/filter_variants_test.py`
- **Sweep CSV**: `scripts/morning_ema_cross/results/morning_ema_cross_atr_sl_sweep.csv`
- **Best trades CSV**: `scripts/morning_ema_cross/results/morning_ema_atr_sl_best_trades.csv`

---

## Quick-start commands

```powershell
# Re-run the SL sweep (when new data arrives)
python scripts/morning_ema_cross/strategy_atr_sl.py

# Verify gamma split is still correct (no lookahead)
python scripts/morning_ema_cross/gamma_split_no_lookahead.py

# Test filter variants
python scripts/morning_ema_cross/filter_variants_test.py
```

---

## When to revisit

Consider deploying if:
- Live performance of the existing 4-strat stack is solid AND
- You want a 5th strategy for portfolio diversification AND
- 10:00 ET RTH window is not already overcrowded with concurrent positions

Skip if:
- You're prop-firm-constrained on cumulative exposure (this adds another concurrent slot)
- Trends die in NQ (the strategy depends on directional momentum)
- 2026-2027 turns into a 2023-style chop year (strategy goes near-flat)
