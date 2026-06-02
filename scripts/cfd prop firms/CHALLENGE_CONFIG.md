# FundingPips 100K 2-Step — CHALLENGE (pass fast) vs FUNDED (keep alive)

The challenge and funded phases want OPPOSITE sizing because of one rule.

## What is RPTI?
**RPTI = Risk Per Trade Idea.** Max loss on a single "trade idea", counting **realized AND
floating (unrealized)** loss across all related positions.
- $100k ($50k+): **2% = $2,000.** Hit $2,000 of loss on one idea → account terminated.
- "One trade idea" = same instrument + same direction held concurrently, **plus** any same-direction
  re-entry within **10 minutes** of closing a loser. (Combined trades one instrument = NAS100.)
- **CRITICAL: RPTI applies on the FUNDED / Master account ONLY. NOT during the challenge.**

## Data + method (all numbers below)
Full 4-way (OD+RV+B2+FB), **martingale OFF**, **floating/MAE-aware** (equity-based floors), MAE
computed consistently from **1-min bar lows (long)/highs (short) over each trade's hold**
(`scripts/montecarlo/results/combined_4way_with_mae_1min.csv`).

## CHALLENGE — RPTI EXEMPT → size UP, pass fast
Rules: $100k, 5% daily loss (of day-start, floating), 10% static floor ($90k), **P1 +$8k / P2 +$5k**.
No RPTI; only the daily + max-loss floors bind. Use the full 4-way and size up.

| MNQ | P(pass BOTH) | median days |
|-----|--------------|-------------|
| 2   | 84.3%        | 195d (~9.3 mo) |
| 3   | 80.8%        | 122d (~5.8 mo) |
| 4   | 73.0%        | 80d (~3.8 mo) |
| **5** | **69.0%**  | **57d (~2.7 mo)** |
| 6   | 62.6%        | 43d (~2.0 mo) |

**To pass quickly: 5 MNQ → ~69% in ~2.7 mo, or 6 MNQ → ~63% in ~2.0 mo.** Pass rate stays high at
size because there's no RPTI and the 5%/10% floors are generous; the +$8k P1 target is the gate.
(Optional: run martingale ON for the challenge only — no RPTI to punish it, DD is generous — to pass
even faster, then turn it off once funded.)

## FUNDED — RPTI APPLIES → keep RV small, size the rest up
RPTI is live and floating-based. Per-strat worst single-trade float (1 NQ, 1-min MAE) and RPTI-safe
size: **OD $5,565 (safe to 3.6 MNQ), RV $16,225 (TIGHTEST — 1.2 MNQ), B2 $6,155 (3.2), FB $6,215
(3.2).** So the lever is **keep RV at 1 MNQ, size OD/B2/FB up.** (NOT "drop OD" — that was a prior
error from a bad OD MAE column.)

| Funded config | E[$ withdrawn]/yr | blow(1yr) |
|---------------|-------------------|-----------|
| ALL 4 @1                 | $5,815  | **0.0%** |
| **RV@1, OD/B2/FB @2**    | **$10,012** | **2.8%** |  ← best risk-adjusted |
| RV@1, OD/B2/FB @3        | $13,995 | 16.7% |
| ALL 4 @2                 | $10,956 | 18.3% |

**Best funded config: RV@1 + OD/B2/FB@2 → ~$10k/yr at ~3% blow** (double the all-@1 EV, nearly
bulletproof). Push the safe three to 3 → ~$14k/yr at ~17% blow. Blows are ~all RPTI (DLL ~0%).

## Per-strat sizing: when it matters (give bad-DD strats less size)
Principle: size each strat inversely to its drawdown — RV (worst float $16k) small, OD/B2/FB (low
float) big. But it ONLY pays off where the constraint is PER-TRADE/PER-POSITION:
- **FUNDED — big win.** The $2k RPTI is a per-trade kill, so one oversized RV trade ends the account.
  RV@1 + OD/B2/FB@2 = $10k/yr @ 2.8% blow vs uniform ALL@2 = $11k @ 18% blow — same money, ~6x less
  blow, purely from shrinking RV. Per-strat sizing is essential on funded.
- **CHALLENGE — marginal.** The 5% daily ($5k) / 10% max are account-level and generous (not a
  per-trade kill); RV rarely floats to $5k even at 4-5 MNQ. Per-strat (RV@3, OD/B2/FB@6) is only a
  touch faster than uniform 5 MNQ at the same ~62-63% pass. Uniform sizing is fine on the challenge.
  (Challenge pass % wobbles a few pts with within-day trade ordering — treat deltas as roughly equal.)

## Summary
- **Challenge: full 4-way, ~5 MNQ uniform, pass ~69% in ~2.7 months** (RPTI doesn't apply — size up).
- **Funded: RV@1 + OD/B2/FB@2 (per-strat), milk ~$10k/yr at ~3% blow** (RPTI caps RV — keep it small).

(Caveat: 1-min MAE may slightly understate true tick-level floats; sequential MAE check doesn't fully
combine concurrent same-direction RTH positions — minor here since RV is the only wide-float leg.)
