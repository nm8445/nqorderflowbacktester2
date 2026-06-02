# FundingPips-style 100K CFD 2-Step — Challenge Pass Rates + Funded EV

> ## ⚠️ CORRECTED 2026-06-01 — read this; everything below is superseded
> Real config: **P1 +8% / P2 +5%**, 5% daily / 10% max, all FLOATING/equity-based. MAE now
> recomputed **consistently from 1-min bars** for all 4 strats (`build_4way_mae.py` ->
> `combined_4way_with_mae_1min.csv`), marti OFF.
>
> **Per-strat worst float (1 NQ): OD $5,565 · RV $16,225 (TIGHTEST) · B2 $6,155 · FB $6,215.**
> The earlier "OD is the RPTI killer / drop OD" was WRONG (bad OD MAE in the old `mae_$` column).
> **RV is the binding leg.**
>
> **CHALLENGE (RPTI-EXEMPT — size up, full 4-way, floating, marti off, P1+8%/P2+5%):**
> | MNQ | P(both) | median days |
> |---|---|---|
> | 3 | 80.8% | 122d (~5.8mo) |
> | 4 | 73.0% | 80d (~3.8mo) |
> | **5** | **69.0%** | **57d (~2.7mo)** |
> | 6 | 62.6% | 43d (~2.0mo) |
> → fast-pass ~5 MNQ uniform (~69% in ~2.7mo). Per-strat sizing only marginal on the challenge.
>
> **FUNDED (RPTI $2k per-trade applies — keep RV small, size others up):**
> | config | E[$ withdrawn]/yr | blow(1yr) |
> |---|---|---|
> | ALL 4 @1 | $5,815 | 0.0% |
> | **RV@1, OD/B2/FB @2** | **$10,012** | **2.8%** |
> | RV@1, OD/B2/FB @3 | $13,995 | 16.7% |
> → best = **RV@1 + OD/B2/FB@2 ≈ $10k/yr at ~3% blow** (per-strat sizing is essential on funded).
> See CHALLENGE_CONFIG.md for the clean writeup. Sections below are kept for history only.

---

Live **4-way combined (OD/B2/RV/FB)** on a $100k CFD 2-step. Daily P&L bootstrapped from
`scripts/rough vol orderflow/results/combined_4way_trades.csv` (1 NQ/strat = 10 MNQ base),
intraday-aware (trades walked within each day by exit time). MNQ $2/pt; ~$4/MNQ round-turn cost.
Realized-PnL DLL model (per-strat sizing makes this robust; flat-MNQ at the edge is a touch
optimistic since OD's overnight *floating* MAE could trip the daily limit harder).

## Two account rule-sets compared

| Account | Daily loss | Max loss | P1 target | P2 target |
|---------|-----------|----------|-----------|-----------|
| **A** | 3% ($3k, of day-start) | 6% (floor $94k) | +$6k | +$6k |
| **B** | 5% ($5k, of day-start) | 10% (floor $90k) | +$10k | +$5k |

## Per-strategy MAE sizing (the smart sizing)

Each strat sized to an equal $-risk-per-trade using its live worst-MAE basis (from
`mt5_executor.sl_pts_per_strat`): `scale_strat = R / (sl_pts * $20)`.
At R = $1,000/trade:

| Strat | MAE basis | ≈ MNQ |
|-------|-----------|-------|
| OD | 600 pt | ~1 |
| B2 | 600 pt | ~1 |
| RV | 200 pt | ~2.5 |
| FB | 150 pt | ~3.3 |

OD/B2 tiny, RV/FB large -> OD's overnight tail days can NO LONGER breach the daily limit
(`fail_DLL = 0%`). Dropping OD does not raise pass rate (harmless when sized small) — keep it.

## CHALLENGE pass rate + days (best sizings)

### Account A — 3% daily / 6% max, +$6k each phase
| Sizing | P(both) | median days | DLL busts |
|--------|---------|-------------|-----------|
| Flat 2 MNQ | 82% | 129 td (~6 mo) | 0% (explodes >2 MNQ) |
| Per-strat $1k/trade | 79% | 136 td | 0% |
Flat 3+ MNQ here busts on the 3% daily limit (OD tails). Per-strat is the safer pick.

### Account B — 5% daily / 10% max, +$10k / +$5k  (EASIER despite bigger P1 target)
| Sizing | P(both) | median days | DLL busts |
|--------|---------|-------------|-----------|
| Flat 2 MNQ | 88.6% | 175 td (~8 mo) | 0% |
| **Flat 3 MNQ** | **87.2%** | **110 td (~5 mo)** | 0% |
| Per-strat $1.5k/trade | 85.2% | 116 td | 0% |
Looser DD lets you size to 3 MNQ -> higher pass AND faster. P1 (+$10k) is the harder gate
(~91-92% pass); P2 (+$5k) is easy. **Account B is the better challenge to buy.**

## REAL FundingPips funded rules (fundingpips_funded_rpti.py) — RPTI is the killer

Funded (Master) rules, EQUITY/floating based, NO hard stop possible (would cut the edge):
  - Daily Loss 5% of max(open balance, open equity); Max Loss 10% static ($90k); reset 5pm ET.
  - **Risk Per Trade Idea (RPTI): $2,000 single-trade-idea floating loss = termination.** Funded
    ONLY (eval exempt). One instrument (NAS100) => concurrent same-direction + 10-min reopen COMBINE.

MAE (floating loss, 1 NQ): OD median $600 / max $30,100; RV median $1,308 / max $16,370;
B2 median $900 / max $16,660. **This is why you size in MNQ, not NQ** — /10 keeps floats under $2k.

Rule-accurate funded EV (flat MNQ, native exits, OD/RV/B2 + MAE; FB excluded ~ adds profit, low RPTI):
| Flat MNQ | E[$ withdrawn]/yr | bust(1yr) | via RPTI | via DLL | via MaxLoss |
|----------|-------------------|-----------|----------|---------|-------------|
| **0.5**  | $3,343            | **0%**    | 0%       | 0%      | 0% |
| 1.0      | $5,288            | 44%       | 44%      | 0%      | 0% |
| 1.5      | $6,494            | 67%       | 67%      | 0%      | 0% |
| 2.0      | $6,211            | 88%       | 87%      | 0%      | 1% |

KEY: RPTI is the ONLY funded killer (DLL/MaxLoss ~0%). OD's overnight gap MAE (unbounded, no stop)
is the binding leg: 0% bust at 0.5 MNQ; 44% at 1 MNQ. RV/B2 only float ~$1.6k at 1 MNQ (safe).
=> trade OD at ~0.5 MNQ (or drop on funded), RV/B2/FB ~1 MNQ. Realistic account_EV (0.86 pass -
$500 fee): ~$2.4k @ 0.5 MNQ (never terminated) .. ~$4k @ 1 MNQ (~44%/yr replacement).
The RPTI-BLIND numbers below were too high — kept only as the (wrong) no-RPTI bound.

## (RPTI-blind, too optimistic) FUNDED EV (bi-weekly 80% payout, 1-yr horizon)

NOTE: trades play out to NATIVE exits — no hard stop / risk cap (a hard cap at the MAE distance
would truncate trades and break the edge). Sizing is pure lot sizing.

### Flat MNQ (uniform size, converted to NAS100 lots — what I'll actually trade)
$2k max cumulative-exposure cap -> ~1.5 MNQ. OD is the binding leg: no hard stop, native ~600pt
adverse move -> 600*$2*MNQ = $2000 -> ~1.7 MNQ. So ~1.5 MNQ keeps open risk <=$2k.

| Flat MNQ | A (3%/6%) E[$]/bust | B (5%/10%) E[$]/bust |
|----------|---------------------|----------------------|
| 1.5 MNQ  | $11,218 / 17%       | $11,540 / **1.6%** |
| 2 MNQ    | $13,866 / 39%       | $15,301 / 7% |
| 3 MNQ    | $13,785 / 84%       | $21,600 / 29% |
| 4 MNQ    | $10,587 / 99%       | $24,202 / 61% |

**Answer: at the $2k-capped size (~1.5 MNQ) the GROSS EV is ~the same (~$11.2k vs ~$11.5k/yr) —
both survive at that small size — but B busts 1.6% vs A's 17%, and B can size PAST the cap
(2-3 MNQ -> $15k-$21.6k) while A busts hard above ~1.5-2 MNQ.** EV depends on size AND the
account's DD rules, not size alone.

### Per-strat lot sizing (OD ~1 MNQ, RV/FB ~2.5-3 MNQ — NOT a hard stop, still native exits)
Sizing only OD down (the leg that breaches $2k) lets RV/FB run bigger at the SAME $2k exposure:
| risk basis | A (3%/6%) | B (5%/10%) |
|------------|-----------|------------|
| ~$1k/trade | $13,516 / 40% | $14,908 / 7% |
Recovers ~$15k on B vs ~$11.5k flat — same $2k exposure, more EV, no edge cut.

`account_EV = P(2-step pass) x E[$ withdrawn] - ~$500 fee`. B @ 1.5 MNQ flat ~ $9.4k;
B @ 3 MNQ ~ $18k; B per-strat $1k ~ $12.3k.

## Recommendation (SEE CORRECTED BLOCK AT TOP — this is superseded)
- Challenge (P1+8%/P2+5%, RPTI-exempt): full 4-way, ~5 MNQ uniform -> ~69% pass in ~2.7 mo.
- Funded: **RV@1 + OD/B2/FB@2** -> ~$10k/yr at ~3% blow. RV is the TIGHTEST leg (worst float $16k),
  so keep RV small and size OD/B2/FB up — NOT "RV/FB large" (that earlier line was the OD-MAE error).

## CAVEATS
1. Everything rides on the combined's edge persisting live (historical bootstrap).
2. Realized-PnL DLL; true intraday floating MAE is stricter at flat high-MNQ (per-strat robust).
3. Fee ~$500 + pass rates 79%/86% are assumptions; payout = bi-weekly 80% (FP also has
   on-demand 90% / monthly 100%, which shift funded EV). "$2k exposure" read as open-risk cap.
4. OD martingale (if present in the backtest log) amplifies tails; per-strat sizing absorbs it.

Files: fundingpips_2step_100k_mc.py (flat MNQ, acct A), fundingpips_2step_variants.py (per-strat
+ OD-drop, acct A), fundingpips_2step_configurable.py (any rule-set; currently acct B),
fundingpips_funded_ev.py (both accounts) + matching CSVs.
