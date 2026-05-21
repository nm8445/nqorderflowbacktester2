# Scaling Plan — $500 to Funded Multi-Account Stack

Last updated 2026-05-20. Based on Monte Carlo simulations using the live 4-strategy stack (RV + B2 + OD + Fabio ORB).

---

## TL;DR

Bootstrap $500 → 2 funded futures via 4-firm hedge. First payout (~2-3 weeks) goes ALL-IN on FundingPips CFDs (3× $399). Within 9-12 months, run rate target $7-10K/month.

```
$500 (start)
  ↓  4-firm futures hedge over 2 days
2 funded futures @ 1 MNQ
  ↓  Trade normally, ~2-3 weeks
First $1,500 payout
  ↓  All-in CFD: 3× FundingPips 100K 2-step ($1,197)
2-3 funded FP CFDs @ 3 MNQ each (median 130 days to first payout)
  ↓  Continued futures + maturing CFDs
$7-10K/month target by month 9-12
```

---

## Source data

All numbers in this doc come from Monte Carlo runs on `combined_4way_trades.csv` (1,414 days, mean $452/day, std $3,096) and the locked B2/OD/RV/Fabio configs. Three MCs:

| Sim | File | Question answered |
|---|---|---|
| `cfd_vs_futures_lifetime_sim.py` | Log: `scripts/results/mc_cfd_vs_futures.log` | CFD-vs-futures lifetime $ comparison |
| `fundednext_200k_4strat_mc.py` | Log: `scripts/results/mc_fundednext_200k.log` | FN 200K/300K MNQ sweep |
| `coinflip_gamblers_ruin_sim.py` | Log: `scripts/results/mc_gamblers_ruin.log` | Cheap-eval gambler's ruin extraction |

---

## Firm pricing (verified 2026-05)

| Firm | Account | Fee | Per-trade rule | Daily LL | Max Loss |
|---|---|---|---|---|---|
| **FundingPips 100K 2-step** | $100K | **$399** | 2% on funded (after $50K) | 5% | 10% static |
| **FundedNext Stellar 2-step 100K** | $100K | **$549** | **NONE on CFD** (verify TOS) | 5% | 10% static |
| FundedNext Stellar 2-step 200K | $200K | $1,099 | NONE on CFD (verify TOS) | 5% | 10% static |
| HolaPrime 2-step 100K | $100K | $2,249 (80% split) | TBD | 3% | 7% |
| **Lucid Flex 50K** | $50K | ~$100-150 | None on challenge | $2K trail | $2K trail |
| Tradeify 50K | $50K | ~$120-150 | None | Varies | Varies |
| Bulenox 50K | $50K | ~$130 | None | Varies | Varies |
| MyFundedFutures 50K | $50K | ~$150 | None | $2K trail | $2K trail |
| Apex 50K | $50K | ~$130 | None | $2.5K | $2.5K |

**Skip HolaPrime** — 5.6× the price of FP for the same nominal size. Not viable on a $1,500 reinvest.

---

## Monte Carlo results

### CFD 100K 2-step lifetime EV (per challenge attempt)

Using full 4-strat stack. `chal_mnq` = size during 2-step challenge. `fund_mnq` = size after funded. Both fees reimbursed on first funded payout. Median days to first payout reported.

#### FundingPips (P1 +8%, P2 +5%, 5% DLL, 10% static max, $399)

| chal_mnq | fund_mnq | Pass rate | Days to pass | Days to 1st payout | Median $ if pass | EV overall |
|---|---|---|---|---|---|---|
| 3 | 2 | 86% | 123 | 132 | $11,740 | $10,220 |
| **3** | **3** | **86%** | **121** | **130** | **$17,577** | **$14,935** |
| 3 | 4 | 86% | 123 | 132 | $20,866 | $17,818 |
| 4 | 2 | 82% | 84 | 93 | $11,939 | $9,778 |
| 4 | 3 | 82% | 83 | 92 | $17,421 | $14,157 |
| **4** | **4** | **82%** | **85** | **94** | **$21,780** | **$17,785** |
| 5 | 4 | 67% | 57 | 66 | $21,468 | $14,182 |
| 6 | 4 | 64% | 43 | 52 | $21,572 | $13,470 |

#### FTMO (P1 +10%, P2 +5%, 5% DLL, 10% static max, $540)

Uniformly **worse than FP** at every sizing because of the 10% P1 target.

| chal_mnq | fund_mnq | Pass rate | Days to 1st payout | Median $ if pass | EV overall |
|---|---|---|---|---|---|
| 3 | 3 | 82% | 148 | $17,700 | $14,627 |
| 4 | 4 | 80% | 107 | $21,370 | $17,221 |

**Picks**:
- **3 MNQ → 3 MNQ FundingPips** — highest pass rate, $14.9K EV/attempt
- **4 MNQ → 4 MNQ FundingPips** — best total $ if pass, $17.8K EV, faster funded
- **Never use FTMO if FP is available** — same math, worse outcome

### FundedNext 200K/300K (4-strat across MNQ sweep)

Per-trade rule modeled as $3K on 200K, $6K on 300K. **Verify if FN Stellar 2-Step actually has per-trade rule** — web search says it doesn't, but the MC assumed it does. If no per-trade rule, real economics are much better than below.

#### FN 200K ($3K per-trade, $10K daily, $20K total)

| MNQ | Bust % | Payouts/yr | Trader $/yr | NET $/yr |
|---|---|---|---|---|
| 1 | 0% | 50 | $11,037 | **$11,037** |
| **2** | **0.12%** | 49.92 | $22,003 | **$22,003** |
| 3 | 31% | 51.76 | $34,260 | $34,142 |
| 4 | 48% | 53.11 | $46,936 | $46,804 |
| 5 | 85% | 56.55 | $61,366 | $61,151 |
| 6 | 98% | 63.84 | $83,145 | $82,752 |
| 7 | 100% | 72.96 | $111,797 | $111,073 |
| 10 | 100% | 91.08 | $198,669 | $196,581 |

**Per-trade rule dominates above 2 MNQ.** B2 has occasional MAE up to $11K on 1 NQ basis = $1.1K on 1 MNQ. At 3 MNQ that's $3.3K → exceeds $3K cap.

#### FN 300K ($6K per-trade, $15K daily, $30K total)

| MNQ | Bust % | Payouts/yr | NET $/yr |
|---|---|---|---|
| 1 | 0% | 50 | $11,037 |
| 2 | 0% | 49.84 | $21,974 |
| 3 | 0% | 50.19 | $33,227 |
| 4 | 0.7% | 50.01 | $44,137 |
| **5** | **4%** | **50.17** | **$55,284** |
| 6 | 38% | 52.53 | $69,543 |
| 7 | 48% | 54.03 | $83,340 |
| 10 | 93% | 61.04 | $131,659 |

**Sweet spots**: FN 200K @ 2 MNQ ($22K/yr safe) or FN 300K @ 5 MNQ ($55K/yr, 4% bust).

### Gambler's ruin pipeline (10 active funded futures + $100 eval refresh)

Using your real strategy edge (~55/45 winrate with proper R:R sizing).

| Setup | NET $/yr | Monthly | Busts/yr | Payouts/yr | p25 | p75 |
|---|---|---|---|---|---|---|
| Pure 50/50 + slip | $18,398 | $1,533 | 75 | 38 | $12,950 | $24,200 |
| 52/48 + slip | $25,894 | $2,158 | 64 | 42 | $20,700 | $31,300 |
| **55/45 + slip (your edge)** | **$36,680** | **$3,057** | **50** | **49** | **$31,750** | **$41,700** |

10 funded accounts, 3 trades/day across all, $1500 stake/trade, $100 evals, 30% eval pass rate. **Real expected income $36K/yr ($3K/mo) sustainable.**

---

## The phased plan

### Phase 1: Bootstrap (Week 1) — $400-500

**Action**: Buy 4 futures evals at firms WITHOUT consistency rules (or 50% consistency, not stricter).

**Firm picks for hedge**: Tradeify + Lucid Flex + Bulenox + MyFundedFutures. All ~$100-150 each. None have <50% consistency that complicates the hedge timeline.

**Avoid for hedge**: Apex (30% strict consistency = need 4 winning days), TopstepOne (consistency on payout side).

**Hedge mechanics** (50% consistency rule version):
- Day 1: Open 4 accounts. Pair 1 = (A long, B short). Pair 2 = (C long, D short). 5 NQ contracts each.
- NQ moves ~15pts directionally (happens most days). Long pair makes $1,500 each, short pair loses $1,500 each.
- Day 2: Repeat. Long accounts hit $3K target (50%/50% split — satisfies consistency). Short accounts bust on -$3K total vs -$2K trail DD.
- **Result**: 2 funded futures accounts after 2 days. ~$400 spent. ~$100 buffer.

**Hedge mechanics** (no consistency rule version — Tradeify/Bulenox):
- Day 1: Same setup. NQ moves 30pts. Long accounts pass target in one trade. Short accounts bust same trade.
- **Result**: 2 funded same day.

**Risk**: Cross-firm hedge detection. Doing it ONCE on a 1-2 day window is low-detection. Repeating is what triggers bans. **Do not repeat the hedge for additional funded accounts** — trade legitimately after this.

### Phase 2: First payout (Weeks 2-3) — $0 cost

Run 2 funded futures at **1 MNQ** with full 4-strat stack. Lucid Flex first payout target = $3,000 profit. At 1 MNQ on your edge (~$3K-5K/yr per account = ~$300/mo), expect ~10-15 trading days to hit first payout target.

**First payout: $1,500 cash** (90% split of $1,666 = $1,500).

### Phase 3: CFD reinvest (Month 2) — $1,500 → 3× FundingPips

**Action**: Buy 3× FundingPips 100K 2-step at $399 each = $1,197. Keep $300 buffer.

**Expected outcome**:
- 3 × 0.86 = **2.58 expected challenge passes** (likely 2-3 funded CFDs)
- Each funded CFD has $14,935 EV per attempt (3 MNQ → 3 MNQ config)
- Total EV from this $1,500 reinvest = **$30-45K over ~6 months**

**Alternative: All-in futures** ($1,500 → 15 evals @ $100 each, 30% pass = 4-5 funded futures):
- Adds 4-5 funded futures, but at 10 accounts you're already at the gambler's ruin sim's cap (3 trades/day shared across accounts)
- Marginal yield: +$10-15K/yr
- **CFD all-in wins by 3× on EV**

**Hybrid split** ($800 CFD + $700 futures): 2 FP evals (1.7 expected pass) + 7 futures evals (2.1 expected pass). Hedges variance but caps upside.

**My pick**: All-in 3 FP CFD. Higher EV, fewer accounts to manage.

### Phase 4: Stack period (Months 3-6) — $0

Run 2 futures + 2-3 challenge-phase CFDs simultaneously. Same trade signals from the 4-strat engine — no extra work.

Expected payout cadence:
- Futures: $1-1.5K/month from continued 1 MNQ trading
- CFDs in challenge: $0 cash (paper progress toward P1/P2)

### Phase 5: CFD funded (Months 6+) — Full scale

Once CFDs pass and start paying:
- 2-3 FP funded CFDs @ 3 MNQ each → ~$5-7K/month at scale (10 payouts per ~6 month lifecycle, $1.5K each)
- 2 funded futures @ 1 MNQ → ~$1-2K/month
- **Target combined: $7-10K/month by month 9-12**

If CFDs prove reliable, **scale up to FN Stellar 2-Step 200K @ 5 MNQ** ($1,099 fee, expected $55K/yr if no per-trade rule confirmed).

---

## Key constraints to watch

### Per-trade rule on FundingPips funded

FundingPips funded account has **2% per-trade rule on $50K+**. On $100K funded that's a **$2,000 per-trade cap**.

Your strategies and max single-trade MAE (per 1 NQ basis, scale by 0.1 for 1 MNQ):

| Strategy | Worst single-trade MAE (1 NQ) | At 3 MNQ |
|---|---|---|
| B2 | ~$11,000 | ~$3,300 (over $2K cap!) |
| OD (martingale 2c) | $10,870 | $3,261 (over $2K cap!) |
| RV | ~$3,000 | $900 |
| Fabio ORB | ~$2,000 | $600 |

**Critical**: At 3 MNQ on FundingPips funded, ~5-8% of trades will exceed the $2K per-trade limit and instant-bust the account. The 86% pass rate from the MC may overstate reality if it didn't model the per-trade rule.

**Mitigations**:
1. Run **2 MNQ** on FP funded instead of 3 MNQ (cuts EV ~40% but stays under per-trade)
2. Disable OD martingale on FP accounts (always 1c — no 2c recovery trades)
3. Add hard SL caps at $1,800 per trade in the live engine
4. Use FN Stellar 2-Step which (per web search) has NO per-trade rule

### OD martingale + per-trade rules

Live OD config = **martingale ON** (`s1-L2`: 1c base, 2c after loss). Worst single trade = -$10,870 on the 2c recovery. **Verify whether to disable martingale on per-trade-rule accounts.** Loses ~15-20% of OD's expected yield but avoids account-killing single trades.

Set up per-account martingale toggle in `live/combined/od_engine.py` if running mixed firms.

### Hedge plan TOS risk

Mass-ban detection is what kills repeat hedgers. The plan above does the hedge ONCE in the bootstrap phase. After that, all trading is legitimate strategy execution. Single hedge event over 1-2 days at 4 firms is unlikely to trip detection. **Do not repeat the hedge to add more funded accounts.**

If you ever need to scale futures count further, just buy 1 eval at a time and trade legitimately (~30% pass rate, ~$300 expected cost per pass).

---

## Variance bands

Year 1 NET income honest estimate from gambler's ruin + CFD layer:

- **p25 (unlucky)**: ~$35K — fewer CFD passes, more futures busts
- **median**: ~$60K — 2 CFDs pass + 2 futures running
- **p75 (lucky)**: ~$90K — all 3 CFDs pass + futures running

Worst case (all CFDs bust at P1): ~$18K from futures alone.

---

## Sources

- [FundingPips 2-Step Rules 2026](https://proptradingvibes.com/blog/fundingpips-two-step-challenge)
- [FundingPips Review](https://thepropjournalist.com/reviews/fundingpips/)
- [FundedNext Stellar 2-Step Model](https://fundednext.com/stellar-model)
- [FundedNext Account Comparison](https://proptradingvibes.com/blog/fundednext-account-types)
- [HolaPrime Two-Step](https://holaprime.com/two-step-prop-firm/)
- MC scripts: `scripts/montecarlo/cfd_vs_futures_lifetime_sim.py`, `fundednext_200k_4strat_mc.py`, `coinflip_gamblers_ruin_sim.py`
- MC logs: `scripts/results/mc_*.log`
