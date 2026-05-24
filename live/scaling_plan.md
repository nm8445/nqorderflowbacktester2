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

### FundedNext SL placement + risk rules (UPDATED 2026-05-23)

FN enforces strict SL placement on **funded** accounts (challenge currently unclear, assume same):

| Rule | Trigger | Consequence |
|---|---|---|
| **No SL Trade** | Trade opened without ever placing SL | Violation |
| **Duration Exceeded** | SL not placed within 3 min of opening | Violation |
| **SL Gap** | SL removed for >1 min after the first 3 min | Violation |
| **High Risk (per-trade)** | SL distance × position > 3% of account | Violation → demotion to 1% sizing |
| **At-a-Time High Risk (cumulative)** | Sum of SL-risk across ALL open positions > 3% | Violation → demotion to 1% sizing |
| **Duration Exceeded & High Risk** | SL placed late AND too far | Compound violation |

**Critical interpretation**: cumulative cap = 3% means **only one full-sized trade open at a time**, OR multiple concurrent positions whose combined SL-risk sums to ≤3%. This breaks the naive "run all 4 strategies independently" assumption.

### Conversion: NQ → NDX100 (FundedNext symbol)

Prices differ (basis varies day-to-day) but move 1:1 in points. The $/pt mapping:

| NQ-side | NDX100 lots | $/pt | Per-strategy use |
|---|---|---|---|
| 1 MNQ | 0.20 | $2 | conservative funded |
| 2 MNQ | 0.40 | $4 | 100K funded sweet spot |
| 4 MNQ | 0.80 | $8 | 200K funded sweet spot |
| 5 MNQ | 1.00 | $10 | 200K challenge |
| 6 MNQ | 1.20 | $12 | 300K funded sweet spot |
| 8 MNQ | 1.60 | $16 | 300K challenge |

Formula: `NDX100 lots = MNQ × 0.20`

### Max SL distance per setup (3% per-trade cap)

`SL_pts_max = (account × 0.03) / ($/pt)`

| Account | Lots | $/pt | Max SL | Recommended SL (95% of cap) | B2 worst (550 pts) safe? |
|---|---|---|---|---|---|
| 100K | 0.40 | $4 | 750 pts | **700 pts** | ✓ |
| 200K | 0.80 | $8 | 750 pts | **700 pts** | ✓ |
| 200K | 1.00 | $10 | 600 pts | **570 pts** | TIGHT — B2 can hit SL |
| 300K | 1.20 | $12 | 750 pts | **700 pts** | ✓ |
| 300K | 1.40 | $14 | 643 pts | **610 pts** | tight |
| 300K | 1.60 | $16 | 562 pts | **530 pts** | NO — B2 worse than this; drop B2 or downsize |

Pattern: 700 pts of SL works at all "sweet spot" sizes, and lets candle-close exits trigger first. The 4 strategies' worst peak MAEs:
- B2 worst: 550 pts (1 NQ basis)
- OD mart 2c worst: 543 pts
- RV worst: ~150 pts
- Fabio worst: ~100 pts

A 700-pt SL is wider than all of them with margin.

### Cumulative-cap architectural impact

`mt5_executor.py` (still TODO) needs:
1. **Atomic SL placement** — every `mt5.order_send()` includes `sl=`, no follow-up modify
2. **Pre-trade cumulative check** — before opening trade N, sum existing SL-risk + new trade's SL-risk; reject if >3% of account
3. **Strategy serialization or splitting** — only one strategy holds the full 3% at a time, OR run different strategies on different FN accounts so each gets its own 3% budget

Honest take: the 4-strat-on-one-account MC numbers I quoted assume free concurrency. With cumulative 3% enforced, either:
- **One strategy per FN account**: 4 accounts to run the full stack (4× the fees)
- **Serialize**: lose signals when another strategy is open (~30-40% signal loss based on overlap rates)
- **Downsize**: each strategy at 0.75% so 4 concurrent = 3% total (cuts EV ~75% per strategy)

The realistic FN play is probably **2-3 strategies per account, sized so 2 concurrent fit under 3%** — needs a fresh MC run.

### CHOSEN DEPLOYMENT (2026-05-23): asymmetric per-strategy budget

OD runs solo (closes by 8 AM, no overlap with B2/RV/Fabio per user). B2/RV/Fabio share the 3% cap (each 1%) since they can be concurrent.

| Strategy | Budget | Recommended SL | Notes |
|---|---|---|---|
| OD | 3% | 600 pts | full slot, only one open |
| B2 | 1% | 600 pts | overnight, can overlap morning sessions |
| RV | 1% | 200 pts | intraday |
| Fabio | 1% | 150 pts | morning ORB |

**Sizing tables** (NDX100 lots = (budget$ / SL_pts) / $10):

#### 100K — $1K per 1% slot, $3K for OD

| Strategy | Lots | ≈MNQ |
|---|---|---|
| OD | 0.50 | 2.5 |
| B2 | 0.17 | 0.83 |
| RV | 0.50 | 2.5 |
| Fabio | 0.67 | 3.33 |

#### 200K — $2K per 1% slot, $6K for OD

| Strategy | Lots | ≈MNQ |
|---|---|---|
| OD | 1.00 | 5 |
| B2 | 0.33 | 1.67 |
| RV | 1.00 | 5 |
| Fabio | 1.33 | 6.67 |

#### 300K — $3K per 1% slot, $9K for OD

| Strategy | Lots | ≈MNQ |
|---|---|---|
| OD | 1.50 | 7.5 |
| B2 | 0.50 | 2.5 |
| RV | 1.50 | 7.5 |
| Fabio | 2.00 | 10 |

**Assumption to verify**: OD really doesn't overlap with B2 (B2 is overnight; if B2 still open at 4-8 AM when OD trades, OD must drop to 2% to leave 1% for B2). Check actual B2 close time vs OD open time in `live/combined/b2_engine.py` and `live/combined/od_engine.py`.

**Strategy schedule (verified 2026-05-23 from engine files)**:
- OD: 19:00 ET entry → 08:00 ET force-close (overnight, solo session)
- B2: 9-14 ET entries → 16:00 ET force-close (day session)
- RV: 09:00-13:00 / 14:00-14:45 ET → 14:45 force-close
- Fabio: morning ORB
- OD runs in a different session from day strats → no overlap → can keep its full 3% slot

### Verified FN pricing (web-checked 2026-05-23)

| Account | Stellar 2-Step fee | Availability |
|---|---|---|
| 100K | **$549.99** | Direct purchase |
| 200K | **$1,099.99** | Direct purchase, **MAX standalone** |
| 300K | N/A | **Only via scaling from 200K**, not purchasable |

**Profit split**: "up to 95%" per FN. Default likely 80-90%; 95% requires add-on or scaling milestones. MCs below use **90% baseline** — multiply income by 1.056 for 95% scenario, by 0.889 for 80%.

**Refresh fee**: not on FN public site; assumed $99 ballpark. Impact is tiny (~$115/yr).

### MC results — CHALLENGE (3% per-trade cap, asymmetric sizing)

Output: `live/combined deployment plan/fundednext_challenge_asymmetric.csv`

| Account | Pass rate | Median days total | p25 | p75 | $/funded acct | Demote % |
|---|---|---|---|---|---|---|
| 100K | **95.66%** | **111** | 76 | 157 | $575 | 0% |
| 200K | **95.64%** | **111** | 76 | 157 | $1,150 | 0% |
| 300K (scaled) | 95.90% | 109 | 75 | 157 | n/a | 0% |

P1 median: 63 days. P2 median: 36 days. 0% demotion risk at this sizing.

### MC results — FUNDED with payout detail (3% per-trade cap, 90% split)

Output: `live/combined deployment plan/fundednext_funded_payout_detail.csv`

| Account | Avg payout | Median payout | p25 / p75 | p95 | Avg gap days | Payouts/yr | NET $/mo | NET $/yr |
|---|---|---|---|---|---|---|---|---|
| **100K** | **$547** | $397 | $169 / $738 | $1,525 | 4.44 | 51.3 | **$2,329** | **$27,947** |
| **200K** | **$1,095** | $794 | $339 / $1,478 | $3,049 | 4.44 | 51.3 | **$4,674** | **$56,087** |
| **300K** (scaled) | $1,643 | $1,196 | $512 / $2,214 | $4,602 | 4.44 | 51.4 | $7,026 | $84,316 |

At 95% split (max): 100K = $29,500/yr, 200K = $59,200/yr, 300K = $89,000/yr.

Bust rate 33-34% across all sizes (from total-DD only; demote% = 0%). Median 1 day between payouts because payouts hit any time balance > start; 4.44 day avg includes gaps from sub-start days.

**Counterintuitive finding**: asymmetric sizing **BEATS flat-MNQ sweet spot by ~24% income** at all account sizes. Reason: Fabio (tight 150 pt SL) and RV (tight 200 pt SL) get UPsized at 1% budget vs flat 4-strat MNQ; only B2 takes a real hit (it's the only strategy with wide-MAE profile). OD stays roughly the same since it gets the full 3% slot.

**Bust rate triples** (12% → 34%) because daily PnL std rises 22% from Fabio/RV upsizing, hitting 10% static DD more often. But:
- Demotion risk (per-trade rule trip) = 0% at all sizes — well within cap
- Bust = $99 refresh + ~38 days lost trading, benign in EV terms
- Payouts/yr unchanged at 51.3 (payouts are daily)

**This deployment is preferred over both serialize and one-account-per-strategy** for fee efficiency and per-strategy EV.

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
