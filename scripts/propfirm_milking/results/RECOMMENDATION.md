# Prop-Firm Milking — Phase 1+4 Results & Recommendation

50K challenge: reach +$3k before $2k trailing EOD DD, 50% consistency (best day <= 50%
of total profit). Funded: $2k trailing EOD, 5 winning days (>=$150) -> withdraw 50% of
profit (balance drops, floor stays). Sizing ~$1k risk/trade via MNQ (cap 40), costs
$2/RT/MNQ + 2 ticks/side slippage. SL-first tick-style first-touch on 1-min bars.

## Recommended CHALLENGE config: RV + FB + MEC (drop OD and B2)

| Strat | atr_len | sl_mult | tp_mult | RR | WR (IS/OOS) | tagged | n |
|-------|---------|---------|---------|------|-------------|--------|------|
| RV    | 10      | 1.50    | 2.00    | 1.33 | 48.9 (49.9/47.5) | 0.76 | 799 |
| FB    | 28      | 2.25    | 3.00    | 1.33 | 48.4 (47.9/49.2) | 0.83 | 628 |
| MEC   | 10      | 2.00    | 2.75    | 1.375| 48.2 (47.7/49.0) | 0.78 | 1386 |

- OD dropped: 0 discipline-compatible configs (overnight ATR too wide -> drift, never tags).
- B2 dropped: only tags at RR~1.8 / WR~42% -> lowers pass rate (more 2-loss blows) more than
  its extra signals help. MC confirms it hurts net.

## Full-cycle Monte Carlo (4000 sims, 30 challenge accts, copy-2 discipline)

Funded = TWO-PHASE (user's actual approach): (1) reach +$3k on the funded acct via the
EVAL discipline ($1k risk); (2) then 1 MNQ combined for 5 winning days -> withdraw 50%
($1500); repeat until bust. Floor stays on withdrawal.

| Config        | P(pass) | #funded/30 | reach-$3k rate | $/funded acct | E[net $/cycle] | P(net>0) | cycle mult |
|---------------|---------|-----------|----------------|---------------|----------------|----------|------------|
| RV+FB+MEC     | 0.313   | 9.4       | 0.273          | $630          | $2,904         | 0.726    | 2.0x |
| RV+FB+MEC+B2  | 0.288   | 8.6       | 0.250          | $574          | $1,959         | 0.643    | 1.65x |

Cycle = buy 30 challenges ($3,000) -> ~9 pass -> ~27% of those reach a payout -> ~$5.9k
withdrawn -> ~$2.9k net (~2x). The funded reach-$3k bust (73% fail) + floor-stays
(~1 payout then death) are the dominant drags.

## OOS robustness (challenge bracket discipline)

- Challenge P(pass): all 0.311 | IS 0.306 | **OOS 0.329** (holds / slightly better OOS).
- Funded reach-$3k uses the same eval discipline -> same OOS-robustness.

## Trading discipline (challenge, per account/day)

Take 1 signal -> WIN stops the day (+~$1.5k); LOSS -> take 1 recovery signal; 2nd LOSS ->
~-$2k -> account blows. Max 2 trades/day. Each signal mirrored to a pair of accounts; losing
pairs free the next signal for the next fresh pair. Max day +$1.5k = 50% of the 2-day pass path.

## Funded reach-$3k risk sweep (results/funded_risk_sweep.csv)

Sweet spot ~$900/trade (vs $1000): reach-rate 31% (was 27%), net/cycle ~$3,697 (2.23x, was 1.96x).
Non-monotonic: $1000 instant-busts on a 2-loss day; <$800 reaches +$3k too slowly (more trailing-DD
exposure). Use ~$900 reach-phase risk. Deeper lever is higher WR (Phase 2) -> cuts blow rate (1-p)^2.

## Funded milking reality

Per funded account: reach +$3k on the eval discipline FIRST (~27% succeed, ~73% bust trying),
then 1 MNQ for 5 winning days -> withdraw 50% ($1500). Floor-stays means after that withdrawal
the cushion is ~$500, so most accounts give ~1 payout then die (0.36 payouts/funded acct).
Avg $/funded acct = $630 (includes the ~73% that get $0).

## Phase 2 — entry-param re-sweep (results/FB_entry_sweep.csv)

- FB: relaxing N_confirm 4 -> 3 (delta 300, bracket atr14/sl2.5/tp3.25) lifts WR to 50.6%
  (IS 48.4 / OOS 53.8) with more signals (753 vs 709). MC: net/cycle $2,890 -> $3,513 (+22%,
  1.96x -> 2.17x). OOS-robust. RECOMMENDED FB upgrade (requires regenerating FB entries from
  the 5-min delta parquet, not the locked log).
- IMPORTANT debunk: the prior sweep's "N=1 -> 57% WR" was an ORB-RR1.0 EXIT artifact (tight TP
  inflates WR). Under the real RR~1.3 ATR bracket, N=1 WR is only 46.7% (worse than N=4). This
  is exactly why entry changes must be re-validated with the actual bracket.
- RV: prior gate sweep (rv_new_gate_sweep.csv) hinted lowz~1.2 gate -> ~2x signals + ~+1.5% WR,
  but its generator script was deleted and (given the FB N=1 mirage) the gain may be an exit
  artifact. NOT integrated -- would need a from-scratch gate rebuild + ATR-bracket re-validation.

## Stacked best estimate

RV + FB(N=3) + MEC, funded reach-phase risk ~$900: ~2.2-2.3x net/cycle (~$3.5-4k on $3k spend),
P(pass) ~31-32%, P(net>0) ~73%.

## ALTERNATIVE Config B — $1500/trade, 1:1 RR (beat Config A in-framework)

`scripts/montecarlo/eval_4strat_vs_coinflip.py`. Bet **$1500/trade, 1:1** (win +1500 / lose -1500),
**1 trade/day**, reach +$3000 before -$2000. Original coinflip ruin sim (static DD, NO consistency):
**50.0% pass at 50% WR, 46.4% at 48% WR** (the "46-50%" config).

Re-run through THIS session's full framework (trailing $2k EOD DD + 50% consistency + two-phase
funded milking), 1 trade/day:
| WR  | challenge pass | funded reach-$3k | net/cycle | P(net>0) | mult |
|-----|----------------|------------------|-----------|----------|------|
| 50% | 44.6%          | 43.3%            | $10,477   | 99%      | 4.5x |
| 48% | 41.0%          | 40.1%            | $8,418    | 97%      | 3.8x |
vs Config A (RV+FB+MEC, $1k/1.5RR) ~$3,500/cycle (2.2x). **B wins ~2.4x.**

WHY B beats A: B takes ONE trade/day with NO recovery trade, so max daily loss = $1500 (survives);
need 2 SEPARATE losing days to blow. A's recovery trade -> a 2-loss DAY = -$2000 = instant blow
(~27% per active day). That one structural difference ~doubles funded reach-rate (40% vs 28%) and
net/cycle. The recovery trade is the culprit -> prefer 1-trade/day 1:1 $1500.

CAVEAT: the 48-50% WR here is a COINFLIP (independent trades, random direction, no edge needed).
Real signals traded at 1:1 brackets would likely have WR>=50% (tighter TP hits more -> better) but
carry serial loss clustering (consecutive correlated losses -> worse). True number is between; not
yet run with the real RV/FB/MEC signals at 1:1 / $1500. See [[project_propfirm_milking]] memory.

## Scaling (30+ accounts) & Annual P&L  (annual_pnl.py, RV+FB[N=3]+MEC, ~$900 reach)

Per-eval economics: each eval ~= P(pass) x $/funded - $100 = 0.316 x $687 - 100 ~= +$117 net.
So net/cycle scales ~linearly with #evals. To net ~$10k/cycle -> ~85 evals (~$8,500 spend).

Per-cycle (joint MC):
| Batch     | spend  | net/cycle mean / median | P(net>0) | recoup | full span |
|-----------|--------|-------------------------|----------|--------|-----------|
| 30 evals  | $3,000 | $4,214 / $3,570         | 80%      | ~32 td | ~47 td |
| 85 evals  | $8,500 | $12,227 / $11,118       | 88%      | ~31 td | ~54 td |

Timing: first payout ~20 td (~1 mo) | recoup the spend ~29 td (~1.4 mo / "6 weeks") |
full cycle ~46 td (~2.2 mo). NOT monthly; ~5 cycles/yr sequential.

Annual P&L distribution (20k simulated years, SEQUENTIAL = one batch at a time):
| Batch    | cycles/yr | median /yr | p10 / p90        | mean /yr | P(losing yr) |
|----------|-----------|------------|------------------|----------|--------------|
| 30 evals | ~5.8      | $24,207    | $10,579 / $38,598| $24,420  | ~1%   |
| 85 evals | ~5.1      | $61,926    | $33,225 / $92,005| $62,437  | ~0.1% |

Aggressive cadence (redeploy at recoup ~6wk, overlapping, ~2x capital) widens variance:
85 evals -> median ~$62k but p90 ~$146k AND P(losing yr) ~14% (vs 0.1% sequential).
"$10k every 6 weeks" literally (~$87k/yr) needs ~120 evals OR overlapping batches.
~22% of INDIVIDUAL cycles are net losers; the year smooths green via cycle count.

## CAVEATS

1. Funded net (~80% of the $22k) depends on the EXISTING combined's edge persisting. Funded
   sim IID-bootstraps historical days. OOS held, but live can diverge (cf. OD 5/14 live event).
2. Funded log is 3-way (OD/RV/B2) and may include OD martingale — verify it matches the
   intended "marti OFF, 1 MNQ" milking engine. (Challenge config uses neither OD nor that log,
   so the robust challenge result is unaffected.)
3. Phase-1 brackets re-bracket the LOCKED entries; flat-gating shift under faster exits is
   ignored (slightly conservative on RV/B2 signal counts). Phase 2 (entry re-sweep) + Phase 3
   (raw-tick first-touch re-validation) are not yet run.

Files: results/{RV,FB,MEC,B2}_bracket_sweep.csv, results/portfolio_report.csv,
results/funded_risk_sweep.csv, results/FB_entry_sweep.csv, configs/challenge_config.json.
Scripts: sweep_exits.py, sweep_funded_risk.py, fb_entry_sweep.py, mc_portfolio.py, annual_pnl.py.
