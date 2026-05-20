# Go-Live Progress — Combined 3-Strategy Deployment

**Last updated:** 2026-05-15

This file captures where we are in converting the OD + RV + B2 combined strategy from backtest to live trading on prop firm accounts. Pick up here next session.

---

## 1. Strategy configs (LOCKED)

All 3 strategies retain their backtest live configs from `live/combined deployment plan/plan.md` with ONE modification:

| Strategy | Status | Modifications for live |
|---|---|---|
| **RV (Rough Vol Orderflow)** | Live config unchanged | None |
| **B2 (OHI/OLO Range Gamma)** | Live config unchanged | None |
| **OD (Overnight Drift)** | Live config — **MARTINGALE DISABLED** | `use_martingale=False`, always 1c |

**Rationale for OD marti OFF**: catastrophic MAE events on 2c recovery trades (worst −$30,100 at 1 NQ basis, Nov 12 2024) blow trailing DD on prop accounts. With marti OFF, worst MAE drops to −$15,150 = ~−$1,500 at 1 MNQ — fits inside $2K trailing floor with $500 cushion.

---

## 2. Payout strategy decision

**Recommended: Max-cap @ $3K trigger**

- Withdraw $2,000 gross whenever cycle profit ≥ $3,000 AND 5 winning days hit
- Trader receives $1,800 per payout (Lucid 90/10)
- After payout: balance drops $2K, leaves ~$1K cushion above locked floor
- Expected: ~$2,085/mo per 10-account copy-trade portfolio

**User's preferred variant: Max-cap @ $4K trigger** (more conservative)
- Wait until balance hits $54K (cycle profit ≥ $4K) before pulling $2K
- Post-payout balance $52K with $2K cushion — sustainable
- Slower cash flow: ~$1,684/mo per 10-account portfolio

**Stagger A** ($1,500 first then $1,000 subsequent at $2K trigger) is dominated by max-cap on both EV and total cash. Don't use.

---

## 3. Multi-firm account allocation

| Firm | Max funded/identity | Recommended slots | Notes |
|---|---:|---:|---|
| Lucid Flex 50K | 5 | 5 | Best 90/10 split, no DLL |
| MFF Flex 50K | 5 | 5 | $5K payout cap, 80/20 split, no DLL |
| TopStep Express 50K | 5 | 5 | Trailing DLL adds bust pressure |
| **Total** | — | **15** | Hard cap without multi-identity (ToS violation) |

Expected monthly NET at 15 copy-traded accounts, 1 MNQ, max-cap @ $3K trigger: **~$3,200**
Realistic ceiling for solo trader on props: **$3-5K/month**

---

## 4a. Execution mode (UPDATED 2026-05-15)

Two modes supported by the architecture; choose based on income vs variance preference:

### Mode A: Copy-trade ALL 15 accounts (RECOMMENDED for $100K/yr target)
- All 15 accounts receive every signal from every strategy
- Higher mean income (~$8.3K/mo at 2 MNQ)
- Higher variance — some months $0, others $15K+
- 9-12% of months have a payout
- Sim verified: $99K/year at 2 MNQ, full stack, marti OFF

### Mode B: Per-strategy split (lower mean, smoother)
- 5 Lucid Flex: RV-only
- 5 MFF Flex: B2-only
- 5 TopStep: OD-only (marti OFF)
- Lower mean income (~$1.7K/mo at 2 MNQ)
- Smoother — 17% of months have payouts
- Better p25 (bad-year floor): $9K vs $0
- Each strategy group sync-busts together (not full portfolio)

**Default: Mode A unless you specifically prioritize income smoothness over mean.** Sim found mode B underperforms because single-strategy accounts lose the combined-stack benefit (slower qualifying day accumulation, weaker per-account economics).

## 4. Architecture decision: UNIFIED Python process

**NOT 3 separate terminals.** Reasons:
1. Hedge prevention requires cross-strategy visibility
2. One position per account at broker level
3. Easier replay/validation
4. Conflict resolver enforces no-hedging rule from plan.md

### Architecture diagram

```
Databento live tick feed
    ↓
Bar builders (20-min anchored midnight ET + 5-min)
    ↓
Three signal engines (RV, B2, OD) running in parallel
    ↓
Conflict resolver:
  - same direction signals: pass both
  - opposite direction to open position: BLOCK
  - log every decision (allowed/blocked + reason)
    ↓
Per-strategy position state (Python is source of truth)
    ↓
Net position computer (sums strategy intents)
    ↓
NT8 HTTP order router (sends NET position changes only)
    ↓
NT8 addon (DUMB EXECUTOR — needs rewrite, see below)
    ↓
Position reconciler (compares Python state to NT8 fills)
    ↓
State persistence (JSON to disk every state change)
```

---

## 5. Audit findings (Step 1 complete)

### Lookahead bias — ALL THREE STRATEGIES CLEAN
- **RV** (`scripts/rough vol orderflow/core.py`): clean. Caveat — orderflow features need full bar data before computing. Live: wait ~1-2s after bar close for Databento delta data.
- **B2** (`scripts/overnight range strat/scripts/lock_v2_k08_lock045_mart_fc_filtered.py`): clean. Caveat — needs BOTH 5-min bars (entry) AND 20-min bars (yellow trailing). Two aligned bar builders.
- **OD** (`scripts/overnight drift strategy/overnight_drift_strategy.py`): clean. Yellow ratchet uses prior-bar floor + current-bar raw. Green target uses ATR(t). All causal.

### NT8 addon (`nt8/NQOrderFlowSignalReceiver.cs`) — MUST BE REWRITTEN

**Critical limitation found:**
```csharp
private Order entryOrder = null;     // ONE entry, ONE stop, ONE target globally
private Order stopOrder = null;
private Order targetOrder = null;
private string entryState;           // single-position state machine
```

Tracks exactly ONE position. No strategy tagging. If RV sends BUY then B2 sends BUY 5min later, second one OVERWRITES the tracked orders → first position's stop/target tracking LOST.

**Rewrite approach: dumb executor** (Option B from analysis):
- Accept only simple MARKET BUY/SELL with quantity (no stop/target tracking in addon)
- Python computes net position change needed across all 3 strategies
- Python sends single market order to NT8 to bring actual position to net target
- All SL/TP logic stays in Python — Python fires NEW market orders when SL/TP hits
- Result: NT8 addon becomes simpler, Python is sole source of truth

---

## 6. Existing scaffold status

| File | Status | Action |
|---|---|---|
| `live/signal_engine.py` | Old VWAP strategy — wrong strategy | **REPLACE with new combined engine** |
| `live/live_trader.py` | Old VWAP orchestrator | Rewire for new coordinator pattern |
| `live/bar_builder.py` | Range bars + 5-min time bars | **Add 20-min anchored-to-midnight builder** |
| `live/signal_client.py` | HTTP client to NT8 — generic | Keep, but route through coordinator only |
| `live/account_manager.py` | Unknown — check | Audit before reuse |
| `live/warm_start.py` | Warm-start logic | Rewire for new strategy ATR/feature seeding |
| `nt8/NQOrderFlowSignalReceiver.cs` | Single-position only | **REWRITE as dumb executor** |

---

## 7. Critical risks for backtest→live divergence (ranked by pain)

1. **Bar boundary alignment** — OD needs 20-min anchored midnight ET. Off-by-1-minute breaks signals.
2. **ATR seed problem** — Pine RMA needs SMA of first 14 bars as seed. Warm-start from Databento history essential.
3. **process_orders_on_close semantics** — backtest fills at bar close; live signal fires at bar close, market order ~50-500ms later. Usually small slippage.
4. **Tick aggressor inference** — Databento MBP-1 has BBO + trades. Inferring aggressor side around simultaneous quote+trade events can be wrong. Matters for RV/B2 delta absorption.
5. **State recovery on restart** — if Python crashes mid-trade, must know OD has position from 19:00, qty=1, entry $21,500. State to disk every signal/fill.
6. **B2 martingale state** — last trade outcome + cooldown flag. Persist.
7. **Time zone weirdness** — ET around DST. Pin all timestamps to `pandas.Timestamp(..., tz='America/New_York')`.
8. **Position size mismatches** — Python "1 MNQ" must map to NT8 "MNQ contract" not "NQ". Test in sim.

---

## 8. Validation plan

**Per-day procedure:**
1. Live process logs `(timestamp, strategy, direction, price, qty, reason)` for every entry/exit
2. End of day: pull Databento ticks → resample → re-run backtest with same locked code
3. Diff live trade list vs backtest trade list
4. Match? Strategy validated. Diverge? Debug.

**Capital scope for first 30-50 trades:** Sim101 paper account in NT8 (no real capital).

---

## 9. Build sequence (Step 2 onward — TODO)

### Step 2: NT8 addon v2 (dumb executor)
- Accept simple MARKET BUY/SELL with quantity
- Return fills via callback or HTTP status poll
- Drop single-position state tracking
- ~1-2 hours of C# work

### Step 3: Python coordinator skeleton
File: `live/coordinator.py`
- Per-strategy position state (dict of strategy → Position obj)
- Net position computer
- Conflict resolver (block opposite-direction concurrent intent)
- HTTP client to send net-position-change orders to NT8
- State persistence (JSON to disk every state change)

### Step 4: Bar builders v2
File: `live/bar_builder_v2.py`
- Tick → 5-min bars (for RV, B2 signals)
- Tick → 20-min anchored-midnight-ET bars (for OD + B2 trailing)
- Both bar streams broadcast to consumers

### Step 5: Per-strategy engine stubs
- `live/od_engine.py` — receives 20-min bars, emits long signals at 19:00 ET, manages yellow/green/force-close
- `live/rv_engine.py` — receives 5-min bars + orderflow data, emits long/short signals with ATR-based SL/TP
- `live/b2_engine.py` — receives 5-min and 20-min bars, emits signals with ratchet trailing

### Step 6: Sim101 paper trading
- 30+ trades minimum
- Daily backtest replay validation
- Compare to historical performance

### Step 7: Deploy to ONE prop account
- Smallest viable: 1 Lucid Flex 50K at 1 MNQ
- Run for 2-4 weeks, monitor performance
- Then scale to 5 Lucid + 5 MFF + 5 TopStep

---

## 10. Open questions for next session

1. NT8 — does it support running multiple addon instances per identity? Affects whether option A (multi-instance) is viable as fallback.
2. Kill switch design — what's the manual shutdown procedure if data feed lags or NT8 disconnects?
3. Eval pass strategy via gambler's ruin — specific risk per trade for fastest pass? Earlier analysis suggested ~$1,500 stake.
4. Conflict resolver detailed spec — what exactly counts as "opposite direction"? If RV is +1 and B2 wants −1, block B2. But what if RV exits and B2 enters −1 on the same bar? Order of operations matters.
5. Data feed redundancy — single Databento connection or failover?
6. B2 5-min vs 20-min bar synchronization — what happens if 5-min bar closes mid-20-min bar? Need precise semantics.

---

## 11. Realistic income expectations (key numbers — UPDATED 2026-05-15)

### Without eval refresh (one-shot annual budget — what if accounts bust, slot dies)

| Setup | Annual NET (mean) | Monthly | Variance |
|---|---:|---:|---|
| 15 accts, **per-strategy split** (5 RV/5 B2/5 OD), 2 MNQ marti OFF | $20,255 | $1,688 | Lower |
| 15 accts, **copy-trade ALL** full stack, 2 MNQ marti OFF | **$31,666** | **$2,639** | Higher |
| 15 accts, copy-trade ALL, 3 MNQ marti OFF | $26,687 | $2,224 | High |
| 15 accts, copy-trade ALL, 1 MNQ marti OFF | $38,059 | $3,172 | Medium |

### With continuous eval refresh (sustainable operational model)

| Setup | Annual NET | Monthly | Notes |
|---|---:|---:|---|
| 15 accts, 1 MNQ, marti OFF, max-cap @ $3K trigger | ~$48K | ~$4K | Sustainable, low scrutiny |
| 15 accts, **2 MNQ**, marti OFF, max-cap @ $3K trigger | **~$99K** | **~$8.3K** | **$100K target ⭐, 12-24mo sustainable** |
| 15 accts, 3 MNQ, marti OFF, max-cap @ $3K trigger | ~$118K | ~$9.8K | Faster, 6-12mo sustainable |

### KEY FINDING: Copy-trade ALL beats per-strategy split on mean income

Earlier I recommended per-strategy split (5 RV / 5 B2 / 5 OD-only). The per-strategy split sim (`scripts/per_strategy_split_sim.py`) showed this is WORSE on mean income because each strategy in isolation loses the combined-stack benefits (trade frequency diversification, faster qualifying day accumulation).

**Copy-trade-all is the recommended mode for max income.** Per-strategy split is only worth it if smoothing month-to-month variance is more important than total cash.

| Tradeoff | Copy-trade ALL | Per-strategy split |
|---|---|---|
| Mean income | Higher | Lower (~35% less) |
| Months with payout | 9-12% (lumpy) | 17% (smoother) |
| Variance | Wild | Smoother |
| p25 year (bad case) | $0 | $9K |

### $100K/yr from 15× 50K accounts — CONFIRMED ACHIEVABLE

Path to $100K:
- 15 × 50K accounts across 3 firms (legal max)
- Copy-trade all 15 with full stack (RV + B2 + OD marti OFF)
- 2 MNQ per account
- Max-cap $2K payouts at $3K cycle profit trigger
- Continuous eval refresh (replace busted accounts immediately)
- Expected: $99K/year, ~$8.3K/mo NET after eval costs

**Operational pace**: ~100-120 evals/year, 70-80 bust replacements/year, ~$10K/yr in eval costs (already in NET).

**Days to first $2K payout** (1 MNQ, full stack):
- Fresh $50K account: median 41 days (74% reach payout, 26% bust first)
- $52K balance with locked floor: median 46 days (88% reach payout, 11% bust first)

**For $10K/mo NET goal**: prop firms cap at ~$8-10K/mo realistic ceiling at 2 MNQ aggressive. Personal capital for OD on top would push total well above $10K/mo target.

---

## 12. Files to know

- `live/combined deployment plan/plan.md` — locked combined strategy plan (canonical)
- `live/combined deployment plan/combined_trades_with_mae.csv` — augmented trade log with MAE column
- `live/overnight drift/live config overnight drift.md` — OD canonical config
- `live/rough vol orderflow/best live config/config.md` — RV canonical
- `live/overnight range gamma strat/best live config/config.md` — B2 canonical
- `scripts/mae_aware_propfirm_sim.py` — main prop firm Monte Carlo
- `scripts/max_cap_payout_sim.py` — payout strategy comparison
- `scripts/days_to_payout_sim.py` — time-to-payout analysis
- `scripts/coinflip_gamblers_ruin_sim.py` — coinflip baseline
- `scripts/staggered_start_vs_coinflip.py` — full strategy vs coinflip comparison

---

## 13. NEXT THING TO DO

**NT8 addon rewrite as dumb executor.** This is the bottleneck — until the addon supports multiple concurrent positions (or Python manages net position), we can't run all 3 strategies on one account safely.

Specifically:
1. Strip `entryOrder`/`stopOrder`/`targetOrder` single-state fields
2. Replace with simple "submit market order with qty and direction" handler
3. Add "current position" query endpoint (Python polls to reconcile)
4. Keep FLATTEN action (closes all positions)
5. Add per-order ID echo so Python can match fills

Once addon is rewritten, build Python coordinator + bar builders + engine stubs in parallel.
