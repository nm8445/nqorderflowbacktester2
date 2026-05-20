# Combined Deployment Plan — Locked

Three locked NQ strategies running on the **same account**:

1. **Rough Vol Orderflow** — intraday, no martingale (`live/rough vol orderflow/best live config/config.md`)
2. **B2 OHI/OLO Range Gamma** — intraday, FC-only martingale (`live/overnight range gamma strat/best live config/config.md`)
3. **Overnight Drift** — overnight, locked martingale (`live/overnight drift/live config overnight drift.md`)

Locked on: **2026-05-13**
Backtest range: **2020-12-01 → 2026-05-01** (~5.4 years)

---

## Combined performance

| Metric | Value |
|---|---:|
| Total trades | **2,782** |
| Total PnL (NQ, 1 contract per strat) | **+$533,364** |
| Max drawdown | **−$24,605** |
| Win rate | **51.5%** |
| Profit factor | **1.33** |
| MAR (PnL / MDD) | **21.68** |
| Annualized Sharpe (daily-bucketed) | **2.40** |
| Annualized Sortino | **3.90** |
| Calmar (annual PnL / MDD) | **4.03** |

Equity curve: see `combined_equity_curve.png`
Trade-level log: see `combined_trades.csv`

### Standalone contribution to combined

| Strat | Trades | Net PnL in combined |
|---|---:|---:|
| Rough Vol (kept after conflict res.) | 806 | +$158,202 |
| B2 OHI/OLO (kept after conflict res.) | 619 | +$166,713 |
| Overnight Drift (no conflict) | 1,357 | +$208,450 |

### Year-by-year COMBINED

| Year | Trades | PF | WR | PnL | Year MDD | Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| 2020 (Dec only) | 31 | 0.77 | 45.2% | −$2,886 | −$5,563 | −1.97 |
| 2021 | 535 | 1.44 | 51.0% | +$97,620 | −$22,429 | 3.13 |
| 2022 | 529 | 1.25 | 51.2% | +$94,000 | −$19,133 | 1.90 |
| 2023 | 583 | 1.19 | 50.8% | +$54,076 | −$24,605 | 1.56 |
| 2024 | 508 | 1.29 | 51.6% | +$79,509 | −$18,363 | 2.15 |
| 2025 | 446 | 1.45 | 52.5% | +$146,103 | −$23,294 | 3.13 |
| 2026 (Jan-May) | 150 | 1.56 | 54.7% | +$64,942 | −$15,841 | 3.66 |

Every full year positive. Largest single-year drawdown is $-24,605 (2023).

---

## Time-window orthogonality

```
00:00         06:00      09:00          14:45    16:00    19:00              23:59
  │             │          │RV ────────────│        │       │OD ───────────────►
  │             │          │B2 ────────────────────│        │
  │             │          │                       │        │
  └─ OD (cont) ─┘          └─────────INTRADAY──────┘        └───── OVERNIGHT ──
```

- **Rough Vol (RV)**: signal window **09:00–14:45 ET** (skip 13:00–13:59), force-close 14:45 ET
- **B2 OHI/OLO**: entry window **09:00–14:59 ET**, force-close 16:00 ET
- **Overnight Drift (OD)**: entry **19:00 ET sharp**, force-close **08:00 ET next morning**

OD is fully disjoint from RV and B2 — **never conflicts**.

RV and B2 share the intraday window. Conflict rule applies (below).

---

## Conflict resolution rule (RV vs B2 only)

When a new RV or B2 signal fires while another **intraday** trade is still open:

| Existing open | New entry | Action |
|---|---|---|
| LONG | LONG | **Both fire** (additive long exposure) |
| SHORT | SHORT | **Both fire** (additive short exposure) |
| LONG | SHORT | **Block the SHORT** (no hedging) |
| SHORT | LONG | **Block the LONG** (no hedging) |
| None | Anything | Take it |

Backtest result over 5+ years:
- Total intraday candidate signals: 1,464
- Blocked by opposing position: **39 (2.7%)**
  - RV blocked: 8 (1.0% of RV signals)
  - B2 blocked: 31 (4.8% of B2 signals)

The conflict rate is low because the two strategies key off different setups (RV: volatility regime, B2: range break + pinbar). Their direction signals naturally align more often than they oppose.

**Implementation note**: when both fire same-direction simultaneously, the broker account simply holds 2 contracts (1 per strategy) until each is exited by its own rules. No netting, no cross-management.

---

## Risk envelope

| Concurrent state | Max contracts open | Worst-case 1-day P&L impact |
|---|---:|---|
| RV alone | 1 | ±2.0 × ATR × $20 (typically ±$1,200–$2,000) |
| B2 alone | 1 (or 2 if mart active) | RV-like, doubled when mart fires |
| RV + B2 same direction | 2–3 | additive |
| OD overnight | 1 (or 2 if mart active) | historically up to −$10,870 |
| All three concurrent (overnight overlap zero, so RV+B2+OD only sequential) | 3 | bounded by individual MDDs |

**Worst observed single-day combined drawdown**: occurred 2023; intraday max DD $-24,605 over multi-day stretch, not a single session.

---

## Capital sizing

| Item | Value |
|---|---:|
| Combined MDD (5+ yr backtest) | $-24,605 |
| **Recommended minimum capital** | **$75,000** (≥ 3× MDD as cushion + margin) |
| Margin per contract NQ (1-contract = 1 NQ) | ~$15,000–22,000 (broker-dependent) |
| **Recommended account capital** | **$100,000** (room for 2–3 concurrent contracts + drawdown buffer) |

**Do NOT run on prop-firm trailing-drawdown accounts** ($2k–5k trailing): the OD strategy's tail-risk events (Russia invasion −$16k, Sahm rule −$15k, etc.) will blow trailing DD limits.

---

## Live runbook

### Pre-flight checklist
- [ ] All three strategies' data dependencies wired (5-min bars, volumetric orderflow, MenthorQ gamma levels, ATR/EMA features)
- [ ] Single broker account holding all 3 strategy positions (no separate sub-accounts needed — netting is handled by the conflict rule)
- [ ] Per-strategy state machine isolated: each strategy tracks its own entries/exits independently
- [ ] **Cross-strategy conflict guard** wired before sending an intraday entry order:
  - On new RV signal: query open positions → if any open is opposite direction → block this entry
  - On new B2 signal: same check
  - OD signals (19:00 ET) need no check — no possible intraday overlap
- [ ] Time zone everywhere is `America/New_York`
- [ ] OD's 08:00 ET force-close fires before RV's 09:00 ET entry window (no overlap edge case)

### Daily flow

```
08:00 ET   OD force-close (if position open)
09:00 ET   RV + B2 start signaling
13:00 ET   RV blackout begins (existing positions managed but no new entries)
14:00 ET   RV resumes signaling
14:45 ET   RV force-close
14:59 ET   B2 last entry allowed
16:00 ET   B2 force-close
19:00 ET   OD entry (long-only)
overnight  OD position managed (yellow ratchet / green target / 08:00 force-close)
```

### Operational guards

| Risk event | Guard |
|---|---|
| Major news (FOMC, NFP, CPI) | Optional: skip OD entry the night before (per OD doc) |
| Earnings on NVDA/AAPL/MSFT/GOOGL/META/AMZN | Optional: skip OD entry |
| Geopolitical event (war, central bank surprise) | Manual override — kill all entries for 1 session |
| Account drawdown > $20,000 in trailing 30 days | Hard pause: cut size to 0 and review |
| Data feed lag > 30 seconds at signal time | Skip that signal |

### Reconciliation

Once per week:
- Compare live executed trades vs backtest signal log per strategy
- Verify P&L attribution per strategy matches expected within ±5%
- Investigate any signal that lived in backtest but didn't execute live (or vice versa)

---

## Why this combination works

1. **Disjoint time windows**: OD vs intraday never overlaps → zero cross-strategy DD compounding.
2. **Different signal sources**:
   - RV: volatility regime + EMA bias + windowed absorption
   - B2: range break (prior-day OHI/OLO) + pinbar + windowed delta absorption
   - OD: time-of-day fade (long at 19:00 ET)
   The signal generation is built from different feature sets, which is why the daily PnL correlations are near zero.
3. **Different exit logic**:
   - RV: fixed ATR-based SL/TP (RR 1:1)
   - B2: ratchet trailing yellow + MFE-guard + fixed TP
   - OD: yellow ratchet + green target + time-stop
4. **Different market regimes favored**:
   - RV: prefers volatility expansion days
   - B2: prefers trend-day continuations
   - OD: prefers low-vol overnight drift
   Composite captures more of the year.

---

## Files in this folder

| File | What it is |
|---|---|
| `plan.md` | This document |
| `combined_equity_curve.png` | 3-strategy combined PnL chart + drawdown |
| `combined_trades.csv` | Full merged trade log (2,782 trades) |

## Files in parent strategy folders (canonical sources)

| File | Strategy |
|---|---|
| `live/rough vol orderflow/best live config/config.md` | Rough vol spec |
| `live/overnight range gamma strat/best live config/config.md` | B2 spec |
| `live/overnight drift/live config overnight drift.md` | OD spec |

---

## Generation source

Combined analysis script: `scripts/rough vol orderflow/combined_3way.py`
B2 trade dump script: `scripts/overnight range strat/scripts/lock_v2_k08_lock045_mart_fc_filtered.py`
Rough vol trade dump script: `scripts/rough vol orderflow/audit_and_log.py`
Overnight drift trade log: `live/overnight drift/trades.csv` (pre-generated)

To regenerate this plan's metrics:

```powershell
python "scripts/overnight range strat/scripts/lock_v2_k08_lock045_mart_fc_filtered.py"
python "scripts/rough vol orderflow/audit_and_log.py"
python "scripts/rough vol orderflow/combined_3way.py"
```

The combined script reads the per-strategy trade logs and re-runs conflict resolution. Outputs are reproducible.
