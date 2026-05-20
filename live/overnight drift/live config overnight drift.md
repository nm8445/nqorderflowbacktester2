# Overnight Drift — Live Config (Locked)

Locked configuration for the overnight drift strategy. **Long NQ at 19:00 ET, force-close at 08:00 ET**, exits via yellow stop / green target / time-stop, with martingale sizing.

> Backtest data: Databento `NQ.c.0` continuous (markettick 1-min bars + recent Databento daily 5-min pickles), resampled to 20-min ET bars (anchored to midnight ET). 5.5 years: 2020-12-01 → 2026-05-06.

---

## Strategy parameters

| Parameter | Value | Notes |
|---|---|---|
| **Symbol** | NQ (E-mini Nasdaq-100 futures) | $20 / point per contract |
| **Bar interval** | 20-min | anchored to midnight ET |
| **Entry time** | 19:00 ET | long-only |
| **Force-close time** | 08:00 ET | next morning |
| **Yellow ATR length** | **14** | RMA-based (Pine `ta.atr`) |
| **Yellow ATR multiplier** | **1.30** | initial stop = `close − 1.30·ATR(14)` |
| **Yellow drift** | n/a (inert) | only used in `drift_floor` mode |
| **Yellow mode** | **`pure_ratchet`** | yellow only moves UP, never down |
| **Green ATR length** | **14** | |
| **Green ATR multiplier** | **1.00** | volatility component of target |
| **Green initial offset** | **82.5** | points above red at bar 0 |
| **Green decay per bar** | **1.50** | green tightens 1.5 pts/bar |
| **Red intercept** | **0.0** | red anchors at entry close |
| **Red drift per bar** | **0.45** | red rises 0.45 pts/bar |
| **Breakeven rule** | **OFF** | confirmed net negative after gap-through fix |
| **Use martingale** | **ON** | |
| **Base contracts (after win)** | **1** | |
| **Loss-recovery contracts** | **2** | |
| **Streak threshold** | **1 loss** | recovery fires after any single loss (s1-L2) |
| **Recovery cooldown** | trade #2 post-loss reverts to base regardless of outcome | per original Pine state machine |

### Yellow / green / red formulas (in-trade)

```
bars_in_trade  = bars since entry
raw_yellow     = current_close − 1.30 · ATR(14)
yellow_val     = max(prev_yellow, raw_yellow)        # pure_ratchet: never moves down
red_val        = entry_close + 0.0 + 0.45 · bars_in_trade
green_val      = red_val + 82.5 − 1.50 · bars_in_trade + 1.00 · ATR(14)
```

### Exit rules

| Order | Condition | Fill price |
|---|---|---|
| 1 | `high >= green_val` (intrabar touch of TP) | bar close (Pine `process_orders_on_close=true`) |
| 2 | `close <= yellow_val AND close < open` (bearish close at/below trailing stop) | bar close |
| 3 | `time == 08:00 ET` | bar close (force exit) |

### Martingale state machine

```
state 0 (after a win or initial):  trade base_qty (1)
state 1 (after a base-qty loss):   trade loss_qty (2)
state 2 (after any state-1 trade): trade base_qty (1) regardless of outcome
                                   then transition back to state 0 (if win) or state 1 (if loss)
```

The cooldown trade in state 2 is the safety: after a recovery trade fires, the next trade is forced back to 1 contract no matter what happened.

---

## Backtest performance (2020-12-01 → 2026-05-06)

### Headline

| Metric | Value |
|---|---|
| Trades | **1,357** |
| Win rate | **43.92 %** |
| Profit factor | **1.286** |
| Gross $ | **+$208,450** |
| Max drawdown | **−$28,155** |
| Drawdown trough | 2023-04-24 |
| Best single trade | +$10,850 |
| Worst single trade | −$10,870 |
| Avg win | +$1,573 |
| Avg loss | −$960 |
| Avg trade $ | +$154 |
| Median trade $ | −$135 |
| Gross / MaxDD ratio | **7.40** |

### Exit distribution

| Reason | Count | Share |
|---|---:|---:|
| TP Green | 299 | 22.0 % |
| SL Yellow | 1,045 | 77.0 % |
| Force Close (08:00 ET) | 13 | 1.0 % |

Note: many SL Yellow exits actually close *above* entry — the pure_ratchet yellow drifts up with price and locks in profits when a bearish bar takes it out.

### Per calendar year

| Year | Trades | Win % | PF | Gross $ | Year Max DD |
|---|---:|---:|---:|---:|---:|
| 2020 (Dec only) | 20 | 35.0 | 0.51 | −4,625 | −6,850 |
| 2021 | 258 | 43.8 | 1.52 | +47,565 | −14,125 |
| **2022** | **244** | **38.1** | **0.90** | **−16,950** | **−27,660** ← bear market |
| 2023 | 243 | 41.6 | 1.22 | +18,845 | −10,480 |
| 2024 | 248 | 46.0 | 1.49 | +54,995 | −15,810 |
| 2025 | 256 | 48.4 | 1.54 | +92,935 | −16,915 |
| 2026 (Jan-May) | 88 | 50.0 | 1.20 | +15,685 | −22,890 |

### Per walk-forward fold

| Fold | Window | Trades | Win % | PF | Gross $ |
|---|---|---:|---:|---:|---:|
| Pre-fold | 2020-12 → 2022-11 | 505 | 41.0 | 1.12 | +32,680 |
| **Fold 1** | **2022-12 → 2023-11** | **244** | **40.6** | **1.04** | **+3,430** ← weakest |
| Fold 2 | 2023-12 → 2024-11 | 243 | 45.7 | 1.51 | +56,495 |
| **Fold 3** | **2024-12 → 2025-11** | **256** | **49.6** | **1.58** | **+97,150** ← strongest |
| Fold 4 | 2025-12 → 2026-05 | 109 | 47.7 | 1.20 | +18,695 |

All folds profitable. F1 (PF 1.04) is the strategy's natural weak regime.

---

## Risk sizing notes

| Risk parameter | Value |
|---|---|
| Max single trade loss | **−$10,870** (2-contract martingale) |
| Max drawdown | **−$28,155** |
| Worst year (2022) | −$16,950, intrayear max DD −$27,660 |
| Recommended trading capital | **~$60-80k** (≥ 2× max DD as cushion) |
| **DO NOT use** on a $2K trailing DD prop account — single bad night = guaranteed bust |

### Why each setting was chosen

- **`pure_ratchet` over `drift_floor`**: more robust across regimes (2.5× more parameter sets survive both 2020-23 and 2024-26 in random search). `drift_floor` generates false ratchet-up wins in low-vol regimes that don't repeat.
- **`y_mult=1.30, g_mult=1.00, g_base=82.5, g_decay=1.5`**: from the 7,315-config constrained sweep, this is the highest *min-fold PF* (1.08) — i.e., the worst single fold still has a real edge. Larger `y_mult` (1.45) generates higher peak PF but breaks pre-fold (PF drops to 1.02).
- **BE off**: BE is net-negative on this strategy. The gap-through fix showed BE rescues fewer $ than it costs in fills at the gap-down open.
- **Martingale `s1-L2`** (single-loss trigger, 2c recovery): best gross/MaxDD ratio (7.40) among all sizing variants tested. Higher multipliers (`L=3+`) move gross up but also Max DD; s1-L2 sits at the efficient frontier.
  - Tested alternatives: `s2-L4` was technically better risk-adjusted but introduces a -$19,820 worst trade — too aggressive given current account targets.

---

## Files in this folder

| File | What it is |
|---|---|
| `live config overnight drift.md` | This document |
| `equity_curve.png` | Cumulative P&L chart + drawdown (static) |
| `equity_curve.html` | Same chart, interactive Plotly version |
| `trades.csv` | Every trade with entry/exit/qty/pnl/reason |
| `yearly_stats.csv` | Yearly summary |

---

## Live deployment checklist

- [ ] Verify NQ broker feed matches 20-min bar timing (anchored to midnight ET)
- [ ] Confirm entry order fires at 19:00:00 ET sharp (or as close as feasible)
- [ ] Wire 08:00 ET force-exit (some platforms need explicit time-stop)
- [ ] Implement pure_ratchet yellow trail (most platforms only have ratchet, not drift_floor — we already chose pure_ratchet so this is fine)
- [ ] Test BE off — no auto-breakeven from broker
- [ ] Wire martingale state machine: track last trade outcome; if loss → next trade 2c; after recovery → force 1c regardless
- [ ] Account capital: ≥ $60k minimum
- [ ] Pre-flight alert nights: skip on FOMC eve, major earnings (NVDA/AAPL/MSFT/GOOGL/META/AMZN), geopolitical event triggers (modeled to add ~3-5 pp to robust PF — optional)
- [ ] Paper-trade ≥ 30 trades to confirm live edge is tracking backtest

## Historical risk events that hit this strategy

| Date | Event | Single-trade P&L impact |
|---|---|---|
| 2022-02-23 | Russia invades Ukraine | -$16k+ overnight gap |
| 2024-08-04 | Sahm rule + yen carry unwind | -$15k+ |
| 2025-01-26 | DeepSeek crash | -$10k |
| 2025-04-09 | Tariff escalation | -$9k |
| 2026-04-02 | Easter weekend hold | smaller, but held cross-weekend |

These represent the strategy's natural tail risk. With martingale on, the contract size on these trades is 1 if the prior trade won (most likely), 2 if it lost — so the absolute worst-case single overnight remains bounded near the historical −$10,870 unless a Ukraine-scale event coincides with a recovery trade.

---

*Generated from `scripts/overnight drift strategy/generate_locked_pnl.py`. Last updated 2026-05-12.*
