# VWAP Reaction Strategy — Prop Firm Configs

Strategy: VWAP Reaction Continuation | ADX 15-30 Filter (14-period, 5-min bars)
Instrument: MNQ ($2/point) | Point value: $2
Backtest period: 2025-03-13 to 2026-04-08 (~205 trading days)

---

## Fullport Challenge (Multi-Account)

Risk ~$1,000 per trade. Dynamic sizing: contracts = floor($1000 / (SL_pts * $2)).
One account per trade, rotate on daily lock. Buy N accounts, pass at least 1.

Account: $50k | Daily loss limit: $1k | Trailing DD: $2k | Target: $3k

### Best Config: SL 0.50 / TP 2.10 (ADX 15-30)
- 377 trades | WR 29.4% | Avg win ~$4,040 | Avg loss ~$984
- One TP hit = ~$3,800 = instant pass

| Accounts | Pass >= 1 | Avg Days | All Blown |
|----------|-----------|----------|-----------|
| 1        | 47.9%     | 1.4d     | 52.1%     |
| 2        | 72.6%     | 1.6d     | 27.4%     |
| 3        | 85.8%     | 1.8d     | 14.2%     |
| 4        | 93.0%     | 1.9d     | 7.0%      |
| 5        | 96.2%     | 1.9d     | 3.8%      |
| 6        | 97.9%     | 2.0d     | 2.1%      |
| 8        | 99.5%     | 2.1d     | 0.5%      |

### Runner-up: SL 0.50 / TP 1.60 (ADX 15-30)
- 379 trades | WR 34.0% | Avg win ~$3,106 | Avg loss ~$983
- Higher WR but smaller wins

| Accounts | Pass >= 1 | Avg Days | All Blown |
|----------|-----------|----------|-----------|
| 1        | 45.2%     | 1.5d     | 54.8%     |
| 2        | 69.9%     | 1.6d     | 30.1%     |
| 3        | 83.7%     | 1.8d     | 16.3%     |
| 4        | 91.0%     | 2.0d     | 9.0%      |
| 5        | 95.2%     | 2.0d     | 4.8%      |

### Also strong: SL 0.50 / TP 1.90 (ADX 15-30)
- 379 trades | WR 31.4% | Avg win ~$3,677 | Avg loss ~$984

| Accounts | Pass >= 1 | Avg Days | All Blown |
|----------|-----------|----------|-----------|
| 1        | 41.7%     | 1.4d     | 58.3%     |
| 2        | 65.4%     | 1.6d     | 34.6%     |
| 3        | 80.1%     | 1.8d     | 19.9%     |
| 4        | 88.4%     | 2.0d     | 11.6%     |
| 5        | 93.4%     | 2.1d     | 6.6%      |

---

## Fixed-Size Funded Payouts (Conservative)

$50k account | Trailing DD: $2k (locks at $50k) | Buffer: $2k | Payout: $3k
5,000 sims | 1,500 max trading days

### Best overall: SL 0.50 / TP 1.90 (ADX 15-30)
- 379 trades | WR 31.4% | PF 1.72 | Exp +$133 | Ret/DD 12.37 (best of all configs)

| MNQ | Pass% | Avg Days | Blowup% | Payouts | Days/Payout | Net Lifetime |
|-----|-------|----------|---------|---------|-------------|-------------|
| 1   | 100%  | 125d     | 0%      | 10.9    | 132 d/p     | $30,641     |
| 2   | 99%   | 64d      | 1%      | 21.8    | 66 d/p      | $63,543     |
| 3   | 96%   | 42d      | 4%      | 28.3    | 44 d/p      | **$82,853** |
| 4   | 90%   | 31d      | 10%     | 23.4    | 34 d/p      | $68,343     |
| 5   | 84%   | 22d      | 16%     | 14.5    | 27 d/p      | $41,480     |
| 6   | 79%   | 18d      | 21%     | 8.7     | 22 d/p      | $24,147     |
| 7   | 73%   | 14d      | 27%     | 5.8     | 19 d/p      | $15,443     |
| 8   | 70%   | 12d      | 30%     | 4.5     | 16 d/p      | $11,525     |
| 9   | 66%   | 10d      | 34%     | 3.4     | 14 d/p      | $8,088      |

Sweet spot: **3 MNQ** — highest net lifetime, 96% pass, only 4% blowup.

### Slippage resistance (SL 0.50 / TP 1.90, challenge pass rates)

| Slippage  | 2 MNQ | 3 MNQ | 4 MNQ | 5 MNQ |
|-----------|-------|-------|-------|-------|
| 0.0 pt    | 99%   | 96%   | 90%   | 84%   |
| 0.5 pt    | 99%   | 94%   | 88%   | 81%   |
| 1.0 pt    | 98%   | 92%   | 84%   | 75%   |
| 1.5 pt    | 97%   | 89%   | 80%   | 73%   |

---

## Backtest Stats (Top Configs)

| Config           | Trades | WR%   | PF   | Exp$  | Total$   | MaxDD$  | Ret/DD |
|------------------|--------|-------|------|-------|----------|---------|--------|
| SL 0.50 / TP 1.90| 379   | 31.4% | 1.72 | +133  | +50,523  | -4,086  | 12.37  |
| SL 0.50 / TP 2.10| 377   | 29.4% | 1.72 | +137  | +51,812  | -5,936  | 8.73   |
| SL 0.50 / TP 1.60| 379   | 34.0% | 1.63 | +112  | +42,509  | -4,795  | 8.86   |
| SL 0.50 / TP 2.40| 376   | 26.9% | 1.72 | +141  | +53,017  | -6,541  | 8.11   |
| SL 0.90 / TP 2.60| 329   | 38.3% | 1.62 | +183  | +60,145  | -6,447  | 9.33   |

All with ADX 15-30 filter, NQ $20/pt, 1 contract.

---

## Martingale Finding

Martingale does NOT work with this strategy at any DD level tested ($2k-$4.5k).
Flat betting is unanimously optimal (walk-forward 4/4 windows, Sharpe 3.4-3.5).
The 31% WR creates frequent long loss streaks that Martingale amplifies into ruin.

---

## Key Parameters
- ADX filter: 14-period on 5-min bars, range 15.0-30.0
- Entry window: 7pm - 4:00pm ET
- Force close: 4:58pm ET
- VWAP zone: +/- 3 points
- ATR: 14-period on 5-min bars (for SL/TP sizing)
- One trade at a time (no overlapping positions)
