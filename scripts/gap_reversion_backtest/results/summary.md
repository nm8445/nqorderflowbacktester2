# Overnight Gap Mean Reversion — NQ 1-min Backtest
Cost model: $4.50 commissions+fees + 0.125 pt slippage per side (=$5.00 slippage RT).
Gap threshold: ±0.3%.  IS/OOS cutoff date: 2023-12-19.
## Variant comparison (IS | OOS)
| Variant | IS n | IS PnL | IS WR | IS Sharpe | IS DD | IS Exp/Trade | OOS n | OOS PnL | OOS WR | OOS Sharpe | OOS DD | OOS Exp/Trade |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A15 | 457 | $+7,128 | 52.5% | 0.24 | $-20,502 | $+15.6 | 272 | $-2,124 | 47.8% | -0.09 | $-19,170 | $-7.8 |
| A30 | 457 | $-2,922 | 51.6% | -0.08 | $-30,940 | $-6.4 | 272 | $+6,431 | 48.9% | 0.21 | $-23,382 | $+23.6 |
| A45 | 457 | $-7,136 | 50.8% | -0.15 | $-34,984 | $-15.6 | 272 | $-12,149 | 48.5% | -0.29 | $-41,540 | $-44.7 |
| A60 | 457 | $+15,038 | 49.9% | 0.30 | $-27,710 | $+32.9 | 272 | $+556 | 49.6% | 0.01 | $-26,995 | $+2.0 |
| A90 | 457 | $-14,606 | 47.7% | -0.25 | $-36,280 | $-32.0 | 272 | $+34,256 | 54.4% | 0.73 | $-21,412 | $+125.9 |
| B | 451 | $-29,571 | 45.9% | -0.49 | $-42,634 | $-65.6 | 272 | $+21,269 | 55.1% | 0.47 | $-33,533 | $+78.2 |
| C | 451 | $+14,991 | 49.9% | 0.30 | $-25,711 | $+33.2 | 272 | $-9,467 | 50.0% | -0.25 | $-34,667 | $-34.8 |

## Best variant (by IS Sharpe): **C**
### In-sample
- Trades: 451
- Total PnL: $+14,991
- Annualized return (vs $25000 margin): +20.1%
- Sharpe (daily, annualized): 0.30
- Max drawdown: $-25,711 (-102.8%)
- Win rate: 49.9%
- Avg win: $+1,465  Avg loss: $-1,392  Payoff: 1.05
- Expected value per trade: $+33.2

### Out-of-sample
- Trades: 272
- Total PnL: $-9,467
- Annualized return (vs $25000 margin): -19.6%
- Sharpe (daily, annualized): -0.25
- Max drawdown: $-34,667 (-138.7%)
- Win rate: 50.0%
- Avg win: $+1,751  Avg loss: $-1,820  Payoff: 0.96
- Expected value per trade: $-34.8

## Direction (OOS)
| direction   |   n |    pnl |   win_rate |   avg_pnl |
|:------------|----:|-------:|-----------:|----------:|
| short       | 167 | -14092 |    49.1018 |  -84.3834 |
| long        | 105 |   4625 |    51.4286 |   44.0476 |

## Magnitude buckets (OOS)
| magnitude   |   n |       pnl |   win_rate |   avg_pnl |
|:------------|----:|----------:|-----------:|----------:|
| 0.3-0.5%    |  77 | -10671.5  |    48.0519 | -138.591  |
| 0.5-0.8%    |  87 |  20317.1  |    57.4713 |  233.53   |
| 0.8-1.2%    |  52 | -13746.1  |    46.1538 | -264.349  |
| 1.2%+       |  56 |  -5366.46 |    44.6429 |  -95.8297 |

## Day of week (OOS) — t-test vs zero
| dow       |   n |       pnl |   win_rate |   avg_pnl |    sharpe |    t_stat |   p_value | flag   |
|:----------|----:|----------:|-----------:|----------:|----------:|----------:|----------:|:-------|
| Monday    |  57 |   6009.39 |    54.386  |   105.428 |  0.687826 |  0.327127 | 0.744792  |        |
| Tuesday   |  52 | -20019.7  |    50      |  -384.995 | -3.04901  | -1.38503  | 0.172071  |        |
| Wednesday |  54 |  -5408.36 |    40.7407 |  -100.155 | -0.847453 | -0.392294 | 0.696414  |        |
| Thursday  |  62 |  35180.5  |    61.2903 |   567.427 |  4.56186  |  2.26275  | 0.0272227 |        |
| Friday    |  47 | -25228.8  |    40.4255 |  -536.783 | -3.33564  | -1.44055  | 0.156484  |        |

## Post-holiday (OOS)
| post_holiday   |   n |       pnl |   win_rate |   avg_pnl |
|:---------------|----:|----------:|-----------:|----------:|
| False          | 210 | -8548.93  |    49.0476 |  -40.7092 |
| True           |  62 |  -918.107 |    53.2258 |  -14.8082 |

## Verdict
**Edge on NQ: NO — OOS PnL non-positive.**
- OOS degraded significantly / flipped sign vs IS.
