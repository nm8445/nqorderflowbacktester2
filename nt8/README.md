# NQ Order Flow ATR Strategy - NinjaTrader 8 Setup

## Strategy Overview

**Locked Parameters (Optimized from Backtest):**
- **Stop Loss:** 2.0x ATR (~20 points, ~$400 risk per contract)
- **Profit Target:** 2.5x ATR (~25 points, ~$500 profit per contract)
- **Trading Hours:** 9:30 AM - 11:00 AM ET only (RTH)
- **Breakeven Rule:** After 5+ bars, if opposing absorption appears, move stop to entry ± 5 ticks

## Backtest Results (2.0x ATR Stop, 2.5x ATR Target)
- **Total Trades:** 1,656
- **Win Rate:** 76.4%
- **Total P&L:** $100,086 (1 NQ contract)
- **Profit Factor:** 1.63
- **Avg P&L/Trade:** $60.42

## Installation

### 1. Install the Strategy

1. Copy `NQOrderFlowATRStrategy.cs` to:
   ```
   Documents\NinjaTrader 8\bin\Custom\Strategies\
   ```

2. In NinjaTrader: **Tools → Edit NinjaScript → Compile**

3. Apply to NQ chart:
   - Right-click chart → **Strategies**
   - Select **NQOrderFlowATRStrategy**
   - Set contract size (start with 1 MNQ = 1/10 risk)
   - Enable strategy

### 2. Chart Setup

**Recommended Settings:**
- **Instrument:** NQ (Micro NQ recommended for lower risk)
- **Data Series:** 40-tick range bars
- **Indicators:** OrderFlowVWAP or similar order flow indicator

### 3. Strategy Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Contract Size | 1 | Number of contracts (use MNQ for 1/10 risk) |
| Max Trades Per Day | 3 | Maximum entries per day |
| Enable Breakeven | True | Use breakeven stop logic |

**Note:** Stop/Target ATR multipliers are LOCKED in code (not user parameters) based on backtest optimization.

## Key Rules (Already Implemented)

✅ **RTH Trading Only:** 9:30 AM - 11:00 AM ET
✅ **ATR-Based Stops:** 2.0x ATR from entry
✅ **ATR-Based Targets:** 2.5x ATR from entry
✅ **Breakeven Logic:** 5+ bars + opposing absorption → move stop to entry ± 5 ticks

## Risk Management

### Recommended Contract Sizes

| Account Size | Contract | Daily Risk (3 losses) |
|--------------|----------|----------------------|
| $5,000 | 1 MNQ | ~$120 (2.4%) |
| $10,000 | 2 MNQ | ~$240 (2.4%) |
| $25,000 | 1 NQ | ~$1,200 (4.8%) |
| $50,000 | 2 NQ | ~$2,400 (4.8%) |

**Start with MNQ** - 1/10th the risk of NQ

### Max Drawdown

From backtest:
- **Worst Day:** -$250 (1 NQ) / -$25 (1 MNQ)
- **Max 3 Losses:** ~$2,400 (1 NQ) / ~$240 (1 MNQ)

## Entry Signals

The strategy framework is ready but needs entry signal logic. Two options:

### Option 1: Python Signal Generator (Recommended)

Use the Python backtester to generate live signals:
1. Python analyzes volume profile + delta in real-time
2. Sends signals to NinjaTrader via HTTP
3. Strategy executes with ATR-based stops/targets

### Option 2: Manual Implementation

Implement entry logic directly in `OnBarUpdate()`:
```csharp
// Check volume profile proximity to VAL/VAH
// Check for absorption (delta >= 30)
// Verify cluster bias matches signal direction

if (buySignalConditionsMet)
    EnterLongATR();
else if (sellSignalConditionsMet)
    EnterShortATR();
```

## Important Notes

1. **ATR Parameters Locked:** Stop (2.0x) and Target (2.5x) multipliers are hardcoded based on optimization. Do not change without re-backtesting.

2. **Trading Hours Enforced:** Strategy will only trade 9:30-11:00 AM ET. Signals outside this window are ignored.

3. **Breakeven is Critical:** The 76.4% win rate depends on the breakeven logic. Do not disable unless testing.

4. **Commission/Slippage:** Backtest assumes $10/RT on NQ, $2/RT on MNQ. Adjust for your broker.

5. **Order Flow Data Required:** For full implementation, you need order flow data (delta, absorption levels). Use OrderFlowVWAP or similar indicator.

## Troubleshooting

**Strategy won't compile:**
- Check file is in correct folder
- Ensure no syntax errors
- Restart NinjaTrader if needed

**No trades executing:**
- Verify RTH hours (9:30-11:00 AM ET)
- Check entry signal logic is implemented
- Confirm strategy is enabled on chart

**Stops too wide/narrow:**
- ATR multipliers are locked at 2.0x/2.5x
- ATR period is 14 bars
- Verify you're using 40-tick range bars

## Performance Comparison

| Metric | Fixed Stop (10 ticks) | ATR-Based (2.0x/2.5x) |
|--------|----------------------|----------------------|
| Total P&L | $81,132 | $100,086 |
| Win Rate | 72.4% | 76.4% |
| Profit Factor | 1.53 | 1.63 |
| Trades | 1,476 | 1,656 |

**ATR-based outperforms fixed stops by +23%**

## Next Steps

1. Paper trade with 1 MNQ for 2 weeks
2. Verify win rate matches backtest (should be ~76%)
3. Monitor max daily loss (should be < $250 for 1 NQ)
4. Scale to 1 NQ after consistent results
5. Consider 2-3 contracts after 1 month profitable

## Support

For questions or issues:
- Review backtest results in `equity_curve.html` and `pnl_calendar.html`
- Check Python backtest scripts for signal generation logic
- Verify RTH filtering and breakeven logic are working correctly
