// NinjaTrader Strategy: NQ Order Flow ATR Strategy
// ATR-based stops & targets with breakeven logic
//
// Installation:
// 1. Copy to: Documents\NinjaTrader 8\bin\Custom\Strategies\
// 2. Compile in NinjaTrader
// 3. Apply to NQ chart with OrderFlowVWAP indicator
//
// Strategy Rules:
// - Entry: Absorption signals at VAL/VAH (9:30-11:00 AM ET only)
// - Stop: 2.0x ATR from entry
// - Target: 2.5x ATR from entry
// - Breakeven: After 5+ bars, if opposing absorption appears, move stop to entry ± 5 ticks

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;

namespace NinjaTrader.NinjaScript.Strategies
{
    public class NQOrderFlowATRStrategy : Strategy
    {
        #region Variables

        // ATR-based parameters (LOCKED)
        private const double STOP_ATR_MULT = 2.0;
        private const double TARGET_ATR_MULT = 2.5;
        private const int ATR_PERIOD = 14;
        private const int BREAKEVEN_OFFSET_TICKS = 5;
        private const int MIN_BARS_FOR_BREAKEVEN = 5;

        // RTH trading hours (ET)
        private readonly TimeSpan RTH_START = new TimeSpan(9, 30, 0);
        private readonly TimeSpan RTH_END = new TimeSpan(11, 0, 0);

        private ATR atr;
        private int entryBar = -1;
        private bool breakevenApplied = false;
        private double entryPrice = 0;

        #endregion

        #region User Parameters

        [NinjaScriptProperty]
        [Display(Name = "Contract Size", Order = 1, GroupName = "Position")]
        public int ContractSize { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Max Trades Per Day", Order = 2, GroupName = "Position")]
        public int MaxTradesPerDay { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable Breakeven", Order = 3, GroupName = "Risk Management")]
        public bool EnableBreakeven { get; set; }

        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = @"NQ Order Flow ATR Strategy - 2.0x ATR Stop, 2.5x ATR Target";
                Name = "NQOrderFlowATRStrategy";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 30;
                IsFillLimitOnTouch = false;
                MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution = OrderFillResolution.Standard;
                Slippage = 0;
                StartBehavior = StartBehavior.WaitUntilFlat;
                TimeInForce = TimeInForce.Day;
                TraceOrders = true;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade = 20;
                IsInstantiatedOnEachOptimizationIteration = true;

                // Defaults
                ContractSize = 1;
                MaxTradesPerDay = 3;
                EnableBreakeven = true;
            }
            else if (State == State.Configure)
            {
                // Use range bars (40 ticks recommended)
                // Add secondary data series if needed for order flow
            }
            else if (State == State.DataLoaded)
            {
                atr = ATR(ATR_PERIOD);
                AddChartIndicator(atr);
            }
        }

        protected override void OnBarUpdate()
        {
            if (CurrentBar < BarsRequiredToTrade)
                return;

            // Check if in RTH trading hours
            if (!IsInRTH())
                return;

            // Breakeven logic for open position
            if (Position.MarketPosition != MarketPosition.Flat && EnableBreakeven && !breakevenApplied)
            {
                int barsSinceEntry = CurrentBar - entryBar;

                if (barsSinceEntry >= MIN_BARS_FOR_BREAKEVEN)
                {
                    // Check for opposing absorption
                    bool hasOpposingAbsorption = CheckOpposingAbsorption();

                    if (hasOpposingAbsorption)
                    {
                        double breakevenStop = Position.MarketPosition == MarketPosition.Long
                            ? entryPrice + (BREAKEVEN_OFFSET_TICKS * TickSize)
                            : entryPrice - (BREAKEVEN_OFFSET_TICKS * TickSize);

                        SetStopLoss(CalculationMode.Price, breakevenStop);
                        breakevenApplied = true;

                        Print($"{Time[0]}: Breakeven stop applied at {breakevenStop}");
                    }
                }
            }

            // Entry logic would go here
            // In practice, this receives signals from Python via the SignalReceiver
            // For manual implementation, you would check:
            // 1. Volume profile (VAL/VAH proximity)
            // 2. Absorption signals (delta clusters)
            // 3. Entry conditions met

            // Example entry (replace with actual signal logic):
            // if (buySignal)
            //     EnterLongATR();
            // else if (sellSignal)
            //     EnterShortATR();
        }

        protected override void OnOrderUpdate(Order order, double limitPrice, double stopPrice,
            int quantity, int filled, double averageFillPrice,
            OrderState orderState, DateTime time, ErrorCode error, string nativeError)
        {
            // Reset on rejected orders
            if (order.Name == "Entry" && orderState == OrderState.Rejected)
            {
                Print($"Entry order rejected: {nativeError}");
            }
        }

        protected override void OnExecutionUpdate(Execution execution, string executionId,
            double price, int quantity, MarketPosition marketPosition, string orderId, DateTime time)
        {
            // Track entry for breakeven logic
            if (execution.Order.Name == "Entry" && execution.Order.OrderState == OrderState.Filled)
            {
                entryBar = CurrentBar;
                entryPrice = execution.Price;
                breakevenApplied = false;

                Print($"{time}: Entry filled at {price}, ATR: {atr[0]:F2}");
            }
        }

        #region Helper Methods

        private bool IsInRTH()
        {
            // Convert to ET (assuming market time is ET, adjust if needed)
            TimeSpan currentTime = Time[0].TimeOfDay;

            // Check if within RTH window
            return currentTime >= RTH_START && currentTime < RTH_END;
        }

        private bool CheckOpposingAbsorption()
        {
            // This needs access to order flow data
            // For NinjaTrader with OrderFlowVWAP or similar indicator:
            // Check if current bar shows absorption counter to position direction

            // Placeholder - implement based on your order flow indicator
            // Should check:
            // - Bar closed opposite direction to position
            // - Has delta cluster >= 30 (absorption)

            bool barClosedBearish = Close[0] < Open[0];
            bool barClosedBullish = Close[0] > Open[0];

            if (Position.MarketPosition == MarketPosition.Long && barClosedBearish)
            {
                // Check for buy absorption in bearish bar
                // Return true if detected
                return false; // Replace with actual check
            }
            else if (Position.MarketPosition == MarketPosition.Short && barClosedBullish)
            {
                // Check for sell absorption in bullish bar
                // Return true if detected
                return false; // Replace with actual check
            }

            return false;
        }

        private void EnterLongATR()
        {
            if (Position.MarketPosition != MarketPosition.Flat)
                return;

            double currentATR = atr[0];
            double stopLoss = Close[0] - (STOP_ATR_MULT * currentATR);
            double profitTarget = Close[0] + (TARGET_ATR_MULT * currentATR);

            EnterLong(ContractSize, "Entry");
            SetStopLoss(CalculationMode.Price, stopLoss);
            SetProfitTarget(CalculationMode.Price, profitTarget);

            Print($"{Time[0]}: LONG entry at {Close[0]:F2}, ATR: {currentATR:F2}");
            Print($"  Stop: {stopLoss:F2} ({STOP_ATR_MULT}x ATR)");
            Print($"  Target: {profitTarget:F2} ({TARGET_ATR_MULT}x ATR)");
        }

        private void EnterShortATR()
        {
            if (Position.MarketPosition != MarketPosition.Flat)
                return;

            double currentATR = atr[0];
            double stopLoss = Close[0] + (STOP_ATR_MULT * currentATR);
            double profitTarget = Close[0] - (TARGET_ATR_MULT * currentATR);

            EnterShort(ContractSize, "Entry");
            SetStopLoss(CalculationMode.Price, stopLoss);
            SetProfitTarget(CalculationMode.Price, profitTarget);

            Print($"{Time[0]}: SHORT entry at {Close[0]:F2}, ATR: {currentATR:F2}");
            Print($"  Stop: {stopLoss:F2} ({STOP_ATR_MULT}x ATR)");
            Print($"  Target: {profitTarget:F2} ({TARGET_ATR_MULT}x ATR)");
        }

        #endregion
    }
}
