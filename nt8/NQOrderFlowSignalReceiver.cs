
// NinjaTrader Add-On: NQ Order Flow Signal Receiver
// Receives live signals from Python signal generator via HTTP
//
// Supports: MARKET, LIMIT, STOP entry orders + CANCEL_ENTRY + BRACKET + FLATTEN
// Python controls all logic — NT8 is a dumb executor.
//
// Installation:
// 1. Copy this file to: Documents\NinjaTrader 8\bin\Custom\AddOns\
// 2. Compile in NinjaTrader (Tools > Edit NinjaScript > Compile)
// 3. Open via: Control Center -> New -> Add-On -> NQ Order Flow Signal Receiver

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Threading;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Tools;
using NinjaTrader.NinjaScript;

namespace NinjaTrader.NinjaScript.AddOns
{
    public class NQOrderFlowSignalReceiver : AddOnBase
    {
        private HttpListener httpListener;
        private Thread listenerThread;
        private bool isRunning = false;

        // UI — accessed only via uiDispatcher
        private Dispatcher uiDispatcher;
        private NTWindow window;
        private TextBlock statusText;
        private TextBlock lastSignalText;
        private Button startButton;
        private Button stopButton;
        private ListBox signalListBox;

        // Settings
        private int    serverPort   = 8080;
        private bool   autoExecute  = false;
        private int    contractSize = 1;        // default, overridden by Python quantity field
        private string accountName  = "Sim101";
        private string instrumentName = "MNQ 06-26";  // MNQ for prop firm sizing
        // Strategy: VWAP Reaction Continuation (SL 0.50x, TP configurable, ADX 15-30)

        // Order tracking
        private Order entryOrder = null;
        private Order stopOrder = null;
        private Order targetOrder = null;
        private DateTime entryTime = DateTime.MinValue;
        private double entryPrice = 0;
        private string entryState = "NONE";  // NONE, PENDING, FILLED
        private Account trackedAccount = null;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "NQ VWAP Reaction Signal Receiver - Receives live trading signals";
                Name        = "NQ Order Flow Signal Receiver";
            }
            else if (State == State.Configure)
            {
                Core.Globals.RandomDispatcher.BeginInvoke(new Action(() =>
                {
                    uiDispatcher = Dispatcher.CurrentDispatcher;
                    ShowControlPanel();
                }));
            }
            else if (State == State.Terminated)
            {
                StopServer();
                if (trackedAccount != null)
                    trackedAccount.OrderUpdate -= OnOrderUpdate;
                DispatchUI(() =>
                {
                    if (window != null) { window.Close(); window = null; }
                });
            }
        }

        private void OnOrderUpdate(object sender, OrderEventArgs e)
        {
            // Track entry fills
            if (e.Order == entryOrder && e.OrderState == OrderState.Filled)
            {
                entryTime = e.Time;
                entryPrice = e.AverageFillPrice;
                entryState = "FILLED";
                Print($"[FILL] Entry filled at {e.AverageFillPrice:F2}");
            }

            // Track entry cancellation
            if (e.Order == entryOrder && e.OrderState == OrderState.Cancelled)
            {
                entryState = "NONE";
                entryOrder = null;
                Print("[CANCEL] Entry order cancelled");
            }
        }

        private void DispatchUI(Action action)
        {
            if (uiDispatcher != null && !uiDispatcher.HasShutdownStarted)
                uiDispatcher.BeginInvoke(action);
            else
                Core.Globals.RandomDispatcher.BeginInvoke(action);
        }

        private void ShowControlPanel()
        {
            if (window != null) { window.Activate(); return; }

            window        = new NTWindow();
            window.Width  = 620;
            window.Height = 600;
            window.Title  = "NQ Order Flow Signal Receiver";
            window.Closed += (s, e) => { window = null; };

            uiDispatcher = window.Dispatcher;

            Grid grid = new Grid();
            grid.Margin = new Thickness(10);

            for (int i = 0; i < 9; i++)
                grid.RowDefinitions.Add(new RowDefinition
                {
                    Height = i == 8
                        ? new GridLength(1, GridUnitType.Star)
                        : GridLength.Auto
                });

            int row = 0;

            // Status
            SetRow(grid, new TextBlock { Text = "Server Status:", FontWeight = FontWeights.Bold, Margin = new Thickness(0,8,0,2) }, row++);
            statusText = new TextBlock { Text = "Stopped", Foreground = Brushes.OrangeRed, Margin = new Thickness(0,0,0,4) };
            SetRow(grid, statusText, row++);

            // Buttons
            StackPanel btnRow = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0,4,0,12) };
            startButton = new Button { Content = "Start Server", Width = 120, Margin = new Thickness(0,0,10,0) };
            startButton.Click += (s, e) => StartServer();
            stopButton  = new Button { Content = "Stop Server", Width = 120, IsEnabled = false };
            stopButton.Click  += (s, e) => StopServer();
            btnRow.Children.Add(startButton);
            btnRow.Children.Add(stopButton);
            SetRow(grid, btnRow, row++);

            // Settings
            SetRow(grid, new TextBlock { Text = "Settings:", FontWeight = FontWeights.Bold, Margin = new Thickness(0,4,0,2) }, row++);
            StackPanel settings = new StackPanel { Margin = new Thickness(0,0,0,8) };
            settings.Children.Add(new TextBlock { Text = $"HTTP Port:      {serverPort}" });
            settings.Children.Add(new TextBlock { Text = $"Instrument:     {instrumentName}" });
            settings.Children.Add(new TextBlock { Text = $"Default Size:   {contractSize} (overridden by Python)" });
            settings.Children.Add(new TextBlock { Text = $"Account:        {accountName}" });

            CheckBox autoBox = new CheckBox
            {
                Content   = "Auto Execute (live order placement)",
                IsChecked = autoExecute,
                Margin    = new Thickness(0,4,0,0)
            };
            autoBox.Checked   += (s, e) => autoExecute = true;
            autoBox.Unchecked += (s, e) => autoExecute = false;
            settings.Children.Add(autoBox);

            SetRow(grid, settings, row++);

            // Last signal
            SetRow(grid, new TextBlock { Text = "Last Signal:", FontWeight = FontWeights.Bold, Margin = new Thickness(0,4,0,2) }, row++);
            lastSignalText = new TextBlock { Text = "No signals yet", TextWrapping = TextWrapping.Wrap, Margin = new Thickness(0,0,0,6) };
            SetRow(grid, lastSignalText, row++);

            // History
            SetRow(grid, new TextBlock { Text = "Signal History:", FontWeight = FontWeights.Bold, Margin = new Thickness(0,4,0,2) }, row++);
            signalListBox = new ListBox();
            SetRow(grid, signalListBox, row++);

            window.Content = grid;
            window.Show();
        }

        private static void SetRow(Grid g, UIElement el, int r)
        {
            Grid.SetRow(el, r);
            g.Children.Add(el);
        }

        // ---------------------------------------------------------------
        // Server
        // ---------------------------------------------------------------

        private void StartServer()
        {
            if (isRunning) return;
            try
            {
                httpListener = new HttpListener();
                httpListener.Prefixes.Add($"http://localhost:{serverPort}/");
                httpListener.Prefixes.Add($"http://localhost:{serverPort}/status/");
                httpListener.Start();

                isRunning      = true;
                listenerThread = new Thread(ListenLoop) { IsBackground = true };
                listenerThread.Start();

                DispatchUI(() =>
                {
                    if (statusText  != null) { statusText.Text = $"Running on port {serverPort}"; statusText.Foreground = Brushes.LimeGreen; }
                    if (startButton != null) startButton.IsEnabled = false;
                    if (stopButton  != null) stopButton.IsEnabled  = true;
                });

                Print($"NQ Signal Receiver: started on port {serverPort}");
            }
            catch (Exception ex) { Print("StartServer error: " + ex.Message); }
        }

        private void StopServer()
        {
            if (!isRunning) return;
            isRunning = false;
            try
            {
                httpListener?.Stop();
                httpListener?.Close();
                httpListener   = null;
                listenerThread = null;

                DispatchUI(() =>
                {
                    if (statusText  != null) { statusText.Text = "Stopped"; statusText.Foreground = Brushes.OrangeRed; }
                    if (startButton != null) startButton.IsEnabled = true;
                    if (stopButton  != null) stopButton.IsEnabled  = false;
                });

                Print("NQ Signal Receiver: stopped");
            }
            catch (Exception ex) { Print("StopServer error: " + ex.Message); }
        }

        private void ListenLoop()
        {
            while (isRunning)
            {
                try
                {
                    HttpListenerContext ctx = httpListener.GetContext();
                    Task.Run(() => HandleRequest(ctx));
                }
                catch (Exception ex)
                {
                    if (isRunning) Print("ListenLoop error: " + ex.Message);
                }
            }
        }

        private void HandleRequest(HttpListenerContext ctx)
        {
            try
            {
                // GET /status — return current state for Python to poll
                if (ctx.Request.HttpMethod == "GET" && ctx.Request.Url.AbsolutePath.TrimEnd('/') == "/status")
                {
                    string statusJson = $"{{\"entry_state\":\"{entryState}\",\"entry_price\":\"{entryPrice:F2}\"}}";
                    byte[] statusBuf = Encoding.UTF8.GetBytes(statusJson);
                    ctx.Response.ContentLength64 = statusBuf.Length;
                    ctx.Response.ContentType     = "application/json";
                    ctx.Response.OutputStream.Write(statusBuf, 0, statusBuf.Length);
                    ctx.Response.OutputStream.Close();
                    return;
                }

                // POST — process signal
                string body;
                using (var sr = new StreamReader(ctx.Request.InputStream, ctx.Request.ContentEncoding))
                    body = sr.ReadToEnd();

                var signal = ParseJson(body);
                if (signal != null) ProcessSignal(signal);

                byte[] buf = Encoding.UTF8.GetBytes("{\"status\":\"ok\"}");
                ctx.Response.ContentLength64 = buf.Length;
                ctx.Response.ContentType     = "application/json";
                ctx.Response.OutputStream.Write(buf, 0, buf.Length);
                ctx.Response.OutputStream.Close();
            }
            catch (Exception ex)
            {
                Print("HandleRequest error: " + ex.Message);
                try { ctx.Response.StatusCode = 500; ctx.Response.Close(); } catch { }
            }
        }

        // ---------------------------------------------------------------
        // Signal handling
        // ---------------------------------------------------------------

        private void ProcessSignal(Dictionary<string, string> sig)
        {
            string action = sig.ContainsKey("action") ? sig["action"] : "";
            string timestamp = sig.ContainsKey("timestamp") ? sig["timestamp"] : "";

            // Read quantity from Python (overrides default contractSize)
            int qty = contractSize;
            if (sig.ContainsKey("quantity") && int.TryParse(sig["quantity"], out int parsedQty))
                qty = parsedQty;

            // Read account from Python (overrides default)
            string acctName = accountName;
            if (sig.ContainsKey("account") && !string.IsNullOrEmpty(sig["account"]))
                acctName = sig["account"];

            string text = $"{timestamp} - {action} x{qty}";
            if (sig.ContainsKey("stop_price"))
                text += $" (Stop: {sig["stop_price"]}, Target: {(sig.ContainsKey("target_price") ? sig["target_price"] : "?")})";
            if (sig.ContainsKey("entry_price"))
                text += $" (Entry: {sig["entry_price"]})";

            Print("Signal received: " + text);

            DispatchUI(() =>
            {
                if (lastSignalText != null)
                {
                    lastSignalText.Text       = text;
                    lastSignalText.Foreground = action == "BUY" ? Brushes.LimeGreen
                        : action == "SELL" ? Brushes.OrangeRed
                        : Brushes.White;
                }
                if (signalListBox != null)
                {
                    signalListBox.Items.Insert(0, text);
                    if (signalListBox.Items.Count > 50)
                        signalListBox.Items.RemoveAt(50);
                }
            });

            if (!autoExecute) return;

            if (action == "FLATTEN")
            {
                string exitType = sig.ContainsKey("exit_type") ? sig["exit_type"] : "UNKNOWN";
                FlattenPosition(exitType, acctName, qty);
            }
            else if (action == "CANCEL_ENTRY")
            {
                CancelEntry(acctName);
            }
            else if (action == "BRACKET")
            {
                string stopStr = sig.ContainsKey("stop_price") ? sig["stop_price"] : "";
                string targetStr = sig.ContainsKey("target_price") ? sig["target_price"] : "";
                SubmitBracket(stopStr, targetStr, acctName, qty);
            }
            else if (action == "BUY" || action == "SELL")
            {
                string orderType = sig.ContainsKey("order_type") ? sig["order_type"] : "MARKET";
                ExecuteSignal(action, sig, orderType, acctName, qty);
            }
        }

        // ---------------------------------------------------------------
        // Order execution
        // ---------------------------------------------------------------

        private void CancelEntry(string acctName)
        {
            try
            {
                Account acct = Account.All.FirstOrDefault(a => a.Name == acctName);
                if (acct == null) { Print($"[ERROR] Account not found: {acctName}"); return; }

                if (entryOrder != null)
                {
                    acct.Cancel(new[] { entryOrder });
                    Print("[CANCEL] Entry order cancellation requested");
                    entryState = "NONE";
                }
                else
                {
                    Print("[CANCEL] No entry order to cancel");
                }
            }
            catch (Exception ex) { Print($"[ERROR] CancelEntry failed: {ex.Message}"); }
        }

        private void SubmitBracket(string stopPriceStr, string targetPriceStr, string acctName, int qty)
        {
            try
            {
                Account acct = Account.All.FirstOrDefault(a => a.Name == acctName);
                if (acct == null) { Print($"[ERROR] Account not found: {acctName}"); return; }

                if (entryOrder == null)
                {
                    Print("[ERROR] No entry order to attach bracket to");
                    return;
                }

                if (!double.TryParse(stopPriceStr, out double stopPrice) ||
                    !double.TryParse(targetPriceStr, out double targetPrice))
                {
                    Print($"[ERROR] Invalid bracket prices: stop={stopPriceStr} target={targetPriceStr}");
                    return;
                }

                Instrument instr = entryOrder.Instrument;
                OrderAction exitAction = entryOrder.OrderAction == OrderAction.Buy
                    ? OrderAction.Sell : OrderAction.Buy;

                string ocoId = "OCO_" + DateTime.Now.Ticks.ToString();

                stopOrder = acct.CreateOrder(
                    instr, exitAction, OrderType.StopMarket, OrderEntry.Manual,
                    TimeInForce.Day, qty, 0, stopPrice, ocoId, "Stop",
                    Core.Globals.MaxDate, null
                );

                targetOrder = acct.CreateOrder(
                    instr, exitAction, OrderType.Limit, OrderEntry.Manual,
                    TimeInForce.Day, qty, targetPrice, 0, ocoId, "Target",
                    Core.Globals.MaxDate, null
                );

                acct.Submit(new[] { stopOrder, targetOrder });
                Print($"[BRACKET] Stop: {stopPrice:F2}, Target: {targetPrice:F2}");

                DispatchUI(() =>
                {
                    if (signalListBox != null)
                        signalListBox.Items.Insert(0, $"[BRACKET] Stop={stopPrice:F2} Target={targetPrice:F2}");
                });
            }
            catch (Exception ex) { Print($"[ERROR] SubmitBracket failed: {ex.Message}"); }
        }

        private void FlattenPosition(string exitType, string acctName, int qty)
        {
            try
            {
                Account acct = Account.All.FirstOrDefault(a => a.Name == acctName);
                if (acct == null)
                {
                    Print($"[ERROR] Account not found: {acctName}");
                    return;
                }

                if (stopOrder != null)
                {
                    acct.Cancel(new[] { stopOrder });
                    Print($"[FLATTEN] Cancelled stop order");
                    stopOrder = null;
                }
                if (targetOrder != null)
                {
                    acct.Cancel(new[] { targetOrder });
                    Print($"[FLATTEN] Cancelled target order");
                    targetOrder = null;
                }

                if (entryOrder == null)
                {
                    Print($"[FLATTEN] No tracked entry order - cannot flatten");
                    return;
                }

                Instrument instr = entryOrder.Instrument;
                OrderAction flattenAction = entryOrder.OrderAction == OrderAction.Buy
                    ? OrderAction.Sell
                    : OrderAction.Buy;

                Order flattenOrder = acct.CreateOrder(
                    instr, flattenAction, OrderType.Market, OrderEntry.Manual,
                    TimeInForce.Day, qty, 0, 0, "", "Flatten",
                    Core.Globals.MaxDate, null
                );

                acct.Submit(new[] { flattenOrder });

                entryOrder = null;
                entryState = "NONE";

                Print($"[FLATTEN] Position flattened via {exitType} - Market order submitted");

                DispatchUI(() =>
                {
                    if (signalListBox != null)
                        signalListBox.Items.Insert(0, $"[FLATTEN] {exitType} - Position flattened");
                });
            }
            catch (Exception ex)
            {
                Print($"[ERROR] Flatten position failed: {ex.Message}");
            }
        }

        private void ExecuteSignal(string action, Dictionary<string, string> sig, string orderType, string acctName, int qty)
        {
            try
            {
                Account acct = Account.All.FirstOrDefault(a => a.Name == acctName);
                if (acct == null)
                {
                    Print($"[ERROR] Account not found: {acctName}");
                    return;
                }

                Instrument instr = Instrument.GetInstrument(instrumentName);
                if (instr == null)
                {
                    Print($"[ERROR] Instrument not found: {instrumentName}");
                    return;
                }

                OrderAction orderAction = action == "BUY" ? OrderAction.Buy : OrderAction.Sell;

                // Subscribe to order updates if not already subscribed
                if (trackedAccount == null)
                {
                    trackedAccount = acct;
                    acct.OrderUpdate += OnOrderUpdate;
                }

                if (orderType == "MARKET")
                {
                    // MARKET: entry + OCO bracket all at once (original behavior)
                    string stopPriceStr = sig.ContainsKey("stop_price") ? sig["stop_price"] : "";
                    string targetPriceStr = sig.ContainsKey("target_price") ? sig["target_price"] : "";

                    if (!double.TryParse(stopPriceStr, out double stopPrice) ||
                        !double.TryParse(targetPriceStr, out double targetPrice))
                    {
                        Print($"[ERROR] Invalid prices: stop={stopPriceStr} target={targetPriceStr}");
                        return;
                    }

                    string ocoId = "OCO_" + DateTime.Now.Ticks.ToString();

                    entryOrder = acct.CreateOrder(
                        instr, orderAction, OrderType.Market, OrderEntry.Manual,
                        TimeInForce.Day, qty, 0, 0, "", "Entry",
                        Core.Globals.MaxDate, null
                    );

                    stopOrder = acct.CreateOrder(
                        instr,
                        orderAction == OrderAction.Buy ? OrderAction.Sell : OrderAction.Buy,
                        OrderType.StopMarket, OrderEntry.Manual,
                        TimeInForce.Day, qty, 0, stopPrice, ocoId, "Stop",
                        Core.Globals.MaxDate, null
                    );

                    targetOrder = acct.CreateOrder(
                        instr,
                        orderAction == OrderAction.Buy ? OrderAction.Sell : OrderAction.Buy,
                        OrderType.Limit, OrderEntry.Manual,
                        TimeInForce.Day, qty, targetPrice, 0, ocoId, "Target",
                        Core.Globals.MaxDate, null
                    );

                    acct.Submit(new[] { entryOrder, stopOrder, targetOrder });
                    entryState = "PENDING";

                    Print($"[EXECUTE] MARKET {action} x{qty} {instrumentName}");
                    Print($"[EXECUTE] Stop: {stopPrice:F2}, Target: {targetPrice:F2}");
                }
                else if (orderType == "LIMIT")
                {
                    // LIMIT: entry only, no bracket. Python sends BRACKET after fill.
                    string entryPriceStr = sig.ContainsKey("entry_price") ? sig["entry_price"] : "";
                    if (!double.TryParse(entryPriceStr, out double limitPrice))
                    {
                        Print($"[ERROR] Invalid limit price: {entryPriceStr}");
                        return;
                    }

                    entryOrder = acct.CreateOrder(
                        instr, orderAction, OrderType.Limit, OrderEntry.Manual,
                        TimeInForce.Day, qty, limitPrice, 0, "", "Entry",
                        Core.Globals.MaxDate, null
                    );

                    acct.Submit(new[] { entryOrder });
                    entryState = "PENDING";

                    Print($"[EXECUTE] LIMIT {action} x{qty} @ {limitPrice:F2}");
                }
                else if (orderType == "STOP")
                {
                    // STOP-LIMIT: entry only, no bracket. Python sends BRACKET after fill.
                    // Limit price == stop price → zero entry slippage (may miss fill if blows through).
                    string entryPriceStr = sig.ContainsKey("entry_price") ? sig["entry_price"] : "";
                    if (!double.TryParse(entryPriceStr, out double stopEntryPrice))
                    {
                        Print($"[ERROR] Invalid stop price: {entryPriceStr}");
                        return;
                    }

                    entryOrder = acct.CreateOrder(
                        instr, orderAction, OrderType.StopLimit, OrderEntry.Manual,
                        TimeInForce.Day, qty, stopEntryPrice, stopEntryPrice, "", "Entry",
                        Core.Globals.MaxDate, null
                    );

                    acct.Submit(new[] { entryOrder });
                    entryState = "PENDING";

                    Print($"[EXECUTE] STOP-LIMIT {action} x{qty} @ {stopEntryPrice:F2}");
                }

                DispatchUI(() =>
                {
                    if (signalListBox != null)
                        signalListBox.Items.Insert(0, $"[ORDER] {orderType} {action} x{qty} submitted");
                });
            }
            catch (Exception ex)
            {
                Print($"[ERROR] ExecuteSignal failed: {ex.Message}");
            }
        }

        private Dictionary<string, string> ParseJson(string json)
        {
            var d = new Dictionary<string, string>();
            json = json.Trim().Trim('{', '}');
            foreach (string pair in json.Split(','))
            {
                string[] kv = pair.Split(new[] { ':' }, 2);
                if (kv.Length == 2)
                    d[kv[0].Trim('"', ' ')] = kv[1].Trim('"', ' ');
            }
            return d;
        }
    }
}
