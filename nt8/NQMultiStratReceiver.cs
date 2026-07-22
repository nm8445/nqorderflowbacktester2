// NinjaTrader Add-On: NQ Multi-Strategy Signal Receiver
//
// Handles 4 concurrent strategy positions (RV, B2, OD, Fabio) on one account.
// Each entry/exit carries a unique tag so flatten commands route to ONLY the
// matching position — manual trades and other addons are not touched.
//
// Heartbeat watchdog: if Python sends no heartbeat for 30 sec, all positions
// tagged by that Python instance are flattened (failsafe).
//
// HTTP endpoints:
//   POST /order      — ENTRY (with optional bracket) or CLOSE_TAG
//   POST /heartbeat  — Python liveness ping (every 10 sec)
//   GET  /status     — diagnostic (open positions, last heartbeat)
//
// Installation:
//   1. Copy to: Documents\NinjaTrader 8\bin\Custom\AddOns\
//   2. Compile in NT (Tools > Edit NinjaScript > Compile)
//   3. Control Center -> New -> Add-On -> NQ Multi-Strat Receiver
//   4. Start the receiver from the UI (button)

using System;
using System.Collections.Concurrent;
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
    public class NQMultiStratReceiver : AddOnBase
    {
        // ============== HTTP server ==============
        private HttpListener httpListener;
        private Thread listenerThread;
        private bool isRunning = false;
        private int serverPort = 8081;

        // ============== UI ==============
        private Dispatcher uiDispatcher;
        private NTWindow window;
        private TextBlock statusText;
        private TextBlock heartbeatText;
        private TextBlock positionsText;
        private Button startButton;
        private Button stopButton;
        private ListBox logListBox;

        // ============== Trading config ==============
        private string accountName    = "MFFUSFFLX606768002";
        // ROOT only ("MNQ"/"NQ"); Resolve() appends the auto front month on EVERY order (no restart
        // needed across a quarterly roll). A full code with a space (e.g. "MNQ 06-26") is taken as-is,
        // so a contract can still be pinned manually if ever needed.
        private string instrumentName = "MNQ";
        private Account trackedAccount = null;
        private NinjaTrader.Cbi.Instrument resolvedInstrument = null;

        // ============== Position tracking by tag ==============
        // Each tag = "STRAT_YYYYMMDD_HHMMSS_DIRECTION" (e.g., "RV_20260519_094000_SHORT")
        private class TaggedPosition
        {
            public string Tag;
            public string Strat;
            public string Direction;
            public int Quantity;
            public Order EntryOrder;
            public Order StopOrder;
            public Order TargetOrder;
            public DateTime EntryTime;
            public string InstanceId;
        }
        private ConcurrentDictionary<string, TaggedPosition> openTaggedPositions = new ConcurrentDictionary<string, TaggedPosition>();

        // ============== Heartbeat watchdog ==============
        private ConcurrentDictionary<string, DateTime> lastHeartbeat = new ConcurrentDictionary<string, DateTime>();
        private Timer watchdogTimer;
        private readonly int HEARTBEAT_TIMEOUT_SEC = 30;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Multi-strategy NQ executor with tagged orders + heartbeat watchdog";
                Name        = "NQ Multi-Strat Receiver";
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
                if (watchdogTimer != null) { watchdogTimer.Dispose(); watchdogTimer = null; }
            }
        }

        // ============== UI ==============
        private void ShowControlPanel()
        {
            window = new NTWindow
            {
                Caption = "NQ Multi-Strat Receiver",
                Width = 600, Height = 480,
                ResizeMode = ResizeMode.CanResize,
            };

            var grid = new Grid();
            grid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(40) });
            grid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(30) });
            grid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(30) });
            grid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(30) });
            grid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });

            var btnPanel = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(5) };
            startButton = new Button { Content = "Start", Width = 80, Margin = new Thickness(2) };
            startButton.Click += (s, e) => StartServer();
            stopButton = new Button { Content = "Stop", Width = 80, Margin = new Thickness(2), IsEnabled = false };
            stopButton.Click += (s, e) => StopServer();
            btnPanel.Children.Add(startButton);
            btnPanel.Children.Add(stopButton);
            Grid.SetRow(btnPanel, 0);
            grid.Children.Add(btnPanel);

            statusText = new TextBlock { Text = "Status: Stopped", Margin = new Thickness(10, 0, 0, 0), Foreground = Brushes.Gray };
            Grid.SetRow(statusText, 1);
            grid.Children.Add(statusText);

            heartbeatText = new TextBlock { Text = "Heartbeat: (no instance connected)", Margin = new Thickness(10, 0, 0, 0), Foreground = Brushes.Gray };
            Grid.SetRow(heartbeatText, 2);
            grid.Children.Add(heartbeatText);

            positionsText = new TextBlock { Text = "Open tagged positions: 0", Margin = new Thickness(10, 0, 0, 0), Foreground = Brushes.Gray };
            Grid.SetRow(positionsText, 3);
            grid.Children.Add(positionsText);

            logListBox = new ListBox { Margin = new Thickness(5) };
            Grid.SetRow(logListBox, 4);
            grid.Children.Add(logListBox);

            window.Content = grid;
            window.Show();
        }

        private void Log(string msg)
        {
            string line = $"{DateTime.Now:HH:mm:ss}  {msg}";
            uiDispatcher?.BeginInvoke(new Action(() =>
            {
                logListBox.Items.Insert(0, line);
                if (logListBox.Items.Count > 200) logListBox.Items.RemoveAt(logListBox.Items.Count - 1);
            }));
        }

        // ============== HTTP server ==============
        private void StartServer()
        {
            if (isRunning) return;
            // Defensive: a previous addon instance may have leaked a listener
            // thread/prefix into NT's process. Tear down any leftover state
            // before allocating a new HttpListener — otherwise Start() throws
            // "prefix conflicts with an existing registration".
            try { httpListener?.Stop(); } catch { }
            try { httpListener?.Close(); } catch { }
            httpListener = null;
            try { listenerThread?.Join(500); } catch { }
            listenerThread = null;
            try
            {
                httpListener = new HttpListener();
                httpListener.Prefixes.Add($"http://localhost:{serverPort}/");
                httpListener.Start();
                isRunning = true;

                // Resolve instrument (auto front month) + account
                trackedAccount = Account.All.FirstOrDefault(a => a.Name == accountName);
                string resolvedName = Resolve(instrumentName);
                resolvedInstrument = Instrument.GetInstrument(resolvedName);

                listenerThread = new Thread(ListenLoop) { IsBackground = true, Name = "NQMS-Listener" };
                listenerThread.Start();

                watchdogTimer = new Timer(_ => CheckHeartbeats(), null, 10000, 10000);

                // Resync state from NT8's live order book — recovers from
                // addon recompiles/restarts mid-trade. Scans Working+Filled
                // orders, filters by our tag prefixes (RV_, B2_, OD_, FB_),
                // and rebuilds openTaggedPositions.
                ResyncFromNT8();

                uiDispatcher?.BeginInvoke(new Action(() =>
                {
                    statusText.Text = $"Status: LISTENING on http://localhost:{serverPort}";
                    statusText.Foreground = Brushes.LimeGreen;
                    startButton.IsEnabled = false;
                    stopButton.IsEnabled = true;
                }));
                Log($"Server started on port {serverPort} - front month {resolvedName}");
            }
            catch (Exception ex)
            {
                Log($"ERROR starting server: {ex.Message}");
            }
        }

        private void StopServer()
        {
            isRunning = false;
            // Aggressive teardown: Stop() unblocks GetContext(), Close() releases
            // the HTTP.sys prefix, Join() waits for the listener thread to die.
            // Without all three the port stays leased after a recompile and the
            // next StartServer() fails with "prefix conflicts with an existing
            // registration".
            try { httpListener?.Stop(); } catch { }
            try { httpListener?.Close(); } catch { }
            httpListener = null;
            try { listenerThread?.Join(2000); } catch { }
            listenerThread = null;
            watchdogTimer?.Dispose();
            watchdogTimer = null;
            uiDispatcher?.BeginInvoke(new Action(() =>
            {
                statusText.Text = "Status: Stopped";
                statusText.Foreground = Brushes.Gray;
                startButton.IsEnabled = true;
                stopButton.IsEnabled = false;
            }));
            Log("Server stopped");
        }

        // Resync openTaggedPositions from NT8's live order book at startup.
        // Each entry/stop/target order carries Name = "{tag}_E" / "_S" / "_T"
        // so we can reconstruct positions by walking the account's orders.
        private void ResyncFromNT8()
        {
            if (trackedAccount == null) return;
            try
            {
                var byTag = new Dictionary<string, TaggedPosition>();
                foreach (var ord in trackedAccount.Orders.ToList())
                {
                    if (string.IsNullOrEmpty(ord.Name)) continue;
                    // Tag is the order Name minus the suffix (_E / _S / _T / _X)
                    string n = ord.Name;
                    string baseTag = null;
                    if (n.EndsWith("_E")) baseTag = n.Substring(0, n.Length - 2);
                    else if (n.EndsWith("_S")) baseTag = n.Substring(0, n.Length - 2);
                    else if (n.EndsWith("_T")) baseTag = n.Substring(0, n.Length - 2);
                    if (baseTag == null) continue;
                    // Must look like {STRAT}_YYYYMMDD_HHMMSS_DIR
                    var parts = baseTag.Split('_');
                    if (parts.Length < 4) continue;
                    string strat = parts[0];
                    if (strat != "RV" && strat != "B2" && strat != "OD" && strat != "FB") continue;
                    string direction = parts[parts.Length - 1];
                    if (direction != "LONG" && direction != "SHORT") continue;

                    if (!byTag.TryGetValue(baseTag, out var pos))
                    {
                        pos = new TaggedPosition {
                            Tag = baseTag, Strat = strat, Direction = direction,
                            Quantity = (int)ord.Quantity, EntryTime = DateTime.Now,
                            InstanceId = "RESYNC"
                        };
                        byTag[baseTag] = pos;
                    }
                    if (n.EndsWith("_E")) pos.EntryOrder = ord;
                    else if (n.EndsWith("_S")) pos.StopOrder = ord;
                    else if (n.EndsWith("_T")) pos.TargetOrder = ord;
                }
                // Only keep positions whose entry order is FILLED AND has open
                // stop or target orders (i.e., position is still active in NT8)
                int adopted = 0;
                foreach (var pos in byTag.Values)
                {
                    bool entryFilled = pos.EntryOrder != null && pos.EntryOrder.OrderState == OrderState.Filled;
                    bool stopWorking = pos.StopOrder != null && pos.StopOrder.OrderState == OrderState.Working;
                    bool tgtWorking  = pos.TargetOrder != null && pos.TargetOrder.OrderState == OrderState.Working;
                    if (entryFilled && (stopWorking || tgtWorking))
                    {
                        openTaggedPositions[pos.Tag] = pos;
                        adopted++;
                        Log($"  RESYNC adopted: {pos.Tag} stop={stopWorking} tgt={tgtWorking}");
                    }
                }
                if (adopted > 0)
                {
                    Log($"Resynced {adopted} active position(s) from NT8 order book");
                    UpdatePositionsLabel();
                }
            }
            catch (Exception ex)
            {
                Log($"ResyncFromNT8 error: {ex.Message}");
            }
        }

        private void ListenLoop()
        {
            while (isRunning)
            {
                try
                {
                    var ctx = httpListener.GetContext();
                    Task.Run(() => HandleRequest(ctx));
                }
                catch { /* listener closed */ }
            }
        }

        private void HandleRequest(HttpListenerContext ctx)
        {
            try
            {
                string body = "";
                using (var reader = new StreamReader(ctx.Request.InputStream, Encoding.UTF8))
                    body = reader.ReadToEnd();

                string path = ctx.Request.Url.AbsolutePath.ToLower();
                string method = ctx.Request.HttpMethod;

                if (method == "POST" && path == "/order")
                {
                    HandleOrder(body);
                }
                else if (method == "POST" && path == "/heartbeat")
                {
                    HandleHeartbeat(body);
                }
                else if (method == "GET" && path == "/status")
                {
                    SendStatusJson(ctx);
                    return;
                }

                ctx.Response.StatusCode = 200;
                ctx.Response.Close();
            }
            catch (Exception ex)
            {
                Log($"Request ERROR: {ex.Message}");
                try { ctx.Response.StatusCode = 500; ctx.Response.Close(); } catch { }
            }
        }

        // Minimal flat-JSON parser (matches existing NQOrderFlowSignalReceiver style).
        // Works for flat objects with string/number/bool values. No nested objects/arrays.
        private Dictionary<string, string> ParseJson(string json)
        {
            var d = new Dictionary<string, string>();
            if (string.IsNullOrWhiteSpace(json)) return d;
            json = json.Trim().Trim('{', '}');
            foreach (string pair in json.Split(','))
            {
                string[] kv = pair.Split(new[] { ':' }, 2);
                if (kv.Length == 2)
                    d[kv[0].Trim().Trim('"', ' ')] = kv[1].Trim().Trim('"', ' ');
            }
            return d;
        }

        // ============== ORDER handler ==============
        private void HandleOrder(string body)
        {
            var p = ParseJson(body);
            string action = p.ContainsKey("action") ? p["action"].ToUpper() : "";
            string tag = p.ContainsKey("tag") ? p["tag"] : null;
            string instanceId = p.ContainsKey("instance_id") ? p["instance_id"] : "unknown";

            if (action == "ENTRY")
            {
                string strat = p["strat"];
                string direction = p["direction"];
                int qty = int.Parse(p["quantity"]);
                // Old API used a "bracket" bool; new API just inspects presence
                // of stop_price / target_price independently.
                bool hasStop  = p.ContainsKey("stop_price");
                bool hasTgt   = p.ContainsKey("target_price");

                Log($"ENTRY {tag} {direction} x{qty} stop={hasStop} tp={hasTgt}");

                OrderAction oaEntry = direction == "LONG" ? OrderAction.Buy : OrderAction.SellShort;
                OrderAction oaStop  = direction == "LONG" ? OrderAction.Sell : OrderAction.BuyToCover;
                OrderAction oaTgt   = oaStop;

                // Resolve the front month PER ORDER so a quarterly roll is picked up without restarting
                // the addon. Exits use EntryOrder.Instrument, so a position always closes on the SAME
                // contract it opened on — correct even if the roll happens while a trade is open.
                // Prefer the contract the DATA FEED is actually on (nt8_executor sends it as "contract",
                // resolved live from Databento) so execution can't diverge from the signals across a roll.
                // Falls back to "instrument", then the addon's own FrontMonth() via instrumentName.
                string contractReq = p.ContainsKey("contract") ? p["contract"]
                                   : p.ContainsKey("instrument") ? p["instrument"] : instrumentName;
                var orderInstr = Instrument.GetInstrument(Resolve(contractReq));
                if (orderInstr == null)
                {
                    Log($"ENTRY {tag}: instrument '{Resolve(contractReq)}' not found — order NOT sent");
                    return;
                }

                var pos = new TaggedPosition
                {
                    Tag = tag, Strat = strat, Direction = direction,
                    Quantity = qty, EntryTime = DateTime.Now, InstanceId = instanceId,
                };

                pos.EntryOrder = trackedAccount.CreateOrder(
                    orderInstr, oaEntry, OrderType.Market,
                    OrderEntry.Manual, TimeInForce.Day, qty, 0, 0,
                    "", tag + "_E", DateTime.MaxValue, null);
                trackedAccount.Submit(new[] { pos.EntryOrder });

                // Place stop and/or target independently. If both → OCO bracket.
                // If only target → standalone limit (B2 green case).
                // If only stop → standalone stop (rare).
                if (hasStop || hasTgt)
                {
                    string ocoId = (hasStop && hasTgt) ? (tag + "_OCO") : "";
                    if (hasTgt)
                    {
                        double tgtPx = double.Parse(p["target_price"]);
                        pos.TargetOrder = trackedAccount.CreateOrder(
                            orderInstr, oaTgt, OrderType.Limit,
                            OrderEntry.Manual, TimeInForce.Day, qty, tgtPx, 0,
                            ocoId, tag + "_T", DateTime.MaxValue, null);
                        trackedAccount.Submit(new[] { pos.TargetOrder });
                        Log($"  target sent: limit={tgtPx:F2}");
                    }
                    if (hasStop)
                    {
                        double stopPx = double.Parse(p["stop_price"]);
                        pos.StopOrder = trackedAccount.CreateOrder(
                            orderInstr, oaStop, OrderType.StopMarket,
                            OrderEntry.Manual, TimeInForce.Day, qty, 0, stopPx,
                            ocoId, tag + "_S", DateTime.MaxValue, null);
                        trackedAccount.Submit(new[] { pos.StopOrder });
                        Log($"  stop sent: {stopPx:F2}");
                    }
                }

                openTaggedPositions[tag] = pos;
                UpdatePositionsLabel();
            }
            else if (action == "CLOSE_TAG")
            {
                if (tag != null && openTaggedPositions.TryGetValue(tag, out var pos))
                {
                    string reason = p.ContainsKey("reason") ? p["reason"] : "";
                    CloseTaggedPosition(pos, reason);
                }
                else
                {
                    Log($"CLOSE_TAG {tag}: position not found (already closed?)");
                }
            }
        }

        private void CloseTaggedPosition(TaggedPosition pos, string reason)
        {
            Log($"CLOSE {pos.Tag} reason={reason}");
            try
            {
                // Close on the contract the position was OPENED on (survives a roll mid-trade); fall
                // back to the startup-resolved instrument if the entry order is somehow missing.
                var posInstr = pos.EntryOrder?.Instrument ?? resolvedInstrument;
                // SAFETY VALIDATION (added 2026-05-27):
                // Before submitting a close order, verify the broker actually has
                // an open position in the same direction. If not (e.g., user
                // manually flattened via NT8 GUI), a BuyToCover/Sell market order
                // with no offsetting position becomes a NEW opening trade — opens
                // a long when we meant to close a short, or vice versa.
                var brokerPos = trackedAccount.Positions
                    .FirstOrDefault(p => p.Instrument == posInstr);
                MarketPosition brokerSide = brokerPos == null
                    ? MarketPosition.Flat
                    : brokerPos.MarketPosition;
                int brokerQty = brokerPos == null ? 0 : brokerPos.Quantity;

                MarketPosition expectedSide = (pos.Direction == "LONG")
                    ? MarketPosition.Long
                    : MarketPosition.Short;

                if (brokerSide != expectedSide || brokerQty == 0)
                {
                    Log($"  CLOSE ABORTED {pos.Tag}: broker side={brokerSide} qty={brokerQty} "
                        + $"but tracked={pos.Direction} qty={pos.Quantity}. "
                        + "Likely closed externally (manual flatten). Untracking only — no order sent.");
                    // Cancel any orphan OCO orders so they don't fire later
                    if (pos.StopOrder != null && pos.StopOrder.OrderState == OrderState.Working)
                        trackedAccount.Cancel(new[] { pos.StopOrder });
                    if (pos.TargetOrder != null && pos.TargetOrder.OrderState == OrderState.Working)
                        trackedAccount.Cancel(new[] { pos.TargetOrder });
                    openTaggedPositions.TryRemove(pos.Tag, out _);
                    UpdatePositionsLabel();
                    return;
                }

                // Cancel any pending OCO orders
                if (pos.StopOrder != null && pos.StopOrder.OrderState == OrderState.Working)
                    trackedAccount.Cancel(new[] { pos.StopOrder });
                if (pos.TargetOrder != null && pos.TargetOrder.OrderState == OrderState.Working)
                    trackedAccount.Cancel(new[] { pos.TargetOrder });

                // Cap close quantity to actual broker position (in case of partial external close)
                int closeQty = Math.Min(pos.Quantity, brokerQty);
                if (closeQty != pos.Quantity)
                    Log($"  CLOSE {pos.Tag}: capping qty {pos.Quantity} -> {closeQty} to match broker");

                // Send market exit
                OrderAction oa = pos.Direction == "LONG" ? OrderAction.Sell : OrderAction.BuyToCover;
                var closeOrder = trackedAccount.CreateOrder(
                    posInstr, oa, OrderType.Market,
                    OrderEntry.Manual, TimeInForce.Day, closeQty, 0, 0,
                    "", pos.Tag + "_X", DateTime.MaxValue, null);
                trackedAccount.Submit(new[] { closeOrder });
            }
            catch (Exception ex)
            {
                Log($"  CLOSE ERROR {pos.Tag}: {ex.Message}");
            }
            openTaggedPositions.TryRemove(pos.Tag, out _);
            UpdatePositionsLabel();
        }

        // ============== HEARTBEAT handler ==============
        private void HandleHeartbeat(string body)
        {
            var p = ParseJson(body);
            string instanceId = p.ContainsKey("instance_id") ? p["instance_id"] : "unknown";
            lastHeartbeat[instanceId] = DateTime.Now;
            uiDispatcher?.BeginInvoke(new Action(() =>
            {
                heartbeatText.Text = $"Heartbeat: {instanceId} @ {DateTime.Now:HH:mm:ss}";
                heartbeatText.Foreground = Brushes.LimeGreen;
            }));
        }

        // Called every 10 sec by watchdogTimer
        private void CheckHeartbeats()
        {
            var now = DateTime.Now;
            var stale = lastHeartbeat
                .Where(kv => (now - kv.Value).TotalSeconds > HEARTBEAT_TIMEOUT_SEC)
                .Select(kv => kv.Key).ToList();

            foreach (var instId in stale)
            {
                Log($"WATCHDOG: heartbeat stale for instance {instId} — flattening tagged positions");
                var toClose = openTaggedPositions.Values
                    .Where(pos => pos.InstanceId == instId).ToList();
                foreach (var pos in toClose)
                    CloseTaggedPosition(pos, "heartbeat_timeout");
                lastHeartbeat.TryRemove(instId, out _);
                uiDispatcher?.BeginInvoke(new Action(() =>
                {
                    heartbeatText.Text = $"Heartbeat: STALE — flattened {toClose.Count} positions";
                    heartbeatText.Foreground = Brushes.OrangeRed;
                }));
            }
        }

        // ============== Front-month resolution (keep in sync with NQMultiStratReceiverTest.cs) ==============
        // Resolve a root ("NQ"/"MNQ") to the full front-month code ("MNQ 06-26"); pass-through if it
        // already has an expiry (contains a space, e.g. a manual pin). Called per ORDER (entry resolves
        // live; exits use the position's own EntryOrder.Instrument), so a roll needs no addon restart.
        private string Resolve(string name)
        {
            return (string.IsNullOrEmpty(name) || name.Contains(" ")) ? name : name + " " + FrontMonth();
        }

        // Front quarterly contract (Mar/Jun/Sep/Dec) as "MM-YY". Holds the front quarterly right up to
        // its 3rd-Friday expiry (matches Databento NQ.c.0 + the broker), rolling ROLL_DAYS_BEFORE_EXPIRY
        // day(s) ahead. NOT the ~8-day CME volume roll (which traded the wrong contract — 2026-06-15 bug).
        private const int ROLL_DAYS_BEFORE_EXPIRY = 1;
        private string FrontMonth()
        {
            DateTime now = DateTime.Now;
            int[] q = { 3, 6, 9, 12 };
            for (int yo = 0; yo <= 1; yo++)
                foreach (int m in q)
                {
                    int year = now.Year + yo;
                    if (ThirdFriday(year, m).AddDays(-ROLL_DAYS_BEFORE_EXPIRY) > now)
                        return string.Format("{0:00}-{1:00}", m, year % 100);
                }
            return string.Format("{0:00}-{1:00}", now.Month, now.Year % 100);
        }

        private DateTime ThirdFriday(int year, int month)
        {
            DateTime d = new DateTime(year, month, 1);
            int toFri = (((int)DayOfWeek.Friday - (int)d.DayOfWeek) + 7) % 7;
            return d.AddDays(toFri + 14);
        }

        // ============== STATUS endpoint ==============
        private void SendStatusJson(HttpListenerContext ctx)
        {
            // Manual JSON construction (matches existing addon style — no Newtonsoft dependency)
            var sb = new StringBuilder();
            sb.Append("{");
            sb.Append($"\"running\":{isRunning.ToString().ToLower()},");
            sb.Append($"\"port\":{serverPort},");
            sb.Append($"\"account\":\"{accountName}\",");
            sb.Append($"\"instrument\":\"{resolvedInstrument?.FullName ?? instrumentName}\",");
            sb.Append($"\"open_positions\":{openTaggedPositions.Count},");
            sb.Append("\"tags\":[");
            sb.Append(string.Join(",", openTaggedPositions.Keys.Select(k => $"\"{k}\"")));
            sb.Append("],\"last_heartbeats\":{");
            sb.Append(string.Join(",", lastHeartbeat.Select(kv =>
                $"\"{kv.Key}\":\"{kv.Value.ToString("yyyy-MM-dd HH:mm:ss")}\"")));
            sb.Append("}}");
            string json = sb.ToString();
            byte[] bytes = Encoding.UTF8.GetBytes(json);
            ctx.Response.ContentType = "application/json";
            ctx.Response.ContentLength64 = bytes.Length;
            ctx.Response.OutputStream.Write(bytes, 0, bytes.Length);
            ctx.Response.Close();
        }

        private void UpdatePositionsLabel()
        {
            int n = openTaggedPositions.Count;
            uiDispatcher?.BeginInvoke(new Action(() =>
            {
                positionsText.Text = $"Open tagged positions: {n}  ({string.Join(", ", openTaggedPositions.Keys.Take(5))})";
            }));
        }
    }
}
