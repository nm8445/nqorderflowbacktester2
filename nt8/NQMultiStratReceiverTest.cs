// NinjaTrader Add-On: NQ Multi-Strat Receiver (TEST / FARM)  — port 8082
//
// The "hands" for the multi-account farm brains (live/farm/*.py). Isolated from the
// production single-account addon (NQMultiStratReceiver, :8081) so testing never touches
// live phase-1 money. See live/farm/TEST_ADDON_SPEC.md for the full design.
//
// PHASE 1: GET /accounts (equity source) + POST /order, /close per-account behind a HARD WHITELIST.
// Orders/closes ONLY ever touch accounts in `orderWhitelist` (the sims). MFF funded + live account are
// readable via /accounts but can NEVER be traded — they are not in the whitelist, so /order rejects them.
//
// HTTP endpoints:
//   GET  /accounts             — every NT8 account: name, cash, unrealized, netliq, dd, positions
//   GET  /account_dump?name=X  — EVERY AccountItem for one account (diagnostic)
//   POST /order                — {account, strat, direction, qty, slPrice?, tpPrice?, instrument?, tag?}
//   POST /close                — {account, tag, reason?}
//   GET  /status               — server diagnostic
//
// Install: copy to Documents\NinjaTrader 8\bin\Custom\AddOns\, compile, then
// Control Center -> New -> Add-On -> "NQ Multi-Strat Receiver (TEST)", press Start.

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
    public class NQMultiStratReceiverTest : AddOnBase
    {
        // ============== HTTP server ==============
        private HttpListener httpListener;
        private Thread listenerThread;
        private bool isRunning = false;
        private int serverPort = 8082;          // distinct from production :8081

        // ============== UI ==============
        private Dispatcher uiDispatcher;
        private NTWindow window;
        private TextBlock statusText;
        private TextBlock accountsText;
        private Button startButton;
        private Button stopButton;
        private ListBox logListBox;
        private Timer uiTimer;

        // ============== Phase 1: order execution (HARD WHITELIST) ==============
        // Orders/closes only ever touch these accounts. Edit to match your NT8 sim names EXACTLY.
        // MFF funded + live account are deliberately absent -> /order and /close reject them.
        private readonly HashSet<string> orderWhitelist = new HashSet<string>
        {
            "SimEval1", "SimEval2", "SimEval3", "SimEval4", "SimEval5"
        };
        private string instrumentName = "MNQ 06-26";   // default order instrument (sizer works in MNQ)

        private class TaggedPosition
        {
            public string Account, Tag, Strat, Direction;
            public int Quantity;
            public Order EntryOrder, StopOrder, TargetOrder;
            public DateTime EntryTime;
        }
        // keyed by "account|tag" so one signal copied across accounts tracks separately
        private ConcurrentDictionary<string, TaggedPosition> openPositions =
            new ConcurrentDictionary<string, TaggedPosition>();

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Multi-account farm executor (TEST) — /accounts equity source on :8082";
                Name        = "NQ Multi-Strat Receiver (TEST)";
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
            }
        }

        // ============== UI ==============
        private void ShowControlPanel()
        {
            window = new NTWindow
            {
                Caption = "NQ Multi-Strat Receiver (TEST) :8082",
                Width = 640, Height = 480,
                ResizeMode = ResizeMode.CanResize,
            };

            var grid = new Grid();
            grid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(40) });
            grid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(30) });
            grid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(110) });
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

            accountsText = new TextBlock { Text = "Accounts: (server stopped)", Margin = new Thickness(10, 2, 0, 0),
                                           Foreground = Brushes.Gray, TextWrapping = TextWrapping.Wrap };
            Grid.SetRow(accountsText, 2);
            grid.Children.Add(accountsText);

            logListBox = new ListBox { Margin = new Thickness(5) };
            Grid.SetRow(logListBox, 3);
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

                listenerThread = new Thread(ListenLoop) { IsBackground = true, Name = "NQMS-Test-Listener" };
                listenerThread.Start();

                uiTimer = new Timer(_ => RefreshAccountsLabel(), null, 0, 3000);  // live UI snapshot

                uiDispatcher?.BeginInvoke(new Action(() =>
                {
                    statusText.Text = $"Status: LISTENING on http://localhost:{serverPort}";
                    statusText.Foreground = Brushes.LimeGreen;
                    startButton.IsEnabled = false;
                    stopButton.IsEnabled = true;
                }));
                string wl = string.Join(", ", orderWhitelist);
                Log($"TEST server started on :{serverPort} - GET /accounts, POST /order + /close (whitelist: {wl})");
            }
            catch (Exception ex)
            {
                Log($"ERROR starting server: {ex.Message}");
            }
        }

        private void StopServer()
        {
            isRunning = false;
            try { uiTimer?.Dispose(); } catch { }
            uiTimer = null;
            try { httpListener?.Stop(); } catch { }
            try { httpListener?.Close(); } catch { }
            httpListener = null;
            try { listenerThread?.Join(2000); } catch { }
            listenerThread = null;
            uiDispatcher?.BeginInvoke(new Action(() =>
            {
                statusText.Text = "Status: Stopped";
                statusText.Foreground = Brushes.Gray;
                accountsText.Text = "Accounts: (server stopped)";
                startButton.IsEnabled = true;
                stopButton.IsEnabled = false;
            }));
            Log("TEST server stopped");
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
                string path = ctx.Request.Url.AbsolutePath.ToLower();
                string method = ctx.Request.HttpMethod;

                if (method == "GET" && path == "/accounts")
                {
                    SendJson(ctx, BuildAccountsJson());
                    return;
                }
                if (method == "GET" && path == "/account_dump")
                {
                    // Diagnostic: dump EVERY AccountItem for one account so we can see whether NT8
                    // exposes the trailing-DD / liquidation level directly (vs us computing it).
                    SendJson(ctx, BuildAccountDumpJson(ctx.Request.QueryString["name"]));
                    return;
                }
                if (method == "GET" && path == "/status")
                {
                    SendJson(ctx, $"{{\"running\":{isRunning.ToString().ToLower()},\"port\":{serverPort}," +
                                  $"\"phase\":\"1 (orders, whitelisted)\",\"open_positions\":{openPositions.Count}}}");
                    return;
                }
                if (method == "POST" && (path == "/order" || path == "/close" || path == "/flatten"))
                {
                    string body;
                    using (var reader = new StreamReader(ctx.Request.InputStream, Encoding.UTF8))
                        body = reader.ReadToEnd();
                    string resp = path == "/order" ? HandleOrder(body)
                                : path == "/close" ? HandleClose(body)
                                : HandleFlatten(body);
                    SendJson(ctx, resp);
                    return;
                }
                ctx.Response.StatusCode = 404;
                ctx.Response.Close();
            }
            catch (Exception ex)
            {
                Log($"Request ERROR: {ex.Message}");
                try { ctx.Response.StatusCode = 500; ctx.Response.Close(); } catch { }
            }
        }

        // ============== /accounts ==============
        // Reports EVERY account NT8 exposes (Account.All). The Python brains classify by name
        // (eval vs funded). Read-only — Phase 0 places no orders.
        private string BuildAccountsJson()
        {
            var sb = new StringBuilder();
            sb.Append("{\"accounts\":[");
            bool firstAcct = true;

            List<Account> accts;
            lock (Account.All) { accts = Account.All.ToList(); }   // snapshot; Account.All is shared

            foreach (var a in accts)
            {
                double cash = SafeGet(a, AccountItem.CashValue);
                double unreal = SafeGet(a, AccountItem.UnrealizedProfitLoss);
                double netliq = SafeGet(a, AccountItem.NetLiquidation);
                double dd = SafeGet(a, AccountItem.TrailingMaxDrawdown);     // live remaining DD
                double realizedToday = SafeGet(a, AccountItem.RealizedProfitLoss);  // today's P&L
                string status = ConnStatus(a);              // "Connected" = live feed; else stale/old
                bool connected = status == "Connected";

                if (!firstAcct) sb.Append(",");
                firstAcct = false;
                var ic = System.Globalization.CultureInfo.InvariantCulture;
                sb.Append("{");
                sb.Append($"\"name\":\"{Esc(a.Name)}\",");
                sb.Append($"\"connected\":{connected.ToString().ToLower()},");
                sb.Append($"\"status\":\"{Esc(status)}\",");
                sb.Append($"\"dd\":{dd.ToString(ic)},");
                sb.Append($"\"realized_today\":{realizedToday.ToString(ic)},");
                sb.Append($"\"cash\":{cash.ToString(System.Globalization.CultureInfo.InvariantCulture)},");
                sb.Append($"\"unrealized\":{unreal.ToString(System.Globalization.CultureInfo.InvariantCulture)},");
                sb.Append($"\"netliq\":{netliq.ToString(System.Globalization.CultureInfo.InvariantCulture)},");
                sb.Append("\"positions\":[");

                bool firstPos = true;
                try
                {
                    foreach (var p in a.Positions.ToList())
                    {
                        if (p.MarketPosition == MarketPosition.Flat) continue;
                        if (!firstPos) sb.Append(",");
                        firstPos = false;
                        sb.Append("{");
                        sb.Append($"\"instrument\":\"{Esc(p.Instrument?.FullName ?? "")}\",");
                        sb.Append($"\"side\":\"{p.MarketPosition}\",");
                        sb.Append($"\"qty\":{p.Quantity},");
                        sb.Append($"\"avgPrice\":{p.AveragePrice.ToString(System.Globalization.CultureInfo.InvariantCulture)}");
                        sb.Append("}");
                    }
                }
                catch { /* positions transiently unavailable */ }

                sb.Append("]}");
            }
            sb.Append("]}");
            return sb.ToString();
        }

        private double SafeGet(Account a, AccountItem item)
        {
            try { return a.Get(item, Currency.UsDollar); }
            catch { return 0.0; }
        }

        // Dump every AccountItem this account actually supports (skip the ones that throw), so we can
        // eyeball whether any of them is the prop's trailing drawdown / liquidation level.
        private string BuildAccountDumpJson(string name)
        {
            Account a;
            lock (Account.All) { a = Account.All.FirstOrDefault(x => x.Name == name); }
            if (a == null)
                return $"{{\"error\":\"account not found (check name, must be connected)\",\"name\":\"{Esc(name ?? "")}\"}}";

            var sb = new StringBuilder();
            sb.Append("{");
            sb.Append($"\"name\":\"{Esc(a.Name)}\",");
            sb.Append($"\"status\":\"{Esc(ConnStatus(a))}\",");
            sb.Append("\"items\":{");
            bool first = true;
            foreach (AccountItem item in Enum.GetValues(typeof(AccountItem)))
            {
                double v;
                try { v = a.Get(item, Currency.UsDollar); }
                catch { continue; }            // item not supported for this account type
                if (!first) sb.Append(",");
                first = false;
                sb.Append($"\"{item}\":{v.ToString(System.Globalization.CultureInfo.InvariantCulture)}");
            }
            sb.Append("}}");
            return sb.ToString();
        }

        // Connection status as a string ("Connected" / "Disconnected" / "ConnectionLost" / ...).
        // We compare the ToString() to avoid referencing the enum members directly (compile-safe).
        private string ConnStatus(Account a)
        {
            try { return a.Connection != null ? a.Connection.Status.ToString() : "NoConnection"; }
            catch { return "Unknown"; }
        }

        private static string Esc(string s)
        {
            return string.IsNullOrEmpty(s) ? "" : s.Replace("\\", "\\\\").Replace("\"", "\\\"");
        }

        private void SendJson(HttpListenerContext ctx, string json)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(json);
            ctx.Response.ContentType = "application/json";
            ctx.Response.ContentLength64 = bytes.Length;
            ctx.Response.OutputStream.Write(bytes, 0, bytes.Length);
            ctx.Response.Close();
        }

        // ============== Phase 1: order execution ==============
        // Minimal flat-JSON parser (matches the production addon style — no Newtonsoft dependency).
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

        // POST /order — place a MARKET entry (+ optional OCO SL/TP) on ONE whitelisted account.
        private string HandleOrder(string body)
        {
            var p = ParseJson(body);
            string account = p.ContainsKey("account") ? p["account"] : "";
            string strat = p.ContainsKey("strat") ? p["strat"] : "";
            string direction = p.ContainsKey("direction") ? p["direction"].ToUpper() : "";
            int qty = 0;
            int.TryParse(p.ContainsKey("qty") ? p["qty"] : (p.ContainsKey("quantity") ? p["quantity"] : "0"), out qty);
            string tag = p.ContainsKey("tag") ? p["tag"] : $"{strat}_{DateTime.Now:yyyyMMdd_HHmmss}_{direction}";

            // ---- HARD WHITELIST GATE ----
            if (!orderWhitelist.Contains(account))
            {
                Log($"REJECT order: '{account}' NOT whitelisted (tag {tag})");
                return $"{{\"ok\":false,\"error\":\"account not whitelisted\",\"account\":\"{Esc(account)}\"}}";
            }
            if (direction != "LONG" && direction != "SHORT") return "{\"ok\":false,\"error\":\"bad direction\"}";
            if (qty < 1) return "{\"ok\":false,\"error\":\"bad qty\"}";

            Account acct;
            lock (Account.All) { acct = Account.All.FirstOrDefault(a => a.Name == account); }
            if (acct == null) return $"{{\"ok\":false,\"error\":\"account not found\",\"account\":\"{Esc(account)}\"}}";

            string instrName = p.ContainsKey("instrument") ? p["instrument"] : instrumentName;
            var instrument = Instrument.GetInstrument(instrName);
            if (instrument == null) return $"{{\"ok\":false,\"error\":\"instrument not found: {Esc(instrName)}\"}}";

            OrderAction oaEntry = direction == "LONG" ? OrderAction.Buy : OrderAction.SellShort;
            OrderAction oaExit  = direction == "LONG" ? OrderAction.Sell : OrderAction.BuyToCover;
            bool hasStop = p.ContainsKey("slprice") || p.ContainsKey("slPrice");
            bool hasTgt  = p.ContainsKey("tpprice") || p.ContainsKey("tpPrice");
            Func<string, double> getPx = k => double.Parse(p.ContainsKey(k) ? p[k] : p[k.ToLower()],
                                                           System.Globalization.CultureInfo.InvariantCulture);

            var pos = new TaggedPosition { Account = account, Tag = tag, Strat = strat,
                                           Direction = direction, Quantity = qty, EntryTime = DateTime.Now };
            try
            {
                pos.EntryOrder = acct.CreateOrder(instrument, oaEntry, OrderType.Market, OrderEntry.Manual,
                    TimeInForce.Day, qty, 0, 0, "", tag + "_E", DateTime.MaxValue, null);
                acct.Submit(new[] { pos.EntryOrder });

                if (hasStop || hasTgt)
                {
                    string ocoId = (hasStop && hasTgt) ? (tag + "_OCO") : "";
                    if (hasTgt)
                    {
                        double tp = getPx("tpPrice");
                        pos.TargetOrder = acct.CreateOrder(instrument, oaExit, OrderType.Limit, OrderEntry.Manual,
                            TimeInForce.Day, qty, tp, 0, ocoId, tag + "_T", DateTime.MaxValue, null);
                        acct.Submit(new[] { pos.TargetOrder });
                    }
                    if (hasStop)
                    {
                        double sl = getPx("slPrice");
                        pos.StopOrder = acct.CreateOrder(instrument, oaExit, OrderType.StopMarket, OrderEntry.Manual,
                            TimeInForce.Day, qty, 0, sl, ocoId, tag + "_S", DateTime.MaxValue, null);
                        acct.Submit(new[] { pos.StopOrder });
                    }
                }
                openPositions[account + "|" + tag] = pos;
                Log($"ORDER {account} {strat} {direction} x{qty} {instrName} sl={hasStop} tp={hasTgt} tag={tag}");
                return $"{{\"ok\":true,\"account\":\"{Esc(account)}\",\"tag\":\"{Esc(tag)}\"}}";
            }
            catch (Exception ex)
            {
                Log($"ORDER ERROR {account} {tag}: {ex.Message}");
                return $"{{\"ok\":false,\"error\":\"{Esc(ex.Message)}\"}}";
            }
        }

        // POST /close — flatten a tagged position on ONE whitelisted account.
        private string HandleClose(string body)
        {
            var p = ParseJson(body);
            string account = p.ContainsKey("account") ? p["account"] : "";
            string tag = p.ContainsKey("tag") ? p["tag"] : null;
            string reason = p.ContainsKey("reason") ? p["reason"] : "";

            if (!orderWhitelist.Contains(account))
            {
                Log($"REJECT close: '{account}' NOT whitelisted");
                return $"{{\"ok\":false,\"error\":\"account not whitelisted\"}}";
            }
            TaggedPosition pos = null;
            if (tag != null) openPositions.TryGetValue(account + "|" + tag, out pos);
            if (pos == null)
            {
                // Tracking lost (e.g. addon recompiled mid-trade). Don't strand the position —
                // fall back to flattening the account (safe: it's whitelisted = farm-only).
                Log($"CLOSE {account} {tag}: not tracked -> flatten-by-account fallback");
                return FlattenAccount(account, reason + " (untracked)");
            }
            return CloseTaggedPosition(pos, reason);
        }

        // POST /flatten — close whatever position the (whitelisted) account holds, tracked or not.
        private string HandleFlatten(string body)
        {
            var p = ParseJson(body);
            string account = p.ContainsKey("account") ? p["account"] : "";
            if (!orderWhitelist.Contains(account))
            {
                Log($"REJECT flatten: '{account}' NOT whitelisted");
                return "{\"ok\":false,\"error\":\"account not whitelisted\"}";
            }
            return FlattenAccount(account, p.ContainsKey("reason") ? p["reason"] : "manual");
        }

        private string FlattenAccount(string accountName, string reason)
        {
            Account acct;
            lock (Account.All) { acct = Account.All.FirstOrDefault(a => a.Name == accountName); }
            if (acct == null) return "{\"ok\":false,\"error\":\"account not found\"}";
            var instrument = Instrument.GetInstrument(instrumentName);
            try
            {
                var bp = acct.Positions.FirstOrDefault(x => x.Instrument == instrument);
                if (bp == null || bp.MarketPosition == MarketPosition.Flat)
                    return "{\"ok\":true,\"note\":\"already flat\"}";
                foreach (var o in acct.Orders.ToList())          // cancel orphan SL/TP for this instrument
                    if (o.Instrument == instrument && o.OrderState == OrderState.Working) acct.Cancel(new[] { o });
                OrderAction oa = bp.MarketPosition == MarketPosition.Long ? OrderAction.Sell : OrderAction.BuyToCover;
                var co = acct.CreateOrder(instrument, oa, OrderType.Market, OrderEntry.Manual, TimeInForce.Day,
                    bp.Quantity, 0, 0, "", "FLATTEN_" + DateTime.Now.ToString("HHmmss"), DateTime.MaxValue, null);
                acct.Submit(new[] { co });
                foreach (var k in openPositions.Keys.Where(k => k.StartsWith(accountName + "|")).ToList())
                    openPositions.TryRemove(k, out _);
                Log($"FLATTEN {accountName} {bp.MarketPosition} x{bp.Quantity} reason={reason}");
                return $"{{\"ok\":true,\"note\":\"flattened\",\"qty\":{bp.Quantity}}}";
            }
            catch (Exception ex)
            {
                Log($"FLATTEN ERROR {accountName}: {ex.Message}");
                return $"{{\"ok\":false,\"error\":\"{Esc(ex.Message)}\"}}";
            }
        }

        private string CloseTaggedPosition(TaggedPosition pos, string reason)
        {
            Account acct;
            lock (Account.All) { acct = Account.All.FirstOrDefault(a => a.Name == pos.Account); }
            if (acct == null) return "{\"ok\":false,\"error\":\"account not found\"}";
            var instrument = pos.EntryOrder?.Instrument ?? Instrument.GetInstrument(instrumentName);
            try
            {
                // Cancel OCO siblings so they can't fire after the manual exit.
                if (pos.StopOrder != null && pos.StopOrder.OrderState == OrderState.Working) acct.Cancel(new[] { pos.StopOrder });
                if (pos.TargetOrder != null && pos.TargetOrder.OrderState == OrderState.Working) acct.Cancel(new[] { pos.TargetOrder });

                // Safety: only send an exit if the broker actually holds the position in our direction
                // (else a market exit would OPEN a new opposite trade). Mirrors the production addon.
                var brokerPos = acct.Positions.FirstOrDefault(x => x.Instrument == instrument);
                MarketPosition side = brokerPos == null ? MarketPosition.Flat : brokerPos.MarketPosition;
                int brokerQty = brokerPos == null ? 0 : brokerPos.Quantity;
                MarketPosition expected = pos.Direction == "LONG" ? MarketPosition.Long : MarketPosition.Short;
                if (side != expected || brokerQty == 0)
                {
                    Log($"CLOSE ABORTED {pos.Account} {pos.Tag}: broker {side} qty {brokerQty} vs tracked {pos.Direction} — untracking only");
                    openPositions.TryRemove(pos.Account + "|" + pos.Tag, out _);
                    return $"{{\"ok\":true,\"note\":\"already flat / external close\",\"tag\":\"{Esc(pos.Tag)}\"}}";
                }

                int closeQty = Math.Min(pos.Quantity, brokerQty);
                OrderAction oa = pos.Direction == "LONG" ? OrderAction.Sell : OrderAction.BuyToCover;
                var closeOrder = acct.CreateOrder(instrument, oa, OrderType.Market, OrderEntry.Manual,
                    TimeInForce.Day, closeQty, 0, 0, "", pos.Tag + "_X", DateTime.MaxValue, null);
                acct.Submit(new[] { closeOrder });
                openPositions.TryRemove(pos.Account + "|" + pos.Tag, out _);
                Log($"CLOSE {pos.Account} {pos.Tag} reason={reason} qty={closeQty}");
                return $"{{\"ok\":true,\"tag\":\"{Esc(pos.Tag)}\"}}";
            }
            catch (Exception ex)
            {
                Log($"CLOSE ERROR {pos.Account} {pos.Tag}: {ex.Message}");
                return $"{{\"ok\":false,\"error\":\"{Esc(ex.Message)}\"}}";
            }
        }

        // Live UI snapshot of accounts (every 3s) so you can eyeball it without curl.
        private void RefreshAccountsLabel()
        {
            try
            {
                List<Account> accts;
                lock (Account.All) { accts = Account.All.ToList(); }
                var conn = accts.Where(a => ConnStatus(a) == "Connected").ToList();
                var lines = conn.Select(a =>
                {
                    double cash = SafeGet(a, AccountItem.CashValue);
                    double netliq = SafeGet(a, AccountItem.NetLiquidation);
                    double unreal = SafeGet(a, AccountItem.UnrealizedProfitLoss);
                    return $"{a.Name}: cash ${cash:N0}  netliq ${netliq:N0}  float ${unreal:+0;-0;0}";
                }).ToList();
                string text = $"Connected accounts ({conn.Count} of {accts.Count}):\n" + string.Join("\n", lines);
                uiDispatcher?.BeginInvoke(new Action(() =>
                {
                    accountsText.Text = text;
                    accountsText.Foreground = Brushes.LightGray;
                }));
            }
            catch (Exception ex)
            {
                Log($"RefreshAccountsLabel error: {ex.Message}");
            }
        }
    }
}
