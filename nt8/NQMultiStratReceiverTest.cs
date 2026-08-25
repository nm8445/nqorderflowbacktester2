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
//   POST /order                — {account, strat, direction, qty, instrument?, tag?,
//                                 slPts?/tpPts? (distance, bracket off the fill) OR slPrice?/tpPrice? (absolute)}
//   POST /close                — {account, tag, reason?}
//   POST /heartbeat            — farm-brain liveness ping; if it stops, watchdog flattens ALL tracked positions
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
        // Account-name PREFIXES that are also tradeable: Lucid Flex evals (LFE...), Lucid Flex FUNDED
        // (LFF...), and sim funded test accounts (Simfunded...). The real MFF funded (MFFUSFFLX...) and
        // live (1422474) accounts never match these, so they stay hard-blocked.
        // LFF added 2026-08-17: the funded accounts existed and app.py's FUNDED_PATTERN was already
        // routing to them, but this list still only had "LFE" — so every funded order was rejected
        // ("REJECT order: 'LFF05068578480001' NOT whitelisted") and the farm silently missed the trade.
        private readonly string[] orderWhitelistPrefixes = { "LFE", "LFF", "Simfunded" };

        // Order/close/flatten gate: an exact-whitelisted name OR a name with an allowed prefix.
        private bool IsTradeAllowed(string account)
        {
            if (string.IsNullOrEmpty(account)) return false;
            if (orderWhitelist.Contains(account)) return true;
            foreach (var pre in orderWhitelistPrefixes)
                if (account.StartsWith(pre)) return true;
            return false;
        }
        private string instrumentName = "MNQ";   // default order ROOT; Resolve() appends the front month

        private class TaggedPosition
        {
            public string Account, Tag, Strat, Direction;
            public int Quantity;
            public Order EntryOrder, StopOrder, TargetOrder;
            public DateTime EntryTime;
            public double SlPts, TpPts;     // bracket distances (attached off the fill price)
            // Set by every close/flatten path (2026-08-17). AttachBracketAfterFill checks it so a
            // bracket can never be attached to (or survive) a position that was closed while the
            // background fill-wait was still running — that race left orphan SL/TP orders.
            public volatile bool Closed;
        }
        // keyed by "account|tag" so one signal copied across accounts tracks separately
        private ConcurrentDictionary<string, TaggedPosition> openPositions =
            new ConcurrentDictionary<string, TaggedPosition>();

        // ============== Heartbeat watchdog ==============
        // The farm brain (live/farm/app.py) pings POST /heartbeat every loop (~3s). If it goes silent for
        // HEARTBEAT_TIMEOUT_SEC (farm or coordinator crashed / was killed), flatten EVERY farm-tracked
        // position so a backend-managed milk leg — OD (no resting order) or B2 (green-TP only) — can't sit
        // open unmanaged. One-shot per stale episode; re-arms when the farm pings again. Mirrors :8081.
        private readonly object heartbeatLock = new object();
        private DateTime lastHeartbeat = DateTime.MinValue;
        private bool heartbeatSeen = false;
        private Timer watchdogTimer;
        private readonly int HEARTBEAT_TIMEOUT_SEC = 30;

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
                lock (heartbeatLock) { heartbeatSeen = false; lastHeartbeat = DateTime.MinValue; }  // don't flatten pre-first-ping
                // Watchdog tick also runs the orphan-order sweep (2026-08-17, :8081 incident
                // 2026-08-13) — cancels any working bracket order that outlived its position.
                watchdogTimer = new Timer(_ => { CheckHeartbeats(); SweepOrphanOrders(); }, null, 10000, 10000);  // farm-liveness watchdog

                // Re-adopt live tagged positions from the order books of tradeable accounts
                // (mirrors :8081's ResyncFromNT8). Required for the sweep to be restart-safe:
                // without it every resting SL/TP would look untracked after a recompile and
                // get cancelled out from under a still-open position.
                ResyncFromNT8();

                uiDispatcher?.BeginInvoke(new Action(() =>
                {
                    statusText.Text = $"Status: LISTENING on http://localhost:{serverPort}";
                    statusText.Foreground = Brushes.LimeGreen;
                    startButton.IsEnabled = false;
                    stopButton.IsEnabled = true;
                }));
                string wl = string.Join(", ", orderWhitelist);
                string wlp = string.Join(", ", orderWhitelistPrefixes);
                Log($"TEST server started on :{serverPort} - front month {FrontMonth()} - whitelist: {wl} - prefixes: {wlp}*");
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
            try { watchdogTimer?.Dispose(); } catch { }
            watchdogTimer = null;
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
                if (method == "POST" && path == "/heartbeat")
                {
                    using (var reader = new StreamReader(ctx.Request.InputStream, Encoding.UTF8))
                        reader.ReadToEnd();                  // body ignored — the ping itself is the signal
                    lock (heartbeatLock) { lastHeartbeat = DateTime.Now; heartbeatSeen = true; }
                    SendJson(ctx, "{\"ok\":true}");
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

        // Resolve a root ("NQ" / "MNQ") to the full front-month code ("NQ 06-26"); pass-through if it
        // already has an expiry (contains a space).
        private string Resolve(string name)
        {
            return (string.IsNullOrEmpty(name) || name.Contains(" ")) ? name : name + " " + FrontMonth();
        }

        // Front quarterly contract (Mar/Jun/Sep/Dec) as "MM-YY". NQ/MNQ trades the front quarterly
        // right up until its 3rd-Friday expiry (this matches the broker/feed — e.g. Topstep & NQ.c.0
        // were still on Jun on 2026-06-15), so we roll only ROLL_DAYS_BEFORE_EXPIRY day(s) ahead of
        // expiry. The old -8 (the CME *volume* roll) put us on Sep while everything else was still Jun
        // -> orders hit the wrong contract ~300 pts off (the 2026-06-15 bug). If your broker rolls a
        // touch earlier/later than this, bump ROLL_DAYS_BEFORE_EXPIRY — it's the one knob.
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
            if (!IsTradeAllowed(account))
            {
                Log($"REJECT order: '{account}' NOT whitelisted (tag {tag})");
                return $"{{\"ok\":false,\"error\":\"account not whitelisted\",\"account\":\"{Esc(account)}\"}}";
            }
            if (direction != "LONG" && direction != "SHORT") return "{\"ok\":false,\"error\":\"bad direction\"}";
            if (qty < 1) return "{\"ok\":false,\"error\":\"bad qty\"}";

            Account acct;
            lock (Account.All) { acct = Account.All.FirstOrDefault(a => a.Name == account); }
            if (acct == null) return $"{{\"ok\":false,\"error\":\"account not found\",\"account\":\"{Esc(account)}\"}}";

            string instrName = Resolve(p.ContainsKey("instrument") ? p["instrument"] : instrumentName);
            var instrument = Instrument.GetInstrument(instrName);
            if (instrument == null) return $"{{\"ok\":false,\"error\":\"instrument not found: {Esc(instrName)}\"}}";

            OrderAction oaEntry = direction == "LONG" ? OrderAction.Buy : OrderAction.SellShort;
            OrderAction oaExit  = direction == "LONG" ? OrderAction.Sell : OrderAction.BuyToCover;
            var ci = System.Globalization.CultureInfo.InvariantCulture;
            bool hasStop = p.ContainsKey("slprice") || p.ContainsKey("slPrice");
            bool hasTgt  = p.ContainsKey("tpprice") || p.ContainsKey("tpPrice");
            Func<string, double> getPx = k => double.Parse(p.ContainsKey(k) ? p[k] : p[k.ToLower()], ci);
            double slPts = p.ContainsKey("slPts") ? double.Parse(p["slPts"], ci) : 0;
            double tpPts = p.ContainsKey("tpPts") ? double.Parse(p["tpPts"], ci) : 0;
            bool hasPtsBracket = slPts > 0 || tpPts > 0;

            var pos = new TaggedPosition { Account = account, Tag = tag, Strat = strat,
                                           Direction = direction, Quantity = qty, EntryTime = DateTime.Now };
            // Track BEFORE submitting (2026-08-17) so the orphan sweep can never race a
            // just-submitted bracket and cancel a legitimate new order. Removed again in the
            // catch below if the submit throws.
            openPositions[account + "|" + tag] = pos;
            try
            {
                pos.EntryOrder = acct.CreateOrder(instrument, oaEntry, OrderType.Market, OrderEntry.Manual,
                    TimeInForce.Day, qty, 0, 0, "", tag + "_E", DateTime.MaxValue, null);
                acct.Submit(new[] { pos.EntryOrder });

                if (hasStop || hasTgt)
                {
                    // ALWAYS assign an OCO id (2026-08-17, :8081 incident 2026-08-13): a
                    // single-leg bracket used to get ocoId "" — no partner to kill the resting
                    // order, so every cancel path had to work perfectly. A single-member OCO
                    // group is valid and harmless; the id is unique per tag.
                    string ocoId = tag + "_OCO";
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
                else if (hasPtsBracket)   // attach SL/TP off the actual fill (background)
                {
                    pos.SlPts = slPts; pos.TpPts = tpPts;
                    Task.Run(() => AttachBracketAfterFill(acct, instrument, pos, oaExit));
                }
                Log($"ORDER {account} {strat} {direction} x{qty} {instrName} sl={hasStop} tp={hasTgt} tag={tag}");
                return $"{{\"ok\":true,\"account\":\"{Esc(account)}\",\"tag\":\"{Esc(tag)}\"}}";
            }
            catch (Exception ex)
            {
                openPositions.TryRemove(account + "|" + tag, out _);   // don't track a failed entry
                Log($"ORDER ERROR {account} {tag}: {ex.Message}");
                return $"{{\"ok\":false,\"error\":\"{Esc(ex.Message)}\"}}";
            }
        }

        // Wait for the market entry to fill, then attach an OCO SL/TP relative to the actual fill
        // price. Runs in a background task so /order returns immediately.
        private void AttachBracketAfterFill(Account acct, Instrument instr, TaggedPosition pos, OrderAction oaExit)
        {
            try
            {
                for (int i = 0; i < 60 && pos.EntryOrder.OrderState != OrderState.Filled; i++)
                    Thread.Sleep(100);                       // up to ~6s
                if (pos.EntryOrder.OrderState != OrderState.Filled)
                {
                    Log($"BRACKET {pos.Tag}: entry not filled in time - no bracket");
                    return;
                }
                // RACE GUARD (2026-08-17): the position can be closed (EXIT broadcast / flatten /
                // watchdog) while we were waiting for the fill. Attaching the bracket then would
                // create instant orphans that could FILL later (:8081 incident 2026-08-13 class).
                if (pos.Closed)
                {
                    Log($"BRACKET {pos.Tag}: position closed while waiting for fill — no bracket attached");
                    return;
                }
                double fill = pos.EntryOrder.AverageFillPrice;
                int sign = pos.Direction == "LONG" ? 1 : -1;
                string ocoId = pos.Tag + "_OCO";   // always OCO (2026-08-17) — see HandleOrder
                if (pos.TpPts > 0)
                {
                    double tp = fill + sign * pos.TpPts;
                    pos.TargetOrder = acct.CreateOrder(instr, oaExit, OrderType.Limit, OrderEntry.Manual,
                        TimeInForce.Day, pos.Quantity, tp, 0, ocoId, pos.Tag + "_T", DateTime.MaxValue, null);
                    acct.Submit(new[] { pos.TargetOrder });
                }
                if (pos.SlPts > 0)
                {
                    double sl = fill - sign * pos.SlPts;
                    pos.StopOrder = acct.CreateOrder(instr, oaExit, OrderType.StopMarket, OrderEntry.Manual,
                        TimeInForce.Day, pos.Quantity, 0, sl, ocoId, pos.Tag + "_S", DateTime.MaxValue, null);
                    acct.Submit(new[] { pos.StopOrder });
                }
                if (pos.Closed)   // closed during the submits above — kill what we just attached
                {
                    Log($"BRACKET {pos.Tag}: position closed mid-attach — cancelling just-attached bracket");
                    CancelBracketOrders(acct, pos, "closed_mid_attach");
                    return;
                }
                Log($"BRACKET {pos.Tag} fill={fill:F2} sl-{pos.SlPts:F0} tp+{pos.TpPts:F0} pts");
            }
            catch (Exception ex) { Log($"BRACKET ERROR {pos.Tag}: {ex.Message}"); }
        }

        // POST /close — flatten a tagged position on ONE whitelisted account.
        private string HandleClose(string body)
        {
            var p = ParseJson(body);
            string account = p.ContainsKey("account") ? p["account"] : "";
            string tag = p.ContainsKey("tag") ? p["tag"] : null;
            string reason = p.ContainsKey("reason") ? p["reason"] : "";

            if (!IsTradeAllowed(account))
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
            if (!IsTradeAllowed(account))
            {
                Log($"REJECT flatten: '{account}' NOT whitelisted");
                return "{\"ok\":false,\"error\":\"account not whitelisted\"}";
            }
            string reason = p.ContainsKey("reason") ? p["reason"] : "manual";
            // Optional "strat" -> flatten ONLY that strat's tracked positions (e.g. RV's 14:45 ET force-
            // close), leaving other strats running (B2 can still be open). No strat -> whole account.
            if (p.ContainsKey("strat") && p["strat"].Length > 0)
                return FlattenStrat(account, p["strat"], reason);
            return FlattenAccount(account, reason);
        }

        // Flatten only the tracked positions for ONE strat on one account. Safe on already-closed (SL/TP)
        // positions: CloseTaggedPosition broker-checks and just untracks when the broker is already flat,
        // so a stale openPositions entry never triggers a fresh order or an account-wide flatten.
        private string FlattenStrat(string accountName, string strat, string reason)
        {
            int n = 0;
            foreach (var kv in openPositions.Where(kv => kv.Key.StartsWith(accountName + "|")
                                                         && kv.Value.Strat == strat).ToList())
            {
                CloseTaggedPosition(kv.Value, reason);
                n++;
            }
            Log($"FLATTEN_STRAT {accountName} {strat}: {n} tracked position(s) reason={reason}");
            return $"{{\"ok\":true,\"note\":\"flattened strat\",\"strat\":\"{Esc(strat)}\",\"closed\":{n}}}";
        }

        private string FlattenAccount(string accountName, string reason)
        {
            Account acct;
            lock (Account.All) { acct = Account.All.FirstOrDefault(a => a.Name == accountName); }
            if (acct == null) return "{\"ok\":false,\"error\":\"account not found\"}";
            try
            {
                // ORPHAN FIX (2026-08-17, :8081 incident 2026-08-13): cancel EVERY non-terminal
                // order on the account FIRST — even when already flat. The old code returned
                // "already flat" without cancelling anything, and only cancelled Working-state
                // orders for instruments with an open position, so orphaned SL/TP from an
                // earlier external close kept resting (and could fill, opening a naked
                // position). Farm accounts are exclusively addon-managed, so an account-wide
                // order cancel is safe here.
                foreach (var pk in openPositions.Where(kv => kv.Key.StartsWith(accountName + "|")))
                    pk.Value.Closed = true;                       // stop any pending bracket-attach
                foreach (var o in acct.Orders.ToList())
                    if (o != null && !IsTerminalOrderState(o.OrderState))
                        CancelOrderChecked(acct, o, "flatten_" + reason);

                // Close EVERY non-flat position (any instrument), so a mixed NQ + MNQ trade flattens fully.
                var positions = acct.Positions.Where(x => x.MarketPosition != MarketPosition.Flat).ToList();
                if (positions.Count == 0)
                {
                    foreach (var k in openPositions.Keys.Where(k => k.StartsWith(accountName + "|")).ToList())
                        openPositions.TryRemove(k, out _);
                    return "{\"ok\":true,\"note\":\"already flat\"}";
                }
                int legs = 0;
                foreach (var bp in positions)
                {
                    OrderAction oa = bp.MarketPosition == MarketPosition.Long ? OrderAction.Sell : OrderAction.BuyToCover;
                    var co = acct.CreateOrder(bp.Instrument, oa, OrderType.Market, OrderEntry.Manual, TimeInForce.Day,
                        bp.Quantity, 0, 0, "", "FLATTEN_" + DateTime.Now.ToString("HHmmss") + "_" + legs, DateTime.MaxValue, null);
                    acct.Submit(new[] { co });
                    legs++;
                }
                foreach (var k in openPositions.Keys.Where(k => k.StartsWith(accountName + "|")).ToList())
                    openPositions.TryRemove(k, out _);
                Log($"FLATTEN {accountName} {legs} leg(s) reason={reason}");
                return $"{{\"ok\":true,\"note\":\"flattened\",\"legs\":{legs}}}";
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
            var instrument = pos.EntryOrder?.Instrument ?? Instrument.GetInstrument(Resolve(instrumentName));
            try
            {
                // Cancel OCO siblings so they can't fire after the manual exit.
                // ORPHAN FIX (2026-08-17, :8081 incident 2026-08-13): was a fire-and-forget
                // Cancel() gated on OrderState == Working — resting orders sitting in
                // Accepted/TriggerPending were silently skipped and could fill AFTER the
                // market exit. CancelBracketOrders cancels any non-terminal state, verifies,
                // retries, and logs. pos.Closed also stops a pending AttachBracketAfterFill.
                pos.Closed = true;
                CancelBracketOrders(acct, pos, "close_" + reason);

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

        // ============== Bracket-cancel invariant (added 2026-08-17) ==============
        // Mirrors the :8081 production fix for the 2026-08-13 incident (a resting bracket
        // order survived its position's close and later FILLED, opening a naked short).
        // INVARIANT: no resting bracket order may outlive the position it protects.
        // Layers: (1) every close/flatten path routes through CancelBracketOrders /
        // CancelOrderChecked — any non-terminal state, async verify + retry, full state
        // logging; (2) SweepOrphanOrders every 10s cancels working _S/_T orders whose tag
        // is no longer tracked and reconciles tracked tags against the broker net position;
        // (3) ResyncFromNT8 at startup re-adopts live tagged positions so the sweep is
        // restart-safe; (4) always-on OCO ids pair even single-leg brackets.

        private static bool IsTerminalOrderState(OrderState s)
        {
            return s == OrderState.Filled || s == OrderState.Cancelled || s == OrderState.Rejected;
        }

        // Cancel one order with state logging + async verify/retry (NT8 Cancel() is async
        // and can be rejected or silently dropped by the adapter).
        private void CancelOrderChecked(Account acct, Order o, string why)
        {
            if (o == null || acct == null) return;
            OrderState before = o.OrderState;
            if (IsTerminalOrderState(before)) return;
            Log($"  cancel {o.Name} state={before} ({why})");
            try { acct.Cancel(new[] { o }); }
            catch (Exception ex) { Log($"  cancel ERROR {o.Name}: {ex.Message}"); }
            Task.Run(() =>
            {
                try
                {
                    for (int i = 0; i < 10; i++)                     // ~5s total
                    {
                        Thread.Sleep(500);
                        OrderState now = o.OrderState;
                        if (IsTerminalOrderState(now))
                        {
                            Log($"  cancel CONFIRMED {o.Name}: {before} -> {now} ({why})");
                            return;
                        }
                        if (i == 3 || i == 7)                        // retry at ~2s and ~4s
                        {
                            Log($"  cancel RETRY {o.Name}: still {now} ({why})");
                            try { acct.Cancel(new[] { o }); }
                            catch (Exception ex) { Log($"  cancel retry ERROR {o.Name}: {ex.Message}"); }
                        }
                    }
                    Log($"  *** CANCEL FAILED {o.Name}: still {o.OrderState} after retries ({why}) — ORPHAN RISK, sweep keeps trying every 10s");
                }
                catch { /* order object torn down */ }
            });
        }

        // Cancel BOTH bracket legs, then sweep the account book for the tag in case the
        // tracked Order references are stale (recompile mid-trade / resync edge cases).
        private void CancelBracketOrders(Account acct, TaggedPosition pos, string why)
        {
            if (pos == null) return;
            pos.Closed = true;
            CancelOrderChecked(acct, pos.StopOrder, why);
            CancelOrderChecked(acct, pos.TargetOrder, why);
            try
            {
                foreach (var o in acct.Orders.ToList())
                {
                    if (o == null || string.IsNullOrEmpty(o.Name)) continue;
                    if (o == pos.StopOrder || o == pos.TargetOrder) continue;
                    if (o.Name != pos.Tag + "_S" && o.Name != pos.Tag + "_T") continue;
                    CancelOrderChecked(acct, o, why + "_tag_sweep");
                }
            }
            catch (Exception ex) { Log($"CancelBracketOrders {pos.Tag} error: {ex.Message}"); }
        }

        // Safety-net reconciliation every 10s across all TRADEABLE accounts: cancel any
        // working _S/_T whose "{account}|{tag}" is not tracked; for tracked tags, auto-fix
        // only the unambiguous broker-flat single-tag case, warn loudly on net mismatch.
        private DateTime lastNetMismatchWarn = DateTime.MinValue;   // rate-limit the warning
        private void SweepOrphanOrders()
        {
            if (!isRunning) return;
            try
            {
                List<Account> accts;
                lock (Account.All) { accts = Account.All.Where(a => IsTradeAllowed(a.Name)).ToList(); }
                foreach (var acct in accts)
                {
                    foreach (var o in acct.Orders.ToList())
                    {
                        if (o == null || string.IsNullOrEmpty(o.Name)) continue;
                        string n = o.Name;
                        if (!n.EndsWith("_S") && !n.EndsWith("_T")) continue;
                        if (IsTerminalOrderState(o.OrderState)) continue;
                        string tag = n.Substring(0, n.Length - 2);
                        var parts = tag.Split('_');
                        if (parts.Length < 4) continue;                   // not one of ours
                        string strat = parts[0];
                        if (strat != "RV" && strat != "B2" && strat != "OD" && strat != "FB") continue;

                        if (!openPositions.TryGetValue(acct.Name + "|" + tag, out var pos))
                        {
                            Log($"SWEEP {acct.Name}: orphan {n} state={o.OrderState} (tag not tracked) — cancelling");
                            CancelOrderChecked(acct, o, "sweep_untracked");
                            continue;
                        }
                        if ((DateTime.Now - pos.EntryTime).TotalSeconds < 30) continue;   // entry/bracket may still be attaching
                        var bp = acct.Positions.FirstOrDefault(x => x.Instrument == o.Instrument);
                        int brokerNet = bp == null || bp.MarketPosition == MarketPosition.Flat ? 0
                                      : (bp.MarketPosition == MarketPosition.Long ? bp.Quantity : -bp.Quantity);
                        var sameInstr = openPositions
                            .Where(kv => kv.Key.StartsWith(acct.Name + "|")
                                         && kv.Value.EntryOrder != null && kv.Value.EntryOrder.Instrument == o.Instrument)
                            .Select(kv => kv.Value).ToList();
                        int expectedNet = sameInstr.Sum(tp => tp.Direction == "LONG" ? tp.Quantity : -tp.Quantity);
                        if (brokerNet == 0 && sameInstr.Count == 1)
                        {
                            Log($"SWEEP {acct.Name}: {n} state={o.OrderState} — broker FLAT on {o.Instrument?.FullName}, sole tracked tag {tag} is stale; cancelling + untracking");
                            CancelBracketOrders(acct, pos, "sweep_broker_flat");
                            openPositions.TryRemove(acct.Name + "|" + tag, out _);
                        }
                        else if (brokerNet != expectedNet && (DateTime.Now - lastNetMismatchWarn).TotalSeconds > 60)
                        {
                            lastNetMismatchWarn = DateTime.Now;
                            Log($"SWEEP WARNING {acct.Name}: broker net {brokerNet} != tracked net {expectedNet} on {o.Instrument?.FullName} ({sameInstr.Count} tracked tag(s)) — check positions manually");
                        }
                    }
                }
            }
            catch (Exception ex) { Log($"SweepOrphanOrders error: {ex.Message}"); }
        }

        // Re-adopt live tagged positions from tradeable accounts' order books at startup
        // (mirrors :8081 ResyncFromNT8). Without this, after a recompile/restart every
        // resting SL/TP would look untracked and the sweep would cancel the brackets of
        // still-open positions.
        private void ResyncFromNT8()
        {
            try
            {
                List<Account> accts;
                lock (Account.All) { accts = Account.All.Where(a => IsTradeAllowed(a.Name)).ToList(); }
                int adopted = 0;
                foreach (var acct in accts)
                {
                    var byTag = new Dictionary<string, TaggedPosition>();
                    foreach (var ord in acct.Orders.ToList())
                    {
                        if (ord == null || string.IsNullOrEmpty(ord.Name)) continue;
                        string n = ord.Name;
                        if (!n.EndsWith("_E") && !n.EndsWith("_S") && !n.EndsWith("_T")) continue;
                        string baseTag = n.Substring(0, n.Length - 2);
                        var parts = baseTag.Split('_');
                        if (parts.Length < 4) continue;
                        string strat = parts[0];
                        if (strat != "RV" && strat != "B2" && strat != "OD" && strat != "FB") continue;
                        // Direction sits at index 3 for both auto tags ({strat}_{ymd}_{hms}_{dir})
                        // and the farm's explicit tags ({strat}_{ymd}_{hms}_{dir}_{acct}_{root}[_FG]).
                        string direction = parts.FirstOrDefault(x => x == "LONG" || x == "SHORT");
                        if (direction == null) continue;

                        if (!byTag.TryGetValue(baseTag, out var pos))
                        {
                            pos = new TaggedPosition { Account = acct.Name, Tag = baseTag, Strat = strat,
                                                       Direction = direction, Quantity = (int)ord.Quantity,
                                                       EntryTime = DateTime.Now };
                            byTag[baseTag] = pos;
                        }
                        if (n.EndsWith("_E")) pos.EntryOrder = ord;
                        else if (n.EndsWith("_S")) pos.StopOrder = ord;
                        else if (n.EndsWith("_T")) pos.TargetOrder = ord;
                    }
                    foreach (var pos in byTag.Values)
                    {
                        bool entryFilled = pos.EntryOrder != null && pos.EntryOrder.OrderState == OrderState.Filled;
                        bool stopOpen = pos.StopOrder != null && !IsTerminalOrderState(pos.StopOrder.OrderState);
                        bool tgtOpen  = pos.TargetOrder != null && !IsTerminalOrderState(pos.TargetOrder.OrderState);
                        if (entryFilled && (stopOpen || tgtOpen))
                        {
                            openPositions[acct.Name + "|" + pos.Tag] = pos;
                            adopted++;
                            Log($"  RESYNC adopted: {acct.Name} {pos.Tag} stop={stopOpen} tgt={tgtOpen}");
                        }
                    }
                }
                if (adopted > 0) Log($"Resynced {adopted} active position(s) from NT8 order books");
            }
            catch (Exception ex) { Log($"ResyncFromNT8 error: {ex.Message}"); }
        }

        // ============== Heartbeat watchdog ==============
        // POST /heartbeat (handled inline in HandleRequest) stamps lastHeartbeat. watchdogTimer calls this
        // every 10s: if the farm has pinged at least once and then gone silent past HEARTBEAT_TIMEOUT_SEC,
        // flatten every tracked position — the farm can no longer manage the backend-closed milk legs.
        // One-shot per stale episode (heartbeatSeen cleared) so it re-arms only on the next ping.
        private void CheckHeartbeats()
        {
            if (!isRunning) return;
            bool stale;
            lock (heartbeatLock)
            {
                stale = heartbeatSeen && (DateTime.Now - lastHeartbeat).TotalSeconds > HEARTBEAT_TIMEOUT_SEC;
                if (stale) heartbeatSeen = false;   // don't re-flatten until the farm pings again
            }
            if (!stale) return;
            var toClose = openPositions.Values.ToList();
            Log($"WATCHDOG: farm heartbeat stale (> {HEARTBEAT_TIMEOUT_SEC}s) — flattening {toClose.Count} tracked position(s)");
            foreach (var pos in toClose)
                CloseTaggedPosition(pos, "heartbeat_timeout");
            uiDispatcher?.BeginInvoke(new Action(() =>
            {
                statusText.Text = $"Status: HEARTBEAT STALE — flattened {toClose.Count} position(s)";
                statusText.Foreground = Brushes.OrangeRed;
            }));
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
