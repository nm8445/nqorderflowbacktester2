// NinjaTrader Add-On: NQ Multi-Strat Receiver (TEST / FARM)  — port 8082
//
// The "hands" for the multi-account farm brains (live/farm/*.py). Isolated from the
// production single-account addon (NQMultiStratReceiver, :8081) so testing never touches
// live phase-1 money. See live/farm/TEST_ADDON_SPEC.md for the full design.
//
// PHASE 0 (this file): GET /accounts only — the equity SOURCE the brains' sync_accounts()
// consumes, and the one thing we must validate live (which field the firm's trailing DD
// keys off). No order routing yet; that's Phase 1 (POST /order + /close behind a whitelist).
//
// HTTP endpoints:
//   GET /accounts             — every NT8 account: name, cash (CashValue), unrealized, netliq, positions
//   GET /account_dump?name=X  — EVERY AccountItem for one account (diagnostic: is the trailing DD readable?)
//   GET /status               — server diagnostic
//
// Install: copy to Documents\NinjaTrader 8\bin\Custom\AddOns\, compile, then
// Control Center -> New -> Add-On -> "NQ Multi-Strat Receiver (TEST)", press Start.

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
                Log($"TEST server started on port {serverPort} (GET /accounts, /status)");
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
                                  $"\"phase\":\"0 (accounts only)\"}}");
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
