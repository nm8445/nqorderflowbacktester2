"""NQ Farm dashboard — a local web app over the read-only monitor.

Press Start in the browser; it polls the NT8 test addon (:8082), feeds the eval brain, and shows live
account state + balances, auto-refreshing. **Read-only** — it never places an order. Only WATCHED
(sim) accounts go through the brain; LIVE accounts (MFF / your live account) are shown balance-only and
are never traded.

Pure stdlib (no FastAPI/npm). Run:  python live/farm/app.py   then open  http://localhost:8090
(or double-click live/farm/start_farm_app.bat)
"""
from __future__ import annotations
import json
import os
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from accounts_client import fetch_accounts
from eval_passer import EvalFarm

APP_PORT = 8090
POLL_SEC = 3.0
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", "eval_farm_state.json")

# Farm-traded (through the brain). Sim accounts report DD=0 -> computed floor from `dd`.
WATCHED = {"SimEval": {"dd": 2000.0, "start_bal": 2000.0, "live_dd": False}}
# Read-only balance display only (your REAL accounts; NEVER traded).
LIVE_DISPLAY = ["MFFUSFFLX606768002", "1422474"]


class FarmState:
    def __init__(self):
        self.farm = EvalFarm(copies=1, day_cap=1500.0, quiet=True)
        self.farm.load_state(STATE_FILE)        # survive restarts/sleep; NT8 refreshes live numbers on sync
        self.lock = threading.Lock()
        self.running = False
        self.addon_up = False
        self.snapshot = {"watched": [], "live": [], "counts": {}, "ts": ""}
        threading.Thread(target=self._loop, daemon=True).start()

    def _watched_cfg(self, name):
        for pat, c in WATCHED.items():
            if pat in name:
                return c
        return None

    def _loop(self):
        while True:
            if not self.running:
                time.sleep(0.4)
                continue
            try:
                accts = fetch_accounts()
                self.addon_up = True
            except Exception:
                self.addon_up = False
                time.sleep(POLL_SEC)
                continue
            by_name = {a["name"]: a for a in accts}
            snap = {}
            for a in accts:
                c = self._watched_cfg(a["name"])
                if c is None:
                    continue
                m = {"cash": a["cash"], "equity": a["netliq"], "realized_today": a.get("realized_today", 0.0)}
                if c["live_dd"] and a.get("dd", 0) > 0:
                    m["dd"] = a["dd"]
                if "start_bal" in c:
                    m["start_bal"] = c["start_bal"]
                snap[a["name"]] = m
            self.farm.sync_accounts(snap)

            watched = []
            for name, acct in self.farm.accounts.items():
                raw = by_name.get(name, {})
                watched.append({"name": name, "state": acct.state.value,
                                "balance": raw.get("cash", acct.cash), "buffer": acct.buffer,
                                "today": raw.get("realized_today", 0.0)})
            live = []
            for pat in LIVE_DISPLAY:
                row = next((a for a in accts if pat in a["name"]), None)
                if row:
                    live.append({"name": row["name"], "balance": row["cash"], "dd": row.get("dd", 0.0),
                                 "today": row.get("realized_today", 0.0), "connected": row.get("connected", True)})
            leads, actives = self.farm.next_signal_takers()
            self.farm.save_state(STATE_FILE)
            with self.lock:
                self.snapshot = {"watched": watched, "live": live, "counts": dict(self.farm.counts()),
                                 "ready": {"leads": leads, "actives": actives, "copies": self.farm.copies},
                                 "ts": datetime.now().strftime("%H:%M:%S")}
            time.sleep(POLL_SEC)

    def get(self):
        with self.lock:
            s = dict(self.snapshot)
        s["running"] = self.running
        s["addon_up"] = self.addon_up
        return s


STATE = FarmState()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, HTML, "text/html")
        elif self.path == "/api/state":
            self._send(200, json.dumps(STATE.get()))
        else:
            self._send(404, "{}")

    def do_POST(self):
        if self.path == "/api/start":
            STATE.running = True
            self._send(200, '{"ok":true}')
        elif self.path == "/api/stop":
            STATE.running = False
            self._send(200, '{"ok":true}')
        else:
            self._send(404, "{}")


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>NQ Farm</title>
<script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-slate-900 text-slate-100 min-h-screen">
<div class="max-w-5xl mx-auto p-6">
  <div class="flex items-center justify-between mb-6">
    <h1 class="text-2xl font-bold">NQ Farm <span id="ts" class="text-slate-500 text-sm font-normal"></span></h1>
    <div class="flex items-center gap-3">
      <span id="conn" class="px-3 py-1 rounded-full text-sm bg-slate-600">Stopped</span>
      <button onclick="ctl('start')" class="px-4 py-2 rounded bg-emerald-600 hover:bg-emerald-500 font-semibold">Start</button>
      <button onclick="ctl('stop')" class="px-4 py-2 rounded bg-slate-700 hover:bg-slate-600 font-semibold">Stop</button>
    </div>
  </div>
  <div id="counts" class="flex flex-wrap gap-2 mb-4 text-sm"></div>
  <div id="ready" class="mb-6 p-3 rounded-lg bg-slate-800 border border-sky-700"></div>
  <h2 class="text-lg font-semibold mb-2 text-slate-300">Eval accounts <span class="text-slate-500 text-sm">(farm-traded)</span></h2>
  <div id="watched" class="grid grid-cols-2 md:grid-cols-3 gap-3 mb-8"></div>
  <h2 class="text-lg font-semibold mb-2 text-slate-300">Live accounts <span class="text-slate-500 text-sm">(read-only, never traded)</span></h2>
  <div id="live" class="grid grid-cols-2 md:grid-cols-3 gap-3"></div>
</div>
<script>
const C = {FRESH:'bg-slate-600', ACTIVE:'bg-sky-600', DONE:'bg-amber-600', PASSED:'bg-emerald-600', BLOWN:'bg-rose-700'};
const OUT = {PASSED:'ring-2 ring-emerald-400 shadow-lg shadow-emerald-500/40', BLOWN:'ring-2 ring-rose-500 shadow-lg shadow-rose-500/40'};
async function ctl(a){ await fetch('/api/'+a,{method:'POST'}); tick(); }
function money(x){ x=x||0; return (x<0?'-$':'$')+Math.abs(x).toLocaleString(undefined,{maximumFractionDigits:0}); }
async function tick(){
  let s; try{ s = await (await fetch('/api/state')).json(); }catch(e){ return; }
  document.getElementById('ts').textContent = s.ts ? '· '+s.ts : '';
  const conn = document.getElementById('conn');
  if(!s.running){ conn.textContent='Stopped'; conn.className='px-3 py-1 rounded-full text-sm bg-slate-600'; }
  else if(s.addon_up){ conn.textContent='Connected to NT8'; conn.className='px-3 py-1 rounded-full text-sm bg-emerald-600'; }
  else { conn.textContent='Addon unreachable :8082'; conn.className='px-3 py-1 rounded-full text-sm bg-rose-600'; }
  document.getElementById('counts').innerHTML = Object.entries(s.counts||{}).map(([k,v])=>
    `<span class="px-2 py-1 rounded ${C[k]||'bg-slate-700'}">${k} ${v}</span>`).join('');
  const r = s.ready || {leads:[], actives:[], copies:1};
  const label = r.copies>1 ? `copy ${r.copies} - next ${r.copies} leads` : 'next lead (copy off)';
  document.getElementById('ready').innerHTML =
    `<div class="text-xs uppercase tracking-wide text-sky-400 mb-1">Ready for next signal - ${label}</div>`+
    `<div class="flex flex-wrap gap-2 items-center">`+
    (r.leads.length ? r.leads.map(n=>`<span class="px-3 py-1 rounded bg-sky-600 font-semibold">${n}</span>`).join('')
                    : '<span class="text-slate-500">no fresh accounts waiting</span>')+
    (r.actives.length ? `<span class="text-slate-400 text-sm">+ copying: ${r.actives.join(', ')}</span>` : '')+
    `</div>`;
  document.getElementById('watched').innerHTML = (s.watched||[]).map(a=>`
    <div class="rounded-lg p-3 bg-slate-800 border border-slate-700 transition ${OUT[a.state]||''}">
      <div class="flex justify-between items-center mb-2">
        <span class="font-semibold">${a.name}</span>
        <span class="text-xs px-2 py-0.5 rounded ${C[a.state]||'bg-slate-700'}">${a.state}</span></div>
      <div class="text-sm text-slate-200">bal ${money(a.balance)}</div>
      <div class="text-xs text-slate-400">buffer ${money(a.buffer)} · today ${money(a.today)}</div>
    </div>`).join('') || '<div class="text-slate-500 col-span-3">no watched accounts in /accounts</div>';
  document.getElementById('live').innerHTML = (s.live||[]).map(a=>`
    <div class="rounded-lg p-3 bg-slate-800/60 border border-slate-700">
      <div class="font-semibold text-slate-200">${a.name}</div>
      <div class="text-sm text-slate-200 mt-2">bal ${money(a.balance)}</div>
      <div class="text-xs text-slate-400">DD left ${money(a.dd)} · today ${money(a.today)}</div>
    </div>`).join('') || '<div class="text-slate-500 col-span-3">no live accounts</div>';
}
setInterval(tick, 2000); tick();
</script></body></html>"""


def main():
    print(f"NQ Farm dashboard -> http://localhost:{APP_PORT}   (Ctrl-C to stop)")
    try:
        ThreadingHTTPServer(("localhost", APP_PORT), Handler).serve_forever()
    except OSError as e:
        print(f"\n*** Cannot start on :{APP_PORT} -> {e}")
        print("*** An OLD copy of the app is still running and serving the previous page.")
        print(f"*** Kill it then re-run. PowerShell:")
        print(f"***   Get-NetTCPConnection -LocalPort {APP_PORT} -State Listen | "
              "%{ Stop-Process -Id $_.OwningProcess -Force }")


if __name__ == "__main__":
    main()
