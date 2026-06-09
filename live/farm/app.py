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
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from zoneinfo import ZoneInfo

from accounts_client import fetch_accounts, _post, ORDER_URL, FLATTEN_URL
from eval_passer import EvalFarm, EState
from funded_state_machine import FundedFarm, Phase as FPhase

APP_PORT = 8090
POLL_SEC = 3.0
CAP_FLATTEN_COOLDOWN_SEC = 10.0    # don't re-spam /flatten on the same account while it closes

# Eval PLANS: per-firm daily cap + consistency thresholds, assigned PER ACCOUNT (a detected eval gets
# DEFAULT_PLAN; reassign any account in the dashboard via dropdown or drag-and-drop between sections).
#   day_cap     = max profit banked per day (also the per-trade 1:1 risk)
#   stop_new_at = day_profit at/above which an account goes DONE (takes no NEW trade today)
#   target      = pass threshold (+$3,000)
#   pass_buffer = force-close the FINAL trade at target+buffer so the realized pass lands just over the
#                 line after slippage/commission (and keeps the biggest day under the consistency %)
PLANS = {
    "futures_50":  {"label": "50% futures firms", "day_cap": 1500.0, "stop_new_at": 1400.0, "target": 3000.0, "pass_buffer": 10.0},
    "tradeify_40": {"label": "Tradeify 40%",      "day_cap": 1200.0, "stop_new_at": 1100.0, "target": 3000.0, "pass_buffer": 10.0},
}
DEFAULT_PLAN = "futures_50"
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", "eval_farm_state.json")
EOD_FLATTEN = (16, 55)             # 4:55pm ET: force-flatten any eval trade still open before the EOD
ET_ZONE = ZoneInfo("America/New_York")

# Farm-traded (through the brain). Sim accounts report DD=0 -> computed floor from `dd`.
WATCHED = {"SimEval": {"dd": 2000.0, "start_bal": 2000.0, "live_dd": False}}
# Read-only balance display only (your REAL accounts; NEVER traded).
LIVE_DISPLAY = ["MFFUSFFLX606768002", "1422474"]
# Funded farm accounts (gambling/milking). Set name prefixes once passed evals convert to funded;
# empty = no funded accounts yet, so those sections show empty.
FUNDED_PATTERN: list = []

# --- sizer: risk$ -> exact NQ + MNQ legs --------------------------------------------------------
PER_PT = 2.0           # MNQ $ per point (NQ = 10x)
MAX_QTY = 60           # sanity ceiling on the MNQ-equivalent (won't bind for normal stops)


def size_legs(risk_usd: float, stop_pts: float) -> list:
    """Exact split: 1 NQ per 10 MNQ-equivalent + MNQ for the remainder (no rounding).
    e.g. 18 -> [('NQ',1),('MNQ',8)]. Returns [(root, qty), ...]; the addon resolves the front month."""
    if stop_pts <= 0:
        return [("MNQ", 1)]
    mnq = max(1, min(MAX_QTY, round(risk_usd / (stop_pts * PER_PT))))
    n_nq, n_mnq = divmod(mnq, 10)
    legs = []
    if n_nq:
        legs.append(("NQ", n_nq))
    if n_mnq:
        legs.append(("MNQ", n_mnq))
    return legs or [("MNQ", 1)]


class FarmState:
    def __init__(self):
        # day_cap = risk per trade AND the daily cap; target = pass threshold. REAL eval values:
        # risk $1500 to make $1500 (1:1), pass at +$3000. The $2k sims have a $2k DD, so they behave
        # exactly like a 50k eval (2 wins pass, ~1.3 losses blow).
        dp = PLANS[DEFAULT_PLAN]
        self.farm = EvalFarm(copies=1, day_cap=dp["day_cap"], target=dp["target"],
                             stop_new_at=dp["stop_new_at"], pass_buffer=dp["pass_buffer"], quiet=True)
        self.farm.load_state(STATE_FILE)        # survive restarts/sleep; NT8 refreshes live numbers on sync
        self.funded = FundedFarm(quiet=True)    # gambling/milking; populated when funded accounts appear
        self.last_flatten_date = None           # EOD sweep guard (fire once/day)
        self._cap_cooldown = {}                 # account -> last cap-flatten monotonic ts (anti-spam)
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

    def _maybe_eod_flatten(self):
        """At 4:55pm ET, force-flatten every eval account once (their 1:1 brackets run to TP/SL, so
        a still-open trade is closed before the 5pm EOD; keep whatever P&L). Funded accounts are NOT
        swept here — milking follows the real strategy. Runs regardless of Start/Stop."""
        now_et = datetime.now(ET_ZONE)
        cutoff = now_et.replace(hour=EOD_FLATTEN[0], minute=EOD_FLATTEN[1], second=0, microsecond=0)
        if cutoff <= now_et < cutoff + timedelta(minutes=5) and self.last_flatten_date != now_et.date():
            self.last_flatten_date = now_et.date()
            names = list(self.farm.accounts.keys())
            print(f"[farm] {now_et:%H:%M} ET EOD sweep -> flattening {len(names)} eval account(s)")
            for name in names:
                try:
                    self.flatten(name)
                except Exception:
                    pass

    def _loop(self):
        while True:
            self._maybe_eod_flatten()           # 4:55pm ET sweep (independent of Start/Stop)
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
                # Guard against bogus reads. A disconnected / just-woke-from-sleep NT8 account
                # reports netliq=$0 while cash is still positive; feeding that $0 equity to the
                # brain false-blows it (equity <= floor). Skip the account until the read is valid.
                if not a.get("connected", True) or a.get("netliq", 0) <= 0:
                    continue
                m = {"cash": a["cash"], "equity": a["netliq"], "realized_today": a.get("realized_today", 0.0)}
                if c["live_dd"] and a.get("dd", 0) > 0:
                    m["dd"] = a["dd"]
                if "start_bal" in c:
                    m["start_bal"] = c["start_bal"]
                snap[a["name"]] = m
            with self.lock:                         # serialize brain access with fire_signal
                self.farm.sync_accounts(snap)
                watched = []
                for name, acct in self.farm.accounts.items():
                    self._apply_plan(acct, acct.plan)    # keep caps in sync with the assigned plan
                    raw = by_name.get(name, {})
                    watched.append({"name": name, "state": acct.state.value,
                                    "balance": raw.get("cash", acct.cash), "buffer": acct.buffer,
                                    "today": raw.get("realized_today", 0.0),
                                    "plan": acct.plan, "day_cap": acct.day_cap, "target": acct.target})
                live = []
                for pat in LIVE_DISPLAY:
                    row = next((a for a in accts if pat in a["name"]), None)
                    if row:
                        live.append({"name": row["name"], "balance": row["cash"], "dd": row.get("dd", 0.0),
                                     "today": row.get("realized_today", 0.0), "connected": row.get("connected", True)})
                leads, actives = self.farm.next_signal_takers()
                self.farm.save_state(STATE_FILE)
                # funded accounts -> gambling / milking sections
                fsnap = {a["name"]: {"cash": a["cash"], "equity": a["netliq"],
                                     **({"dd": a["dd"]} if a.get("dd", 0) > 0 else {})}
                         for a in accts if FUNDED_PATTERN and any(pat in a["name"] for pat in FUNDED_PATTERN)
                         and a.get("connected", True) and a.get("netliq", 0) > 0}
                self.funded.sync_accounts(fsnap)
                funded = {"gambling": [], "milking": [], "retired": []}
                for name, fa in self.funded.accounts.items():
                    key = ("gambling" if fa.phase is FPhase.GAMBLING else
                           "milking" if fa.phase is FPhase.MILKING else "retired")
                    funded[key].append({"name": name, "balance": fa.cash, "buffer": fa.buffer,
                                        "payouts": fa.payouts_done})
                plans = [{"key": k, "label": v["label"], "day_cap": v["day_cap"], "target": v["target"]}
                         for k, v in PLANS.items()]
                self.snapshot = {"watched": watched, "live": live, "counts": dict(self.farm.counts()),
                                 "ready": {"leads": leads, "actives": actives, "copies": self.farm.copies},
                                 "funded": funded, "plans": plans, "ts": datetime.now().strftime("%H:%M:%S")}
            # Cap force-close: close any open eval winner that would overshoot the daily cap / pass line
            # (computed under the lock, flattened here outside it to avoid holding the lock on network I/O).
            for name, unreal, room, day_cap, land in self._cap_flatten_candidates(by_name):
                print(f"[farm] CAP force-close {name}: open +${unreal:,.0f} >= room ${room:,.0f} "
                      f"-> lands day ~${day_cap:,.0f} (or pass ~${land:,.0f})")
                try:
                    self.flatten(name, reason="cap")
                except Exception:
                    pass
            time.sleep(POLL_SEC)

    def get(self):
        with self.lock:
            s = dict(self.snapshot)
        s["running"] = self.running
        s["addon_up"] = self.addon_up
        return s

    def fire_signal(self, strat, direction, stop_pts):
        """Route a signal through the brain (promote leads + copy to actives), size each taker, and
        place the orders on the whitelisted sims via the addon's /order. Bare market for now (no
        bracket yet — that needs the fill price). Returns what was placed."""
        with self.lock:
            routes = self.farm.route_signal(strat, date.today())
        placed = []
        for r in routes:
            acct = self.farm.accounts.get(r.account_id)
            risk = acct.day_cap if acct else self.farm.day_cap   # per-account plan risk
            for root, q in size_legs(risk, stop_pts):
                try:
                    resp = _post(ORDER_URL, {"account": r.account_id, "strat": strat,
                                             "direction": direction.upper(), "qty": q, "instrument": root,
                                             "slPts": stop_pts, "tpPts": stop_pts})  # 1:1 bracket
                except Exception as e:
                    resp = {"ok": False, "error": str(e)}
                placed.append({"account": r.account_id, "instr": root, "qty": q, "ok": resp.get("ok")})
        with self.lock:
            self.farm.save_state(STATE_FILE)
        return placed

    def flatten(self, account, reason="manual"):
        try:
            return _post(FLATTEN_URL, {"account": account, "reason": reason})
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @staticmethod
    def _apply_plan(acct, plan_key):
        """Resolve a plan key into the account's caps. Unknown key -> DEFAULT_PLAN."""
        key = plan_key if plan_key in PLANS else DEFAULT_PLAN
        p = PLANS[key]
        acct.plan = key
        acct.day_cap, acct.stop_new_at = p["day_cap"], p["stop_new_at"]
        acct.target, acct.pass_buffer = p["target"], p["pass_buffer"]

    def set_plan(self, account, plan):
        """Assign an eval account to a plan (dropdown / drag-drop). Persists immediately."""
        if plan not in PLANS:
            return {"ok": False, "error": f"unknown plan {plan}"}
        with self.lock:
            acct = self.farm.accounts.get(account)
            if acct is None:
                return {"ok": False, "error": f"unknown account {account}"}
            self._apply_plan(acct, plan)
            self.farm.save_state(STATE_FILE)
        print(f"[farm] {account} -> plan {plan} ({PLANS[plan]['label']})")
        return {"ok": True, "account": account, "plan": plan}

    def _cap_flatten_candidates(self, by_name):
        """Eval cap force-close: under the lock, find ACTIVE accounts whose OPEN-trade unrealized has
        reached the close-room (daily-cap room OR pass-line room, whichever is nearer). Returns a list
        of (name, unrealized, room) to flatten OUTSIDE the lock (network I/O). Only winners are capped;
        losing trades run to their own SL/TP/EOD. Skips bad reads and accounts on cooldown."""
        out = []
        now = time.monotonic()
        with self.lock:
            for name, acct in self.farm.accounts.items():
                if acct.state is not EState.ACTIVE:
                    continue
                raw = by_name.get(name)
                if not raw or not raw.get("connected", True) or raw.get("netliq", 0) <= 0:
                    continue
                unreal = raw["netliq"] - raw["cash"]          # open-position P&L (equity - balance)
                if unreal <= 0:
                    continue                                   # only cap winners approaching a line
                room = self.farm.cap_close_room(acct)
                if unreal >= room and (now - self._cap_cooldown.get(name, 0)) >= CAP_FLATTEN_COOLDOWN_SEC:
                    self._cap_cooldown[name] = now
                    out.append((name, unreal, room, acct.day_cap, acct.target + acct.pass_buffer))
        return out


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
        elif self.path == "/api/fire_signal":
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n).decode() if n else "{}")
            placed = STATE.fire_signal(p.get("strat", "OD"), p.get("direction", "LONG"),
                                       float(p.get("stop_pts", 30)))
            self._send(200, json.dumps({"ok": True, "placed": placed}))
        elif self.path == "/api/signal":          # live signal broadcast from the coordinator
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n).decode() if n else "{}")
            placed = STATE.fire_signal(p.get("strat", "OD"), p.get("direction", "LONG"),
                                       float(p.get("stop_pts", 30)))
            self._send(200, json.dumps({"ok": True, "placed": placed}))
        elif self.path == "/api/flatten":
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n).decode() if n else "{}")
            self._send(200, json.dumps(STATE.flatten(p.get("account", ""))))
        elif self.path == "/api/set_plan":
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n).decode() if n else "{}")
            self._send(200, json.dumps(STATE.set_plan(p.get("account", ""), p.get("plan", ""))))
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
  <div id="ready" class="mb-3 p-3 rounded-lg bg-slate-800 border border-sky-700"></div>
  <div class="mb-6 flex items-center gap-2 text-sm">
    <span class="text-slate-400">Fire test signal:</span>
    <select id="fStrat" class="bg-slate-700 rounded px-2 py-1"><option>OD</option><option>RV</option><option>B2</option><option>FB</option></select>
    <select id="fDir" class="bg-slate-700 rounded px-2 py-1"><option>LONG</option><option>SHORT</option></select>
    <span class="text-slate-400">stop</span>
    <input id="fStop" value="30" class="bg-slate-700 rounded px-2 py-1 w-14" title="stop distance in points - sizes the position (qty = risk / stop_pts / $2)">
    <span class="text-slate-500 text-xs">pts</span>
    <button onclick="fire()" class="px-3 py-1 rounded bg-indigo-600 hover:bg-indigo-500 font-semibold">Fire</button>
    <span id="fResult" class="text-slate-400"></span>
  </div>
  <h2 class="text-lg font-semibold mb-2 text-slate-300">Eval accounts
    <span class="text-slate-500 text-sm">(farm-traded · drag a card to another plan, or use its dropdown)</span></h2>
  <div id="evalSections" class="mb-8"></div>
  <h2 class="text-lg font-semibold mb-2 text-amber-300">Funded - gambling</h2>
  <div id="gambling" class="grid grid-cols-2 md:grid-cols-3 gap-3 mb-6"></div>
  <h2 class="text-lg font-semibold mb-2 text-emerald-300">Funded - milking</h2>
  <div id="milking" class="grid grid-cols-2 md:grid-cols-3 gap-3 mb-8"></div>
  <h2 class="text-lg font-semibold mb-2 text-slate-300">Live accounts <span class="text-slate-500 text-sm">(read-only, never traded)</span></h2>
  <div id="live" class="grid grid-cols-2 md:grid-cols-3 gap-3"></div>
</div>
<script>
const C = {FRESH:'bg-slate-600', ACTIVE:'bg-sky-600', DONE:'bg-amber-600', PASSED:'bg-emerald-600', BLOWN:'bg-rose-700'};
const OUT = {PASSED:'ring-2 ring-emerald-400 shadow-lg shadow-emerald-500/40', BLOWN:'ring-2 ring-rose-500 shadow-lg shadow-rose-500/40'};
async function ctl(a){ await fetch('/api/'+a,{method:'POST'}); tick(); }
async function flat(acct){
  await fetch('/api/flatten',{method:'POST',body:JSON.stringify({account:acct})});
  tick();
}
function evalCard(a, plans){
  const opts = (plans||[]).map(p=>`<option value="${p.key}" ${p.key===a.plan?'selected':''}>${p.label}</option>`).join('');
  return `<div draggable="true" ondragstart="event.dataTransfer.setData('text/plain','${a.name}')"
      class="rounded-lg p-3 bg-slate-800 border border-slate-700 transition cursor-move ${OUT[a.state]||''}">
      <div class="flex justify-between items-center mb-2">
        <span class="font-semibold">${a.name}</span>
        <span class="text-xs px-2 py-0.5 rounded ${C[a.state]||'bg-slate-700'}">${a.state}</span></div>
      <div class="text-sm text-slate-200">bal ${money(a.balance)}</div>
      <div class="flex justify-between items-center mt-1">
        <span class="text-xs text-slate-400">buffer ${money(a.buffer)} · today ${money(a.today)}</span>
        <button onclick="flat('${a.name}')" class="text-xs px-2 py-0.5 rounded bg-rose-800 hover:bg-rose-700">flatten</button>
      </div>
      <select onchange="setPlan('${a.name}', this.value)" class="mt-2 w-full bg-slate-700 rounded px-1 py-1 text-xs">${opts}</select>
    </div>`;
}
async function setPlan(acct, plan){
  await fetch('/api/set_plan',{method:'POST',body:JSON.stringify({account:acct, plan})});
  tick();
}
async function drop(ev, plan){
  ev.preventDefault();
  const acct = ev.dataTransfer.getData('text/plain');
  if(acct) await setPlan(acct, plan);
}
async function fire(){
  const body = {strat: fStrat.value, direction: fDir.value, stop_pts: parseFloat(fStop.value)};
  fResult.textContent = 'firing...';
  try {
    const r = await (await fetch('/api/fire_signal',{method:'POST',body:JSON.stringify(body)})).json();
    const ok = (r.placed||[]).filter(x=>x.ok).map(x=>x.account+' '+x.qty+x.instr);
    fResult.textContent = ok.length ? ('placed: '+ok.join(', ')) : 'nothing routed';
  } catch(e){ fResult.textContent = 'error'; }
  tick();
}
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
  const label = r.copies>1 ? `copy ${r.copies} (round-robin, ${r.copies} at a time)` : 'de-correlated (1 account)';
  document.getElementById('ready').innerHTML =
    `<div class="text-xs uppercase tracking-wide text-sky-400 mb-1">Ready for next signal - ${label}</div>`+
    `<div class="flex flex-wrap gap-2 items-center">`+
    (r.leads.length ? r.leads.map(n=>`<span class="px-3 py-1 rounded bg-sky-600 font-semibold">${n}</span>`).join('')
                    : '<span class="text-slate-500">no accounts ready</span>')+
    (r.actives.length ? `<span class="text-slate-400 text-sm">+ copying: ${r.actives.join(', ')}</span>` : '')+
    `</div>`;
  const plans = s.plans || [];
  const byPlan = {};
  (s.watched||[]).forEach(a=>{ (byPlan[a.plan]=byPlan[a.plan]||[]).push(a); });
  document.getElementById('evalSections').innerHTML = plans.map(p=>{
    const accts = byPlan[p.key]||[];
    return `<div class="mb-5" ondragover="event.preventDefault()" ondrop="drop(event,'${p.key}')">
      <div class="flex items-baseline gap-2 mb-2">
        <h3 class="text-base font-semibold text-slate-200">${p.label}</h3>
        <span class="text-xs text-slate-500">cap ${money(p.day_cap)} · pass ${money(p.target)} · ${accts.length} acct(s)</span>
      </div>
      <div class="grid grid-cols-2 md:grid-cols-3 gap-3 rounded-lg border border-dashed border-slate-700 p-2 min-h-[64px]">
      ${accts.map(a=>evalCard(a, plans)).join('') ||
        '<div class="text-slate-600 col-span-3 text-sm text-center py-3">drop accounts here</div>'}
      </div></div>`;
  }).join('') || '<div class="text-slate-500">no eval accounts detected</div>';
  const f = s.funded || {gambling:[], milking:[]};
  const fcard = (a,b)=>`<div class="rounded-lg p-3 bg-slate-800 border ${b}"><div class="font-semibold">${a.name}</div>`+
    `<div class="text-sm text-slate-200 mt-1">bal ${money(a.balance)}</div>`+
    `<div class="text-xs text-slate-400">buffer ${money(a.buffer)} · payouts ${a.payouts}</div></div>`;
  document.getElementById('gambling').innerHTML = (f.gambling||[]).length ? f.gambling.map(a=>fcard(a,'border-amber-700')).join('') : '<div class="text-slate-500 col-span-3">no gambling accounts yet</div>';
  document.getElementById('milking').innerHTML = (f.milking||[]).length ? f.milking.map(a=>fcard(a,'border-emerald-700')).join('') : '<div class="text-slate-500 col-span-3">no milking accounts yet</div>';
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
