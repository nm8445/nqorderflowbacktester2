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

from accounts_client import fetch_accounts, _post, send_heartbeat, ORDER_URL, FLATTEN_URL
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
# consistency = biggest day must be <= this fraction of total at pass time -> true pass target =
#   max(target, biggest_day / consistency). milk_at = total at/above which an account leaves the
#   $-bracket rotation and milks 1 MNQ 4-strat up to that (consistency-adjusted) target.
PLANS = {
    "futures_50":  {"label": "50% futures firms", "day_cap": 1500.0, "stop_new_at": 1400.0, "target": 3000.0, "pass_buffer": 10.0, "consistency": 0.50, "milk_at": 2800.0},
    "tradeify_40": {"label": "Tradeify 40%",      "day_cap": 1200.0, "stop_new_at": 1100.0, "target": 3000.0, "pass_buffer": 10.0, "consistency": 0.40, "milk_at": 2800.0},
}
DEFAULT_PLAN = "futures_50"
MILK_QTY = 1           # eval-milking size: fixed 1 MNQ per strat (4-strat combined)
# EXIT-broadcast reasons that mean a SESSION FORCE-CLOSE (close everyone holding the strat: eval
# takers + funded gamblers + milkers). Everything else is a DYNAMIC exit -> close MILK only (gamblers
# and eval-takers run to their own static bracket()). Reasons per engine: RV force_close/stop/target,
# B2 TP_FIXED/SL_TRAIL/FORCE_CLOSE/EOD, OD TP_GREEN/SL_YELLOW/FORCE_CLOSE, FB SL/TP/EOD.
FORCE_CLOSE_REASONS = {"force_close", "FORCE_CLOSE", "EOD"}
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", "eval_farm_state.json")
FUNDED_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", "funded_farm_state.json")
# The live combined system (active_contract.py) writes the feed's real contract month here; we stamp
# orders with it so the farm executes the SAME contract the data feed is on (no roll mismatch). If it's
# missing/stale, we send the bare root and the test addon appends its own FrontMonth() — graceful.
ACTIVE_CONTRACT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "combined", "state", "active_contract.json")
EOD_FLATTEN = (16, 55)             # 4:55pm ET: force-flatten any eval trade still open before the EOD
# Per-strat EFFECTIVE force-close times (wall-clock when the live combined actually flattens, i.e. when
# the relevant bar CLOSES). The farm only places each strat's SL/TP and never hears the force-close exit
# (farm_broadcast is ENTRY-only), so we sweep each strat at its own time (strat-scoped — others keep
# running). Derived from the engines:
#   FB  EOD_TIME=14:00 on 5-min closes               -> 14:00
#   RV  threshold 14:45 on 20-min closes (…14:40,15:00) -> 15:00
#   B2  bar OPEN>=16:00, exit at that bar's close     -> 16:20  (MT5 should match NT8, not run 20m later)
#   OD  bar OPEN==08:00, exit at that bar's close     -> 08:20  (overnight: enters 19:00, FC next AM)
FB_FLATTEN = (14, 0)
RV_FLATTEN = (15, 0)
B2_FLATTEN = (16, 20)
OD_FLATTEN = (8, 20)
# OD is an OVERNIGHT trade: enters 19:00 ET (Asia) and force-closes ~08:20 ET the NEXT morning (NY pre-
# market) if still open. The sweep must run in the morning. We bound it to a morning window rather than
# open-ended ">= 08:20" because the once-per-CALENDAR-day guard keys on the date: on an evening cold-start
# (e.g. 22:00, after that day's 19:00 entry) 22:00 is also ">= 08:20 of the same date" and would wrongly
# flatten the just-entered overnight position. The window confines the sweep to the AM.
OD_FLATTEN_WINDOW_MIN = 240        # 08:20–12:20 ET
# 6:05pm ET: roll the brain to the new trading day (DONE->ACTIVE, reset day tallies, funded winning-days
# + payouts). Must be AFTER the firm's daily reset of realized_today (5pm settle / 6pm CME reopen) so the
# next sync doesn't re-read today's profit and bounce accounts straight back to DONE, and BEFORE OD's
# 7pm ET entry so reset accounts are eligible. Bump later if your firm resets realized_today after 6pm.
EOD_ROLL = (18, 5)
ET_ZONE = ZoneInfo("America/New_York")

# Farm-traded (through the brain). Sim accounts report DD=0 -> computed floor from `dd`.
# Real Lucid 50k evals (LFE...0005..0014): $50k start, $2k max loss, traded on the futures_50 plan; live_dd
# True reads NT8's reported remaining DD (falls back to the computed trailing floor if the addon reports 0).
WATCHED = {"SimEval": {"dd": 2000.0, "start_bal": 2000.0, "live_dd": False},
           "LFE0506857848": {"dd": 2000.0, "start_bal": 50000.0, "live_dd": True}}
# Read-only balance display only (your REAL accounts; NEVER traded).
LIVE_DISPLAY = ["MFFUSFFLX606768002", "1422474"]
# Funded farm accounts (gambling/milking). Set name prefixes once passed evals convert to funded;
# empty = no funded accounts yet, so those sections show empty. Match is a case-sensitive SUBSTRING
# of the NT8 account name, so "Sim funded" matches "Sim funded 1", "Sim funded 2", ...
FUNDED_PATTERN: list = ["Simfunded", "LFF0506857848"]   # LFF0506857848... = real Lucid funded (LFF...0001, ...0002)

# --- sizer: risk$ -> exact NQ + MNQ legs --------------------------------------------------------
PER_PT = 2.0           # MNQ $ per point (NQ = 10x)
MAX_QTY = 60           # sanity ceiling on the MNQ-equivalent (won't bind for normal stops)

# LucidFlex 50k contract caps (micro-equivalent; 1 mini = 10 micro). Cap-at-max: a trade that would need
# more contracts than allowed is sized DOWN to the cap (risks less than the target $), never exceeding it.
LUCID50K_EVAL_MAX = 40                       # eval: flat 4 mini / 40 micro, no scaling
def lucid50k_funded_cap(profit_at_session: float) -> int:
    """Funded scaling tier by simulated profit AS OF THE LAST SESSION CLOSE (Lucid updates EOD, not live)."""
    if profit_at_session < 1000: return 20   # 2 mini / 20 micro
    if profit_at_session < 2000: return 30   # 3 mini / 30 micro
    return 40                                # 4 mini / 40 micro


TIGHT_RISK = 750.0       # DD-aware tight-tier risk while remaining DD (buffer) < RESUME_DD
RESUME_DD = 2000.0       # remaining DD ($) at/above which full-size risk resumes
AUTO_BLOW_DD = 100.0     # remaining DD ($) below which a FLAT account is auto-marked blown (firm-dead)


def tier_risk(buffer_usd: float, mode: str, full: float) -> float:
    """DD-aware per-trade risk. 'auto' -> $750 while remaining DD (buffer) < $2000, else `full`.
    '750'/'1500' force the tight/full tier (manual per-account override). Flat $750 even at a tiny
    buffer (no cap) — user's rule; the mark-blown control handles truly-dead accounts."""
    tight = min(TIGHT_RISK, full)
    if mode == "750":
        return tight
    if mode == "1500":
        return full
    return tight if buffer_usd < RESUME_DD else full


def size_legs(risk_usd: float, stop_pts: float, max_micros: int = MAX_QTY) -> list:
    """Exact split: 1 NQ per 10 MNQ-equivalent + MNQ for the remainder (no rounding).
    e.g. 18 -> [('NQ',1),('MNQ',8)]. Returns [(root, qty), ...]; the addon resolves the front month.
    `max_micros` caps the MNQ-equivalent at the firm's contract limit (LucidFlex eval / funded scaling):
    when a tight stop would need more contracts than allowed, it sizes DOWN here (risks less, never over)."""
    if stop_pts <= 0:
        return [("MNQ", 1)]
    # round to the NEAREST contract but break ties DOWN (never over-risk): e.g. target $750 with a
    # high-vol $500/1-MNQ stop -> 1.5 contracts -> 1 MNQ ($500), not 2 ($1000).
    raw = risk_usd / (stop_pts * PER_PT)
    nearest = int(raw) + (1 if (raw - int(raw)) > 0.5 else 0)
    mnq = max(1, min(MAX_QTY, int(max_micros), nearest))
    n_nq, n_mnq = divmod(mnq, 10)
    legs = []
    if n_nq:
        legs.append(("NQ", n_nq))
    if n_mnq:
        legs.append(("MNQ", n_mnq))
    return legs or [("MNQ", 1)]


def _active_contract_code():
    """Feed's current contract month ('MM-YY') from the live system's cache, only if it's today's.
    None -> caller sends the bare root and the addon appends its own FrontMonth()."""
    try:
        d = json.loads(open(ACTIVE_CONTRACT_FILE).read())
        if d.get("date") == date.today().isoformat() and d.get("code"):
            return d["code"]
    except Exception:
        pass
    return None


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
        self.funded.load_state(FUNDED_STATE_FILE)   # restore manual phase locks + payout progress
        self.last_flatten_date = None           # EOD sweep guard (fire once/day)
        self.last_eod_roll_date = None          # EOD brain-roll guard (fire once/day)
        self.last_strat_flatten_date = {"FB": None, "RV": None, "B2": None, "OD": None}  # per-strat FC guards
        self._cap_cooldown = {}                 # account -> last cap-flatten monotonic ts (anti-spam)
        self._dead_polls = {}                   # account -> consecutive flat+DD<$100 polls (auto-blow debounce)
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
        """At 4:55pm ET, force-flatten every farm account once (eval + funded) before the 5pm EOD; keep
        whatever P&L. Backstop to the EXIT-broadcast force-close (handle_exit) and the per-strat sweeps —
        a safe no-op on anything already flat. Runs regardless of Start/Stop."""
        now_et = datetime.now(ET_ZONE)
        cutoff = now_et.replace(hour=EOD_FLATTEN[0], minute=EOD_FLATTEN[1], second=0, microsecond=0)
        if cutoff <= now_et < cutoff + timedelta(minutes=5) and self.last_flatten_date != now_et.date():
            self.last_flatten_date = now_et.date()
            names = list(self.farm.accounts.keys()) + list(self.funded.accounts.keys())
            print(f"[farm] {now_et:%H:%M} ET EOD sweep -> flattening {len(names)} account(s) (eval+funded)")
            for name in names:
                try:
                    self.flatten(name)
                except Exception:
                    pass

    def _maybe_strat_flatten(self, strat, hhmm, window_min=None):
        """Once/day at/after `hhmm` ET, flatten one strat across all farm accounts to mirror the live
        combined's force-close (the farm never hears the exit — broadcast is ENTRY-only). Strat-scoped
        (addon /flatten with strat=...) so other strats keep running, and a no-op on already-closed
        positions. Open-ended (>= cutoff) by default so a late poll still sweeps; `window_min` bounds it
        to [cutoff, cutoff+window_min) for OD — an overnight trade (enters 19:00 ET, force-closes the next
        morning ~08:20). Because the guard keys on the calendar date, an evening cold-start (22:00 is also
        >= 08:20 of the same date) would otherwise flatten that night's fresh entry; the morning window
        prevents that. Runs regardless of Start/Stop."""
        now_et = datetime.now(ET_ZONE)
        if self.last_strat_flatten_date.get(strat) == now_et.date():
            return
        cutoff = now_et.replace(hour=hhmm[0], minute=hhmm[1], second=0, microsecond=0)
        if now_et < cutoff:
            return
        if window_min is not None and now_et >= cutoff + timedelta(minutes=window_min):
            return
        self.last_strat_flatten_date[strat] = now_et.date()
        names = list(self.farm.accounts.keys()) + list(self.funded.accounts.keys())
        print(f"[farm] {now_et:%H:%M} ET {strat} force-close -> sweeping {strat} on {len(names)} account(s)")
        for name in names:
            try:
                self.flatten_strat(name, strat, f"{strat.lower()}_force_close")
            except Exception:
                pass

    def _maybe_eod_roll(self):
        """Once per day at/after 6:05pm ET, roll both brains to the new trading day. This is the only
        thing that returns DONE eval accounts to ACTIVE (and resets their day tally) and that tallies
        funded winning-days / fires payouts. Without it, an eval that caps once stays DONE forever and
        is never eligible for the next signal. Open-ended (>= cutoff) so a late app start still rolls.
        Runs regardless of Start/Stop, mirroring the EOD flatten."""
        now_et = datetime.now(ET_ZONE)
        cutoff = now_et.replace(hour=EOD_ROLL[0], minute=EOD_ROLL[1], second=0, microsecond=0)
        if now_et < cutoff or self.last_eod_roll_date == now_et.date():
            return
        with self.lock:
            self.last_eod_roll_date = now_et.date()
            done_before = sum(1 for a in self.farm.accounts.values() if a.state is EState.DONE)
            self.farm.end_of_day(now_et.date())
            self.funded.end_of_day(now_et.date())
            self.farm.save_state(STATE_FILE)
            self.funded.save_state(FUNDED_STATE_FILE)
        print(f"[farm] {now_et:%H:%M} ET EOD roll -> new trading day "
              f"({done_before} eval acct(s) reset DONE->ACTIVE; funded winning-days tallied)")

    def _loop(self):
        while True:
            try:                                # liveness ping -> addon watchdog flattens if we go silent
                send_heartbeat()                # (protects backend-managed milk legs: OD no-rest / B2 green-only)
            except Exception:
                pass
            self._maybe_eod_flatten()           # 4:55pm ET account-wide sweep (independent of Start/Stop)
            self._maybe_strat_flatten("FB", FB_FLATTEN)                              # 14:00 ET
            self._maybe_strat_flatten("RV", RV_FLATTEN)                              # 15:00 ET
            self._maybe_strat_flatten("B2", B2_FLATTEN)                              # 16:20 ET
            self._maybe_strat_flatten("OD", OD_FLATTEN, window_min=OD_FLATTEN_WINDOW_MIN)  # 08:20 ET (AM)
            self._maybe_eod_roll()              # 6:05pm ET brain roll (DONE->ACTIVE, winning-days)
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
                # Prune accounts deleted in NT8: absent from the live feed ENTIRELY (a merely DISCONNECTED
                # account is still listed in by_name with connected=false, so it's kept). Only when the
                # feed is valid (non-empty), so a momentary addon glitch can't wipe the roster.
                if accts:
                    for nm in [n for n in self.farm.accounts if n not in by_name]:
                        del self.farm.accounts[nm]
                watched = []
                for name, acct in self.farm.accounts.items():
                    self._apply_plan(acct, acct.plan)    # keep caps in sync with the assigned plan
                    raw = by_name.get(name, {})
                    watched.append({"name": name, "state": acct.state.value,
                                    "balance": raw.get("cash", acct.cash), "buffer": acct.buffer,
                                    "nt8_dd": acct.dd_remaining,
                                    "today": raw.get("realized_today", 0.0),
                                    "plan": acct.plan, "day_cap": acct.day_cap, "target": acct.target,
                                    "total": acct.total, "true_target": acct.true_target,
                                    "peak_day": acct.peak_day_profit, "manual": acct.manual_phase,
                                    "risk_mode": acct.risk_mode})
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
                if accts:                            # prune funded accounts deleted in NT8 (see eval prune above)
                    for nm in [n for n in self.funded.accounts if n not in by_name]:
                        del self.funded.accounts[nm]
                # one-trade-at-a-time: a funded account holding an open position is skipped for new signals.
                # Refreshed from the live /accounts positions each poll (placement also marks it immediately).
                self.funded.busy_ids = {a["name"] for a in accts
                                        if FUNDED_PATTERN and any(p in a["name"] for p in FUNDED_PATTERN)
                                        and a.get("positions")}
                # AUTO-BLOW: an in-play account whose live remaining DD (buffer) has dropped below $100
                # while FLAT (its trade has closed) is effectively dead — the firm will/has force-closed
                # it. Mark it BLOWN (sticky) so it drops out of rotation + sizing. Skips PASSED (NT8
                # under-reads those) and any account holding an open position (wait for the close).
                # DEBOUNCED: requires the condition on DEAD_POLLS consecutive reads so one transient bad
                # NT8 read can't wrongly retire a live account (the false-positive direction).
                DEAD_POLLS = 2
                def _dead_now(nm, buf, raw, terminal):
                    if (terminal or not raw or not raw.get("connected", True)
                            or raw.get("netliq", 0) <= 0 or raw.get("positions") or buf >= AUTO_BLOW_DD):
                        self._dead_polls.pop(nm, None)     # condition broken -> reset the streak
                        return False
                    self._dead_polls[nm] = self._dead_polls.get(nm, 0) + 1
                    return self._dead_polls[nm] >= DEAD_POLLS
                auto_blown = []
                for nm, a in list(self.farm.accounts.items()):
                    if _dead_now(nm, a.buffer, by_name.get(nm),
                                 a.state in (EState.BLOWN, EState.PASSED) or a.manual_blown):
                        self.farm.mark_blown(nm); auto_blown.append(nm)
                for nm, fa in list(self.funded.accounts.items()):
                    if _dead_now(nm, fa.buffer, by_name.get(nm),
                                 fa.phase is FPhase.RETIRED or fa.manual_blown):
                        self.funded.mark_blown(nm); auto_blown.append(nm)
                if auto_blown:
                    self.farm.save_state(STATE_FILE)
                    self.funded.save_state(FUNDED_STATE_FILE)
                    print(f"[farm] AUTO-BLOWN (remaining DD < ${AUTO_BLOW_DD:.0f} & flat): {', '.join(auto_blown)}")
                # De-risk GAMBLING -> MILKING once an account has banked its buffer (derisk_at, default
                # = its DD = the floor-lock point). Balance-driven (the app has no per-trade close hook),
                # so NOT "any profit". Manual phase locks are respected.
                for fa in self.funded.accounts.values():
                    if (fa.phase is FPhase.GAMBLING and not fa.manual_phase
                            and self.funded._should_derisk(fa)):
                        fa.phase = FPhase.MILKING
                        print(f"[farm] funded {fa.id}: banked +${fa.cash-fa.start_bal:,.0f} "
                              f"(>= ${self.funded._derisk_target(fa):,.0f}) -> DE-RISK -> MILKING")
                self.funded.save_state(FUNDED_STATE_FILE)
                funded = {"gambling": [], "milking": [], "retired": []}
                for name, fa in self.funded.accounts.items():
                    key = ("gambling" if fa.phase is FPhase.GAMBLING else
                           "milking" if fa.phase is FPhase.MILKING else "retired")
                    funded[key].append({"name": name, "balance": fa.cash, "buffer": fa.buffer,
                                        "nt8_dd": fa.dd_remaining,
                                        "payouts": fa.payouts_done, "manual": fa.manual_phase,
                                        "risk_mode": fa.risk_mode, "reason": fa.retired_reason,
                                        "blown": fa.manual_blown,
                                        "forced": self.funded.forced_gambler_id == name})
                plans = [{"key": k, "label": v["label"], "day_cap": v["day_cap"], "target": v["target"]}
                         for k, v in PLANS.items()]
                # Milkers that will COPY the next signal (they take every signal, not a rotation): eval
                # milkers past the consistency gate (or manually locked), + every funded milker.
                milkers_ready = ([a.id for a in self.farm.accounts.values()
                                  if a.state is EState.MILKING
                                  and (a.manual_phase or a.day_profit < a.peak_day_profit - 1e-9)]
                                 + [fa.id for fa in self.funded.accounts.values()
                                    if fa.phase is FPhase.MILKING])
                self.snapshot = {"watched": watched, "live": live, "counts": dict(self.farm.counts()),
                                 "ready": {"leads": leads, "actives": actives, "copies": self.farm.copies},
                                 "milkers_ready": milkers_ready,
                                 "funded": funded, "funded_risk": self.funded.gamble_risk,
                                 "funded_derisk": self.funded.derisk_at, "funded_milk": self.funded.milk_qty,
                                 "funded_lock": self.funded.profit_lock,
                                 "plans": plans, "ts": datetime.now().strftime("%H:%M:%S")}
            # Cap force-close: close any open eval winner that would overshoot the daily cap / pass line
            # (computed under the lock, flattened here outside it to avoid holding the lock on network I/O).
            for name, unreal, room, day_cap, land in self._cap_flatten_candidates(by_name):
                print(f"[farm] CAP force-close {name}: open +${unreal:,.0f} >= room ${room:,.0f} "
                      f"-> lands day ~${day_cap:,.0f} (or pass ~${land:,.0f})")
                try:
                    self.flatten(name, reason="cap")
                except Exception:
                    pass
            # Funded profit-lock: force-close a GAMBLING account's open trade once its account profit
            # (equity - start_bal) reaches the selected lock level (mirrors the eval cap, banks the buffer).
            for name, profit, lock in self._funded_lock_candidates(by_name):
                print(f"[farm] funded PROFIT-LOCK {name}: +${profit:,.0f} >= ${lock:,.0f} -> force-close gamble")
                try:
                    self.flatten(name, reason="profit_lock")
                except Exception:
                    pass
            time.sleep(POLL_SEC)

    def get(self):
        with self.lock:
            s = dict(self.snapshot)
        s["running"] = self.running
        s["addon_up"] = self.addon_up
        return s

    def fire_signal(self, strat, direction, stop_pts, sl_price=None, tp_price=None,
                    green=None, fb_tp=None):
        """Route a signal through the brain (promote leads + copy to actives), size each taker, and
        place the orders on the whitelisted sims via the addon's /order. Returns what was placed.

        Bracket mode (GAMBLERS + EVAL-TAKERS): if absolute sl_price/tp_price are supplied (RV — natively
        1:1, so we mirror the engine's exact stop/target levels) the addon attaches a resting OCO at those
        prices. Otherwise it builds a 1:1 bracket off the fill (slPts=tpPts=stop_pts, OD/B2/FB — their
        native targets aren't 1:1). stop_pts still drives position sizing either way.

        MILK routes (eval + funded MILKING) instead use milk_shape() -> the :8081 per-strat shape, closed
        by the EXIT broadcast (STATE.handle_exit). `green` (B2) and `fb_tp` (FB) carry the native levels."""
        absolute = sl_price is not None and tp_price is not None
        def bracket():
            if absolute:
                return {"slPrice": round(float(sl_price), 2), "tpPrice": round(float(tp_price), 2)}
            return {"slPts": stop_pts, "tpPts": stop_pts}
        def milk_shape():
            """MILKING order shape — mirror the live 4-way combined (:8081), NOT the 1:1 bracket:
            B2 = green-TP resting limit only (yellow closed by the EXIT broadcast); OD = no resting orders
            (fully EXIT-broadcast / force-close-swept); RV/FB = their native brackets. A missing native
            level falls back to bracket() so a manual test fire still gets a protective order."""
            if strat == "B2":
                return {"tpPrice": round(float(green), 2)} if green is not None else bracket()
            if strat == "OD":
                return {}   # no resting orders (parity with :8081 OD — closed by handle_exit/sweeps)
            if strat == "FB":
                # TP-only resting limit (like B2). The yellow stop is engine-managed and closes FB milk
                # legs via the EXIT broadcast (SL_YELLOW = milk-only); the flatten cancels this resting TP.
                return {"tpPrice": round(float(fb_tp), 2)} if fb_tp is not None else bracket()
            return bracket()   # RV native bracket == the absolute sl/tp already in bracket()
        with self.lock:
            routes = self.farm.route_signal(strat, date.today())
        code = _active_contract_code()
        instr = lambda root: f"{root} {code}" if code else root   # stamp the feed's real contract
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        placed = []
        for r in routes:
            acct = self.farm.accounts.get(r.account_id)
            # DD-aware two-tier: $750 while remaining DD < $2k, else the plan's full risk (day_cap).
            # Per-account risk_mode ('auto'|'750'|'1500') overrides. Sized to the account's live buffer.
            risk = tier_risk(acct.buffer, acct.risk_mode, acct.day_cap) if acct else self.farm.day_cap
            for root, q in size_legs(risk, stop_pts, max_micros=LUCID50K_EVAL_MAX):   # 50k eval cap = 40 micro
                # UNIQUE tag per leg (incl. root). Without an explicit tag the addon auto-generates the
                # SAME {strat}_{timestamp}_{dir} for both NQ and MNQ legs of a $1500 trade, so they share
                # one OCO id — when one leg hits SL/TP, NT8 cancels the OTHER leg's bracket too, leaving
                # it open with no stop (bleeds forever). A per-leg tag gives each its own OCO bracket.
                leg_tag = f"{strat}_{ts}_{direction.upper()}_{r.account_id}_{root}"
                try:
                    resp = _post(ORDER_URL, {"account": r.account_id, "strat": strat, "tag": leg_tag,
                                             "direction": direction.upper(), "qty": q, "instrument": instr(root),
                                             **bracket()})  # absolute (RV) or 1:1-off-fill
                except Exception as e:
                    resp = {"ok": False, "error": str(e)}
                placed.append({"account": r.account_id, "instr": root, "qty": q, "ok": resp.get("ok")})
        # Eval-milking feed: every MILKING account also takes this strat at a fixed 1 MNQ (4-strat
        # combined -> gently walk total up to the consistency target). Skip if today's profit already
        # reached the biggest day (don't grow a new biggest day on the same day as the big win).
        with self.lock:
            # AUTO milkers respect the consistency ceiling (don't grow a new biggest day); a MANUALLY
            # locked milker (dashboard -> Milk) milks unconditionally.
            milkers = [a.id for a in self.farm.accounts.values()
                       if a.state is EState.MILKING
                       and (a.manual_phase or a.day_profit < a.peak_day_profit - 1e-9)]
        for aid in milkers:
            try:
                resp = _post(ORDER_URL, {"account": aid, "strat": strat, "direction": direction.upper(),
                                         "qty": MILK_QTY, "instrument": instr("MNQ"),
                                         **milk_shape()})
            except Exception as e:
                resp = {"ok": False, "error": str(e)}
            placed.append({"account": aid, "instr": "MNQ", "qty": MILK_QTY, "ok": resp.get("ok"), "milk": True})
        # Funded farm: route the gamble (ONE decorrelated GAMBLING account, sized to its whole buffer)
        # + milk (every funded MILKING account at 1 MNQ). Gamble->milk de-risk is balance-driven in _loop.
        with self.lock:
            # capture the gamblers' pre-route last_gamble_date so we can ROLL BACK if the order fails to
            # place (route_signal optimistically marks 'gambled today' before placement — without this a
            # failed/no-fill gamble would burn the 1/day slot with no trade).
            prev_lgd = {aid: a.last_gamble_date for aid, a in self.funded.accounts.items()}
            froutes = self.funded.route_signal(strat, date.today())
        for r in froutes:
            if r.intent == "GAMBLE":
                facct = self.funded.accounts.get(r.account_id)
                buf = facct.buffer if facct else 1500.0
                mode = facct.risk_mode if facct else "auto"
                # DD-aware two-tier (same rule as evals): $750 while buffer < $2k, $1500 once >= $2k.
                risk = tier_risk(buf, mode, 1500.0)
                if self.funded.gamble_risk:            # global manual gamble-risk cap still applies if set
                    risk = min(risk, self.funded.gamble_risk)
                cap = lucid50k_funded_cap(facct.scaling_profit if facct else 0.0)   # scaling tier (EOD-based)
                gamble_ok = False
                for root, q in size_legs(risk, stop_pts, max_micros=cap):
                    leg_tag = f"{strat}_{ts}_{direction.upper()}_{r.account_id}_{root}_FG"
                    try:
                        resp = _post(ORDER_URL, {"account": r.account_id, "strat": strat, "tag": leg_tag,
                                                 "direction": direction.upper(), "qty": q,
                                                 "instrument": instr(root), **bracket()})
                    except Exception as e:
                        resp = {"ok": False, "error": str(e)}
                    if resp.get("ok"):
                        gamble_ok = True
                    placed.append({"account": r.account_id, "instr": root, "qty": q,
                                   "ok": resp.get("ok"), "gamble": True})
                if not gamble_ok:        # nothing placed -> don't burn the daily gamble slot; allow retry
                    with self.lock:
                        a2 = self.funded.accounts.get(r.account_id)
                        if a2 is not None:
                            a2.last_gamble_date = prev_lgd.get(r.account_id)
                    print(f"[farm] funded GAMBLE {r.account_id}: order placed NOTHING "
                          f"-> rolled back last_gamble_date (retryable)")
                else:                    # placed -> mark busy so it won't take a 2nd position this cycle
                    with self.lock:
                        self.funded.busy_ids.add(r.account_id)
            else:  # MILK -> funded milk size (settable, default 1 MNQ), capped at the scaling tier
                facct = self.funded.accounts.get(r.account_id)
                cap = lucid50k_funded_cap(facct.scaling_profit if facct else 0.0)
                mq = min(self.funded.milk_qty, cap)
                try:
                    resp = _post(ORDER_URL, {"account": r.account_id, "strat": strat,
                                             "direction": direction.upper(), "qty": mq,
                                             "instrument": instr("MNQ"), **milk_shape()})
                except Exception as e:
                    resp = {"ok": False, "error": str(e)}
                # NOTE: do NOT mark milkers busy — they run the full 4-way concurrently (take every signal).
                # busy_ids is a GAMBLER-only one-trade-at-a-time gate.
                placed.append({"account": r.account_id, "instr": "MNQ", "qty": mq,
                               "ok": resp.get("ok"), "fmilk": True})
        with self.lock:
            self.farm.save_state(STATE_FILE)
            self.funded.save_state(FUNDED_STATE_FILE)
        return placed

    def handle_exit(self, strat, direction, reason):
        """React to a live-combined EXIT broadcast (:8081 parity for MILK). A FORCE-CLOSE reason
        (force_close / FORCE_CLOSE / EOD) closes EVERYONE holding this strat — eval takers + funded
        gamblers + milkers — mirroring the live session force-close. A DYNAMIC exit (TP_FIXED/SL_TRAIL/
        TP_GREEN/SL_YELLOW/SL/TP/stop/target) closes MILK ONLY; gamblers and eval-takers keep running to
        their own static bracket(). flatten_strat is broker-checked and a safe no-op if the position
        already closed (a rested bracket filled), so it coexists with the time-sweep backstops."""
        if not strat:
            return []
        # news_blackout_<EVENT> is a RISK flatten -> force-close everyone (not milk-only).
        force = reason in FORCE_CLOSE_REASONS or str(reason).startswith("news_blackout")
        with self.lock:
            if force:
                targets = set(self.farm.accounts.keys()) | set(self.funded.accounts.keys())
            else:
                targets = ({a.id for a in self.farm.accounts.values() if a.state is EState.MILKING}
                           | {a.id for a in self.funded.accounts.values() if a.phase is FPhase.MILKING})
        closed = []
        for aid in targets:
            try:
                resp = self.flatten_strat(aid, strat, f"{strat.lower()}_{str(reason).lower()}")
                closed.append({"account": aid, "ok": resp.get("ok")})
            except Exception:
                pass
        print(f"[farm] EXIT {strat} {reason} -> "
              f"{'FORCE-CLOSE all' if force else 'milk-only'}: {len(targets)} account(s)")
        return closed

    def flatten(self, account, reason="manual"):
        try:
            return _post(FLATTEN_URL, {"account": account, "reason": reason})
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def flatten_strat(self, account, strat, reason="strat_force_close"):
        """Flatten only one strat's tracked positions on an account (addon /flatten with strat=...)."""
        try:
            return _post(FLATTEN_URL, {"account": account, "strat": strat, "reason": reason})
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def reset_rotation(self):
        """Reset the eval round-robin to a clean slate: zero every account's recency marker (last_seq)
        and the global tick, and clear today's tallies (day_profit / peak_day_profit). Use after firing
        TEST signals — those advance the rotation and dent the day tally, skewing who's 'up next'. Live
        cash/equity/DD are NT8-owned and refresh on the next sync, so we leave them alone. Persists."""
        with self.lock:
            self.farm._tick = 0
            for a in self.farm.accounts.values():
                a.last_seq = 0
                a.day_profit = 0.0
                a.peak_day_profit = 0.0
            self.farm.save_state(STATE_FILE)
        print("[farm] rotation reset -> last_seq/tick zeroed, day tallies cleared")
        return {"ok": True, "reset": len(self.farm.accounts)}

    @staticmethod
    def _apply_plan(acct, plan_key):
        """Resolve a plan key into the account's caps. Unknown key -> DEFAULT_PLAN."""
        key = plan_key if plan_key in PLANS else DEFAULT_PLAN
        p = PLANS[key]
        acct.plan = key
        acct.day_cap, acct.stop_new_at = p["day_cap"], p["stop_new_at"]
        acct.target, acct.pass_buffer = p["target"], p["pass_buffer"]
        acct.consistency, acct.milk_at = p["consistency"], p["milk_at"]

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

    def set_funded_phase(self, account, phase):
        """Manually move a FUNDED account between GAMBLING / MILKING (locks out auto de-risk), or AUTO
        to release the lock. Persists immediately."""
        with self.lock:
            r = self.funded.set_phase(account, phase)
            if r.get("ok"):
                self.funded.save_state(FUNDED_STATE_FILE)
        print(f"[farm] funded {account} -> phase {phase}: {r}")
        return r

    def set_eval_phase(self, account, state):
        """Manually move an EVAL account MILKING <-> ACTIVE (locks out the auto milk_at/DONE logic), or
        AUTO to release the lock. Persists immediately."""
        with self.lock:
            r = self.farm.set_state(account, state)
            if r.get("ok"):
                self.farm.save_state(STATE_FILE)
        print(f"[farm] eval {account} -> state {state}: {r}")
        return r

    def gamble_next(self, account):
        """Tag a gambling funded account to take the next decorrelated signal (one-shot)."""
        with self.lock:
            r = self.funded.set_forced_gambler(account)
            if r.get("ok"):
                self.funded.save_state(FUNDED_STATE_FILE)
        print(f"[farm] funded gamble-next -> {account}: {r}")
        return r

    def set_funded_risk(self, risk):
        """Fixed $ risk per funded GAMBLE trade. risk in {number, 'buffer', '', None}: a number sizes
        every funded gamble to that $ (capped at the account's buffer); 'buffer'/None risks the whole
        buffer (a touch = blow). Persists with the funded state."""
        with self.lock:
            self.funded.gamble_risk = None if risk in (None, "buffer", "") else float(risk)
            self.funded.save_state(FUNDED_STATE_FILE)
        print(f"[farm] funded gamble risk -> {self.funded.gamble_risk}")
        return {"ok": True, "risk": self.funded.gamble_risk}

    def set_funded_derisk(self, val):
        """De-risk threshold (gamble -> milk). val in {'lock', '0', a number}: 'lock'/None -> the account's
        DD (lock the floor); '0' -> de-risk at the FIRST profit; a number -> de-risk once profit >= that $."""
        with self.lock:
            self.funded.derisk_at = None if val in (None, "lock", "") else float(val)
            self.funded.save_state(FUNDED_STATE_FILE)
        print(f"[farm] funded de-risk at -> {self.funded.derisk_at}")
        return {"ok": True, "derisk_at": self.funded.derisk_at}

    def set_funded_milk(self, qty):
        """MNQ per signal for MILKING funded accounts (default 1)."""
        with self.lock:
            try:
                self.funded.milk_qty = max(1, int(qty))
            except (TypeError, ValueError):
                self.funded.milk_qty = 1
            self.funded.save_state(FUNDED_STATE_FILE)
        print(f"[farm] funded milk size -> {self.funded.milk_qty} MNQ")
        return {"ok": True, "milk_qty": self.funded.milk_qty}

    def set_funded_lock(self, val):
        """$ account profit at which to force-close a GAMBLING account's open trade. 'off'/None -> disabled."""
        with self.lock:
            self.funded.profit_lock = None if val in (None, "off", "") else float(val)
            self.funded.save_state(FUNDED_STATE_FILE)
        print(f"[farm] funded profit lock -> {self.funded.profit_lock}")
        return {"ok": True, "profit_lock": self.funded.profit_lock}

    def set_risk_mode(self, account, mode):
        """Per-account risk tier: 'auto' ($750 while buffer < $2k, else full), '750', or '1500'.
        Tries the eval brain first, then funded. Persists immediately."""
        with self.lock:
            r = self.farm.set_risk_mode(account, mode)
            if not r.get("ok"):
                r = self.funded.set_risk_mode(account, mode)
            if r.get("ok"):
                self.farm.save_state(STATE_FILE)
                self.funded.save_state(FUNDED_STATE_FILE)
        print(f"[farm] {account} -> risk mode {mode}: {r}")
        return r

    def mark_blown(self, account):
        """Manually mark an account BLOWN (firm force-closed it). Sticky — sync won't revive it.
        Tries eval then funded. Persists immediately. Does NOT touch rotation ordering."""
        with self.lock:
            r = self.farm.mark_blown(account)
            if not r.get("ok"):
                r = self.funded.mark_blown(account)
            if r.get("ok"):
                self.farm.save_state(STATE_FILE)
                self.funded.save_state(FUNDED_STATE_FILE)
        print(f"[farm] {account} -> MARK BLOWN: {r}")
        return r

    def mark_passed(self, account):
        """Manually mark an eval account PASSED (firm passed it, farm missed the +$3k). Sticky."""
        with self.lock:
            r = self.farm.mark_passed(account)
            if r.get("ok"):
                self.farm.save_state(STATE_FILE)
        print(f"[farm] {account} -> MARK PASSED: {r}")
        return r

    def revive(self, account):
        """Undo a (manual/auto) blow -> back into play. Safety valve for a false blow. Clears the
        auto-blow debounce streak so the account gets a fresh window (re-blows only if truly dead)."""
        with self.lock:
            self._dead_polls.pop(account, None)
            r = self.farm.revive(account)
            if not r.get("ok"):
                r = self.funded.revive(account)
            if r.get("ok"):
                self.farm.save_state(STATE_FILE)
                self.funded.save_state(FUNDED_STATE_FILE)
        print(f"[farm] {account} -> REVIVE: {r}")
        return r

    def remove(self, account):
        """Remove an account from the farm entirely (dead/passed acct NT8 still lists). Blocklisted so
        sync won't re-add it. Tries eval then funded. Rotation of the rest is unaffected."""
        with self.lock:
            self._dead_polls.pop(account, None)
            r = self.farm.remove(account)
            if not r.get("ok"):
                r = self.funded.remove(account)
            if r.get("ok"):
                self.farm.save_state(STATE_FILE)
                self.funded.save_state(FUNDED_STATE_FILE)
        print(f"[farm] {account} -> REMOVE: {r}")
        return r

    def _cap_flatten_candidates(self, by_name):
        """Eval cap force-close: under the lock, find ACTIVE accounts whose OPEN-trade unrealized has
        reached the close-room (daily-cap room OR pass-line room, whichever is nearer). Returns a list
        of (name, unrealized, room) to flatten OUTSIDE the lock (network I/O). Only winners are capped;
        losing trades run to their own SL/TP/EOD. Skips bad reads and accounts on cooldown."""
        out = []
        now = time.monotonic()
        with self.lock:
            for name, acct in self.farm.accounts.items():
                if acct.state not in (EState.ACTIVE, EState.MILKING):
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

    def _funded_lock_candidates(self, by_name):
        """Funded profit-lock: GAMBLING accounts whose live account profit (netliq - start_bal) has
        reached the selected $ lock level -> flatten the open gamble to bank it. Gamble-phase only;
        milk runs to its own exits. Returns [(name, profit, lock), ...]. Skips bad reads / cooldown."""
        out = []
        lock = self.funded.profit_lock
        if not lock:
            return out
        now = time.monotonic()
        with self.lock:
            for name, fa in self.funded.accounts.items():
                if fa.phase is not FPhase.GAMBLING:
                    continue
                raw = by_name.get(name)
                if not raw or not raw.get("connected", True) or raw.get("netliq", 0) <= 0:
                    continue
                if not raw.get("positions"):
                    continue                                   # nothing open to lock
                profit = raw["netliq"] - fa.start_bal          # account profit incl. the open gamble
                if profit >= lock and (now - self._cap_cooldown.get(name, 0)) >= CAP_FLATTEN_COOLDOWN_SEC:
                    self._cap_cooldown[name] = now
                    out.append((name, profit, lock))
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
                                       float(p.get("stop_pts", 30)),
                                       sl_price=p.get("sl_price"), tp_price=p.get("tp_price"))
            self._send(200, json.dumps({"ok": True, "placed": placed}))
        elif self.path == "/api/signal":          # live signal broadcast from the coordinator
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n).decode() if n else "{}")
            if p.get("event") == "EXIT":           # close backend-managed milk legs / force-close all
                closed = STATE.handle_exit(p.get("strat", ""), p.get("direction", "LONG"),
                                           p.get("reason", ""))
                self._send(200, json.dumps({"ok": True, "closed": closed}))
            else:                                  # ENTRY (default): route + size + place
                placed = STATE.fire_signal(p.get("strat", "OD"), p.get("direction", "LONG"),
                                           float(p.get("stop_pts", 30)),
                                           sl_price=p.get("sl_price"), tp_price=p.get("tp_price"),
                                           green=p.get("green"), fb_tp=p.get("fb_tp"))
                self._send(200, json.dumps({"ok": True, "placed": placed}))
        elif self.path == "/api/reset_rotation":
            self._send(200, json.dumps(STATE.reset_rotation()))
        elif self.path == "/api/flatten":
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n).decode() if n else "{}")
            self._send(200, json.dumps(STATE.flatten(p.get("account", ""))))
        elif self.path == "/api/set_plan":
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n).decode() if n else "{}")
            self._send(200, json.dumps(STATE.set_plan(p.get("account", ""), p.get("plan", ""))))
        elif self.path == "/api/set_funded_phase":
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n).decode() if n else "{}")
            self._send(200, json.dumps(STATE.set_funded_phase(p.get("account", ""), p.get("phase", ""))))
        elif self.path == "/api/set_eval_phase":
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n).decode() if n else "{}")
            self._send(200, json.dumps(STATE.set_eval_phase(p.get("account", ""), p.get("state", ""))))
        elif self.path == "/api/gamble_next":
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n).decode() if n else "{}")
            self._send(200, json.dumps(STATE.gamble_next(p.get("account", ""))))
        elif self.path == "/api/set_funded_risk":
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n).decode() if n else "{}")
            self._send(200, json.dumps(STATE.set_funded_risk(p.get("risk"))))
        elif self.path == "/api/set_funded_derisk":
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n).decode() if n else "{}")
            self._send(200, json.dumps(STATE.set_funded_derisk(p.get("derisk"))))
        elif self.path == "/api/set_funded_milk":
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n).decode() if n else "{}")
            self._send(200, json.dumps(STATE.set_funded_milk(p.get("milk"))))
        elif self.path == "/api/set_funded_lock":
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n).decode() if n else "{}")
            self._send(200, json.dumps(STATE.set_funded_lock(p.get("lock"))))
        elif self.path == "/api/set_risk_mode":
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n).decode() if n else "{}")
            self._send(200, json.dumps(STATE.set_risk_mode(p.get("account", ""), p.get("mode", "auto"))))
        elif self.path == "/api/mark_blown":
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n).decode() if n else "{}")
            self._send(200, json.dumps(STATE.mark_blown(p.get("account", ""))))
        elif self.path == "/api/mark_passed":
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n).decode() if n else "{}")
            self._send(200, json.dumps(STATE.mark_passed(p.get("account", ""))))
        elif self.path == "/api/revive":
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n).decode() if n else "{}")
            self._send(200, json.dumps(STATE.revive(p.get("account", ""))))
        elif self.path == "/api/remove":
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n).decode() if n else "{}")
            self._send(200, json.dumps(STATE.remove(p.get("account", ""))))
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
    <button onclick="resetRotation()" class="px-3 py-1 rounded bg-amber-700 hover:bg-amber-600 font-semibold" title="zero last_seq/tick + clear day tallies after test fires (cash/equity untouched, NT8 re-syncs)">Reset rotation</button>
    <span id="fResult" class="text-slate-400"></span>
  </div>
  <h2 class="text-lg font-semibold mb-2 text-slate-300">Eval accounts
    <span class="text-slate-500 text-sm">(farm-traded · drag a card to another plan, or use its dropdown)</span></h2>
  <div id="evalSections" class="mb-6"></div>
  <h2 class="text-lg font-semibold mb-2 text-teal-300">Eval — milking
    <span class="text-slate-500 text-sm">(1 MNQ 4-strat → consistency-adjusted pass target)</span></h2>
  <div id="evalMilking" class="grid grid-cols-2 md:grid-cols-3 gap-3 mb-8"></div>
  <div class="flex items-center justify-between mb-2">
    <h2 class="text-lg font-semibold text-amber-300">Funded - gambling</h2>
    <div class="flex items-center gap-4">
      <label class="text-sm text-slate-400 flex items-center gap-2">gamble risk / trade
        <select id="fRisk" onchange="setFundedRisk(this.value)" class="bg-slate-700 rounded px-2 py-1 text-sm" title="$ risked per funded gamble (capped at the buffer). Full buffer = a touch of SL blows the account.">
          <option value="buffer">Full buffer</option>
          <option value="2000">$2,000</option>
          <option value="1500">$1,500</option>
          <option value="1000">$1,000</option>
          <option value="750">$750</option>
          <option value="500">$500</option>
        </select>
      </label>
      <label class="text-sm text-slate-400 flex items-center gap-2">de-risk at
        <select id="fDerisk" onchange="setFundedDerisk(this.value)" class="bg-slate-700 rounded px-2 py-1 text-sm" title="When a gambling account switches to milk. First profit = safest (low blow); +$2k lock = bank the DD; +$3k = bigger buffer, higher blow.">
          <option value="0">First profit</option>
          <option value="lock">+$2k lock</option>
          <option value="3000">+$3k</option>
        </select>
      </label>
      <label class="text-sm text-slate-400 flex items-center gap-2">milk size
        <select id="fMilk" onchange="setFundedMilk(this.value)" class="bg-slate-700 rounded px-2 py-1 text-sm" title="MNQ per signal once an account is milking">
          <option value="1">1 MNQ</option>
          <option value="2">2 MNQ</option>
          <option value="3">3 MNQ</option>
          <option value="4">4 MNQ</option>
          <option value="5">5 MNQ</option>
          <option value="6">6 MNQ</option>
        </select>
      </label>
      <label class="text-sm text-slate-400 flex items-center gap-2">profit lock
        <select id="fLock" onchange="setFundedLock(this.value)" class="bg-slate-700 rounded px-2 py-1 text-sm" title="Force-close a gambling account's open trade when its account profit reaches this level (mid-trade, like the eval cap). Gamble phase only.">
          <option value="off">Off</option>
          <option value="2000">$2k</option>
          <option value="3000">$3k</option>
          <option value="4000">$4k</option>
        </select>
      </label>
    </div>
  </div>
  <div id="gambling" class="grid grid-cols-2 md:grid-cols-3 gap-3 mb-6"></div>
  <h2 class="text-lg font-semibold mb-2 text-emerald-300">Funded - milking</h2>
  <div id="milking" class="grid grid-cols-2 md:grid-cols-3 gap-3 mb-8"></div>
  <h2 class="text-lg font-semibold mb-2 text-rose-300">Funded - retired / blown</h2>
  <div id="fretired" class="grid grid-cols-2 md:grid-cols-3 gap-3 mb-8"></div>
  <h2 class="text-lg font-semibold mb-2 text-slate-300">Live accounts <span class="text-slate-500 text-sm">(read-only, never traded)</span></h2>
  <div id="live" class="grid grid-cols-2 md:grid-cols-3 gap-3"></div>
</div>
<script>
const C = {FRESH:'bg-slate-600', ACTIVE:'bg-sky-600', DONE:'bg-amber-600', MILKING:'bg-teal-600', PASSED:'bg-emerald-600', BLOWN:'bg-rose-700'};
const OUT = {PASSED:'ring-2 ring-emerald-400 shadow-lg shadow-emerald-500/40', BLOWN:'ring-2 ring-rose-500 shadow-lg shadow-rose-500/40'};
async function ctl(a){ await fetch('/api/'+a,{method:'POST'}); tick(); }
async function flat(acct){
  await fetch('/api/flatten',{method:'POST',body:JSON.stringify({account:acct})});
  tick();
}
function evalCard(a, plans){
  const opts = (plans||[]).map(p=>`<option value="${p.key}" ${p.key===a.plan?'selected':''}>${p.label}</option>`).join('');
  const lock = a.manual ? ' <span title="manual lock: auto milk/DONE suppressed">🔒</span>' : '';
  const auto = a.manual ? `<button onclick="setEvalPhase('${a.name}','AUTO')" class="text-xs px-2 py-0.5 rounded bg-slate-700 hover:bg-slate-600">Auto</button>` : '';
  return `<div draggable="true" ondragstart="event.dataTransfer.setData('text/plain','${a.name}')"
      class="rounded-lg p-3 bg-slate-800 border border-slate-700 transition cursor-move ${OUT[a.state]||''}">
      <div class="flex justify-between items-center mb-2">
        <span class="font-semibold">${a.name}${lock}</span>
        <span class="text-xs px-2 py-0.5 rounded ${C[a.state]||'bg-slate-700'}">${a.state}</span></div>
      <div class="text-sm text-slate-200">bal ${money(a.balance)}</div>
      <div class="flex justify-between items-center mt-1">
        <span class="text-xs text-slate-400">buffer ${money(a.buffer)} <span class="text-slate-500" title="raw NT8 TrailingMaxDrawdown — intraday, unreliable; buffer left of this is the true EOD-trailing value">(nt8 ${a.nt8_dd==null?'—':money(a.nt8_dd)})</span> · today ${money(a.today)}</span>
        <button onclick="flat('${a.name}')" class="text-xs px-2 py-0.5 rounded bg-rose-800 hover:bg-rose-700">flatten</button>
      </div>
      <div class="flex gap-1 mt-2">
        <button onclick="setEvalPhase('${a.name}','MILKING')" class="text-xs px-2 py-0.5 rounded bg-teal-800 hover:bg-teal-700">→ Milk</button>${auto}
      </div>
      <div class="flex flex-wrap gap-1 mt-2">${evalActions(a)}</div>
      <select onchange="setPlan('${a.name}', this.value)" class="mt-2 w-full bg-slate-700 rounded px-1 py-1 text-xs">${opts}</select>
    </div>`;
}
async function setPlan(acct, plan){
  await fetch('/api/set_plan',{method:'POST',body:JSON.stringify({account:acct, plan})});
  tick();
}
async function setPhase(acct, phase){
  await fetch('/api/set_funded_phase',{method:'POST',body:JSON.stringify({account:acct, phase})});
  tick();
}
async function setEvalPhase(acct, state){
  await fetch('/api/set_eval_phase',{method:'POST',body:JSON.stringify({account:acct, state})});
  tick();
}
async function gambleNext(acct){
  await fetch('/api/gamble_next',{method:'POST',body:JSON.stringify({account:acct})});
  tick();
}
async function setRiskMode(acct, mode){
  await fetch('/api/set_risk_mode',{method:'POST',body:JSON.stringify({account:acct, mode})});
  tick();
}
async function markBlown(acct){
  if(!confirm('Mark '+acct+' BLOWN? (firm force-closed it). Sticky — sync will NOT revive it.')) return;
  await fetch('/api/mark_blown',{method:'POST',body:JSON.stringify({account:acct})});
  tick();
}
async function revive(acct){
  await fetch('/api/revive',{method:'POST',body:JSON.stringify({account:acct})});
  tick();
}
function reviveBtn(a){
  return `<button onclick="revive('${a.name}')" class="text-xs px-1.5 py-0.5 rounded bg-emerald-800 hover:bg-emerald-700" title="undo a false blow -> back into play (re-blows if truly dead)">revive</button>`;
}
async function removeAcct(acct){
  if(!confirm('Remove '+acct+' from the farm? Blocklisted so sync will not re-add it (use for dead/passed accounts NT8 still lists).')) return;
  await fetch('/api/remove',{method:'POST',body:JSON.stringify({account:acct})});
  tick();
}
function removeBtn(a){
  return `<button onclick="removeAcct('${a.name}')" class="text-xs px-1.5 py-0.5 rounded bg-slate-600 hover:bg-slate-500" title="remove from farm (dead/passed account)">remove</button>`;
}
async function markPassed(acct){
  if(!confirm('Mark '+acct+' PASSED? (firm passed it). Sticky — takes it out of rotation.')) return;
  await fetch('/api/mark_passed',{method:'POST',body:JSON.stringify({account:acct})});
  tick();
}
function passedBtn(a){
  return `<button onclick="markPassed('${a.name}')" class="text-xs px-1.5 py-0.5 rounded bg-emerald-900 hover:bg-emerald-800" title="firm passed it -> mark passed (sticky)">passed</button>`;
}
function evalActions(a){
  if(a.state==='PASSED') return reviveBtn(a)+removeBtn(a);
  if(a.state==='BLOWN') return reviveBtn(a)+removeBtn(a);
  return riskBtns(a)+blownBtn(a)+passedBtn(a);
}
function riskBtns(a){
  const m = a.risk_mode||'auto';
  const b = (val,lbl)=>`<button onclick="setRiskMode('${a.name}','${val}')" class="text-xs px-1.5 py-0.5 rounded ${m===val?'bg-indigo-600':'bg-slate-700 hover:bg-slate-600'}" title="risk tier">${lbl}</button>`;
  return b('auto','Auto')+b('750','$750')+b('1500','$1500');
}
function blownBtn(a){
  return `<button onclick="markBlown('${a.name}')" class="text-xs px-1.5 py-0.5 rounded bg-rose-900 hover:bg-rose-800" title="firm force-closed -> mark blown (sticky)">blown</button>`;
}
async function setFundedRisk(v){
  await fetch('/api/set_funded_risk',{method:'POST',body:JSON.stringify({risk:v})});
  tick();
}
async function setFundedDerisk(v){
  await fetch('/api/set_funded_derisk',{method:'POST',body:JSON.stringify({derisk:v})});
  tick();
}
async function setFundedMilk(v){
  await fetch('/api/set_funded_milk',{method:'POST',body:JSON.stringify({milk:v})});
  tick();
}
async function setFundedLock(v){
  await fetch('/api/set_funded_lock',{method:'POST',body:JSON.stringify({lock:v})});
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
async function resetRotation(){
  if(!confirm('Reset rotation? Zeros last_seq/tick + clears day tallies (cash/equity untouched).')) return;
  fResult.textContent = 'resetting...';
  try {
    const r = await (await fetch('/api/reset_rotation',{method:'POST'})).json();
    fResult.textContent = r.ok ? ('rotation reset ('+r.reset+' accounts)') : 'reset failed';
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
  const mr = s.milkers_ready || [];
  document.getElementById('ready').innerHTML =
    `<div class="text-xs uppercase tracking-wide text-sky-400 mb-1">Ready for next signal - ${label}</div>`+
    `<div class="flex flex-wrap gap-2 items-center">`+
    (r.leads.length ? r.leads.map(n=>`<span class="px-3 py-1 rounded bg-sky-600 font-semibold">${n}</span>`).join('')
                    : '<span class="text-slate-500">no accounts ready</span>')+
    (r.actives.length ? `<span class="text-slate-400 text-sm">+ copying: ${r.actives.join(', ')}</span>` : '')+
    `</div>`+
    `<div class="text-xs uppercase tracking-wide text-teal-400 mt-3 mb-1">Milkers ready (copy every signal)</div>`+
    `<div class="flex flex-wrap gap-2 items-center">`+
    (mr.length ? mr.map(n=>`<span class="px-3 py-1 rounded bg-teal-600 font-semibold">${n}</span>`).join('')
               : '<span class="text-slate-500">no milkers ready</span>')+
    `</div>`;
  const plans = s.plans || [];
  const milkers = (s.watched||[]).filter(a=>a.state==='MILKING');
  const byPlan = {};
  (s.watched||[]).filter(a=>a.state!=='MILKING').forEach(a=>{ (byPlan[a.plan]=byPlan[a.plan]||[]).push(a); });
  document.getElementById('evalMilking').innerHTML = milkers.map(a=>{
    const tgt = a.true_target||a.target||3000, need = Math.max(0, tgt - (a.total||0));
    const pct = Math.min(100, Math.max(0, (a.total||0)/tgt*100));
    const lock = a.manual ? ' <span title="manual lock: auto milk/DONE suppressed">🔒</span>' : '';
    const auto = a.manual ? `<button onclick="setEvalPhase('${a.name}','AUTO')" class="text-xs px-2 py-0.5 rounded bg-slate-700 hover:bg-slate-600">Auto</button>` : '';
    return `<div class="rounded-lg p-3 bg-slate-800 border border-teal-700">
      <div class="flex justify-between items-center mb-1"><span class="font-semibold">${a.name}${lock}</span>
        <span class="text-xs px-2 py-0.5 rounded bg-teal-600">MILKING</span></div>
      <div class="text-sm text-slate-200">${money(a.total)} / ${money(tgt)}
        <span class="text-xs text-slate-400">need ${money(need)}</span></div>
      <div class="w-full bg-slate-700 rounded h-1.5 mt-2"><div class="bg-teal-500 h-1.5 rounded" style="width:${pct}%"></div></div>
      <div class="flex justify-between items-center mt-1">
        <span class="text-xs text-slate-500">biggest day ${money(a.peak_day)}</span>
        <button onclick="flat('${a.name}')" class="text-xs px-2 py-0.5 rounded bg-rose-800 hover:bg-rose-700">flatten</button>
      </div>
      <div class="flex gap-1 mt-2">
        <button onclick="setEvalPhase('${a.name}','ACTIVE')" class="text-xs px-2 py-0.5 rounded bg-sky-800 hover:bg-sky-700">→ Active</button>${auto}
      </div>
      <div class="flex flex-wrap gap-1 mt-2">${evalActions(a)}</div>
    </div>`;
  }).join('') || '<div class="text-slate-500 col-span-3">no accounts milking</div>';
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
  const fr = document.getElementById('fRisk');
  if(fr && document.activeElement!==fr) fr.value = (s.funded_risk==null ? 'buffer' : String(s.funded_risk));
  const fd2 = document.getElementById('fDerisk');
  if(fd2 && document.activeElement!==fd2) fd2.value = (s.funded_derisk==null ? 'lock' : (s.funded_derisk==0 ? '0' : String(s.funded_derisk)));
  const fm = document.getElementById('fMilk');
  if(fm && document.activeElement!==fm) fm.value = String(s.funded_milk||1);
  const fl = document.getElementById('fLock');
  if(fl && document.activeElement!==fl) fl.value = (s.funded_lock==null ? 'off' : String(s.funded_lock));
  const f = s.funded || {gambling:[], milking:[]};
  const fcard = (a,b,isGamble)=>{
    const lock = a.manual ? '<span class="text-xs px-1.5 py-0.5 rounded bg-slate-600" title="manual lock: auto de-risk suppressed">🔒</span>' : '';
    // gambling cards -> "→ Milk" + "gamble next"; milking cards -> "→ Gamble". Locked cards -> "Auto".
    const toggle = isGamble
      ? `<button onclick="setPhase('${a.name}','MILKING')" class="text-xs px-2 py-0.5 rounded bg-emerald-800 hover:bg-emerald-700">→ Milk</button>`
      : `<button onclick="setPhase('${a.name}','GAMBLING')" class="text-xs px-2 py-0.5 rounded bg-amber-800 hover:bg-amber-700">→ Gamble</button>`;
    const auto = a.manual ? `<button onclick="setPhase('${a.name}','AUTO')" class="text-xs px-2 py-0.5 rounded bg-slate-700 hover:bg-slate-600">Auto</button>` : '';
    const gnext = isGamble ? `<button onclick="gambleNext('${a.name}')" class="text-xs px-2 py-0.5 rounded ${a.forced?'bg-sky-600':'bg-slate-700 hover:bg-slate-600'}" title="take the next decorrelated signal">${a.forced?'next ✓':'gamble next'}</button>` : '';
    return `<div class="rounded-lg p-3 bg-slate-800 border ${b}">`+
      `<div class="flex justify-between items-center"><span class="font-semibold">${a.name}</span>${lock}</div>`+
      `<div class="text-sm text-slate-200 mt-1">bal ${money(a.balance)}</div>`+
      `<div class="text-xs text-slate-400">buffer ${money(a.buffer)} <span class="text-slate-500" title="raw NT8 TrailingMaxDrawdown — trails intraday netliq, unreliable; the buffer left of this is the true EOD-trailing value">(nt8 ${a.nt8_dd==null?'—':money(a.nt8_dd)})</span> · payouts ${a.payouts}</div>`+
      `<div class="flex flex-wrap gap-1 mt-2">${toggle}${gnext}${auto}</div>`+
      `<div class="flex flex-wrap gap-1 mt-2">${evalActions(a)}</div></div>`;
  };
  document.getElementById('gambling').innerHTML = (f.gambling||[]).length ? f.gambling.map(a=>fcard(a,'border-amber-700',true)).join('') : '<div class="text-slate-500 col-span-3">no gambling accounts yet</div>';
  document.getElementById('milking').innerHTML = (f.milking||[]).length ? f.milking.map(a=>fcard(a,'border-emerald-700',false)).join('') : '<div class="text-slate-500 col-span-3">no milking accounts yet</div>';
  document.getElementById('fretired').innerHTML = (f.retired||[]).length ? (f.retired||[]).map(a=>
    `<div class="rounded-lg p-3 bg-slate-800/60 border border-rose-800">
      <div class="flex justify-between items-center"><span class="font-semibold">${a.name}</span>
        <span class="text-xs px-2 py-0.5 rounded ${a.blown?'bg-rose-700':'bg-slate-600'}">${a.blown?'BLOWN':'RETIRED'}</span></div>
      <div class="text-xs text-slate-400 mt-1">${a.reason||''}</div>
      <div class="flex gap-1 mt-2">${a.blown?reviveBtn(a):''}${removeBtn(a)}</div>
    </div>`).join('') : '<div class="text-slate-500 col-span-3">none</div>';
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
