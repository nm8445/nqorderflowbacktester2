# Test Addon Spec — `NQMultiStratReceiverTest` (port 8082)

The **hands** for the two farm brains (`funded_state_machine.py`, `eval_passer.py`). A second NT8
addon, isolated from the production single-account addon (`NQMultiStratReceiver.cs`, :8081), that:

1. **Reports per-account equity** (`GET /accounts`) — the snapshot both brains' `sync_accounts()`
   consume, and the one genuine live unknown we must validate.
2. **Routes orders to a SPECIFIC named account** (`POST /order`, `POST /close`) — driven by the
   `Route` objects the brains emit.

It runs alongside production without touching it: different port, an order **whitelist**, and
sim-only flatten until cutover.

---

## Why a second addon (not edit :8081)

- :8081 runs live phase-1 money. We validate the farm on demo/eval accounts without risking it.
- `Account.All` is **global** across NT8 connections, so one addon *could* see every account — but
  two addons both acting on one signal would double-fire. The test addon **reports** all accounts
  (read-only) yet only **acts** on its whitelisted (test) accounts; production stays on :8081.
- After validation: cut over — retire :8081, or keep it for phase-1 and let :8082 own only the farm
  accounts.

The delta vs :8081: that addon is **single-account** (`accountName` hardcoded ~L60,
`Account.All.FirstOrDefault(a => a.Name == accountName)` ~L186). The test addon generalizes that to
**per-order account selection + a whitelist**, and adds `/accounts`.

---

## Endpoints

### `GET /accounts` — the equity source (validate this FIRST)

Returns every account NT8 exposes (`Account.All`), so the brains classify by name themselves:

```json
{ "accounts": [
  { "name": "APEX-12345", "cash": 50000.0, "unrealized": 0.0, "netliq": 50000.0,
    "positions": [ { "instrument": "NQ 06-26", "qty": 1, "avgPrice": 24310.0, "tag": "B2_..." } ] }
] }
```

| field | NT8 source | brain uses it for |
|---|---|---|
| `cash` | `Account.Get(AccountItem.CashValue)` | realized balance → EOD winning-days / pass |
| `unrealized` | `Account.Get(AccountItem.UnrealizedProfitLoss)` | floating (open-trade swing) |
| `netliq` | `Account.Get(AccountItem.NetLiquidation)` | floor / blow / de-risk (equity) |

Python polls this every ~3–5 s and feeds **both** `EvalFarm.sync_accounts` and
`FundedFarm.sync_accounts`. **The one live unknown:** which field the firm's trailing DD keys off
(EOD `cash` vs intraday `netliq`). We pin that before trusting any routing (Test Plan, Phase 0).

### `POST /order` — per-account entry (the new bit vs :8081)

```json
{ "account": "APEX-12345", "strat": "B2", "direction": "LONG", "qty": 1,
  "slPrice": 24180.0, "tpPrice": 24470.0, "tag": "B2_20260604_142000_LONG" }
```

- `account` resolves to `Account.All.First(a => a.Name == account)`; submit there (same submission
  path the :8081 addon already uses for entry + OCO SL/TP bracket).
- `qty`, `slPrice`, `tpPrice` are computed **Python-side** (sizing is NOT in the addon — see below).
- **Whitelist gate:** reject the order if `account` isn't in the configured test whitelist.

### `POST /close` — per-account flatten of a tagged position

```json
{ "account": "APEX-12345", "strat": "B2", "tag": "...", "reason": "FORCE_CLOSE" }
```

- Closes the tagged position on the named account. Mirror the **retry-on-fail + stale-position
  reconcile** we just added on the MT5 side (`mt5_executor.py`): a 10013-style reject retries, and a
  position that vanished (manual flatten / broker SL fill) self-clears so it never blocks re-entry.

---

## Where sizing happens — NOT in the addon

The brains emit `risk~$X` (a dollar risk target), never a contract count. Converting to contracts
needs the **stop distance**, which the *strategy* owns (ATR / ORB-low). So a thin Python **sizer**
sits between brains and addon:

```
strategy fires (direction + stop_pts)
   -> Brain.route_signal()  -> Routes [(account, strat, intent, risk$)]
   -> Sizer: qty = clamp(round(risk$ / (stop_pts * $per_pt)), 1, MAX)     # NQ $20/pt, MNQ $2/pt
   -> POST /order per Route  (account, qty, slPrice, tpPrice, tag)
```

The addon stays dumb (execute on the named account). This matches the existing split: Python =
brains + sizing, addon = execution.

---

## Account differentiation & handoff

The addon reports **all** accounts; the brains classify with `funded_pattern` / `eval_pattern`
regexes (TBD once evals are bought). A name matching the funded pattern routes to `FundedFarm`; an
eval pattern to `EvalFarm`. A **newly-appeared funded-pattern name** is the eval→funded handoff —
`FundedFarm.sync_accounts` auto-adopts it as `GAMBLING`. The addon needs no knowledge of any of this.

---

## Safety (isolation from production is the whole point)

- **Order whitelist** — `/order` and `/close` only act on accounts in the test whitelist. All other
  accounts are **reported but never touched**. Production phase-1 money is read-only to this addon.
- **Heartbeat flatten** — like :8081, flatten tagged positions on Python-silence (~30 s) **but only
  on whitelisted accounts**, so a test-side crash can never flatten production.
- **Per-account retry/reconcile** — port the MT5-side fixes: failed close → retry; tracked position
  with no live match (older than a grace window) → drop, so it self-heals without a restart.

---

## Test plan (validate before a single live routed order)

**Phase 0 — `/accounts` only, zero routing.**
1. Point the addon at the one live funded 50k (read-only).
2. Poll `/accounts`; log `cash` / `unrealized` / `netliq`.
3. Manually open then close a trade; watch the three fields move.
4. Compare to the firm dashboard's balance + trailing-DD number → decide which field the DD keys off.
   **Lock that field** as the floor basis in both brains' `sync_accounts`.

**Phase 1 — routing on a demo account (whitelisted).**
5. Run both brains against a demo account; place real (demo) orders via `/order`; confirm per-account
   routing, SL/TP bracket, `/close`, and whitelisted heartbeat-flatten all fire on the right account.

**Phase 2 — one real eval (whitelisted, small).**
6. A single eval account end-to-end through `EvalFarm`.

**Phase 3 — cutover.** Retire :8081 (or keep it for phase-1 and give :8082 only the farm accounts).

---

## Build order

1. `GET /accounts` (Account.All → JSON). **Stop and run Phase 0.**
2. Python poller → `sync_accounts` on both brains, **read-only** (log the Routes, place nothing).
3. `POST /order` + `POST /close` per-account + the sizer layer. **Run Phase 1 on demo.**
4. Whitelisted heartbeat flatten + per-account retry/reconcile.
5. Cutover.

Each step is independently testable, and nothing places a live order until Phase 0 has told us the
equity source is correct.
```
