# Live System Slow Path Audit — Databento "slow_client" errors

**Date written**: 2026-05-21
**Trigger**: Databento gateway dropped records on 2026-05-21 (09:53 + 09:55 ET):
```
[ERROR] databento.live.protocol: gateway error code=skipped_records_after_slow_reading
err='Slow client detected for mbp-1. Skipped records'
```

This means Python consumed ticks slower than they arrived, gateway dropped data to catch up, and bars built from those windows have **incomplete tick coverage**. The errors clustered around cash open (peak tick rate), which is the architecture's worst-case load.

**Status**: NOT FIXED YET. This document lists the issues to address, ranked by impact.

---

## 🔴🔴 CONFIRMED 2026-05-21: bar high/low are corrupted, not just delta

User observed that during high-volatility candles where the `slow_client` error fires, the **bar high and low values are wrong** for those bars and the bars that follow.

This is more severe than initially documented. The implications cascade through the entire engine:

- **OHLC corruption** — bar builder's `_apply_tick` only updates `b.high` / `b.low` when it sees the relevant tick. Dropped ticks at the extremes never reach the builder, so the bar reports a narrower range than reality.
- **SL / TP triggers misfire** — your engines check `low <= sl_price` and `high >= tp_price` against the BAR's high/low. If those are wrong, an SL that *should have hit* won't, and you stay in a losing trade. Conversely, a TP that *should have triggered* might miss too.
- **ATR is wrong** — all your engines compute ATR from bar OHLC. A run of bars with shrunken ranges produces under-estimated ATR. This is consistent with the **May 20 B2 trade I flagged earlier** (live ATR 62 vs replay ATR 79 — the live ATR was lower because the lookback bars had truncated ranges from earlier slow_client events).
- **z_vol is wrong** — RV's `z_vol` uses close-to-close returns and a rolling std. Closes ARE usually captured (close = last seen tick price), but std normalization uses prior bars whose ranges are wrong.
- **Pinbar ratios are wrong** — B2's signal detection uses `(close - low) / (high - low)` style ratios. Wrong high/low → wrong pinbar score.
- **Volume profile / orderflow** — `level_volumes` dict only fires for ticks the builder saw. So per-price-level absorption analysis is operating on a partial picture.

The May 20 ATR mismatch I attributed to "warm-start divergence" was actually **this bug**. Live's 20-min ATR at 10:55 was 62 vs replay's 79 — same data source, but live's accumulated bars from earlier in the session had truncated highs/lows due to dropped ticks during volatile minutes. Replay reads complete tick history so its bars are correct.

This makes **fixes #1 and #2 (queue + async subscribers) the only acceptable path forward** before any real-money trading. Layers 1/2 of the disaster fallback (wide broker SL + Python watchdog) don't help if the engine's view of price is itself wrong — a position will simply be held longer than intended because the engine doesn't know its stop level was hit.

### How to verify the extent of corruption

Run for today's session (2026-05-21):
```bash
python live/combined/replay_today.py --date 2026-05-21
```

Then compare the bar high/low/volume from the replay against what live recorded in `D:/trading_pythonbacktest_data/live_warmstart_cache/nq_5min_2026-05-21.pkl`. Any bar where replay shows a higher high or lower low than live = corrupted live bar = trades during that bar (or any subsequent bar relying on its OHLC for ATR/EMA/range calcs) had bad inputs.

Likely fingerprint: the 09:30-10:00 ET window (cash open, peak tick rate) will show the worst divergence. Bars there may be wrong by 5-15 NQ points on high/low.

### Until the queue fix lands

**Do not place real-money entries based on live signals.** Any signal generated within ~5-15 minutes of a `slow_client` error in the log is suspect. The bar that triggered the signal, plus the ~14 prior bars feeding ATR(14), may have wrong ranges.

If you must trade today:
- Check the log for any `slow_client` error within the last 60 minutes before any new signal
- If present, skip the signal and validate via replay before trusting future signals
- Or: stop trading until the queue architecture is in place

---

## ✅ Recommended NT8-Specific Architecture (the actual fix)

Three threads with bounded queues between them. The Databento reader becomes the hot path's ONLY responsibility; everything else is decoupled.

```
┌───────────────────────────────┐
│  Databento Live Iterator       │  ← gateway sees only this loop
│  (data_feed.py reader thread) │
└──────────────┬────────────────┘
               │ tick_queue.put_nowait((ts, price, size, side))
               │   ~5 microseconds per tick — never blocks the gateway
               ↓
┌───────────────────────────────┐
│  Engine consumer thread        │
│  1. bar_builder.on_tick()      │  builds bars
│  2. subscribers fire on close  │  → RV, B2, OD, Fabio, persistor
│  3. Coordinator no-hedge       │
│  4. signal emitted to NT8 queue│
└──────────────┬────────────────┘
               │ signal_queue.put((tag, action, payload))
               │   tick consumer NEVER waits on NT8
               ↓
┌───────────────────────────────┐
│  NT8 executor thread           │
│  - drains signal_queue          │
│  - POSTs to localhost:8081      │
│  - retries on transient fail    │
└───────────────────────────────┘
```

### Concrete changes (NT8-only scope)

#### Change 1 — `live/combined/data_feed.py` — make it a queue producer

Replace the synchronous `for msg in self.client: self.on_tick(...)` loop with:

```python
import queue, threading

class DatabentoLiveFeed:
    def __init__(self, ...):
        self.tick_queue: queue.Queue = queue.Queue(maxsize=200_000)
        self._reader_thread: threading.Thread | None = None
        self._consumer_thread: threading.Thread | None = None
        self.on_tick_callback = on_tick   # bar_builder.on_tick, etc.

    def start(self):
        self._reader_thread = threading.Thread(target=self._reader_loop,
                                               name="DBN-reader", daemon=True)
        self._consumer_thread = threading.Thread(target=self._consumer_loop,
                                                  name="DBN-consumer", daemon=True)
        self._reader_thread.start()
        self._consumer_thread.start()
        # Block main thread on consumer (so Ctrl+C still works)
        self._consumer_thread.join()

    def _reader_loop(self):
        """HOT PATH. Do almost nothing here. Just parse + enqueue."""
        client = db.Live(key=self._api_key)
        client.subscribe(...)
        for msg in client:
            if self._stopped: break
            action = getattr(msg, "action", None)
            if action is None: continue
            ac = chr(action) if isinstance(action, int) else action.decode() if isinstance(action, bytes) else str(action)
            if ac != "T": continue
            try:
                self.tick_queue.put_nowait((msg.ts_recv, msg.price, msg.size, msg.side))
            except queue.Full:
                self.n_dropped += 1   # log + alert; engine fell behind

    def _consumer_loop(self):
        """Drain queue; do all the heavy work here."""
        while not self._stopped:
            try:
                ts_ns, price_raw, size, side_raw = self.tick_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            ts_et = pd.Timestamp(ts_ns, unit="ns", tz="UTC").tz_convert(ET_TZ)
            price = float(price_raw) / PRICE_SCALE
            side = chr(side_raw) if isinstance(side_raw, int) else (
                side_raw.decode() if isinstance(side_raw, bytes) else str(side_raw))
            if side not in ("A", "B", "N"): side = "N"
            try:
                self.on_tick_callback(ts_et, price, size, side)
            except Exception as e:
                log.exception(f"[engine] on_tick error: {e}")
```

Key properties:
- **Reader thread does 5 microseconds of work per tick.** Cannot fall behind the gateway.
- **All parsing moved to consumer** — including the tz conversion (`tz_convert` is ~30µs, was on hot path).
- **`put_nowait` + Full counter** — if engine is so slow the queue fills 200k deep, we log + count the drop locally rather than letting the gateway drop it. Acts as an early-warning system.

#### Change 2 — `live/combined/nt8_executor.py` — make HTTP async

Add a signal queue + dedicated POST thread:

```python
class NT8Executor:
    def __init__(self, ...):
        ...
        self.signal_queue: queue.Queue = queue.Queue(maxsize=10_000)
        self._post_thread = threading.Thread(target=self._post_loop,
                                              name="NT8-poster", daemon=True)
        self._post_thread.start()

    def _post_loop(self):
        while not self._stopped:
            try:
                endpoint, payload = self.signal_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            self._post_sync(endpoint, payload)   # rename old _post → _post_sync

    def send_entry(self, ...):
        # build payload as before, but instead of self._post():
        self.signal_queue.put_nowait(("/order", payload))
        return tag    # tag tracking still happens inline in engine thread
```

Key properties:
- **`send_entry` / `send_close_tag` / `send_close_all` return instantly** (queue put is microseconds).
- **NT8 POST happens on a separate thread.** A slow NT8 response no longer blocks the engine.
- **Heartbeat thread already exists** (`_hb_thread`) — leave it as-is. It already runs separately.

#### Change 3 — `live/combined/bar_persistor.py` — move pickle write to background

Replace `_atomic_write` synchronous call inside `on_bar` with a queued write:

```python
class LiveBarPersistor:
    def __init__(self, ...):
        ...
        self._write_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._write_thread = threading.Thread(target=self._write_loop,
                                               name="bar-persistor", daemon=True)
        self._write_thread.start()

    def on_bar(self, bar: Bar) -> None:
        # ... existing in-memory update of self._today_bars ...
        # Snapshot the list and enqueue
        self._write_queue.put_nowait((session_date, list(self._today_bars)))

    def _write_loop(self):
        while True:
            session_date, bars_snapshot = self._write_queue.get()
            self._atomic_write(session_date, bars_snapshot)
```

20ms pickle write moved off the engine thread. Critical for 5-min bar close events that previously stalled tick consumption.

#### Change 4 — kill `print()` on hot paths

Replace these `print()` calls with a single shared `logging.Logger` using `QueueHandler`:

- `live/combined/run_phase1.py:102-103` (per-signal prints in `_log`)
- `live/combined/nt8_executor.py:89` (per POST log)
- `live/combined/b2_signal_engine.py:253` and the `[B2-SIG]` line
- Any `print(f"[bar_builder ...]")` in `bar_builder.py`

Setup once at startup:

```python
import logging, logging.handlers
log_q = queue.Queue(-1)
queue_handler = logging.handlers.QueueHandler(log_q)
listener = logging.handlers.QueueListener(log_q, logging.StreamHandler())
listener.start()
logging.getLogger().addHandler(queue_handler)
```

Engine threads call `log.info(...)` — that's just an `enqueue` op. A dedicated listener thread does the actual `stderr` write. Removes Windows console rendering from the hot path.

### What to monitor (verify the fix is working)

Once these are in, add a single instrumentation thread that prints every 60s:

```python
def monitor_loop():
    while True:
        time.sleep(60)
        log.info(f"[mon] tick_q={tick_queue.qsize()}/{tick_queue.maxsize}  "
                 f"signal_q={signal_queue.qsize()}  "
                 f"write_q={write_queue.qsize()}  "
                 f"dropped={dropped_total}")
```

Healthy state:
- `tick_q` near 0 most of the time, occasional bursts to a few thousand during cash open
- `signal_q` near 0 (signals are rare events)
- `write_q` near 0
- `dropped` stays at 0

If `tick_q` is consistently > 10k or hitting maxsize → engine genuinely too slow, need to profile further (issue #6 — B2 signal engine work per bar).

### Effort estimate

- Change 1 (data_feed queue split): **2-3 hours** including testing
- Change 2 (NT8 async POST): **1 hour**
- Change 3 (bar persistor async): **30 min**
- Change 4 (logging refactor): **1-2 hours** (~20 print calls to convert)
- Monitor + testing: **1 hour**

**Total: ~6-8 hours of focused work.** All in `live/combined/` — no NT8 addon changes needed.

### Risk profile

These changes don't change the strategy logic at all. The coordinator, engines, and gate logic are untouched. Only the data plumbing changes.

Verification path:
1. After changes, run live for one session
2. Watch for `slow_client` errors → should be GONE
3. Run `replay_today.py --date <today>` and diff bars vs live's cached pickle → should match exactly
4. Compare today's live signals vs replay signals → should match exactly (modulo the existing 5.3% RV mismatch from earlier validation)

If anything diverges, the queue depth metrics will point at the culprit.

---

## TL;DR — root architecture

The current pipeline is **fully synchronous in one thread**:

```
Databento gateway
      ↓ tick
data_feed.py:69  for msg in self.client:
      ↓ direct call (no queue)
data_feed.py:112  self.on_tick(ts_et, price, size, side)
      ↓
bar_builder.py:83  on_tick → _apply_tick  (fast: ~10µs per tick)
      ↓ on bar close
bar_builder.py:107  for cb in self.subscribers:  cb(completed)  (SYNCHRONOUS)
      ↓
   ├── settle_recorder.on_bar       (cheap, only writes JSON at 16:00 ET)
   ├── make_print_handler           (cheap, suppressed after 5 bars)
   ├── bar_persistor.on_bar         ← ⚠️ pickle write per 5-min bar
   ├── rv.on_bar  /  b2.on_20min_bar  /  od.on_20min_bar  ← engine work
   ├── b2_sig.on_5min_bar           ← ⚠️ heavy signal computation
   ├── fb.on_5min_bar               ← Fabio
   └── (coordinator → nt8_executor.send_entry)   ← ⚠️ blocking HTTP POST
```

**If any subscriber takes >50ms, ticks pile up at the Databento gateway during that window.** Bar boundaries are when multiple subscribers fire in sequence — that's when the gateway drops records.

The 09:53 + 09:55 timing fits this exactly: the 09:55 5-min bar close triggers heavy B2/Fabio signal work, the cash-open tick burst is highest in those minutes, gateway buffer fills, records get dropped.

---

## Issues ranked by impact

### 🔴 #1 — No queue between Databento client and bar builder
**File**: `live/combined/data_feed.py:69-114`

```python
for msg in self.client:                    # ← gateway reads back-pressured to this loop
    ...
    self.on_tick(ts_et, price, size, side)  # ← whatever happens next blocks the loop
```

The Databento iterator backpressures based on consumption speed. If `on_tick` blocks, gateway eventually drops records.

**Fix sketch**: Insert a `queue.Queue` between the iterator and the bar builder. Dedicated reader thread shoves records onto queue (microseconds per op). Engine thread drains queue at its own pace.

```python
tick_queue = queue.Queue(maxsize=100_000)

def _reader_thread():
    for msg in self.client:
        if msg has action == 'T':
            tick_queue.put((ts_et, price, size, side), block=False)
        # everything else (parsing, dispatch) moves to engine thread

def _engine_thread():
    while not stopped:
        try:
            tick = tick_queue.get(timeout=1.0)
            bar_builder.on_tick(*tick)
        except queue.Empty:
            continue
```

Monitor `tick_queue.qsize()` — if consistently >10k, the engine thread is too slow and needs further profiling.

---

### 🔴 #2 — Synchronous bar subscriber dispatch
**File**: `live/combined/bar_builder.py:107-113` and `:148-152`

```python
if bar_open != self.current.open_time:
    completed = self.current
    for cb in self.subscribers:        # ← serial, blocking, all in tick thread
        try:
            cb(completed)
        except Exception as e:
            ...
```

When a bar closes, every subscriber runs in series in the same thread that's consuming ticks. If RV's bar handler takes 30ms and B2's takes 80ms and bar_persistor takes 20ms, that's **130ms of blocked tick consumption** at every bar boundary.

**Fix sketch**: Either
- (a) Move subscriber dispatch onto the engine thread (combined with fix #1), OR
- (b) Use a separate thread/queue per subscriber for non-critical ones (bar_persistor, settle_recorder)

The trading engines themselves must remain ordered because the coordinator depends on signal sequencing — those stay synchronous in the engine thread. But auxiliary subscribers (persistor, settle, print) can run async.

---

### 🟠 #3 — Bar persistor writes entire day's pickle on every 5-min bar
**File**: `live/combined/bar_persistor.py:93-98`

```python
def _atomic_write(self, session_date, bars):
    path = _pickle_path(session_date, self.cache_dir)
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(bars, f)        # ← serializes ENTIRE day's bars every 5 min
    os.replace(tmp, path)
```

Cost: ~5-20ms per bar close (depends on day's bar count + disk). 288 bars/day max for 24h session.

**Fix sketch**:
- Use `pickle.dump` with `protocol=5` + append-mode if possible, OR
- Write delta-only (just the new bar) to a sidecar log, periodically merge to canonical pickle
- Move this whole operation to a separate IO thread so the bar-close path doesn't wait on disk

---

### 🟠 #4 — NT8 HTTP POST is blocking with 5-second timeout
**File**: `live/combined/nt8_executor.py:79-102`

```python
def _post(self, endpoint, payload, timeout=5.0):
    ...
    r = self.session.post(full_url, json=payload, timeout=timeout)  # ← BLOCKS THREAD
    ...
```

On every entry/exit signal, the tick thread (or engine thread, post-fix) blocks for the round-trip. Typical RTT is ~3-10ms when NT8 is healthy, but if NT8 is slow, deadlocked, or unreachable, this stalls for up to **5 SECONDS**.

5 seconds at peak tick rate = potentially **5,000-10,000 ticks** lost.

**Fix sketch**:
- Move signal POSTing to a dedicated executor thread with a bounded queue
- Engine puts (strat, direction, qty, ...) on `signal_queue` instantly, returns
- Executor thread pulls from queue, does HTTP POST, retries on failure
- Bonus: this also decouples the engine from NT8 connectivity issues

Reduce timeout to 1.0s — if NT8 doesn't respond in 1s, it's not going to respond, queue the retry.

---

### 🟠 #5 — Terminal printing on hot paths (Windows console is slow)
**Files**: multiple

`live/combined/run_phase1.py:102-103`:
```python
print(f"  [{strat}-{ev}] {ts_et.strftime('%H:%M')} {direction} @ ...")
```

`live/combined/nt8_executor.py:89`:
```python
print(f"  [nt8] POST {endpoint} OK ({rtt_ms:.0f}ms): {payload.get('tag','')}")
```

`live/combined/b2_signal_engine.py:253` (and a few others):
```python
print(f"  [b2_signal_engine] RTH morning reload: ...")
print(f"  [B2-SIG] {ts.strftime(...)} {direction} ...")
```

On Windows, `print()` to a PowerShell terminal can be **5-50ms per call** depending on terminal renderer (worse with conhost.exe redrawing, faster with Windows Terminal). Multiple prints per signal × multiple strategies firing simultaneously = noticeable.

**Fix sketch**:
- Replace `print()` with `logging.info()` and route logs through `QueueHandler` → background `QueueListener` thread that does the actual `stderr` write
- Lazy string formatting via `log.info("foo %s", bar)` (only formats if level enabled)
- Or just nuke the per-signal prints on the hot path entirely and rely on the CSV trade logs

---

### 🟡 #6 — B2 signal engine heavy work on every 5-min RTH bar
**File**: `live/combined/b2_signal_engine.py:212-274`

`on_5min_bar` does on every RTH 5-min bar (78 bars/RTH session):
- Bias state update (~µs)
- Confirmation eval if pending (some pandas ops, ~ms range)
- New signal detection: pinbar calc + level proximity scan + windowed orderflow check (~5-20ms typical)

Not catastrophic individually, but combined with #2 (synchronous subscriber chain), it stacks up.

**Fix sketch**: profile `_try_new_signal` with `cProfile`. The "windowed orderflow scan" uses `level_volumes` dict — make sure it's not iterating the entire price-level dict (which can be 100+ keys on volatile bars). Use precomputed indices.

---

### 🟡 #7 — Signal CSV file opened/closed per write
**File**: `live/combined/run_phase1.py:100-101`

```python
with open(log_files[strat], "a", newline="") as f:
    csv.writer(f).writerow(line)
```

Opens + closes the file on every signal. Fine for sparse signals (a few per day), but if any code path emits signals rapidly (e.g., signal-spam bug) this becomes IO-heavy.

**Fix sketch**: Keep file handles open at engine startup, flush after each write. Or batch writes via a buffered logger.

---

## What I DIDN'T find as an issue

- `bar_builder._apply_tick` is fast (microseconds per tick — pure dict mutation)
- `settle_recorder.on_bar` only writes JSON at 16:00 ET — single event per day
- Heartbeat thread is properly threaded (`nt8_executor.py:69`)
- `print_handler` correctly suppresses after first 5 bars
- Gamma parquet reload is rare (session boundary only) and small (~1300 rows)

---

## Recommended fix order

1. **Issue #1 + #2 combined** — add `tick_queue` + dedicated engine thread. This is THE foundational change; everything else is secondary.
2. **Issue #4** — async HTTP POST executor thread. Critical before going live with real money.
3. **Issue #3** — move bar persistor to its own IO thread.
4. **Issue #5** — replace prints with proper async logging.
5. **Issue #6** — profile B2 signal engine and optimize windowed scan if needed.
6. **Issue #7** — keep CSV files open, batch flushes.

After (1) + (2) are in place, add a queue depth monitor:
- Log `tick_queue.qsize()` every 30s
- Alert if size > 10K (engine falling behind)
- Alert if size hits maxsize (data loss — gateway will drop)

---

## Verification after fixes

To confirm the fix works, run for a full session and check:

1. **No `skipped_records_after_slow_reading` errors** in the log
2. **`tick_queue.qsize()` stays below 1000** even during cash open
3. **Bar volume matches replay** — run `python live/combined/replay_today.py --date <today>` and verify `total_vol` per 5-min bar matches what live recorded in the pickles
4. **No engine signal differences** vs replay (already validated at 100% match for B2/OD/FB before — should remain)

If bar volumes diverge or replay shows different signals, the architecture fix alone isn't enough — there's still data loss happening upstream.

---

## Notes

- The current architecture worked during paper testing because paper sessions didn't see the full tick rate (warm-start replay is rate-limited by your CPU, not the gateway). The issue only surfaced when going live with real-time Databento at peak rate.
- All 4 strategies are validated at 100% (B2/OD/FB) and 94.7% (RV) replay-vs-backtest match. That validation is on COMPLETE tick data. If live drops ticks, the live results will diverge from backtest in proportion to how many ticks were missed.
- The 09:55 cash-open window is the worst case. Off-peak (afternoon, overnight) shouldn't see this error unless the engine has a different bottleneck.
