# MT5 Setup — FundedNext Connection Reference

**Status as of 2026-05-22**: Connection verified end-to-end. Read-only smoke test passed (account info + symbol info + bar history + position query). Order placement not yet tested (market closed at time of setup).

---

## Verified working environment

| Item | Value |
|---|---|
| MT5 build | 5836 (28 Apr 2026) |
| Install path | `C:\Program Files\MetaTrader 5\terminal64.exe` |
| Python package | `MetaTrader5==5.0.5735` |
| Python version known working | **3.12** (3.14 has no ABI issue here — same timeout symptom from MT5 side) |
| Account | `34019900` (Nathaniel Mais FundedNext) |
| Server | **`FundedNext-Server 3`** (note the SPACE before the 3) |
| Currency | USD |
| Balance | $100,000.58 |
| Leverage | 100:1 |
| NAS100 symbol name on FundedNext | **`NDX100`** (NOT `NAS100`) |
| Min lot | 0.01 |
| Max lot | 40.0 |
| Lot step | 0.01 |
| Contract size | 10 USD per point per lot |

### Lot sizing reference (relative to NQ futures)

| Strategy size (NQ futures basis) | NDX100 lots |
|---|---|
| 1 MNQ (= 0.1 NQ) | **0.20 lots** |
| 2 MNQ | 0.40 lots |
| 3 MNQ | 0.60 lots |
| 5 MNQ | 1.00 lot |
| 10 MNQ (= 1 NQ) | 2.00 lots |

Math: 1 MNQ = $2/pt. 1 NDX100 lot = $10/pt (contract size 10 × 1 USD per point). So `1 MNQ / 1 lot = $2/$10 = 0.20 lots`.

---

## `.env` config

```
MT5_LOGIN=34019900
MT5_PASSWORD=yprKF73##
MT5_SERVER=FundedNext-Server 3
MT5_TERMINAL_PATH=C:/Program Files/MetaTrader 5/terminal64.exe
```

**Note**: `.env` is gitignored. Do not commit.

---

## Test scripts

| Script | Purpose | Status |
|---|---|---|
| `scripts/test_mt5_connection.py` | Read-only smoke test (init + login + account + symbol + bars + positions) | ✅ Passes |
| `scripts/test_mt5_order.py` | Place tiny 0.01-lot test order + close | Pending — run when Globex reopens Sunday 18:00 ET |

---

## Gotchas (learned the hard way)

### 1. Server name has a space
The server is literally `FundedNext-Server 3` (with a space before the 3). Most documentation writes it without — wrong. The MT5 server dropdown shows the correct version.

### 2. Symbol is `NDX100`, not `NAS100`
FundedNext uses `NDX100` as their Nasdaq 100 CFD ticker. The order/feed/info calls need this exact string. Common alternative names (`NAS100`, `US100`, `USTEC`, etc.) do not exist on this server.

### 3. MT5 IPC requires successful manual login first
The Python `MetaTrader5` package only works after MT5 has logged into a broker session. **First-time setup MUST do a manual `File → Login to Trading Account` and verify the Journal tab shows `'34019900' authorized on FundedNext-Server 3`** before Python's `mt5.initialize()` will work.

After the first manual login, MT5 saves credentials and auto-reconnects on launch.

### 4. Python 3.14 incompatibility — not yet observed
Tested both Python 3.12 and 3.14 against `MetaTrader5==5.0.5735`. Both produced the same `IPC timeout` error when MT5 wasn't logged in. Once MT5 was logged in, both worked. So Python 3.14 appears OK against this MT5 build (despite the package's PyPI classifiers).

### 5. MT5 settings that must be configured
- **Tools → Options → Expert Advisors**:
  - ✓ `Allow algorithmic trading` (checked)
  - ☐ `Disable algorithmic trading via external Python API` (UNCHECKED — double-negative, unchecking = enabling Python)
- **Toolbar** AutoTrading button: green/enabled

### 6. Login dialog can close without submitting
The MT5 "Login to Trading Account" dialog sometimes closes when you hit Save or Enter but does NOT actually attempt the login. **The Journal tab is the only reliable check** — if you don't see an auth attempt line within 5 seconds of clicking Login, the dialog never submitted.

### 7. Bid/ask shows 0 outside market hours
Globex closes Friday 17:00 ET, reopens Sunday 18:00 ET. During the close, `symbol_info_tick("NDX100")` returns all zeros even though MT5 is connected. This is normal — wait for market open.

---

## Quick health check sequence

After any MT5 restart or computer reboot, verify the connection works:

```
1. Open MT5 desktop
2. Wait until Journal shows "'34019900' authorized on FundedNext-Server 3"
   (saved credentials usually auto-connect within 5 seconds of launch)
3. Run: python scripts/test_mt5_connection.py
4. All 7 steps should pass
```

If step 4 fails with `IPC timeout`, MT5 isn't actually authenticated — check Journal.

---

## Architecture for live integration (NOT YET BUILT)

Per `live/mt5_external_watchdog_spec.md` and `live/live_slow_path_audit.md`:

1. **`live/combined/mt5_executor.py`** — port of `nt8_executor.py`, replaces NT8 HTTP POST with `mt5.order_send()` calls. Same `send_entry()` / `send_close_tag()` interface.
2. **Layer 1**: include wide disaster SL on entry order (e.g., 150 pts away) so Python death doesn't leave a naked position.
3. **Layer 2**: Python self-watchdog thread monitors main loop tick.
4. **Layer 3**: external `mt5_watchdog.py` process pings main engine; on 30s silence, calls `mt5.positions_close_all()` for tagged positions.

Existing coordinator + engines + signal logic stay unchanged. Estimated effort: 1-2 days.

### Multi-firm note
Each prop firm = separate MT5 terminal + separate Python process. `MetaTrader5` package can only attach to ONE terminal per Python process at a time. To trade 3 funded CFDs simultaneously (e.g., FundedNext + FundingPips + HolaPrime), need 3 MT5 installs and 3 Python instances.
