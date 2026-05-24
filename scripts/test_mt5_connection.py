"""Connection smoke test for MT5 (FundedNext).

What this verifies (in order):
  1. MetaTrader5 package import + initialize
  2. Login with credentials from .env
  3. Account info (balance, equity, leverage, server)
  4. Available symbols (search for NAS100 variants)
  5. Current tick (bid/ask) on NAS100
  6. Recent 5-min bars for NAS100
  7. Open positions (should be empty)

NO orders are placed. This is read-only verification.
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
import MetaTrader5 as mt5

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

LOGIN = int(os.getenv("MT5_LOGIN", "0"))
PASSWORD = os.getenv("MT5_PASSWORD", "")
SERVER = os.getenv("MT5_SERVER", "")
# Optional: path to terminal64.exe if not in default location
TERMINAL_PATH = os.getenv("MT5_TERMINAL_PATH", "")

# Common NAS100 symbol names across brokers — we'll search all
NAS100_CANDIDATES = [
    "NAS100", "NAS100.cash", "NAS100cash", "NAS100.fs",
    "USTECH100", "USTEC", "USTEC.cash",
    "US100", "US100.cash", "US100cash",
    "NDX100", "NDX", "NDX.cash",
    "USA100", "USA100.cash",
    "NQ100", "NQ100.cash",
]


def hr(t=""):
    print(f"\n{'='*70}\n{t}\n{'='*70}" if t else "-"*70)


def main():
    hr("Step 1: initialize MT5")
    init_ok = mt5.initialize(path=TERMINAL_PATH) if TERMINAL_PATH else mt5.initialize()
    if not init_ok:
        print(f"FAIL initialize: {mt5.last_error()}")
        print("Check: is the MT5 terminal installed?")
        print("If installed at non-default path, set MT5_TERMINAL_PATH in .env, e.g.:")
        print("  MT5_TERMINAL_PATH=C:/Program Files/FundedNext MT5/terminal64.exe")
        return
    print(f"OK   initialize")
    print(f"     terminal version: {mt5.version()}")
    print(f"     terminal info:    {mt5.terminal_info()._asdict() if mt5.terminal_info() else 'NONE'}")

    hr(f"Step 2: login (account={LOGIN}, server={SERVER!r})")
    if not LOGIN or not PASSWORD or not SERVER:
        print("FAIL: MT5_LOGIN / MT5_PASSWORD / MT5_SERVER not set in .env")
        mt5.shutdown()
        return
    ok = mt5.login(login=LOGIN, password=PASSWORD, server=SERVER)
    if not ok:
        print(f"FAIL login: {mt5.last_error()}")
        print("Common causes:")
        print(" - Server name slightly off (try 'FundedNext-Server03' or 'FundedNext-Server-3')")
        print(" - Account on a different MT5 build (FundedNext may require their custom installer)")
        print(" - Password contains special chars that .env couldn't parse — try quoting it")
        mt5.shutdown()
        return
    print(f"OK   login successful")

    hr("Step 3: account info")
    ai = mt5.account_info()
    if ai is None:
        print(f"FAIL account_info: {mt5.last_error()}")
    else:
        for k in ["login", "name", "server", "currency", "balance", "equity",
                  "margin", "margin_free", "leverage", "trade_allowed",
                  "trade_expert", "limit_orders"]:
            print(f"     {k:>18}: {getattr(ai, k, '?')}")

    hr("Step 4: search for NAS100 symbol")
    all_syms = mt5.symbols_get()
    if all_syms is None:
        print(f"FAIL symbols_get: {mt5.last_error()}")
        mt5.shutdown(); return
    print(f"     {len(all_syms):,} total symbols available")
    found_nas = []
    for cand in NAS100_CANDIDATES:
        info = mt5.symbol_info(cand)
        if info is not None:
            found_nas.append(cand)
    if not found_nas:
        # broad text search
        nas_like = [s.name for s in all_syms
                    if any(k in s.name.upper() for k in ["NAS", "NDX", "US100", "USTEC", "USTECH"])]
        print(f"     none of {NAS100_CANDIDATES} found.")
        print(f"     fuzzy-match candidates ({len(nas_like)}): {nas_like[:20]}")
        mt5.shutdown(); return
    print(f"OK   matching symbols: {found_nas}")
    nas_sym = found_nas[0]
    info = mt5.symbol_info(nas_sym)
    if not info.visible:
        # Symbol exists but not in Market Watch — enable
        if not mt5.symbol_select(nas_sym, True):
            print(f"FAIL symbol_select({nas_sym}): {mt5.last_error()}")
            mt5.shutdown(); return
        print(f"     enabled in Market Watch")
    print(f"     symbol: {nas_sym}")
    print(f"     digits: {info.digits}   point: {info.point}")
    print(f"     min lot: {info.volume_min}   max lot: {info.volume_max}   step: {info.volume_step}")
    print(f"     contract size: {info.trade_contract_size}")
    print(f"     spread (current): {info.spread} points")
    print(f"     swap_long/short: {info.swap_long} / {info.swap_short}")

    hr(f"Step 5: current tick on {nas_sym}")
    tick = mt5.symbol_info_tick(nas_sym)
    if tick is None:
        print(f"FAIL symbol_info_tick: {mt5.last_error()}")
    else:
        for k in ["time", "bid", "ask", "last", "volume", "flags"]:
            print(f"     {k:>10}: {getattr(tick, k, '?')}")
        print(f"     spread: {tick.ask - tick.bid:.{info.digits}f}")

    hr(f"Step 6: last 5 bars (5-min) on {nas_sym}")
    bars = mt5.copy_rates_from_pos(nas_sym, mt5.TIMEFRAME_M5, 0, 5)
    if bars is None or len(bars) == 0:
        print(f"FAIL copy_rates: {mt5.last_error()}")
    else:
        print(f"     got {len(bars)} bars")
        for b in bars:
            from datetime import datetime
            t = datetime.fromtimestamp(b["time"]).strftime("%Y-%m-%d %H:%M")
            print(f"     {t}  O={b['open']:.2f} H={b['high']:.2f} L={b['low']:.2f} C={b['close']:.2f}  vol={b['tick_volume']}")

    hr("Step 7: open positions")
    pos = mt5.positions_get()
    print(f"     {0 if pos is None else len(pos)} open positions")
    if pos:
        for p in pos:
            print(f"     ticket={p.ticket} sym={p.symbol} vol={p.volume} type={p.type} px={p.price_open} pnl={p.profit}")

    hr("All checks done")
    mt5.shutdown()


if __name__ == "__main__":
    main()
