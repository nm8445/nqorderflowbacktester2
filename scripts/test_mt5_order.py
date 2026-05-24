"""MT5 order placement smoke test on NDX100 (FundedNext NAS100).

Places a TINY market BUY for 0.01 lots NDX100, then immediately closes it.
Approx exposure: $10/pt × 0.01 lots = $0.10/pt. A 5-pt slippage = $0.50.
Effectively zero real risk; the point is to verify the round-trip works.

What this validates:
  1. order_send() returns a successful retcode (TRADE_RETCODE_DONE)
  2. The position appears in positions_get()
  3. We can extract the position ticket and price
  4. A closing market order using `position=ticket` closes it cleanly
  5. positions_get() is empty after close

REQUIRES: market open (Globex). NDX100 bid/ask must be non-zero.
Run after Sunday 18:00 ET when Globex reopens.

SAFETY:
  - Only the FundedNext account (34019900) is touched
  - 0.01 lot = smallest legal size on this broker
  - Order is closed within ~1 second of fill
  - If anything goes wrong, you can manually close in MT5 (Trade tab → right-click position → Close)

CLI: python scripts/test_mt5_order.py
       python scripts/test_mt5_order.py --skip-close  (leaves position open for manual inspection)
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
import MetaTrader5 as mt5

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

LOGIN = int(os.getenv("MT5_LOGIN", "0"))
PASSWORD = os.getenv("MT5_PASSWORD", "")
SERVER = os.getenv("MT5_SERVER", "")
TERMINAL_PATH = os.getenv("MT5_TERMINAL_PATH", "")
SYMBOL = os.getenv("MT5_SYMBOL", "NDX100")   # broker-specific (NAS100/US100/USTECH100)
LOT_SIZE = 0.01
TAG_PREFIX = "TEST"


def hr(t=""):
    print(f"\n{'='*70}\n{t}\n{'='*70}" if t else "-"*70)


def init_and_login():
    init_ok = mt5.initialize(path=TERMINAL_PATH) if TERMINAL_PATH else mt5.initialize()
    if not init_ok:
        print(f"FAIL init: {mt5.last_error()}")
        return False
    if not mt5.login(LOGIN, PASSWORD, SERVER):
        print(f"FAIL login: {mt5.last_error()}")
        mt5.shutdown(); return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-close", action="store_true",
                    help="Leave the test position open (for manual inspection)")
    args = ap.parse_args()

    hr("Init + login")
    if not init_and_login():
        return
    ai = mt5.account_info()
    print(f"OK   account={ai.login} balance=${ai.balance:.2f}")

    # Verify market open
    info = mt5.symbol_info(SYMBOL)
    if not info or not info.visible:
        if info and not mt5.symbol_select(SYMBOL, True):
            print(f"FAIL: can't enable {SYMBOL} in Market Watch")
            mt5.shutdown(); return
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None or tick.ask == 0 or tick.bid == 0:
        print(f"FAIL: market closed (ask={getattr(tick,'ask',0)}, bid={getattr(tick,'bid',0)})")
        print("       NDX100 follows Globex. Closed Fri 17:00 ET — Sun 18:00 ET.")
        print("       Re-run when market is open.")
        mt5.shutdown(); return
    print(f"OK   tick: bid={tick.bid:.2f} ask={tick.ask:.2f} spread={tick.ask-tick.bid:.2f}")

    # Pick a supported filling mode (brokers vary: FOK, IOC, or RETURN).
    fm_bits = info.filling_mode
    if fm_bits & 2:
        filling = mt5.ORDER_FILLING_IOC; fname = "IOC"
    elif fm_bits & 1:
        filling = mt5.ORDER_FILLING_FOK; fname = "FOK"
    else:
        filling = mt5.ORDER_FILLING_RETURN; fname = "RETURN"
    print(f"OK   filling mode chosen: {fname} (broker bitmask={fm_bits})")

    # Build entry order
    tag = f"{TAG_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}"
    entry_req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT_SIZE,
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick.ask,
        "deviation": 20,           # max slippage 20 points
        "magic": 99999,             # identifier for this script
        "comment": tag,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }

    hr(f"Sending BUY {LOT_SIZE} {SYMBOL} @ market (tag={tag})")
    t0 = time.perf_counter()
    res = mt5.order_send(entry_req)
    rtt_ms = (time.perf_counter() - t0) * 1000

    if res is None:
        print(f"FAIL order_send returned None: {mt5.last_error()}")
        mt5.shutdown(); return
    print(f"     retcode={res.retcode} ({_retcode_str(res.retcode)})")
    print(f"     ticket={res.order}  deal={res.deal}  price={res.price}  vol={res.volume}")
    print(f"     RTT: {rtt_ms:.0f}ms")
    print(f"     comment from broker: {res.comment}")
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"FAIL retcode {res.retcode}: {res.comment}")
        mt5.shutdown(); return
    print(f"OK   ENTRY FILLED at {res.price}")

    # Confirm position exists
    time.sleep(0.5)
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        print(f"WARN positions_get returned 0 — position may have closed instantly")
    else:
        for p in positions:
            print(f"     position ticket={p.ticket} sym={p.symbol} vol={p.volume} "
                  f"px_open={p.price_open} px_cur={p.price_current} pnl={p.profit}")
        # Find OUR position by magic/comment
        our_pos = next((p for p in positions if p.comment == tag or p.magic == 99999), None)
        if our_pos is None:
            print(f"WARN couldn't find our position by tag/magic. Using first one.")
            our_pos = positions[0]

    if args.skip_close:
        hr("--skip-close set; leaving position open. Close manually in MT5.")
        mt5.shutdown(); return

    # Close it
    if our_pos is None:
        print("WARN no position to close; exiting")
        mt5.shutdown(); return

    hr(f"Closing position ticket={our_pos.ticket}")
    tick2 = mt5.symbol_info_tick(SYMBOL)
    close_req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": our_pos.volume,
        "type": mt5.ORDER_TYPE_SELL,           # opposite side
        "position": our_pos.ticket,            # CRITICAL: this is what makes it a CLOSE, not a new short
        "price": tick2.bid,
        "deviation": 20,
        "magic": 99999,
        "comment": f"{tag}_CLOSE",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }
    t0 = time.perf_counter()
    res2 = mt5.order_send(close_req)
    rtt_ms = (time.perf_counter() - t0) * 1000
    if res2 is None:
        print(f"FAIL close order_send returned None: {mt5.last_error()}")
        mt5.shutdown(); return
    print(f"     retcode={res2.retcode} ({_retcode_str(res2.retcode)})")
    print(f"     ticket={res2.order}  deal={res2.deal}  price={res2.price}")
    print(f"     RTT: {rtt_ms:.0f}ms")
    if res2.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"FAIL retcode {res2.retcode}: {res2.comment}")
        print(f"     **WARNING: position may still be open. Check MT5.**")
        mt5.shutdown(); return
    print(f"OK   CLOSED at {res2.price}")

    # Confirm flat
    time.sleep(0.5)
    positions = mt5.positions_get(symbol=SYMBOL)
    n_remaining = 0 if positions is None else len(positions)
    if n_remaining == 0:
        print(f"OK   confirmed FLAT — 0 open positions on {SYMBOL}")
    else:
        print(f"WARN {n_remaining} position(s) still open on {SYMBOL}:")
        for p in positions:
            print(f"     ticket={p.ticket} vol={p.volume} tag={p.comment}")

    # Final account state
    ai2 = mt5.account_info()
    pnl_usd = ai2.balance - ai.balance
    print(f"\nFinal balance: ${ai2.balance:.2f}  (delta: ${pnl_usd:+.2f})")
    print(f"Round trip complete. End-to-end RTT for entry: see above.")

    mt5.shutdown()


def _retcode_str(rc: int) -> str:
    """Map MT5 retcode int -> readable name."""
    codes = {
        10004: "REQUOTE", 10006: "REJECT", 10007: "CANCEL", 10008: "PLACED",
        10009: "DONE", 10010: "DONE_PARTIAL", 10011: "ERROR", 10012: "TIMEOUT",
        10013: "INVALID", 10014: "INVALID_VOLUME", 10015: "INVALID_PRICE",
        10016: "INVALID_STOPS", 10017: "TRADE_DISABLED", 10018: "MARKET_CLOSED",
        10019: "NO_MONEY", 10020: "PRICE_CHANGED", 10021: "PRICE_OFF",
        10022: "INVALID_EXPIRATION", 10023: "ORDER_CHANGED",
        10024: "TOO_MANY_REQUESTS", 10025: "NO_CHANGES",
        10026: "SERVER_DISABLES_AT", 10027: "CLIENT_DISABLES_AT",
        10028: "LOCKED", 10029: "FROZEN", 10030: "INVALID_FILL",
        10031: "CONNECTION", 10032: "ONLY_REAL", 10033: "LIMIT_ORDERS",
        10034: "LIMIT_VOLUME", 10035: "INVALID_ORDER", 10036: "POSITION_CLOSED",
        10038: "INVALID_CLOSE_VOLUME", 10039: "CLOSE_ORDER_EXIST",
        10040: "LIMIT_POSITIONS", 10041: "REJECT_CANCEL",
        10042: "LONG_ONLY", 10043: "SHORT_ONLY", 10044: "CLOSE_ONLY",
        10045: "FIFO_CLOSE", 10046: "HEDGE_PROHIBITED",
    }
    return codes.get(rc, f"UNKNOWN_{rc}")


if __name__ == "__main__":
    main()