"""
NT8 Signal Client - sends signals to NQ Order Flow Signal Receiver add-on.

Posts JSON signals to http://localhost:8080/
Supports: MARKET entry, LIMIT/STOP entry, CANCEL_ENTRY, BRACKET, FLATTEN
"""
import logging
import time
from datetime import datetime
from typing import Optional

import requests

log = logging.getLogger(__name__)

NT8_URL = "http://localhost:8080/"

# Persistent HTTP session — keeps TCP connection alive across sends,
# saves ~10-30ms of handshake per request vs creating a new connection each time.
_session = requests.Session()


def send_signal_to_nt8(
    action: str,
    stop_price: float,
    target_price: float,
    quantity: int = 1,
    order_type: str = "MARKET",
    entry_price: Optional[float] = None,
    account: Optional[str] = None,
) -> bool:
    """
    Send entry signal to NT8 add-on.

    For MARKET orders: entry + OCO bracket sent together (current behavior).
    For LIMIT/STOP orders: entry only, no bracket. Bracket sent separately after fill.
    """
    signal = {
        "action": action.upper(),
        "order_type": order_type.upper(),
        "quantity": str(quantity),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if order_type.upper() == "MARKET":
        signal["stop_price"] = f"{stop_price:.2f}"
        signal["target_price"] = f"{target_price:.2f}"
    else:
        # LIMIT or STOP — entry only, bracket sent after fill
        if entry_price is not None:
            signal["entry_price"] = f"{entry_price:.2f}"

    if account:
        signal["account"] = account

    try:
        t_post = time.perf_counter()
        response = _session.post(NT8_URL, json=signal, timeout=5)
        rtt_ms = (time.perf_counter() - t_post) * 1000
        log.info(f"[latency] NT8 POST roundtrip: {rtt_ms:.1f}ms")

        if response.status_code == 200:
            if order_type.upper() == "MARKET":
                log.info(f"[nt8] MARKET {action} x{quantity} sent | stop={stop_price:.2f} target={target_price:.2f}")
            else:
                log.info(f"[nt8] {order_type} {action} x{quantity} sent | entry={entry_price:.2f}")
            return True
        else:
            log.error(f"[nt8] Error: status {response.status_code} - {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        log.error("[nt8] Connection failed - is NT8 Add-On running?")
        return False
    except Exception as e:
        log.error(f"[nt8] Error sending signal: {e}")
        return False


def send_bracket_to_nt8(
    stop_price: float,
    target_price: float,
    quantity: int = 1,
    account: Optional[str] = None,
) -> bool:
    """Send OCO bracket after a LIMIT/STOP entry is filled."""
    signal = {
        "action": "BRACKET",
        "stop_price": f"{stop_price:.2f}",
        "target_price": f"{target_price:.2f}",
        "quantity": str(quantity),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if account:
        signal["account"] = account

    try:
        t_post = time.perf_counter()
        response = _session.post(NT8_URL, json=signal, timeout=5)
        rtt_ms = (time.perf_counter() - t_post) * 1000
        log.info(f"[latency] NT8 POST roundtrip: {rtt_ms:.1f}ms")
        if response.status_code == 200:
            log.info(f"[nt8] BRACKET sent | stop={stop_price:.2f} target={target_price:.2f}")
            return True
        else:
            log.error(f"[nt8] Bracket error: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        log.error("[nt8] Connection failed - is NT8 Add-On running?")
        return False
    except Exception as e:
        log.error(f"[nt8] Error sending bracket: {e}")
        return False


def send_cancel_to_nt8(account: Optional[str] = None) -> bool:
    """Cancel unfilled LIMIT/STOP entry order."""
    signal = {
        "action": "CANCEL_ENTRY",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if account:
        signal["account"] = account

    try:
        t_post = time.perf_counter()
        response = _session.post(NT8_URL, json=signal, timeout=5)
        rtt_ms = (time.perf_counter() - t_post) * 1000
        log.info(f"[latency] NT8 POST roundtrip: {rtt_ms:.1f}ms")
        if response.status_code == 200:
            log.info("[nt8] CANCEL_ENTRY sent")
            return True
        else:
            log.error(f"[nt8] Cancel error: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        log.error("[nt8] Connection failed - is NT8 Add-On running?")
        return False
    except Exception as e:
        log.error(f"[nt8] Error sending cancel: {e}")
        return False


def get_nt8_status() -> Optional[dict]:
    """Get NT8 status (position, fill state) via GET request."""
    try:
        response = requests.get(NT8_URL + "status", timeout=3)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def send_flatten_to_nt8(exit_type: str, exit_price: float, account: Optional[str] = None) -> bool:
    """Send flatten signal to NT8 add-on to close position."""
    signal = {
        "action": "FLATTEN",
        "exit_type": exit_type,
        "exit_price": f"{exit_price:.2f}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if account:
        signal["account"] = account

    try:
        t_post = time.perf_counter()
        response = _session.post(NT8_URL, json=signal, timeout=5)
        rtt_ms = (time.perf_counter() - t_post) * 1000
        log.info(f"[latency] NT8 POST roundtrip: {rtt_ms:.1f}ms")

        if response.status_code == 200:
            log.info(f"[nt8] Flatten sent: {exit_type} @ {exit_price:.2f}")
            return True
        else:
            log.error(f"[nt8] Error: status {response.status_code} - {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        log.error("[nt8] Connection failed - is NT8 Add-On running?")
        return False
    except Exception as e:
        log.error(f"[nt8] Error sending flatten: {e}")
        return False
