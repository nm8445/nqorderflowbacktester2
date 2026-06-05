"""Client for the test addon's GET /accounts (port 8082). Phase-0 verification tool and the seed of
the app backend — this is what will feed EvalFarm/FundedFarm.sync_accounts() and, later, the FastAPI
WebSocket the dashboard subscribes to.

Run `python live/farm/accounts_client.py` to poll and print every NT8 account's balance/equity once
the addon is compiled and started in NinjaTrader.
"""
from __future__ import annotations
import json
import os
import sys
import time
import urllib.parse
import urllib.request

URL = "http://localhost:8082/accounts"
DUMP_URL = "http://localhost:8082/account_dump"
ORDER_URL = "http://localhost:8082/order"
CLOSE_URL = "http://localhost:8082/close"
FLATTEN_URL = "http://localhost:8082/flatten"
APP_FIRE_URL = "http://localhost:8090/api/fire_signal"   # routes via the dashboard's brain

# --- which accounts to show ---------------------------------------------------------------------
# NT8's Account.All includes every dead/old account ever added, and a FRESH account reads $0 just
# like a dead one (and shares the prop's naming prefix), so neither balance nor prefix can separate
# "current" from "old". The only reliable filter is an explicit list of the exact accounts you're
# using now — i.e. the account registry the farm needs anyway. Add a new eval's id when you buy it,
# remove one when you're done. Empty list = show everything (so you can find new ids).
ALLOW: list[str] = ["SimEval1", "SimEval2","SimEval3","SimEval4","SimEval5"]   # funded 50k + live; add evals as you buy them


def fetch_accounts(url: str = URL, timeout: float = 3.0) -> list[dict]:
    """GET /accounts -> [{name, connected, status, cash, unrealized, netliq, positions:[...]}, ...]."""
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8")).get("accounts", [])


def _visible(a: dict) -> bool:
    return (not ALLOW) or any(s in a["name"] for s in ALLOW)


def _print_table(accts: list[dict]) -> None:
    shown = [a for a in accts if _visible(a)]
    hidden = len(accts) - len(shown)
    os.system("cls" if os.name == "nt" else "clear")            # refresh in place, no endless scroll
    flt = f"ALLOW={ALLOW}" if ALLOW else "showing ALL (set ALLOW to pin your accounts)"
    print(f"{URL}   {time.strftime('%H:%M:%S')}   ({len(shown)} shown, {hidden} hidden — {flt})\n")
    print(f"{'account':<24}{'cash':>10}{'netliq':>10}{'DD left':>9}{'today':>9}{'float':>8}  positions")
    print("  " + "-" * 82)
    for a in shown:
        pos = ", ".join(f"{p['side']} {p['qty']} {p['instrument']}" for p in a.get("positions", [])) or "flat"
        print(f"{a['name']:<24}{a['cash']:>10,.0f}{a['netliq']:>10,.0f}{a.get('dd',0):>9,.0f}"
              f"{a.get('realized_today',0):>+9,.0f}{a['unrealized']:>+8,.0f}  {pos}")


def dump_account(name: str) -> None:
    """GET /account_dump?name=... and print every AccountItem sorted, so we can spot the trailing DD."""
    url = f"{DUMP_URL}?name={urllib.parse.quote(name)}"
    with urllib.request.urlopen(url, timeout=5.0) as r:
        data = json.loads(r.read().decode("utf-8"))
    if "error" in data:
        print(f"  {data['error']}  (name={data.get('name')})")
        return
    print(f"\n{data['name']}   status={data['status']}   ({len(data['items'])} AccountItems)\n")
    for k in sorted(data["items"]):
        print(f"  {k:<28}{data['items'][k]:>16,.2f}")
    print("\nLook for a field that equals your dashboard's kill-level or DD-available "
          "(balance - $2,000-ish). If none match, NT8 doesn't expose it -> we compute it.")


def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5.0) as r:
        return json.loads(r.read().decode("utf-8"))


def place_order(account: str, direction: str, qty, sl=None, tp=None) -> None:
    """python accounts_client.py order <account> <LONG|SHORT> <qty> [slPrice] [tpPrice]"""
    p = {"account": account, "strat": "TEST", "direction": direction.upper(), "qty": int(qty)}
    if sl:
        p["slPrice"] = float(sl)
    if tp:
        p["tpPrice"] = float(tp)
    print("ORDER ->", _post(ORDER_URL, p))


def close_order(account: str, tag: str) -> None:
    """python accounts_client.py close <account> <tag>"""
    print("CLOSE ->", _post(CLOSE_URL, {"account": account, "tag": tag, "reason": "manual"}))


def flatten_account(account: str) -> None:
    """python accounts_client.py flatten <account>   (closes whatever the account holds, tag or not)"""
    print("FLATTEN ->", _post(FLATTEN_URL, {"account": account, "reason": "manual"}))


def fire(strat: str, direction: str, stop_pts) -> None:
    """python accounts_client.py fire <strat> <LONG|SHORT> <stop_pts>  (routes via the app to the sims)"""
    print("FIRE ->", _post(APP_FIRE_URL, {"strat": strat, "direction": direction.upper(),
                                          "stop_pts": float(stop_pts)}))


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "dump":   # python accounts_client.py dump <ACCOUNT_NAME>
        dump_account(sys.argv[2])
        return
    if len(sys.argv) >= 5 and sys.argv[1] == "order":  # order <account> <LONG|SHORT> <qty> [sl] [tp]
        place_order(*sys.argv[2:])
        return
    if len(sys.argv) >= 4 and sys.argv[1] == "close":  # close <account> <tag>
        close_order(sys.argv[2], sys.argv[3])
        return
    if len(sys.argv) >= 3 and sys.argv[1] == "flatten":  # flatten <account>
        flatten_account(sys.argv[2])
        return
    if len(sys.argv) >= 5 and sys.argv[1] == "fire":     # fire <strat> <LONG|SHORT> <stop_pts>
        fire(sys.argv[2], sys.argv[3], sys.argv[4])
        return
    try:
        while True:
            try:
                _print_table(fetch_accounts())
            except Exception as e:
                print(f"  [poll failed: {e}]  is the TEST addon compiled + started on :8082?")
            time.sleep(3.0)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
