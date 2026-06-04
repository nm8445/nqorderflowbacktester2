"""Read-only farm monitor — the safe Phase 0 -> 1 bridge.

Polls the test addon's GET /accounts, feeds the eval brain (EvalFarm.sync_accounts), and prints each
account's live state. **Places NO orders.** Manually trade a sim account in NT8 and watch the brain
track FRESH -> ACTIVE -> BLOWN / PASSED purely from the equity feed — this validates the whole
equity -> brain pipeline before any order routing exists.

Sim accounts report TrailingMaxDrawdown = 0 (no prop risk plugin), so for them we OMIT the live DD and
let the brain compute the floor from the configured DD (start $2k, DD $2k -> floor $0, buffer $2k).
Real prop accounts (live_dd=True) pass NT8's TrailingMaxDrawdown straight through.

Run: python live/farm/farm_monitor.py
"""
from __future__ import annotations
import os
import time
from accounts_client import fetch_accounts
from eval_passer import EvalFarm

# Accounts the farm WATCHES, and how to source each one's DD.
#   live_dd=True  -> trust NT8's TrailingMaxDrawdown (real prop accounts)
#   live_dd=False -> NT8 reports 0 (sim accounts) -> brain computes the floor from `dd`
# MFF funded + your live account are deliberately ABSENT -> never tracked, never traded.
REGISTRY = {
    "SimEval": {"dd": 2000.0, "live_dd": False},
}


def _cfg(name: str):
    for pat, c in REGISTRY.items():
        if pat in name:
            return c
    return None


def main():
    farm = EvalFarm(copies=1, day_cap=1500.0, quiet=False)   # quiet=False: print transitions as they happen
    print("READ-ONLY farm monitor — no orders are ever placed. Ctrl-C to stop.\n")
    try:
        while True:
            try:
                accts = fetch_accounts()
            except Exception as e:
                print(f"  [poll failed: {e}]  is the TEST addon up on :8082?")
                time.sleep(3.0)
                continue
            snap = {}
            for a in accts:
                c = _cfg(a["name"])
                if c is None:
                    continue                       # not a watched account -> ignored entirely
                m = {"cash": a["cash"], "equity": a["netliq"]}
                if c["live_dd"] and a.get("dd", 0) > 0:
                    m["dd"] = a["dd"]              # real account: trust NT8's remaining DD
                # sim: omit dd -> brain computes floor from its dd field (default $2k = our config)
                snap[a["name"]] = m
            farm.sync_accounts(snap)
            os.system("cls" if os.name == "nt" else "clear")
            print(f"FARM MONITOR (read-only)  {time.strftime('%H:%M:%S')}   {dict(farm.counts())}")
            print(f"watching: {sorted(farm.accounts)}\n")
            print(farm.state_table() or "  (no watched accounts found in /accounts yet)")
            time.sleep(3.0)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
