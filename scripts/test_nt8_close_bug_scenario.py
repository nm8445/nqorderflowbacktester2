"""Test the EXACT bug scenario on NT8 Sim101 with the patched addon.

Sequence:
1. POST ENTRY (SHORT 1 contract, no SL/TP) — addon submits sell-short on broker,
   tracks the tag locally.
2. WAIT for user to manually flatten the short via NT8 GUI (right-click position
   -> Close). After this: broker is FLAT, but addon's tracking dict STILL has
   the tag.
3. POST CLOSE_TAG for that tag.
   - WITH PATCH: addon checks broker positions, sees Flat, ABORTS — no order sent.
                  No new long opens. Tag is untracked locally.
   - WITHOUT PATCH: addon sends BuyToCover -> opens a new LONG (bug).
4. GET /status to confirm tracking is clean.

Prereqs:
- Patched addon recompiled (F5 in NinjaScript editor) and reloaded.
- Account in addon set to Sim101 (or any safe sim).
- MNQ instrument trades during normal hours.
"""
import requests
import json
import time
import sys
from datetime import datetime

NT8_URL = "http://localhost:8081"
TAG = f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}_SHORT"


def get_status():
    try:
        r = requests.get(f"{NT8_URL}/status", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def post_order(payload, label):
    print(f"\n--- POST /order {label} ---")
    print(f"  payload: {json.dumps(payload, indent=2)}")
    try:
        r = requests.post(f"{NT8_URL}/order", json=payload, timeout=5)
        rtt_ms = r.elapsed.total_seconds() * 1000
        print(f"  Status: {r.status_code}  RTT: {rtt_ms:.0f}ms")
        if r.text:
            print(f"  Response: {r.text[:500]}")
        return r.status_code == 200
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    print("=" * 70)
    print(f"NT8 CLOSE-BUG SCENARIO TEST  (tag={TAG})")
    print("=" * 70)

    s0 = get_status()
    if s0 is None:
        print("ABORT: NT8 addon not reachable at " + NT8_URL)
        return
    print(f"\nInitial addon state:")
    print(f"  account={s0.get('account')}  instrument={s0.get('instrument')}")
    print(f"  tracked positions: {s0.get('open_positions', 0)}  tags={s0.get('tags', [])}")

    # ---- Step 1: send ENTRY (SHORT 1 ct, no SL/TP) ----
    entry_payload = {
        "action": "ENTRY",
        "tag": TAG,
        "strat": "TEST",
        "direction": "SHORT",
        "quantity": 1,
        "order_type": "MARKET",
        "instance_id": "test-bug-script",
    }
    if not post_order(entry_payload, "ENTRY SHORT 1ct"):
        print("ABORT: ENTRY failed")
        return
    time.sleep(2)

    s1 = get_status()
    print(f"\nAfter ENTRY:")
    print(f"  tracked: {s1.get('open_positions', 0)}  tags={s1.get('tags', [])}")
    if TAG not in (s1.get("tags") or []):
        print(f"WARN: tag {TAG} not in tracking after ENTRY — addon may not have processed it")

    # ---- Step 2: wait for user to manually flatten ----
    print("\n" + "=" * 70)
    print("STEP 2: NOW GO TO NT8 CONTROL CENTER → Positions tab")
    print(f"   Right-click the short MNQ position → Close")
    print(f"   (This simulates the manual interference that caused the bug)")
    print("=" * 70)
    input("Press ENTER once you've manually closed the position in NT8...")

    # Confirm broker is actually flat
    print("\n(Optionally verify in NT8 Positions tab: should show 0 contracts on MNQ)")

    # ---- Step 3: send CLOSE_TAG ----
    close_payload = {
        "action": "CLOSE_TAG",
        "tag": TAG,
        "reason": "test_bug_scenario",
        "instance_id": "test-bug-script",
    }
    print(f"\nSending CLOSE_TAG for {TAG}")
    print("Expected addon log WITH PATCH:")
    print(f"  CLOSE ABORTED {TAG}: broker side=Flat qty=0 but tracked=SHORT qty=1.")
    print(f"  Likely closed externally (manual flatten). Untracking only — no order sent.")
    print("\nExpected addon log WITHOUT PATCH (BUG):")
    print(f"  CLOSE {TAG} reason=test_bug_scenario")
    print(f"  ... and then a NEW LONG MNQ position opens.")

    post_order(close_payload, "CLOSE_TAG")
    time.sleep(2)

    # ---- Step 4: verify state ----
    s2 = get_status()
    print(f"\nAfter CLOSE_TAG:")
    print(f"  tracked: {s2.get('open_positions', 0)}  tags={s2.get('tags', [])}")

    print("\n" + "=" * 70)
    print("RESULT — Verify in NT8:")
    print("=" * 70)
    print("1. CHECK NT8 Control Center → Positions tab:")
    print("     If FLAT (0 contracts on MNQ) → patch works ✓")
    print("     If you see a NEW LONG position → patch did NOT apply (still old code)")
    print(f"\n2. CHECK NT8 addon UI log (the receiver window):")
    print(f"     Look for the CLOSE ABORTED line for tag {TAG}")
    print(f"\n3. Tracking should be empty: tags={s2.get('tags', [])}")
    print("=" * 70)


if __name__ == "__main__":
    main()
