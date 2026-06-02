"""NT8 close-path test — safe (won't risk opening positions).

Steps:
1. GET /status — see what positions the addon currently tracks
2. POST CLOSE_TAG with a NON-EXISTENT tag — should log "position not found"
   in the addon. No order sent regardless of old/new patch state.
3. Optionally: POST CLOSE_TAG for a REAL tracked tag (only do this if you've
   already recompiled the patched addon). Set REAL_TAG below.
"""
import requests
import json
import sys

NT8_URL = "http://localhost:8081"

FAKE_TAG = "TEST_20260101_000000_LONG"   # guaranteed not in tracking

# If you've recompiled the patched addon and want to test the safety abort
# against a real tracked tag (where broker is flat), set this:
REAL_TAG: str | None = None    # e.g., "RV_20260527_142000_SHORT"


def get_status():
    print("\n--- GET /status ---")
    try:
        r = requests.get(f"{NT8_URL}/status", timeout=5)
        if r.status_code == 200:
            try:
                data = r.json()
                print(json.dumps(data, indent=2))
                return data
            except Exception:
                print(r.text)
                return None
        else:
            print(f"  HTTP {r.status_code}: {r.text}")
            return None
    except requests.exceptions.ConnectionError:
        print(f"  CONNECTION FAILED. Is the NT8 addon running on {NT8_URL}?")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def post_close(tag: str):
    payload = {
        "action": "CLOSE_TAG",
        "tag": tag,
        "reason": "manual_test",
        "instance_id": "test-script",
    }
    print(f"\n--- POST /order CLOSE_TAG  tag={tag} ---")
    try:
        r = requests.post(f"{NT8_URL}/order", json=payload, timeout=5)
        rtt_ms = r.elapsed.total_seconds() * 1000
        print(f"  Status: {r.status_code}  RTT: {rtt_ms:.0f}ms")
        print(f"  Response: {r.text[:500]}")
    except requests.exceptions.ConnectionError:
        print(f"  CONNECTION FAILED. Is the NT8 addon running on {NT8_URL}?")
    except Exception as e:
        print(f"  ERROR: {e}")


def main():
    print("=" * 70)
    print("NT8 CLOSE-PATH TEST")
    print("=" * 70)

    status = get_status()

    print("\n=== TEST 1: CLOSE_TAG with FAKE tag (always safe) ===")
    print("Expected addon log: 'CLOSE_TAG <tag>: position not found (already closed?)'")
    post_close(FAKE_TAG)

    if REAL_TAG:
        print(f"\n=== TEST 2: CLOSE_TAG with REAL tag {REAL_TAG} ===")
        print("If patched: expect 'CLOSE ABORTED ... broker side=Flat' (safe abort)")
        print("If NOT patched: WILL OPEN A NEW POSITION (dangerous!)")
        confirm = input("Proceed with REAL_TAG close? (y/N): ").strip().lower()
        if confirm == "y":
            post_close(REAL_TAG)
            print("\n--- GET /status (after) ---")
            get_status()
        else:
            print("  Skipped.")

    print("\n" + "=" * 70)
    print("DONE. Check NT8 addon UI log for the response messages.")
    print("=" * 70)


if __name__ == "__main__":
    main()
