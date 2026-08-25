"""Lane rotation — the farm's de-correlation unit.

WHAT A LANE IS
  A lane is a GROUP of accounts that trade together. One signal goes to ONE lane; every eligible
  account in that lane takes it. Variance therefore comes from the NUMBER OF LANES, not the number
  of accounts: 10 lanes x 3 accounts is 10 independent bets sized 3x, not 30 independent bets.

  Typical layout (user, 2026-08-17): 30 accounts = 10 lanes x (1 Topstep + 1 Lucid + 1 Tradeify).
  The three firms in a lane see identical signals but size to their OWN plan (Topstep/Lucid 50%
  consistency -> $1,500; Tradeify 40% -> $1,200), which the per-account plan already handles.

TWO INDEPENDENT BOOKS: EVALS AND FUNDEDS
  The eval farm and the funded farm each get their OWN LaneBook -- own on/off switch, own lane count,
  own turn order. They are not linked, because the two populations are different sizes: 10 eval lanes
  alongside 3 funded accounts would, on a shared rotation, let a funded account gamble only about
  once every 4 trading days. FarmState picks one lane per book per signal.
  MILKERS ARE EXEMPT: milking (eval and funded) is deliberately correlated -- every milker copies
  every signal, lanes or not. The de-correlation budget is spent on the gamble/eval-taking phase.

OPT-IN, PER ACCOUNT
  `enabled` is False by default and an account with no lane assignment keeps the legacy behaviour
  (the global least-recently-traded round-robin). So lanes can be switched on with only some accounts
  assigned: the assigned ones rotate by lane, the rest carry on exactly as before.

BROKER-AGNOSTIC
  Lanes are keyed by account-id string only. Nothing here knows about NT8, so a Topstep account
  arriving over its own API drops into a lane the same way an NT8 account does.
"""
from __future__ import annotations

import json
import os

MIN_LANES = 2
MAX_LANES = 20
DEFAULT_LANES = 10


class LaneBook:
    """Lane assignments + the lane round-robin. Owned by FarmState; mutate under its lock."""

    def __init__(self, enabled: bool = False, n_lanes: int = DEFAULT_LANES):
        self.enabled = bool(enabled)
        self.n_lanes = self._clamp(n_lanes)
        self.assign: dict[str, int] = {}     # account_id -> lane number (1-based)
        self._seq: dict[int, int] = {}       # lane -> recency marker (bigger = more recently served)
        self._tick = 0

    @staticmethod
    def _clamp(n: int) -> int:
        return max(MIN_LANES, min(MAX_LANES, int(n)))

    # -- assignment ----------------------------------------------------------------------------
    def lane_of(self, account_id: str) -> int | None:
        """The account's ACTIVE lane, or None if unassigned. An assignment above the current
        n_lanes reads as None (the account falls back to the legacy rotation) but is KEPT, so
        widening n_lanes again restores it instead of losing the layout."""
        ln = self.assign.get(account_id)
        return ln if ln is not None and 1 <= ln <= self.n_lanes else None

    def set_lane(self, account_id: str, lane: int | None) -> dict:
        """Assign an account to a lane. lane None/0 -> unassign (back to the legacy rotation)."""
        if lane in (None, 0, "", "none"):
            self.assign.pop(account_id, None)
            return {"ok": True, "account": account_id, "lane": None}
        try:
            ln = int(lane)
        except (TypeError, ValueError):
            return {"ok": False, "error": f"bad lane {lane!r}"}
        if not 1 <= ln <= MAX_LANES:
            return {"ok": False, "error": f"lane {ln} out of range 1..{MAX_LANES}"}
        self.assign[account_id] = ln
        return {"ok": True, "account": account_id, "lane": ln}

    def set_config(self, enabled: bool | None = None, n_lanes: int | None = None) -> dict:
        if enabled is not None:
            self.enabled = bool(enabled)
        if n_lanes is not None:
            self.n_lanes = self._clamp(n_lanes)
        return {"ok": True, "enabled": self.enabled, "n_lanes": self.n_lanes}

    def auto_assign(self, account_ids: list[str]) -> dict:
        """Round-robin the given accounts across lanes 1..n_lanes, in id order, PRESERVING any
        account that already has a lane. Fills the emptiest lanes first so a partial layout tops up
        evenly instead of restarting from lane 1."""
        counts = {ln: 0 for ln in range(1, self.n_lanes + 1)}
        for aid in account_ids:
            ln = self.lane_of(aid)
            if ln is not None:
                counts[ln] += 1
        placed = 0
        for aid in sorted(account_ids):
            if self.lane_of(aid) is not None:
                continue
            ln = min(counts, key=lambda k: (counts[k], k))
            self.assign[aid] = ln
            counts[ln] += 1
            placed += 1
        return {"ok": True, "assigned": placed, "n_lanes": self.n_lanes}

    def clear(self) -> dict:
        n = len(self.assign)
        self.assign.clear()
        return {"ok": True, "cleared": n}

    # -- rotation ------------------------------------------------------------------------------
    def pick(self, candidate_ids) -> int | None:
        """The lane whose turn it is: the least-recently-served lane that actually has an eligible
        account in it. Lanes with nobody eligible are SKIPPED rather than burning their turn, so a
        lane whose accounts are all done/blown doesn't stall the rotation."""
        lanes = {self.lane_of(a) for a in candidate_ids}
        lanes.discard(None)
        if not lanes:
            return None
        return min(lanes, key=lambda ln: (self._seq.get(ln, 0), ln))

    def mark_served(self, lane: int | None) -> None:
        if lane is None:
            return
        self._tick += 1
        self._seq[lane] = self._tick

    def reset_rotation(self) -> None:
        self._seq.clear()
        self._tick = 0

    def members(self, account_ids) -> dict[int, list[str]]:
        """lane -> [account ids], for every lane 1..n_lanes (empty lanes included, for the UI)."""
        out: dict[int, list[str]] = {ln: [] for ln in range(1, self.n_lanes + 1)}
        for aid in account_ids:
            ln = self.lane_of(aid)
            if ln is not None:
                out[ln].append(aid)
        for v in out.values():
            v.sort()
        return out

    # -- state ---------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"enabled": self.enabled, "n_lanes": self.n_lanes, "assign": self.assign,
                "seq": {str(k): v for k, v in self._seq.items()}, "tick": self._tick}

    def from_dict(self, d: dict) -> None:
        self.enabled = d.get("enabled", False)
        self.n_lanes = self._clamp(d.get("n_lanes", DEFAULT_LANES))
        self.assign = {k: int(v) for k, v in d.get("assign", {}).items()}
        self._seq = {int(k): v for k, v in d.get("seq", {}).items()}
        self._tick = d.get("tick", 0)

    def save_state(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_state(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path) as f:
            self.from_dict(json.load(f))


# ---- the farm keeps TWO independent books: evals and fundeds ----------------------------------
# They are deliberately NOT one shared rotation. A funded account inherits nothing from the eval
# rotation because the two populations are different sizes: with 10 eval lanes and only 3 funded
# accounts, a shared turn order would let a funded account gamble roughly once every 4 trading days.
# Separate books let you run (say) 10 eval lanes and 3 funded lanes at their own natural cadence,
# each with its own on/off switch and lane count.
def save_books(path: str, eval_book: LaneBook, funded_book: LaneBook) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"eval": eval_book.to_dict(), "funded": funded_book.to_dict()}, f, indent=2)


def load_books(path: str, eval_book: LaneBook, funded_book: LaneBook) -> None:
    """Tolerates the single-book file format written before the eval/funded split (reads it as the
    eval book, leaving fundeds unassigned)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        d = json.load(f)
    if "eval" in d or "funded" in d:
        eval_book.from_dict(d.get("eval", {}))
        funded_book.from_dict(d.get("funded", {}))
    else:
        eval_book.from_dict(d)
