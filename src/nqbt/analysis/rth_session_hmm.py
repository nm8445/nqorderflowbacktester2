"""
2-state GaussianHMM for RTH session-level directional classification.

Architecture
------------
Unlike the overnight HMM which passes per-bar feature matrices (n_bars, 8)
with a lengths array to Baum-Welch, this model receives one feature vector
per session.  Training input is shape (n_sessions, 6) with no lengths array —
each row is a complete session observation, not a bar within a sequence.

This eliminates the per-session z-scoring problem: with only 90 one-minute
bars in an RTH session, slow features (obi_avg, vwap_deviation) have near-zero
within-session variance, collapsing to zero under StandardScaler.fit_transform().
At session-level each observation is already a scalar summary, so a
cross-session StandardScaler fitted on training sessions normalizes cleanly.

State alignment
---------------
Uses labeled examples instead of a formula.  10 dates confirmed visually as
clean trending RTH sessions are used to vote for the directional state:
whichever state the majority of examples land in gets labeled directional.

Confidence
----------
predict() uses model.predict_proba() on a single shape-(6,) vector.
Since there is no bar sequence to decode, the Viterbi path has length 1 —
the posterior probability for each state is the meaningful quantity.
Values below the threshold (default 0.70) return "ambiguous".
"""

from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM

from nqbt.analysis.rth_session_features import RTH_SESSION_FEATURE_NAMES

_DIRECTIONAL_EXAMPLES = [
    "2025-12-08",
    "2025-12-19",
    "2025-12-30",
    "2025-12-31",
    "2026-01-02",
    "2026-01-07",
    "2026-01-27",
    "2026-01-29",
    "2026-02-03",
    "2026-02-18",
]


class RTHSessionHMM:
    """
    2-state GaussianHMM for RTH session-level directional classification.

    Input
    -----
    Each session is represented by one shape-(6,) feature vector produced by
    compute_rth_session_features().  Normalization is cross-session: fit a
    StandardScaler on all training session vectors stacked, then apply to
    train and test before passing to this class.

    fit() receives a list of already-scaled shape-(6,) vectors.
    predict() receives a single already-scaled shape-(6,) vector.
    """

    STATES = ["directional", "non_directional"]

    def __init__(self, n_iter: int = 200, random_state: int = 42):
        self.model = GaussianHMM(
            n_components=2,
            covariance_type="diag",   # full needs ~hundreds of obs/state; diag correct at 180 sessions
            n_iter=n_iter,
            random_state=random_state,
        )
        self.state_map: dict[int, str] = {}
        self._fitted = False

    def fit(self, session_vectors: list[np.ndarray]) -> "RTHSessionHMM":
        """
        Train on pre-scaled session feature vectors.

        Parameters
        ----------
        session_vectors : list of np.ndarray, shape (8,) each
            One vector per RTH session, already cross-session scaled.
            No lengths array — each vector is one complete observation.
        """
        X = np.vstack([v.reshape(1, -1) for v in session_vectors])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.model.fit(X)   # no lengths argument
        self._fitted = True
        return self

    def align_states(
        self,
        example_vectors: dict[str, np.ndarray],
    ) -> None:
        """
        Map integer states to directional / non_directional using labeled examples.

        Parameters
        ----------
        example_vectors : dict mapping date string -> shape-(6,) scaled vector
            Should include the 10 directional example dates where available.
            Whichever state captures the majority of examples is labeled
            directional.

        Prints per-example state assignments and vote counts for verification.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before align_states()")

        votes = {0: 0, 1: 0}
        print("  align_states() — per-example state assignments:")
        print(f"  {'Date':<14}  {'State':>5}  {'prob_s0':>8}  {'prob_s1':>8}")
        print(f"  {'-'*40}")

        for date_str in _DIRECTIONAL_EXAMPLES:
            if date_str not in example_vectors:
                print(f"  {date_str:<14}  {'MISSING':>5}")
                continue
            vec   = example_vectors[date_str]
            clean = np.nan_to_num(vec.reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)
            proba = self.model.predict_proba(clean)[0]
            state = int(np.argmax(proba))
            votes[state] += 1
            print(
                f"  {date_str:<14}  {state:>5}  "
                f"{proba[0]:>8.4f}  {proba[1]:>8.4f}"
            )

        print(f"  {'-'*40}")
        print(f"  Votes: state 0 = {votes[0]},  state 1 = {votes[1]}")

        directional     = 0 if votes[0] >= votes[1] else 1
        non_directional = 1 - directional

        # ── Emission mean sanity check ─────────────────────────────────────────
        # directional state should have trend_ratio mean > 0 and
        # realized_vol mean < 0 (lower vol on clean trend days in scaled space).
        tr_idx  = RTH_SESSION_FEATURE_NAMES.index("trend_ratio")
        vol_idx = RTH_SESSION_FEATURE_NAMES.index("realized_vol")
        tr_mean  = float(self.model.means_[directional, tr_idx])
        vol_mean = float(self.model.means_[directional, vol_idx])
        check_ok = (tr_mean > 0) and (vol_mean < 0)
        if not check_ok:
            directional, non_directional = non_directional, directional
            print(
                f"  WARNING: vote result failed emission mean sanity check "
                f"(trend_ratio={tr_mean:+.4f}, realized_vol={vol_mean:+.4f}) "
                f"— labels flipped"
            )
        else:
            print(
                f"  Sanity check passed "
                f"(trend_ratio={tr_mean:+.4f} > 0, realized_vol={vol_mean:+.4f} < 0)"
            )

        print(
            f"  -> state {directional} = directional,  "
            f"state {non_directional} = non_directional"
        )
        self.state_map = {
            directional:     "directional",
            non_directional: "non_directional",
        }

    def predict(
        self,
        session_vector: np.ndarray,
    ) -> tuple[str, float]:
        """
        Classify a single session and return (label, confidence).

        confidence = posterior probability of the predicted state from
        model.predict_proba(), not a Viterbi bar fraction.

        Parameters
        ----------
        session_vector : np.ndarray, shape (8,)
            Already cross-session scaled.

        Returns
        -------
        (label, confidence) — label is "directional", "non_directional",
        or "unknown" if the model is not fitted.
        """
        if not self._fitted or not self.state_map:
            return "unknown", 0.0
        clean = np.nan_to_num(
            session_vector.reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0
        )
        proba  = self.model.predict_proba(clean)[0]   # shape (2,)
        state  = int(np.argmax(proba))
        conf   = float(proba[state])
        label  = self.state_map.get(state, "unknown")
        return label, conf

    def classify(
        self,
        session_vector: np.ndarray,
        confidence_threshold: float = 0.70,
    ) -> tuple[str, float]:
        """
        Classify a session; return "ambiguous" if confidence is below threshold.
        """
        label, conf = self.predict(session_vector)
        if label == "unknown":
            return "unknown", conf
        if conf < confidence_threshold:
            return "ambiguous", conf
        return label, conf
