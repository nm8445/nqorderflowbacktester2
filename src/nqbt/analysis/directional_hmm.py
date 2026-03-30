"""
2-state GaussianHMM for overnight directional classification.

States
------
  DIRECTIONAL     — sustained one-way movement regardless of direction.
                    Price is outside or breaking VAH/VAL.
                    Continuation setups in the direction of the break are
                    favored over fading back into the value area.

  NON_DIRECTIONAL — balanced, ranging, or choppy overnight.
                    Price is at or near VAH/VAL.
                    Reversal setups back toward the POC and the opposite
                    value area boundary are favored over continuation breaks.

State alignment
---------------
After training, the state with the higher composite score:
    abs_delta_pct_mean + trend_ratio_mean + max(0, autocorr_r1_mean)
is mapped to DIRECTIONAL.  The other state is NON_DIRECTIONAL.
This avoids the 3-state model's error of using realized_vol as the primary
discriminator (which misclassified high-volatility macro selloff days as chop).

Confidence
----------
classify() returns both a label and a confidence score (fraction of bars
in the dominant Viterbi state).  Values below the threshold (default 0.70)
return "ambiguous" — the session is treated as mixed and signal thresholds
are raised for both continuation and reversal setups.

Live trading integration
------------------------
Called each morning before the RTH open:
    label, conf = hmm.classify(overnight_features)
    if label == "directional" and conf >= 0.70:
        # enable continuation setups when price breaks outside VAH/VAL
    elif label == "non_directional" and conf >= 0.70:
        # enable reversal setups when price is at VAH/VAL
    else:
        # ambiguous — raise signal threshold, reduce size or skip
"""

from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM

from nqbt.analysis.overnight_features import OVERNIGHT_FEATURE_NAMES


class DirectionalHMM:
    """2-state GaussianHMM wrapper for overnight directional classification."""

    STATES = ["directional", "non_directional"]

    def __init__(self, n_iter: int = 200, random_state: int = 42):
        self.model = GaussianHMM(
            n_components=2,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
        )
        self.state_map: dict[int, str] = {}
        self._fitted = False

    def fit(self, sessions: list[np.ndarray], min_bars: int = 5) -> "DirectionalHMM":
        """
        Train on a list of overnight session feature matrices.

        Parameters
        ----------
        sessions : list of np.ndarray, shape (n_bars, 8) each
        min_bars : int
            Sessions with fewer bars are excluded from training.
        """
        valid   = [X for X in sessions if X.shape[0] >= min_bars]
        lengths = [X.shape[0] for X in valid]
        X_all   = np.vstack(valid)
        X_all   = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
        self.model.fit(X_all, lengths)
        self._fitted = True
        return self

    def align_states(self) -> None:
        """
        Map integer states to DIRECTIONAL / NON_DIRECTIONAL using emission means.

        Composite score = abs_delta_pct_mean + trend_ratio_mean
                        + max(0, autocorr_r1_mean)

        The state with the higher score is DIRECTIONAL.
        """
        means      = self.model.means_
        delta_idx  = OVERNIGHT_FEATURE_NAMES.index("abs_delta_pct")
        trend_idx  = OVERNIGHT_FEATURE_NAMES.index("trend_ratio")
        autocorr_idx = OVERNIGHT_FEATURE_NAMES.index("autocorr_r1")

        scores = [
            means[s, delta_idx] + means[s, trend_idx] + max(0.0, float(means[s, autocorr_idx]))
            for s in range(2)
        ]
        directional     = int(np.argmax(scores))
        non_directional = 1 - directional
        self.state_map  = {
            directional:     "directional",
            non_directional: "non_directional",
        }

    def predict_session(self, X: np.ndarray) -> str:
        """Viterbi decode; return mode state label for the session."""
        if not self._fitted or X.shape[0] == 0:
            return "unknown"
        X_clean   = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        state_seq = self.model.predict(X_clean)
        mode      = int(np.bincount(state_seq).argmax())
        return self.state_map.get(mode, "unknown")

    def predict_proba_session(self, X: np.ndarray) -> dict[str, float]:
        """Return fraction of bars spent in each state."""
        if not self._fitted or X.shape[0] == 0:
            return {s: 0.0 for s in self.STATES}
        X_clean   = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        state_seq = self.model.predict(X_clean)
        n         = len(state_seq)
        return {
            name: float(np.sum(state_seq == state) / n)
            for state, name in self.state_map.items()
        }

    def classify(
        self,
        X: np.ndarray,
        confidence_threshold: float = 0.70,
    ) -> tuple[str, float]:
        """
        Classify a session and return (label, confidence).

        confidence = fraction of Viterbi bars in the dominant state.
        If confidence < threshold, returns ("ambiguous", confidence).
        """
        proba      = self.predict_proba_session(X)
        label      = max(proba, key=lambda k: proba[k])
        confidence = proba[label]
        if confidence < confidence_threshold:
            return "ambiguous", confidence
        return label, confidence
