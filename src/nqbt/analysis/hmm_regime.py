"""
Hidden Markov Model regime classifiers for NQ session analysis.

Two models are trained and saved:

  OvernightHMM (2 states: directional / non_directional)
    - 10 per-bar features only (no session-level excursion features)
    - State alignment: abs(delta_pct_mean) + trend_ratio_mean
                       + max(0, autocorr_r1_mean)
    - Highest-scoring state → directional, other → non_directional

  RegimeHMM (3 states: trending / ranging / chop) — RTH session
    - 13 features: 10 per-bar + 3 session-level excursion constants
    - State alignment: trending = highest abs(delta_pct_mean),
                       ranging  = lowest realized_vol of remainder,
                       chop     = remaining state

The overnight → RTH predictability study outputs:
  - Contingency table: P(RTH regime | overnight regime), row-normalized
  - Chi-squared test of independence (p < 0.05 = statistically significant)
  - Per-state match rates for each overnight regime
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy import stats

from nqbt.analysis.features import FEATURE_NAMES, N_BAR_FEATURES
from nqbt.analysis.overnight_features import OVERNIGHT_FEATURE_NAMES


class RegimeHMM:
    """GaussianHMM wrapper with regime state alignment."""

    REGIMES = ["trending", "ranging", "chop"]

    def __init__(self, n_components: int = 3, n_iter: int = 200, random_state: int = 42):
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
        )
        self.state_map: dict[int, str] = {}
        self._fitted = False

    def fit(self, sessions: list[np.ndarray], min_bars: int = 5) -> "RegimeHMM":
        """
        Train on a list of session feature matrices.

        Parameters
        ----------
        sessions : list of np.ndarray, shape (n_bars, n_features) each
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

    def align_states(self, feature_names: list[str] = FEATURE_NAMES) -> None:
        """
        Map HMM integer states to regime labels using emission means.
        Must be called after fit().
        """
        means      = self.model.means_
        delta_idx  = feature_names.index("delta_pct")
        rvol_idx   = feature_names.index("realized_vol")

        remaining = list(range(self.model.n_components))

        trending = max(remaining, key=lambda s: abs(means[s, delta_idx]))
        remaining.remove(trending)

        ranging = min(remaining, key=lambda s: means[s, rvol_idx])
        remaining.remove(ranging)

        chop = remaining[0]

        self.state_map = {trending: "trending", ranging: "ranging", chop: "chop"}

    def predict_session(self, X: np.ndarray) -> str:
        """
        Viterbi decode and return the mode regime label for the session.
        Returns 'unknown' if model not fitted or X is empty.
        """
        if not self._fitted or X.shape[0] == 0:
            return "unknown"
        X_clean   = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        state_seq = self.model.predict(X_clean)
        mode      = int(np.bincount(state_seq).argmax())
        return self.state_map.get(mode, "unknown")

    def predict_proba_session(self, X: np.ndarray) -> dict[str, float]:
        """Return fraction of bars spent in each regime."""
        if not self._fitted or X.shape[0] == 0:
            return {r: 0.0 for r in self.REGIMES}
        X_clean   = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        state_seq = self.model.predict(X_clean)
        n         = len(state_seq)
        return {
            name: float(np.sum(state_seq == state) / n)
            for state, name in self.state_map.items()
        }


class OvernightHMM:
    """
    2-state GaussianHMM for overnight directional classification.

    States: directional / non_directional
    Features: 10 per-bar features only (no session-level excursion columns).
    Alignment: state with higher
        abs(delta_pct_mean) + trend_ratio_mean + max(0, autocorr_r1_mean)
    is labeled directional.
    """

    REGIMES = ["directional", "non_directional"]

    def __init__(self, n_iter: int = 200, random_state: int = 42):
        self.model = GaussianHMM(
            n_components=2,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
        )
        self.state_map: dict[int, str] = {}
        self._fitted = False

    def fit(self, sessions: list[np.ndarray], min_bars: int = 5) -> "OvernightHMM":
        valid   = [X for X in sessions if X.shape[0] >= min_bars]
        lengths = [X.shape[0] for X in valid]
        X_all   = np.vstack(valid)
        X_all   = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
        self.model.fit(X_all, lengths)
        self._fitted = True
        return self

    def align_states(self, label: str = "") -> None:
        """
        Map integer states to directional / non_directional.

        Trained on 8-feature overnight data from compute_overnight_features()
        (OVERNIGHT_FEATURE_NAMES).  Uses that index set so autocorr_r1 is
        correctly found at index 7, not index 8 (which would be out of bounds
        for an 8-feature model).

        score = abs_delta_pct_mean + trend_ratio_mean + max(0, autocorr_r1_mean)
        Higher-scoring state -> directional.

        Prints per-component scores for both states for verification.
        """
        means        = self.model.means_
        delta_idx    = OVERNIGHT_FEATURE_NAMES.index("abs_delta_pct")   # 0
        trend_idx    = OVERNIGHT_FEATURE_NAMES.index("trend_ratio")     # 1
        autocorr_idx = OVERNIGHT_FEATURE_NAMES.index("autocorr_r1")    # 7

        tag = f"[{label}] " if label else ""
        print(f"  {tag}align_states scoring (n_features={means.shape[1]}):")
        scores = []
        for s in range(2):
            d_comp  = float(means[s, delta_idx])     # already abs in overnight features
            tr_comp = float(means[s, trend_idx])
            ac_comp = max(0.0, float(means[s, autocorr_idx]))
            score   = d_comp + tr_comp + ac_comp
            scores.append(score)
            print(
                f"    state {s}: abs_delta_pct={d_comp:+.4f}  "
                f"trend_ratio={tr_comp:+.4f}  "
                f"max(0,autocorr_r1)={ac_comp:+.4f}  "
                f"score={score:.4f}"
            )

        directional     = int(np.argmax(scores))
        non_directional = 1 - directional
        print(f"    -> state {directional} = directional, state {non_directional} = non_directional")
        self.state_map  = {
            directional:     "directional",
            non_directional: "non_directional",
        }

    def predict_session(self, X: np.ndarray) -> str:
        if not self._fitted or X.shape[0] == 0:
            return "unknown"
        X_clean   = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        state_seq = self.model.predict(X_clean)
        mode      = int(np.bincount(state_seq).argmax())
        return self.state_map.get(mode, "unknown")

    def predict_proba_session(self, X: np.ndarray) -> dict[str, float]:
        if not self._fitted or X.shape[0] == 0:
            return {r: 0.0 for r in self.REGIMES}
        X_clean   = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        state_seq = self.model.predict(X_clean)
        n         = len(state_seq)
        return {
            name: float(np.sum(state_seq == state) / n)
            for state, name in self.state_map.items()
        }


def run_overnight_rth_study(
    train_overnight: list[np.ndarray],
    train_rth:       list[np.ndarray],
    test_overnight:  list[np.ndarray],
    test_rth:        list[np.ndarray],
    test_dates:      list,
    min_bars:        int = 5,
) -> dict:
    """
    Train overnight and RTH HMMs, then evaluate predictive relationship on test set.

    Parameters
    ----------
    train_*/test_* : list of np.ndarray
        Per-session feature matrices, already z-score normalized per session.
    test_dates : list
        Trading dates corresponding to each test session pair.
    min_bars : int
        Minimum bars in a session for it to be included.

    Returns
    -------
    dict with keys:
        overnight_hmm, rth_hmm — trained RegimeHMM objects
        pairs                  — DataFrame of (date, overnight_regime, rth_regime)
        contingency            — row-normalized P(RTH | overnight) table
        chi2, p_value, dof     — chi-squared test results
        per_state_match        — dict of per-overnight-regime RTH distributions
        n_test_days            — number of valid test days used
    """
    print("Training overnight HMM...")
    overnight_hmm = OvernightHMM().fit(train_overnight, min_bars=min_bars)
    overnight_hmm.align_states(label="overnight")

    print("Training RTH HMM...")
    rth_hmm = RegimeHMM().fit(train_rth, min_bars=min_bars)
    rth_hmm.align_states()

    print("Predicting on test set...")
    records = []
    for i, date in enumerate(test_dates):
        X_on  = test_overnight[i]
        X_rth = test_rth[i]
        if X_on.shape[0] < min_bars or X_rth.shape[0] < min_bars:
            continue
        records.append({
            "date":      date,
            "overnight": overnight_hmm.predict_session(X_on),
            "rth":       rth_hmm.predict_session(X_rth),
        })

    pairs = pd.DataFrame(records)

    if pairs.empty or len(pairs) < 3:
        return {
            "overnight_hmm": overnight_hmm, "rth_hmm": rth_hmm,
            "pairs": pairs, "contingency": None,
            "chi2": None, "p_value": None, "dof": None,
            "per_state_match": {}, "n_test_days": 0,
        }

    raw_ct  = pd.crosstab(pairs["overnight"], pairs["rth"])
    norm_ct = pd.crosstab(pairs["overnight"], pairs["rth"], normalize="index")

    chi2, p_value, dof, _ = stats.chi2_contingency(raw_ct.values)

    per_state = {}
    for regime in OvernightHMM.REGIMES:
        subset = pairs[pairs["overnight"] == regime]
        if len(subset) > 0:
            per_state[regime] = subset["rth"].value_counts(normalize=True).to_dict()

    return {
        "overnight_hmm":   overnight_hmm,
        "rth_hmm":         rth_hmm,
        "pairs":           pairs,
        "contingency":     norm_ct,
        "raw_contingency": raw_ct,
        "chi2":            round(chi2, 4),
        "p_value":         round(p_value, 6),
        "dof":             dof,
        "per_state_match": per_state,
        "n_test_days":     len(records),
    }
