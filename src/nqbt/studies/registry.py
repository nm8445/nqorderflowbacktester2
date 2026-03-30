"""
Study registry — all 21 research studies organised by block.

Import BLOCKS (ordered dict: block_label → list[BaseStudy instance]) for the UI.
Import ALL_STUDIES (flat list) for lookup by id.

To implement a study, add a run() method to its class below.
The stub in BaseStudy.run() is used until then.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseStudy, Param, StudyResult

# ── Data paths ────────────────────────────────────────────────────────────────

_SIGNALS_CSV = Path("output/absorption_signals_labeled.csv")
_CLUSTER_CSV = Path("output/diagnostics/signals_with_cluster_tags.csv")
_ET          = "America/New_York"

# ── Shared data helpers ───────────────────────────────────────────────────────

def _load_signals(start: str, end: str, cluster_tags: bool = False) -> pd.DataFrame:
    path = _CLUSTER_CSV if (cluster_tags and _CLUSTER_CSV.exists()) else _SIGNALS_CSV
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["signal_time"])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    if "sustained" in df.columns:
        df["sustained"] = df["sustained"].astype(bool)
    s = pd.to_datetime(start).date()
    e = pd.to_datetime(end).date()
    if "date" in df.columns:
        df = df[(df["date"] >= s) & (df["date"] <= e)]
    return df.reset_index(drop=True)


def _small_sample(n: int) -> str:
    return (f"\n⚠ SMALL SAMPLE WARNING: only {n} signals — treat with caution.\n"
            if n < 20 else "")


def _dir_bias_note(b: str) -> str:
    if b in ("bullish", "bearish"):
        return (f"Note: dir_bias_filter='{b}' requested but directional bias is not yet "
                f"computable from this dataset. Results shown unfiltered.\n")
    return ""


def _abs_filter(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    if "percentile_rank" not in df.columns or df.empty or thr <= 0:
        return df
    return df[df["percentile_rank"] >= thr].copy()


def _imbal_filter(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    if "imbalance_ratio" not in df.columns or df.empty or thr <= 0:
        return df
    return df[df["imbalance_ratio"].abs() >= thr].copy()


def _stats_row(sub: pd.DataFrame, label: str) -> dict:
    n    = len(sub)
    sust = int(sub["sustained"].sum()) if ("sustained" in sub.columns and n > 0) else 0
    az   = sub["move_z_score"].mean()   if ("move_z_score"   in sub.columns and n > 0) else float("nan")
    am   = sub["mae_normalized"].mean() if ("mae_normalized"  in sub.columns and n > 0) else float("nan")
    ai   = sub["imbalance_ratio"].mean() if ("imbalance_ratio" in sub.columns and n > 0) else float("nan")
    return {
        "group":          label,
        "n":              n,
        "sustained":      sust,
        "sust_pct":       round(sust / n * 100, 1) if n > 0 else 0.0,
        "avg_move_z":     round(az, 3) if not (isinstance(az, float) and np.isnan(az)) else float("nan"),
        "avg_mae":        round(am, 4) if not (isinstance(am, float) and np.isnan(am)) else float("nan"),
        "avg_imbalance":  round(ai, 3) if not (isinstance(ai, float) and np.isnan(ai)) else float("nan"),
    }


def _regime_table(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    rows = []
    rows.append(_stats_row(df, f"{prefix}ALL"))
    if "overnight_regime" in df.columns:
        for regime in ["directional", "non_directional", "ambiguous"]:
            sub = df[df["overnight_regime"] == regime]
            if not sub.empty:
                rows.append(_stats_row(sub, f"{prefix}{regime}"))
    return pd.DataFrame(rows)

# ── Shared parameter presets ──────────────────────────────────────────────────

_DIR_BIAS = Param(
    "dir_bias_filter", "Directional Bias Filter", "choice", "any",
    "Only include sessions where the computed directional bias matches this.",
    choices=["any", "bullish", "bearish"],
)
_REGIME = Param(
    "regime_filter", "Regime Filter", "choice", "any",
    "Only process sessions where the RTH HMM regime matches.",
    choices=["any", "trending", "ranging", "chop"],
)
_MIN_DELTA_PCT = Param(
    "min_delta_pct", "Min |delta_pct| at Level", "float", 0.10,
    "Minimum absolute delta_pct at the structural level to qualify.",
    mn=0.0, mx=1.0, step=0.01,
)
_PROX_TICKS = Param(
    "proximity_ticks", "Proximity Threshold (ticks)", "int", 4,
    "How close price must be to the level (in 0.25-pt ticks) to count.",
    mn=1, mx=20,
)
_MIN_MOVE = Param(
    "min_move_ticks", "Min Reaction Move (ticks)", "int", 8,
    "Minimum move size (ticks) to qualify as a meaningful reaction.",
    mn=1, mx=80,
)
_ABS_SCORE = Param(
    "abs_score_min", "Min Absorption Score (× avg)", "float", 2.0,
    "Minimum absorption score multiple of rolling average.",
    mn=1.0, mx=10.0, step=0.25,
)
_TOTAL_VOL_PCT = Param(
    "total_vol_percentile", "Min Total Vol Percentile (rolling 20-session)", "int", 60,
    "Keep only signals whose total_vol ranks at or above this percentile within a rolling "
    "20-session window ending on the signal date. Adapts to changing volume regimes rather "
    "than using a fixed contract threshold that would overfit to training-period volume levels.",
    mn=0, mx=99,
)


# ── Block 1 — Value Area Behavior ─────────────────────────────────────────────

class Study11(BaseStudy):
    id = "1.1"
    name = "VAH/VAL Buying and Selling Effort Reaction Rate"
    block = "Block 1 — Value Area Behavior"
    description = (
        "Given a statistically correct directional bias, measure the percentage "
        "of occurrences where buying effort on a retest of the VAH or selling "
        "effort on a retest of the VAL produces a meaningful reaction.  Measures "
        "reaction frequency and average move magnitude."
    )
    variables = [
        _DIR_BIAS,
        _MIN_DELTA_PCT,
        _PROX_TICKS,
        _MIN_MOVE,
    ]

    def run(self, start_date: str, end_date: str, **p) -> StudyResult:
        return super().run(start_date, end_date, **p)


class Study12(BaseStudy):
    id = "1.2"
    name = "Break-and-Retest Delta Threshold"
    block = "Block 1 — Value Area Behavior"
    description = (
        "Identify all break-and-retest events of VAH and VAL.  Calculate the "
        "average delta required during the retest to produce a sustained price "
        "reaction."
    )
    variables = [
        Param("break_ticks", "Break Definition (min ticks beyond level)", "int", 4,
              "Minimum ticks price must close beyond the level to count as a break.",
              mn=1, mx=20),
        Param("retest_ticks", "Retest Definition (return within N ticks)", "int", 3,
              "Return within this many ticks of the broken level counts as a retest.",
              mn=1, mx=15),
        _MIN_DELTA_PCT,
        Param("sustained_ticks", "Sustained Move Definition (ticks)", "int", 12,
              "Minimum continuation ticks after retest to classify as sustained.",
              mn=4, mx=60),
    ]

    def run(self, start_date: str, end_date: str, **p) -> StudyResult:
        return super().run(start_date, end_date, **p)


class Study13(BaseStudy):
    id = "1.3"
    name = "Post-Break VAH/VAL Continuation vs Failure"
    block = "Block 1 — Value Area Behavior"
    description = (
        "After a clean break of VAH or VAL, measure whether the break continues "
        "or fails and reverts back into the value area.  Identify what regime "
        "conditions, delta quality, and absorption scores at the break point "
        "predict continuation versus failure."
    )
    variables = [
        Param("break_quality_min", "Min Break Quality Score", "float", 0.5,
              "Minimum delta_pct * trend_ratio composite score at breakout bar.",
              mn=0.0, mx=1.0, step=0.05),
        _ABS_SCORE,
        _MIN_DELTA_PCT,
        Param("continuation_ticks", "Min Continuation (ticks)", "int", 16,
              "Minimum ticks beyond break to classify as continuation.",
              mn=4, mx=80),
        Param("failure_ticks", "Max Revert Before Failure (ticks)", "int", 8,
              "Reversion beyond this many ticks back inside value area = failure.",
              mn=2, mx=30),
    ]

    def run(self, start_date: str, end_date: str, **p) -> StudyResult:
        return super().run(start_date, end_date, **p)


class Study14(BaseStudy):
    id = "1.4"
    name = "POC Reach Probability from VAH/VAL Reversal"
    block = "Block 1 — Value Area Behavior"
    description = (
        "Given a statistically valid directional bias and a reversal starting "
        "from VAH or VAL, determine the probability that price reaches the POC "
        "before returning to the reversal origin.  Measures average distance "
        "traveled and time taken."
    )
    variables = [
        _DIR_BIAS,
        Param("reversal_confirm_ticks", "Reversal Confirmation (ticks)", "int", 6,
              "Minimum ticks away from origin to confirm a reversal has started.",
              mn=2, mx=30),
        _MIN_DELTA_PCT,
    ]

    def run(self, start_date: str, end_date: str, **p) -> StudyResult:
        return super().run(start_date, end_date, **p)


# ── Block 2 — VWAP and Standard Deviation Band ────────────────────────────────

class Study21(BaseStudy):
    id = "2.1"
    name = "VWAP Band Reaction Rates: Solo vs Confluence"
    block = "Block 2 — VWAP and Standard Deviation Bands"
    description = (
        "Given a statistically valid directional bias, analyze how frequently "
        "price reacts to VWAP and its standard deviation bands.  Measure whether "
        "reactions show stronger statistical significance when coinciding with "
        "VAH, VAL, or POC.  Quantify delta and average follow-through distance."
    )
    variables = [
        Param("band_level", "Band Level", "choice", "1σ",
              "Which VWAP band to study.", choices=["VWAP", "1σ", "2σ", "3σ"]),
        Param("confluence_flag", "Confluence Mode", "choice", "both",
              "Study solo reactions, confluent only, or both.",
              choices=["both", "solo", "confluent"]),
        _DIR_BIAS,
        _MIN_DELTA_PCT,
        _MIN_MOVE,
        _PROX_TICKS,
    ]


class Study22(BaseStudy):
    id = "2.2"
    name = "Extreme VWAP Extension Reversal Rates by Band"
    block = "Block 2 — VWAP and Standard Deviation Bands"
    description = (
        "At 2σ, 3σ, and 4σ VWAP deviations, measure the base rate of reversal "
        "regardless of regime.  Then measure whether regime changes that "
        "probability meaningfully or whether extreme extensions reverse at a "
        "consistent rate regardless of market conditions."
    )
    variables = [
        Param("min_band", "Minimum Deviation Band", "choice", "2σ",
              "Starting band for the study.", choices=["2σ", "3σ", "4σ"]),
        _REGIME,
        _DIR_BIAS,
        Param("inside_value_area", "Value Area Position", "choice", "both",
              "Filter by whether price is inside or outside the value area.",
              choices=["both", "inside", "outside"]),
        _MIN_MOVE,
    ]


class Study23(BaseStudy):
    id = "2.3"
    name = "VWAP Band + VAH/VAL Confluence Zone Behavior"
    block = "Block 2 — VWAP and Standard Deviation Bands"
    description = (
        "Price is near or outside VAH/VAL and simultaneously near a VWAP std "
        "dev band.  Measure P(large reversal) and P(break continuation) given "
        "current regime.  Find the minimum confluence proximity that produces "
        "statistically meaningful results."
    )
    variables = [
        _PROX_TICKS,
        Param("vwap_prox_ticks", "VWAP Band Proximity (ticks)", "int", 4,
              "Price must be within this many ticks of a VWAP band.",
              mn=1, mx=20),
        _REGIME,
        _ABS_SCORE,
        _MIN_DELTA_PCT,
        Param("delta_flip", "Require Delta Flip Confirmation", "bool", True,
              "Require delta_pct sign flip at the confluence zone."),
    ]


# ── Block 3 — Absorption and Aggression ───────────────────────────────────────

class Study31(BaseStudy):
    id = "3.1"
    name = "Absorption Followed by Immediate Buying Aggression"
    block = "Block 3 — Absorption and Aggression"
    description = (
        "Quantify the probability that absorption followed by immediate buying "
        "aggression leads to a directional move.  Measure typical move size and "
        "frequency."
    )
    variables = [
        _TOTAL_VOL_PCT,
        Param("aggression_window_s", "Aggression Confirmation Window (seconds)", "int", 30,
              "Seconds after absorption bar close to detect delta spike.",
              mn=5, mx=300, step=5),
        Param("min_delta_spike", "Min Delta Spike after Absorption", "float", 0.25,
              "Minimum delta_pct value on the first bar after absorption.",
              mn=0.05, mx=1.0, step=0.05),
        _PROX_TICKS,
    ]


class Study32(BaseStudy):
    id = "3.2"
    name = "Absorption Size Optimization for Signal Quality"
    block = "Block 3 — Absorption and Aggression"
    description = (
        "Run an optimization on total_vol percentile thresholds.  Determine whether "
        "raising the volume percentile cutoff reduces trade count but improves win rate and "
        "expectancy enough to justify the reduction in frequency.  Percentile is computed "
        "within a rolling 20-session window to remain adaptive across volume regimes."
    )
    variables = [
        Param("vol_pct_sweep_lo", "Volume Percentile Sweep — Low", "int", 20,
              "Lower bound of total_vol percentile sweep (rolling 20-session window).",
              mn=0, mx=80),
        Param("vol_pct_sweep_hi", "Volume Percentile Sweep — High", "int", 90,
              "Upper bound of total_vol percentile sweep.",
              mn=10, mx=99),
        Param("vol_pct_sweep_step", "Volume Percentile Sweep — Step", "int", 10,
              "Step size for the sweep.", mn=1, mx=20),
        Param("signal_type", "Signal Type", "choice", "all",
              "Which signal type to optimize for.",
              choices=["trend_retest", "structure_break", "range_reversal", "all"]),
    ]


class Study33(BaseStudy):
    id = "3.3"
    name = "Optimal Delta Threshold for Entry Signals"
    block = "Block 3 — Absorption and Aggression"
    description = (
        "Run an optimization on the delta required on entry signals.  Measure "
        "whether higher delta minimums improve win rate and expectancy at the "
        "cost of trade frequency."
    )
    variables = [
        Param("min_delta_sweep_lo", "delta_pct Sweep — Low", "float", 0.05,
              "Lower bound of imbalance_ratio sweep.", mn=0.0, mx=0.5, step=0.01),
        Param("min_delta_sweep_hi", "delta_pct Sweep — High", "float", 0.60,
              "Upper bound of imbalance_ratio sweep.", mn=0.1, mx=1.0, step=0.01),
        Param("delta_step", "delta_pct Sweep — Step", "float", 0.05,
              "Step size for the sweep.", mn=0.01, mx=0.2, step=0.01),
        Param("signal_type", "Signal Type", "choice", "all",
              "Which signal type to optimize for.",
              choices=["all", "trend_retest", "structure_break", "range_reversal"]),
        _TOTAL_VOL_PCT,
        Param("time_of_day", "Time-of-Day Filter", "choice", "all",
              "Restrict to a specific time window.",
              choices=["all", "09:30–11:00", "11:00–14:00", "14:00–16:00"]),
    ]


# ── Block 4 — Historical Delta Node ───────────────────────────────────────────

_NODE_MAG = Param(
    "node_delta_min", "Min Node Delta Magnitude (contracts)", "int", 500,
    "Minimum absolute delta at the historical node to qualify.",
    mn=50, mx=5000, step=50,
)
_NODE_AGE = Param(
    "max_node_age_hours", "Max Node Age (hours)", "int", 48,
    "Only consider nodes formed within this many hours.",
    mn=1, mx=120,
)
_DECAY_HL = Param(
    "decay_halflife_hours", "Decay Halflife (hours)", "float", 12.0,
    "Halflife for exponential node relevance decay.",
    mn=1.0, mx=72.0, step=1.0,
)


class Study41(BaseStudy):
    id = "4.1"
    name = "Historical Sell-Delta Node Reversal Rate"
    block = "Block 4 — Historical Delta Nodes"
    description = (
        "When price is in a bullish regime and approaches a level where large "
        "sell delta occurred in the recent past, measure the probability of a "
        "reversal from that level.  Measure how node age, magnitude, and current "
        "regime interact."
    )
    variables = [
        _NODE_MAG,
        _NODE_AGE,
        Param("at_structural_level", "Require Structural Level Coincidence", "bool", False,
              "Only count nodes that sit at a VAH/VAL/POC or prev day high/low."),
        _REGIME,
        _MIN_DELTA_PCT,
        _DECAY_HL,
    ]


class Study42(BaseStudy):
    id = "4.2"
    name = "Historical Buy-Delta Node Reversal Rate"
    block = "Block 4 — Historical Delta Nodes"
    description = (
        "When price is in a bearish regime and approaches a level where large "
        "buy delta occurred in the recent past, measure the probability of a "
        "reversal.  Mirror of Study 4.1 for the sell side."
    )
    variables = [
        _NODE_MAG,
        _NODE_AGE,
        Param("at_structural_level", "Require Structural Level Coincidence", "bool", False,
              "Only count nodes that sit at a VAH/VAL/POC or prev day high/low."),
        _REGIME,
        _MIN_DELTA_PCT,
        _DECAY_HL,
    ]


class Study43(BaseStudy):
    id = "4.3"
    name = "Delta Node Age Decay — Optimal Halflife"
    block = "Block 4 — Historical Delta Nodes"
    description = (
        "Measure what halflife parameter best predicts node relevance.  Compare "
        "multiple decay halflives against observed reversal rates at returned-to "
        "nodes.  Determine whether structural-level nodes decay more slowly than "
        "intraday nodes."
    )
    variables = [
        Param("halflives", "Halflife Values to Compare (hours, comma-separated)", "str",
              "6,12,24,48,72",
              "Comma-separated list of halflives to sweep, e.g. 6,12,24,48,72."),
        Param("node_type", "Node Type", "choice", "both",
              "Filter to intraday nodes, structural-level nodes, or both.",
              choices=["both", "intraday", "structural"]),
        Param("return_distance_ticks", "Return-to-Node Distance (ticks)", "int", 4,
              "Price must return within this many ticks of the node level.",
              mn=1, mx=20),
    ]


# ── Block 5 — Regime Classification ───────────────────────────────────────────

class Study51(BaseStudy):
    id = "5.1"
    name = "Overnight Directional Score as RTH Setup-Type Predictor"
    block = "Block 5 — Regime Classification"
    description = (
        "Measure whether a directional overnight session (high trend ratio, high "
        "absolute delta) predicts that continuation setups outperform reversal "
        "setups in the 9:30–11:00 RTH window.  Measure the opposite for "
        "non-directional overnight sessions."
    )
    variables = [
        Param("on_trend_ratio_min", "Overnight Trend Ratio Threshold", "float", 0.4,
              "Minimum |trend_ratio| averaged across overnight bars.",
              mn=0.1, mx=0.9, step=0.05),
        Param("on_abs_delta_min", "Overnight |delta_pct| Threshold", "float", 0.2,
              "Minimum |delta_pct| averaged across overnight bars.",
              mn=0.05, mx=0.8, step=0.05),
        Param("on_decel_score", "Overnight Deceleration Score Threshold", "float", 0.3,
              "Deceleration score above this suggests trend slowed into open.",
              mn=0.0, mx=1.0, step=0.05),
        Param("setup_type", "Setup Type to Measure", "choice", "both",
              "Compare continuation setups, reversal setups, or both.",
              choices=["both", "continuation", "reversal"]),
    ]


class Study52(BaseStudy):
    id = "5.2"
    name = "Overnight Trend Deceleration as Failed Auction Predictor"
    block = "Block 5 — Regime Classification"
    description = (
        "Measure whether an overnight session that trended strongly but "
        "decelerated in the final 2 hours before the open predicts a failed "
        "auction at the RTH open.  A failed auction is defined as the overnight "
        "trend reversing aggressively within the first 30 minutes of RTH."
    )
    variables = [
        Param("on_trend_magnitude_min", "Overnight Trend Magnitude Min", "float", 0.35,
              "Minimum trend_ratio in the first half of the overnight session.",
              mn=0.1, mx=0.9, step=0.05),
        Param("decel_ratio_max", "Deceleration Ratio Max", "float", 0.5,
              "Second-half trend_ratio / first-half trend_ratio below this = decelerating.",
              mn=0.1, mx=0.9, step=0.05),
        Param("reversal_window_minutes", "Reversal Window (minutes from open)", "int", 30,
              "Minutes after RTH open to look for the reversal.",
              mn=10, mx=90, step=5),
        Param("reversal_magnitude_ticks", "Reversal Magnitude (ticks)", "int", 20,
              "Minimum ticks against the overnight trend direction within the window.",
              mn=4, mx=100, step=4),
    ]


class Study53(BaseStudy):
    id = "5.3"
    name = "RTH Real-Time Regime Bar-by-Bar State Detection"
    block = "Block 5 — Regime Classification"
    description = (
        "Using the rolling RTH HMM updated every 40-range bar, measure at what "
        "bar number after the open the regime classification stabilizes above a "
        "confidence threshold.  Measure setup win rate before vs after that point."
    )
    variables = [
        Param("confidence_threshold", "Regime Confidence Threshold", "float", 0.70,
              "Minimum HMM state probability to call the regime 'confirmed'.",
              mn=0.50, mx=0.95, step=0.05),
        _REGIME,
        Param("signal_type", "Signal Type", "choice", "all",
              "Which signal type to measure win rate for.",
              choices=["all", "trend_retest", "structure_break", "range_reversal"]),
    ]


class Study54(BaseStudy):
    id = "5.4"
    name = "Overnight HMM Training with Extended Observation Window"
    block = "Block 5 — Regime Classification"
    description = (
        "Train the overnight HMM on the 18:00–11:00 window instead of "
        "18:00–09:30 and evaluate whether the longer window more accurately "
        "classifies market regimes.  Compare classification accuracy against "
        "the shorter window using held-out test data."
    )
    variables = [
        Param("window_end", "Observation Window End", "choice", "11:00",
              "End time of the training observation window.",
              choices=["09:30", "10:00", "10:30", "11:00"]),
        Param("feature_set", "Feature Set", "choice", "full",
              "Which feature set to use for training.",
              choices=["full", "tier1_only"]),
        Param("post_window_signal_type", "Post-Window Signal Type", "choice", "all",
              "Signal type to measure win rate for after the window.",
              choices=["all", "trend_retest", "structure_break", "range_reversal"]),
    ]


class Study55(BaseStudy):
    id = "5.5"
    name = "Post-11:00 Favorable Entry Study"
    block = "Block 5 — Regime Classification"
    description = (
        "Evaluate whether statistically favorable entry opportunities exist after "
        "11:00 AM.  Measure hit rate, expectancy, and average move size for all "
        "signal types in the 11:00–14:00 window versus the 9:30–11:00 window."
    )
    variables = [
        Param("time_window", "Time Window", "choice", "11:00–14:00",
              "Which window to study.", choices=["09:30–11:00", "11:00–14:00", "14:00–16:00"]),
        Param("signal_type", "Signal Type", "choice", "all",
              "Which signal type to measure.", choices=["all", "trend_retest", "structure_break", "range_reversal"]),
        _REGIME,
        Param("profile_position", "Volume Profile Position", "choice", "any",
              "Restrict to signals occurring at specific profile levels.",
              choices=["any", "at_poc", "at_vah", "at_val", "outside_value_area"]),
    ]


# ── Block 6 — Session Structure ───────────────────────────────────────────────

class Study61(BaseStudy):
    id = "6.1"
    name = "Reaction Location Mapping Given Directional Bias"
    block = "Block 6 — Session Structure"
    description = (
        "Given a statistically valid directional bias, determine where meaningful "
        "upside (or downside) reactions most frequently occur relative to the "
        "volume profile.  Measure probability and average size of reactions at "
        "POC, VAH, VAL, prev day high/low, and VWAP bands."
    )
    variables = [
        _DIR_BIAS,
        Param("location_type", "Reaction Location", "choice", "all",
              "Which structural location to map.",
              choices=["all", "poc", "vah", "val", "prev_day_high", "prev_day_low",
                       "vwap_1s", "vwap_2s", "vwap_3s"]),
        _MIN_MOVE,
        _REGIME,
        Param("time_window", "Time Window", "choice", "09:30–11:00",
              "Restrict to this session window.",
              choices=["all_day", "09:30–11:00", "11:00–14:00", "14:00–16:00"]),
    ]


class Study62(BaseStudy):
    id = "6.2"
    name = "Intraday Session State Sequence"
    block = "Block 6 — Session Structure"
    description = (
        "Measure how frequently the intraday session follows specific structural "
        "sequences (e.g. open chop then continuation, immediate trend, range all "
        "session).  Map each sequence to its regime classification and measure "
        "which signal types work best in each sequence."
    )
    variables = [
        Param("sequence_type", "Sequence Type", "choice", "all",
              "Which session sequence to study.",
              choices=["all", "chop_then_trend", "immediate_trend", "range_all_day",
                       "trend_reverse"]),
        Param("chop_window_bars", "Initial Chop Window (bars)", "int", 5,
              "Number of opening bars to classify as the chop phase.",
              mn=2, mx=15),
        Param("signal_type", "Signal Type", "choice", "all",
              "Which signal type to measure win rate for.",
              choices=["all", "trend_retest", "structure_break", "range_reversal"]),
        _REGIME,
    ]


# ── Block 7 — Prop Firm Optimization ─────────────────────────────────────────
#
# Evaluation framework for all Block 7 studies
# ─────────────────────────────────────────────
# Every study produces results for both contexts side-by-side.
# The comparison metric is EV per dollar of personal capital at risk, not
# raw profit.  This is the only metric that makes strategies comparable
# across contexts.
#
# Prop firm context:
#   Personal capital at risk = total fees paid across all active accounts.
#   Monthly return = (monthly payout × n_accounts) / total_fees_paid
#   Example: $500/month × 20 accounts on $6,000 fees = 8.3% monthly
#
# Live account context:
#   Personal capital at risk = live account size.
#   Monthly return = monthly profit / account_size
#   Example: $500/month on $50,000 account = 1.0% monthly
#
# A strategy is not concluded "not worth trading" until the EV/capital
# comparison has been run for both contexts.
#
# Shared live account comparison parameters (added to every Block 7 study):
_LIVE_CAPITAL = Param(
    "live_account_size", "Live Account Size ($)", "int", 50000,
    "Capital at risk in the equivalent live account scenario.",
    mn=5000, mx=500000, step=5000,
)
_PROP_FEE = Param(
    "prop_fee_per_account", "Prop Firm Fee per Account ($)", "int", 300,
    "One-time evaluation fee per funded account attempt.",
    mn=50, mx=2000, step=50,
)
_N_ACCOUNTS = Param(
    "n_prop_accounts", "Number of Prop Accounts", "int", 5,
    "Number of simultaneously active funded accounts.",
    mn=1, mx=50,
)


class Study71(BaseStudy):
    id = "7.1"
    name = "Monte Carlo Risk Optimization per Funded Account Mode"
    block = "Block 7 — Prop Firm Optimization"
    description = (
        "Run Monte Carlo simulations on empirical trade outcome distributions to "
        "find optimal parameters for each funded account mode: danger zone mode "
        "(near trailing threshold), buffer building mode, and payout mode "
        "(securing 5 winning days).  "
        "Results are reported for both prop firm and live account contexts.  "
        "The comparison metric is EV per dollar of personal capital at risk — "
        "prop firm capital at risk is total fees paid, live account capital at "
        "risk is the account size."
    )
    variables = [
        Param("account_mode", "Account Mode", "choice", "buffer_building",
              "Which funded account mode to optimize for.",
              choices=["danger_zone", "buffer_building", "payout_securing"]),
        Param("contracts", "Contract Size", "int", 1,
              "Number of NQ contracts per trade.", mn=1, mx=10),
        Param("daily_stop_loss", "Daily Stop Loss ($)", "int", 500,
              "Maximum daily loss before stopping.", mn=100, mx=2000, step=50),
        Param("daily_profit_target", "Daily Profit Target ($)", "int", 300,
              "Daily profit target before stopping.", mn=50, mx=2000, step=50),
        Param("n_simulations", "Monte Carlo Simulations", "int", 10000,
              "Number of Monte Carlo runs.", mn=1000, mx=100000, step=1000),
        _ABS_SCORE,
        _MIN_DELTA_PCT,
        # Live account comparison
        _LIVE_CAPITAL,
        _PROP_FEE,
        _N_ACCOUNTS,
    ]


class Study72(BaseStudy):
    id = "7.2"
    name = "Prop Firm Challenge Pass Rate Optimization"
    block = "Block 7 — Prop Firm Optimization"
    description = (
        "Optimize parameters for passing the Lucid evaluation in minimum days at "
        "maximum pass percentage.  Sweeps position size, signal threshold, daily "
        "targets, and regime filter strictness.  "
        "Results are reported for both prop firm and live account contexts.  "
        "The comparison metric is EV per dollar of personal capital at risk — "
        "total fees across all accounts versus equivalent live account capital."
    )
    variables = [
        Param("contracts", "Contract Size", "int", 1,
              "Number of NQ contracts per trade.", mn=1, mx=5),
        Param("daily_profit_target", "Daily Profit Target ($)", "int", 300,
              "Minimum daily profit target.", mn=100, mx=1000, step=50),
        Param("max_trades_per_day", "Max Trades Per Day", "int", 3,
              "Stop taking trades after this many per day.", mn=1, mx=10),
        _REGIME,
        _ABS_SCORE,
        Param("n_simulations", "Monte Carlo Simulations", "int", 5000,
              "Number of Monte Carlo runs.", mn=500, mx=50000, step=500),
        # Live account comparison
        _LIVE_CAPITAL,
        _PROP_FEE,
        _N_ACCOUNTS,
    ]


class Study73(BaseStudy):
    id = "7.3"
    name = "Funded Phase Payout Probability Optimization"
    block = "Block 7 — Prop Firm Optimization"
    description = (
        "Given funded phase rules (5 profitable days per payout cycle, EOD "
        "trailing drawdown), find the parameter set that maximizes P(5 qualifying "
        "days) while minimizing P(drawdown breach).  "
        "Results are reported for both prop firm and live account contexts.  "
        "The comparison metric is EV per dollar of personal capital at risk — "
        "total fees across all accounts versus equivalent live account capital."
    )
    variables = [
        Param("daily_profit_target", "Daily Profit Target ($)", "int", 200,
              "Target profit per qualifying day (minimum $150).", mn=150, mx=1000, step=25),
        Param("stop_at_target", "Stop Trading at Daily Target", "bool", True,
              "Stop accepting new signals once daily target is reached."),
        _ABS_SCORE,
        _REGIME,
        Param("contracts", "Contract Size", "int", 1,
              "Number of NQ contracts per trade.", mn=1, mx=5),
        Param("n_simulations", "Monte Carlo Simulations", "int", 10000,
              "Number of Monte Carlo runs.", mn=1000, mx=100000, step=1000),
        # Live account comparison
        _LIVE_CAPITAL,
        _PROP_FEE,
        _N_ACCOUNTS,
    ]


class Study74(BaseStudy):
    id = "7.4"
    name = "Martingale Viability Study"
    block = "Block 7 — Prop Firm Optimization"
    description = (
        "Measure whether a martingale position sizing approach applied to the "
        "highest-confidence signal type produces positive expectancy within prop "
        "firm risk constraints.  Measure capital required, max consecutive losses "
        "before breach, and expected payout before drawdown breach.  "
        "Results are reported for both prop firm and live account contexts.  "
        "The comparison metric is EV per dollar of personal capital at risk — "
        "the martingale approach is only viable if its EV/capital exceeds flat "
        "sizing in both contexts before concluding it adds or destroys value."
    )
    variables = [
        Param("base_contracts", "Base Contract Size", "int", 1,
              "Starting number of contracts.", mn=1, mx=3),
        Param("multiplier", "Martingale Multiplier", "float", 2.0,
              "Position size multiplier after each loss.", mn=1.5, mx=4.0, step=0.25),
        Param("max_contracts_cap", "Max Contract Cap", "int", 4,
              "Hard cap on position size regardless of losses.", mn=2, mx=10),
        Param("reset_condition", "Reset Condition", "choice", "on_win",
              "When to reset position size back to base.",
              choices=["on_win", "on_profit_target", "on_new_day"]),
        Param("signal_confidence_min", "Min Signal Confidence (HMM prob)", "float", 0.70,
              "Minimum HMM state probability to take a martingale trade.",
              mn=0.50, mx=0.95, step=0.05),
        _REGIME,
        Param("n_simulations", "Monte Carlo Simulations", "int", 10000,
              "Number of Monte Carlo runs.", mn=1000, mx=100000, step=1000),
        # Live account comparison
        _LIVE_CAPITAL,
        _PROP_FEE,
        _N_ACCOUNTS,
    ]


# ── Registry ──────────────────────────────────────────────────────────────────

_ALL: list[BaseStudy] = [
    Study11(), Study12(), Study13(), Study14(),
    Study21(), Study22(), Study23(),
    Study31(), Study32(), Study33(),
    Study41(), Study42(), Study43(),
    Study51(), Study52(), Study53(), Study54(), Study55(),
    Study61(), Study62(),
    Study71(), Study72(), Study73(), Study74(),
]

# Ordered dict: block label → list of study instances
BLOCKS: dict[str, list[BaseStudy]] = {}
for _s in _ALL:
    BLOCKS.setdefault(_s.block, []).append(_s)

# Flat lookup by id
ALL_STUDIES: dict[str, BaseStudy] = {s.id: s for s in _ALL}
