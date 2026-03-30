"""
Run signal quality diagnostics on output/absorption_signals_labeled.csv.
Saves full output to output/diagnostics/signal_baseline.txt
Saves cluster-tagged signals to output/diagnostics/signals_with_cluster_tags.csv
"""

import sys
from pathlib import Path

# Ensure src/ is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from nqbt.analysis.signal_diagnostics import (
    absorption_quality_baseline,
    signal_clustering,
    time_of_day_buckets,
    prior_absorption_count,
    move_magnitude_baseline,
    node_qualification_baseline,
    vwap_confluence_analysis,
    sustained_tier_comparison,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CSV_PATH = PROJECT_ROOT / "output" / "absorption_signals_labeled.csv"
OUT_DIR = PROJECT_ROOT / "output" / "diagnostics"
TXT_OUT = OUT_DIR / "signal_baseline.txt"
CLUSTER_CSV_OUT = OUT_DIR / "signals_with_cluster_tags.csv"
NODE_REPORT_OUT = OUT_DIR / "node_qualification_report.txt"


def main():
    # ------------------------------------------------------------------
    # 1. Check CSV exists
    # ------------------------------------------------------------------
    if not CSV_PATH.exists():
        print(f"ERROR: Input CSV not found at {CSV_PATH}")
        print("       Run the absorption labeling pipeline first to generate this file.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Load CSV
    # ------------------------------------------------------------------
    print(f"Loading {CSV_PATH} ...")
    parse_dates = ["signal_time"]
    # Also parse 'date' as date if present
    df = pd.read_csv(CSV_PATH, parse_dates=parse_dates)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    # Coerce boolean
    if "sustained" in df.columns:
        df["sustained"] = df["sustained"].astype(bool)

    # Note on optional columns
    optional_cols = ["signal_price", "end_price"]
    missing_optional = [c for c in optional_cols if c not in df.columns]
    if missing_optional:
        print(f"NOTE: Optional columns not found in CSV (may be from older run): {missing_optional}")

    # ------------------------------------------------------------------
    # 3. Dataset summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    n_signals = len(df)
    print(f"Total signals: {n_signals}")

    if "date" in df.columns:
        dates = pd.to_datetime(pd.Series([str(d) for d in df["date"]]))
        print(f"Date range:    {df['date'].min()}  to  {df['date'].max()}")

    if "signal_time" in df.columns:
        st = pd.to_datetime(df["signal_time"], utc=True)
        print(f"Signal time:   {st.min()}  to  {st.max()}")

    if "sustained" in df.columns:
        sus_rate = df["sustained"].mean() * 100
        print(f"Sustained rate: {sus_rate:.1f}%  ({df['sustained'].sum()} of {n_signals})")

    if "absorption_side" in df.columns:
        print()
        print("absorption_side breakdown:")
        for val, cnt in df["absorption_side"].value_counts().items():
            print(f"  {val:<20} {cnt:>6}  ({cnt / n_signals * 100:.1f}%)")

    if "price_location" in df.columns:
        print()
        print("price_location breakdown:")
        for val, cnt in df["price_location"].value_counts().items():
            print(f"  {val:<20} {cnt:>6}  ({cnt / n_signals * 100:.1f}%)")

    if "overnight_regime" in df.columns:
        print()
        print("overnight_regime breakdown:")
        for val, cnt in df["overnight_regime"].value_counts().items():
            print(f"  {val:<20} {cnt:>6}  ({cnt / n_signals * 100:.1f}%)")

    print()

    # ------------------------------------------------------------------
    # 4. Create output directory
    # ------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_text_sections = []

    # ------------------------------------------------------------------
    # Diagnostic 1
    # ------------------------------------------------------------------
    print("Running diagnostic 1/8: Absorption Quality Baseline...")
    stats_df, text1 = absorption_quality_baseline(df)
    print(text1)
    all_text_sections.append(text1)

    # ------------------------------------------------------------------
    # Diagnostic 2
    # ------------------------------------------------------------------
    print("Running diagnostic 2/8: Signal Clustering...")
    df_clustered, text2 = signal_clustering(df)
    print(text2)
    all_text_sections.append(text2)

    # ------------------------------------------------------------------
    # Diagnostic 3
    # ------------------------------------------------------------------
    print("Running diagnostic 3/8: Time-of-Day Buckets...")
    agg_df, text3 = time_of_day_buckets(df)
    print(text3)
    all_text_sections.append(text3)

    # ------------------------------------------------------------------
    # Diagnostic 4
    # ------------------------------------------------------------------
    print("Running diagnostic 4/8: Prior Absorption Count...")
    df_prior, text4 = prior_absorption_count(df)
    print(text4)
    all_text_sections.append(text4)

    # ------------------------------------------------------------------
    # Diagnostic 5
    # ------------------------------------------------------------------
    print("Running diagnostic 5/8: Move Magnitude Baseline...")
    baseline_df, text5 = move_magnitude_baseline(df)
    print(text5)
    all_text_sections.append(text5)

    # ------------------------------------------------------------------
    # Diagnostic 6: Node Qualification
    # ------------------------------------------------------------------
    print("Running diagnostic 6/8: Node Qualification Baseline...")
    node_df, text6 = node_qualification_baseline(df)
    print(text6)
    all_text_sections.append(text6)

    NODE_REPORT_OUT.write_text(text6, encoding="utf-8")
    print(f"Saved node qualification report to {NODE_REPORT_OUT}")

    # ------------------------------------------------------------------
    # Diagnostic 7: VWAP Confluence
    # ------------------------------------------------------------------
    print("Running diagnostic 7/8: VWAP Confluence Analysis...")
    _, text7 = vwap_confluence_analysis(df)
    print(text7)
    all_text_sections.append(text7)

    # ------------------------------------------------------------------
    # Diagnostic 8: Sustained Tier Comparison
    # ------------------------------------------------------------------
    print("Running diagnostic 8/8: Sustained Tier Comparison...")
    _, text8 = sustained_tier_comparison(df)
    print(text8)
    all_text_sections.append(text8)

    # ------------------------------------------------------------------
    # 5. Save full text output
    # ------------------------------------------------------------------
    full_text = "\n\n".join(all_text_sections)
    TXT_OUT.write_text(full_text, encoding="utf-8")
    print(f"Saved diagnostic text to {TXT_OUT}")

    # ------------------------------------------------------------------
    # 6. Build cluster-tagged CSV with both cluster_position and prior_absorption_count
    # ------------------------------------------------------------------
    # df_clustered has cluster_position; df_prior has prior_absorption_count
    # Merge them on index (both derived from original df)
    cluster_col = df_clustered[["cluster_position"]] if "cluster_position" in df_clustered.columns else pd.DataFrame(index=df.index)
    prior_col = df_prior[["prior_absorption_count"]] if "prior_absorption_count" in df_prior.columns else pd.DataFrame(index=df.index)

    tagged_df = df.copy()
    if not cluster_col.empty:
        tagged_df = tagged_df.join(cluster_col, how="left")
    if not prior_col.empty:
        tagged_df = tagged_df.join(prior_col, how="left")

    tagged_df.to_csv(CLUSTER_CSV_OUT, index=False)
    print(f"Saved cluster-tagged signals to {CLUSTER_CSV_OUT}")

    # ------------------------------------------------------------------
    # 7. Final summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("All diagnostics complete.")
    print(f"  Text output:  {TXT_OUT}")
    print(f"  Tagged CSV:   {CLUSTER_CSV_OUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
