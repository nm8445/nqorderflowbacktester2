"""
NQ Orderflow — Research Study Runner (Streamlit)

Launch with:
    streamlit run scripts/study_runner.py

Select a study from the sidebar, configure parameters, and click Run Study.
Results are displayed inline and saved to output/studies/<id>_<start>_<end>.txt
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import streamlit as st

from nqbt.studies.registry import ALL_STUDIES, BLOCKS
from nqbt.studies.base import Param, StudyResult

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NQ Study Runner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUTPUT_DIR = Path("output/studies")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def render_param(p: Param):
    """Render a single Param as the appropriate Streamlit widget."""
    if p.ptype == "float":
        mn   = float(p.mn)   if p.mn   is not None else 0.0
        mx   = float(p.mx)   if p.mx   is not None else 1.0
        step = float(p.step) if p.step is not None else 0.01
        return st.slider(p.label, mn, mx, float(p.default), step, help=p.description)

    if p.ptype == "int":
        mn = int(p.mn) if p.mn is not None else 0
        mx = int(p.mx) if p.mx is not None else 1000
        return st.number_input(p.label, mn, mx, int(p.default), help=p.description)

    if p.ptype == "bool":
        return st.checkbox(p.label, bool(p.default), help=p.description)

    if p.ptype == "choice":
        choices = p.choices or []
        default_idx = choices.index(p.default) if p.default in choices else 0
        return st.selectbox(p.label, choices, index=default_idx, help=p.description)

    # str / fallback
    return st.text_input(p.label, str(p.default), help=p.description)


# ── Sidebar — study selection ─────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 NQ Study Runner")
    st.markdown("---")

    block_labels = list(BLOCKS.keys())
    selected_block = st.selectbox("Block", block_labels)

    studies_in_block = BLOCKS[selected_block]
    study_options = {f"{s.id}  {s.name}": s for s in studies_in_block}
    selected_label = st.selectbox("Study", list(study_options.keys()))
    study = study_options[selected_label]

    st.markdown("---")
    st.markdown("**Date Range**")
    start_date = st.date_input("Start", value=date(2025, 3, 17))
    end_date   = st.date_input("End",   value=date(2026, 3, 15))

    if start_date >= end_date:
        st.error("Start date must be before end date.")

    st.markdown("---")
    st.caption(f"Study ID: `{study.id}`")
    total = sum(len(v) for v in BLOCKS.values())
    st.caption(f"{total} studies registered")


# ── Main — description ────────────────────────────────────────────────────────

st.header(f"Study {study.id} — {study.name}")
st.markdown(f"**Block:** {study.block}")
st.markdown(study.description)
st.markdown("---")

# ── Parameters ────────────────────────────────────────────────────────────────

params: dict = {}

if not study.variables:
    st.info("This study has no configurable parameters.")
else:
    st.subheader("Parameters")
    n_cols = 3
    cols = st.columns(n_cols)
    for idx, p in enumerate(study.variables):
        with cols[idx % n_cols]:
            params[p.name] = render_param(p)

st.markdown("---")

# ── Run button ────────────────────────────────────────────────────────────────

run_col, _ = st.columns([1, 5])
with run_col:
    run_clicked = st.button("▶  Run Study", type="primary", use_container_width=True)

if run_clicked:
    if start_date >= end_date:
        st.error("Fix the date range before running.")
        st.stop()

    with st.spinner(f"Running study {study.id}…"):
        result: StudyResult = study.run(str(start_date), str(end_date), **params)

    st.success("Done.")
    st.subheader("Results")

    # Summary text
    st.text_area("Summary", result.summary, height=400)

    # Tables
    for table_name, df in result.tables:
        st.subheader(table_name)
        st.dataframe(df, use_container_width=True)

    # Save to file
    slug = f"{study.id}_{start_date}_{end_date}".replace(".", "_")
    out_path = OUTPUT_DIR / f"{slug}.txt"
    out_path.write_text(result.summary, encoding="utf-8")
    st.info(f"Results saved → `{out_path}`")

    st.download_button(
        label="⬇  Download .txt",
        data=result.summary.encode("utf-8"),
        file_name=out_path.name,
        mime="text/plain",
    )
