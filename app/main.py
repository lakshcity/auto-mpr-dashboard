# ========================= # Path setup (MUST be first) # =========================
import sys
import os
from pathlib import Path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(ROOT_DIR) / "data"
INDEX_FILE = "data/case_index_master.faiss"
META_FILE = "data/case_meta_master.pkl"
STATE_FILE = "data/case_hash_state.pkl"  # optional

# --- FIX: SMART LOGO FINDER
def get_logo_path():
    """Tries to find the logo in multiple locations to prevent crashes."""
    candidates = [
        BASE_DIR / "ui" / "assets" / "company_logo.png",  # Standard
        BASE_DIR.parent / "assets" / "company_logo.png",  # Root assets
        BASE_DIR / "assets" / "company_logo.png",         # Local assets
        Path("assets/company_logo.png")                   # Current Working Dir
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None

LOGO_PATH = get_logo_path()

# ✅ FIX 2: Merged all imports from user_insights to prevent reload issues
from services.user_insights import (
    # existing (kept)
    get_recent_cases,
    get_user_or_case_insights,
    get_pending_cases,
    get_overdue_cases,
    get_critical_cases,
    get_latest_resolved_cases,
    # NEW business-day flow (added)
    get_all_owners,
    filter_user_fy_fq_fm,
    compute_user_summary_bd,
    compute_sla_metrics_bd,
    add_business_ageing
)

from services.retriever import search_redash_mpr
from services.agent import pdf_agent


# ========================= # Imports # =========================
import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

Path("data/case_index_master.faiss")

# ========================= # Streamlit UI Config # =========================
st.set_page_config(page_title="Auto MPR Recommendation", layout="wide")

# ========================= # Load CSS (Merged Styles) # =========================
def load_css():
    try:
        # Try finding css relative to this file
        css_path = BASE_DIR / "ui" / "style.css"
        if css_path.exists():
            with open(css_path) as f:
                st.markdown(f"", unsafe_allow_html=True)
    except:
        pass
    # (Keep placeholder styled blocks if you had them)
    st.markdown("""""", unsafe_allow_html=True)

load_css()

# ========================= # Helper: Build Recommendation #
# (Works for both CSV columns and PDF Chunks) # =========================
def build_recommendation(results, top_n=3):
    recommendations = []
    # Priority list of fields to look for solution text
    POSSIBLE_TEXT_FIELDS = [
        "Resolution", "resolution", "solution", "answer",  # CSV fields
        "page_content", "chunk", "content", "details"      # PDF/Chunk fields
    ]
    for r in results[:top_n]:
        for field in POSSIBLE_TEXT_FIELDS:
            # Check if field exists and has content
            if r.get(field) and str(r[field]).strip():
                clean_text = str(r[field]).strip()
                # Avoid duplicates
                if clean_text not in recommendations:
                    recommendations.append(clean_text)
                break  # Found the best field for this result, move to next result
    if not recommendations:
        return "No clear resolution found in historical data."
    # Format as bullet points
    formatted_points = [f"• {rec}" for rec in recommendations]
    return "\n".join(formatted_points)

# --- Tiny UI helpers for Executive Summary visuals ---

def _dot(color_hex: str, size: int = 12) -> str:
    """Return an HTML span representing a colored dot."""
    return f"""<span style="
        display:inline-block; width:{size}px; height:{size}px; 
        background:{color_hex}; border-radius:50%;
        border:1px solid rgba(0,0,0,0.08); vertical-align:middle;
    "></span>"""

def _legend_row(label: str, dot_html: str, value: str) -> str:
    """One line for the visuals panel (dot + label + value)."""
    return f"""
    <div style="display:flex; align-items:center; justify-content:space-between; gap:8px; padding:4px 6px;">
        <div style="display:flex; align-items:center; gap:8px;">
            {dot_html}
            <span style="font-size:0.92rem;">{label}</span>
        </div>
        <span style="font-weight:600;">{value}</span>
    </div>
    """

def _resolution_ring_chart(percent: float, color_hex: str):
    """Small donut ring (Altair) for Resolution Rate."""
    import altair as alt
    import pandas as pd

    pct = max(0, min(100, float(percent)))
    data = pd.DataFrame({
        'segment': ['filled', 'empty'],
        'value': [pct, 100 - pct]
    })
    color_empty = '#E6E8EB'  # light grey
    chart = alt.Chart(data, width=90, height=90).mark_arc(innerRadius=28).encode(
        theta='value:Q',
        color=alt.Color('segment:N',
                        scale=alt.Scale(domain=['filled', 'empty'],
                                        range=[color_hex, color_empty]),
                        legend=None)
    )
    # center label
    text = alt.Chart(pd.DataFrame({'label':[f'{pct:.0f}%']})).mark_text(
        fontSize=14, fontWeight='bold', color='#2C3E50'
    ).encode(text='label:N')
    return chart + text

def _rate_color(percent: float) -> str:
    """Red/Amber/Green thresholds for resolution rate."""
    if percent < 50:
        return '#e74c3c'   # red
    elif percent < 75:
        return '#f39c12'   # amber
    else:
        return '#2ecc71'   # green

def render_confidence_bar(score):

    color = ""
    if score < 20:
        color = "#d9534f"  # red
    elif score < 40:
        color = "#f0ad4e"  # orange
    elif score < 60:
        color = "#ffd700"  # yellow
    elif score < 80:
        color = "#5bc0de"  # light blue
    else:
        color = "#5cb85c"  # green

    st.markdown(
        f"""
        <div style='background:#eee;width:100%;height:18px;border-radius:4px;'>
            <div style='width:{score}%;background:{color};height:18px;border-radius:4px;'></div>
        </div>
        <p style='font-size:12px;'>Confidence: {score}%</p>
        """,
        unsafe_allow_html=True
    )


# --- Add near your other helpers ---
import streamlit.components.v1 as components

def _avg_resolution_gauge(avg_bd: float, max_bd: int = 20):
    """
    5-band horizontal gauge for Avg Resolution Time (BD), with a rule & label at the current value.
    Bands (BD): 0–2 (green), 2–4 (teal), 4–7 (yellow), 7–14 (amber), 14–max (red).
    """
    import altair as alt
    import pandas as pd

    # Clamp values
    try:
        v = float(avg_bd)
    except:
        v = 0.0
    v = max(0.0, min(v, float(max_bd)))

    bands = [
        {"label": "≤2",          "start": 0,  "end": min(2, max_bd),   "color": "#2ecc71"},  # green
        {"label": "2–4",         "start": 2,  "end": min(4, max_bd),   "color": "#1abc9c"},  # teal
        {"label": "4–7",         "start": 4,  "end": min(7, max_bd),   "color": "#f1c40f"},  # yellow
        {"label": "7–14",        "start": 7,  "end": min(14, max_bd),  "color": "#f39c12"},  # amber
        {"label": f"≥{min(14,max_bd)}", "start": 14, "end": max_bd,    "color": "#e74c3c"},  # red
    ]

    # Normalize any inverted ranges if max_bd < some band cut
    bands = [b for b in bands if b["start"] < b["end"]]

    df = pd.DataFrame(bands)

    base = alt.Chart(df).mark_bar(height=16).encode(
        x=alt.X("start:Q",
                scale=alt.Scale(domain=[0, max_bd]),
                axis=alt.Axis(title="BD", values=[0,2,4,7,14,max_bd], labelFontSize=10, titleFontSize=11)),
        x2="end:Q",
        color=alt.Color("label:N", scale=alt.Scale(
            domain=[b["label"] for b in bands],
            range=[b["color"] for b in bands]),
            legend=None),
        tooltip=[alt.Tooltip("label:N"), alt.Tooltip("start:Q"), alt.Tooltip("end:Q")]
    )

    marker_df = pd.DataFrame({"x": [v], "txt": [f"{avg_bd:.2f} BD"]})

    rule = alt.Chart(marker_df).mark_rule(color="#2c3e50", size=2).encode(x="x:Q")
    text = alt.Chart(marker_df).mark_text(dy=-8, fontSize=12, fontWeight="bold", color="#2c3e50").encode(
        x="x:Q", text="txt:N"
    )

    chart = (base + rule + text).properties(height=48, width=260) \
        .configure_view(stroke=None) \
        .configure_axis(grid=False, domain=False)

    return chart


def _trend_arrow(avg_bd: float) -> str:
    """
    Trend arrow based on absolute Avg Resolution Time (BD).
    ≤3 BD: green ↘︎ (good/low), 3–7 BD: amber →, >7 BD: red ↗︎
    """
    try:
        v = float(avg_bd)
    except:
        v = 0.0
    if v <= 3:
        return '<span style="color:#2ecc71;font-weight:700;">↘︎</span>'
    elif v <= 7:
        return '<span style="color:#f39c12;font-weight:700;">→</span>'
    else:
        return '<span style="color:#e74c3c;font-weight:700;">↗︎</span>'

# ========================= # Sidebar # =========================
with st.sidebar:
    if LOGO_PATH:
        st.image(LOGO_PATH, use_container_width=True)
    st.markdown("#### Auto MPR", unsafe_allow_html=True)
    st.caption("Internal AI Demo")
    if st.button("Clear Cache & Reset"):
        st.cache_resource.clear()
        st.rerun()

# ========================= # Header # =========================
st.markdown("\n", unsafe_allow_html=True)
if LOGO_PATH:
    st.image(LOGO_PATH, width=160)
st.markdown("\n", unsafe_allow_html=True)
st.markdown(""" """, unsafe_allow_html=True)
st.markdown("## Auto MPR Response Recommendation", unsafe_allow_html=True)

# ========================= # Query Mode # =========================
st.markdown("\n", unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    query_mode = st.radio(
        "Query Mode",
        ["General MPR Issue", "User-Specific View"],
        horizontal=True,
        label_visibility="collapsed"
    )

# ========================= # Load Resources # =========================
scale = "master"

@st.cache_resource(ttl="30m")
def load_resources():
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # ROOT_DIR is defined at the top of main.py
        index_path = DATA_DIR / "case_index_master.faiss"
        meta_path = DATA_DIR / "case_meta_master.pkl"
        if not index_path.exists():
            # Fallback for local UI development if data is inside ui/data
            index_path = BASE_DIR / "data" / "case_index_master.faiss"
            meta_path = BASE_DIR / "data" / "case_meta_master.pkl"
        if index_path.exists():
            index = faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
            return model, index, metadata
        else:
            st.error(f"Files not found at {index_path}")
            return model, None, []
    except Exception as e:
        st.error(f"Resource Load Error: {e}")
        return None, None, []

model, index, metadata = load_resources()

# ========================= # Inputs # =========================
st.markdown('\n', unsafe_allow_html=True)
ci1, ci2, ci3 = st.columns([1, 2, 1])
with ci2:
    if query_mode == "General MPR Issue":
        query = st.text_area("Enter MPR Issue", height=100, placeholder="Describe the issue...")
        # Keep run button consistent across modes
        run_clicked = st.button("Run", use_container_width=True)
    else:
        # --- NEW: user dropdown and FY/FQ/FM filters based on reportedon
        owners = get_all_owners()
        selected_user = st.selectbox("Select User", owners, index=0 if owners else None)
        # Filters appear AFTER user
        fy = fq = fm = None
        if selected_user:
            ff1, ff2, ff3 = st.columns(3)
            with ff1:
                fy = st.selectbox("Financial Year (Apr–Mar)", ["All", "FY24", "FY25", "FY26"])
            with ff2:
                fq = st.selectbox("Financial Quarter", ["All", "Q1", "Q2", "Q3", "Q4"])
            with ff3:
                # --- FIX: Quarter selection restricts FM options to the 3 months in that quarter
                quarter_to_fm = {
                    "Q1": [1, 2, 3],   # Apr–Jun
                    "Q2": [4, 5, 6],   # Jul–Sep
                    "Q3": [7, 8, 9],   # Oct–Dec
                    "Q4": [10, 11, 12] # Jan–Mar
                }
                if fq and fq != "All":
                    fm_options = ["All"] + quarter_to_fm.get(fq, list(range(1, 13)))
                else:
                    fm_options = ["All"] + list(range(1, 13))
                fm = st.selectbox("Financial Month (Apr=1 ... Mar=12)", fm_options)
            # Optional visual hint of month names without changing values
            if fq == "Q1":
                st.caption("Q1 months: Apr (1), May (2), Jun (3)")
            elif fq == "Q2":
                st.caption("Q2 months: Jul (4), Aug (5), Sep (6)")
            elif fq == "Q3":
                st.caption("Q3 months: Oct (7), Nov (8), Dec (9)")
            elif fq == "Q4":
                st.caption("Q4 months: Jan (10), Feb (11), Mar (12)")
            # Gate note
            st.caption(":information_source: All metrics use **reported on** date and **exclude weekends**.")
        # Keep a button (same label) to avoid changing user habits / LOC
        run_clicked = st.button("Run", use_container_width=True)
        # keep original variable to avoid breaking dependent code (not used in new flow)
        user_id = ""

st.markdown('\n', unsafe_allow_html=True)

# ========================= # State Management # =========================
if "user_summary" not in st.session_state:
    st.session_state.user_summary = None
if "active_owner" not in st.session_state:
    st.session_state.active_owner = None
# --- FIX: persist the filtered frame to avoid “blank page” on radio change
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = None

# Reset state on mode switch
if "last_query_mode" not in st.session_state:
    st.session_state.last_query_mode = query_mode
if st.session_state.last_query_mode != query_mode:
    st.session_state.user_summary = None
    st.session_state.active_owner = None
    st.session_state.filtered_df = None
    st.session_state.last_query_mode = query_mode

# =========================
# Logic: General Search (CLEAN HYBRID FLOW)
# =========================

if run_clicked and query_mode == "General MPR Issue":

    if not query.strip():
        st.warning("Please enter an MPR issue.")

    elif index is None:
        st.error("Index not found. Please check data files.")

    else:
        with st.spinner("Searching similar past MPRs..."):
            results = search_redash_mpr(query)

        if not results:
            st.warning("No similar MPR subjects found in Redash.")
            st.session_state.pop("similar_results", None)
            st.session_state.pop("recommendation_text", None)

        else:
            # Sort by confidence once
            results = sorted(results, key=lambda x: x.get("confidence", 0), reverse=True)

            # Store results in session
            st.session_state["similar_results"] = results
            st.session_state["last_query"] = query

            # Clear previous recommendation when new search happens
            st.session_state.pop("recommendation_text", None)


# =========================
# DISPLAY SIMILAR RESULTS (ONLY ONCE)
# =========================

if query_mode == "General MPR Issue" and "similar_results" in st.session_state:

    results = st.session_state["similar_results"]

    st.subheader("🔍 Similar Historical Cases")

    for i, r in enumerate(results, 1):

        conf = round(r.get("confidence", 0), 2)
        header_subject = r.get("mpr_subject") or "Unknown Subject"
        header_id = r.get("caseid") or f"Doc-{i}"

        with st.expander(f"Case {i} — {header_id} — {conf}%"):
            render_confidence_bar(conf)

            st.write(f"**Subject:** {header_subject}")
            if r.get("statuscode"):
                st.write(f"**Status:** {r.get('statuscode')}")
            if r.get("reportedon"):
                st.write(f"**Reported On:** {r.get('reportedon')}")

    # -------------------------
    # Recommendation Button
    # -------------------------
    if st.button("🚀 Get Recommendation"):

        best_subject = results[0].get("mpr_subject", "")

        with st.spinner("Generating recommendation..."):
            try:
                recommendation_text = pdf_agent(best_subject)
                st.session_state["recommendation_text"] = recommendation_text
            except Exception as e:
                st.session_state["recommendation_text"] = f"PDF RAG Error: {str(e)}"


# =========================
# DISPLAY RECOMMENDATION (PERSISTENT)
# =========================

if query_mode == "General MPR Issue" and "recommendation_text" in st.session_state:

    st.markdown("### ✅ Recommended Solution")
    st.markdown(st.session_state["recommendation_text"])


# ========================= # Logic: User-Specific View (NEW dropdown + filters + strict gating) # =========================
if query_mode == "User-Specific View" and run_clicked:
    if not owners:
        st.warning("No owners found in data.")
    elif not selected_user:
        st.warning("Please select a user.")
    elif fy is None or fq is None or fm is None:
        st.info("Select FY, FQ and FM to view report.")
    else:
        # 1) Filter strictly on reportedon (FY/FQ/FM), weekends excluded in metrics
        filtered_df = filter_user_fy_fq_fm(selected_user, fy, fq, fm)
        if filtered_df.empty:
            st.error("No data found for the selected filters.")
        else:
            # persist to session for subsequent UI interactions
            st.session_state.filtered_df = filtered_df.copy()

            # 2) Build user summary (adds 'New Cases' and uses business-day ageing)
            s = compute_user_summary_bd(filtered_df)
            st.session_state.user_summary = s
            st.session_state.active_owner = s["owner"]

            # ----------- RENDERING ORDER -----------
            st.markdown('\n', unsafe_allow_html=True)
            st.subheader(f"👤 User Summary — {s['owner']}")

            # Metrics row: Total, New, Open, Overdue, Critical
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Cases", s["total_cases"])
            m2.metric("New Cases (≤2 BD)", s["new_cases"])
            m3.metric("Open Cases", s["open_cases"])
            m4.metric("Overdue (≥7 & <14 BD)", s["overdue_cases"])
            m5.metric("Critical (≥14 BD)", s["critical_cases"])

            # --- Executive Summary + SLA side-by-side
            es_col, sla_col = st.columns(2)
            with es_col:
                st.markdown("### Executive Summary")

                # Business-day aware metrics
                sla_table = compute_sla_metrics_bd(filtered_df)

                # Resolved-in-period = closed within the filtered frame
                resolved_in_period = int(filtered_df["closeddate"].notna().sum())
                total_cases = int(s["total_cases"])
                resolution_rate = round((resolved_in_period / total_cases) * 100, 2) if total_cases > 0 else 0.0

                # Build the core table (Metric/Value) on the left
                exec_rows = [
                    {"Metric": "Total Cases",               "Value": total_cases},
                    {"Metric": "Open Cases",                "Value": int(s["open_cases"])},
                    {"Metric": "Overdue (≥7 BD)",           "Value": int(sla_table["overdue_active"])},
                    {"Metric": "Critical (≥14 BD)",         "Value": int(sla_table["critical_active"])},
                    {"Metric": "Awaiting Input (>2 BD)",    "Value": int(sla_table["awaiting_input"])},
                    {"Metric": "Resolution Rate (%)",       "Value": resolution_rate},
                    {"Metric": "Avg Resolution Time (BD)",  "Value": float(sla_table["avg_resolution"])},
                ]
                es_df = pd.DataFrame(exec_rows, columns=["Metric", "Value"])

                tcol, vcol = st.columns([2, 1])
                with tcol:
                    st.dataframe(es_df, use_container_width=True)

                # Visuals panel on the right
                with vcol:
                    st.subheader("Visuals", help="All visuals computed on business days (weekends excluded).")

                    open_ct      = int(s["open_cases"])
                    overdue_ct   = int(sla_table["overdue_active"])
                    critical_ct  = int(sla_table["critical_active"])
                    awaiting_ct  = int(sla_table["awaiting_input"])

                    total_cases = int(s["total_cases"])
                    open_ratio = (open_ct / total_cases) if total_cases > 0 else 0

                    # Color rules
                    # Open: green if 0; amber if >0 and <5% of total; red if ≥5%
                    if open_ct == 0:
                        open_color = '#2ecc71'
                    elif open_ratio < 0.05:
                        open_color = '#f39c12'
                    else:
                        open_color = '#e74c3c'

                    overdue_color  = '#2ecc71' if overdue_ct == 0  else '#f39c12'
                    critical_color = '#2ecc71' if critical_ct == 0 else '#e74c3c'
                    await_color    = '#2ecc71' if awaiting_ct == 0 else '#e74c3c'

                    dot_open     = _dot(open_color)
                    dot_overdue  = _dot(overdue_color)
                    dot_critical = _dot(critical_color)
                    dot_await    = _dot(await_color)

                    # Build list HTML (no stray </div> by using components.html)
                    rows_html = ""
                    rows_html += _legend_row("Open Cases",              dot_open,     f"{open_ct}")
                    rows_html += _legend_row("Overdue (≥7 BD)",         dot_overdue,  f"{overdue_ct}")
                    rows_html += _legend_row("Critical (≥14 BD)",       dot_critical, f"{critical_ct}")
                    rows_html += _legend_row("Awaiting Input (>2 BD)",  dot_await,    f"{awaiting_ct}")

                    visual_box = f"""
                    <div style="border:1px solid #EEF0F3; border-radius:8px; background:#FFF; padding:6px;">
                        {rows_html}
                    </div>
                    """
                    # Dynamically size the iframe height to avoid scrollbars
                    components.html(visual_box, height=(4 * 36) + 20, scrolling=False)

                    # Resolution Rate donut
                    resolved_in_period = int(filtered_df["closeddate"].notna().sum())
                    resolution_rate = round((resolved_in_period / total_cases) * 100, 2) if total_cases > 0 else 0.0
                    st.markdown("##### Resolution Rate")
                    st.altair_chart(_resolution_ring_chart(resolution_rate, _rate_color(resolution_rate)), use_container_width=False)

                    # Avg Resolution Time (5‑band horizontal gauge)
                    st.markdown("##### Avg Resolution Time (BD)")
                    st.altair_chart(_avg_resolution_gauge(sla_table["avg_resolution"], max_bd=20), use_container_width=True)

                st.caption("All metrics calculated on **business days** (weekends excluded).")

            with sla_col:
                st.markdown("### SLA Metrics")
                sla = compute_sla_metrics_bd(filtered_df)
                sm1, sm2, sm3 = st.columns(3)
                sm1.metric("Active in SLA (<7 BD)", sla["active_in_sla"])
                sm2.metric("Near Breach (5–<7 BD)", sla["near_breach"])
                sm3.metric("SLA Breached (>7 BD)", sla["sla_breached"])
                sm4, sm5, sm6 = st.columns(3)
                sm4.metric("Overdue Active (≥7 & <14 BD)", sla["overdue_active"])
                sm5.metric("Critical Active (≥14 BD)", sla["critical_active"])
                sm6.metric("Awaiting Input (>2 BD)", sla["awaiting_input"])
                sm7, sm8 = st.columns(2)
                sm7.metric("SLA Compliance (%)", sla["sla_compliance"])
                sm8.metric("Avg Resolution Time (BD)", sla["avg_resolution"])

            # --- Status Breakdown + Active/Closed Load charts
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("#### Status Breakdown")
                # --- FIX: normalize labels, drop 'Unknown', add 'Others'
                status_series = (
                    filtered_df["statuscode"].astype(str).str.strip()
                    .replace("", "Unknown").str.title().value_counts()
                )
                if "Unknown" in status_series.index:
                    status_series = status_series.drop("Unknown")
                total_ct = int(status_series.sum())
                top = status_series.head(7)
                others_ct = total_ct - int(top.sum())
                labels = top.index.tolist()
                counts = top.values.tolist()
                if others_ct > 0:
                    labels += ["Others"]
                    counts += [others_ct]
                if sum(counts) <= 0:
                    st.info("No cases available for charting.")
                else:
                    fig1, ax1 = plt.subplots(figsize=(4, 3))
                    wedges, texts, autotexts = ax1.pie(
                        counts, autopct='%1.0f%%', startangle=90,
                        textprops={'color': "white", 'weight': 'bold'}
                    )
                    ax1.legend(
                        wedges, labels, title="Status",
                        loc="center left", bbox_to_anchor=(1, 0, 0.5, 1)
                    )
                    ax1.axis('equal')
                    fig1.patch.set_alpha(0)
                    st.pyplot(fig1, use_container_width=False)

            with chart_col2:
                st.markdown("#### Active Load Severity")
                active_df = filtered_df[filtered_df["closeddate"].isna()].copy()
                if "ageing_bd" not in active_df.columns:
                    active_df = add_business_ageing(active_df)  # defensive
                fresh_a = int((active_df["ageing_bd"] < 7).sum())
                overdue_a = int(((active_df["ageing_bd"] >= 7) & (active_df["ageing_bd"] < 14)).sum())
                critical_a = int((active_df["ageing_bd"] >= 14).sum())
                aging_data = pd.DataFrame({
                    "Category": ["Fresh (<7 BD)", "Overdue (≥7 & <14 BD)", "Critical (≥14 BD)"],
                    "Cases": [fresh_a, overdue_a, critical_a]
                })
                c = alt.Chart(aging_data).mark_bar().encode(
                    x=alt.X('Category', sort=["Fresh (<7 BD)", "Overdue (≥7 & <14 BD)", "Critical (≥14 BD)"]),
                    y='Cases',
                    color=alt.Color('Category', scale=alt.Scale(
                        domain=["Fresh (<7 BD)", "Overdue (≥7 & <14 BD)", "Critical (≥14 BD)"],
                        range=['#2ecc71', '#f39c12', '#e74c3c']
                    )),
                    tooltip=['Category', 'Cases']
                ).properties(height=300)
                st.altair_chart(c, use_container_width=True)

            # --- FIX: add Resolution Mix (closed in the selected period)
            st.markdown("#### Resolution Mix (Closed in period)")
            closed_df = filtered_df[filtered_df["closeddate"].notna()].copy()
            if not closed_df.empty and "ageing_bd" not in closed_df.columns:
                closed_df = add_business_ageing(closed_df)  # at closure
            if closed_df.empty:
                st.info("No cases closed in this period.")
            else:
                fresh_c = int((closed_df["ageing_bd"] < 7).sum())
                overdue_c = int(((closed_df["ageing_bd"] >= 7) & (closed_df["ageing_bd"] < 14)).sum())
                critical_c = int((closed_df["ageing_bd"] >= 14).sum())
                res_data = pd.DataFrame({
                    "Category": ["Fresh (<7 BD)", "Overdue (≥7 & <14 BD)", "Critical (≥14 BD)"],
                    "Cases": [fresh_c, overdue_c, critical_c]
                })
                rc = alt.Chart(res_data).mark_bar().encode(
                    x=alt.X('Category', sort=["Fresh (<7 BD)", "Overdue (≥7 & <14 BD)", "Critical (≥14 BD)"]),
                    y='Cases',
                    color=alt.Color('Category', scale=alt.Scale(
                        domain=["Fresh (<7 BD)", "Overdue (≥7 & <14 BD)", "Critical (≥14 BD)"],
                        range=['#2ecc71', '#f39c12', '#e74c3c']
                    )),
                    tooltip=['Category', 'Cases']
                ).properties(height=300)
                st.altair_chart(rc, use_container_width=True)

            # --- Latest Top 3 Resolved
            st.markdown("---")
            st.subheader("✅ Latest Resolved Cases (Top 3)")
            latest_resolved = get_latest_resolved_cases(s["owner"], top_n=3)
            if not latest_resolved:
                st.info("No resolved cases found for this user.")
            else:
                for rc in latest_resolved:
                    case_id = rc.get("caseid", "NA")
                    subj = rc.get("subject", "") or rc.get("MPR_Subject", "") or ""
                    reported_on = pd.to_datetime(rc.get("reportedon", None), errors="coerce")
                    closed_on = pd.to_datetime(rc.get("closeddate", None), errors="coerce")
                    header = f"✅ Case {case_id} \n {subj}"
                    with st.expander(header):
                        st.write(f"**Reported On:** {reported_on}")
                        st.write(f"**Closed On:** {closed_on}")
                        st.write(f"**Resolution Time:** {round(float(rc.get('resolution_days', 0)), 2)} days")
                        st.write(f"**Details:** {rc.get('details','')}")
                        # 1) Gantt (Reported -> Closed)
                        if pd.notnull(reported_on) and pd.notnull(closed_on):
                            gantt_df = pd.DataFrame([{
                                "Task": f"Case {case_id}",
                                "Start": reported_on,
                                "End": closed_on
                            }])
                            gantt = alt.Chart(gantt_df).mark_bar(size=18).encode(
                                y=alt.Y("Task:N", title=""),
                                x=alt.X("Start:T", title="Timeline"),
                                x2="End:T",
                                color=alt.value("#7b2ff7"),
                                tooltip=[
                                    alt.Tooltip("Task:N"),
                                    alt.Tooltip("Start:T"),
                                    alt.Tooltip("End:T")
                                ]
                            ).properties(height=70)
                            st.altair_chart(gantt, use_container_width=True)
                        else:
                            st.info("Timeline not available (missing reportedon/closeddate).")
                        # 2) Effort Breakdown
                        effort_df = pd.DataFrame({
                            "Effort Type": ["Configuration", "Testing", "Total"],
                            "Effort": [
                                float(rc.get("configurationeffort", 0) or 0),
                                float(rc.get("testingeffort", 0) or 0),
                                float(rc.get("totaleffort", 0) or 0),
                            ],
                        })
                        effort_chart = alt.Chart(effort_df).mark_bar().encode(
                            y=alt.Y("Effort Type:N", sort=["Configuration", "Testing", "Total"], title=""),
                            x=alt.X("Effort:Q", title="Effort"),
                            color=alt.Color(
                                "Effort Type:N", legend=None,
                                scale=alt.Scale(
                                    domain=["Configuration", "Testing", "Total"],
                                    range=["#2ecc71", "#f39c12", "#e74c3c"]
                                )
                            ),
                            tooltip=["Effort Type", "Effort"]
                        ).properties(height=140)
                        st.altair_chart(effort_chart, use_container_width=True)

            # --- 🆕 Focused Case View (UPDATED)
            st.markdown('\n', unsafe_allow_html=True)
            st.subheader("📌 Focused Case View")

            # Add two new categories: "New (≤2 BD)" and "Open (All Active)"
            case_type = st.radio(
                "Select category",
                ["New (≤2 BD)", "Fresh (<7 BD)", "Overdue (≥7 & <14 BD)", "Critical (≥14 BD)", "Open (All Active)"],
                horizontal=True
            )

            base_df = filtered_df.copy()
            if "ageing_bd" not in base_df.columns:
                base_df = add_business_ageing(base_df)  # ensures ageing_bd is present

            # --- Color / badge map to keep visuals consistent
            COLOR_BADGE = {
                "New (≤2 BD)":           ("#1abc9c",  "🟢 New"),
                "Fresh (<7 BD)":         ("#2ecc71",  "🟢 Fresh"),
                "Overdue (≥7 & <14 BD)": ("#f39c12",  "🟠 Overdue"),
                "Critical (≥14 BD)":     ("#e74c3c",  "🔴 Critical"),
                "Open (All Active)":     ("#3498db",  "🔵 Open")
            }

            # --- Build the filtered view for the selected category
            if "New" in case_type:
                # New = very recent active cases (≤2 business days)
                cdf = base_df[(base_df["closeddate"].isna()) & (base_df["ageing_bd"] <= 2)] \
                    .sort_values(by="ageing_bd", ascending=False)
                empty_msg = "No new cases (≤2 business days) found."
            elif "Fresh" in case_type:
                cdf = base_df[base_df["ageing_bd"] < 7] \
                    .sort_values(by="ageing_bd", ascending=False)
                empty_msg = "No fresh cases (<7 business days) found."
            elif "Overdue" in case_type:
                cdf = base_df[(base_df["ageing_bd"] >= 7) & (base_df["ageing_bd"] < 14)] \
                    .sort_values(by="ageing_bd", ascending=False)
                empty_msg = "No overdue cases (≥7 & <14 business days) found."
            elif "Critical" in case_type:
                cdf = base_df[base_df["ageing_bd"] >= 14] \
                    .sort_values(by="ageing_bd", ascending=False)
                empty_msg = "No critical cases (≥14 business days) found."
            else:
                # Open = all active (irrespective of ageing bucket)
                cdf = base_df[base_df["closeddate"].isna()] \
                    .sort_values(by="ageing_bd", ascending=False)
                empty_msg = "No open cases found."

            if cdf.empty:
                st.info(empty_msg)
            else:
                color, badge = COLOR_BADGE.get(case_type, ("#7b2ff7", ""))
                for _, row in cdf.head(5).iterrows():
                    header_text = f"{badge}\nID: {row.get('caseid')}\n{row.get('subject','No Subject')}"
                    with st.expander(header_text):
                        st.write(f"**Aging (BD):** {row.get('ageing_bd', 0)}")
                        st.write(f"**Status:** {row.get('statuscode')}")
                        st.write(f"**Reported On:** {row.get('reportedon')}")
                        st.write(f"**Closed On:** {row.get('closeddate')}")
                        st.write(f"**Details:** {row.get('details','')}")

# --- FIX: Render from session if user toggles radio or interacts without clicking Run again
if query_mode == "User-Specific View" and not run_clicked and st.session_state.filtered_df is not None:
    filtered_df = st.session_state.filtered_df.copy()
    s = st.session_state.user_summary or compute_user_summary_bd(filtered_df)

    st.markdown('\n', unsafe_allow_html=True)
    st.subheader(f"👤 User Summary — {s['owner']}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Cases", s["total_cases"])
    m2.metric("New Cases (≤2 BD)", s["new_cases"])
    m3.metric("Open Cases", s["open_cases"])
    m4.metric("Overdue (≥7 & <14 BD)", s["overdue_cases"])
    m5.metric("Critical (≥14 BD)", s["critical_cases"])

    es_col, sla_col = st.columns(2)
    with es_col:
        st.markdown("### Executive Summary")

        sla_table = compute_sla_metrics_bd(filtered_df)
        resolved_in_period = int(filtered_df["closeddate"].notna().sum())
        total_cases = int(s["total_cases"])
        resolution_rate = round((resolved_in_period / total_cases) * 100, 2) if total_cases > 0 else 0.0

        exec_rows = [
            {"Metric": "Total Cases",               "Value": total_cases},
            {"Metric": "Open Cases",                "Value": int(s["open_cases"])},
            {"Metric": "Overdue (≥7 BD)",           "Value": int(sla_table["overdue_active"])},
            {"Metric": "Critical (≥14 BD)",         "Value": int(sla_table["critical_active"])},
            {"Metric": "Awaiting Input (>2 BD)",    "Value": int(sla_table["awaiting_input"])},
            {"Metric": "Resolution Rate (%)",       "Value": resolution_rate},
            {"Metric": "Avg Resolution Time (BD)",  "Value": float(sla_table["avg_resolution"])},
        ]
        es_df = pd.DataFrame(exec_rows, columns=["Metric", "Value"])

        tcol, vcol = st.columns([2, 1])
        with tcol:
            st.dataframe(es_df, use_container_width=True)

        with vcol:
            st.markdown("#### Visuals", help="All visuals computed on business days (weekends excluded).")

            open_ct      = int(s["open_cases"])
            overdue_ct   = int(sla_table["overdue_active"])
            critical_ct  = int(sla_table["critical_active"])
            awaiting_ct  = int(sla_table["awaiting_input"])

            open_ratio = (open_ct / total_cases) if total_cases > 0 else 0
            if open_ct == 0:
                open_color = '#2ecc71'
            elif open_ratio < 0.05:
                open_color = '#f39c12'
            else:
                open_color = '#e74c3c'

            overdue_color  = '#2ecc71' if overdue_ct == 0  else '#f39c12'
            critical_color = '#2ecc71' if critical_ct == 0 else '#e74c3c'
            await_color    = '#2ecc71' if awaiting_ct == 0 else '#e74c3c'

            html = ""
            html += _legend_row("Open Cases",              _dot(open_color),     f"{open_ct}")
            html += _legend_row("Overdue (≥7 BD)",         _dot(overdue_color),  f"{overdue_ct}")
            html += _legend_row("Critical (≥14 BD)",       _dot(critical_color), f"{critical_ct}")
            html += _legend_row("Awaiting Input (>2 BD)",  _dot(await_color),    f"{awaiting_ct}")

            st.markdown(f"""
            <div style="border:1px solid #EEF0F3; border-radius:8px; background:#FFF; padding:6px;">
                {html}
            </div>
            """, unsafe_allow_html=True)

            rate_color = _rate_color(resolution_rate)
            st.markdown("##### Resolution Rate")
            st.altair_chart(_resolution_ring_chart(resolution_rate, rate_color), use_container_width=False)

            st.markdown("##### Avg Resolution Time (BD)")
            st.markdown(_trend_arrow(sla_table["avg_resolution"]), unsafe_allow_html=True)

        st.caption("All metrics calculated on **business days** (weekends excluded).")

    with sla_col:
        st.markdown("### SLA Metrics")
        sla = compute_sla_metrics_bd(filtered_df)
        sm1, sm2, sm3 = st.columns(3)
        sm1.metric("Active in SLA (<7 BD)", sla["active_in_sla"])
        sm2.metric("Near Breach (5–<7 BD)", sla["near_breach"])
        sm3.metric("SLA Breached (>7 BD)", sla["sla_breached"])
        sm4, sm5, sm6 = st.columns(3)
        sm4.metric("Overdue Active (≥7 & <14 BD)", sla["overdue_active"])
        sm5.metric("Critical Active (≥14 BD)", sla["critical_active"])
        sm6.metric("Awaiting Input (>2 BD)", sla["awaiting_input"])
        sm7, sm8 = st.columns(2)
        sm7.metric("SLA Compliance (%)", sla["sla_compliance"])
        sm8.metric("Avg Resolution Time (BD)", sla["avg_resolution"])

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("#### Status Breakdown")
        status_series = (
            filtered_df["statuscode"].astype(str).str.strip()
            .replace("", "Unknown").str.title().value_counts()
        )
        if "Unknown" in status_series.index:
            status_series = status_series.drop("Unknown")
        total_ct = int(status_series.sum())
        top = status_series.head(7)
        others_ct = total_ct - int(top.sum())
        labels = top.index.tolist()
        counts = top.values.tolist()
        if others_ct > 0:
            labels += ["Others"]
            counts += [others_ct]
        if sum(counts) <= 0:
            st.info("No cases available for charting.")
        else:
            fig1, ax1 = plt.subplots(figsize=(4, 3))
            wedges, texts, autotexts = ax1.pie(
                counts, autopct='%1.0f%%', startangle=90,
                textprops={'color': "white", 'weight': 'bold'}
            )
            ax1.legend(
                wedges, labels, title="Status",
                loc="center left", bbox_to_anchor=(1, 0, 0.5, 1)
            )
            ax1.axis('equal')
            fig1.patch.set_alpha(0)
            st.pyplot(fig1, use_container_width=False)

    with chart_col2:
        st.markdown("#### Active Load Severity")
        active_df = filtered_df[filtered_df["closeddate"].isna()].copy()
        if "ageing_bd" not in active_df.columns:
            active_df = add_business_ageing(active_df)
        fresh_a = int((active_df["ageing_bd"] < 7).sum())
        overdue_a = int(((active_df["ageing_bd"] >= 7) & (active_df["ageing_bd"] < 14)).sum())
        critical_a = int((active_df["ageing_bd"] >= 14).sum())
        aging_data = pd.DataFrame({
            "Category": ["Fresh (<7 BD)", "Overdue (≥7 & <14 BD)", "Critical (≥14 BD)"],
            "Cases": [fresh_a, overdue_a, critical_a]
        })
        c = alt.Chart(aging_data).mark_bar().encode(
            x=alt.X('Category', sort=["Fresh (<7 BD)", "Overdue (≥7 & <14 BD)", "Critical (≥14 BD)"]),
            y='Cases',
            color=alt.Color('Category', scale=alt.Scale(
                domain=["Fresh (<7 BD)", "Overdue (≥7 & <14 BD)", "Critical (≥14 BD)"],
                range=['#2ecc71', '#f39c12', '#e74c3c']
            )),
            tooltip=['Category', 'Cases']
        ).properties(height=300)
        st.altair_chart(c, use_container_width=True)

    st.markdown("#### Resolution Mix (Closed in period)")
    closed_df = filtered_df[filtered_df["closeddate"].notna()].copy()
    if not closed_df.empty and "ageing_bd" not in closed_df.columns:
        closed_df = add_business_ageing(closed_df)
    if closed_df.empty:
        st.info("No cases closed in this period.")
    else:
        fresh_c = int((closed_df["ageing_bd"] < 7).sum())
        overdue_c = int(((closed_df["ageing_bd"] >= 7) & (closed_df["ageing_bd"] < 14)).sum())
        critical_c = int((closed_df["ageing_bd"] >= 14).sum())
        res_data = pd.DataFrame({
            "Category": ["Fresh (<7 BD)", "Overdue (≥7 & <14 BD)", "Critical (≥14 BD)"],
            "Cases": [fresh_c, overdue_c, critical_c]
        })
        rc = alt.Chart(res_data).mark_bar().encode(
            x=alt.X('Category', sort=["Fresh (<7 BD)", "Overdue (≥7 & <14 BD)", "Critical (≥14 BD)"]),
            y='Cases',
            color=alt.Color('Category', scale=alt.Scale(
                domain=["Fresh (<7 BD)", "Overdue (≥7 & <14 BD)", "Critical (≥14 BD)"],
                range=['#2ecc71', '#f39c12', '#e74c3c']
            )),
            tooltip=['Category', 'Cases']
        ).properties(height=300)
        st.altair_chart(rc, use_container_width=True)

    # --- 🆕 Focused Case View (UPDATED - session branch)
    st.markdown('\n', unsafe_allow_html=True)
    st.subheader("📌 Focused Case View")

    case_type = st.radio(
        "Select category",
        ["New (≤2 BD)", "Fresh (<7 BD)", "Overdue (≥7 & <14 BD)", "Critical (≥14 BD)", "Open (All Active)"],
        horizontal=True
    )

    base_df = filtered_df.copy()
    if "ageing_bd" not in base_df.columns:
        base_df = add_business_ageing(base_df)

    COLOR_BADGE = {
        "New (≤2 BD)":           ("#1abc9c",  "🟢 New"),
        "Fresh (<7 BD)":         ("#2ecc71",  "🟢 Fresh"),
        "Overdue (≥7 & <14 BD)": ("#f39c12",  "🟠 Overdue"),
        "Critical (≥14 BD)":     ("#e74c3c",  "🔴 Critical"),
        "Open (All Active)":     ("#3498db",  "🔵 Open")
    }

    if "New" in case_type:
        cdf = base_df[(base_df["closeddate"].isna()) & (base_df["ageing_bd"] <= 2)] \
            .sort_values(by="ageing_bd", ascending=False)
        empty_msg = "No new cases (≤2 business days) found."
    elif "Fresh" in case_type:
        cdf = base_df[base_df["ageing_bd"] < 7] \
            .sort_values(by="ageing_bd", ascending=False)
        empty_msg = "No fresh cases (<7 business days) found."
    elif "Overdue" in case_type:
        cdf = base_df[(base_df["ageing_bd"] >= 7) & (base_df["ageing_bd"] < 14)] \
            .sort_values(by="ageing_bd", ascending=False)
        empty_msg = "No overdue cases (≥7 & <14 business days) found."
    elif "Critical" in case_type:
        cdf = base_df[base_df["ageing_bd"] >= 14] \
            .sort_values(by="ageing_bd", ascending=False)
        empty_msg = "No critical cases (≥14 business days) found."
    else:
        cdf = base_df[base_df["closeddate"].isna()] \
            .sort_values(by="ageing_bd", ascending=False)
        empty_msg = "No open cases found."

    if cdf.empty:
        st.info(empty_msg)
    else:
        color, badge = COLOR_BADGE.get(case_type, ("#7b2ff7", ""))
        for _, row in cdf.head(5).iterrows():
            header_text = f"{badge}\nID: {row.get('caseid')}\n{row.get('subject','No Subject')}"
            with st.expander(header_text):
                st.write(f"**Aging (BD):** {row.get('ageing_bd', 0)}")
                st.write(f"**Status:** {row.get('statuscode')}")
                st.write(f"**Reported On:** {row.get('reportedon')}")
                st.write(f"**Closed On:** {row.get('closeddate')}")
                st.write(f"**Details:** {row.get('details','')}")

# =========================
# (KEPT) Old "Dashboard Display" block — disabled to avoid duplicates
# =========================
if False and st.session_state.user_summary:
    s = st.session_state.user_summary
    st.markdown('\n', unsafe_allow_html=True)
    st.subheader(f"👤 User Summary — {s['owner']}")
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Cases", s["total_cases"])
    m2.metric("Fresh (<7d)", s["pending_cases"])
    m3.metric("Overdue (8-21d)", s["overdue_cases"])
    m4.metric("Critical (>21d)", s["critical_cases"])
    recent_cases = get_recent_cases(s["owner"], days=5)
    st.metric("Recent Cases (Last 5 Days)", len(recent_cases))
    # Latest Resolved, Charts, Recent Activity, Focused View (all kept but disabled)
    # (Original rendering code preserved here)