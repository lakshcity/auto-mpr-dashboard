# =========================
# Path setup (MUST be first)
# =========================
import sys
import os
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

BASE_DIR = Path(__file__).resolve().parent

# --- FIX: SMART LOGO FINDER ---
def get_logo_path():
    """Tries to find the logo in multiple locations to prevent crashes."""
    candidates = [
        BASE_DIR / "ui" / "assets" / "company_logo.png",   # Standard
        BASE_DIR.parent / "assets" / "company_logo.png",   # Root assets
        BASE_DIR / "assets" / "company_logo.png",          # Local assets
        Path("assets/company_logo.png")                    # Current Working Dir
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None

LOGO_PATH = get_logo_path()

# ✅ FIX 2: Merged all imports from user_insights to prevent reload issues
from services.user_insights import (
    get_recent_cases,
    get_user_or_case_insights,
    get_pending_cases,
    get_overdue_cases,
    get_critical_cases,
    get_latest_resolved_cases
)

# =========================
# Imports
# =========================
import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from services.retriever import find_similar_cases
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

CSV_PATH = "backend_api/data/cases_master.csv"

# =========================
# Streamlit UI Config
# =========================
st.set_page_config(
    page_title="Auto MPR Recommendation",
    layout="wide"
)

# =========================
# Load CSS (Merged Styles)
# =========================
def load_css():
    try:
        # Try finding css relative to this file
        css_path = BASE_DIR / "ui" / "style.css"
        if css_path.exists():
            with open(css_path) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

    # Merged CSS: Control Panel Hidden + Section Card + Recommendation Card
    st.markdown("""
        <style>
            .control-panel-hidden h3 {
                display: none;
            }
            .section-card {
                background-color: #ffffff;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            /* PDF/Resolutioning Styles */
            .recommendation-card {
                background: #f1f8f4;
                padding: 16px;
                border-left: 5px solid #2e7d32;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            .recommendation-title {
                font-weight: 600;
                font-size: 18px;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .recommendation-text {
                font-size: 16px;
                line-height: 1.5;
                color: #1f1f1f;
            }
        </style>
    """, unsafe_allow_html=True)

load_css()

# =========================
# Helper: Build Recommendation
# (Works for both CSV columns and PDF Chunks)
# =========================
def build_recommendation(results, top_n=3):
    recommendations = []
    
    # Priority list of fields to look for solution text
    POSSIBLE_TEXT_FIELDS = [
        "Resolution", "resolution", "solution", "answer", # CSV fields
        "page_content", "chunk", "content", "details"     # PDF/Chunk fields
    ]

    for r in results[:top_n]:
        for field in POSSIBLE_TEXT_FIELDS:
            # Check if field exists and has content
            if r.get(field) and str(r[field]).strip():
                clean_text = str(r[field]).strip()
                # Avoid duplicates
                if clean_text not in recommendations:
                    recommendations.append(clean_text)
                break # Found the best field for this result, move to next result

    if not recommendations:
        return "No clear resolution found in historical data."

    # Format as bullet points
    formatted_points = [f"• {rec}" for rec in recommendations]
    return "\n".join(formatted_points)

# =========================
# Sidebar
# =========================
with st.sidebar:
    if LOGO_PATH:
        st.image(LOGO_PATH, use_container_width=True)
    st.markdown("<h3 style='margin-top:10px;'>Auto MPR</h3>", unsafe_allow_html=True)
    st.caption("Internal AI Demo")
    if st.button("Clear Cache & Reset"):
        st.cache_resource.clear()
        st.rerun()

# =========================
# Header
# =========================
st.markdown("<div style='display:flex; justify-content:center; margin-top:16px;'>", unsafe_allow_html=True)
if LOGO_PATH:
    st.image(LOGO_PATH, width=160)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="height:5px; margin:18px auto 30px auto; border-radius:14px; max-width:1100px; background: linear-gradient(90deg, #7b2ff7, #f107a3);"></div>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align:center;'>Auto MPR Response Recommendation</h1>", unsafe_allow_html=True)

# =========================
# Query Mode
# =========================
st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    query_mode = st.radio(
    "Query Mode",
    ["General MPR Issue", "User-Specific View"],
    horizontal=True,
    label_visibility="collapsed"
)


# =========================
# Load Resources
# =========================
scale = "master"

# Cache with TTL so index reloads periodically (e.g. every 30 minutes)
@st.cache_resource(ttl="30m")
def load_resources(scale):
    try:
        # -------------------------
        # PRIMARY: CSV / cases_master FAISS index
        # -------------------------
        model = SentenceTransformer("all-MiniLM-L6-v2")

        index_candidates = [
            BASE_DIR.parent / "data" / f"case_index_master.faiss",
            BASE_DIR / "data" / f"case_index_master.faiss",
            Path(f"data/case_index_master.faiss")
        ]

        index_path = None
        for p in index_candidates:
            if p.exists():
                index_path = p
                break

        if not index_path:
            raise FileNotFoundError("CSV FAISS index not found")

        index = faiss.read_index(str(index_path))

        meta_path = (
            str(index_path)
            .replace(".faiss", ".pkl")
            .replace("case_index_", "case_meta_")
        )

        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        return model, index, metadata

    except Exception as e_csv:
        # -------------------------
        # FALLBACK: PDF FAISS index
        # -------------------------
        try:
            index = faiss.read_index("data/pdf_index.faiss")
            with open("data/pdf_meta.pkl", "rb") as f:
                metadata = pickle.load(f)

            return SentenceTransformer("all-MiniLM-L6-v2"), index, metadata

        except Exception as e_pdf:
            # Last-resort safe fallback (app should not crash)
            st.error(
                f"Error loading FAISS resources.\n"
                f"CSV index error: {e_csv}\n"
                f"PDF index error: {e_pdf}"
            )
            return SentenceTransformer("all-MiniLM-L6-v2"), None, []



model, index, metadata = load_resources(scale)

# =========================
# Inputs
# =========================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    if query_mode == "General MPR Issue":
        query = st.text_area("Enter MPR Issue", height=100, placeholder="Describe the issue...")
    else:
        user_id = st.text_input("Enter CaseID or User Name", placeholder="e.g. Kumar Sanu")
    run_clicked = st.button("Run", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# State Management
# =========================
if "user_summary" not in st.session_state:
    st.session_state.user_summary = None
if "active_owner" not in st.session_state:
    st.session_state.active_owner = None

# Reset state on mode switch
if "last_query_mode" not in st.session_state:
    st.session_state.last_query_mode = query_mode
if st.session_state.last_query_mode != query_mode:
    st.session_state.user_summary = None
    st.session_state.active_owner = None
    st.session_state.last_query_mode = query_mode

# =========================
# Logic: General Search (INTEGRATED PDF/CSV LOGIC)
# =========================
if run_clicked and query_mode == "General MPR Issue":
    if not query.strip():
        st.warning("Please enter an MPR issue.")
    elif index is None:
        st.error("Index not found. Please check data files.")
    else:
        with st.spinner("Searching similar past MPRs..."):
            results = find_similar_cases(query, model, index, metadata)
        
        results = sorted(results, key=lambda x: x.get("confidence", 0), reverse=True)
        best_conf = round(results[0].get("confidence", 0), 2) if results else 0

        # --- 1. RECOMMENDED RESOLUTION (Integrated Feature) ---
        recommendation_text = build_recommendation(results)
        
        st.markdown(f"""
            <div class="recommendation-card">
                <div class="recommendation-title">
                    <span>✅</span> Recommended Solution
                    <span style="font-size:12px; color:#555; font-weight:normal; margin-left:10px;">
                        (Confidence: {best_conf}%)
                    </span>
                </div>
                <div class="recommendation-text">
                    {recommendation_text}
                </div>
            </div>
        """, unsafe_allow_html=True)

        # --- 2. HISTORICAL CASES LIST ---
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("🔍 Similar Historical Cases")
        
        # Fields to display in the expander (Handles both CSV and PDF fields)
        DISPLAY_FIELDS = [
            "caseid", "subject", "Resolution", "details", # CSV
            "page_content", "source", "page"              # PDF
        ]

        for i, r in enumerate(results, 1):
            conf = round(r.get("confidence", 0), 2)
            label = "🟢 Best Match" if i == 1 else ""
            
            # Dynamic Header based on available data
            header_subject = r.get('subject') or r.get('source') or "Unknown Subject"
            header_id = r.get('caseid') or f"Doc-{i}"
            
            with st.expander(f"Case {i} {label} — {header_id} — {conf}%"):
                for field in DISPLAY_FIELDS:
                    if r.get(field):
                        st.write(f"**{field.capitalize()}:** {r[field]}")
                        
        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Logic: User View
# =========================
if run_clicked and query_mode == "User-Specific View":
    if not user_id.strip():
        st.warning("Please enter a Name.")
    else:
        insights = get_user_or_case_insights(user_id)
        if insights["data"] is None:
            st.error("No data found.")
        elif insights["type"] == "case":
            st.json(insights["data"])
        else:
            st.session_state.user_summary = insights["data"]
            st.session_state.active_owner = insights["data"]["owner"]

# =========================
# Dashboard Display
# =========================
if st.session_state.user_summary:
    s = st.session_state.user_summary
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader(f"👤 User Summary — {s['owner']}")
    
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Cases", s["total_cases"])
    m2.metric("Fresh (<7d)", s["pending_cases"])
    m3.metric("Overdue (8-21d)", s["overdue_cases"])
    m4.metric("Critical (>21d)", s["critical_cases"])

    # ✅ FIX 1: Defined 'owner' using the 's' summary dictionary
    recent_cases = get_recent_cases(s["owner"], days=5)

    st.metric(
        "Recent Cases (Last 5 Days)",
        len(recent_cases)
    )

    # =========================
    # ✅ Latest Resolved Cases (Top 3)
    # =========================
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

            header = f"✅ Case {case_id} | {subj}"

            with st.expander(header):
                st.write(f"**Reported On:** {reported_on}")
                st.write(f"**Closed On:** {closed_on}")
                st.write(f"**Resolution Time:** {round(float(rc.get('resolution_days', 0)), 2)} days")
                st.write(f"**Details:** {rc.get('details','')}")

                # -------------------------
                # 1) Gantt (Reported -> Closed)
                # -------------------------
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

                # -------------------------
                # 2) Effort Breakdown (Horizontal Bar)
                # -------------------------
                effort_df = pd.DataFrame({
                    "Effort Type": ["Configuration", "Testing", "Total"],
                    "Effort": [
                        float(rc.get("configurationeffort", 0) or 0),
                        float(rc.get("testingeffort", 0) or 0),
                        float(rc.get("totaleffort", 0) or 0)
                    ]
                })

                effort_chart = alt.Chart(effort_df).mark_bar().encode(
                    y=alt.Y("Effort Type:N", sort=["Configuration", "Testing", "Total"], title=""),
                    x=alt.X("Effort:Q", title="Effort"),
                    color=alt.Color("Effort Type:N", legend=None,
                        scale=alt.Scale(domain=["Configuration", "Testing", "Total"],
                                        range=["#2ecc71", "#f39c12", "#e74c3c"])
                    ),
                    tooltip=["Effort Type", "Effort"]
                ).properties(height=140)

                st.altair_chart(effort_chart, use_container_width=True)
    
    # CHARTS
    chart_col1, chart_col2 = st.columns(2)
    
    # 1. SMART PIE CHART (Status or Fallback)
    with chart_col1:
        st.markdown("#### Status Breakdown")
        
        # Prepare Data: Try Status first
        status_df = pd.DataFrame(list(s["status_breakdown"].items()), columns=["Label", "Count"])
        # Filter out "Unknown" or trivial statuses if necessary
        status_df_clean = status_df[status_df["Label"] != "Unknown"]

        # Decision: Use Status or Fallback to Aging?
        if not status_df_clean.empty and status_df_clean["Count"].sum() > 0:
            pie_df = status_df_clean
            legend_title = "Status"
        else:
            # Fallback: Use Aging buckets so chart is never empty
            pie_df = pd.DataFrame({
                "Label": ["Fresh (<7d)", "Overdue (8-21d)", "Critical (>21d)"],
                "Count": [s["pending_cases"], s["overdue_cases"], s["critical_cases"]]
            })
            legend_title = "Aging"

        # Remove zero counts to avoid messy chart
        pie_df = pie_df[pie_df["Count"] > 0]

        if pie_df.empty:
            st.info("No active cases available for charting.")
        else:
            # Create Plot
            fig1, ax1 = plt.subplots(figsize=(4, 3))
            wedges, texts, autotexts = ax1.pie(
                pie_df["Count"], 
                autopct='%1.0f%%', 
                startangle=90,
                textprops={'color':"white", 'weight':'bold'}
            )
            # Legend outside
            ax1.legend(wedges, pie_df["Label"], title=legend_title, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            ax1.axis('equal')
            fig1.patch.set_alpha(0)
            st.pyplot(fig1, use_container_width=False)

    # 2. BAR CHART (Altair)
    with chart_col2:
        st.markdown("#### Active Load Severity")
        aging_data = pd.DataFrame({
            "Category": ["Fresh (<7d)", "Overdue (8-21d)", "Critical (>21d)"],
            "Cases": [s["pending_cases"], s["overdue_cases"], s["critical_cases"]]
        })
        
        c = alt.Chart(aging_data).mark_bar().encode(
            x=alt.X('Category', sort=["Fresh (<7d)", "Overdue (8-21d)", "Critical (>21d)"]),
            y='Cases',
            color=alt.Color('Category', scale=alt.Scale(
                domain=["Fresh (<7d)", "Overdue (8-21d)", "Critical (>21d)"],
                range=['#2ecc71', '#f39c12', '#e74c3c']
            )),
            tooltip=['Category', 'Cases']
        ).properties(height=300)
        
        st.altair_chart(c, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("🆕 Recent Activity (Last 5 Days)")

    if not recent_cases:
        st.info("No cases reported in the last 5 days.")
    else:
        for c in recent_cases:
            with st.expander(f"Case {c['caseid']} | {c.get('subject','')}"):
                st.write(f"Reported On: {c['reportedon']}")
                st.write(f"Status: {c['statuscode']}")
                st.write(c.get("details", ""))

    
    # Focused List
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📌 Focused Case View")
    
    case_type = st.radio("Select category", ["Fresh (<7d)", "Overdue (8-21d)", "Critical (>21d)"], horizontal=True)
    
    if "Fresh" in case_type:
        cases = get_pending_cases(s["owner"], top_n=5)
        badge = "🟢 Fresh"
        empty_msg = "No fresh cases (<7 days) found."
    elif "Overdue" in case_type:
        cases = get_overdue_cases(s["owner"], top_n=3)
        badge = "🟠 Overdue"
        empty_msg = "No overdue cases (8-21 days) found."
    else:
        cases = get_critical_cases(s["owner"], top_n=3)
        badge = "🔴 Critical"
        empty_msg = "No critical cases (>21 days) found."
        
    if not cases:
        st.info(empty_msg)
    else:
        for c in cases:
            # Added "Subject" to the header for quick scanning
            header_text = f"{badge} | ID: {c['caseid']} | {c.get('subject', 'No Subject')}"
            
            with st.expander(header_text):
                st.write(f"**Aging:** {c['aging_num']} days")
                st.write(f"**Status:** {c['statuscode']}")
                st.write(f"**Reported On:** {c['reportedon']}")
                st.write(f"**Details:** {c.get('details','')}")

    st.markdown('</div>', unsafe_allow_html=True)