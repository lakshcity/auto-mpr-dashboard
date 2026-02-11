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

# ========================= # Imports # =========================
import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from services.retriever import find_similar_cases
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

# ========================= # Logic: General Search (INTEGRATED PDF/CSV LOGIC) # =========================
if run_clicked and query_mode == "General MPR Issue":
    if not query.strip():
        st.warning("Please enter an MPR issue.")
    elif index is None:
        st.error("Index not found. Please check data files.")
    else:
        with st.spinner("Searching similar past MPRs..."):
            # 🔧 FIX: retriever now manages its own model/index; pass only the query
            results = find_similar_cases(query)

            results = sorted(results, key=lambda x: x.get("confidence", 0), reverse=True)
            best_conf = round(results[0].get("confidence", 0), 2) if results else 0

            # --- 1. RECOMMENDED RESOLUTION (Integrated Feature)
            recommendation_text = build_recommendation(results)
            st.markdown(f"""
✅ **Recommended Solution** (Confidence: {best_conf}%)
{recommendation_text}
""")

            # --- 2. HISTORICAL CASES LIST
            st.markdown('\n', unsafe_allow_html=True)
            st.subheader("🔍 Similar Historical Cases")
            DISPLAY_FIELDS = [
                "caseid", "subject", "Resolution", "details",  # CSV
                "page_content", "source", "page"               # PDF
            ]
            for i, r in enumerate(results, 1):
                conf = round(r.get("confidence", 0), 2)
                label = "🟢 Best Match" if i == 1 else ""
                header_subject = r.get('subject') or r.get('source') or "Unknown Subject"
                header_id = r.get('caseid') or f"Doc-{i}"
                with st.expander(f"Case {i} {label} — {header_id} — {conf}%"):
                    for field in DISPLAY_FIELDS:
                        if r.get(field):
                            st.write(f"**{field.capitalize()}:** {r[field]}")

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
                st.write({
                    "Owner": s["owner"],
                    "Scope (cases)": s["total_cases"],
                    "Open": s["open_cases"],
                    "New (≤2 BD)": s["new_cases"]
                })
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
        st.write({
            "Owner": s["owner"],
            "Scope (cases)": s["total_cases"],
            "Open": s["open_cases"],
            "New (≤2 BD)": s["new_cases"]
        })
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