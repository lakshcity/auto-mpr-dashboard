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

# ========================= # Imports # =========================
import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from services.auth import verify_user

# --- FIX: SMART LOGO FINDER (moved above login usage)
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

# --- Auth state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# ========================= # Streamlit UI Config # =========================
st.set_page_config(page_title="Auto MPR Recommendation", layout="wide")

# ========================= # Load CSS (Merged Styles) # =========================
def load_css():
    try:
        css_path = BASE_DIR / "ui" / "style.css"
        if css_path.exists():
            with open(css_path, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        # Fail silently but don't break the app
        pass
    # (Keep placeholder styled blocks if you had them)
    st.markdown("", unsafe_allow_html=True)

load_css()

# =========================
# LOGIN (centered, with role dropdown)
# =========================
if not st.session_state.authenticated:

    login_col1, login_col2, login_col3 = st.columns([1, 2, 1])

    with login_col2:
        if LOGO_PATH:
            st.image(LOGO_PATH, width=220)
        st.markdown("## 🔐 Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        role_selected = st.selectbox(
            "Login As",
            ["consultant", "admin", "admin_analyst"]
        )

        login_clicked = st.button("Login", use_container_width=True)

        if login_clicked:

            role = verify_user(username.strip(), password.strip())

            if role:
                if role == role_selected:
                    st.session_state.authenticated = True
                    st.session_state.role = role
                    st.rerun()
                else:
                    st.error("Selected role does not match your assigned role.")
            else:
                st.error("Invalid credentials")

    # Stop rendering the rest of the app until user is authenticated
    st.stop()


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
from services.feedback_manager import (
    save_feedback,
    get_subject_success_rate,
    get_reward_trend,
    get_subject_feedback_count,
    get_feedback_stats,
    get_low_performing_subjects,
    get_weighted_subject_score
)
Path("data/case_index_master.faiss")

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
        return '#e74c3c'  # red
    elif percent < 75:
        return '#f39c12'  # amber
    else:
        return '#2ecc71'  # green

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
        {"label": "≤2", "start": 0, "end": min(2, max_bd), "color": "#2ecc71"},  # green
        {"label": "2–4", "start": 2, "end": min(4, max_bd), "color": "#1abc9c"},  # teal
        {"label": "4–7", "start": 4, "end": min(7, max_bd), "color": "#f1c40f"},  # yellow
        {"label": "7–14", "start": 7, "end": min(14, max_bd), "color": "#f39c12"},  # amber
        {"label": f"≥{min(14,max_bd)}", "start": 14, "end": max_bd, "color": "#e74c3c"},  # red
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
        st.image(LOGO_PATH, width=180)
    st.markdown("#### Auto MPR", unsafe_allow_html=True)

    if st.button("Clear Cache & Reset"):
        st.cache_resource.clear()
        st.rerun()

    if st.session_state.get("authenticated"):
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

# ========================= # Header # =========================
st.markdown("\n", unsafe_allow_html=True)
hcol1, hcol2 = st.columns([1,4])
with hcol1:
    if LOGO_PATH:
        st.image(LOGO_PATH, width=120)
with hcol2:
    st.markdown("## Auto MPR Response Recommendation")

# ========================= # Query Mode # =========================
st.markdown("\n", unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 2, 1])

role = st.session_state.role

if role == "consultant":
    available_modes = ["General MPR Issue"]

elif role == "admin":
    available_modes = ["General MPR Issue", "User-Specific View"]

elif role == "admin_analyst":
    available_modes = ["General MPR Issue", "User-Specific View", "Analytics Dashboard"]


query_mode = st.radio(
    "Query Mode",
    available_modes,
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

# =========================
# Inputs
# =========================
st.markdown('\n', unsafe_allow_html=True)
ci1, ci2, ci3 = st.columns([1, 2, 1])

with ci2:

    # -------------------------
    # General MPR Issue
    # -------------------------
    if query_mode == "General MPR Issue":

        query = st.text_area(
            "Enter MPR Issue",
            height=100,
            placeholder="Describe the issue..."
        )

        run_clicked = st.button("Run", use_container_width=True)


    # -------------------------
    # User-Specific View ONLY
    # -------------------------
    elif query_mode == "User-Specific View":

        owners = get_all_owners()
        selected_user = st.selectbox(
            "Select User",
            owners,
            index=0 if owners else None
        )

        fy = fq = fm = None

        if selected_user:

            ff1, ff2, ff3 = st.columns(3)

            with ff1:
                fy = st.selectbox(
                    "Financial Year (Apr–Mar)",
                    ["All", "FY24", "FY25", "FY26"]
                )

            with ff2:
                fq = st.selectbox(
                    "Financial Quarter",
                    ["All", "Q1", "Q2", "Q3", "Q4"]
                )

            with ff3:
                quarter_to_fm = {
                    "Q1": [1, 2, 3],
                    "Q2": [4, 5, 6],
                    "Q3": [7, 8, 9],
                    "Q4": [10, 11, 12]
                }

                if fq and fq != "All":
                    fm_options = ["All"] + quarter_to_fm.get(fq, list(range(1, 13)))
                else:
                    fm_options = ["All"] + list(range(1, 13))

                fm = st.selectbox(
                    "Financial Month (Apr=1 ... Mar=12)",
                    fm_options
                )

            if fq == "Q1":
                st.caption("Q1 months: Apr (1), May (2), Jun (3)")
            elif fq == "Q2":
                st.caption("Q2 months: Jul (4), Aug (5), Sep (6)")
            elif fq == "Q3":
                st.caption("Q3 months: Oct (7), Nov (8), Dec (9)")
            elif fq == "Q4":
                st.caption("Q4 months: Jan (10), Feb (11), Mar (12)")

            st.caption(":information_source: All metrics use **reported on** date and **exclude weekends**.")

        run_clicked = st.button("Run", use_container_width=True)

        # Keep compatibility variable
        user_id = ""


    # -------------------------
    # Analytics Dashboard
    # -------------------------
    elif query_mode == "Analytics Dashboard":

        # No user filters here
        run_clicked = True  # Always load analytics immediately


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

# --- Feedback state flag (initialize once) ---
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

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
                # Clear previous state if no results
                st.session_state.pop("similar_results", None)
                st.session_state.pop("recommendation_text", None)
                st.session_state.pop("last_query", None)
                # Ensure feedback stays disabled until a valid recommendation is generated again
                st.session_state.feedback_submitted = False
            else:
                # Sort by best available score once (adaptive > confidence)
                results = sorted(
                    results,
                    key=lambda x: x.get("adaptive_score", x.get("confidence", 0)),
                    reverse=True,
                )
                # Store results and last query
                st.session_state["similar_results"] = results
                st.session_state["last_query"] = query
                # Clear previous recommendation when new search happens
                st.session_state.pop("recommendation_text", None)
                # New query => allow feedback again once a recommendation is generated
                st.session_state.feedback_submitted = False

# =========================
# DISPLAY SIMILAR RESULTS (ONLY ONCE)
# =========================
if query_mode == "General MPR Issue" and "similar_results" in st.session_state:
    results = st.session_state["similar_results"]
    if results:
        st.subheader("🔍 Similar Historical Cases")

        # Render expandable cards for each result
        for i, r in enumerate(results, 1):
            # Prefer composite_confidence; fallback to confidence
            conf_val = r.get("composite_confidence", r.get("confidence", 0))
            try:
                conf = round(float(conf_val), 2)
            except Exception:
                conf = 0.0

            header_subject = r.get("mpr_subject") or "Unknown Subject"
            header_id = r.get("caseid") or f"Doc-{i}"

            with st.expander(f"Case {i} — {header_id} — {conf}%"):
                # Visual confidence bar
                try:
                    render_confidence_bar(conf)
                except Exception:
                    # Fail-safe textual display
                    st.write(f"**Similarity:** {conf}%")

                st.write(f"**Subject:** {header_subject}")
                if r.get("statuscode"):
                    st.write(f"**Status:** {r.get('statuscode')}")
                if r.get("reportedon"):
                    st.write(f"**Reported On:** {r.get('reportedon')}")

        # Visualize similarity distribution
        import pandas as pd
        import altair as alt

        scores = []
        for r in results:
            val = r.get("confidence", 0)
            try:
                scores.append(float(val))
            except Exception:
                scores.append(0.0)

        df_scores = pd.DataFrame({
            "Case Rank": list(range(1, len(scores) + 1)),
            "Similarity (%)": scores
        })

        st.markdown("### 📊 Similarity Distribution")
        chart = (
            alt.Chart(df_scores, title="Similarity across top retrieved cases")
            .mark_bar(color="#4E79A7")
            .encode(
                x=alt.X("Case Rank:O", title="Rank (1 = most similar)"),
                y=alt.Y("Similarity (%):Q", title="Similarity (%)", scale=alt.Scale(domain=[0, 100])),
                tooltip=[
                    alt.Tooltip("Case Rank:O", title="Rank"),
                    alt.Tooltip("Similarity (%):Q", title="Similarity (%)", format=".2f"),
                ],
            )
        )
        st.altair_chart(chart, use_container_width=True)

        # -------------------------
        # Recommendation Button
        # -------------------------
        if st.button("🚀 Get Recommendation", key="btn_get_reco"):
            # Safety guard in case results changed
            if results and isinstance(results, list):
                best_subject = results[0].get("mpr_subject", "") or ""
                if not best_subject.strip():
                    st.warning("Top result has no subject to query. Try another search.")
                else:
                    with st.spinner("Generating recommendation..."):
                        try:
                            recommendation_text = pdf_agent(best_subject)
                            st.session_state["recommendation_text"] = recommendation_text
                            # Reset feedback lock since this is a fresh recommendation
                            st.session_state.feedback_submitted = False
                        except Exception as e:
                            st.session_state["recommendation_text"] = f"PDF RAG Error: {str(e)}"
                            # Keep feedback locked off to avoid rating an error
                            st.session_state.feedback_submitted = True
            else:
                st.warning("No results available to generate a recommendation.")

# =========================
# DISPLAY RECOMMENDATION (PERSISTENT)
# =========================
if (
    query_mode == "General MPR Issue"
    and "recommendation_text" in st.session_state
    and st.session_state.get("recommendation_text")
):
    st.markdown("### ✅ Recommended Solution")

    # Reliability badge using historical feedback
    try:
        from services.feedback_manager import get_subject_success_rate
        best_subject_safe = ""
        if "similar_results" in st.session_state and st.session_state["similar_results"]:
            best_subject_safe = st.session_state["similar_results"][0].get("mpr_subject", "") or ""
        success_rate = get_subject_success_rate(best_subject_safe) if best_subject_safe else 0.0
    except Exception:
        success_rate = 0.0

    # Thresholds (>= logic)
    if success_rate >= 4:
        badge = "🟢 High Reliability"
    elif success_rate >= 2.5:
        badge = "🟡 Moderate Confidence"
    else:
        badge = "🔴 Needs Review"
    st.markdown(f"**Reliability:** {badge}")

    # The recommendation content
    st.markdown(st.session_state["recommendation_text"])

    # Explainability / provenance
    with st.expander("🔍 Why was this recommended?"):
        results = st.session_state.get("similar_results", [])
        st.write(f"Top Similar Cases Considered: {len(results)}")

        top_sim = 0.0
        if results:
            try:
                top_sim = float(results[0].get("confidence", 0))
            except Exception:
                top_sim = 0.0
        st.write(f"Top Case Similarity: {round(top_sim, 2)}%")
        st.write(f"Historical Success Score: {round(float(success_rate or 0), 2)} / 5")

        if results and "adaptive_score" in results[0]:
            st.write(f"Adaptive Score Used: {results[0]['adaptive_score']}")

        # Trend chart (if available)
        try:
            from services.feedback_manager import get_reward_trend
            trend_df = get_reward_trend()
            if trend_df is None or getattr(trend_df, "empty", True):
                st.info("Performance trend will appear after more feedback submissions.")
            elif len(trend_df) < 3:
                st.info("Performance trend will appear after more feedback submissions.")
            else:
                import altair as alt
                trend_chart = (
                    alt.Chart(trend_df, title="Rolling Average Reward Over Time")
                    .mark_line(color="#E15759")
                    .encode(
                        x=alt.X("timestamp:T", title="Time"),
                        y=alt.Y("rolling_avg:Q", title="Rolling Avg Reward"),
                        tooltip=[
                            alt.Tooltip("timestamp:T", title="Time"),
                            alt.Tooltip("rolling_avg:Q", title="Rolling Avg", format=".2f"),
                        ],
                    )
                )
                st.altair_chart(trend_chart, use_container_width=True)
        except Exception:
            # Silently ignore trend failures to avoid breaking UX
            pass

    # =========================
    # Feedback Form (Disabled after submission until new recommendation)
    # =========================
    st.markdown("### 📝 Rate This Recommendation")
    with st.form("feedback_form_general_issue", clear_on_submit=True):
        quality = st.slider("Quality (1–5)", 1, 5, 3, key="fb_quality")
        relevance = st.slider("Relevance (1–5)", 1, 5, 3, key="fb_relevance")
        clarity = st.slider("Clarity (1–5)", 1, 5, 3, key="fb_clarity")
        match = st.radio("Did it match historical cases?", ["Yes", "No"], index=0, key="fb_match")
        applicability = st.selectbox(
            "Applicability",
            ["Immediate", "Needs Customization", "Not Useful"],
            index=0,
            key="fb_applicability",
        )
        submit = st.form_submit_button(
            "Submit Feedback",
            disabled=st.session_state.feedback_submitted  # ✅ lock after submit
        )

        if submit:
            try:
                from services.feedback_manager import save_feedback
            except Exception as e:
                st.error(f"Feedback module not available: {e}")
                save_feedback = None

            if save_feedback is not None:
                # Guard best subject and similarity from session
                sr = st.session_state.get("similar_results", [])
                best_subject = sr[0].get("mpr_subject", "") if sr else ""
                try:
                    similarity_score = float(sr[0].get("confidence", 0)) if sr else 0.0
                except Exception:
                    similarity_score = 0.0

                try:
                    reward = save_feedback({
                        "mpr_subject": best_subject,
                        "similarity_score": similarity_score,
                        "quality_rating": int(quality),
                        "relevance_rating": int(relevance),
                        "clarity_rating": int(clarity),
                        "match_accuracy": match,
                        "applicability": applicability,
                    })
                    st.success(f"Feedback saved. Reward score: {reward}")
                    # ✅ lock feedback until new query/recommendation
                    st.session_state.feedback_submitted = True
                except Exception as e:
                    st.error(f"Failed to save feedback: {e}")
                    # Keep unlocked to allow retry
                    st.session_state.feedback_submitted = False

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

            # --- Executive Summary | Visuals | SLA in three columns (fresh run)
            # Precompute once for all columns
            sla_table = compute_sla_metrics_bd(filtered_df)
            resolved_in_period = int(filtered_df["closeddate"].notna().sum())
            total_cases = int(s["total_cases"])
            resolution_rate = round((resolved_in_period / total_cases) * 100, 2) if total_cases > 0 else 0.0

            exec_rows = [
                {"Metric": "Total Cases", "Value": total_cases},
                {"Metric": "Open Cases", "Value": int(s["open_cases"])},
                {"Metric": "Overdue (≥7 BD)", "Value": int(sla_table["overdue_active"])},
                {"Metric": "Critical (≥14 BD)", "Value": int(sla_table["critical_active"])},
                {"Metric": "Awaiting Input (>2 BD)", "Value": int(sla_table["awaiting_input"])},
                {"Metric": "Resolution Rate (%)", "Value": resolution_rate},
                {"Metric": "Avg Resolution Time (BD)", "Value": float(sla_table["avg_resolution"])},
            ]
            es_df = pd.DataFrame(exec_rows, columns=["Metric", "Value"])

            # Build visuals content pieces
            open_ct = int(s["open_cases"])
            overdue_ct = int(sla_table["overdue_active"])
            critical_ct = int(sla_table["critical_active"])
            awaiting_ct = int(sla_table["awaiting_input"])
            open_ratio = (open_ct / total_cases) if total_cases > 0 else 0

            # Color rules
            if open_ct == 0:
                open_color = '#2ecc71'
            elif open_ratio < 0.05:
                open_color = '#f39c12'
            else:
                open_color = '#e74c3c'
            overdue_color = '#2ecc71' if overdue_ct == 0 else '#f39c12'
            critical_color = '#2ecc71' if critical_ct == 0 else '#e74c3c'
            await_color = '#2ecc71' if awaiting_ct == 0 else '#e74c3c'

            rows_html = ""
            rows_html += _legend_row("Open Cases", _dot(open_color), f"{open_ct}")
            rows_html += _legend_row("Overdue (≥7 BD)", _dot(overdue_color), f"{overdue_ct}")
            rows_html += _legend_row("Critical (≥14 BD)", _dot(critical_color), f"{critical_ct}")
            rows_html += _legend_row("Awaiting Input (>2 BD)", _dot(await_color), f"{awaiting_ct}")
            visual_box = f"""
            <div style="border:1px solid #EEF0F3; border-radius:8px; background:#FFF; padding:6px;">
              {rows_html}
            </div>
            """

            # Three columns
            col_exec, col_visual, col_sla = st.columns([2, 1.2, 1.5])

            # Column 1 — Executive Summary
            with col_exec:
                st.markdown("### Executive Summary")
                st.dataframe(es_df, use_container_width=True)

            # Column 2 — Visuals
            with col_visual:
                st.markdown("### Visuals")
                components.html(visual_box, height=(4 * 36) + 20, scrolling=False)

                st.markdown("##### Resolution Rate")
                st.altair_chart(_resolution_ring_chart(resolution_rate, _rate_color(resolution_rate)), use_container_width=False)

                st.markdown("##### Avg Resolution Time (BD)")
                st.altair_chart(_avg_resolution_gauge(sla_table["avg_resolution"], max_bd=20), use_container_width=True)

            # Column 3 — SLA Metrics
            with col_sla:
                st.markdown("### SLA Metrics")
                sm1, sm2 = st.columns(2)
                sm1.metric("Active in SLA", sla_table["active_in_sla"],help="Active cases with ageing less than 7 business days")
                sm2.metric("Near Breach",  sla_table["near_breach"],help="Active cases with ageing between 5 and 7 business days(approaching SLA breach)")

                sm3, sm4 = st.columns(2)
                sm3.metric("SLA Breached",    sla_table["sla_breached"],help="Cases (active or closed) with ageing greater than 7 business days")
                sm4.metric("Compliance %",    sla_table["sla_compliance"],help="Percentage of cases closed within 7 business days")

                # Keep additional SLA KPIs for completeness (no LOC loss), but tuck them into an expander
                with st.expander("More SLA details"):
                    smx1, smx2, smx3 = st.columns(3)
                    smx1.metric("Overdue Active",   sla_table["overdue_active"])
                    smx2.metric("Critical Active",  sla_table["critical_active"])
                    smx3.metric("Awaiting Input",   sla_table["awaiting_input"])
                    smx4, smx5 = st.columns(2)
                    smx4.metric("Avg Resolution (BD)", sla_table["avg_resolution"])
                    smx5.metric("Total Cases", total_cases)

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
                "New (≤2 BD)": ("#1abc9c", "🟢 New"),
                "Fresh (<7 BD)": ("#2ecc71", "🟢 Fresh"),
                "Overdue (≥7 & <14 BD)": ("#f39c12", "🟠 Overdue"),
                "Critical (≥14 BD)": ("#e74c3c", "🔴 Critical"),
                "Open (All Active)": ("#3498db", "🔵 Open")
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

    # --- Executive Summary | Visuals | SLA in three columns (session reuse)
    sla_table = compute_sla_metrics_bd(filtered_df)
    resolved_in_period = int(filtered_df["closeddate"].notna().sum())
    total_cases = int(s["total_cases"])
    resolution_rate = round((resolved_in_period / total_cases) * 100, 2) if total_cases > 0 else 0.0

    exec_rows = [
        {"Metric": "Total Cases", "Value": total_cases},
        {"Metric": "Open Cases", "Value": int(s["open_cases"])},
        {"Metric": "Overdue (≥7 BD)", "Value": int(sla_table["overdue_active"])},
        {"Metric": "Critical (≥14 BD)", "Value": int(sla_table["critical_active"])},
        {"Metric": "Awaiting Input (>2 BD)", "Value": int(sla_table["awaiting_input"])},
        {"Metric": "Resolution Rate (%)", "Value": resolution_rate},
        {"Metric": "Avg Resolution Time (BD)", "Value": float(sla_table["avg_resolution"])},
    ]
    es_df = pd.DataFrame(exec_rows, columns=["Metric", "Value"])

    open_ct = int(s["open_cases"])
    overdue_ct = int(sla_table["overdue_active"])
    critical_ct = int(sla_table["critical_active"])
    awaiting_ct = int(sla_table["awaiting_input"])
    open_ratio = (open_ct / total_cases) if total_cases > 0 else 0

    if open_ct == 0:
        open_color = '#2ecc71'
    elif open_ratio < 0.05:
        open_color = '#f39c12'
    else:
        open_color = '#e74c3c'
    overdue_color = '#2ecc71' if overdue_ct == 0 else '#f39c12'
    critical_color = '#2ecc71' if critical_ct == 0 else '#e74c3c'
    await_color = '#2ecc71' if awaiting_ct == 0 else '#e74c3c'

    html = ""
    html += _legend_row("Open Cases", _dot(open_color), f"{open_ct}")
    html += _legend_row("Overdue (≥7 BD)", _dot(overdue_color), f"{overdue_ct}")
    html += _legend_row("Critical (≥14 BD)", _dot(critical_color), f"{critical_ct}")
    html += _legend_row("Awaiting Input (>2 BD)", _dot(await_color), f"{awaiting_ct}")
    visual_box = f"""
    <div style="border:1px solid #EEF0F3; border-radius:8px; background:#FFF; padding:6px;">
      {html}
    </div>
    """

    col_exec, col_visual, col_sla = st.columns([2, 1.2, 1.5])

    with col_exec:
        st.markdown("### Executive Summary")
        st.dataframe(es_df, use_container_width=True)

    with col_visual:
        st.markdown("### Visuals")
        components.html(visual_box, height=(4 * 36) + 20, scrolling=False)

        st.markdown("##### Resolution Rate")
        st.altair_chart(_resolution_ring_chart(resolution_rate, _rate_color(resolution_rate)), use_container_width=False)
        st.caption("Resolution Rate = (Resolved Cases / Total Cases) × 100")


        st.markdown("##### Avg Resolution Time (BD)")
        st.altair_chart(_avg_resolution_gauge(sla_table["avg_resolution"], max_bd=20), use_container_width=True)

    with col_sla:
        st.markdown("### SLA Metrics")
        sm1, sm2 = st.columns(2)
        sm1.metric("Active in SLA", sla_table["active_in_sla"])
        sm2.metric("Near Breach",  sla_table["near_breach"])

        sm3, sm4 = st.columns(2)
        sm3.metric("SLA Breached",    sla_table["sla_breached"])
        sm4.metric("Compliance %",    sla_table["sla_compliance"])

        with st.expander("More SLA details"):
            smx1, smx2, smx3 = st.columns(3)
            smx1.metric("Overdue Active",   sla_table["overdue_active"])
            smx2.metric("Critical Active",  sla_table["critical_active"])
            smx3.metric("Awaiting Input",   sla_table["awaiting_input"])
            smx4, smx5 = st.columns(2)
            smx4.metric("Avg Resolution (BD)", sla_table["avg_resolution"])
            smx5.metric("Total Cases", total_cases)

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
        st.caption("Distribution of active cases by business-day ageing buckets.")

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
        "New (≤2 BD)": ("#1abc9c", "🟢 New"),
        "Fresh (<7 BD)": ("#2ecc71", "🟢 Fresh"),
        "Overdue (≥7 & <14 BD)": ("#f39c12", "🟠 Overdue"),
        "Critical (≥14 BD)": ("#e74c3c", "🔴 Critical"),
        "Open (All Active)": ("#3498db", "🔵 Open")
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

# =========================
# Analytics Dashboard (Admin Analyst Only)
# =========================
if query_mode == "Analytics Dashboard" and role == "admin_analyst":

    from services.user_insights import add_business_ageing, compute_sla_metrics_bd
    from services.feedback_manager import get_reward_trend, get_feedback_stats

    st.markdown("## 📊 Admin Analytics Control Tower")

    try:
        full_df = pd.DataFrame(metadata)
    except:
        st.error("Metadata not available.")
        st.stop()

    if full_df.empty:
        st.warning("No data available.")
        st.stop()

    # -------------------------
    # Normalize + Business-Day Ageing
    # -------------------------
    full_df["reportedon"] = pd.to_datetime(full_df.get("reportedon"), errors="coerce")
    full_df["closeddate"] = pd.to_datetime(full_df.get("closeddate"), errors="coerce")
    full_df = add_business_ageing(full_df)

    total_cases = len(full_df)
    active_cases = full_df["is_open"].sum()
    closed_cases = total_cases - active_cases

    # -------------------------
    # SLA Metrics (BD Based)
    # -------------------------
    sla_metrics = compute_sla_metrics_bd(full_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cases", total_cases)
    col2.metric("Active Cases", active_cases)
    col3.metric("Closed Cases", closed_cases)
    col4.metric("SLA Compliance (%)", sla_metrics["sla_compliance"])

    col5, col6 = st.columns(2)
    col5.metric("Avg Resolution (BD)", sla_metrics["avg_resolution"])
    col6.metric("Awaiting Input (>2BD)", sla_metrics["awaiting_input"])

    st.markdown("---")

    # =========================
    # SLA Distribution (BD-Based)
    # =========================
    st.markdown("### 🚦 SLA Health (Business-Day Based)")

    sla_dist_df = pd.DataFrame({
        "Category": [
            "Active <7BD",
            "Near Breach (5–7BD)",
            "Overdue Active (7–14BD)",
            "Critical Active (≥14BD)",
            "Total Breached (>7BD)"
        ],
        "Count": [
            sla_metrics["active_in_sla"],
            sla_metrics["near_breach"],
            sla_metrics["overdue_active"],
            sla_metrics["critical_active"],
            sla_metrics["sla_breached"]
        ]
    })

    sla_chart = alt.Chart(sla_dist_df).mark_bar().encode(
        x="Category",
        y="Count"
    )
    st.altair_chart(sla_chart, use_container_width=True)

    # =========================
    # Case Inflow Trend
    # =========================
    st.markdown("### 📈 Case Inflow Trend")

    trend_df = (
        full_df
        .groupby(full_df["reportedon"].dt.date)
        .size()
        .reset_index(name="cases")
    )

    trend_chart = alt.Chart(trend_df).mark_line(point=True).encode(
        x="reportedon:T",
        y="cases:Q"
    )
    st.altair_chart(trend_chart, use_container_width=True)

    # =========================
    # Resolution Time Distribution (BD)
    # =========================
    st.markdown("### ⏱ Resolution Time Distribution (Closed Cases - BD)")

    closed_df = full_df[~full_df["is_open"]]

    if not closed_df.empty:
        hist = alt.Chart(closed_df).mark_bar().encode(
            alt.X("ageing_bd:Q", bin=True),
            y="count()"
        )
        st.altair_chart(hist, use_container_width=True)
    else:
        st.info("No closed cases available.")

    # =========================
    # User Performance (Resolution Rate)
    # =========================
    if "currentowner" in full_df.columns:

        st.markdown("### 👥 Top 10 Performers (Resolution %)")

        user_perf = (
            full_df.groupby("currentowner")
            .agg(
                total=("currentowner", "count"),
                closed=("closeddate", lambda x: x.notna().sum())
            )
        )

        user_perf["resolution_rate"] = (
            user_perf["closed"] / user_perf["total"] * 100
        ).round(2)

        user_perf = user_perf.sort_values("resolution_rate", ascending=False).head(10)

        chart = alt.Chart(user_perf.reset_index()).mark_bar().encode(
            x="resolution_rate:Q",
            y="currentowner:N"
        )
        st.altair_chart(chart, use_container_width=True)

    # =========================
    # Status Distribution
    # =========================
    st.markdown("### 🧩 Status Distribution")

    status_df = (
        full_df["statuscode"]
        .astype(str)
        .value_counts()
        .reset_index()
    )
    status_df.columns = ["Status", "Count"]

    pie = alt.Chart(status_df).mark_arc().encode(
        theta="Count",
        color="Status"
    )
    st.altair_chart(pie, use_container_width=True)

    # =========================
    # Subject Intelligence
    # =========================
    if "mpr_subject" in full_df.columns:
        st.markdown("### 📚 Top 10 Recurring Subjects")

        subject_df = (
            full_df["mpr_subject"]
            .astype(str)
            .value_counts()
            .head(10)
            .reset_index()
        )
        subject_df.columns = ["Subject", "Count"]

        subject_chart = alt.Chart(subject_df).mark_bar().encode(
            x="Count:Q",
            y="Subject:N"
        )
        st.altair_chart(subject_chart, use_container_width=True)

    

    # =========================
    # AI Learning Intelligence
    # =========================
    
   
    st.markdown("---")
    st.markdown("## 🤖 AI Learning Intelligence")

    stats = get_feedback_stats()
    reward_trend = get_reward_trend()

    colA, colB, colC, colD = st.columns(4)

    colA.metric("Total Feedback Entries", stats["total_feedback"])
    colB.metric("Global Avg Reward", stats["global_avg_reward"])
    colC.metric("Model Stability (Std Dev)", stats["reward_std"])
    colD.metric("Learning Velocity", stats["learning_velocity"])

    if not reward_trend.empty:
        reward_chart = alt.Chart(reward_trend).mark_line(point=True).encode(
            x="timestamp:T",
            y="rolling_avg:Q"
        )
        st.altair_chart(reward_chart, use_container_width=True)

    # Subject Reliability Leaderboard
    st.markdown("### 🏆 Subject Reliability Leaderboard")

    feedback_df = stats["raw_df"]

    if not feedback_df.empty:
        subject_stats = (
            feedback_df.groupby("mpr_subject")
            .agg(
                count=("quality_rating", "count"),
                avg_reward=("final_reward", "mean")
            )
            .reset_index()
        )

        subject_stats = subject_stats[subject_stats["count"] >= 3]
        subject_stats = subject_stats.sort_values("avg_reward", ascending=False).head(10)

        chart = alt.Chart(subject_stats).mark_bar().encode(
            x="avg_reward:Q",
            y="mpr_subject:N"
        )

        st.altair_chart(chart, use_container_width=True)


    # =========================
    # Risk Alerts
    # =========================

    st.markdown("---")
    st.markdown("## ⚠ Risk Intelligence Engine")

    risk_score = 0

    if sla_metrics["critical_active"] > 0:
        st.error(f"{sla_metrics['critical_active']} cases in CRITICAL SLA band (≥14 BD).")
        risk_score += 2

    if sla_metrics["near_breach"] > 15:
        st.warning("High Near-Breach Volume Detected.")
        risk_score += 1

    if stats["reward_std"] > 1.5:
        st.warning("AI Reward Variance High → Model unstable.")
        risk_score += 1

    if stats["learning_velocity"] < 0:
        st.warning("Model Performance Declining.")
        risk_score += 1

    st.metric("Composite Risk Score", risk_score)

