import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
import sys, os
import hashlib
from openai import APITimeoutError

#  To Run : python -m streamlit run app.py --server.port 8501

sys.path.insert(0, os.path.dirname(__file__))
from utils.db import (
    init_db, load_data, insert_entry, delete_entry,
    export_csv
)
from utils.ai import (
    compute_scores, summarize_logs, generate_checked_result,
)
from utils.rag import (
    KB_PATH, load_passages, prep_documents, get_embeddings_function,
    build_vector_store, retrieve_relevant_passages,
)
from utils.grounding import (
    bounded_history, resolve_question, reference_resolution_reason, log_summary_fields,
    describe_rate_limit, describe_validation_error,
)
from utils.dashboard import dashboard_view, shift_period, trend_charts, can_advance

st.set_page_config(
    page_title="NutriRecall",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  :root {
    --nr-border: #29232f;
    --nr-panel: #100d15;
    --nr-muted: #b5afbf;
  }
  [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: #050505;
  }
  [data-testid="stMainBlockContainer"] {
    padding-top: 3rem;
    padding-bottom: 3rem;
    max-width: 1440px;
  }
  h1, h2 { font-size: 1.75rem !important; font-weight: 600 !important; }
  h3 { font-size: 1rem !important; font-weight: 600 !important; }
  [data-testid="stCaptionContainer"] { color: var(--nr-muted); }
  hr { border-color: var(--nr-border); margin: 1.5rem 0; }
  [data-testid="stMetric"] {
    background: var(--nr-panel);
    border: 1px solid var(--nr-border);
    border-radius: 10px;
    padding: 1rem;
    height: 100%;
    min-height: 9rem;
  }
  [data-testid="stMetricLabel"] p { font-size: 0.875rem; color: var(--nr-muted); }
  [data-testid="stMetricValue"] { font-size: 1.75rem; font-weight: 600; }
  .st-key-dashboard_workouts [data-testid="stMetricValue"],
  .st-key-dashboard_workouts [data-testid="stMetricValue"] p {
    font-size: 1rem;
    white-space: normal;
  }
  [data-testid="stMetricDelta"] {
    color: var(--nr-muted) !important;
    background: transparent !important;
  }
  [data-testid="stPlotlyChart"] {
    background: var(--nr-panel);
    /* Keep the outline inside Streamlit's fixed chart dimensions. */
    border: 0;
    box-shadow: inset 0 0 0 1px var(--nr-border);
    border-radius: 10px;
  }
  [data-testid="stButton"] button[kind="secondary"],
  [data-testid="stDownloadButton"] button {
    background: var(--nr-panel);
    border: 1px solid #44394e;
    border-radius: 8px;
    color: #eeeaf2;
  }
  [data-testid="stButton"] button[kind="secondary"]:hover,
  [data-testid="stDownloadButton"] button:hover {
    background: #21132e;
    border-color: #00e5ff;
    color: #eeeaf2;
  }
  [data-testid="stSidebar"] {
    background: #0b090e;
    border-right: 1px solid var(--nr-border);
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 0.5rem 0.625rem;
    border: 1px solid transparent;
    border-radius: 8px;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: #231130;
    border-color: #453052;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #17101e;
  }
  .score-green { color: #22c55e; font-weight: 600; }
  .score-amber { color: #f59e0b; font-weight: 600; }
  .score-red   { color: #ef4444; font-weight: 600; }

  /* Compact sidebar spacing without changing text or control sizes. */
  [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-bottom: 2rem;
  }
  [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.75rem;
  }
  [data-testid="stSidebar"] h1 {
    font-size: 1rem !important;
    padding-top: 0.75rem;
    padding-bottom: 0.5rem;
  }
  [data-testid="stSidebar"] hr {
    margin-top: 1rem;
    margin-bottom: 1rem;
  }
</style>
""", unsafe_allow_html=True)


def show_chart(fig):
    """Apply the shared visual theme without changing plotted values or ranges."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Arial, sans-serif", size=14, color="#EEEAF2"),
        colorway=["#00E5FF", "#9572B5"],
        hoverlabel=dict(bgcolor="#21132E", font_color="#EEEAF2"),
        legend=dict(font_size=14),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#29232F", automargin=True)
    fig.update_yaxes(gridcolor="#29232F", zeroline=False, linecolor="#29232F", automargin=True)
    st.plotly_chart(fig, use_container_width=True, theme=None,
                    config={
                        "displaylogo": False,
                        "displayModeBar": "hover",
                        "modeBarButtonsToRemove": [
                            "zoom2d", "pan2d", "select2d", "lasso2d",
                            "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d",
                        ],
                    })


def navigate_dashboard(step):
    if step > 0 and not can_advance(st.session_state.dashboard_period,
                                   st.session_state.dashboard_anchor):
        return
    anchor = shift_period(st.session_state.dashboard_period,
                          st.session_state.dashboard_anchor, step)
    st.session_state.dashboard_anchor = anchor
    st.session_state.dashboard_picker = anchor


def current_dashboard_week():
    st.session_state.dashboard_period = "Week"
    st.session_state.dashboard_anchor = date.today()
    st.session_state.dashboard_picker = date.today()


def pick_dashboard_date():
    st.session_state.dashboard_anchor = st.session_state.dashboard_picker


def reset_conversation():
    st.session_state.rag_chat_history = []
    for key in ("rag_pending_query", "rag_pending_safety", "rag_weight_help"):
        st.session_state.pop(key, None)


with st.sidebar:
    st.title("❤️ NutriRecall")
    st.caption("Your smart health assistant")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📥 Daily Log", "📊 Dashboard", "📅 History", "🔄 Week Compare", "✨ AI Assistant"],
        label_visibility="collapsed",
    )

    st.divider()
    with st.expander("AI settings & data sharing"):
        use_local = st.toggle("Use local Ollama model", value=False)
        local_model = "phi3:mini"
        if not use_local:
            groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
        else:
            groq_key = ""
            local_model = st.text_input("Ollama model tag", value="phi3:mini",
                                        help="Use the exact installed tag from ollama list on your other machine.")
        include_logs = st.toggle("Include my recent logs", value=True, key="rag_include_logs",
                                 on_change=reset_conversation,
                                 help="Changing sharing starts a new conversation so old log figures are not reused.")
        summary_end = st.date_input("Summary ending on", value=date.today(),
                                    max_value=date.today(), key="rag_summary_end")
        st.caption(
            "Cloud requests share your question, up to three completed conversation turns, "
            "and public evidence with Groq. Fresh log figures are rendered locally; earlier figures can "
            "appear in session history. No permanent chat memory. Contextual follow-ups can add "
            "a reference-resolution call. Disable log sharing here or select local Ollama."
        )

    if page == "✨ AI Assistant":
        with st.expander("Help & safety", expanded=False):
            st.caption("Optional curated guidance; not automatic detection or general medical triage.")
            if st.button("Sudden or unexplained weight loss", key="rag_weight_help_button"):
                st.session_state.rag_weight_help = True
                st.rerun()
            if st.session_state.get("rag_weight_help"):
                st.info(
                    "Is this about sudden or unexplained weight loss you are experiencing? "
                    "Confirm to read the curated guidance without a model call."
                )
                if st.button("Show curated guidance", key="rag_weight_confirm"):
                    st.session_state.rag_pending_query = "Show me the guidance for sudden or unexplained weight loss."
                    st.session_state.rag_pending_safety = True
                    st.session_state.rag_weight_help = False
                    st.rerun()
                if st.button("Cancel", key="rag_weight_cancel"):
                    st.session_state.rag_weight_help = False
                    st.rerun()

    st.divider()
    data_all = load_data()
    if not data_all.empty:
        csv_bytes = export_csv()
        st.download_button("⬇️ Export CSV", csv_bytes, "nutrirecall_export.csv", "text/csv")


init_db()
data = load_data()

if not data.empty:
    try:
        data = compute_scores(data)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        data = pd.DataFrame()


def get_week(df, offset_weeks=0):
    today = pd.Timestamp(date.today())
    end   = today - timedelta(weeks=offset_weeks)
    start = end   - timedelta(days=6)
    mask  = (df["date"] >= start) & (df["date"] <= end)
    return df[mask]


@st.cache_resource(show_spinner=False)
def get_knowledge_store(kb_signature):
    """Cache public knowledge resources only, never keys or personal logs.

    The content hash invalidates the cache when the curated file changes.
    Initialization is lazy: ordinary tracking pages do not load it.
    """
    documents = prep_documents(load_passages())
    embeddings = get_embeddings_function()
    return build_vector_store(documents, embeddings)


def render_assistant_result(result):
    if result.get("response_mode") == "confirmed_safety":
        st.caption("Curated guidance — selected through your explicit confirmation.")
    if result.get("citation_warning"):
        st.warning(result["citation_warning"])
    if result.get("grounding", {}).get("provenance_issues"):
        if result["grounding"].get("selection_contract") == "source_units_v1":
            st.warning("An invalid evidence selection was removed. See review details.")
        else:
            st.warning("Some draft content could not be matched to its evidence and was removed. See review details.")
    st.markdown(result["answer"])
    if result.get("source_review_status"):
        st.caption(result["source_review_status"])
    if result.get("source_footer"):
        with st.expander("Sources and review notes"):
            st.markdown(result["source_footer"])
    if result.get("evidence_used"):
        with st.expander("Exact evidence"):
            st.markdown(result["evidence_used"])
    if result.get("log_header"):
        usage = result.get("log_usage", {})
        if usage.get("rendered_fields"):
            label = "Logs used for this answer"
            detail = "These fields were rendered from the recorded summary: " + ", ".join(usage["rendered_fields"]) + "."
        elif usage.get("fresh_summary_sent_to_model") is True:
            label = "Log context sent with this request"
            detail = "The fresh summary was sent to the model. The final answer did not display any record fields."
        else:
            label = "Log coverage snapshot (usage not recorded)"
            detail = "This older response did not record whether its summary was sent or rendered."
        with st.expander(label):
            st.caption(detail)
            st.markdown(result["log_header"])
            st.caption("Sharing details here refer to the fresh summary. Session history can contain earlier log figures.")
    if result.get("grounding"):
        with st.expander("Review details"):
            if result["grounding"].get("support_review") == "deterministic_log_summary":
                st.caption("Log values and coverage were rendered by code. No model or knowledge passages were used.")
            elif result["grounding"].get("selection_contract") == "source_units_v1":
                st.caption("Text and food-list relationships come from source units. Valid indices do not prove relevance; model selection can still be wrong.")
            elif result["grounding"].get("support_review"):
                st.caption("Citation/quote checks are mechanical. Model support review can still be wrong.")
            st.json(result["grounding"])


if page == "📥 Daily Log":
    st.header("📥 Daily Entry")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        log_date = st.date_input("Date", value=date.today())
    with col2:
        weight = st.number_input("Weight (kg)", min_value=0.0, value=None, placeholder="e.g. 72.5", step=1.0)
    with col3:
        protein = st.number_input("Protein (g)", min_value=0.0, value=None, placeholder="e.g. 120", step=1.0)
    with col4:
        sleep = st.number_input("Sleep (hrs)", min_value=0.0, max_value=24.0, value=None, placeholder="e.g. 7.5", step=1.0)

    workout = st.selectbox("Did you work out today?", ["No", "Yes"])

    c1, c2 = st.columns([5, 1])
    with c1:
        if st.button("💾 Save Entry", use_container_width=True):
            if weight is None or protein is None or sleep is None:
                st.error("Please fill in all fields before saving.")
            elif weight <= 0:
                st.error("Weight must be greater than zero. Protein and sleep may be zero.")
            else:
                date_str = str(log_date)
                ok = insert_entry(date_str, weight, protein, sleep, 1 if workout == "Yes" else 0)
                if ok:
                    st.success(f"✅ Entry saved for {log_date.strftime('%d %b %Y')}!")
                    st.rerun()
                else:
                    st.warning(f"An entry for {log_date} already exists. Go to History to edit it.")
    with c2:
        delete_button = st.empty()
        last_entry = None if data.empty else data.loc[data["id"].idxmax()]
        confirm_last = False
        if last_entry is not None:
            confirm_last = st.checkbox(
                f"Confirm deletion of {last_entry['date']:%Y-%m-%d} (last added entry)",
                key=f"confirm_last_{int(last_entry['id'])}",
            )
        if delete_button.button("Delete Last Entry", disabled=not confirm_last, use_container_width=True):
            if last_entry is not None:
                delete_entry(int(last_entry["id"]))
                st.warning("Last entry deleted.")
                st.rerun()

    # Quick stats below the form
    if not data.empty:
        st.divider()
        st.subheader("Today's context")
        w7 = get_week(data, offset_weeks=0)
        if not w7.empty:
            avg_p  = w7["protein"].mean()
            avg_pr = w7["protein_required"].mean()
            avg_s  = w7["sleep_hours"].mean()
            wo     = int(w7["workout"].sum())
            hs     = w7["health_score_10"].mean()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg Protein (7d)", f"{avg_p:.0f}g", f"need {avg_pr:.0f}g")
            m2.metric("Avg Sleep (7d)", f"{avg_s:.1f} hrs")
            m3.metric("Workout Days (7d)", f"{wo}/7")
            score_color = "score-green" if hs >= 7 else ("score-amber" if hs >= 5 else "score-red")
            m4.metric("Health Score", f"{hs:.1f}/10")



elif page == "📊 Dashboard":
    st.header("📊 Dashboard")
    today = date.today()
    latest = pd.to_datetime(data["date"]).max().date() if not data.empty else today
    st.session_state.setdefault("dashboard_anchor", min(latest, today))
    st.session_state.dashboard_anchor = min(st.session_state.dashboard_anchor, today)
    if st.session_state.get("dashboard_picker", today) > today:
        st.session_state.dashboard_picker = today
    st.radio("View", ["Week", "Month", "Year", "All time"],
             horizontal=True, key="dashboard_period")
    mode = st.session_state.dashboard_period
    if mode != "All time":
        previous, next_period = st.columns(2)
        previous.button("← Previous", key="dashboard_previous", on_click=navigate_dashboard,
                        args=(-1,), use_container_width=True)
        next_period.button("Next →", key="dashboard_next", on_click=navigate_dashboard,
                           args=(1,), use_container_width=True,
                           disabled=not can_advance(mode, st.session_state.dashboard_anchor, today))
        st.date_input("Date in selected period", value=st.session_state.dashboard_anchor,
                      key="dashboard_picker", max_value=today, on_change=pick_dashboard_date,
                      help="Choose any date within the week, month or year to view. Weeks run Monday–Sunday.")
    st.button("Go to current week", on_click=current_dashboard_week, key="dashboard_current")
    view = dashboard_view(data, mode, st.session_state.dashboard_anchor, today=today)
    if view["start"] is not None:
        st.markdown(f"**{mode}: {view['start']:%d %b %Y} – {view['end']:%d %b %Y}**")
        st.caption(
            f"Coverage: {view['coverage_logged_days']} of {view['coverage_days']} days logged; "
            f"{view['missing_days']} days unlogged. "
            "Averages use recorded days only; unlogged days are unknown."
        )
    else:
        st.markdown("**All time: no recorded dates**")
        st.caption("Coverage: 0 logged days; no recorded date range.")
    if view["rows"].empty:
        if mode == "All time":
            st.info("No entries yet. Add an entry in Daily Log, or select Week, Month or Year "
                    "to browse calendar periods.")
        else:
            st.info("No entries in this period. Use Previous, Next or the date picker to browse another "
                    "period, choose All time, or add an entry in Daily Log.")
    else:
        avg_p = view["means"]["protein"]
        avg_pr = view["means"]["protein_required"]
        avg_s = view["means"]["sleep_hours"]
        hs = view["means"]["health_score_10"]
        m1, m2, m3, m4 = st.columns(4)
        delta_p = round(avg_p - avg_pr, 1)
        m1.metric("Avg Protein", f"{avg_p:.0f}g", f"{delta_p:+.0f}g vs target")
        m2.metric("Protein Target", f"{avg_pr:.0f}g", "based on weight")
        m3.metric("Avg Sleep", f"{avg_s:.1f} hrs", f"{'✓' if avg_s >= 7 else '⚠ below 7'}")
        with m4.container(key="dashboard_workouts"):
            st.metric("Workout days", f"{view['workout_days']} of {view['logged_days']} logged days")

        st.subheader("Health Score")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(hs, 1),
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [1, 10], "tickwidth": 1},
                "bar": {"color": "#00E5FF"},
                "steps": [
                    {"range": [1, 5],  "color": "#24172F"},
                    {"range": [5, 7],  "color": "#3B0A6B"},
                    {"range": [7, 10], "color": "#63417E"},
                ],
                "threshold": {"line": {"color": "#EEEAF2", "width": 2}, "thickness": 0.75, "value": hs},
            },
            number={"suffix": "/10", "font": {"size": 28}},
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=20, b=20, l=40, r=40))
        show_chart(fig_gauge)

        st.divider()
        if view["monthly"]:
            st.caption("Charts: monthly averages from recorded days only; hover for logged-day counts. "
                       "Headline metrics average the recorded days directly, not the monthly averages.")
        else:
            st.caption("Charts: daily readings in the selected range. Gaps are unlogged days, not zeros. "
                       "All time switches to monthly averages for spans longer than 366 days.")
        fig_w, fig_p, fig_s = trend_charts(view)
        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Weight progress")
            show_chart(fig_w)
        with col_r:
            st.subheader("Protein vs Target")
            show_chart(fig_p)
        st.subheader("Sleep duration")
        show_chart(fig_s)


elif page == "📅 History":
    st.header("📅 Log History")

    if data.empty:
        st.info("No entries yet. Start logging in Daily Log.")
    else:
        display = data.copy()
        display["date"] = display["date"].dt.strftime("%Y-%m-%d")
        display["workout"] = display["workout"].map({1: "✅ Yes", 0: "❌ No"})
        display["health_score_10"] = display["health_score_10"].round(1)
        display["protein_required"] = display["protein_required"].round(1)
        display = display.rename(columns={
            "date": "Date", "weight": "Weight (kg)", "protein": "Protein (g)",
            "sleep_hours": "Sleep (hrs)", "workout": "Workout",
            "protein_required": "Target Protein (g)", "health_score_10": "Health Score",
        })

        show_cols = ["Date", "Weight (kg)", "Protein (g)", "Target Protein (g)",
                     "Sleep (hrs)", "Workout", "Health Score"]

        st.dataframe(
            display[show_cols].sort_values("Date", ascending=False).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            height=420,
        )

        st.divider()
        st.subheader("Edit or delete an entry")

        raw = data.copy()
        raw["label"] = raw["date"].dt.strftime("%Y-%m-%d") + "  |  " + raw["weight"].astype(str) + "kg"
        selected_label = st.selectbox("Select entry", raw["label"].tolist()[::-1])
        selected_row = raw[raw["label"] == selected_label].iloc[0]

        ec1, ec2, ec3, ec4 = st.columns(4)
        new_w  = ec1.number_input("Weight (kg)", min_value=0.0, value=float(selected_row["weight"]), step=1.0)
        new_p  = ec2.number_input("Protein (g)", min_value=0.0, value=float(selected_row["protein"]), step=1.0)
        new_sl = ec3.number_input("Sleep (hrs)", min_value=0.0, max_value=24.0, value=float(selected_row["sleep_hours"]), step=1.0)
        new_wo = ec4.selectbox("Workout", ["Yes", "No"],
                               index=0 if selected_row["workout"] == 1 else 1)
        confirm_delete = st.checkbox(
            f"Confirm deletion of {selected_row['date']:%Y-%m-%d}",
            key=f"confirm_entry_{int(selected_row['id'])}",
        )

        b1, b2, _ = st.columns([1, 1, 4])
        with b1:
            if st.button("💾 Save changes", use_container_width=True):
                if new_w <= 0:
                    st.error("Weight must be greater than zero. Protein and sleep may be zero.")
                else:
                    from utils.db import update_entry
                    update_entry(
                        int(selected_row["id"]), new_w, new_p, new_sl,
                        1 if new_wo == "Yes" else 0,
                    )
                    st.success("Entry updated!")
                    st.rerun()
        with b2:
            if st.button("Delete entry", disabled=not confirm_delete, use_container_width=True):
                delete_entry(int(selected_row["id"]))
                st.warning("Entry deleted.")
                st.rerun()


elif page == "🔄 Week Compare":
    st.header("🔄 Week-over-Week Comparison")

    if data.empty:
        st.info("Add entries to compare weeks.")
    else:
        this_w = get_week(data, offset_weeks=0)
        last_w = get_week(data, offset_weeks=1)
        if this_w.empty or last_w.empty:
            st.info("Entries are needed in both 7-day periods to calculate changes. Missing data is not zero.")

        metrics = ["protein", "sleep_hours", "workout", "health_score_10"]
        labels  = ["Avg Protein (g)", "Avg Sleep (hrs)", "Workout Days", "Health Score"]

        def week_avg(df, col):
            if df.empty:
                return None
            if col == "workout":
                return float(df[col].sum())
            return float(df[col].mean())

        this_vals = [week_avg(this_w, m) for m in metrics]
        last_vals = [week_avg(last_w, m) for m in metrics]

        # Summary cards
        cols = st.columns(4)
        for i, (col, label, tv, lv) in enumerate(zip(cols, labels, this_vals, last_vals)):
            delta = None if tv is None or lv is None else tv - lv
            fmt = ".0f" if i in [0, 2] else ".1f"
            col.metric(label, "No data" if tv is None else f"{tv:{fmt}}",
                       None if delta is None else f"{delta:+{fmt}} vs last week")

        st.divider()

        
        fig = go.Figure()
        fig.add_trace(go.Bar(name="This week", x=labels, y=[None if v is None else round(v, 1) for v in this_vals],
                             marker_color="#00E5FF"))
        fig.add_trace(go.Bar(name="Last week", x=labels, y=[None if v is None else round(v, 1) for v in last_vals],
                             marker_color="#9572B5"))
        fig.update_layout(
            barmode="group", height=350,
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", y=1.1),
        )
        show_chart(fig)

        
        if not this_w.empty:
            st.subheader("This week — daily protein vs target")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=this_w["date"].dt.strftime("%a %d"),
                y=this_w["protein"],
                name="Protein (g)", marker_color="#00E5FF",
            ))
            fig2.add_trace(go.Scatter(
                x=this_w["date"].dt.strftime("%a %d"),
                y=this_w["protein_required"],
                mode="lines+markers", name="Target",
                line=dict(color="#B5AFBF", dash="dash"),
            ))
            fig2.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10),
                               hovermode="x unified")
            show_chart(fig2)


elif page == "✨ AI Assistant":
    st.header("✨ NutriRecall AI")
    st.caption("Personal wellness chat · Experimental — see answer quality and privacy details.")
    with st.expander("About answer quality"):
        st.write(
            "Health guidance displays selected curated passages with their qualifications; "
            "personal figures are calculated from fresh logs. Evidence selection is fallible; "
            "earlier evaluations exposed grounding defects. Sources, exact evidence and "
            "review details remain inspectable below answers. This is not medical diagnosis."
        )

    log_summary = None
    if include_logs:
        logs = []
        if not data.empty:
            summary_data = data[["date", "weight", "protein", "sleep_hours", "workout"]].copy()
            summary_data["date"] = summary_data["date"].dt.strftime("%Y-%m-%d")
            logs = summary_data.to_dict("records")
        try:
            log_summary = summarize_logs(logs, as_of=summary_end.isoformat())
        except (ValueError, TypeError, KeyError):
            st.error("Could not summarize these logs. Check their dates and values in History.")
            st.stop()
        st.markdown(
            f"**Available logs: {log_summary['window_start']} – {log_summary['window_end']} "
            f"· {log_summary['logged_days']}/7 days recorded**"
        )
        st.caption("Logged dates: " + (", ".join(log_summary["logged_dates"]) or "none")
                   + ". Missing days are unknown, not zero.")
    else:
        st.caption("Log sharing is off. Change it under AI settings & data sharing.")

    if st.button("New conversation", key="rag_new"):
        reset_conversation()
        st.rerun()

    st.session_state.setdefault("rag_chat_history", [])
    for msg in st.session_state.rag_chat_history:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "✨"):
            if "result" in msg:
                render_assistant_result(msg["result"])
            elif msg.get("error"):
                st.error(msg["content"])
            else:
                st.markdown(msg["content"])

    if not st.session_state.rag_chat_history:
        st.markdown("**Try asking:**")
        quick_qs = ["Summarise my protein and sleep", "What foods provide protein without powder?",
                    "What habits could help me sleep better?"]
        qcols = st.columns(3)
        for i, question in enumerate(quick_qs):
            if qcols[i].button(question, use_container_width=True, key=f"rag_qq_{i}"):
                st.session_state.rag_pending_query = question
                st.rerun()

    query = st.chat_input("Ask about your logs, nutrition, sleep or activity...", key="rag_query")
    final_query = st.session_state.pop("rag_pending_query", None) or query
    confirmed = st.session_state.pop("rag_pending_safety", False)

    if final_query and final_query.strip():
        history = bounded_history(st.session_state.rag_chat_history)
        st.session_state.rag_chat_history.append({"role": "user", "content": final_query})
        requests = []
        flow = {"reference_resolution": "code_route"}
        with st.spinner("Finding evidence and preparing the response..."):
            try:
                options = {"use_local": use_local, "local_model": local_model,
                           "groq_api_key": groq_key.strip() if not use_local else None,
                           "request_metrics": requests}
                matches = []
                if confirmed:
                    result = generate_checked_result(
                        final_query, matches, confirmed_unexplained_weight_loss=True,
                        log_summary=log_summary, **options,
                    )
                elif log_summary_fields(final_query, history) is not None:
                    result = generate_checked_result(
                        final_query, [], log_summary=log_summary, history=history, **options,
                    )
                else:
                    if not use_local and not groq_key.strip():
                        raise ValueError("missing_api_key")
                    reason = reference_resolution_reason(final_query, history)
                    flow["reference_resolution"] = reason
                    resolved = (resolve_question(final_query, history, **options)
                                if reason in ("contextual_reference", "uncertain_reference") else
                                {"query": final_query, "clarification": ""})
                    if not resolved["query"].strip():
                        result = {"answer": resolved["clarification"],
                                  "response_mode": "clarification", "log_header": "",
                                  "log_usage": {"summary_available": log_summary is not None,
                                                "fresh_summary_sent_to_model": False, "rendered_fields": []}}
                    else:
                        signature = hashlib.sha256(KB_PATH.read_bytes()).hexdigest()
                        store = get_knowledge_store(signature)
                        matches = retrieve_relevant_passages(resolved["query"], store, k=6, max_distance=0.60)
                        result = generate_checked_result(
                            final_query, matches, log_summary=log_summary, history=history, **options,
                        )
                        result["resolved_question"] = resolved["query"]
                flow["model_stages"] = [request["stage"] for request in requests]
                flow["automatic_retries"] = False
                result["request_flow"] = flow
                result.setdefault("grounding", {})["request_flow"] = flow
                st.session_state.rag_chat_history.append({"role": "assistant", "result": result})
            except Exception as error:
                status = getattr(error, "status_code", None)
                if isinstance(error, ValueError) and str(error) == "missing_api_key":
                    message = "Add your Groq API key under AI settings & data sharing, then retry."
                elif status == 429:
                    message = describe_rate_limit(error, provider="Ollama" if use_local else "Groq")
                elif status in (401, 403):
                    message = "The model service rejected access. Check your key and account permissions."
                elif isinstance(error, APITimeoutError):
                    message = "The Groq request timed out (10-second timeout per API call). No answer was generated. Please retry."
                elif isinstance(error, ModuleNotFoundError):
                    message = "An AI dependency is missing. Run python -m pip install -r requirements.txt in NTR."
                elif isinstance(error, ValueError):
                    message = describe_validation_error(error)
                else:
                    message = "Could not complete the request. Check the selected model service and connection, then retry."
                st.session_state.rag_chat_history.append({"role": "assistant", "content": message, "error": True})
        st.session_state.rag_chat_history = st.session_state.rag_chat_history[-40:]
        st.rerun()
