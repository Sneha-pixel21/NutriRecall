import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from utils.db import (
    init_db, load_data, insert_entry, delete_entry,
    delete_last_entry, export_csv, entry_exists
)
from utils.ai import compute_scores, generate_insights, nutrirecall_ai

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NutriRecall",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .score-green { color: #22c55e; font-weight: 600; }
  .score-amber { color: #f59e0b; font-weight: 600; }
  .score-red   { color: #ef4444; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar — nav + settings ─────────────────────────────────────────────────

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
    st.subheader("AI Settings")
    use_local = st.toggle("Use local Ollama model", value=False)
    if not use_local:
        groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
        st.caption("Get a free key at console.groq.com")
    else:
        groq_key = ""
        st.caption("Make sure Ollama is running with phi3:mini")

    st.divider()
    data_all = load_data()
    if not data_all.empty:
        csv_bytes = export_csv()
        st.download_button("⬇️ Export CSV", csv_bytes, "nutrirecall_export.csv", "text/csv")

# ─── Shared data load + scoring ───────────────────────────────────────────────

init_db()
data = load_data()

# ── Debug info (always visible so we can diagnose issues) ─────────────────────
import sqlite3 as _sqlite3
from pathlib import Path as _Path
_db = _Path("data/health_logs.db")
with st.sidebar:
    st.divider()
    st.markdown("**Debug**")
    st.caption(f"DB exists: {_db.exists()}")
    st.caption(f"DB path: {_db.resolve() if _db.exists() else 'NOT FOUND'}")
    st.caption(f"Rows in DB: {len(data)}")
    if _db.exists():
        with _sqlite3.connect(_db) as _con:
            _rows = _con.execute("SELECT date, weight, protein FROM logs ORDER BY date DESC LIMIT 3").fetchall()
        for _r in _rows:
            st.caption(f"  {_r[0]} | {_r[1]}kg | {_r[2]}g")

if not data.empty:
    try:
        data = compute_scores(data)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        data = pd.DataFrame()

# ─── Helper: week slices ──────────────────────────────────────────────────────

def get_week(df, offset_weeks=0):
    today = pd.Timestamp(date.today())
    end   = today - timedelta(weeks=offset_weeks)
    start = end   - timedelta(days=6)
    mask  = (df["date"] >= start) & (df["date"] <= end)
    return df[mask]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DAILY LOG
# ══════════════════════════════════════════════════════════════════════════════

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

    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("💾 Save Entry", use_container_width=True):
            if weight is None or protein is None or sleep is None:
                st.error("Please fill in all fields before saving.")
            else:
                date_str = str(log_date)
                ok = insert_entry(date_str, weight, protein, sleep, 1 if workout == "Yes" else 0)
                if ok:
                    st.success(f"✅ Entry saved for {log_date.strftime('%d %b %Y')}!")
                    st.rerun()
                else:
                    st.warning(f"An entry for {log_date} already exists. Go to History to edit it.")
    with c2:
        if st.button("Delete Last Entry", use_container_width=True):
            if data.empty:
                st.warning("No entries to delete.")
            else:
                delete_last_entry()
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

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Dashboard":
    st.header("📊 Dashboard")

    if data.empty:
        st.info("Add some entries first to see your dashboard.")
    else:
        w7 = get_week(data, offset_weeks=0)
        if w7.empty:
            w7 = data.tail(7)

        avg_p  = w7["protein"].mean()
        avg_pr = w7["protein_required"].mean()
        avg_s  = w7["sleep_hours"].mean()
        wo     = int(w7["workout"].sum())
        hs     = w7["health_score_10"].mean()

        # ── Health score gauge ──
        st.subheader("Health Score")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(hs, 1),
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [1, 10], "tickwidth": 1},
                "bar": {"color": "#1a7a4a" if hs >= 7 else ("#b45309" if hs >= 5 else "#b91c1c")},
                "steps": [
                    {"range": [1, 5],  "color": "#fee2e2"},
                    {"range": [5, 7],  "color": "#fef3c7"},
                    {"range": [7, 10], "color": "#d1fae5"},
                ],
                "threshold": {"line": {"color": "black", "width": 2}, "thickness": 0.75, "value": hs},
            },
            number={"suffix": "/10", "font": {"size": 40}},
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=20, b=20, l=40, r=40))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── KPI row ──
        m1, m2, m3, m4 = st.columns(4)
        delta_p = round(avg_p - avg_pr, 1)
        m1.metric("Avg Protein (7d)", f"{avg_p:.0f}g", f"{delta_p:+.0f}g vs target")
        m2.metric("Protein Target", f"{avg_pr:.0f}g", "based on weight")
        m3.metric("Avg Sleep (7d)", f"{avg_s:.1f} hrs", f"{'✓' if avg_s >= 7 else '⚠ below 7'}")
        m4.metric("Workout Days (7d)", f"{wo}/7")

        st.divider()

        # ── Weight trend ──
        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Weight progress")
            fig_w = go.Figure()
            fig_w.add_trace(go.Scatter(
                x=data["date"], y=data["weight"],
                mode="lines+markers", name="Weight",
                line=dict(color="#3b82f6", width=2),
                marker=dict(size=5),
            ))
            fig_w.update_layout(
                height=260, margin=dict(t=10, b=10, l=10, r=10),
                xaxis_title="", yaxis_title="kg",
                hovermode="x unified",
            )
            st.plotly_chart(fig_w, use_container_width=True)

        with col_r:
            st.subheader("Protein vs Target")
            fig_p = go.Figure()
            fig_p.add_trace(go.Bar(
                x=data["date"], y=data["protein"],
                name="Protein (g)", marker_color="#6366f1",
            ))
            fig_p.add_trace(go.Scatter(
                x=data["date"], y=data["protein_required"],
                mode="lines", name="Target",
                line=dict(color="#ef4444", dash="dash", width=2),
            ))
            fig_p.update_layout(
                height=260, margin=dict(t=10, b=10, l=10, r=10),
                xaxis_title="", yaxis_title="grams",
                hovermode="x unified", legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_p, use_container_width=True)

        # ── Sleep trend ──
        st.subheader("Sleep quality")
        fig_s = go.Figure()
        fig_s.add_trace(go.Bar(
            x=data["date"], y=data["sleep_hours"],
            name="Sleep (hrs)",
            marker_color=[
                "#22c55e" if h >= 7 else ("#f59e0b" if h >= 6 else "#ef4444")
                for h in data["sleep_hours"]
            ],
        ))
        fig_s.add_hline(y=7, line_dash="dash", line_color="gray", annotation_text="7h target")
        fig_s.update_layout(
            height=220, margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title="", yaxis_title="hours",
        )
        st.plotly_chart(fig_s, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📅 History":
    st.header("📅 Log History")

    if data.empty:
        st.info("No entries yet. Start logging in Daily Log.")
    else:
        display = data.copy()
        display["date"] = display["date"].dt.strftime("%Y-%m-%d")
        display["workout"] = display["workout"].map({1: "✅ Yes", 0: "❌ No"})
        display["health_score_10"] = display["health_score_10"].round(1)
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
            height=420,
        )

        st.divider()
        st.subheader("Edit or delete an entry")

        raw = data.copy()
        raw["label"] = raw["date"].dt.strftime("%Y-%m-%d") + "  |  " + raw["weight"].astype(str) + "kg"
        selected_label = st.selectbox("Select entry", raw["label"].tolist()[::-1])
        selected_row = raw[raw["label"] == selected_label].iloc[0]

        ec1, ec2, ec3, ec4 = st.columns(4)
        new_w  = ec1.number_input("Weight (kg)",  value=float(selected_row["weight"]), step=1.0)
        new_p  = ec2.number_input("Protein (g)",  value=float(selected_row["protein"]), step=1.0)
        new_sl = ec3.number_input("Sleep (hrs)",  value=float(selected_row["sleep_hours"]), step=1.0)
        new_wo = ec4.selectbox("Workout", ["Yes", "No"],
                               index=0 if selected_row["workout"] == 1 else 1)

        b1, b2, _ = st.columns([1, 1, 4])
        with b1:
            if st.button("💾 Save changes", use_container_width=True):
                from utils.db import update_entry
                update_entry(
                    int(selected_row["id"]), new_w, new_p, new_sl,
                    1 if new_wo == "Yes" else 0,
                )
                st.success("Entry updated!")
                st.rerun()
        with b2:
            if st.button("Delete entry", use_container_width=True):
                delete_entry(int(selected_row["id"]))
                st.warning("Entry deleted.")
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — WEEK COMPARE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔄 Week Compare":
    st.header("🔄 Week-over-Week Comparison")

    if data.empty or len(data) < 2:
        st.info("Add at least a few entries to compare weeks.")
    else:
        this_w = get_week(data, offset_weeks=0)
        last_w = get_week(data, offset_weeks=1)

        metrics = ["protein", "sleep_hours", "workout", "health_score_10"]
        labels  = ["Avg Protein (g)", "Avg Sleep (hrs)", "Workout Days", "Health Score"]

        def week_avg(df, col):
            if df.empty:
                return 0.0
            if col == "workout":
                return float(df[col].sum())
            return float(df[col].mean())

        this_vals = [week_avg(this_w, m) for m in metrics]
        last_vals = [week_avg(last_w, m) for m in metrics]

        # Summary cards
        cols = st.columns(4)
        for i, (col, label, tv, lv) in enumerate(zip(cols, labels, this_vals, last_vals)):
            delta = tv - lv
            fmt = ".0f" if i in [0, 2] else ".1f"
            col.metric(label, f"{tv:{fmt}}", f"{delta:+{fmt}} vs last week")

        st.divider()

        # Grouped bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(name="This week", x=labels, y=[round(v, 1) for v in this_vals],
                             marker_color="#6366f1"))
        fig.add_trace(go.Bar(name="Last week", x=labels, y=[round(v, 1) for v in last_vals],
                             marker_color="#a5b4fc"))
        fig.update_layout(
            barmode="group", height=350,
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Day-by-day protein for this week
        if not this_w.empty:
            st.subheader("This week — daily protein vs target")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=this_w["date"].dt.strftime("%a %d"),
                y=this_w["protein"],
                name="Protein (g)", marker_color="#6366f1",
            ))
            fig2.add_trace(go.Scatter(
                x=this_w["date"].dt.strftime("%a %d"),
                y=this_w["protein_required"],
                mode="lines+markers", name="Target",
                line=dict(color="#ef4444", dash="dash"),
            ))
            fig2.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10),
                               hovermode="x unified")
            st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "✨ AI Assistant":
    st.header("✨ NutriRecall AI")
    st.caption("Personalised advice based on your actual health data.")

    if data.empty:
        # Clear empty state — tell user exactly what to do
        st.info("📭 No data yet! Go to **📥 Daily Log** in the sidebar and save at least one entry. Then come back here.")
        st.stop()

    # ── API key check (only when using Groq) ──────────────────────────────────
    if not use_local and not groq_key:
        st.warning("Paste your **Groq API Key** in the sidebar to use the AI. Get a free key at [console.groq.com](https://console.groq.com)")
        st.info("Or toggle **Use local Ollama model** in the sidebar if Ollama is installed.")
        st.stop()

    # ── Context expander ──────────────────────────────────────────────────────
    with st.expander("📋 Data being sent to AI", expanded=False):
        st.code(generate_insights(data), language="text")

    # ── Chat history ──────────────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render existing messages
    for msg in st.session_state.chat_history:
        role   = msg["role"]
        avatar = "🧑" if role == "user" else "✨"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])

    # ── Quick question buttons (only when chat is empty) ──────────────────────
    if not st.session_state.chat_history:
        st.markdown("**Try asking:**")
        qcols = st.columns(3)
        quick_qs = [
            "Why am I not gaining muscle?",
            "How can I improve my sleep?",
            "What should I eat today?",
        ]
        for i, q in enumerate(quick_qs):
            if qcols[i].button(q, use_container_width=True, key=f"qq_{i}"):
                st.session_state.pending_query = q
                st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────────
    query = st.chat_input("Ask anything about your health data...")

    final_query = query
    if "pending_query" in st.session_state:
        final_query = st.session_state.pop("pending_query")

    if final_query:
        st.session_state.chat_history.append({"role": "user", "content": final_query})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(final_query)

        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("Thinking..."):
                try:
                    response = nutrirecall_ai(
                        final_query, data,
                        use_local=use_local,
                        groq_api_key=groq_key if not use_local else "",
                    )
                except Exception as e:
                    response = f"Error: {e}"
            st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # ── Clear button ──────────────────────────────────────────────────────────
    if st.session_state.get("chat_history"):
        st.divider()
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()