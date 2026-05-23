"""
Streamlit dashboard for ML Monitor.
Run with:  streamlit run ml_monitor/dashboard/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(page_title="ML Monitor", page_icon="📡", layout="wide")

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("📡 ML Monitoring Dashboard")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    db_path = st.text_input("SQLite DB path", value="ml_monitor.db")
    days = st.slider("History window (days)", 1, 30, 7)
    st.divider()
    if st.button("🔄 Refresh"):
        st.rerun()

# ── Connect ───────────────────────────────────────────────────────────────────
try:
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
except Exception as e:
    st.error(f"Cannot connect to DB: {e}")
    st.stop()

# ── Load all data upfront ─────────────────────────────────────────────────────
with engine.connect() as conn:
    # All model IDs across both tables
    drift_models = [r[0] for r in conn.execute(
        text("SELECT DISTINCT model_id FROM drift_records")
    ).fetchall()]
    metric_models = [r[0] for r in conn.execute(
        text("SELECT DISTINCT model_id FROM metric_records")
    ).fetchall()]
    all_models = sorted(set(drift_models + metric_models))

if not all_models:
    st.info("No models found yet. Run `python examples/train_baseline.py` and `python examples/simulate_drift.py` first.")
    st.stop()

model_id = st.selectbox("🤖 Select Model", all_models)

# ── Raw data queries ──────────────────────────────────────────────────────────
with engine.connect() as conn:
    df_drift = pd.read_sql(
        text("""
            SELECT timestamp, feature, test_name, statistic, p_value, drift_detected
            FROM drift_records
            WHERE model_id = :mid
            ORDER BY timestamp
        """),
        conn, params={"mid": model_id}
    )

    df_metrics = pd.read_sql(
        text("""
            SELECT timestamp, metric_name, value
            FROM metric_records
            WHERE model_id = :mid
            ORDER BY timestamp
        """),
        conn, params={"mid": model_id}
    )

    df_alerts = pd.read_sql(
        text("""
            SELECT timestamp, alert_type, severity, message, resolved
            FROM alert_records
            WHERE model_id = :mid
            ORDER BY timestamp DESC
        """),
        conn, params={"mid": model_id}
    )

# Parse timestamps
for df in [df_drift, df_metrics, df_alerts]:
    if "timestamp" in df.columns and not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

# ── SECTION 1: KPIs ───────────────────────────────────────────────────────────
st.subheader("📊 Snapshot")
k1, k2, k3, k4 = st.columns(4)

total_tests  = len(df_drift)
drifted      = int(df_drift["drift_detected"].sum()) if not df_drift.empty else 0
open_alerts  = int((df_alerts["resolved"] == 0).sum()) if not df_alerts.empty else 0
drift_pct    = round(drifted / total_tests * 100, 1) if total_tests > 0 else 0.0

k1.metric("Total Drift Tests", total_tests)
k2.metric("Drift Detected", drifted, delta=f"{drift_pct}% of tests", delta_color="inverse")
k3.metric("Open Alerts", open_alerts, delta_color="inverse")
if not df_metrics.empty:
    latest_metric = df_metrics.iloc[-1]
    k4.metric(f"Latest · {latest_metric['metric_name']}", f"{latest_metric['value']:.4f}")
else:
    k4.metric("Performance Metrics", "N/A")

st.divider()

# ── SECTION 2: Alerts ─────────────────────────────────────────────────────────
st.subheader("🚨 Alerts")
if df_alerts.empty:
    st.success("No alerts recorded for this model.")
else:
    open_df   = df_alerts[df_alerts["resolved"] == 0]
    closed_df = df_alerts[df_alerts["resolved"] == 1]

    tab_open, tab_all = st.tabs([f"Open ({len(open_df)})", f"All ({len(df_alerts)})"])
    with tab_open:
        if open_df.empty:
            st.success("No open alerts. ✅")
        else:
            for _, row in open_df.iterrows():
                color = "🔴" if row["severity"] == "critical" else "🟡"
                st.warning(f"{color} **{row['alert_type']}** — {row['timestamp'].strftime('%Y-%m-%d %H:%M')}  \n{row['message']}")
    with tab_all:
        st.dataframe(df_alerts, use_container_width=True)

st.divider()

# ── SECTION 3: Performance Metrics ───────────────────────────────────────────
st.subheader("📈 Performance Metrics")
if df_metrics.empty:
    st.info("No performance metrics for this model. Make sure you pass `y_true` and `y_pred` to `monitor.run()`.")
else:
    metric_names = df_metrics["metric_name"].unique().tolist()

    # Latest value cards
    cols = st.columns(len(metric_names))
    for i, m in enumerate(metric_names):
        vals = df_metrics[df_metrics["metric_name"] == m]["value"]
        cols[i].metric(m, f"{vals.iloc[-1]:.4f}")

    # Trend lines
    fig = px.line(
        df_metrics, x="timestamp", y="value", color="metric_name",
        markers=True, title="Metric Trends Over Time",
        labels={"value": "Score", "timestamp": "Time", "metric_name": "Metric"},
    )
    fig.update_layout(legend_title="Metric", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── SECTION 4: Drift Overview ─────────────────────────────────────────────────
st.subheader("🌊 Drift Detection")

if df_drift.empty:
    st.info("No drift records for this model.")
else:
    tab1, tab2, tab3 = st.tabs(["Drift Rate by Feature", "Heatmap", "Raw Records"])

    # ── Tab 1: Bar chart ──────────────────────────────────────────────────────
    with tab1:
        drift_rate = (
            df_drift.groupby("feature")["drift_detected"]
            .agg(drift_rate="mean", tests="count", drifted="sum")
            .reset_index()
            .sort_values("drift_rate", ascending=False)
        )
        drift_rate["drift_rate_pct"] = (drift_rate["drift_rate"] * 100).round(1)

        fig_bar = px.bar(
            drift_rate, x="feature", y="drift_rate_pct",
            color="drift_rate_pct",
            color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
            range_color=[0, 100],
            title="Drift Rate per Feature (%)",
            labels={"drift_rate_pct": "Drift Rate (%)", "feature": "Feature"},
            text="drift_rate_pct",
        )
        fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_bar.update_layout(coloraxis_showscale=False, yaxis_range=[0, 110])
        st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(
            drift_rate.rename(columns={
                "feature": "Feature", "drift_rate_pct": "Drift Rate %",
                "tests": "Total Tests", "drifted": "Times Drifted"
            }),
            use_container_width=True, hide_index=True
        )

    # ── Tab 2: Heatmap ────────────────────────────────────────────────────────
    with tab2:
        df_drift["date"] = df_drift["timestamp"].dt.date

        # Aggregate: max drift_detected per feature per day
        pivot_data = (
            df_drift.groupby(["feature", "date"])["drift_detected"]
            .max()
            .reset_index()
        )

        features = sorted(pivot_data["feature"].unique())
        dates    = sorted(pivot_data["date"].unique())

        # Build matrix
        matrix = np.zeros((len(features), len(dates)))
        feat_idx = {f: i for i, f in enumerate(features)}
        date_idx = {d: i for i, d in enumerate(dates)}

        for _, row in pivot_data.iterrows():
            r = feat_idx[row["feature"]]
            c = date_idx[row["date"]]
            matrix[r][c] = row["drift_detected"]

        if len(dates) == 0 or len(features) == 0:
            st.info("Not enough data to render heatmap yet.")
        else:
            fig_heat = go.Figure(data=go.Heatmap(
                z=matrix,
                x=[str(d) for d in dates],
                y=features,
                colorscale=[[0, "#2ecc71"], [1, "#e74c3c"]],
                showscale=True,
                colorbar=dict(
                    title="Drift",
                    tickvals=[0, 1],
                    ticktext=["None", "Detected"],
                ),
                hoverongaps=False,
                hovertemplate="Feature: %{y}<br>Date: %{x}<br>Drift: %{z}<extra></extra>",
            ))
            fig_heat.update_layout(
                title="Drift Heatmap — Feature × Day",
                xaxis_title="Date",
                yaxis_title="Feature",
                height=max(300, len(features) * 40 + 100),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    # ── Tab 3: Raw table ──────────────────────────────────────────────────────
    with tab3:
        display_df = df_drift.copy()
        display_df["drift_detected"] = display_df["drift_detected"].map(
            {0: "❌ No", 1: "⚠️ Yes", False: "❌ No", True: "⚠️ Yes"}
        )
        st.dataframe(display_df, use_container_width=True)

st.divider()

# ── SECTION 5: Per-feature drill-down ────────────────────────────────────────
st.subheader("🔍 Feature Drill-down")
if not df_drift.empty:
    selected_feature = st.selectbox("Select Feature", sorted(df_drift["feature"].unique()))
    feat_df = df_drift[df_drift["feature"] == selected_feature].copy()

    col_a, col_b = st.columns(2)
    with col_a:
        fig_stat = px.line(
            feat_df, x="timestamp", y="statistic", color="test_name",
            markers=True, title=f"Test Statistic — {selected_feature}",
            labels={"statistic": "Statistic", "test_name": "Test"},
        )
        st.plotly_chart(fig_stat, use_container_width=True)

    with col_b:
        p_df = feat_df.dropna(subset=["p_value"])
        if not p_df.empty:
            fig_p = px.line(
                p_df, x="timestamp", y="p_value", color="test_name",
                markers=True, title=f"p-value — {selected_feature}",
                labels={"p_value": "p-value", "test_name": "Test"},
            )
            fig_p.add_hline(y=0.05, line_dash="dash", line_color="red",
                            annotation_text="α = 0.05")
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.info("No p-values available for this feature (PSI test doesn't produce one).")
else:
    st.info("No drift data available.")