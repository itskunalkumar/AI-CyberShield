import os
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# ============================================================
# CONFIGURATION
# ============================================================

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8001"
)

SUMMARY_URL = f"{API_BASE_URL}/api/v1/audit/summary"
RECENT_URL = f"{API_BASE_URL}/api/v1/audit/recent"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI-CyberShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# AUTO REFRESH
# ============================================================

st_autorefresh(
    interval=5000,
    key="cybershield_refresh"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 52px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.8;
        margin-bottom: 30px;
    }

    .status-box {
        padding: 18px 24px;
        border-radius: 12px;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 20px;
    }

    .alert-card {
        padding: 18px 22px;
        border-radius: 12px;
        margin-bottom: 12px;
        border: 1px solid rgba(255,255,255,0.10);
    }

    .explanation-card {
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 15px;
        border: 1px solid rgba(255,255,255,0.10);
    }

    .small-text {
        font-size: 14px;
        opacity: 0.75;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API FUNCTIONS
# ============================================================

def get_summary():

    try:

        response = requests.get(
            SUMMARY_URL,
            timeout=10
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:

        st.error(
            f"Could not connect to FastAPI: {e}"
        )

        return None


def get_recent_events(limit=100):

    try:

        response = requests.get(
            RECENT_URL,
            params={"limit": limit},
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        if isinstance(data, dict):

            if "events" in data:
                return data["events"]

            if "data" in data:
                return data["data"]

        if isinstance(data, list):
            return data

        return []

    except requests.exceptions.RequestException as e:

        st.error(
            f"Could not retrieve audit events: {e}"
        )

        return []


# ============================================================
# SHAP HELPER FUNCTIONS
# ============================================================

def normalize_shap_data(shap_data):
    """
    Convert SHAP JSON returned by FastAPI/PostgreSQL
    into a consistent list of dictionaries.

    Expected formats supported:

    [
        {
            "feature": "...",
            "shap_value": 0.52
        }
    ]

    or

    {
        "feature": 0.52,
        "another_feature": -0.31
    }
    """

    if shap_data is None:
        return []

    if isinstance(shap_data, list):

        normalized = []

        for item in shap_data:

            if not isinstance(item, dict):
                continue

            feature = (
                item.get("feature")
                or item.get("name")
                or item.get("feature_name")
            )

            value = (
                item.get("shap_value")
                if item.get("shap_value") is not None
                else item.get("value")
            )

            if feature is None or value is None:
                continue

            try:

                normalized.append(
                    {
                        "feature": str(feature),
                        "shap_value": float(value),
                    }
                )

            except (TypeError, ValueError):

                continue

        return normalized

    if isinstance(shap_data, dict):

        normalized = []

        # Handle dictionaries such as:
        # {"feature": "...", "shap_value": 0.4}

        if (
            "feature" in shap_data
            and (
                "shap_value" in shap_data
                or "value" in shap_data
            )
        ):

            value = shap_data.get(
                "shap_value",
                shap_data.get("value")
            )

            try:

                return [
                    {
                        "feature": str(
                            shap_data["feature"]
                        ),
                        "shap_value": float(value),
                    }
                ]

            except (TypeError, ValueError):

                return []

        # Handle:
        # {"feature_1": 0.4, "feature_2": -0.2}

        for feature, value in shap_data.items():

            try:

                normalized.append(
                    {
                        "feature": str(feature),
                        "shap_value": float(value),
                    }
                )

            except (TypeError, ValueError):

                continue

        return normalized

    return []


def format_feature_name(feature):

    feature = str(feature)

    return (
        feature
        .replace("__rollmean10", " | Rolling Mean 10")
        .replace("__rollmean5", " | Rolling Mean 5")
        .replace("__rollstd10", " | Rolling Std 10")
        .replace("__rollstd5", " | Rolling Std 5")
        .replace("__delta1", " | Delta 1")
        .replace("__pctchange", " | % Change")
        .replace("_", " ")
    )


# ============================================================
# LOAD DATA
# ============================================================

summary = get_summary()

events = get_recent_events(100)


if summary is None:
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ AI-CyberShield</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered cybersecurity monitoring for '
    'grid-connected microgrid telemetry.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# DATAFRAME
# ============================================================

if events:

    events_df = pd.DataFrame(events)

    if "timestamp" in events_df.columns:

        events_df["timestamp"] = pd.to_datetime(
            events_df["timestamp"],
            errors="coerce"
        )

        events_df = events_df.sort_values(
            "timestamp"
        )

else:

    events_df = pd.DataFrame()


# ============================================================
# CURRENT SECURITY STATUS
# ============================================================

risk_distribution = summary.get(
    "risk_distribution",
    {}
)

critical_count = risk_distribution.get(
    "CRITICAL",
    0
)

high_count = risk_distribution.get(
    "HIGH",
    0
)

medium_count = risk_distribution.get(
    "MEDIUM",
    0
)

low_count = risk_distribution.get(
    "LOW",
    0
)


if critical_count > 0:

    status_text = (
        "🔴 CRITICAL SECURITY EVENT DETECTED"
    )

    status_bg = "#5c1f1f"

elif high_count > 0:

    status_text = (
        "🟠 HIGH-RISK SECURITY EVENT DETECTED"
    )

    status_bg = "#5c3b1f"

elif medium_count > 0:

    status_text = (
        "🟡 MEDIUM-RISK SECURITY EVENTS DETECTED"
    )

    status_bg = "#5c511f"

else:

    status_text = (
        "🟢 NO HIGH OR CRITICAL SECURITY EVENTS DETECTED"
    )

    status_bg = "#123d2a"


st.markdown(
    f"""
    <div class="status-box"
         style="background:{status_bg};">

        {status_text}

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# KPI SECTION
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Total Events",
        summary.get("total_events", 0)
    )


with col2:

    st.metric(
        "Attack Events",
        summary.get("attack_events", 0)
    )


with col3:

    st.metric(
        "Anomaly Events",
        summary.get("anomaly_events", 0)
    )


with col4:

    st.metric(
        "Alerts",
        summary.get("alert_events", 0)
    )


with col5:

    st.metric(
        "Average Risk",
        f"{summary.get('average_risk_score', 0):.2f}"
    )


st.divider()


# ============================================================
# ACTIVE ALERTS
# ============================================================

st.header("🚨 Active Security Alerts")


if (
    not events_df.empty
    and "alert" in events_df.columns
):

    alerts_df = events_df[
        events_df["alert"] == True
    ].copy()

else:

    alerts_df = pd.DataFrame()


if alerts_df.empty:

    st.success(
        "No active security alerts."
    )

else:

    alerts_df = alerts_df.sort_values(
        "timestamp",
        ascending=False
    )

    for _, row in alerts_df.head(10).iterrows():

        risk_level = row.get(
            "risk_level",
            "UNKNOWN"
        )

        risk_score = row.get(
            "risk_score",
            0
        )

        endpoint = row.get(
            "endpoint",
            "unknown"
        )

        action = row.get(
            "recommended_action",
            "UNKNOWN"
        )

        attack_probability = row.get(
            "attack_probability",
            0
        )

        anomaly_score = row.get(
            "anomaly_score",
            0
        )

        if risk_level == "CRITICAL":

            icon = "🔴"

        elif risk_level == "HIGH":

            icon = "🟠"

        else:

            icon = "🟡"

        st.markdown(
            f"""
            <div class="alert-card">

            <h4>
            {icon} {risk_level} SECURITY EVENT
            </h4>

            <b>Endpoint:</b> {endpoint}<br>

            <b>Risk Score:</b> {risk_score:.2f}<br>

            <b>Attack Probability:</b>
            {attack_probability * 100:.2f}%<br>

            <b>Anomaly Score:</b>
            {anomaly_score * 100:.2f}%<br>

            <b>Recommended Action:</b>
            {action}

            </div>
            """,
            unsafe_allow_html=True
        )


st.divider()


# ============================================================
# RISK DISTRIBUTION + DETECTION STATISTICS
# ============================================================

col_left, col_right = st.columns(2)


with col_left:

    st.subheader("📊 Risk Distribution")

    risk_data = pd.DataFrame(
        {
            "Risk Level": [
                "LOW",
                "MEDIUM",
                "HIGH",
                "CRITICAL",
            ],
            "Events": [
                low_count,
                medium_count,
                high_count,
                critical_count,
            ],
        }
    )

    fig = px.bar(
        risk_data,
        x="Risk Level",
        y="Events",
        text="Events",
        title="Security Risk Distribution",
    )

    fig.update_layout(
        height=400,
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col_right:

    st.subheader("🔍 Detection Statistics")

    detection_data = pd.DataFrame(
        {
            "Detection Type": [
                "Attack",
                "Anomaly",
                "Alerts",
            ],
            "Events": [
                summary.get("attack_events", 0),
                summary.get("anomaly_events", 0),
                summary.get("alert_events", 0),
            ],
        }
    )

    fig = px.bar(
        detection_data,
        x="Detection Type",
        y="Events",
        text="Events",
        title="Security Detection Overview",
    )

    fig.update_layout(
        height=400,
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


st.divider()


# ============================================================
# SHAP / AI MODEL EXPLAINABILITY
# ============================================================

st.header("🧠 AI Model Explainability")

st.markdown(
    """
    Select a security event to understand which telemetry
    features contributed most to the XGBoost attack prediction.
    """
)


if (
    not events_df.empty
    and "shap_explanation" in events_df.columns
):

    shap_events = events_df[
        events_df["shap_explanation"].notna()
    ].copy()

    if not shap_events.empty:

        shap_events = shap_events.sort_values(
            "timestamp",
            ascending=False
        )

        # Create readable event labels
        event_options = []

        for idx, row in shap_events.iterrows():

            endpoint = str(
                row.get(
                    "endpoint",
                    "unknown"
                )
            )

            timestamp = row.get(
                "timestamp",
                ""
            )

            risk_level = row.get(
                "risk_level",
                "UNKNOWN"
            )

            risk_score = row.get(
                "risk_score",
                0
            )

            label = (
                f"{endpoint} | "
                f"{risk_level} | "
                f"Risk {risk_score:.2f} | "
                f"{timestamp}"
            )

            event_options.append(
                (idx, label)
            )

        selected_idx, selected_label = st.selectbox(
            "Select Security Event",
            event_options,
            format_func=lambda x: x[1],
            key="shap_event_selector"
        )

        selected_event = shap_events.loc[
            selected_idx
        ]

        # ----------------------------------------------------
        # Selected event summary
        # ----------------------------------------------------

        explain_col1, explain_col2, explain_col3, explain_col4 = (
            st.columns(4)
        )

        with explain_col1:

            st.metric(
                "Risk Level",
                selected_event.get(
                    "risk_level",
                    "UNKNOWN"
                )
            )

        with explain_col2:

            st.metric(
                "Risk Score",
                f"{float(selected_event.get('risk_score', 0)):.2f}"
            )

        with explain_col3:

            st.metric(
                "Attack Probability",
                f"{float(selected_event.get('attack_probability', 0)) * 100:.2f}%"
            )

        with explain_col4:

            st.metric(
                "Anomaly Score",
                f"{float(selected_event.get('anomaly_score', 0)) * 100:.2f}%"
            )

        st.caption(
            f"Endpoint: {selected_event.get('endpoint', 'unknown')}"
        )

        # ----------------------------------------------------
        # Parse SHAP
        # ----------------------------------------------------

        shap_data = normalize_shap_data(
            selected_event.get(
                "shap_explanation"
            )
        )

        if shap_data:

            shap_df = pd.DataFrame(
                shap_data
            )

            shap_df["feature_display"] = (
                shap_df["feature"]
                .apply(format_feature_name)
            )

            shap_df["direction"] = (
                shap_df["shap_value"]
                .apply(
                    lambda x:
                    "Increases attack probability"
                    if x > 0
                    else "Decreases attack probability"
                )
            )

            shap_df = shap_df.sort_values(
                "shap_value",
                key=lambda x: x.abs(),
                ascending=False
            )

            top_shap = shap_df.head(10).copy()

            # ------------------------------------------------
            # SHAP chart
            # ------------------------------------------------

            st.subheader(
                "Top Feature Contributions"
            )

            fig = px.bar(
                top_shap.sort_values(
                    "shap_value"
                ),
                x="shap_value",
                y="feature_display",
                orientation="h",
                text="shap_value",
                title="SHAP Feature Contributions",
                labels={
                    "shap_value": "SHAP Contribution",
                    "feature_display": "Feature",
                },
            )

            fig.update_traces(
                texttemplate="%{text:.4f}",
                textposition="outside"
            )

            fig.update_layout(
                height=550,
                showlegend=False,
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # ------------------------------------------------
            # SHAP table
            # ------------------------------------------------

            display_shap = top_shap[
                [
                    "feature_display",
                    "shap_value",
                    "direction",
                ]
            ].copy()

            display_shap.columns = [
                "Feature",
                "SHAP Contribution",
                "Effect",
            ]

            display_shap[
                "SHAP Contribution"
            ] = display_shap[
                "SHAP Contribution"
            ].round(6)

            st.dataframe(
                display_shap,
                use_container_width=True,
                hide_index=True
            )

            st.info(
                "Positive SHAP values indicate that the feature "
                "pushes the model toward a higher attack probability. "
                "Negative SHAP values push the prediction toward "
                "a lower attack probability."
            )

        else:

            st.warning(
                "SHAP explanation is not available for this event."
            )

    else:

        st.info(
            "No events with SHAP explanations are available yet."
        )

else:

    st.info(
        "The API is not currently returning SHAP explanations. "
        "Update the /api/v1/audit/recent endpoint first."
    )


st.divider()


# ============================================================
# SECURITY EVENT TIMELINE
# ============================================================

st.header("📈 Security Event Timeline")


if not events_df.empty:

    timeline_df = events_df.copy()

    timeline_df = timeline_df.sort_values(
        "timestamp"
    )

    # --------------------------------------------------------
    # Risk timeline
    # --------------------------------------------------------

    if "risk_score" in timeline_df.columns:

        fig = px.line(
            timeline_df,
            x="timestamp",
            y="risk_score",
            markers=True,
            title="Risk Score Over Time",
        )

        fig.update_layout(
            height=400,
            yaxis_title="Risk Score",
            xaxis_title="Time",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Attack probability
    # --------------------------------------------------------

    if "attack_probability" in timeline_df.columns:

        fig = px.line(
            timeline_df,
            x="timestamp",
            y="attack_probability",
            markers=True,
            title="Attack Probability Over Time",
        )

        fig.update_layout(
            height=400,
            yaxis_title="Attack Probability",
            xaxis_title="Time",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Anomaly score
    # --------------------------------------------------------

    if "anomaly_score" in timeline_df.columns:

        fig = px.line(
            timeline_df,
            x="timestamp",
            y="anomaly_score",
            markers=True,
            title="Anomaly Score Over Time",
        )

        fig.update_layout(
            height=400,
            yaxis_title="Anomaly Score",
            xaxis_title="Time",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

else:

    st.info(
        "No security events available yet."
    )


st.divider()


# ============================================================
# RECENT SECURITY EVENTS
# ============================================================

st.header("📋 Recent Security Events")


if not events_df.empty:

    display_columns = [
        "timestamp",
        "endpoint",
        "attack_probability",
        "attack_prediction",
        "anomaly_score",
        "anomaly_prediction",
        "risk_score",
        "risk_level",
        "recommended_action",
        "alert",
    ]

    available_columns = [
        col
        for col in display_columns
        if col in events_df.columns
    ]

    table_df = events_df[
        available_columns
    ].copy()

    if "attack_probability" in table_df.columns:

        table_df["attack_probability"] = (
            table_df["attack_probability"] * 100
        ).round(2)

    if "anomaly_score" in table_df.columns:

        table_df["anomaly_score"] = (
            table_df["anomaly_score"] * 100
        ).round(2)

    if "risk_score" in table_df.columns:

        table_df["risk_score"] = (
            table_df["risk_score"]
            .round(2)
        )

    table_df = table_df.sort_values(
        "timestamp",
        ascending=False
    ).head(20)

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No audit events found."
    )


st.divider()


# ============================================================
# SCENARIO MONITORING
# ============================================================

st.header("🎯 Scenario Security Monitoring")


if (
    not events_df.empty
    and "endpoint" in events_df.columns
):

    scenario_df = events_df.copy()

    scenario_df["scenario"] = (
        scenario_df["endpoint"]
        .astype(str)
        .str.replace(
            "microgrid-simulator-",
            "",
            regex=False
        )
    )

    scenario_summary = (
        scenario_df
        .groupby("scenario")
        .agg(
            events=("scenario", "size"),
            attacks=("attack_prediction", "sum"),
            anomalies=("anomaly_prediction", "sum"),
            alerts=("alert", "sum"),
            max_risk=("risk_score", "max"),
            avg_risk=("risk_score", "mean"),
        )
        .reset_index()
    )

    def get_status(row):

        if row["max_risk"] >= 80:
            return "🔴 CRITICAL"

        elif row["max_risk"] >= 60:
            return "🟠 HIGH"

        elif row["attacks"] > 0:
            return "🟡 ATTACK DETECTED"

        elif row["anomalies"] > 0:
            return "🟡 ANOMALY DETECTED"

        else:
            return "🟢 NORMAL"

    scenario_summary["status"] = (
        scenario_summary.apply(
            get_status,
            axis=1
        )
    )

    scenario_summary["avg_risk"] = (
        scenario_summary["avg_risk"]
        .round(2)
    )

    scenario_summary["max_risk"] = (
        scenario_summary["max_risk"]
        .round(2)
    )

    st.dataframe(
        scenario_summary[
            [
                "scenario",
                "events",
                "attacks",
                "anomalies",
                "alerts",
                "avg_risk",
                "max_risk",
                "status",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No scenario data available yet."
    )


st.divider()


# ============================================================
# SAFETY ENGINE
# ============================================================

st.header("🛡️ Safety Engine Status")


safety_col1, safety_col2, safety_col3 = st.columns(3)


with safety_col1:

    st.metric(
        "Physical Breaker Control",
        "DISABLED"
    )


with safety_col2:

    logical_isolation_active = False

    if (
        not events_df.empty
        and "logical_isolation" in events_df.columns
    ):

        logical_isolation_active = (
            events_df["logical_isolation"]
            .fillna(False)
            .astype(bool)
            .any()
        )

    st.metric(
        "Logical Isolation",
        "ACTIVE"
        if logical_isolation_active
        else "STANDBY"
    )


with safety_col3:

    st.metric(
        "Automated Physical Control",
        "BLOCKED"
    )


st.info(
    "AI-CyberShield uses ML for detection and risk assessment. "
    "The ML layer does not directly control physical breakers."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; opacity:0.65;">

    AI-CyberShield |
    ML-based Microgrid Cybersecurity Monitoring |
    FastAPI + XGBoost + Isolation Forest + PostgreSQL + SHAP

    </div>
    """,
    unsafe_allow_html=True
)