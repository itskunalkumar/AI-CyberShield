import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# Flatten HTML blocks so Markdown never renders indented tags as a code block.
_st_markdown = st.markdown


def _flat_markdown(body, *args, **kwargs):
    if kwargs.get("unsafe_allow_html") and isinstance(body, str):
        body = " ".join(line.strip() for line in body.splitlines() if line.strip())
    return _st_markdown(body, *args, **kwargs)


st.markdown = _flat_markdown


# ============================================================
# CONFIG
# ============================================================

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8001",
).rstrip("/")

SUMMARY_URL = f"{API_BASE_URL}/api/v1/audit/summary"
RECENT_URL = f"{API_BASE_URL}/api/v1/audit/recent"
HEALTH_URL = f"{API_BASE_URL}/health"
MODEL_URL = f"{API_BASE_URL}/model-info"

st.set_page_config(
    page_title="AI-CyberShield | Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st_autorefresh(
    interval=5000,
    key="cybershield_live_refresh",
)


# ============================================================
# CYBER UI
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --bg:#030711;
        --panel:#071321;
        --panel2:#0b1b2e;
        --line:rgba(73,190,255,.18);
        --cyan:#22d3ee;
        --blue:#60a5fa;
        --violet:#a78bfa;
        --green:#34d399;
        --yellow:#fbbf24;
        --orange:#fb923c;
        --red:#fb7185;
        --text:#eef8ff;
        --muted:#8ca7bf;
    }

    .stApp {
        background:
          radial-gradient(circle at 50% -15%,rgba(34,211,238,.13),transparent 32%),
          radial-gradient(circle at 100% 30%,rgba(96,165,250,.08),transparent 24%),
          linear-gradient(180deg,#020610 0%,#050b15 55%,#020610 100%);
    }

    [data-testid="stHeader"] { background:transparent; }

    .block-container {
        max-width:1650px;
        padding-top:1rem;
        padding-bottom:3rem;
    }

    .topbar {
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:16px;
        padding:16px 20px;
        margin-bottom:14px;
        border:1px solid var(--line);
        border-radius:22px;
        background:linear-gradient(135deg,rgba(13,31,52,.92),rgba(3,10,21,.95));
        box-shadow:0 18px 55px rgba(0,0,0,.35);
    }

    .brand {
        display:flex;
        align-items:center;
        gap:13px;
    }

    .logo {
        width:52px;
        height:52px;
        display:flex;
        align-items:center;
        justify-content:center;
        border-radius:16px;
        font-size:28px;
        background:linear-gradient(145deg,rgba(34,211,238,.20),rgba(167,139,250,.15));
        border:1px solid rgba(116,226,255,.28);
        box-shadow:0 0 35px rgba(34,211,238,.14);
    }

    .title {
        font-size:29px;
        font-weight:900;
        letter-spacing:.4px;
        color:#eef8ff;
    }

    .subtitle {
        color:var(--muted);
        font-size:10px;
        text-transform:uppercase;
        letter-spacing:1.7px;
        margin-top:5px;
        font-weight:800;
    }

    .badges {
        display:flex;
        flex-wrap:wrap;
        gap:7px;
        justify-content:flex-end;
    }

    .badge {
        padding:7px 10px;
        border-radius:999px;
        border:1px solid rgba(255,255,255,.08);
        background:rgba(255,255,255,.035);
        color:#d8eafd;
        font-size:10px;
        font-weight:800;
        letter-spacing:.5px;
    }

    .hero-grid {
        display:grid;
        grid-template-columns:1.1fr .9fr;
        gap:14px;
        margin-bottom:14px;
    }

    .hero, .signals, .panel, .metric {
        border:1px solid var(--line);
        background:linear-gradient(145deg,rgba(12,29,50,.90),rgba(3,11,22,.94));
        box-shadow:0 22px 60px rgba(0,0,0,.28);
    }

    .hero {
        min-height:420px;
        border-radius:26px;
        position:relative;
        overflow:hidden;
        display:flex;
        align-items:center;
        justify-content:center;
    }

    .hero-gridlines {
        position:absolute;
        inset:0;
        background-image:
          linear-gradient(rgba(34,211,238,.045) 1px,transparent 1px),
          linear-gradient(90deg,rgba(34,211,238,.045) 1px,transparent 1px);
        background-size:42px 42px;
        mask-image:radial-gradient(circle at center,#000,transparent 76%);
    }

    .globe {
        width:315px;
        height:315px;
        border-radius:50%;
        position:relative;
        display:flex;
        align-items:center;
        justify-content:center;
        background:
          radial-gradient(circle at 35% 30%,rgba(255,255,255,.12),transparent 15%),
          radial-gradient(circle at 50% 45%,rgba(34,211,238,.24),rgba(7,25,49,.75) 45%,rgba(3,9,19,.95) 72%);
        border:1px solid rgba(94,210,255,.30);
        box-shadow:
          0 0 40px rgba(34,211,238,.16),
          inset 0 0 55px rgba(34,211,238,.09);
        transform:perspective(800px) rotateX(8deg);
    }

    .globe:before, .globe:after {
        content:"";
        position:absolute;
        inset:9%;
        border:1px solid rgba(96,165,250,.19);
        border-radius:50%;
    }

    .globe:after {
        inset:19%;
        border-style:dashed;
        border-color:rgba(167,139,250,.22);
        animation:rotate 12s linear infinite;
    }

    .orbit {
        position:absolute;
        border:1px solid rgba(34,211,238,.27);
        border-radius:50%;
    }

    .orbit.one {
        width:355px;
        height:120px;
        transform:rotate(22deg);
        animation:rotate 8s linear infinite;
    }

    .orbit.two {
        width:350px;
        height:145px;
        transform:rotate(-55deg);
        border-color:rgba(167,139,250,.24);
        animation:rotateReverse 11s linear infinite;
    }

    .core-shield {
        width:145px;
        height:170px;
        clip-path:polygon(
          50% 0, 88% 14%, 100% 48%, 82% 79%,
          50% 100%, 18% 79%, 0 48%, 12% 14%
        );
        background:
          linear-gradient(
            145deg,
            rgba(211,251,255,.22),
            rgba(34,211,238,.20) 32%,
            rgba(44,96,190,.30) 62%,
            rgba(167,139,250,.25)
          );
        border:1px solid rgba(178,241,255,.45);
        box-shadow:
          0 0 30px rgba(34,211,238,.24),
          inset 0 0 35px rgba(255,255,255,.10);
        display:flex;
        align-items:center;
        justify-content:center;
        position:relative;
        z-index:5;
        animation:float 4s ease-in-out infinite;
    }

    .core-shield:after {
        content:"";
        position:absolute;
        left:17%;
        right:17%;
        height:2px;
        background:linear-gradient(90deg,transparent,#fff,transparent);
        box-shadow:0 0 16px rgba(255,255,255,.6);
        animation:scan 2.3s ease-in-out infinite;
    }

    .shield-icon {
        font-size:62px;
        filter:drop-shadow(0 0 18px rgba(255,255,255,.25));
    }

    .hero-caption {
        position:absolute;
        bottom:20px;
        left:0;
        right:0;
        text-align:center;
        z-index:7;
    }

    .eyebrow {
        color:var(--muted);
        font-size:10px;
        text-transform:uppercase;
        letter-spacing:2px;
        font-weight:850;
    }

    .hero-state {
        margin-top:4px;
        font-size:23px;
        font-weight:900;
        letter-spacing:1px;
    }

    .hero-risk {
        margin-top:3px;
        color:#b8cde0;
        font-size:12px;
    }

    .signals {
        min-height:420px;
        border-radius:26px;
        padding:19px;
    }

    .section-title {
        color:#b0c5da;
        font-size:11px;
        font-weight:900;
        letter-spacing:1.4px;
        text-transform:uppercase;
        margin:4px 0 14px;
    }

    .signal {
        margin-bottom:17px;
    }

    .signal-head {
        display:flex;
        justify-content:space-between;
        font-size:12px;
        margin-bottom:7px;
    }

    .signal-label { color:#9fb7ce; font-weight:700; }
    .signal-value { font-weight:900; }

    .bar {
        height:8px;
        border-radius:999px;
        background:rgba(255,255,255,.05);
        border:1px solid rgba(255,255,255,.06);
        overflow:hidden;
    }

    .fill {
        height:100%;
        border-radius:999px;
        background:linear-gradient(90deg,var(--cyan),var(--blue),var(--violet));
        box-shadow:0 0 16px rgba(34,211,238,.24);
    }

    .live-grid {
        display:grid;
        grid-template-columns:1fr 1fr;
        gap:9px;
        margin-top:19px;
    }

    .live-cell {
        padding:12px;
        border-radius:14px;
        background:rgba(255,255,255,.025);
        border:1px solid rgba(255,255,255,.06);
    }

    .live-label {
        font-size:9px;
        color:#718aa2;
        text-transform:uppercase;
        letter-spacing:1px;
        font-weight:800;
    }

    .live-value {
        margin-top:4px;
        font-size:14px;
        font-weight:900;
        line-height:1.2;
    }

    .ticker {
        overflow:hidden;
        white-space:nowrap;
        padding:9px 0;
        margin-bottom:14px;
        border:1px solid var(--line);
        border-radius:14px;
        background:rgba(4,12,24,.72);
        color:#9fc2dc;
        font-size:11px;
    }

    .ticker-inner {
        display:inline-block;
        padding-left:100%;
        animation:marquee 35s linear infinite;
    }

    .metric-grid {
        display:grid;
        grid-template-columns:repeat(5,1fr);
        gap:10px;
        margin-bottom:14px;
    }

    .metric {
        border-radius:18px;
        padding:15px;
        position:relative;
        overflow:hidden;
    }

    .metric:before {
        content:"";
        position:absolute;
        top:0;
        left:-60%;
        width:60%;
        height:2px;
        background:linear-gradient(90deg,transparent,var(--cyan),transparent);
        animation:sweep 3s linear infinite;
    }

    .metric-label {
        color:#7892aa;
        font-size:9px;
        font-weight:850;
        text-transform:uppercase;
        letter-spacing:1px;
    }

    .metric-value {
        font-size:27px;
        font-weight:900;
        margin-top:5px;
    }

    .metric-note {
        font-size:9px;
        color:#5f7890;
        margin-top:3px;
    }

    .status {
        display:flex;
        justify-content:space-between;
        align-items:center;
        gap:14px;
        padding:12px 15px;
        margin-bottom:14px;
        border-radius:17px;
        border:1px solid rgba(34,211,238,.13);
        background:linear-gradient(90deg,rgba(34,211,238,.05),rgba(167,139,250,.045));
    }

    .status-text {
        font-size:11px;
        font-weight:900;
        letter-spacing:.8px;
        text-transform:uppercase;
    }

    .status-meta {
        color:#7892aa;
        font-size:9px;
    }

    .panel {
        border-radius:20px;
        padding:14px;
        margin-bottom:14px;
    }

    .alert {
        padding:14px;
        border-radius:15px;
        margin-bottom:9px;
        background:rgba(255,255,255,.024);
        border:1px solid rgba(255,255,255,.07);
        border-left:3px solid var(--red);
    }

    .alert-head {
        display:flex;
        justify-content:space-between;
        gap:10px;
        color:#98b0c7;
        font-size:10px;
    }

    .alert-title {
        color:#eff7ff;
        font-size:13px;
        font-weight:900;
        margin-top:5px;
    }

    .alert-grid {
        display:grid;
        grid-template-columns:repeat(3,1fr);
        gap:8px;
        margin-top:9px;
    }

    .alert-box {
        padding:8px;
        background:rgba(255,255,255,.025);
        border-radius:10px;
    }

    .alert-box span {
        display:block;
        color:#6d859c;
        text-transform:uppercase;
        font-size:8px;
        letter-spacing:.8px;
    }

    .alert-box b {
        display:block;
        margin-top:3px;
        font-size:12px;
    }

    .scenario-grid {
        display:grid;
        grid-template-columns:repeat(4,1fr);
        gap:9px;
    }

    .scenario {
        min-height:98px;
        padding:11px;
        border-radius:15px;
        background:rgba(255,255,255,.022);
        border:1px solid rgba(255,255,255,.06);
    }

    .scenario-name {
        color:#c9dbe9;
        font-size:10px;
        font-weight:850;
        white-space:nowrap;
        overflow:hidden;
        text-overflow:ellipsis;
    }

    .scenario-level {
        margin-top:8px;
        font-size:12px;
        font-weight:900;
    }

    .scenario-meta {
        margin-top:5px;
        color:#7590a7;
        font-size:9px;
        line-height:1.35;
    }

    .topology {
        display:flex;
        align-items:center;
        gap:8px;
        padding:16px;
        border-radius:18px;
        border:1px solid var(--line);
        background:rgba(6,15,28,.78);
        overflow-x:auto;
    }

    .node {
        min-width:112px;
        padding:11px 8px;
        border-radius:14px;
        text-align:center;
        background:rgba(255,255,255,.024);
        border:1px solid rgba(255,255,255,.06);
    }

    .node-icon { font-size:22px; }
    .node-name {
        color:#a4bad0;
        font-size:9px;
        text-transform:uppercase;
        letter-spacing:.7px;
        font-weight:850;
        margin-top:4px;
    }

    .arrow {
        min-width:30px;
        height:2px;
        background:linear-gradient(90deg,transparent,rgba(34,211,238,.55),transparent);
        position:relative;
    }

    .arrow:after {
        content:"";
        position:absolute;
        top:-3px;
        left:0;
        width:7px;
        height:7px;
        border-radius:50%;
        background:var(--cyan);
        box-shadow:0 0 13px var(--cyan);
        animation:packet 1.8s linear infinite;
    }

    .footer {
        text-align:center;
        color:#536b82;
        font-size:9px;
        padding:13px 0 2px;
        letter-spacing:.5px;
    }

    @keyframes rotate {
        from { transform:rotate(0deg); }
        to { transform:rotate(360deg); }
    }

    @keyframes rotateReverse {
        from { transform:rotate(360deg); }
        to { transform:rotate(0deg); }
    }

    @keyframes float {
        0%,100% { transform:perspective(800px) rotateX(7deg) rotateY(-7deg) translateY(0); }
        50% { transform:perspective(800px) rotateX(2deg) rotateY(7deg) translateY(-9px); }
    }

    @keyframes scan {
        0%,100% { top:28%; opacity:.15; }
        50% { top:65%; opacity:1; }
    }

    @keyframes marquee {
        to { transform:translateX(-100%); }
    }

    @keyframes sweep {
        to { left:120%; }
    }

    @keyframes packet {
        from { left:0; }
        to { left:calc(100% - 7px); }
    }

    @media(max-width:1100px) {
        .hero-grid { grid-template-columns:1fr; }
        .metric-grid { grid-template-columns:repeat(2,1fr); }
        .scenario-grid { grid-template-columns:repeat(2,1fr); }
        .topbar { flex-direction:column; align-items:flex-start; }
        .badges { justify-content:flex-start; }
    }

    @media(max-width:650px) {
        .metric-grid,.scenario-grid,.live-grid { grid-template-columns:1fr; }
        .title { font-size:24px; }
        .globe { width:270px; height:270px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API HELPERS
# ============================================================

def get_json(url, params=None):
    try:
        response = requests.get(
            url,
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def pct(value):
    try:
        return float(value) * 100
    except (TypeError, ValueError):
        return 0.0


def get_level(score):
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "LOW"

    if score >= 80:
        return "CRITICAL"
    if score >= 60:
        return "HIGH"
    if score >= 30:
        return "MEDIUM"
    return "LOW"


def get_level_color(level):
    return {
        "LOW": "#34d399",
        "MEDIUM": "#fbbf24",
        "HIGH": "#fb923c",
        "CRITICAL": "#fb7185",
    }.get(level, "#22d3ee")


def clean_endpoint(value):
    return str(
        value or "microgrid-simulator"
    ).replace(
        "microgrid-simulator-",
        "",
    )


# ============================================================
# LOAD DATA
# ============================================================

summary = get_json(SUMMARY_URL)
events_payload = get_json(
    RECENT_URL,
    {"limit": 100},
)
health = get_json(HEALTH_URL)
model_info = get_json(MODEL_URL)

if summary is None:
    st.error(
        "AI-CyberShield backend is unavailable. "
        "Check FastAPI and the ALB."
    )
    st.stop()

if isinstance(events_payload, dict):
    events = events_payload.get(
        "events",
        events_payload.get("data", []),
    )
else:
    events = (
        events_payload
        if isinstance(events_payload, list)
        else []
    )

events_df = pd.DataFrame(events)

if not events_df.empty:
    if "timestamp" in events_df.columns:
        events_df["timestamp"] = pd.to_datetime(
            events_df["timestamp"],
            errors="coerce",
        )
        events_df = events_df.sort_values(
            "timestamp"
        )

    for column in [
        "attack_probability",
        "anomaly_score",
        "risk_score",
        "criticality",
    ]:
        if column in events_df.columns:
            events_df[column] = pd.to_numeric(
                events_df[column],
                errors="coerce",
            )


# ============================================================
# SUMMARY
# ============================================================

risk_distribution = (
    summary.get("risk_distribution", {})
    or {}
)

low_count = int(
    risk_distribution.get("LOW", 0) or 0
)
medium_count = int(
    risk_distribution.get("MEDIUM", 0) or 0
)
high_count = int(
    risk_distribution.get("HIGH", 0) or 0
)
critical_count = int(
    risk_distribution.get("CRITICAL", 0) or 0
)

latest = (
    events_df.iloc[-1].to_dict()
    if not events_df.empty
    else {}
)

latest_attack = float(
    latest.get("attack_probability", 0) or 0
)
latest_anomaly = float(
    latest.get("anomaly_score", 0) or 0
)
latest_criticality = float(
    latest.get("criticality", 0.5) or 0.5
)
latest_risk = float(
    latest.get(
        "risk_score",
        summary.get("average_risk_score", 0),
    )
    or 0
)

latest_level = str(
    latest.get("risk_level")
    or get_level(latest_risk)
)

latest_action = str(
    latest.get(
        "recommended_action",
        "NORMAL_OPERATION",
    )
)

level_color = get_level_color(
    latest_level
)

state_text = {
    "LOW": "SYSTEM SECURE",
    "MEDIUM": "ELEVATED RISK",
    "HIGH": "HIGH RISK",
    "CRITICAL": "CRITICAL THREAT",
}.get(
    latest_level,
    "MONITORING",
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
    <div class="topbar">
        <div class="brand">
            <div class="logo">🛡️</div>
            <div>
                <div class="title">AI-CyberShield</div>
                <div class="subtitle">
                    Microgrid Cybersecurity Command Center
                </div>
            </div>
        </div>

        <div class="badges">
            <div class="badge">🟢 AWS ONLINE</div>
            <div class="badge">🧠 AI ENGINE READY</div>
            <div class="badge">📡 AUDIT STREAM LIVE</div>
            <div class="badge">◉ SIMULATED TELEMETRY</div>
            <div class="badge">🔒 PHYSICAL ACTUATION OFF</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# EVENT TICKER
# ============================================================

ticker_items = []

if not events_df.empty:
    for _, row in events_df.tail(16).iterrows():
        score = float(
            row.get("risk_score", 0) or 0
        )
        lvl = get_level(score)

        ticker_items.append(
            f'<span style="color:{get_level_color(lvl)}">◆</span> '
            f'{clean_endpoint(row.get("endpoint"))} '
            f'· {lvl} · Risk {score:.1f}'
        )

ticker = (
    " &nbsp;&nbsp;&nbsp;&nbsp; ".join(ticker_items)
    if ticker_items
    else "Awaiting security telemetry..."
)

st.markdown(
    f"""
    <div class="ticker">
        <div class="ticker-inner">{ticker}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO + SIGNAL MATRIX
# ============================================================

hero_col, signal_col = st.columns(
    [1.1, 0.9],
    gap="medium",
)

with hero_col:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-gridlines"></div>

            <div class="globe">
                <div class="orbit one"></div>
                <div class="orbit two"></div>

                <div
                    class="core-shield"
                    style="
                        box-shadow:
                        0 0 30px {level_color}55,
                        0 0 85px {level_color}20,
                        inset 0 0 35px rgba(255,255,255,.10);
                    "
                >
                    <div class="shield-icon">🛡️</div>
                </div>
            </div>

            <div class="hero-caption">
                <div class="eyebrow">
                    LIVE DEFENSE STATUS
                </div>

                <div
                    class="hero-state"
                    style="color:{level_color};
                           text-shadow:0 0 18px {level_color};"
                >
                    {state_text}
                </div>

                <div class="hero-risk">
                    Threat Index
                    <b>{latest_risk:.2f}/100</b>
                    · Action
                    <b>{latest_action.replace("_", " ")}</b>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with signal_col:
    def signal_block(name, value, scale=1.0):
        width = max(
            0,
            min(
                100,
                float(value) * scale,
            ),
        )

        st.markdown(
            f"""
            <div class="signal">
                <div class="signal-head">
                    <span class="signal-label">{name}</span>
                    <span class="signal-value">{width:.1f}%</span>
                </div>
                <div class="bar">
                    <div
                        class="fill"
                        style="width:{width:.1f}%"
                    ></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="signals">'
        '<div class="section-title">'
        '⚡ Live Signal Matrix'
        '</div>',
        unsafe_allow_html=True,
    )

    signal_block(
        "Attack Probability",
        latest_attack,
        100,
    )

    signal_block(
        "Anomaly Score",
        latest_anomaly,
        100,
    )

    signal_block(
        "Asset Criticality",
        latest_criticality,
        100,
    )

    signal_block(
        "Composite Risk",
        latest_risk,
        1,
    )

    latest_endpoint = clean_endpoint(
        latest.get(
            "endpoint",
            "microgrid-simulator",
        )
    )

    attack_signal = (
        "DETECTED"
        if bool(
            latest.get(
                "attack_prediction",
                False,
            )
        )
        else "CLEAR"
    )

    anomaly_signal = (
        "ANOMALY"
        if bool(
            latest.get(
                "anomaly_prediction",
                False,
            )
        )
        else "NORMAL"
    )

    st.markdown(
        f"""
            <div class="live-grid">
                <div class="live-cell">
                    <div class="live-label">Endpoint</div>
                    <div class="live-value">{latest_endpoint}</div>
                </div>

                <div class="live-cell">
                    <div class="live-label">Defense Mode</div>
                    <div class="live-value">{latest_action.replace("_", " ")}</div>
                </div>

                <div class="live-cell">
                    <div class="live-label">Attack Signal</div>
                    <div class="live-value">{attack_signal}</div>
                </div>

                <div class="live-cell">
                    <div class="live-label">Anomaly Signal</div>
                    <div class="live-value">{anomaly_signal}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SECURITY STATUS + KPIs
# ============================================================

overall_status = (
    "CRITICAL SECURITY EVENT DETECTED"
    if critical_count > 0
    else "HIGH-RISK SECURITY EVENT DETECTED"
    if high_count > 0
    else "ELEVATED SECURITY ACTIVITY"
    if medium_count > 0
    else "NO HIGH OR CRITICAL SECURITY EVENTS DETECTED"
)

overall_color = (
    "#fb7185"
    if critical_count > 0
    else "#fb923c"
    if high_count > 0
    else "#fbbf24"
    if medium_count > 0
    else "#34d399"
)

st.markdown(
    f"""
    <div class="status">
        <div>
            <span
                style="
                    display:inline-block;
                    width:9px;
                    height:9px;
                    border-radius:50%;
                    background:{overall_color};
                    box-shadow:0 0 16px {overall_color};
                    margin-right:8px;
                "
            ></span>

            <span
                class="status-text"
                style="color:{overall_color};"
            >
                {overall_status}
            </span>
        </div>

        <div class="status-meta">
            AWS audit database · 5s refresh · ML + safety policy
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

kpis = [
    (
        "TOTAL EVENTS",
        summary.get("total_events", 0),
        "audit history",
    ),
    (
        "ATTACK SIGNALS",
        summary.get("attack_events", 0),
        "XGBoost detector",
    ),
    (
        "ANOMALIES",
        summary.get("anomaly_events", 0),
        "Isolation Forest",
    ),
    (
        "ACTIVE ALERTS",
        summary.get("alert_events", 0),
        "safety engine",
    ),
    (
        "AVERAGE RISK",
        f"{float(summary.get('average_risk_score', 0) or 0):.2f}",
        "0–100 composite",
    ),
]

kpi_html = '<div class="metric-grid">'

for label, value, note in kpis:
    kpi_html += f"""
        <div class="metric">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
    """

kpi_html += "</div>"

st.markdown(
    kpi_html,
    unsafe_allow_html=True,
)


# ============================================================
# THREAT RADAR
# ============================================================

st.markdown(
    '<div class="panel">'
    '<div class="section-title">'
    '🌌 THREAT RADAR · ATTACK × ANOMALY × RISK'
    '</div>',
    unsafe_allow_html=True,
)

if not events_df.empty:
    threat = events_df.tail(100).copy()

    for col in [
        "attack_probability",
        "anomaly_score",
        "risk_score",
    ]:
        if col not in threat.columns:
            threat[col] = 0

        threat[col] = pd.to_numeric(
            threat[col],
            errors="coerce",
        ).fillna(0)

    threat["risk_level"] = threat[
        "risk_score"
    ].apply(get_level)

    threat_colors = threat[
        "risk_level"
    ].map(
        {
            "LOW": "#34d399",
            "MEDIUM": "#fbbf24",
            "HIGH": "#fb923c",
            "CRITICAL": "#fb7185",
        }
    ).fillna("#22d3ee")

    threat["bubble_size"] = (
        10
        + threat["risk_score"].clip(
            0, 100
        ) / 4
    )

    threat["hover"] = threat.apply(
        lambda row: (
            f"<b>{clean_endpoint(row.get('endpoint'))}</b><br>"
            f"Attack: {pct(row.get('attack_probability')):.1f}%<br>"
            f"Anomaly: {pct(row.get('anomaly_score')):.1f}%<br>"
            f"Risk: {float(row.get('risk_score', 0)):.1f}<br>"
            f"Level: {row.get('risk_level', 'LOW')}"
        ),
        axis=1,
    )

    fig = go.Figure()

    # Main threat field.
    fig.add_trace(
        go.Scatter(
            x=threat["attack_probability"],
            y=threat["anomaly_score"],
            mode="markers",
            marker={
                "size": threat["bubble_size"],
                "color": threat_colors.tolist(),
                "opacity": 0.88,
                "line": {
                    "width": 1,
                    "color": "rgba(255,255,255,.35)",
                },
            },
            text=threat["hover"],
            hovertemplate="%{text}<extra></extra>",
            name="Security Events",
        )
    )

    # Risk contour bands as background guides.
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line={
            "color": "rgba(52,211,153,.18)",
            "width": 1,
        },
        fillcolor="rgba(52,211,153,.015)",
    )

    fig.add_shape(
        type="rect",
        x0=0.5,
        y0=0.5,
        x1=1,
        y1=1,
        line={
            "color": "rgba(251,146,60,.22)",
            "width": 1,
        },
        fillcolor="rgba(251,146,60,.03)",
    )

    fig.add_shape(
        type="line",
        x0=0.7,
        y0=0,
        x1=0.7,
        y1=1,
        line={
            "color": "rgba(251,113,133,.20)",
            "dash": "dash",
        },
    )

    fig.add_shape(
        type="line",
        x0=0,
        y0=0.7,
        x1=1,
        y1=0.7,
        line={
            "color": "rgba(251,113,133,.20)",
            "dash": "dash",
        },
    )

    fig.add_annotation(
        x=0.86,
        y=0.88,
        text="HIGH THREAT ZONE",
        showarrow=False,
        font={
            "color": "#fb7185",
            "size": 11,
        },
    )

    fig.update_layout(
        height=470,
        margin=dict(
            l=20,
            r=20,
            t=15,
            b=30,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "#a8bed3",
            "size": 10,
        },
        xaxis={
            "title": "Attack Probability",
            "range": [0, 1],
            "gridcolor": "rgba(96,165,250,.10)",
            "zeroline": False,
        },
        yaxis={
            "title": "Anomaly Score",
            "range": [0, 1],
            "gridcolor": "rgba(167,139,250,.10)",
            "zeroline": False,
        },
        showlegend=False,
        hoverlabel={
            "bgcolor": "#071321",
            "font": {"color": "#eef8ff"},
        },
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )

else:
    st.info(
        "Waiting for telemetry events to populate the threat radar."
    )

st.markdown(
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# ALERT CENTER + THREAT PULSE
# ============================================================

left, right = st.columns(
    [1, 1],
    gap="large",
)

with left:
    st.markdown(
        '<div class="panel">'
        '<div class="section-title">'
        '🚨 ALERT COMMAND CENTER'
        '</div>',
        unsafe_allow_html=True,
    )

    alerts_df = events_df.copy()

    if (
        not alerts_df.empty
        and "alert" in alerts_df.columns
    ):
        alerts_df = alerts_df[
            alerts_df["alert"]
            .fillna(False)
            .astype(bool)
        ].sort_values(
            "timestamp",
            ascending=False,
        )
    else:
        alerts_df = pd.DataFrame()

    if alerts_df.empty:
        st.success(
            "No active alerts in the latest audit stream."
        )
    else:
        for _, row in alerts_df.head(8).iterrows():
            lvl = str(
                row.get(
                    "risk_level",
                    get_level(
                        row.get(
                            "risk_score",
                            0,
                        )
                    ),
                )
            )

            color = get_level_color(lvl)

            timestamp = pd.to_datetime(
                row.get("timestamp"),
                errors="coerce",
            )

            timestamp_text = (
                timestamp.strftime(
                    "%d %b %H:%M:%S"
                )
                if pd.notna(timestamp)
                else "—"
            )

            st.markdown(
                f"""
                <div
                    class="alert"
                    style="
                        border-left-color:{color};
                        box-shadow:
                        inset 8px 0 22px -20px {color};
                    "
                >
                    <div class="alert-head">
                        <span>{lvl} SECURITY EVENT</span>
                        <span>{timestamp_text}</span>
                    </div>

                    <div class="alert-title">
                        🛰️ {clean_endpoint(row.get("endpoint"))}
                    </div>

                    <div class="alert-grid">
                        <div class="alert-box">
                            <span>Risk</span>
                            <b>{float(row.get("risk_score", 0) or 0):.2f}</b>
                        </div>

                        <div class="alert-box">
                            <span>Attack</span>
                            <b>{pct(row.get("attack_probability", 0)):.1f}%</b>
                        </div>

                        <div class="alert-box">
                            <span>Anomaly</span>
                            <b>{pct(row.get("anomaly_score", 0)):.1f}%</b>
                        </div>
                    </div>

                    <div
                        style="
                            color:#93abc1;
                            font-size:10px;
                            margin-top:9px;
                        "
                    >
                        Action:
                        <b style="color:#eef8ff;">
                            {str(row.get("recommended_action", "MONITOR")).replace("_", " ")}
                        </b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


with right:
    st.markdown(
        '<div class="panel">'
        '<div class="section-title">'
        '📡 THREAT PULSE'
        '</div>',
        unsafe_allow_html=True,
    )

    if (
        not events_df.empty
        and "timestamp" in events_df.columns
    ):
        pulse = events_df.tail(100).copy()

        for col in [
            "risk_score",
            "attack_probability",
            "anomaly_score",
        ]:
            if col not in pulse.columns:
                pulse[col] = 0

            pulse[col] = pd.to_numeric(
                pulse[col],
                errors="coerce",
            ).fillna(0)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=pulse["timestamp"],
                y=pulse["risk_score"],
                mode="lines+markers",
                name="Risk",
                line={
                    "width": 3,
                    "color": "#22d3ee",
                },
                marker={
                    "size": 5,
                },
            )
        )

        fig.add_trace(
            go.Scatter(
                x=pulse["timestamp"],
                y=pulse["attack_probability"] * 100,
                mode="lines",
                name="Attack",
                line={
                    "width": 2,
                    "dash": "dot",
                    "color": "#fb7185",
                },
            )
        )

        fig.add_trace(
            go.Scatter(
                x=pulse["timestamp"],
                y=pulse["anomaly_score"] * 100,
                mode="lines",
                name="Anomaly",
                line={
                    "width": 2,
                    "dash": "dash",
                    "color": "#a78bfa",
                },
            )
        )

        fig.update_layout(
            height=400,
            margin=dict(
                l=10,
                r=10,
                t=18,
                b=20,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={
                "color": "#a8bed3",
                "size": 10,
            },
            xaxis={
                "gridcolor": "rgba(120,180,255,.07)",
                "zeroline": False,
            },
            yaxis={
                "gridcolor": "rgba(120,180,255,.07)",
                "zeroline": False,
            },
            legend={
                "orientation": "h",
                "y": 1.02,
                "x": 0,
            },
        )

        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "displayModeBar": False,
            },
        )
    else:
        st.info(
            "Threat pulse will appear after telemetry arrives."
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# RISK + SCENARIOS
# ============================================================

left, right = st.columns(
    [.75, 1.25],
    gap="large",
)

with left:
    st.markdown(
        '<div class="panel">'
        '<div class="section-title">'
        '🎯 RISK DISTRIBUTION'
        '</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure(
        go.Bar(
            x=[
                low_count,
                medium_count,
                high_count,
                critical_count,
            ],
            y=[
                "LOW",
                "MEDIUM",
                "HIGH",
                "CRITICAL",
            ],
            orientation="h",
            marker={
                "color": [
                    "#34d399",
                    "#fbbf24",
                    "#fb923c",
                    "#fb7185",
                ]
            },
            text=[
                low_count,
                medium_count,
                high_count,
                critical_count,
            ],
            textposition="outside",
        )
    )

    fig.update_layout(
        height=350,
        margin=dict(
            l=0,
            r=25,
            t=15,
            b=5,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "#a8bed3",
            "size": 10,
        },
        xaxis={
            "gridcolor": "rgba(120,180,255,.07)",
            "zeroline": False,
        },
        yaxis={
            "gridcolor": "rgba(120,180,255,.05)",
        },
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displayModeBar": False,
        },
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


with right:
    st.markdown(
        '<div class="panel">'
        '<div class="section-title">'
        '🎯 SCENARIO THREAT MATRIX'
        '</div>',
        unsafe_allow_html=True,
    )

    if not events_df.empty:
        work = events_df.copy()

        work["scenario"] = (
            work.get(
                "endpoint",
                pd.Series(
                    "microgrid-simulator",
                    index=work.index,
                ),
            )
            .astype(str)
            .map(clean_endpoint)
        )

        for col in [
            "attack_prediction",
            "anomaly_prediction",
            "alert",
            "risk_score",
        ]:
            if col not in work.columns:
                work[col] = 0

        grouped = (
            work.groupby("scenario")
            .agg(
                events=("scenario", "size"),
                attacks=(
                    "attack_prediction",
                    "sum",
                ),
                anomalies=(
                    "anomaly_prediction",
                    "sum",
                ),
                alerts=("alert", "sum"),
                max_risk=(
                    "risk_score",
                    "max",
                ),
            )
            .reset_index()
            .sort_values(
                "max_risk",
                ascending=False,
            )
        )

        html = '<div class="scenario-grid">'

        for _, row in grouped.iterrows():
            score = float(
                row.get(
                    "max_risk",
                    0,
                )
                or 0
            )

            lvl = get_level(score)
            color = get_level_color(lvl)

            html += f"""
            <div
                class="scenario"
                style="border-color:{color}44;"
            >
                <div class="scenario-name">
                    {row["scenario"]}
                </div>

                <div
                    class="scenario-level"
                    style="color:{color};"
                >
                    {lvl}
                </div>

                <div class="scenario-meta">
                    Events {int(row["events"])} ·
                    Attacks {int(row["attacks"])} ·
                    Anomalies {int(row["anomalies"])}
                    <br>
                    Alerts {int(row["alerts"])} ·
                    Max Risk {score:.1f}
                </div>
            </div>
            """

        html += "</div>"

        st.markdown(
            html,
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "Scenario matrix will populate after telemetry arrives."
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# DEFENSE TOPOLOGY
# ============================================================

st.markdown(
    '<div class="panel">'
    '<div class="section-title">'
    '🌐 DEFENSE TOPOLOGY'
    '</div>',
    unsafe_allow_html=True,
)

nodes = [
    ("📡", "Telemetry"),
    ("🌐", "AWS ALB"),
    ("⚙️", "FastAPI"),
    ("🧠", "ML Engine"),
    ("⚖️", "Risk Engine"),
    ("🛡️", "Safety Engine"),
    ("🗄️", "PostgreSQL"),
]

topology = '<div class="topology">'

for index, (icon, name) in enumerate(nodes):
    topology += f"""
        <div class="node">
            <div class="node-icon">{icon}</div>
            <div class="node-name">{name}</div>
        </div>
    """

    if index < len(nodes) - 1:
        topology += '<div class="arrow"></div>'

topology += "</div>"

st.markdown(
    topology,
    unsafe_allow_html=True,
)

st.markdown(
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL + SAFETY
# ============================================================

st.markdown(
    '<div class="panel">'
    '<div class="section-title">'
    '🧠 MODEL INTELLIGENCE & SAFETY'
    '</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)

base_features = (
    model_info.get(
        "base_features",
        "—",
    )
    if isinstance(model_info, dict)
    else "—"
)

temporal_features = (
    model_info.get(
        "temporal_features",
        "—",
    )
    if isinstance(model_info, dict)
    else "—"
)

total_features = (
    model_info.get(
        "total_model_features",
        "—",
    )
    if isinstance(model_info, dict)
    else "—"
)

m1.metric(
    "Base Features",
    base_features,
    width="stretch",
)

m2.metric(
    "Temporal Features",
    temporal_features,
    width="stretch",
)

m3.metric(
    "Total Model Features",
    total_features,
    width="stretch",
)

m4.metric(
    "Physical Breaker Control",
    "DISABLED",
    width="stretch",
)

st.info(
    "ML performs detection and risk assessment. "
    "The deterministic safety layer does not directly "
    "actuate physical breakers."
)

st.markdown(
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# LATEST EVENTS
# ============================================================

st.markdown(
    '<div class="panel">'
    '<div class="section-title">'
    '📋 LATEST SECURITY EVENTS'
    '</div>',
    unsafe_allow_html=True,
)

if not events_df.empty:
    table_cols = [
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

    available_cols = [
        col
        for col in table_cols
        if col in events_df.columns
    ]

    table = (
        events_df
        .sort_values(
            "timestamp",
            ascending=False,
        )[available_cols]
        .head(100)
        .copy()
    )

    if "timestamp" in table.columns:
        table["timestamp"] = table[
            "timestamp"
        ].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    if "attack_probability" in table.columns:
        table["attack_probability"] = (
            table["attack_probability"] * 100
        ).round(2)

    if "anomaly_score" in table.columns:
        table["anomaly_score"] = (
            table["anomaly_score"] * 100
        ).round(2)

    if "risk_score" in table.columns:
        table["risk_score"] = (
            table["risk_score"]
            .round(2)
        )

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
    )
else:
    st.info(
        "No audit events available yet."
    )

st.markdown(
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="footer">
        AI-CyberShield · XGBoost + Isolation Forest + SHAP ·
        Last refresh {datetime.now().strftime("%d %b %Y %H:%M:%S")} ·
        Simulated telemetry for controlled demonstration
    </div>
    """,
    unsafe_allow_html=True,
)
