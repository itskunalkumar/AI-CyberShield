<div align="center">

<img src="assets/banner.svg" alt="AI-CyberShield" width="100%"/>

<br/>

[![CI](https://github.com/itskunalkumar/AI-CyberShield/actions/workflows/ci.yml/badge.svg)](https://github.com/itskunalkumar/AI-CyberShield/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Detector-EB5E28)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Isolation%20Forest-F7931E?logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-7c3aed)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Audit%20Log-4169E1?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-ECR%20%2B%20ALB-FF9900?logo=amazonaws&logoColor=white)

**Detect. Score. Explain. Protect.**
An end-to-end ML security pipeline that watches microgrid telemetry, scores every event from 0 to 100, explains *why*, and hands a deterministic safety policy to the operator, **without ever touching a physical breaker.**

[Architecture](#-architecture) ·
[Quick start](#-quick-start) ·
[Dashboard](#-command-center-dashboard) ·
[API](#-api-reference) ·
[Results](#-model-performance) ·
[Limitations](#-limitations--responsible-use)

</div>

---

## ✨ At a glance

| 🧪 Attack scenarios | 🧠 Temporal features | 🎯 Precision (held-out) | 🛡️ Safety levels |
| :---: | :---: | :---: | :---: |
| **7** (S0 - S6) | **845** | **0.97** | **4** deterministic |

> Built on the **Data4Cyber** dataset: benign baseline plus Industroyer/Modbus, ARP-spoofing false-data-injection and MQTT supply-chain attacks against a grid-connected microgrid.

---

## 🔥 Features

- **Two-signal detection**: a supervised **XGBoost** attack detector and an unsupervised **Isolation Forest** trained only on benign behaviour.
- **Temporal intelligence**: rolling and delta features over telemetry history (845 engineered features) instead of single-row snapshots.
- **Explainable by design**: every prediction ships with a **SHAP** explanation of the features that drove it.
- **Transparent risk score**: one 0 - 100 number from a documented, auditable formula.
- **Deterministic safety layer**: risk maps to a fixed operator policy; unknown states **fail safe**.
- **Full audit trail**: every decision is written to **PostgreSQL** and surfaced live on the dashboard.
- **Live command center**: animated Streamlit dashboard with a 3D threat space, alert feed, scenario matrix and SHAP view.
- **Production plumbing**: Dockerised services, API-key auth, unit tests, and a GitHub Actions pipeline that pushes images to **Amazon ECR**.

---

## 🏗 Architecture

```mermaid
flowchart LR
    T["📡 Telemetry<br/>(microgrid / simulator)"] --> A["⚙️ FastAPI<br/>/api/v1/predict"]
    A --> F["🧮 Temporal<br/>feature engineering"]
    F --> X["🌲 XGBoost<br/>attack probability"]
    F --> I["🧭 Isolation Forest<br/>anomaly score"]
    X --> R["⚖️ Risk Engine<br/>score 0-100"]
    I --> R
    R --> S["🛡️ Safety Engine<br/>deterministic policy"]
    S --> E["🔍 SHAP<br/>explanation"]
    E --> D[("🗄️ PostgreSQL<br/>audit log")]
    D --> UI["🖥️ Streamlit<br/>command center"]
```

**Design principle:** machine learning *assesses*; a deterministic policy *decides*; the human *acts*. The `physical_breaker_control` flag in every response is always `false`.

---

## ⚖️ Risk engine

```text
risk = 100 × ( 0.60 × attack_probability
             + 0.30 × anomaly_score
             + 0.10 × asset_criticality )
```

| Risk score | Level | Safety action | Automated control | Operator confirmation | Logical isolation | Alert |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| `0 - 29` | 🟢 LOW | `NORMAL_OPERATION` | ✅ allowed | - | - | - |
| `30 - 59` | 🟡 MEDIUM | `MONITOR` | ✅ allowed | - | - | 🔔 |
| `60 - 79` | 🟠 HIGH | `PROTECTIVE_MODE` | ⛔ blocked | ✅ required | ✅ | 🔔 |
| `80 - 100` | 🔴 CRITICAL | `SAFE_STATE_REVIEW` | ⛔ blocked | ✅ required | ✅ | 🔔 |

Any unrecognised risk level falls back to `SAFE_STATE_REVIEW`. The system never allows automated control when it cannot interpret its own output.

---

## 📊 Model performance

Temporal XGBoost attack detector (845 features, decision threshold `0.30`):

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Validation | 0.847 | 0.839 | 0.865 | 0.852 | 0.876 | 0.858 |
| **Held-out test** (7,198 rows, S3 - S6) | 0.548 | **0.970** | 0.449 | 0.613 | 0.856 | **0.960** |

<details>
<summary><b>Per-scenario results on the held-out test set</b></summary>

| Scenario | Attack type | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: |
| S3 · ARP spoof, BSS meter half values | False data injection | 1.000 | 0.624 | 0.769 |
| S4 · ARP spoof, loads + PV two-phase | False data injection | 0.925 | 0.453 | 0.609 |
| S5 · ARP spoof, loads + PV + BSS two-phase | False data injection | 0.873 | 0.174 | 0.290 |
| S6 · MQTT supply-chain compromise | Price-signal manipulation | 1.000 | 0.482 | 0.651 |

</details>

Isolation Forest (trained on 3,545 benign rows, threshold `0.20`), used as a supporting signal at 30% of the risk score:

| Split | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: |
| Validation | 0.540 | 0.843 | 0.659 |
| Held-out test | 0.785 | 0.506 | 0.615 |

> **Reading these numbers honestly:** when the detector raises an alert it is almost always right (precision ≈ 0.97), but it misses a large share of attacks from scenarios it never trained on (recall ≈ 0.45). See [Limitations](#-limitations--responsible-use).

---

## 🖥 Command center dashboard

A live, auto-refreshing (5 s) Streamlit interface built for a security operations feel:

| Panel | What it shows |
| :--- | :--- |
| 🛡️ **Live Defense Status** | Animated shield reflecting the current threat level |
| ⚡ **Signal Matrix** | Attack probability, anomaly score, asset criticality and composite risk |
| 🌌 **3D Threat Space** | Events plotted as Attack × Anomaly × Risk |
| 🚨 **Alert Command Center** | Latest alerts with risk, attack and anomaly readings |
| 📡 **Threat Pulse** | Risk, attack and anomaly trends over time |
| 🎯 **Scenario Threat Matrix** | Per-scenario events, attacks, anomalies, alerts and peak risk |
| 🌐 **Defense Topology** | Telemetry → ALB → FastAPI → ML → Risk → Safety → PostgreSQL |
| 🔍 **Explainable AI** | SHAP feature contributions for the latest event |
| 📋 **Audit table** | The most recent 100 security events |

<!-- Add a screenshot or GIF here:
<p align="center"><img src="docs/dashboard.png" alt="AI-CyberShield dashboard" width="100%"/></p>
-->

---

## 🚀 Quick start

### Option A · Docker (recommended)

```bash
git clone https://github.com/itskunalkumar/AI-CyberShield.git
cd AI-CyberShield

cp .env.example .env          # set API_KEY (and your own database credentials)
docker compose up --build
```

| Service | URL |
| :--- | :--- |
| 🖥️ Dashboard (Local) | http://localhost:8501 |
| 🖥️ Dashboard (AWS Live) | http://ai-cybershield-alb-870998626.ap-south-1.elb.amazonaws.com/ |
| ⚙️ API + interactive docs (Local) | http://localhost:8001/docs |
| ⚙️ API + interactive docs (AWS Live) | http://ai-cybershield-alb-870998626.ap-south-1.elb.amazonaws.com/docs |
| 🗄️ PostgreSQL (Local) | `localhost:5432` |
| 🗄️ PostgreSQL (AWS RDS) | Managed PostgreSQL on AWS RDS |

### Option B · Local

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export DATABASE_URL="postgresql+psycopg2://USER:PASSWORD@localhost:5432/ai_cybershield"
export API_KEY="change-me"

uvicorn src.api:app --port 8001 --reload      # terminal 1: API
streamlit run dashboard/app.py                # terminal 2: dashboard
```

> `DATABASE_URL` is required. The API refuses to start without it.

### Feed it live telemetry

Replay the cleaned dataset through the API and watch the dashboard react:

```bash
API_KEY=change-me DELAY_SECONDS=1 python -m src.telemetry_simulator
```

The simulator sends 100 new rows per call (with 19 rows of history for temporal context) and creates one audit event per batch. Tune it with `API_URL`, `BATCH_SIZE`, `CONTEXT_ROWS`, `DELAY_SECONDS`, `CRITICALITY` and `MAX_NEW_ROWS`.

### Train from scratch

```bash
python scripts/prepare_data.py /path/to/data4cyber_dataset.zip     # or set DATA_ROOT
python -m src.train                                                # baseline models
python -m src.train_temporal                                       # temporal detector
python -m src.train_anomaly                                        # Isolation Forest
```

---

## 🔌 API reference

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `GET` | `/` | Service information |
| `GET` | `/health` | Health check |
| `GET` | `/model-info` | Loaded model and feature counts |
| `POST` | `/api/v1/predict` | Full security assessment for a telemetry window |
| `GET` | `/api/v1/audit/recent` | Most recent audit events (`limit` query parameter) |
| `GET` | `/api/v1/audit/summary` | Aggregate counts and risk distribution |

Authentication: send `X-API-Key: <your key>`. The header is required whenever the `API_KEY` environment variable is set.

```bash
curl -X POST http://localhost:8001/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me" \
  -d '{
        "endpoint": "microgrid-simulator",
        "criticality": 0.8,
        "telemetry": [ { "BSS-Inverter-AC-Meter.freq": 50.0 } ]
      }'
```

<details>
<summary><b>Response fields</b></summary>

| Field | Meaning |
| :--- | :--- |
| `attack_probability`, `attack_prediction` | XGBoost output and thresholded decision |
| `anomaly_score`, `anomaly_prediction`, `anomaly_threshold` | Isolation Forest output |
| `criticality` | Asset criticality used in the score |
| `risk_score`, `risk_level`, `recommended_action` | Risk engine result |
| `allow_automated_control`, `operator_confirmation_required`, `logical_isolation`, `alert` | Safety engine policy |
| `risk_reason`, `safety_reason` | Human-readable explanations |
| `shap_explanation` | Top feature contributions |
| `physical_breaker_control` | Always `false` |

</details>

---

## 🗂 Project structure

```text
AI-CyberShield/
├── src/
│   ├── api.py                  # FastAPI service (predict, audit, health)
│   ├── inference.py            # Unified pipeline: features → models → risk → safety → SHAP
│   ├── temporal_features.py    # Rolling / delta feature engineering
│   ├── risk_engine.py          # 0-100 risk score, levels, recommended actions
│   ├── safety_engine.py        # Deterministic, fail-safe policy engine
│   ├── explainability.py       # SHAP explanations
│   ├── train*.py               # Detector, temporal, robust and anomaly training
│   ├── telemetry_simulator.py  # Replays the dataset through the API
│   ├── database.py             # SQLAlchemy + PostgreSQL
│   └── audit_repository.py     # Audit-log persistence and queries
├── dashboard/app.py            # Streamlit command center
├── api/main.py                 # Lightweight API entry point
├── models/                     # Trained artifacts and metrics
├── tests/                      # Unit tests
├── scripts/prepare_data.py     # Extracts the dataset archive
├── .github/workflows/ci.yml    # Test → Docker build → push to Amazon ECR
├── Dockerfile
└── docker-compose.yml          # API + dashboard + PostgreSQL
```

---

## 🧪 Testing & CI/CD

```bash
pytest -q
```

Every push and pull request to `main` runs the test suite. Pushes to `main` that pass then build the Docker image and push it to **Amazon ECR** (`ap-south-1`), tagged with the commit SHA and `latest`, authenticated through GitHub OIDC rather than stored keys.

---

## 🔬 Evaluation design

Models are evaluated by **scenario**, not by random row split, so adjacent timestamps from one scenario never leak between train and test.

| Scenario | Family |
| :--- | :--- |
| `S0` benign baseline | Benign |
| `S1` `S2` Industroyer (PV, BSS) | Modbus attack |
| `S3` `S4` `S5` ARP spoofing | Man-in-the-middle false data injection |
| `S6` MQTT supply chain | Price-signal manipulation |

Feature selection uses numeric telemetry only. Labels and attacker-annotation columns are excluded to prevent leakage. The Isolation Forest is trained on the benign baseline only.

---

## ⚠️ Limitations & responsible use

- This is a **research prototype** on a single dataset. It is not a production security evaluation and should not be the only defence for real infrastructure.
- **Generalisation is limited.** Recall on held-out scenarios is ~0.45, and attack families not seen in training are not reliably detected.
- The Isolation Forest is a **weak standalone detector** (test ROC-AUC below 0.5); it works as a supporting signal in the risk score, not on its own.
- The safety engine produces **logical responses only**. It never sends physical control commands.
- Telemetry in the demo is simulated.

---

## 🛣 Roadmap

- [ ] Raise recall on unseen attack families (more scenarios, domain-shift analysis)
- [ ] Add API and inference tests to the suite
- [ ] Drift monitoring and scheduled model retraining
- [ ] Alert routing (email / Slack) for HIGH and CRITICAL events
- [ ] Dashboard screenshots and a short demo GIF

---

## 👤 Author

**Kunal Kumar**
Mechanical Engineering graduate transitioning into Data Science & ML Engineering
🔗 [GitHub](https://github.com/itskunalkumar)

<div align="center">

<sub>Built with 🛡️ for safer, smarter energy systems.</sub>

</div>
