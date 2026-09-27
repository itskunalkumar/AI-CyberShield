# AI-CyberShield

ML-powered cybersecurity smart controller for a grid-connected microgrid. Built on the Data4Cyber dataset.

## Pipeline

1. **Data loading**: Reads primary scenarios S0-S6 from Data4Cyber `dataset.csv` files.
2. **Feature selection**: Numeric telemetry only; label and attacker annotation columns are excluded to prevent leakage.
3. **Attack detection**: XGBoost binary classifier with balanced sample weights and a validation-tuned decision threshold.
4. **Attack-family classification**: XGBoost multiclass model, evaluated on held-out scenarios.
5. **Anomaly detection**: Isolation Forest trained on benign baseline (S0) only.
6. **Risk score (0-100)**: 70% attack probability, 20% anomaly score, 10% asset criticality.
7. **Safety engine**: Deterministic policy mapping risk to NORMAL, MONITOR, PROTECTIVE or SAFE_MODE. **ML never directly operates a physical breaker.**
8. **Serving**: FastAPI inference API and Streamlit command-center dashboard.

## Evaluation design

- **Detector**: scenario-level holdout. Train on S0, S1, S3, S4 (S4 used for threshold selection); test on S2, S5, S6.
- **Attack-family classifier**: evaluated on held-out S2 and S5. S6 (MQTT) is excluded from training because that family is unseen; it is not reported as a classifier result.

## Project structure

```text
AI-CyberShield/
├── data/
│   ├── raw/                 # extracted Data4Cyber dataset (not committed)
│   └── processed/
├── notebooks/               # exploration (see notebooks/README.md)
├── src/
│   ├── config.py            # paths, scenario splits
│   ├── data_loader.py       # scenario CSV loading
│   ├── features.py          # leakage-aware feature selection
│   ├── preprocessing.py     # inf/NaN handling with training medians
│   ├── train.py            # trains and evaluates all models
│   ├── predict.py          # inference pipeline
│   ├── risk_engine.py      # risk score and level
│   └── safety_engine.py    # policy engine
├── models/                  # trained artifacts (generated)
├── reports/metrics.json     # generated evaluation report
├── api/main.py              # FastAPI service
├── dashboard/app.py         # Streamlit dashboard
├── scripts/prepare_data.py  # extract dataset archive
├── tests/                   # unit and API tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

```bash
python scripts/prepare_data.py /path/to/data4cyber_dataset.zip
```

Or set `DATA_ROOT` to the extracted folder (see `.env.example`).

## Train

```bash
python -m src.train
```

Writes `models/*.joblib` and `reports/metrics.json`.

## Run

```bash
pytest -q                                          # tests
uvicorn api.main:app --reload                      # API, docs at /docs
streamlit run dashboard/app.py                     # dashboard
docker compose up --build                          # containerised
```

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me" \
  -d '{"features": {"BSS-Inverter-AC-Meter.freq": 50.0}, "criticality": 0.8}'
```

Missing features are filled with training medians. The `X-API-Key` header is only required when `API_KEY` is set.

## Limitations

- Results are for a research prototype on one dataset; they are not a production security evaluation.
- Cross-scenario generalisation is limited: unseen attack families are not reliably detected.
- The safety engine produces logical responses only and does not send physical control commands.
