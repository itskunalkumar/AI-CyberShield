"""
AI-CyberShield
Unified Security Inference Pipeline

Pipeline:

Telemetry History
        |
        v
Temporal Feature Engineering
        |
        +----------------------+
        |                      |
        v                      v
XGBoost Attack Detector    Isolation Forest
        |                      |
        v                      v
Attack Probability        Anomaly Score
        |                      |
        +----------+-----------+
                   |
                   v
              Risk Engine
                   |
                   v
             Safety Engine
                   |
                   v
          SHAP Explainability
                   |
                   v
          Final Security Result

IMPORTANT:
This system provides a security recommendation.
It does NOT directly operate physical breakers.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from .risk_engine import RiskEngine
from .safety_engine import SafetyEngine


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "temporal_features.json"
)

XGB_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "xgboost_temporal_detector.json"
)

ISOLATION_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "isolation_forest.joblib"
)

ANOMALY_SCALER_PATH = (
    BASE_DIR
    / "models"
    / "anomaly_scaler.joblib"
)

ANOMALY_CALIBRATION_PATH = (
    BASE_DIR
    / "models"
    / "anomaly_calibration.json"
)


# Artifacts are loaded only when inference is requested. This keeps importing
# the module safe in environments that do not include the trained model files.
BASE_FEATURES = None
TEMPORAL_FEATURES = None
ALL_FEATURES = None
xgb_model = None
isolation_model = None
anomaly_scaler = None
CALIBRATION_MIN = None
CALIBRATION_MAX = None
ANOMALY_THRESHOLD = None
shap_explainer = None
risk_engine = RiskEngine()
safety_engine = SafetyEngine()


def _load_feature_config():
    global BASE_FEATURES, TEMPORAL_FEATURES, ALL_FEATURES
    if BASE_FEATURES is not None:
        return

    with open(FEATURES_PATH, "r", encoding="utf-8") as feature_file:
        feature_config = json.load(feature_file)

    BASE_FEATURES = feature_config["base_features"]
    TEMPORAL_FEATURES = feature_config["temporal_features"]
    ALL_FEATURES = feature_config["all_features"]


def _load_artifacts():
    global xgb_model, isolation_model, anomaly_scaler
    global CALIBRATION_MIN, CALIBRATION_MAX, ANOMALY_THRESHOLD
    global shap_explainer

    if xgb_model is not None:
        return

    _load_feature_config()

    model = xgb.XGBClassifier()
    model.load_model(XGB_MODEL_PATH)
    model_features = model.get_booster().feature_names

    if model_features is None:
        raise ValueError("XGBoost model does not contain feature names.")
    if len(model_features) != len(ALL_FEATURES):
        raise ValueError(
            "XGBoost feature count mismatch: "
            f"model={len(model_features)}, config={len(ALL_FEATURES)}"
        )
    if model_features != ALL_FEATURES:
        raise ValueError(
            "XGBoost feature order does not match temporal_features.json"
        )

    with open(ANOMALY_CALIBRATION_PATH, "r", encoding="utf-8") as calibration_file:
        anomaly_calibration = json.load(calibration_file)

    xgb_model = model
    isolation_model = joblib.load(ISOLATION_MODEL_PATH)
    anomaly_scaler = joblib.load(ANOMALY_SCALER_PATH)
    CALIBRATION_MIN = float(anomaly_calibration["min"])
    CALIBRATION_MAX = float(anomaly_calibration["max"])
    ANOMALY_THRESHOLD = float(anomaly_calibration["threshold"])
    shap_explainer = shap.TreeExplainer(xgb_model)


# ============================================================
# TEMPORAL FEATURE ENGINEERING
# ============================================================

def create_temporal_features(
    telemetry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create temporal features required by the trained
    AI-CyberShield temporal XGBoost model.

    IMPORTANT
    ---------
    - Uses only historical/past observations.
    - Rolling features use shift(1), so the current observation
      is NOT included in its own rolling statistics.
    - Supports scenario-based historical telemetry.
    - Produces exactly the 845 features expected by XGBoost.
    """

    df = telemetry.copy()

    if df.empty:
        raise ValueError("Telemetry dataframe is empty.")

    _load_feature_config()

    # --------------------------------------------------------
    # 1. Remove duplicate column names
    # --------------------------------------------------------

    if df.columns.duplicated().any():

        duplicated = (
            df.columns[df.columns.duplicated()].tolist()
        )

        print(
            "Warning: duplicate columns detected. "
            f"Removing: {duplicated}"
        )

        df = df.loc[
            :,
            ~df.columns.duplicated()
        ].copy()

    # --------------------------------------------------------
    # 2. Timestamp
    # --------------------------------------------------------

    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        df = (
            df.sort_values("timestamp")
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # 3. Validate base features
    # --------------------------------------------------------

    missing_base_features = [
        feature
        for feature in BASE_FEATURES
        if feature not in df.columns
    ]

    if missing_base_features:
        raise ValueError(
            "Missing required base telemetry features: "
            f"{missing_base_features[:20]}"
        )

    # --------------------------------------------------------
    # 4. Convert base features to numeric
    # --------------------------------------------------------

    for feature in BASE_FEATURES:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

    # --------------------------------------------------------
    # 5. Replace invalid values
    # --------------------------------------------------------

    df[BASE_FEATURES] = (
        df[BASE_FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    # --------------------------------------------------------
    # 6. Fill historical telemetry values
    # --------------------------------------------------------

    df[BASE_FEATURES] = (
        df[BASE_FEATURES]
        .ffill()
        .fillna(0)
    )

    # --------------------------------------------------------
    # 7. Define inference group
    # --------------------------------------------------------

    if "scenario" in df.columns:

        group_key = df["scenario"].fillna(
            "__UNKNOWN_SCENARIO__"
        )

    else:

        group_key = pd.Series(
            0,
            index=df.index,
            name="_inference_group",
        )

    # --------------------------------------------------------
    # 8. Delta features
    # --------------------------------------------------------

    for feature in BASE_FEATURES:

        feature_name = f"{feature}__delta1"

        if feature_name not in TEMPORAL_FEATURES:
            continue

        delta = (
            df[feature]
            .groupby(
                group_key,
                sort=False,
            )
            .diff(1)
        )

        df[feature_name] = (
            delta
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0)
        )

    # --------------------------------------------------------
    # 9. Rolling features
    # --------------------------------------------------------

    for feature in BASE_FEATURES:

        shifted = (
            df[feature]
            .groupby(
                group_key,
                sort=False,
            )
            .shift(1)
        )

        # Rolling mean 5
        feature_name = f"{feature}__rollmean5"

        if feature_name in TEMPORAL_FEATURES:

            df[feature_name] = (
                shifted
                .groupby(
                    group_key,
                    sort=False,
                )
                .transform(
                    lambda x:
                    x.rolling(
                        window=5,
                        min_periods=1,
                    ).mean()
                )
                .fillna(0)
            )

        # Rolling std 5
        feature_name = f"{feature}__rollstd5"

        if feature_name in TEMPORAL_FEATURES:

            df[feature_name] = (
                shifted
                .groupby(
                    group_key,
                    sort=False,
                )
                .transform(
                    lambda x:
                    x.rolling(
                        window=5,
                        min_periods=1,
                    ).std()
                )
                .fillna(0)
            )

        # Rolling mean 10
        feature_name = f"{feature}__rollmean10"

        if feature_name in TEMPORAL_FEATURES:

            df[feature_name] = (
                shifted
                .groupby(
                    group_key,
                    sort=False,
                )
                .transform(
                    lambda x:
                    x.rolling(
                        window=10,
                        min_periods=1,
                    ).mean()
                )
                .fillna(0)
            )

        # Rolling std 10
        feature_name = f"{feature}__rollstd10"

        if feature_name in TEMPORAL_FEATURES:

            df[feature_name] = (
                shifted
                .groupby(
                    group_key,
                    sort=False,
                )
                .transform(
                    lambda x:
                    x.rolling(
                        window=10,
                        min_periods=1,
                    ).std()
                )
                .fillna(0)
            )

    # --------------------------------------------------------
    # 10. Percentage-change features
    # --------------------------------------------------------

    pct_candidates = [
        feature
        for feature in BASE_FEATURES
        if any(
            keyword in feature.lower()
            for keyword in [
                "power",
                "kw",
                "kvar",
                "freq",
                "soc",
                "voltage",
                "irradiance",
            ]
        )
    ]

    pct_candidates = pct_candidates[:40]

    for feature in pct_candidates:

        feature_name = f"{feature}__pctchange"

        if feature_name not in TEMPORAL_FEATURES:
            continue

        pct_change = (
            df[feature]
            .groupby(
                group_key,
                sort=False,
            )
            .pct_change()
        )

        df[feature_name] = (
            pct_change
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0)
        )

    # --------------------------------------------------------
    # 11. Replace remaining invalid values
    # --------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # --------------------------------------------------------
    # 12. Verify all required model features exist
    # --------------------------------------------------------

    missing_temporal_features = [
        feature
        for feature in ALL_FEATURES
        if feature not in df.columns
    ]

    if missing_temporal_features:

        raise ValueError(
            "Temporal feature generation failed.\n"
            f"Missing {len(missing_temporal_features)} "
            "model features.\n"
            f"Examples: "
            f"{missing_temporal_features[:20]}"
        )

    # --------------------------------------------------------
    # 13. Select EXACT model feature order
    # --------------------------------------------------------

    temporal_df = df[ALL_FEATURES].copy()

    # --------------------------------------------------------
    # 14. Ensure numeric data
    # --------------------------------------------------------

    temporal_df = temporal_df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    temporal_df = temporal_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    temporal_df = temporal_df.fillna(0)

    # --------------------------------------------------------
    # 15. Final feature-count validation
    # --------------------------------------------------------

    if temporal_df.shape[1] != len(ALL_FEATURES):

        raise ValueError(
            "Final feature count mismatch: "
            f"generated={temporal_df.shape[1]}, "
            f"expected={len(ALL_FEATURES)}"
        )

    # --------------------------------------------------------
    # 16. Final feature-order validation
    # --------------------------------------------------------

    if list(temporal_df.columns) != list(ALL_FEATURES):

        raise ValueError(
            "Final temporal feature order does not "
            "match the trained XGBoost model."
        )

    print(
        "Temporal features generated successfully: "
        f"{temporal_df.shape[1]}"
    )

    return temporal_df


# ============================================================
# ANOMALY SCORE
# ============================================================

def calculate_anomaly_score(
    X: pd.DataFrame,
) -> float:

    """
    Calculate anomaly score using the SAME calibration
    learned from validation data.
    """

    _load_artifacts()

    X_scaled = anomaly_scaler.transform(
        X
    )


    raw_score = (
        -isolation_model
        .decision_function(X_scaled)
    )


    normalized = (
        (
            raw_score
            - CALIBRATION_MIN
        )
        /
        (
            CALIBRATION_MAX
            - CALIBRATION_MIN
            + 1e-12
        )
    )


    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )


    return float(
        normalized[-1]
    )


# ============================================================
# SHAP EXPLANATION
# ============================================================

def create_shap_explanation(
    X: pd.DataFrame,
    top_n: int = 10,
):

    """
    Explain the final prediction using SHAP.
    """

    _load_artifacts()

    sample = X.iloc[[-1]]


    shap_values = (
        shap_explainer
        .shap_values(sample)
    )


    if isinstance(
        shap_values,
        list,
    ):

        shap_values = shap_values[0]


    shap_values = shap_values[0]


    explanation = pd.DataFrame(
        {
            "feature": ALL_FEATURES,

            "feature_value":
                sample.iloc[0].values,

            "shap_value":
                shap_values,
        }
    )


    explanation[
        "abs_shap"
    ] = explanation[
        "shap_value"
    ].abs()


    explanation = explanation.sort_values(
        "abs_shap",
        ascending=False,
    )


    top = explanation.head(
        top_n
    )


    results = []


    for _, row in top.iterrows():

        results.append(
            {
                "feature": row["feature"],

                "value": float(
                    row["feature_value"]
                ),

                "shap_value": float(
                    row["shap_value"]
                ),

                "direction": (
                    "attack"
                    if row["shap_value"] > 0
                    else "benign"
                ),
            }
        )


    return results


# ============================================================
# UNIFIED PREDICTION
# ============================================================

def predict_security_event(
    telemetry: pd.DataFrame,
    criticality: float = 0.5,
    endpoint: str | None = None,
):
    """
    Complete AI-CyberShield prediction.

    Parameters
    ----------
    telemetry:
        Recent telemetry history.

    criticality:
        System/asset criticality from 0 to 1.

    endpoint:
        Logical endpoint identifier.

    Returns
    -------
    dict
    """

    _load_artifacts()

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if telemetry is None:

        raise ValueError(
            "Telemetry cannot be None."
        )


    if not isinstance(
        telemetry,
        pd.DataFrame,
    ):

        raise TypeError(
            "Telemetry must be a pandas DataFrame."
        )


    if telemetry.empty:

        raise ValueError(
            "Telemetry dataframe is empty."
        )


    criticality = float(
        np.clip(
            criticality,
            0.0,
            1.0,
        )
    )


    # --------------------------------------------------------
    # Temporal features
    # --------------------------------------------------------

    temporal_df = create_temporal_features(
        telemetry
    )


    # --------------------------------------------------------
    # Verify features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in ALL_FEATURES
        if feature not in temporal_df.columns
    ]


    if missing_features:

        raise ValueError(
            "Missing model features: "
            f"{missing_features[:20]}"
        )


    # --------------------------------------------------------
    # XGBoost input
    # --------------------------------------------------------

    X = temporal_df[
        ALL_FEATURES
    ].copy()


    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )


    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )


    X = X.fillna(0)


    # --------------------------------------------------------
    # XGBoost prediction
    # --------------------------------------------------------

    final_row = X.iloc[[-1]]


    attack_probability = float(
        xgb_model
        .predict_proba(final_row)[
            0,
            1
        ]
    )


    attack_prediction = int(
        attack_probability >= 0.30
    )


    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------

    anomaly_score = (
        calculate_anomaly_score(
            final_row
        )
    )


    anomaly_prediction = int(
        anomaly_score >= ANOMALY_THRESHOLD
    )


    # --------------------------------------------------------
    # Risk Engine
    # --------------------------------------------------------

    assessment = risk_engine.assess(

        attack_probability=
            attack_probability,

        anomaly_score=
            anomaly_score,

        criticality=
            criticality,
    )


    # --------------------------------------------------------
    # Safety Engine
    # --------------------------------------------------------

    safety_decision = (
        safety_engine.evaluate(
            assessment,
            endpoint=endpoint,
        )
    )


    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    explanation = (
        create_shap_explanation(
            final_row,
            top_n=10,
        )
    )


    # --------------------------------------------------------
    # Convert dataclass/object values
    # --------------------------------------------------------

    risk_score = float(
        assessment.risk_score
    )


    risk_level = str(
        assessment.risk_level
    )


    recommended_action = str(
        assessment.recommended_action
    )


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    result = {

        "attack_probability":
            round(
                attack_probability,
                6,
            ),

        "attack_prediction":
            attack_prediction,

        "anomaly_score":
            round(
                anomaly_score,
                6,
            ),

        "anomaly_prediction":
            anomaly_prediction,

        "anomaly_threshold":
            round(
                ANOMALY_THRESHOLD,
                6,
            ),

        "criticality":
            round(
                criticality,
                6,
            ),

        "risk_score":
            round(
                risk_score,
                4,
            ),

        "risk_level":
            risk_level,

        "recommended_action":
            recommended_action,

        "operator_confirmation_required":
            bool(
                safety_decision
                .operator_confirmation_required
            ),

        "logical_isolation":
            bool(
                safety_decision
                .logical_isolation
            ),

        "alert":
            bool(
                safety_decision
                .alert
            ),

        "allow_automated_control":
            bool(
                safety_decision
                .allow_control
            ),

        "risk_reason":
            getattr(
                assessment,
                "reason",
                "",
            ),

        "safety_reason":
            getattr(
                safety_decision,
                "reason",
                "",
            ),

        "endpoint":
            endpoint,

        "shap_explanation":
            explanation,

        "physical_breaker_control":
            False,
    }


    return result
