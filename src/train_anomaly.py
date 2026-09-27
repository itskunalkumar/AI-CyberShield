"""
AI-CyberShield
Unsupervised Anomaly Detection

Isolation Forest is trained ONLY on benign training data.

TRAIN SCENARIOS:
    S0
    S1_alt
    S2

VALIDATION:
    S1

TEST:
    S3
    S4
    S5
    S6
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path(
    "data/processed/temporal_dataset.csv"
)

FEATURES_PATH = Path(
    "data/processed/temporal_features.json"
)

MODEL_PATH = Path(
    "models/isolation_forest.joblib"
)

SCALER_PATH = Path(
    "models/anomaly_scaler.joblib"
)

CALIBRATION_PATH = Path(
    "models/anomaly_calibration.json"
)

METRICS_PATH = Path(
    "models/anomaly_metrics.json"
)

TARGET = "attack_active"
SCENARIO = "scenario"


TRAIN_SCENARIOS = [
    "S0_benign_baseline",
    "S1_industroyer_pv_alt",
    "S2_industroyer_bss",
]


VALIDATION_SCENARIOS = [
    "S1_industroyer_pv"
]


TEST_SCENARIOS = [
    "S3_arp_spoof_bss_meter_half_values",
    "S4_arp_spoof_loads_pv_two_phase",
    "S5_arp_spoof_loads_pv_bss_two_phase",
    "S6_mqtt_supply_chain_compromise",
]


CONTAMINATION = 0.01
RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("AI-CyberShield - Isolation Forest Anomaly Detector")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

if not FEATURES_PATH.exists():
    raise FileNotFoundError(
        f"Feature configuration not found: {FEATURES_PATH}"
    )


df = pd.read_csv(DATA_PATH)


with open(
    FEATURES_PATH,
    "r",
    encoding="utf-8"
) as f:

    feature_data = json.load(f)


# Use all 845 features
features = feature_data["all_features"]


# Keep only features actually present
features = [
    feature
    for feature in features
    if feature in df.columns
]


print(f"Dataset shape: {df.shape}")
print(f"Features used: {len(features)}")


if len(features) == 0:
    raise ValueError(
        "No model features were found in the dataset."
    )


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    TARGET,
    SCENARIO,
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# SPLIT BY SCENARIO
# ============================================================

print("\nCreating scenario-based split...")


train_df = df[
    df[SCENARIO].isin(TRAIN_SCENARIOS)
].copy()


val_df = df[
    df[SCENARIO].isin(VALIDATION_SCENARIOS)
].copy()


test_df = df[
    df[SCENARIO].isin(TEST_SCENARIOS)
].copy()


print(
    f"Training rows:   {len(train_df)}"
)

print(
    f"Validation rows: {len(val_df)}"
)

print(
    f"Test rows:       {len(test_df)}"
)


# ============================================================
# TRAIN ONLY ON BENIGN DATA
# ============================================================

train_benign = train_df[
    train_df[TARGET] == 0
].copy()


print(
    f"\nBenign training rows: "
    f"{len(train_benign)}"
)


if len(train_benign) == 0:

    raise ValueError(
        "No benign training rows found."
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

X_train = train_benign[
    features
].copy()


X_val = val_df[
    features
].copy()


X_test = test_df[
    features
].copy()


y_val = val_df[
    TARGET
].astype(int)


y_test = test_df[
    TARGET
].astype(int)


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\nChecking missing values...")


train_missing = int(
    X_train.isna().sum().sum()
)

val_missing = int(
    X_val.isna().sum().sum()
)

test_missing = int(
    X_test.isna().sum().sum()
)


print(
    f"Training missing values:   {train_missing}"
)

print(
    f"Validation missing values: {val_missing}"
)

print(
    f"Test missing values:       {test_missing}"
)


if (
    train_missing > 0
    or val_missing > 0
    or test_missing > 0
):

    raise ValueError(
        "Missing values detected. "
        "Clean the temporal dataset before training."
    )


# ============================================================
# SCALER
# ============================================================

print("\nFitting StandardScaler...")


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_val_scaled = scaler.transform(
    X_val
)


X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# ISOLATION FOREST
# ============================================================

print("\nTraining Isolation Forest...")


model = IsolationForest(

    n_estimators=300,

    contamination=CONTAMINATION,

    random_state=RANDOM_STATE,

    n_jobs=-1,
)


model.fit(
    X_train_scaled
)


print("Isolation Forest training complete.")


# ============================================================
# RAW ANOMALY SCORE
# ============================================================

def raw_anomaly_score(X):
    """
    Isolation Forest decision_function:

        larger value  = more normal
        smaller value = more anomalous

    Therefore we multiply by -1 so:

        larger score = more anomalous
    """

    score = -model.decision_function(
        X
    )

    return score


# ============================================================
# VALIDATION CALIBRATION
# ============================================================

print("\nCreating fixed anomaly-score calibration...")


val_raw_anomaly = raw_anomaly_score(
    X_val_scaled
)


calibration_min = float(
    val_raw_anomaly.min()
)


calibration_max = float(
    val_raw_anomaly.max()
)


print(
    f"Calibration minimum: "
    f"{calibration_min:.8f}"
)


print(
    f"Calibration maximum: "
    f"{calibration_max:.8f}"
)


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def normalize_anomaly_score(
    raw_score,
    min_value,
    max_value,
):
    """
    Convert raw anomaly score to 0-1.

    IMPORTANT:
    min_value and max_value come ONLY from
    validation calibration.

    They are NOT recalculated for test/live data.
    """

    normalized = (
        (raw_score - min_value)
        /
        (
            max_value
            - min_value
            + 1e-12
        )
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    return normalized


# ============================================================
# VALIDATION ANOMALY SCORE
# ============================================================

val_anomaly = normalize_anomaly_score(

    val_raw_anomaly,

    calibration_min,

    calibration_max,
)


# ============================================================
# TEST ANOMALY SCORE
# ============================================================

test_raw_anomaly = raw_anomaly_score(
    X_test_scaled
)


test_anomaly = normalize_anomaly_score(

    test_raw_anomaly,

    calibration_min,

    calibration_max,
)


# ============================================================
# THRESHOLD SELECTION
# ============================================================

print("\nSearching for best validation threshold...")


validation_thresholds = np.linspace(
    0.10,
    0.90,
    17,
)


best_threshold = 0.50
best_f1 = -1.0


for threshold in validation_thresholds:

    predictions = (
        val_anomaly >= threshold
    ).astype(int)


    score = f1_score(

        y_val,

        predictions,

        zero_division=0,
    )


    if score > best_f1:

        best_f1 = score

        best_threshold = float(
            threshold
        )


print(
    f"\nSelected anomaly threshold: "
    f"{best_threshold:.2f}"
)


print(
    f"Validation F1 at threshold: "
    f"{best_f1:.4f}"
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    name,
    y,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)


    metrics = {

        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            y,
            predictions,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),

        "pr_auc": average_precision_score(
            y,
            probabilities,
        ),
    }


    print("\n" + "=" * 70)

    print(name)

    print("=" * 70)


    for key, value in metrics.items():

        print(
            f"{key.upper():<12}: "
            f"{value:.4f}"
        )


    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y,
            predictions,
        )
    )


    return metrics


# ============================================================
# VALIDATION EVALUATION
# ============================================================

val_metrics = evaluate(

    "ANOMALY VALIDATION",

    y_val,

    val_anomaly,

    best_threshold,
)


# ============================================================
# TEST EVALUATION
# ============================================================

test_metrics = evaluate(

    "ANOMALY TEST",

    y_test,

    test_anomaly,

    best_threshold,
)


# ============================================================
# PER-SCENARIO EVALUATION
# ============================================================

print("\n" + "=" * 70)

print(
    "ANOMALY DETECTION BY SCENARIO"
)

print("=" * 70)


test_output = test_df.copy()


test_output[
    "anomaly_probability"
] = test_anomaly


test_output[
    "anomaly_prediction"
] = (
    test_anomaly >= best_threshold
).astype(int)


scenario_results = {}


for scenario in TEST_SCENARIOS:

    subset = test_output[
        test_output[SCENARIO] == scenario
    ]


    if len(subset) == 0:

        print(
            f"\n{scenario}: no rows found"
        )

        continue


    y_true = subset[
        TARGET
    ].astype(int)


    probabilities = subset[
        "anomaly_probability"
    ]


    predictions = subset[
        "anomaly_prediction"
    ]


    # ROC-AUC requires both classes
    if y_true.nunique() >= 2:

        scenario_roc_auc = (
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

    else:

        scenario_roc_auc = None


    # PR-AUC can technically be calculated
    # for a single positive class
    scenario_pr_auc = (
        average_precision_score(
            y_true,
            probabilities,
        )
    )


    result = {

        "rows": int(
            len(subset)
        ),

        "attack_rate": float(
            y_true.mean()
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "roc_auc": (
            float(scenario_roc_auc)
            if scenario_roc_auc is not None
            else None
        ),

        "pr_auc": float(
            scenario_pr_auc
        ),
    }


    scenario_results[
        scenario
    ] = result


    print(
        f"\n{scenario}"
    )


    for key, value in result.items():

        if isinstance(
            value,
            float,
        ):

            print(
                f"{key:<18}: "
                f"{value:.4f}"
            )

        else:

            print(
                f"{key:<18}: "
                f"{value}"
            )


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SAVE ISOLATION FOREST
# ============================================================

print("\nSaving Isolation Forest...")


joblib.dump(
    model,
    MODEL_PATH,
)


# ============================================================
# SAVE SCALER
# ============================================================

print("Saving scaler...")


joblib.dump(
    scaler,
    SCALER_PATH,
)


# ============================================================
# SAVE CALIBRATION
# ============================================================

calibration = {

    "method": "validation_min_max",

    "min": calibration_min,

    "max": calibration_max,

    "threshold": best_threshold,

    "description": (
        "Anomaly scores are normalized using "
        "fixed validation-set calibration values. "
        "These values must be reused during "
        "test and live inference."
    ),
}


with open(
    CALIBRATION_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        calibration,
        f,
        indent=2,
    )


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "feature_count": len(
        features
    ),

    "training_rows": len(
        train_df
    ),

    "training_benign_rows": len(
        train_benign
    ),

    "validation_rows": len(
        val_df
    ),

    "test_rows": len(
        test_df
    ),

    "threshold": best_threshold,

    "calibration": {

        "method": "validation_min_max",

        "min": calibration_min,

        "max": calibration_max,
    },

    "validation": val_metrics,

    "test": test_metrics,

    "scenarios": scenario_results,
}


with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics,
        f,
        indent=2,
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)

print(
    "ANOMALY DETECTOR COMPLETE"
)

print("=" * 70)


print(
    f"Model:       {MODEL_PATH}"
)

print(
    f"Scaler:      {SCALER_PATH}"
)

print(
    f"Calibration: {CALIBRATION_PATH}"
)

print(
    f"Metrics:     {METRICS_PATH}"
)

print("=" * 70)