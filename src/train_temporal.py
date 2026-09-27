"""
AI-CyberShield
Temporal XGBoost Experiment

TRAIN:
    S0 + S1_alt + S2

VALIDATION:
    S1

TEST:
    S3 + S4 + S5 + S6

The scenario split remains unchanged.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/processed/temporal_dataset.csv"
)

FEATURES_PATH = Path(
    "data/processed/temporal_features.json"
)

MODEL_PATH = Path(
    "models/xgboost_temporal_detector.json"
)

METRICS_PATH = Path(
    "models/temporal_metrics.json"
)

IMPORTANCE_PATH = Path(
    "models/temporal_feature_importance.csv"
)

TARGET = "attack_active"

SCENARIO_COLUMN = "scenario"

THRESHOLD = 0.30

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AI-CyberShield - Temporal XGBoost Training")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

with open(FEATURES_PATH, "r") as f:
    feature_data = json.load(f)

features = feature_data["all_features"]

features = [
    feature
    for feature in features
    if feature in df.columns
]

print(f"\nDataset shape: {df.shape}")
print(f"Features: {len(features)}")


# ============================================================
# SCENARIO SPLIT
# ============================================================

TRAIN_SCENARIOS = [
    "S0_benign_baseline",
    "S1_industroyer_pv_alt",
    "S2_industroyer_bss",
]

VALIDATION_SCENARIOS = [
    "S1_industroyer_pv",
]

TEST_SCENARIOS = [
    "S3_arp_spoof_bss_meter_half_values",
    "S4_arp_spoof_loads_pv_two_phase",
    "S5_arp_spoof_loads_pv_bss_two_phase",
    "S6_mqtt_supply_chain_compromise",
]


train_df = df[
    df[SCENARIO_COLUMN].isin(TRAIN_SCENARIOS)
].copy()

val_df = df[
    df[SCENARIO_COLUMN].isin(VALIDATION_SCENARIOS)
].copy()

test_df = df[
    df[SCENARIO_COLUMN].isin(TEST_SCENARIOS)
].copy()


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

for data in [train_df, val_df, test_df]:

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce"
    )

    data.sort_values(
        [SCENARIO_COLUMN, "timestamp"],
        inplace=True
    )


# ============================================================
# PREPARE X / y
# ============================================================

X_train = train_df[features]
y_train = train_df[TARGET].astype(int)

X_val = val_df[features]
y_val = val_df[TARGET].astype(int)

X_test = test_df[features]
y_test = test_df[TARGET].astype(int)


print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    f"Train      : {X_train.shape}"
)

print(
    f"Validation : {X_val.shape}"
)

print(
    f"Test       : {X_test.shape}"
)

print("\nTrain target:")
print(y_train.value_counts())

print("\nValidation target:")
print(y_val.value_counts())

print("\nTest target:")
print(y_test.value_counts())


# ============================================================
# CLASS WEIGHT
# ============================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print(
    f"\nscale_pos_weight: "
    f"{scale_pos_weight:.4f}"
)


# ============================================================
# MODEL
# ============================================================

model = xgb.XGBClassifier(

    n_estimators=500,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    tree_method="hist",

    random_state=RANDOM_STATE,

    n_jobs=-1,

    early_stopping_rounds=40,
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining temporal XGBoost...")

model.fit(

    X_train,
    y_train,

    eval_set=[
        (X_train, y_train),
        (X_val, y_val),
    ],

    verbose=False,
)


print(
    f"Best iteration: "
    f"{model.best_iteration}"
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    name,
    X,
    y,
    threshold=0.30
):

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {

        "accuracy": accuracy_score(
            y,
            predictions
        ),

        "precision": precision_score(
            y,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y,
            probabilities
        ),

        "pr_auc": average_precision_score(
            y,
            probabilities
        ),
    }

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    for key, value in metrics.items():

        print(
            f"{key.upper():<10}: "
            f"{value:.4f}"
        )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y,
            predictions
        )
    )

    return metrics, probabilities, predictions


# ============================================================
# VALIDATION
# ============================================================

val_metrics, _, _ = evaluate(
    "TEMPORAL VALIDATION RESULTS",
    X_val,
    y_val,
    THRESHOLD
)


# ============================================================
# TEST
# ============================================================

test_metrics, test_probabilities, test_predictions = evaluate(
    "TEMPORAL TEST RESULTS",
    X_test,
    y_test,
    THRESHOLD
)


# ============================================================
# PER-SCENARIO TEST RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL TEST RESULTS BY SCENARIO")
print("=" * 70)

test_output = test_df.copy()

test_output["probability"] = test_probabilities

test_output["prediction"] = test_predictions

scenario_results = {}

for scenario in TEST_SCENARIOS:

    mask = (
        test_output[SCENARIO_COLUMN]
        == scenario
    )

    subset = test_output[mask]

    y_true = subset[TARGET].astype(int)

    y_pred = subset["prediction"].astype(int)

    y_prob = subset["probability"]

    metrics = {

        "rows": len(subset),

        "attack_rate": y_true.mean(),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_true,
            y_prob
        ),

        "pr_auc": average_precision_score(
            y_true,
            y_prob
        ),

        "predicted_attacks": int(
            y_pred.sum()
        ),

        "actual_attacks": int(
            y_true.sum()
        ),
    }

    scenario_results[scenario] = metrics

    print(f"\n{scenario}")

    for key, value in metrics.items():

        if isinstance(value, float):

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
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature": features,

    "importance": model.feature_importances_,

})

importance = importance.sort_values(
    "importance",
    ascending=False
)

importance.to_csv(
    IMPORTANCE_PATH,
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

model.save_model(
    MODEL_PATH
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_output = {

    "threshold": THRESHOLD,

    "feature_count": len(features),

    "best_iteration": int(
        model.best_iteration
    ),

    "validation": val_metrics,

    "test": test_metrics,

    "scenarios": scenario_results,
}


with open(METRICS_PATH, "w") as f:

    json.dump(
        metrics_output,
        f,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL MODEL TRAINING COMPLETE")
print("=" * 70)

print(
    f"Model saved: {MODEL_PATH}"
)

print(
    f"Metrics saved: {METRICS_PATH}"
)

print(
    f"Importance saved: {IMPORTANCE_PATH}"
)