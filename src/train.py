import os
import json
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


TRAIN_FILE = "data/processed/splits/train.csv"
VALIDATION_FILE = "data/processed/splits/validation.csv"
TEST_FILE = "data/processed/splits/test.csv"

MODEL_DIR = "models"
MODEL_FILE = os.path.join(
    MODEL_DIR,
    "xgboost_attack_detector.json"
)

FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "feature_names.json"
)


TARGET = "attack_active"

EXCLUDED_COLUMNS = [
    TARGET,
    "timestamp",
    "attack_phase",
    "attack_phase_all",
    "scenario",
    "Attacker.event",
    "Attacker.new_value",
    "Attacker.old_value",
    "MQTT-Price-Signal.event",
    "Profile.timestamp",
]


def prepare_features(df):

    X = df.drop(
        columns=[
            col
            for col in EXCLUDED_COLUMNS
            if col in df.columns
        ],
        errors="ignore"
    )

    # Ensure numeric matrix
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # The cleaning stage should already have handled this.
    # This is a safety check.
    if X.isna().any().any():

        missing_columns = (
            X.columns[
                X.isna().any()
            ].tolist()
        )

        raise ValueError(
            "Missing values found in features: "
            f"{missing_columns}"
        )

    return X


def calculate_metrics(y_true, probabilities):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities
        ),

        "pr_auc": average_precision_score(
            y_true,
            probabilities
        ),
    }

    return metrics, predictions


def print_metrics(
    name,
    y_true,
    probabilities
):

    metrics, predictions = calculate_metrics(
        y_true,
        probabilities
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    for metric, value in metrics.items():

        print(
            f"{metric.upper():12s}: "
            f"{value:.4f}"
        )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_true,
            predictions
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            predictions,
            digits=4,
            zero_division=0
        )
    )

    return metrics


def main():

    print("=" * 70)
    print("AI-CyberShield - XGBoost Attack Detector")
    print("=" * 70)

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_FILE,
        low_memory=False
    )

    validation_df = pd.read_csv(
        VALIDATION_FILE,
        low_memory=False
    )

    test_df = pd.read_csv(
        TEST_FILE,
        low_memory=False
    )

    print("\nLoaded datasets:")

    print(
        f"Train      : {train_df.shape}"
    )

    print(
        f"Validation : {validation_df.shape}"
    )

    print(
        f"Test       : {test_df.shape}"
    )

    # ---------------------------------------------------------
    # Prepare features
    # ---------------------------------------------------------

    X_train = prepare_features(
        train_df
    )

    X_validation = prepare_features(
        validation_df
    )

    X_test = prepare_features(
        test_df
    )

    y_train = train_df[TARGET].astype(int)
    y_validation = validation_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    # ---------------------------------------------------------
    # Feature consistency
    # ---------------------------------------------------------

    if list(X_train.columns) != list(
        X_validation.columns
    ):

        raise ValueError(
            "Train and validation features do not match."
        )

    if list(X_train.columns) != list(
        X_test.columns
    ):

        raise ValueError(
            "Train and test features do not match."
        )

    print(
        f"\nNumber of features: "
        f"{X_train.shape[1]}"
    )

    # ---------------------------------------------------------
    # Class imbalance
    # ---------------------------------------------------------

    negative = (
        y_train == 0
    ).sum()

    positive = (
        y_train == 1
    ).sum()

    scale_pos_weight = (
        negative / positive
    )

    print("\nTraining class distribution:")

    print(
        f"Benign : {negative}"
    )

    print(
        f"Attack : {positive}"
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # ---------------------------------------------------------
    # XGBoost
    # ---------------------------------------------------------

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,

        objective="binary:logistic",
        eval_metric="logloss",

        scale_pos_weight=scale_pos_weight,

        tree_method="hist",

        random_state=42,
        n_jobs=-1,

        early_stopping_rounds=40,
    )

    print("\nTraining XGBoost...")

    model.fit(
        X_train,
        y_train,

        eval_set=[
            (
                X_validation,
                y_validation
            )
        ],

        verbose=False
    )

    print(
        f"\nBest iteration: "
        f"{model.best_iteration}"
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    validation_metrics = print_metrics(
        "VALIDATION RESULTS",
        y_validation,
        validation_probabilities
    )

    # ---------------------------------------------------------
    # Final test
    # ---------------------------------------------------------

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    test_metrics = print_metrics(
        "FINAL TEST RESULTS",
        y_test,
        test_probabilities
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    model.save_model(
        MODEL_FILE
    )

    print(
        f"\nModel saved: "
        f"{MODEL_FILE}"
    )

    # ---------------------------------------------------------
    # Save feature names
    # ---------------------------------------------------------

    with open(
        FEATURE_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            list(X_train.columns),
            file,
            indent=2
        )

    print(
        f"Features saved: "
        f"{FEATURE_FILE}"
    )

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------

    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("TOP 30 FEATURES")
    print("=" * 70)

    print(
        importance.head(30).to_string(
            index=False
        )
    )

    importance.to_csv(
        os.path.join(
            MODEL_DIR,
            "feature_importance.csv"
        ),
        index=False
    )

    # ---------------------------------------------------------
    # Per-scenario test performance
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST PERFORMANCE BY SCENARIO")
    print("=" * 70)

    test_results = test_df[
        [
            "scenario",
            TARGET
        ]
    ].copy()

    test_results["probability"] = (
        test_probabilities
    )

    test_results["prediction"] = (
        test_probabilities >= 0.5
    ).astype(int)

    for scenario, group in test_results.groupby(
        "scenario"
    ):

        print(
            f"\n{scenario}"
        )

        print(
            f"Rows: {len(group)}"
        )

        print(
            f"Attack rate: "
            f"{group[TARGET].mean():.4f}"
        )

        print(
            f"Precision: "
            f"{precision_score(group[TARGET], group['prediction'], zero_division=0):.4f}"
        )

        print(
            f"Recall: "
            f"{recall_score(group[TARGET], group['prediction'], zero_division=0):.4f}"
        )

        print(
            f"F1: "
            f"{f1_score(group[TARGET], group['prediction'], zero_division=0):.4f}"
        )

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------

    metrics_output = {
        "validation": validation_metrics,
        "test": test_metrics,
        "best_iteration": int(
            model.best_iteration
        ),
        "feature_count": int(
            X_train.shape[1]
        )
    }

    with open(
        os.path.join(
            MODEL_DIR,
            "metrics.json"
        ),
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics_output,
            file,
            indent=2
        )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()