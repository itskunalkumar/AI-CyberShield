import os
import json
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

FEATURE_FILE = "data/processed/robust_features.json"

MODEL_DIR = "models"
MODEL_FILE = os.path.join(
    MODEL_DIR,
    "xgboost_robust_detector.json"
)


TARGET = "attack_active"


def load_features():

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        config = json.load(file)

    return config["features"]


def prepare_data(df, features):

    missing_features = [
        col
        for col in features
        if col not in df.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing features: {missing_features}"
        )

    X = df[features].copy()

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if X.isna().any().any():

        bad_columns = (
            X.columns[
                X.isna().any()
            ].tolist()
        )

        raise ValueError(
            f"Missing values in: {bad_columns}"
        )

    y = df[TARGET].astype(int)

    return X, y


def evaluate(
    name,
    y_true,
    probabilities
):

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
    print("AI-CyberShield - Robust XGBoost Experiment")
    print("=" * 70)

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Load robust feature list
    # ---------------------------------------------------------

    features = load_features()

    print(
        f"\nRobust feature count: "
        f"{len(features)}"
    )

    # ---------------------------------------------------------
    # Load datasets
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

    # ---------------------------------------------------------
    # Prepare data
    # ---------------------------------------------------------

    X_train, y_train = prepare_data(
        train_df,
        features
    )

    X_validation, y_validation = prepare_data(
        validation_df,
        features
    )

    X_test, y_test = prepare_data(
        test_df,
        features
    )

    # ---------------------------------------------------------
    # Class weighting
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

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # ---------------------------------------------------------
    # Same XGBoost configuration as baseline
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

    print("\nTraining robust XGBoost...")

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
        f"Best iteration: "
        f"{model.best_iteration}"
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    validation_probability = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    validation_metrics = evaluate(
        "ROBUST VALIDATION RESULTS",
        y_validation,
        validation_probability
    )

    # ---------------------------------------------------------
    # Test
    # ---------------------------------------------------------

    test_probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    test_metrics = evaluate(
        "ROBUST FINAL TEST RESULTS",
        y_test,
        test_probability
    )

    # ---------------------------------------------------------
    # Per scenario
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ROBUST TEST PERFORMANCE BY SCENARIO")
    print("=" * 70)

    results = test_df[
        [
            "scenario",
            TARGET
        ]
    ].copy()

    results["probability"] = (
        test_probability
    )

    results["prediction"] = (
        test_probability >= 0.5
    ).astype(int)

    for scenario, group in results.groupby(
        "scenario"
    ):

        precision = precision_score(
            group[TARGET],
            group["prediction"],
            zero_division=0
        )

        recall = recall_score(
            group[TARGET],
            group["prediction"],
            zero_division=0
        )

        f1 = f1_score(
            group[TARGET],
            group["prediction"],
            zero_division=0
        )

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
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall:    {recall:.4f}"
        )

        print(
            f"F1:        {f1:.4f}"
        )

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("TOP 20 ROBUST FEATURES")
    print("=" * 70)

    print(
        importance.head(20).to_string(
            index=False
        )
    )

    importance.to_csv(
        os.path.join(
            MODEL_DIR,
            "robust_feature_importance.csv"
        ),
        index=False
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
    # Save metrics
    # ---------------------------------------------------------

    with open(
        os.path.join(
            MODEL_DIR,
            "robust_metrics.json"
        ),
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "feature_count": len(features),
                "best_iteration": int(
                    model.best_iteration
                ),
                "validation": validation_metrics,
                "test": test_metrics,
            },
            file,
            indent=2
        )

    print(
        "\nMetrics saved:"
    )

    print(
        "models/robust_metrics.json"
    )

    print("\n" + "=" * 70)
    print("ROBUST EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()