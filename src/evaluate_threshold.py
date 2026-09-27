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
)


TEST_FILE = "data/processed/splits/test.csv"
FEATURE_FILE = "data/processed/robust_features.json"
MODEL_FILE = "models/xgboost_robust_detector.json"

TARGET = "attack_active"

# Frozen using validation results
THRESHOLD = 0.30


def main():

    print("=" * 70)
    print("AI-CyberShield - Frozen Threshold Test Evaluation")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load features
    # ---------------------------------------------------------

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        config = json.load(file)

    features = config["features"]

    # ---------------------------------------------------------
    # Load test data
    # ---------------------------------------------------------

    df = pd.read_csv(
        TEST_FILE,
        low_memory=False
    )

    X = df[features].copy()

    y = df[TARGET].astype(int)

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    model = XGBClassifier()

    model.load_model(
        MODEL_FILE
    )

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    # ---------------------------------------------------------
    # Overall metrics
    # ---------------------------------------------------------

    print(
        f"\nFrozen threshold: "
        f"{THRESHOLD:.2f}"
    )

    print("\n" + "=" * 70)
    print("OVERALL TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy : "
        f"{accuracy_score(y, predictions):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y, predictions, zero_division=0):.4f}"
    )

    print(
        f"Recall   : "
        f"{recall_score(y, predictions, zero_division=0):.4f}"
    )

    print(
        f"F1       : "
        f"{f1_score(y, predictions, zero_division=0):.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{roc_auc_score(y, probabilities):.4f}"
    )

    print(
        f"PR-AUC   : "
        f"{average_precision_score(y, probabilities):.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y,
            predictions
        )
    )

    # ---------------------------------------------------------
    # Probability distribution
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREDICTION PROBABILITY DISTRIBUTION")
    print("=" * 70)

    print(
        pd.Series(probabilities)
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
    )

    # ---------------------------------------------------------
    # Per-scenario evaluation
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST RESULTS BY SCENARIO")
    print("=" * 70)

    results = df[
        [
            "scenario",
            TARGET
        ]
    ].copy()

    results["probability"] = probabilities
    results["prediction"] = predictions

    for scenario, group in results.groupby(
        "scenario"
    ):

        scenario_y = group[TARGET]
        scenario_pred = group["prediction"]

        precision = precision_score(
            scenario_y,
            scenario_pred,
            zero_division=0
        )

        recall = recall_score(
            scenario_y,
            scenario_pred,
            zero_division=0
        )

        f1 = f1_score(
            scenario_y,
            scenario_pred,
            zero_division=0
        )

        print(
            f"\n{scenario}"
        )

        print(
            f"Rows       : {len(group)}"
        )

        print(
            f"Attack rate: "
            f"{scenario_y.mean():.4f}"
        )

        print(
            f"Precision  : {precision:.4f}"
        )

        print(
            f"Recall     : {recall:.4f}"
        )

        print(
            f"F1         : {f1:.4f}"
        )

        print(
            f"Predicted attacks: "
            f"{scenario_pred.sum()}"
        )

        print(
            f"Actual attacks   : "
            f"{scenario_y.sum()}"
        )

    print("\n" + "=" * 70)
    print("Evaluation complete")
    print("=" * 70)


if __name__ == "__main__":
    main()