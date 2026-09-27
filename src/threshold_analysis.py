import json
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


VALIDATION_FILE = "data/processed/splits/validation.csv"

FEATURE_FILE = "data/processed/robust_features.json"

MODEL_FILE = "models/xgboost_robust_detector.json"


TARGET = "attack_active"


def main():

    print("=" * 70)
    print("AI-CyberShield - Validation Threshold Analysis")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load feature configuration
    # ---------------------------------------------------------

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        config = json.load(file)

    features = config["features"]

    # ---------------------------------------------------------
    # Load validation data
    # ---------------------------------------------------------

    df = pd.read_csv(
        VALIDATION_FILE,
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

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    # ---------------------------------------------------------
    # Threshold analysis
    # ---------------------------------------------------------

    thresholds = np.arange(
        0.10,
        0.91,
        0.05
    )

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y,
            predictions,
            zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y,
            predictions
        ).ravel()

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "true_negatives": tn
        })

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 70)
    print("VALIDATION THRESHOLD TRADE-OFF")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ---------------------------------------------------------
    # Best F1
    # ---------------------------------------------------------

    best_f1 = results_df.loc[
        results_df["f1"].idxmax()
    ]

    print("\n" + "=" * 70)
    print("BEST VALIDATION F1")
    print("=" * 70)

    print(
        best_f1.to_string()
    )

    # ---------------------------------------------------------
    # Threshold achieving recall >= 90%
    # ---------------------------------------------------------

    high_recall = results_df[
        results_df["recall"] >= 0.90
    ]

    if not high_recall.empty:

        selected = high_recall.sort_values(
            [
                "precision",
                "recall"
            ],
            ascending=False
        ).iloc[0]

        print("\n" + "=" * 70)
        print("BEST PRECISION AMONG THRESHOLDS WITH RECALL >= 90%")
        print("=" * 70)

        print(
            selected.to_string()
        )

    else:

        print(
            "\nNo tested threshold achieved "
            "recall >= 90%."
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()