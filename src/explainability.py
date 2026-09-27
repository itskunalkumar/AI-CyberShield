import json
from pathlib import Path

import pandas as pd
import shap
import xgboost as xgb


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "temporal_dataset.csv"
FEATURES_PATH = BASE_DIR / "data" / "processed" / "temporal_features.json"
MODEL_PATH = BASE_DIR / "models" / "xgboost_temporal_detector.json"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "shap_example.csv"


def load_features():
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_config = json.load(f)

    feature_names = feature_config["all_features"]

    return feature_names


def main():

    print("=" * 70)
    print("AI-CyberShield - SHAP Explainability")
    print("=" * 70)

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------
    # Load correct 845 model features
    # --------------------------------------------------
    feature_names = load_features()

    print(f"Number of configured model features: {len(feature_names)}")

    # --------------------------------------------------
    # Load XGBoost model
    # --------------------------------------------------
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    print("XGBoost model loaded.")

    # --------------------------------------------------
    # Verify model feature count
    # --------------------------------------------------
    model_features = model.get_booster().feature_names

    print(f"Number of model features: {len(model_features)}")

    if len(model_features) != len(feature_names):
        raise ValueError(
            f"Feature count mismatch: "
            f"configuration={len(feature_names)}, "
            f"model={len(model_features)}"
        )

    # --------------------------------------------------
    # Verify feature order
    # --------------------------------------------------
    if model_features != feature_names:

        print("\nFeature order mismatch detected.")

        mismatches = []

        for i, (model_feature, config_feature) in enumerate(
            zip(model_features, feature_names)
        ):
            if model_feature != config_feature:
                mismatches.append(
                    (i, model_feature, config_feature)
                )

                if len(mismatches) >= 10:
                    break

        for item in mismatches:
            print(item)

        raise ValueError(
            "Model feature order does not match "
            "temporal_features.json"
        )

    print("Feature count and feature order verified.")

    # --------------------------------------------------
    # Prepare input
    # --------------------------------------------------
    X = df[feature_names]

    sample = X.iloc[[0]]

    print(f"Sample shape: {sample.shape}")

    # --------------------------------------------------
    # SHAP
    # --------------------------------------------------
    print("\nCreating SHAP TreeExplainer...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(sample)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = shap_values[0]

    # --------------------------------------------------
    # Create explanation
    # --------------------------------------------------
    explanation = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_value": sample.iloc[0].values,
            "shap_value": shap_values,
        }
    )

    explanation["abs_shap"] = (
        explanation["shap_value"].abs()
    )

    explanation = explanation.sort_values(
        "abs_shap",
        ascending=False
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    explanation.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------
    # Display
    # --------------------------------------------------
    print("\nTop 15 contributing features:")
    print("-" * 70)

    print(
        explanation[
            [
                "feature",
                "feature_value",
                "shap_value",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("SHAP explanation successfully generated.")
    print(f"Saved to: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()