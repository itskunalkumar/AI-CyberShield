"""
AI-CyberShield
Leakage-Safe Temporal Feature Engineering

Creates temporal/behavioral features separately within each scenario.
The purpose is to capture changes and short-term behavior rather than
only absolute telemetry values.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path("data/processed/cleaned_dataset.csv")
OUTPUT_PATH = Path("data/processed/temporal_dataset.csv")
FEATURES_PATH = Path("data/processed/temporal_features.json")

GROUP_COLUMN = "scenario"
TIME_COLUMN = "timestamp"
TARGET_COLUMN = "attack_active"

# Existing robust features
ROBUST_FEATURES_PATH = Path(
    "data/processed/robust_features.json"
)

# Number of previous observations used for temporal calculations
ROLLING_WINDOWS = [5, 10]


# ============================================================
# LOAD ROBUST FEATURES
# ============================================================

def load_robust_features():

    with open(ROBUST_FEATURES_PATH, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    return data["features"]


# ============================================================
# CREATE TEMPORAL FEATURES
# ============================================================

def create_temporal_features(df, base_features):

    result = df.copy()

    # Make sure scenarios are chronologically ordered
    result[TIME_COLUMN] = pd.to_datetime(
        result[TIME_COLUMN],
        errors="coerce"
    )

    result = result.sort_values(
        [GROUP_COLUMN, TIME_COLUMN]
    ).reset_index(drop=True)

    temporal_features = []

    # --------------------------------------------------------
    # DELTA FEATURES
    # --------------------------------------------------------

    print("\nCreating delta features...")

    for feature in base_features:

        if feature not in result.columns:
            continue

        delta_name = f"{feature}__delta1"

        result[delta_name] = (
            result.groupby(GROUP_COLUMN)[feature]
            .diff(1)
        )

        temporal_features.append(delta_name)

    # --------------------------------------------------------
    # ROLLING FEATURES
    # --------------------------------------------------------

    print("Creating rolling features...")

    for window in ROLLING_WINDOWS:

        for feature in base_features:

            if feature not in result.columns:
                continue

            group = result.groupby(GROUP_COLUMN)[feature]

            mean_name = f"{feature}__rollmean{window}"
            std_name = f"{feature}__rollstd{window}"

            result[mean_name] = (
                group.transform(
                    lambda x: x.shift(1)
                    .rolling(window=window, min_periods=2)
                    .mean()
                )
            )

            result[std_name] = (
                group.transform(
                    lambda x: x.shift(1)
                    .rolling(window=window, min_periods=2)
                    .std()
                )
            )

            temporal_features.extend(
                [mean_name, std_name]
            )

    # --------------------------------------------------------
    # RATE-OF-CHANGE FEATURES
    # --------------------------------------------------------

    print("Creating percentage-change features...")

    # Use only selected physically meaningful features.
    ratio_features = [
        feature
        for feature in base_features
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

    # Limit this group to avoid an unnecessarily huge feature space.
    ratio_features = ratio_features[:40]

    for feature in ratio_features:

        pct_name = f"{feature}__pctchange"

        previous = (
            result.groupby(GROUP_COLUMN)[feature]
            .shift(1)
        )

        denominator = previous.abs().replace(0, np.nan)

        result[pct_name] = (
            (result[feature] - previous)
            / denominator
        )

        result[pct_name] = (
            result[pct_name]
            .replace([np.inf, -np.inf], np.nan)
            .clip(-10, 10)
        )

        temporal_features.append(pct_name)

    # --------------------------------------------------------
    # FILL TEMPORAL NaN VALUES
    # --------------------------------------------------------

    print("Cleaning temporal feature values...")

    for feature in temporal_features:

        result[feature] = pd.to_numeric(
            result[feature],
            errors="coerce"
        )

        result[feature] = (
            result.groupby(GROUP_COLUMN)[feature]
            .transform(
                lambda x: x.replace(
                    [np.inf, -np.inf],
                    np.nan
                )
            )
        )

        # Forward/backward filling only within each scenario.
        result[feature] = (
            result.groupby(GROUP_COLUMN)[feature]
            .transform(
                lambda x: x.ffill().bfill()
            )
        )

        result[feature] = result[feature].fillna(0)

    # --------------------------------------------------------
    # REMOVE CONSTANT TEMPORAL FEATURES
    # --------------------------------------------------------

    print("Removing constant temporal features...")

    useful_temporal_features = []

    for feature in temporal_features:

        if feature not in result.columns:
            continue

        if result[feature].nunique(dropna=False) > 1:
            useful_temporal_features.append(feature)

    removed = len(temporal_features) - len(
        useful_temporal_features
    )

    print(
        f"Temporal features created : {len(temporal_features)}"
    )
    print(
        f"Constant temporal removed : {removed}"
    )
    print(
        f"Useful temporal features  : "
        f"{len(useful_temporal_features)}"
    )

    return result, useful_temporal_features


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI-CyberShield - Temporal Feature Engineering")
    print("=" * 70)

    print(f"\nLoading: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    print(f"Original shape: {df.shape}")

    robust_features = load_robust_features()

    print(
        f"Existing robust features: "
        f"{len(robust_features)}"
    )

    # Only features actually present in dataset
    base_features = [
        feature
        for feature in robust_features
        if feature in df.columns
    ]

    print(
        f"Usable base features: "
        f"{len(base_features)}"
    )

    # --------------------------------------------------------
    # CREATE FEATURES
    # --------------------------------------------------------

    result, temporal_features = create_temporal_features(
        df,
        base_features
    )

    # --------------------------------------------------------
    # SAVE DATASET
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # SAVE FEATURE LIST
    # --------------------------------------------------------

    feature_list = {
        "base_features": base_features,
        "temporal_features": temporal_features,
        "all_features": (
            base_features +
            temporal_features
        )
    }

    with open(FEATURES_PATH, "w") as f:

        json.dump(
            feature_list,
            f,
            indent=2
        )

       # --------------------------------------------------------
    # FINAL CHECKS
    # --------------------------------------------------------

    model_features = base_features + temporal_features

    missing_values = (
        result[model_features]
        .isna()
        .sum()
        .sum()
    )

    print("\n" + "=" * 70)
    print("TEMPORAL DATASET SUMMARY")
    print("=" * 70)

    print(f"Rows                  : {len(result)}")

    print(
        f"Base features         : "
        f"{len(base_features)}"
    )

    print(
        f"Temporal features     : "
        f"{len(temporal_features)}"
    )

    print(
        f"Total model features  : "
        f"{len(model_features)}"
    )

    print(
        f"Missing values        : "
        f"{missing_values}"
    )

    print(
        f"Output dataset        : "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Feature configuration : "
        f"{FEATURES_PATH}"
    )

    print("\nTemporal feature engineering complete.")
    
    
if __name__ == "__main__":
    main() 