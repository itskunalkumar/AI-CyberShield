import os
import pandas as pd
import numpy as np


INPUT_FILE = "data/processed/cleaned_dataset.csv"
OUTPUT_FILE = "data/processed/model_dataset.csv"


def main():

    print("=" * 70)
    print("AI-CyberShield - Feature Selection")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    target = "attack_active"

    # ---------------------------------------------------------
    # Columns that must never become model features
    # ---------------------------------------------------------

    metadata_columns = [
        "timestamp",
        "attack_active",
        "attack_phase",
        "attack_phase_all",
        "scenario",
        "Attacker.event",
        "Attacker.new_value",
        "Attacker.old_value",
        "MQTT-Price-Signal.event",
        "Profile.timestamp",
    ]

    feature_columns = [
        col
        for col in df.columns
        if col not in metadata_columns
    ]

    X = df[feature_columns].copy()
    y = df[target].copy()

    print(f"\nInitial feature count: {X.shape[1]}")

    # ---------------------------------------------------------
    # Remove constant features
    # ---------------------------------------------------------

    constant_features = [
        col
        for col in X.columns
        if X[col].nunique(dropna=False) <= 1
    ]

    print(
        f"Constant features removed: "
        f"{len(constant_features)}"
    )

    for col in constant_features:
        print(f"  - {col}")

    X = X.drop(
        columns=constant_features
    )

    # ---------------------------------------------------------
    # Check remaining missing values
    # ---------------------------------------------------------

    missing = X.isna().sum()

    missing = missing[missing > 0]

    print("\nRemaining missing values:")

    if len(missing) == 0:
        print("None")
    else:
        print(missing)

    # ---------------------------------------------------------
    # Remove duplicate feature columns
    # ---------------------------------------------------------

    duplicate_columns = []

    columns = X.columns

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            if X[columns[i]].equals(
                X[columns[j]]
            ):
                duplicate_columns.append(
                    columns[j]
                )

    duplicate_columns = list(
        dict.fromkeys(duplicate_columns)
    )

    print(
        f"\nExactly duplicate feature columns: "
        f"{len(duplicate_columns)}"
    )

    if duplicate_columns:

        for col in duplicate_columns[:50]:
            print(f"  - {col}")

        X = X.drop(
            columns=duplicate_columns
        )

    # ---------------------------------------------------------
    # Final dataset
    # ---------------------------------------------------------

    model_df = pd.concat(
        [
            X,
            y
        ],
        axis=1
    )

    print("\n" + "=" * 70)
    print("FINAL MODEL DATASET")
    print("=" * 70)

    print(
        f"Rows:     {model_df.shape[0]:,}"
    )

    print(
        f"Features: {X.shape[1]:,}"
    )

    print("\nTarget distribution:")

    print(
        y.value_counts()
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    model_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()