import os
import numpy as np
import pandas as pd


INPUT_FILE = "data/processed/combined_dataset.csv"
OUTPUT_FILE = "data/processed/cleaned_dataset.csv"


def main():

    print("=" * 70)
    print("AI-CyberShield - Leakage-Safe Data Cleaning")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    print(f"Original shape: {df.shape}")

    # ---------------------------------------------------------
    # 1. Required columns
    # ---------------------------------------------------------

    required_columns = [
        "timestamp",
        "attack_active",
        "attack_phase",
        "scenario"
    ]

    missing_required = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_required:
        raise ValueError(
            f"Missing required columns: {missing_required}"
        )

    # ---------------------------------------------------------
    # 2. Timestamp
    # ---------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    invalid_timestamp = df["timestamp"].isna().sum()

    print(f"Invalid timestamps: {invalid_timestamp}")

    # ---------------------------------------------------------
    # 3. Target
    # ---------------------------------------------------------

    target_map = {
        True: 1,
        False: 0,
        1: 1,
        0: 0,
        "True": 1,
        "False": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0
    }

    df["attack_active"] = df["attack_active"].map(target_map)

    invalid_target = df["attack_active"].isna().sum()

    print(f"Invalid target values: {invalid_target}")

    # ---------------------------------------------------------
    # 4. Remove invalid rows
    # ---------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=[
            "timestamp",
            "attack_active",
            "scenario"
        ]
    )

    print(
        f"Rows removed due to invalid target/timestamp/scenario: "
        f"{before - len(df)}"
    )

    # ---------------------------------------------------------
    # 5. Replace infinity
    # ---------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ---------------------------------------------------------
    # 6. Remove duplicates
    # ---------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows: {duplicate_count}")

    df = df.drop_duplicates()

    # ---------------------------------------------------------
    # 7. Columns that MUST NOT become ML features
    # ---------------------------------------------------------

    leakage_columns = [
        "attack_active",
        "attack_phase",
        "attack_phase_all",
        "scenario",

        # attacker metadata
        "Attacker.event",
        "Attacker.new_value",
        "Attacker.old_value",

        # explicit MQTT attack messages
        "MQTT-Price-Signal.event",

        # raw timestamps
        "timestamp",
        "Profile.timestamp",
    ]

    leakage_columns = [
        col for col in leakage_columns
        if col in df.columns
    ]

    print("\nExcluded leakage / metadata columns:")

    for col in leakage_columns:
        print(f"  - {col}")

    # ---------------------------------------------------------
    # 8. Separate ML feature candidates
    # ---------------------------------------------------------

    feature_df = df.drop(
        columns=leakage_columns,
        errors="ignore"
    ).copy()

    # ---------------------------------------------------------
    # 9. Convert remaining columns to numeric
    # ---------------------------------------------------------

    numeric_columns = []

    for col in feature_df.columns:

        feature_df[col] = pd.to_numeric(
            feature_df[col],
            errors="coerce"
        )

        numeric_columns.append(col)

    print(
        f"\nNumeric feature candidates: "
        f"{len(numeric_columns)}"
    )

    # ---------------------------------------------------------
    # 10. Median imputation
    # ---------------------------------------------------------

    all_nan_columns = []

    for col in numeric_columns:

        if feature_df[col].isna().all():
            all_nan_columns.append(col)

    if all_nan_columns:

        print("\nRemoving completely empty columns:")

        for col in all_nan_columns:
            print(f"  - {col}")

        feature_df = feature_df.drop(
            columns=all_nan_columns
        )

    # Median imputation

    for col in feature_df.columns:

        median_value = feature_df[col].median()

        if pd.isna(median_value):

            print(
                f"Warning: Could not calculate median for {col}"
            )

        else:

            feature_df[col] = feature_df[col].fillna(
                median_value
            )

    # ---------------------------------------------------------
    # 11. Reconstruct clean dataset
    # ---------------------------------------------------------

    clean_df = pd.concat(
        [
            df[
                [
                    "timestamp",
                    "attack_active",
                    "attack_phase",
                    "scenario"
                ]
            ].reset_index(drop=True),

            feature_df.reset_index(drop=True)
        ],
        axis=1
    )

    # ---------------------------------------------------------
    # 12. Sort chronologically
    # ---------------------------------------------------------

    clean_df = clean_df.sort_values(
        by=[
            "scenario",
            "timestamp"
        ]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 13. Final validation
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL DATASET")
    print("=" * 70)

    print(f"Shape: {clean_df.shape}")

    print("\nTarget distribution:")
    print(
        clean_df["attack_active"].value_counts()
    )

    print("\nRemaining missing values:")

    missing = clean_df.isna().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:
        print("None")
    else:
        print(missing)

    print("\nScenario distribution:")
    print(
        clean_df["scenario"].value_counts()
    )

    # ---------------------------------------------------------
    # 14. Save
    # ---------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    clean_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()