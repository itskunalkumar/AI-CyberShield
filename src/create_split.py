import os
import pandas as pd


INPUT_FILE = "data/processed/cleaned_dataset.csv"
OUTPUT_DIR = "data/processed/splits"


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


def main():

    print("=" * 70)
    print("AI-CyberShield - Scenario-Aware Dataset Split")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # Validate scenarios
    # ---------------------------------------------------------

    defined_scenarios = set(
        TRAIN_SCENARIOS
        + VALIDATION_SCENARIOS
        + TEST_SCENARIOS
    )

    actual_scenarios = set(
        df["scenario"].unique()
    )

    missing = defined_scenarios - actual_scenarios
    unexpected = actual_scenarios - defined_scenarios

    if missing:
        raise ValueError(
            f"Missing scenarios: {sorted(missing)}"
        )

    if unexpected:
        raise ValueError(
            f"Unexpected scenarios: {sorted(unexpected)}"
        )

    # ---------------------------------------------------------
    # Split
    # ---------------------------------------------------------

    train_df = df[
        df["scenario"].isin(TRAIN_SCENARIOS)
    ].copy()

    validation_df = df[
        df["scenario"].isin(VALIDATION_SCENARIOS)
    ].copy()

    test_df = df[
        df["scenario"].isin(TEST_SCENARIOS)
    ].copy()

    # ---------------------------------------------------------
    # Sort chronologically within each split
    # ---------------------------------------------------------

    train_df = train_df.sort_values(
        ["scenario", "timestamp"]
    ).reset_index(drop=True)

    validation_df = validation_df.sort_values(
        ["scenario", "timestamp"]
    ).reset_index(drop=True)

    test_df = test_df.sort_values(
        ["scenario", "timestamp"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    train_df.to_csv(
        f"{OUTPUT_DIR}/train.csv",
        index=False
    )

    validation_df.to_csv(
        f"{OUTPUT_DIR}/validation.csv",
        index=False
    )

    test_df.to_csv(
        f"{OUTPUT_DIR}/test.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Reporting
    # ---------------------------------------------------------

    print("\nTRAIN")
    print("-" * 70)
    print(f"Rows: {len(train_df):,}")
    print(train_df["scenario"].value_counts())
    print("\nTarget:")
    print(train_df["attack_active"].value_counts())

    print("\nVALIDATION")
    print("-" * 70)
    print(f"Rows: {len(validation_df):,}")
    print(validation_df["scenario"].value_counts())
    print("\nTarget:")
    print(validation_df["attack_active"].value_counts())

    print("\nTEST")
    print("-" * 70)
    print(f"Rows: {len(test_df):,}")
    print(test_df["scenario"].value_counts())
    print("\nTarget:")
    print(test_df["attack_active"].value_counts())

    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(f"{OUTPUT_DIR}/train.csv")
    print(f"{OUTPUT_DIR}/validation.csv")
    print(f"{OUTPUT_DIR}/test.csv")


if __name__ == "__main__":
    main()