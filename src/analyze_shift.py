import pandas as pd
import numpy as np


TRAIN_FILE = "data/processed/splits/train.csv"
TEST_FILE = "data/processed/splits/test.csv"


EXCLUDED_COLUMNS = [
    "attack_active",
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


def prepare(df):

    X = df.drop(
        columns=[
            col
            for col in EXCLUDED_COLUMNS
            if col in df.columns
        ],
        errors="ignore"
    )

    return X.select_dtypes(
        include=np.number
    )


def main():

    print("=" * 70)
    print("AI-CyberShield - Feature Distribution Shift Analysis")
    print("=" * 70)

    train_df = pd.read_csv(
        TRAIN_FILE,
        low_memory=False
    )

    test_df = pd.read_csv(
        TEST_FILE,
        low_memory=False
    )

    X_train = prepare(train_df)

    s4 = test_df[
        test_df["scenario"]
        == "S4_arp_spoof_loads_pv_two_phase"
    ]

    s5 = test_df[
        test_df["scenario"]
        == "S5_arp_spoof_loads_pv_bss_two_phase"
    ]

    X_s4 = prepare(s4)
    X_s5 = prepare(s5)

    # ---------------------------------------------------------
    # Compare means using standardized difference
    # ---------------------------------------------------------

    train_mean = X_train.mean()
    train_std = X_train.std()

    s4_mean = X_s4.mean()
    s5_mean = X_s5.mean()

    # Prevent division by zero
    safe_std = train_std.replace(
        0,
        np.nan
    )

    s4_shift = (
        (s4_mean - train_mean)
        / safe_std
    ).abs()

    s5_shift = (
        (s5_mean - train_mean)
        / safe_std
    ).abs()

    # ---------------------------------------------------------
    # S4
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("TOP S4 DISTRIBUTION SHIFTS")
    print("-" * 70)

    print(
        s4_shift
        .sort_values(
            ascending=False
        )
        .head(30)
    )

    # ---------------------------------------------------------
    # S5
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("TOP S5 DISTRIBUTION SHIFTS")
    print("-" * 70)

    print(
        s5_shift
        .sort_values(
            ascending=False
        )
        .head(30)
    )

    # ---------------------------------------------------------
    # Compare S4 vs S5
    # ---------------------------------------------------------

    s4_s5_difference = (
        (
            X_s4.mean()
            - X_s5.mean()
        )
        / safe_std
    ).abs()

    print("\n" + "-" * 70)
    print("S4 vs S5 DISTRIBUTION DIFFERENCES")
    print("-" * 70)

    print(
        s4_s5_difference
        .sort_values(
            ascending=False
        )
        .head(30)
    )

    print("\n" + "=" * 70)
    print("Analysis complete")
    print("=" * 70)


if __name__ == "__main__":
    main()