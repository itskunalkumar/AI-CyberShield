import os
import json
import numpy as np
import pandas as pd


TRAIN_FILE = "data/processed/splits/train.csv"
TEST_FILE = "data/processed/splits/test.csv"

OUTPUT_FILE = "data/processed/robust_features.json"


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


S4_SCENARIO = "S4_arp_spoof_loads_pv_two_phase"
S5_SCENARIO = "S5_arp_spoof_loads_pv_bss_two_phase"


SHIFT_THRESHOLD = 5.0


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
    print("AI-CyberShield - Robust Feature Analysis")
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

    s4_df = test_df[
        test_df["scenario"] == S4_SCENARIO
    ]

    s5_df = test_df[
        test_df["scenario"] == S5_SCENARIO
    ]

    X_s4 = prepare(s4_df)
    X_s5 = prepare(s5_df)

    # ---------------------------------------------------------
    # Calculate standardized mean shifts
    # ---------------------------------------------------------

    train_mean = X_train.mean()
    train_std = X_train.std()

    train_std = train_std.replace(
        0,
        np.nan
    )

    s4_shift = (
        (X_s4.mean() - train_mean)
        / train_std
    ).abs()

    s5_shift = (
        (X_s5.mean() - train_mean)
        / train_std
    ).abs()

    max_shift = pd.concat(
        [
            s4_shift.rename("s4"),
            s5_shift.rename("s5")
        ],
        axis=1
    ).max(axis=1)

    # ---------------------------------------------------------
    # Identify extreme-shift features
    # ---------------------------------------------------------

    extreme_features = (
        max_shift[
            max_shift > SHIFT_THRESHOLD
        ]
        .sort_values(
            ascending=False
        )
    )

    robust_features = [
        feature
        for feature in X_train.columns
        if feature not in extreme_features.index
    ]

    print(
        f"\nOriginal features: "
        f"{len(X_train.columns)}"
    )

    print(
        f"Extreme-shift features: "
        f"{len(extreme_features)}"
    )

    print(
        f"Robust candidate features: "
        f"{len(robust_features)}"
    )

    print("\nFeatures removed:")

    for feature, shift in extreme_features.items():

        print(
            f"{feature}: "
            f"max shift = {shift:.3f}"
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "shift_threshold": SHIFT_THRESHOLD,
                "original_feature_count": len(
                    X_train.columns
                ),
                "removed_feature_count": len(
                    extreme_features
                ),
                "robust_feature_count": len(
                    robust_features
                ),
                "removed_features": list(
                    extreme_features.index
                ),
                "features": robust_features
            },
            file,
            indent=2
        )

    print(
        f"\nSaved feature configuration:"
    )

    print(OUTPUT_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()