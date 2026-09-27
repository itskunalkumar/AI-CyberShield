import json
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# CONFIGURATION
# ============================================================

TEST_PATH = "data/processed/splits/test.csv"
FEATURES_PATH = "data/processed/robust_features.json"
MODEL_PATH = "models/xgboost_robust_detector.json"

THRESHOLD = 0.30


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AI-CyberShield - Scenario Probability Analysis")
print("=" * 70)

print(f"\nFrozen threshold: {THRESHOLD}")
print(f"Model: {MODEL_PATH}")

test_df = pd.read_csv(TEST_PATH)

with open(FEATURES_PATH, "r") as f:
    feature_data = json.load(f)

# Support either a direct list or {"features": [...]}
if isinstance(feature_data, list):
    features = feature_data
else:
    features = feature_data["features"]

print(f"\nTest rows: {len(test_df)}")
print(f"Features used: {len(features)}")


# ============================================================
# LOAD MODEL
# ============================================================

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

X_test = test_df[features]
y_test = test_df["attack_active"].astype(int)

probabilities = model.predict_proba(X_test)[:, 1]

test_df = test_df.copy()
test_df["attack_probability"] = probabilities
test_df["predicted_attack"] = (
    test_df["attack_probability"] >= THRESHOLD
).astype(int)


# ============================================================
# OVERALL PROBABILITY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("OVERALL PROBABILITY DISTRIBUTION")
print("=" * 70)

overall_stats = test_df["attack_probability"].describe(
    percentiles=[
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    ]
)

print(overall_stats)


# ============================================================
# SCENARIO PROBABILITY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PROBABILITY DISTRIBUTION BY SCENARIO")
print("=" * 70)

scenarios = sorted(test_df["scenario"].unique())

scenario_rows = []

for scenario in scenarios:

    df = test_df[test_df["scenario"] == scenario]

    probs = df["attack_probability"]

    attacks = df[df["attack_active"] == 1]
    benign = df[df["attack_active"] == 0]

    row = {
        "scenario": scenario,
        "rows": len(df),
        "attack_rate": df["attack_active"].mean(),

        "mean": probs.mean(),
        "min": probs.min(),
        "p01": probs.quantile(0.01),
        "p05": probs.quantile(0.05),
        "p10": probs.quantile(0.10),
        "p25": probs.quantile(0.25),
        "median": probs.quantile(0.50),
        "p75": probs.quantile(0.75),
        "p90": probs.quantile(0.90),
        "p95": probs.quantile(0.95),
        "p99": probs.quantile(0.99),
        "max": probs.max(),

        "predicted_attacks": df["predicted_attack"].sum(),

        "attack_mean_prob": (
            attacks["attack_probability"].mean()
            if len(attacks) else np.nan
        ),

        "attack_median_prob": (
            attacks["attack_probability"].median()
            if len(attacks) else np.nan
        ),

        "benign_mean_prob": (
            benign["attack_probability"].mean()
            if len(benign) else np.nan
        ),

        "benign_median_prob": (
            benign["attack_probability"].median()
            if len(benign) else np.nan
        ),
    }

    scenario_rows.append(row)


scenario_stats = pd.DataFrame(scenario_rows)

print(
    scenario_stats.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# ATTACK VS BENIGN PROBABILITY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ATTACK VS BENIGN PROBABILITY DISTRIBUTION")
print("=" * 70)

for scenario in scenarios:

    df = test_df[test_df["scenario"] == scenario]

    print(f"\n--- {scenario} ---")

    for label, value in [(0, "BENIGN"), (1, "ATTACK")]:

        subset = df[df["attack_active"] == label]

        if len(subset) == 0:
            print(f"{value}: No samples")
            continue

        print(f"\n{value} ({len(subset)} samples)")

        print(
            subset["attack_probability"]
            .describe(
                percentiles=[
                    0.10,
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                    0.95,
                    0.99,
                ]
            ).to_string()
        )


# ============================================================
# S4 THRESHOLD COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("S4 ATTACK COVERAGE AT DIFFERENT THRESHOLDS")
print("=" * 70)

s4 = test_df[
    test_df["scenario"] == "S4_arp_spoof_loads_pv_two_phase"
]

s4_attacks = s4[s4["attack_active"] == 1]

thresholds = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]

print(
    f"{'Threshold':<12}"
    f"{'Detected':<12}"
    f"{'Total':<10}"
    f"{'Recall':<12}"
)

for threshold in thresholds:

    detected = (
        s4_attacks["attack_probability"] >= threshold
    ).sum()

    recall = detected / len(s4_attacks)

    print(
        f"{threshold:<12.2f}"
        f"{detected:<12}"
        f"{len(s4_attacks):<10}"
        f"{recall:<12.4f}"
    )


# ============================================================
# S4 ATTACK VS BENIGN SEPARATION
# ============================================================

print("\n" + "=" * 70)
print("S4 ATTACK VS BENIGN SEPARATION")
print("=" * 70)

s4_benign = s4[s4["attack_active"] == 0]

print("\nS4 ATTACK:")
print(
    s4_attacks["attack_probability"]
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
            0.99,
        ]
    )
)

print("\nS4 BENIGN:")
print(
    s4_benign["attack_probability"]
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
            0.99,
        ]
    )
)


# ============================================================
# S4 AUC / PR-AUC
# ============================================================

if s4["attack_active"].nunique() == 2:

    s4_auc = roc_auc_score(
        s4["attack_active"],
        s4["attack_probability"]
    )

    s4_pr_auc = average_precision_score(
        s4["attack_active"],
        s4["attack_probability"]
    )

    print("\n" + "=" * 70)
    print("S4 RANKING PERFORMANCE")
    print("=" * 70)

    print(f"ROC-AUC : {s4_auc:.4f}")
    print(f"PR-AUC  : {s4_pr_auc:.4f}")


# ============================================================
# SAVE RESULTS
# ============================================================

scenario_stats.to_csv(
    "data/processed/scenario_probability_analysis.csv",
    index=False
)

test_df[
    [
        "scenario",
        "attack_active",
        "attack_probability",
        "predicted_attack",
    ]
].to_csv(
    "data/processed/test_probabilities.csv",
    index=False
)

print("\n" + "=" * 70)
print("Analysis complete")
print("=" * 70)

print(
    "\nSaved:"
    "\n  data/processed/scenario_probability_analysis.csv"
    "\n  data/processed/test_probabilities.csv"
)