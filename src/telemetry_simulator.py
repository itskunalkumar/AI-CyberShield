import os
import time
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8001/api/v1/predict"
)

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError(
        "API_KEY environment variable is not set. "
        "Please set API_KEY before running the simulator."
    )

DATA_PATH = Path("data/processed/cleaned_dataset.csv")

WINDOW_SIZE = 20

# For testing, keep this at 1–2 seconds.
DELAY_SECONDS = 2

CRITICALITY = 0.5

# Number of windows to process from each scenario
WINDOWS_PER_SCENARIO = int(
    os.getenv("WINDOWS_PER_SCENARIO", "10")
)

ENDPOINT = "microgrid-simulator"

# API authentication header
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("AI-CyberShield Telemetry Simulator")
print("=" * 70)

print(f"\nLoading dataset: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# TIMESTAMP
# ============================================================

if "timestamp" not in df.columns:
    raise ValueError("Column 'timestamp' not found.")

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "timestamp",
    "scenario",
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# SORT DATA
# ============================================================

df = df.sort_values(
    ["scenario", "timestamp"]
).reset_index(drop=True)


# ============================================================
# REMOVE NON-TELEMETRY COLUMNS
# ============================================================

excluded_columns = {
    "attack_active",
    "attack_phase",
    "attack_phase_all",
    "scenario",
    "Attacker.event",
    "Attacker.new_value",
    "Attacker.old_value",
    "MQTT-Price-Signal.event",
    "timestamp",
    "Profile.timestamp",
}

feature_columns = [
    col
    for col in df.columns
    if col not in excluded_columns
]


# ============================================================
# KEEP ONLY NUMERIC FEATURES
# ============================================================

feature_columns = [
    col
    for col in feature_columns
    if pd.api.types.is_numeric_dtype(df[col])
]

print(
    f"Telemetry features available: "
    f"{len(feature_columns)}"
)


# ============================================================
# CLEAN TELEMETRY
# ============================================================

telemetry_df = df[
    ["timestamp", "scenario"] + feature_columns
].copy()

telemetry_df[feature_columns] = (
    telemetry_df[feature_columns]
    .replace(
        [float("inf"), float("-inf")],
        pd.NA
    )
)

telemetry_df[feature_columns] = (
    telemetry_df[feature_columns]
    .fillna(
        telemetry_df[feature_columns].median()
    )
)


# ============================================================
# CHECK ACTUAL SCENARIOS
# ============================================================

print("\nAvailable scenarios:")

scenario_counts = (
    telemetry_df["scenario"]
    .value_counts()
)

for scenario, count in scenario_counts.items():
    print(
        f"  {scenario:<45} "
        f"{count} rows"
    )


# ============================================================
# DEMO SCENARIO ORDER
# ============================================================

demo_scenarios = [
    "S0_benign_baseline",
    "S1_industroyer_pv_alt",
    "S2_industroyer_bss",
    "S1_industroyer_pv",
    "S3_arp_spoof_bss_meter_half_values",
    "S4_arp_spoof_loads_pv_two_phase",
    "S5_arp_spoof_loads_pv_bss_two_phase",
    "S6_mqtt_supply_chain_compromise",
]

print("\n" + "=" * 70)
print("DEMO SCENARIO SEQUENCE")
print("=" * 70)

for scenario in demo_scenarios:
    print(f"  → {scenario}")


# ============================================================
# SIMULATION
# ============================================================

total_events = 0
attack_events = 0
alert_events = 0


for scenario in demo_scenarios:

    # --------------------------------------------------------
    # Get only this scenario
    # --------------------------------------------------------

    scenario_df = telemetry_df[
        telemetry_df["scenario"] == scenario
    ].reset_index(drop=True)

    if len(scenario_df) < WINDOW_SIZE:
        print(
            f"\nSkipping {scenario}: "
            f"not enough rows."
        )
        continue

    # --------------------------------------------------------
    # Number of windows
    # --------------------------------------------------------

    available_windows = (
        len(scenario_df) - WINDOW_SIZE + 1
    )

    windows_to_run = min(
        available_windows,
        WINDOWS_PER_SCENARIO
    )

    print("\n" + "=" * 70)
    print(
        f"STARTING SCENARIO: {scenario}"
    )
    print("=" * 70)

    print(
        f"Rows: {len(scenario_df)} | "
        f"Windows: {windows_to_run}"
    )

    # --------------------------------------------------------
    # Process sliding windows
    # --------------------------------------------------------

    for start in range(windows_to_run):

        end = start + WINDOW_SIZE

        window = scenario_df.iloc[start:end]

        # ----------------------------------------------------
        # Build telemetry records
        # ----------------------------------------------------

        telemetry_records = []

        for _, row in window.iterrows():

            record = {
                col: float(row[col])
                for col in feature_columns
            }

            record["timestamp"] = str(
                row["timestamp"]
            )

            record["scenario"] = str(
                row["scenario"]
            )

            telemetry_records.append(record)

        # ----------------------------------------------------
        # API PAYLOAD
        # ----------------------------------------------------

        payload = {
            "endpoint": (
                f"{ENDPOINT}-{scenario}"
            ),
            "criticality": CRITICALITY,
            "telemetry": telemetry_records,
        }

        # ----------------------------------------------------
        # SEND TO FASTAPI
        # ----------------------------------------------------

        try:

            response = requests.post(
                API_URL,
                json=payload,
                headers=HEADERS,
                timeout=30
            )

            response.raise_for_status()

            result = response.json()

            # ------------------------------------------------
            # READ RESPONSE
            # ------------------------------------------------

            attack_probability = result.get(
                "attack_probability",
                0
            )

            attack_prediction = result.get(
                "attack_prediction",
                0
            )

            anomaly_score = result.get(
                "anomaly_score",
                0
            )

            anomaly_prediction = result.get(
                "anomaly_prediction",
                0
            )

            risk_score = result.get(
                "risk_score",
                0
            )

            risk_level = result.get(
                "risk_level",
                "UNKNOWN"
            )

            action = result.get(
                "recommended_action",
                "UNKNOWN"
            )

            alert = result.get(
                "alert",
                False
            )

            operator_confirmation = result.get(
                "operator_confirmation_required",
                False
            )

            logical_isolation = result.get(
                "logical_isolation",
                False
            )

            # ------------------------------------------------
            # STATISTICS
            # ------------------------------------------------

            total_events += 1

            if attack_prediction == 1:
                attack_events += 1

            if alert:
                alert_events += 1

            # ------------------------------------------------
            # TERMINAL OUTPUT
            # ------------------------------------------------

            print(
                f"[{scenario}] "
                f"Window={start + 1:02d} | "
                f"Attack={attack_probability:.3f} | "
                f"Pred={attack_prediction} | "
                f"Anomaly={anomaly_score:.3f} | "
                f"AnomPred={anomaly_prediction} | "
                f"Risk={risk_score:6.2f} | "
                f"Level={risk_level:<8} | "
                f"Alert={str(alert):<5} | "
                f"Action={action}"
            )

            # ------------------------------------------------
            # SHOW SAFETY ACTION FOR HIGH/CRITICAL
            # ------------------------------------------------

            if risk_level in [
                "HIGH",
                "CRITICAL"
            ]:

                print(
                    "   ⚠ SECURITY EVENT"
                )

                print(
                    f"   Operator confirmation: "
                    f"{operator_confirmation}"
                )

                print(
                    f"   Logical isolation: "
                    f"{logical_isolation}"
                )

        except requests.exceptions.HTTPError as e:

            print(
                f"[HTTP ERROR] {scenario} "
                f"Window={start + 1}: "
                f"{e}"
            )

            # Show server response when available
            try:
                print(
                    f"   Response: "
                    f"{response.text}"
                )
            except Exception:
                pass

        except requests.exceptions.RequestException as e:

            print(
                f"[REQUEST ERROR] {scenario} "
                f"Window={start + 1}: {e}"
            )

        except Exception as e:

            print(
                f"[ERROR] {scenario} "
                f"Window={start + 1}: {e}"
            )

        # ----------------------------------------------------
        # WAIT BEFORE NEXT EVENT
        # ----------------------------------------------------

        time.sleep(DELAY_SECONDS)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SIMULATION COMPLETED")
print("=" * 70)

print(
    f"Total events:   {total_events}"
)

print(
    f"Attack events:  {attack_events}"
)

print(
    f"Alert events:   {alert_events}"
)

print("=" * 70)