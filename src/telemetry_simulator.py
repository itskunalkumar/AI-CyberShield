"""
AI-CyberShield - Full Dataset Telemetry Simulator

Processes the COMPLETE cleaned dataset in manageable batches and sends each
batch to FastAPI. Every dataset row is included at least once; temporal
context rows are reused between adjacent batches so rolling/delta features
have historical context.

One API call creates one audit event for the newest row of that batch.
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8001/api/v1/predict",
)
API_KEY = os.getenv("API_KEY")
DATA_PATH = Path(
    os.getenv(
        "DATA_PATH",
        "data/processed/cleaned_dataset.csv",
    )
)

# 100 new rows are processed on each API call.
BATCH_SIZE = max(1, int(os.getenv("BATCH_SIZE", "100")))

# Historical rows included for temporal feature continuity.
CONTEXT_ROWS = max(0, int(os.getenv("CONTEXT_ROWS", "19")))

# Pause between requests. Set to 0 for fastest execution.
DELAY_SECONDS = max(
    0.0,
    float(os.getenv("DELAY_SECONDS", "1")),
)

CRITICALITY = min(
    1.0,
    max(0.0, float(os.getenv("CRITICALITY", "0.5"))),
)

# 0 = process every row. Set a number to test/resume with a limited run.
MAX_NEW_ROWS = max(
    0,
    int(os.getenv("MAX_NEW_ROWS", "0")),
)

ENDPOINT = os.getenv(
    "ENDPOINT",
    "microgrid-simulator",
)

if not API_KEY:
    raise RuntimeError(
        "API_KEY environment variable is not set. "
        "Set API_KEY before running the simulator."
    )

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}

EXCLUDED_COLUMNS = {
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


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def send_batch(
    scenario: str,
    batch_number: int,
    window: pd.DataFrame,
    new_start: int,
    new_end: int,
) -> dict:
    telemetry_records = []

    for _, row in window.iterrows():
        record = {
            column: safe_float(row[column])
            for column in FEATURE_COLUMNS
        }
        record["timestamp"] = row["timestamp"].isoformat()
        record["scenario"] = str(row["scenario"])
        telemetry_records.append(record)

    payload = {
        "endpoint": f"{ENDPOINT}-{scenario}",
        "criticality": CRITICALITY,
        "telemetry": telemetry_records,
    }

    response = requests.post(
        API_URL,
        json=payload,
        headers=HEADERS,
        timeout=60,
    )
    response.raise_for_status()

    result = response.json()

    attack_probability = safe_float(
        result.get("attack_probability", 0)
    )
    anomaly_score = safe_float(
        result.get("anomaly_score", 0)
    )
    risk_score = safe_float(
        result.get("risk_score", 0)
    )

    print(
        f"[{scenario}] "
        f"Batch={batch_number:03d} | "
        f"NewRows={new_start + 1:05d}-{new_end:05d} | "
        f"Payload={len(window):03d} | "
        f"Attack={attack_probability:.3f} | "
        f"Pred={result.get('attack_prediction', 0)} | "
        f"Anomaly={anomaly_score:.3f} | "
        f"AnomPred={result.get('anomaly_prediction', 0)} | "
        f"Risk={risk_score:6.2f} | "
        f"Level={result.get('risk_level', 'UNKNOWN'):<8} | "
        f"Alert={str(result.get('alert', False)):<5} | "
        f"Action={result.get('recommended_action', 'UNKNOWN')}"
    )

    return result


print("\n" + "=" * 90)
print("AI-CYBERSHIELD - FULL DATASET TELEMETRY SIMULATOR")
print("=" * 90)
print(f"API URL        : {API_URL}")
print(f"Dataset        : {DATA_PATH}")
print(f"Batch size     : {BATCH_SIZE} NEW rows/request")
print(f"Context rows   : {CONTEXT_ROWS}")
print(f"Delay          : {DELAY_SECONDS}s")
print(f"Criticality    : {CRITICALITY}")
print(
    "Max new rows   : "
    + ("ALL" if MAX_NEW_ROWS == 0 else str(MAX_NEW_ROWS))
)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

required_columns = {"timestamp", "scenario"}
missing = required_columns.difference(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
)

if df["timestamp"].isna().any():
    bad_rows = int(df["timestamp"].isna().sum())
    raise ValueError(
        f"Found {bad_rows} rows with invalid timestamps."
    )

FEATURE_COLUMNS = [
    column
    for column in df.columns
    if column not in EXCLUDED_COLUMNS
    and pd.api.types.is_numeric_dtype(df[column])
]

if not FEATURE_COLUMNS:
    raise ValueError(
        "No numeric telemetry features were found."
    )

work_df = df[
    ["timestamp", "scenario"] + FEATURE_COLUMNS
].copy()

work_df[FEATURE_COLUMNS] = (
    work_df[FEATURE_COLUMNS]
    .replace([float("inf"), float("-inf")], pd.NA)
    .ffill()
    .bfill()
    .fillna(0)
)

preferred_scenarios = [
    "S0_benign_baseline",
    "S1_industroyer_pv_alt",
    "S2_industroyer_bss",
    "S1_industroyer_pv",
    "S3_arp_spoof_bss_meter_half_values",
    "S4_arp_spoof_loads_pv_two_phase",
    "S5_arp_spoof_loads_pv_bss_two_phase",
    "S6_mqtt_supply_chain_compromise",
]

actual_scenarios = [
    str(value)
    for value in work_df["scenario"].dropna().unique()
]

scenario_order = [
    scenario
    for scenario in preferred_scenarios
    if scenario in actual_scenarios
]
scenario_order.extend(
    scenario
    for scenario in actual_scenarios
    if scenario not in scenario_order
)

print("\nScenario row counts:")
for scenario in scenario_order:
    count = int(
        (work_df["scenario"] == scenario).sum()
    )
    print(f"  {scenario:<48} {count:>5,} rows")

total_rows_processed = 0
total_api_events = 0
total_attack_events = 0
total_alert_events = 0
failed_batches = 0
remaining_limit = (
    MAX_NEW_ROWS
    if MAX_NEW_ROWS > 0
    else None
)

print("\n" + "=" * 90)
print("PROCESSING COMPLETE DATASET")
print("=" * 90)

for scenario in scenario_order:
    scenario_df = (
        work_df[work_df["scenario"] == scenario]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    scenario_rows = len(scenario_df)
    if scenario_rows == 0:
        continue

    print("\n" + "-" * 90)
    print(
        f"START SCENARIO: {scenario} | "
        f"Rows={scenario_rows:,}"
    )
    print("-" * 90)

    batch_number = 0
    new_position = 0

    while new_position < scenario_rows:
        if remaining_limit is not None and remaining_limit <= 0:
            break

        batch_number += 1
        new_start = new_position
        new_end = min(
            new_position + BATCH_SIZE,
            scenario_rows,
        )

        if remaining_limit is not None:
            new_end = min(
                new_end,
                new_start + remaining_limit,
            )

        if new_end <= new_start:
            break

        # Include historical rows only as warm-up context.
        context_start = max(
            0,
            new_start - CONTEXT_ROWS,
        )
        window = scenario_df.iloc[
            context_start:new_end
        ].copy()

        try:
            result = send_batch(
                scenario=scenario,
                batch_number=batch_number,
                window=window,
                new_start=new_start,
                new_end=new_end,
            )

            total_api_events += 1
            total_rows_processed += new_end - new_start

            if result.get("attack_prediction", 0):
                total_attack_events += 1

            if result.get("alert", False):
                total_alert_events += 1

        except requests.HTTPError as exc:
            failed_batches += 1
            print(
                f"\n[HTTP ERROR] {scenario} "
                f"batch={batch_number}: {exc}"
            )
            try:
                print(
                    f"Server response: {response.text}"
                )
            except Exception:
                pass
            raise

        except requests.RequestException as exc:
            failed_batches += 1
            print(
                f"\n[REQUEST ERROR] {scenario} "
                f"batch={batch_number}: {exc}"
            )
            raise

        except Exception as exc:
            failed_batches += 1
            print(
                f"\n[ERROR] {scenario} "
                f"batch={batch_number}: {exc}"
            )
            raise

        new_position = new_end

        if remaining_limit is not None:
            remaining_limit -= new_end - new_start

        if new_position < scenario_rows:
            time.sleep(DELAY_SECONDS)

    print(
        f"COMPLETED {scenario}: "
        f"{new_position:,}/{scenario_rows:,} NEW rows processed"
    )

    if remaining_limit is not None and remaining_limit <= 0:
        break

expected_rows = (
    MAX_NEW_ROWS
    if MAX_NEW_ROWS > 0
    else len(work_df)
)

print("\n" + "=" * 90)
print("SIMULATION COMPLETED")
print("=" * 90)
print(f"New dataset rows processed : {total_rows_processed:,}")
print(f"API inference events       : {total_api_events:,}")
print(f"Attack events detected     : {total_attack_events:,}")
print(f"Alert events               : {total_alert_events:,}")
print(f"Failed batches             : {failed_batches:,}")
print(f"Expected new rows          : {expected_rows:,}")

if total_rows_processed == expected_rows:
    print("STATUS                     : COMPLETE ✅")
else:
    print("STATUS                     : INCOMPLETE ⚠️")

print("=" * 90)
