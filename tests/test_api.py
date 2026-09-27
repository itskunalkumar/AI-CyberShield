import json
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "temporal_dataset.csv"
)

API_URL = "http://127.0.0.1:8000/api/v1/predict"


def main():

    print("=" * 70)
    print("AI-CyberShield - FastAPI Prediction Test")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset shape: {df.shape}")

    # --------------------------------------------------------
    # Use test telemetry window
    # --------------------------------------------------------

    telemetry = df.iloc[0:20].copy()

    # Convert DataFrame to JSON-compatible records
    telemetry_records = telemetry.to_dict(
        orient="records"
    )

    payload = {
        "telemetry": telemetry_records,
        "criticality": 0.5,
        "endpoint": "microgrid-test-endpoint",
    }

    print(
        f"Telemetry rows sent to API: "
        f"{len(telemetry_records)}"
    )

    # --------------------------------------------------------
    # Send request
    # --------------------------------------------------------

    response = requests.post(
        API_URL,
        json=payload,
        timeout=120,
    )

    print(
        f"\nHTTP Status: {response.status_code}"
    )

    # --------------------------------------------------------
    # Display response
    # --------------------------------------------------------

    try:

        result = response.json()

        print("\nAPI RESPONSE")
        print("=" * 70)

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

    except Exception:

        print(
            "\nRaw response:"
        )

        print(response.text)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if response.status_code == 200:

        result = response.json()

        required_fields = [
            "attack_probability",
            "attack_prediction",
            "anomaly_score",
            "risk_score",
            "risk_level",
            "recommended_action",
            "operator_confirmation_required",
            "logical_isolation",
            "alert",
            "allow_automated_control",
            "shap_explanation",
            "physical_breaker_control",
        ]

        missing = [
            field
            for field in required_fields
            if field not in result
        ]

        if missing:

            print(
                "\n❌ Missing response fields:"
            )

            print(missing)

        else:

            print(
                "\n✅ FastAPI prediction test passed."
            )

            print(
                f"Risk Level: "
                f"{result['risk_level']}"
            )

            print(
                f"Risk Score: "
                f"{result['risk_score']}"
            )

            print(
                f"Attack Probability: "
                f"{result['attack_probability']}"
            )

            print(
                "Physical Breaker Control: "
                f"{result['physical_breaker_control']}"
            )

    else:

        print(
            "\n❌ FastAPI prediction request failed."
        )


if __name__ == "__main__":
    main()