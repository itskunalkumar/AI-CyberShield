import json
from pathlib import Path

import pandas as pd

from src.inference import predict_security_event


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "temporal_dataset.csv"
)


def main():

    print("=" * 70)
    print("AI-CyberShield - End-to-End Inference Test")
    print("=" * 70)

    # --------------------------------------------------------
    # Load real telemetry
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset shape: {df.shape}")

    # --------------------------------------------------------
    # Select a telemetry history
    #
    # Use 20 consecutive observations so that
    # temporal features have historical context.
    # --------------------------------------------------------

    telemetry = df.iloc[
        0:20
    ].copy()

    print(
        f"Telemetry window: "
        f"{len(telemetry)} rows"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    result = predict_security_event(
        telemetry=telemetry,
        criticality=0.5,
        endpoint="microgrid-test-endpoint",
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SECURITY PREDICTION")
    print("=" * 70)

    print(
        f"Attack Probability : "
        f"{result['attack_probability']}"
    )

    print(
        f"Attack Prediction  : "
        f"{result['attack_prediction']}"
    )

    print(
        f"Anomaly Score      : "
        f"{result['anomaly_score']}"
    )

    print(
        f"Anomaly Prediction : "
        f"{result['anomaly_prediction']}"
    )

    print(
        f"Risk Score         : "
        f"{result['risk_score']}"
    )

    print(
        f"Risk Level         : "
        f"{result['risk_level']}"
    )

    print(
        f"Recommended Action : "
        f"{result['recommended_action']}"
    )

    print(
        f"Operator Approval  : "
        f"{result['operator_confirmation_required']}"
    )

    print(
        f"Logical Isolation  : "
        f"{result['logical_isolation']}"
    )

    print(
        f"Alert              : "
        f"{result['alert']}"
    )

    print(
        f"Automated Control  : "
        f"{result['allow_automated_control']}"
    )

    print(
        f"Physical Breaker   : "
        f"{result['physical_breaker_control']}"
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP SHAP CONTRIBUTORS")
    print("=" * 70)

    for item in result[
        "shap_explanation"
    ]:

        print(
            f"{item['feature']:<55} "
            f"SHAP={item['shap_value']:+.6f} "
            f"→ {item['direction']}"
        )

    # --------------------------------------------------------
    # Full JSON
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FULL RESULT")
    print("=" * 70)

    print(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()