"""
AI-CyberShield
Risk Assessment Engine

Combines:

    1. Supervised attack probability
    2. Unsupervised anomaly score
    3. Asset criticality

The engine does NOT directly control physical equipment.

It produces a risk assessment for the deterministic
safety layer.
"""

from dataclasses import dataclass


# ============================================================
# RISK ASSESSMENT
# ============================================================

@dataclass
class RiskAssessment:
    """
    Result produced by the Risk Engine.
    """

    attack_probability: float
    anomaly_score: float
    criticality: float

    risk_score: float
    risk_level: str
    recommended_action: str

    reason: str


# ============================================================
# RISK ENGINE
# ============================================================

class RiskEngine:
    """
    Deterministic risk assessment engine.

    Risk score:

        60%  Attack Probability
        30%  Anomaly Score
        10%  Asset Criticality

    Final score is converted to a 0-100 scale.

    The Risk Engine does NOT directly control
    physical equipment.
    """

    def __init__(
        self,
        attack_weight: float = 0.60,
        anomaly_weight: float = 0.30,
        criticality_weight: float = 0.10,
    ):

        total = (
            attack_weight
            + anomaly_weight
            + criticality_weight
        )

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "Risk weights must sum to 1.0"
            )

        self.attack_weight = attack_weight
        self.anomaly_weight = anomaly_weight
        self.criticality_weight = criticality_weight

    # ========================================================
    # RISK SCORE
    # ========================================================

    def calculate_risk_score(
        self,
        attack_probability: float,
        anomaly_score: float,
        criticality: float = 0.5,
    ) -> float:
        """
        Calculate overall risk score from 0 to 100.
        """

        attack_probability = self._clip(
            attack_probability
        )

        anomaly_score = self._clip(
            anomaly_score
        )

        criticality = self._clip(
            criticality
        )

        score = (
            self.attack_weight
            * attack_probability
            +
            self.anomaly_weight
            * anomaly_score
            +
            self.criticality_weight
            * criticality
        )

        return round(
            score * 100,
            2
        )

    # ========================================================
    # RISK LEVEL
    # ========================================================

    @staticmethod
    def get_risk_level(
        risk_score: float
    ) -> str:
        """
        Convert numeric risk score into a risk level.
        """

        if risk_score < 30:
            return "LOW"

        elif risk_score < 60:
            return "MEDIUM"

        elif risk_score < 80:
            return "HIGH"

        else:
            return "CRITICAL"

    # ========================================================
    # RECOMMENDED ACTION
    # ========================================================

    @staticmethod
    def get_recommended_action(
        risk_level: str
    ) -> str:
        """
        Return the recommended action for a risk level.
        """

        actions = {
            "LOW": "NORMAL",
            "MEDIUM": "MONITOR",
            "HIGH": "PROTECTIVE",
            "CRITICAL": "SAFE_STATE_REVIEW",
        }

        if risk_level not in actions:

            raise ValueError(
                f"Unknown risk level: {risk_level}"
            )

        return actions[risk_level]

    # ========================================================
    # RISK REASON
    # ========================================================

    @staticmethod
    def build_reason(
        attack_probability: float,
        anomaly_score: float,
        criticality: float,
        risk_score: float,
        risk_level: str,
    ) -> str:
        """
        Generate a human-readable explanation
        for the calculated risk.
        """

        attack_pct = attack_probability * 100
        anomaly_pct = anomaly_score * 100
        criticality_pct = criticality * 100

        if risk_level == "LOW":

            return (
                f"Attack probability is low "
                f"({attack_pct:.2f}%), "
                f"anomaly score is low "
                f"({anomaly_pct:.2f}%), "
                f"and overall calculated risk is "
                f"LOW ({risk_score:.2f}/100). "
                f"Asset criticality is "
                f"{criticality_pct:.2f}%."
            )

        elif risk_level == "MEDIUM":

            return (
                f"Security indicators require increased "
                f"monitoring. Attack probability is "
                f"{attack_pct:.2f}%, anomaly score is "
                f"{anomaly_pct:.2f}%, and calculated risk "
                f"is MEDIUM ({risk_score:.2f}/100). "
                f"Asset criticality is "
                f"{criticality_pct:.2f}%."
            )

        elif risk_level == "HIGH":

            return (
                f"Elevated security risk detected. "
                f"Attack probability is "
                f"{attack_pct:.2f}%, anomaly score is "
                f"{anomaly_pct:.2f}%, and calculated risk "
                f"is HIGH ({risk_score:.2f}/100). "
                f"Asset criticality is "
                f"{criticality_pct:.2f}%. "
                f"Protective policy should be evaluated."
            )

        else:

            return (
                f"Critical security risk detected. "
                f"Attack probability is "
                f"{attack_pct:.2f}%, anomaly score is "
                f"{anomaly_pct:.2f}%, and calculated risk "
                f"is CRITICAL ({risk_score:.2f}/100). "
                f"Asset criticality is "
                f"{criticality_pct:.2f}%. "
                f"Safe-state review is required."
            )

    # ========================================================
    # COMPLETE ASSESSMENT
    # ========================================================

    def assess(
        self,
        attack_probability: float,
        anomaly_score: float,
        criticality: float = 0.5,
    ) -> RiskAssessment:
        """
        Calculate the complete risk assessment.
        """

        # ----------------------------------------------------
        # Validate / clip input
        # ----------------------------------------------------

        attack_probability = self._clip(
            attack_probability
        )

        anomaly_score = self._clip(
            anomaly_score
        )

        criticality = self._clip(
            criticality
        )

        # ----------------------------------------------------
        # Calculate risk
        # ----------------------------------------------------

        risk_score = self.calculate_risk_score(
            attack_probability=attack_probability,
            anomaly_score=anomaly_score,
            criticality=criticality,
        )

        # ----------------------------------------------------
        # Risk level
        # ----------------------------------------------------

        risk_level = self.get_risk_level(
            risk_score
        )

        # ----------------------------------------------------
        # Recommended action
        # ----------------------------------------------------

        action = self.get_recommended_action(
            risk_level
        )

        # ----------------------------------------------------
        # Human-readable reason
        # ----------------------------------------------------

        reason = self.build_reason(
            attack_probability=attack_probability,
            anomaly_score=anomaly_score,
            criticality=criticality,
            risk_score=risk_score,
            risk_level=risk_level,
        )

        # ----------------------------------------------------
        # Return assessment
        # ----------------------------------------------------

        return RiskAssessment(
            attack_probability=round(
                attack_probability,
                4,
            ),

            anomaly_score=round(
                anomaly_score,
                4,
            ),

            criticality=round(
                criticality,
                4,
            ),

            risk_score=risk_score,

            risk_level=risk_level,

            recommended_action=action,

            reason=reason,
        )

    # ========================================================
    # UTILITY
    # ========================================================

    @staticmethod
    def _clip(
        value: float
    ) -> float:
        """
        Restrict a value to the range [0, 1].
        """

        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )


# Backward-compatible function API used by ``src.predict`` and older tests.
def risk_score(
    attack_probability: float,
    anomaly_score: float,
    criticality: float = 0.5,
) -> float:
    return RiskEngine().calculate_risk_score(
        attack_probability,
        anomaly_score,
        criticality,
    )


def risk_level(score: float) -> str:
    return RiskEngine.get_risk_level(float(score))


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    engine = RiskEngine()

    examples = [
        {
            "name": "Normal operation",
            "attack": 0.05,
            "anomaly": 0.05,
            "criticality": 0.30,
        },
        {
            "name": "Suspicious operation",
            "attack": 0.45,
            "anomaly": 0.60,
            "criticality": 0.50,
        },
        {
            "name": "High risk",
            "attack": 0.75,
            "anomaly": 0.70,
            "criticality": 0.70,
        },
        {
            "name": "Critical",
            "attack": 0.95,
            "anomaly": 0.90,
            "criticality": 0.90,
        },
    ]

    print("=" * 70)
    print("AI-CyberShield - Risk Engine Test")
    print("=" * 70)

    for example in examples:

        result = engine.assess(
            attack_probability=example["attack"],
            anomaly_score=example["anomaly"],
            criticality=example["criticality"],
        )

        print(
            f"\n{example['name']}"
        )

        print(
            f"Attack Probability : "
            f"{result.attack_probability}"
        )

        print(
            f"Anomaly Score      : "
            f"{result.anomaly_score}"
        )

        print(
            f"Criticality        : "
            f"{result.criticality}"
        )

        print(
            f"Risk Score         : "
            f"{result.risk_score}"
        )

        print(
            f"Risk Level         : "
            f"{result.risk_level}"
        )

        print(
            f"Recommended Action : "
            f"{result.recommended_action}"
        )

        print(
            f"Reason             : "
            f"{result.reason}"
        )
