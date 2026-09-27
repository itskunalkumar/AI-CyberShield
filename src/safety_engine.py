"""
AI-CyberShield
Deterministic Safety / Policy Engine

IMPORTANT:
This engine does NOT directly operate physical breakers.

The ML system provides a security assessment.
This deterministic engine converts that assessment into
a safe logical response and operator-control policy.
"""

from dataclasses import dataclass
from typing import Optional

from .risk_engine import RiskAssessment


@dataclass
class SafetyDecision:
    """
    Deterministic safety decision produced from a RiskAssessment.
    """

    action: str
    allow_control: bool
    operator_confirmation_required: bool
    logical_isolation: bool
    alert: bool
    reason: str
    endpoint: Optional[str] = None


class SafetyEngine:
    """
    Deterministic safety policy engine.

    Risk policy:

        LOW
            -> NORMAL_OPERATION

        MEDIUM
            -> MONITOR

        HIGH
            -> PROTECTIVE_MODE

        CRITICAL
            -> SAFE_STATE_REVIEW

    No physical breaker control is performed here.
    """

    def __init__(self):
        pass

    # ========================================================
    # REASON BUILDER
    # ========================================================

    def _build_reason(
        self,
        risk_level: str,
        risk_score: float,
        endpoint: Optional[str] = None,
    ) -> str:

        endpoint_text = (
            f" for endpoint '{endpoint}'"
            if endpoint
            else ""
        )

        if risk_level == "LOW":

            return (
                f"Risk score {risk_score:.2f} is LOW"
                f"{endpoint_text}. "
                "Continue normal monitoring."
            )

        if risk_level == "MEDIUM":

            return (
                f"Risk score {risk_score:.2f} is MEDIUM"
                f"{endpoint_text}. "
                "Continue operation with increased monitoring "
                "and alerting."
            )

        if risk_level == "HIGH":

            return (
                f"Risk score {risk_score:.2f} is HIGH"
                f"{endpoint_text}. "
                "Automated control is blocked. "
                "Operator confirmation is required and the "
                "logical endpoint should be isolated."
            )

        if risk_level == "CRITICAL":

            return (
                f"Risk score {risk_score:.2f} is CRITICAL"
                f"{endpoint_text}. "
                "Automated control is blocked. "
                "Operator confirmation and safe-state review "
                "are required."
            )

        return (
            f"Risk score {risk_score:.2f} has an unknown "
            f"risk level '{risk_level}'"
            f"{endpoint_text}. "
            "Fail safely and require operator review."
        )

    # ========================================================
    # SAFETY EVALUATION
    # ========================================================

    def evaluate(
        self,
        assessment: RiskAssessment,
        endpoint: Optional[str] = None,
    ) -> SafetyDecision:

        risk_level = str(
            assessment.risk_level
        ).upper()

        risk_score = float(
            assessment.risk_score
        )

        # ----------------------------------------------------
        # LOW
        # ----------------------------------------------------

        if risk_level == "LOW":

            return SafetyDecision(
                action="NORMAL_OPERATION",
                allow_control=True,
                operator_confirmation_required=False,
                logical_isolation=False,
                alert=False,
                reason=self._build_reason(
                    risk_level,
                    risk_score,
                    endpoint,
                ),
                endpoint=endpoint,
            )

        # ----------------------------------------------------
        # MEDIUM
        # ----------------------------------------------------

        if risk_level == "MEDIUM":

            return SafetyDecision(
                action="MONITOR",
                allow_control=True,
                operator_confirmation_required=False,
                logical_isolation=False,
                alert=True,
                reason=self._build_reason(
                    risk_level,
                    risk_score,
                    endpoint,
                ),
                endpoint=endpoint,
            )

        # ----------------------------------------------------
        # HIGH
        # ----------------------------------------------------

        if risk_level == "HIGH":

            return SafetyDecision(
                action="PROTECTIVE_MODE",
                allow_control=False,
                operator_confirmation_required=True,
                logical_isolation=True,
                alert=True,
                reason=self._build_reason(
                    risk_level,
                    risk_score,
                    endpoint,
                ),
                endpoint=endpoint,
            )

        # ----------------------------------------------------
        # CRITICAL
        # ----------------------------------------------------

        if risk_level == "CRITICAL":

            return SafetyDecision(
                action="SAFE_STATE_REVIEW",
                allow_control=False,
                operator_confirmation_required=True,
                logical_isolation=True,
                alert=True,
                reason=self._build_reason(
                    risk_level,
                    risk_score,
                    endpoint,
                ),
                endpoint=endpoint,
            )

        # ----------------------------------------------------
        # UNKNOWN RISK LEVEL
        # ----------------------------------------------------
        #
        # Fail safely.
        # Never allow automated control if the risk level
        # cannot be interpreted.
        # ----------------------------------------------------

        return SafetyDecision(
            action="SAFE_STATE_REVIEW",
            allow_control=False,
            operator_confirmation_required=True,
            logical_isolation=True,
            alert=True,
            reason=self._build_reason(
                risk_level,
                risk_score,
                endpoint,
            ),
            endpoint=endpoint,
        )