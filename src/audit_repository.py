from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    JSON,
)

from .database import Base


class AuditLog(Base):
    """
    Stores security inference decisions for auditability.
    """

    __tablename__ = "security_audit_logs"

    id = Column(Integer, primary_key=True, index=True)

    timestamp = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    endpoint = Column(String(255), nullable=True)

    attack_probability = Column(Float, nullable=False)

    attack_prediction = Column(Integer, nullable=False)

    anomaly_score = Column(Float, nullable=False)

    anomaly_prediction = Column(Integer, nullable=False)

    criticality = Column(Float, nullable=False)

    risk_score = Column(Float, nullable=False)

    risk_level = Column(String(50), nullable=False)

    recommended_action = Column(String(100), nullable=False)

    operator_confirmation_required = Column(
        Boolean,
        nullable=False,
    )

    logical_isolation = Column(
        Boolean,
        nullable=False,
    )

    alert = Column(
        Boolean,
        nullable=False,
    )

    allow_automated_control = Column(
        Boolean,
        nullable=False,
    )

    risk_reason = Column(Text, nullable=True)

    safety_reason = Column(Text, nullable=True)

    physical_breaker_control = Column(
        Boolean,
        nullable=False,
        default=False,
    )

    # SHAP explainability results
    shap_explanation = Column(
        JSON,
        nullable=True,
    )


def create_audit_log(db, prediction: dict) -> AuditLog:
    """
    Save one inference result to PostgreSQL.
    """

    audit_log = AuditLog(
        endpoint=prediction.get("endpoint"),

        attack_probability=float(
            prediction["attack_probability"]
        ),

        attack_prediction=int(
            prediction["attack_prediction"]
        ),

        anomaly_score=float(
            prediction["anomaly_score"]
        ),

        anomaly_prediction=int(
            prediction["anomaly_prediction"]
        ),

        criticality=float(
            prediction["criticality"]
        ),

        risk_score=float(
            prediction["risk_score"]
        ),

        risk_level=prediction["risk_level"],

        recommended_action=prediction[
            "recommended_action"
        ],

        operator_confirmation_required=bool(
            prediction["operator_confirmation_required"]
        ),

        logical_isolation=bool(
            prediction["logical_isolation"]
        ),

        alert=bool(
            prediction["alert"]
        ),

        allow_automated_control=bool(
            prediction["allow_automated_control"]
        ),

        risk_reason=prediction.get(
            "risk_reason"
        ),

        safety_reason=prediction.get(
            "safety_reason"
        ),

        physical_breaker_control=bool(
            prediction.get(
                "physical_breaker_control",
                False,
            )
        ),

        # Store SHAP feature contributions
        shap_explanation=prediction.get(
            "shap_explanation"
        ),
    )

    db.add(audit_log)
    db.commit()
    db.refresh(audit_log)

    return audit_log