"""
AI-CyberShield
FastAPI Security Inference API

Exposes the trained cybersecurity inference engine
through REST endpoints.

IMPORTANT:
The API does NOT directly operate physical breakers.
"""

import os
from typing import Any, Dict, List, Optional

import pandas as pd

from fastapi import Depends, FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from .audit_repository import AuditLog, create_audit_log
from .database import get_db, init_db
from .inference import predict_security_event


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="AI-CyberShield API",
    description=(
        "AI-powered cybersecurity inference API for "
        "grid-connected microgrid telemetry."
    ),
    version="1.0.0",
)


# ============================================================
# DATABASE STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    """
    Initialize database tables when the API starts.
    """
    init_db()


# ============================================================
# API KEY AUTHENTICATION
# ============================================================

def verify_api_key(
    x_api_key: Optional[str] = Header(default=None),
):
    """
    Verify API key for protected endpoints.

    The API key is loaded from the API_KEY environment variable.
    """

    expected_key = os.getenv("API_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=500,
            detail="API key is not configured on the server.",
        )

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key.",
        )

    if x_api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )

    return True


# ============================================================
# REQUEST MODELS
# ============================================================

class TelemetryRequest(BaseModel):
    """
    Request payload for security prediction.

    telemetry:
        List of telemetry records.

    criticality:
        Asset/system criticality from 0 to 1.

    endpoint:
        Logical endpoint identifier.
    """

    telemetry: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        description=(
            "Historical telemetry records used to generate "
            "temporal features."
        ),
    )

    criticality: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Criticality of the monitored asset. "
            "0 = low criticality, 1 = highest criticality."
        ),
    )

    endpoint: Optional[str] = Field(
        default=None,
        description="Logical microgrid endpoint identifier.",
    )


# ============================================================
# RESPONSE MODEL
# ============================================================

class SecurityPredictionResponse(BaseModel):
    """
    Standardized security prediction response.
    """

    attack_probability: float
    attack_prediction: int

    anomaly_score: float
    anomaly_prediction: int
    anomaly_threshold: float

    criticality: float

    risk_score: float
    risk_level: str
    recommended_action: str

    operator_confirmation_required: bool
    logical_isolation: bool
    alert: bool

    allow_automated_control: bool

    risk_reason: str
    safety_reason: str

    endpoint: Optional[str]

    shap_explanation: List[Dict[str, Any]]

    physical_breaker_control: bool


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/health",
    tags=["System"],
)
def health_check() -> Dict[str, Any]:
    """
    Check whether the AI-CyberShield API is running.
    """

    return {
        "status": "healthy",
        "service": "AI-CyberShield",
        "version": "1.0.0",
        "inference_engine": "loaded",
        "database": "connected",
        "physical_breaker_control": False,
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get(
    "/model-info",
    tags=["System"],
)
def model_info() -> Dict[str, Any]:
    """
    Return information about the currently loaded
    inference configuration.
    """

    try:

        from .inference import (
            ALL_FEATURES,
            ANOMALY_THRESHOLD,
            BASE_FEATURES,
            TEMPORAL_FEATURES,
        )

        return {
            "service": "AI-CyberShield",
            "model_type": "XGBoost + Isolation Forest",
            "base_features": len(BASE_FEATURES),
            "temporal_features": len(TEMPORAL_FEATURES),
            "total_model_features": len(ALL_FEATURES),
            "anomaly_threshold": ANOMALY_THRESHOLD,
            "attack_probability_threshold": 0.30,
            "shap_enabled": True,
            "risk_engine": True,
            "safety_engine": True,
            "audit_logging": True,
            "api_key_authentication": True,
            "physical_breaker_control": False,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to retrieve model information: "
                f"{str(exc)}"
            ),
        )


# ============================================================
# SECURITY PREDICTION
# ============================================================

@app.post(
    "/api/v1/predict",
    response_model=SecurityPredictionResponse,
    tags=["Security"],
    dependencies=[Depends(verify_api_key)],
)
def predict_security(
    request: TelemetryRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Analyze microgrid telemetry and return a complete
    cybersecurity assessment.

    Pipeline:

        Telemetry
            ↓
        Temporal Features
            ↓
        XGBoost
            ↓
        Isolation Forest
            ↓
        Risk Engine
            ↓
        Safety Engine
            ↓
        SHAP
            ↓
        PostgreSQL Audit Log
            ↓
        Security Decision

    The endpoint does NOT operate physical breakers.
    """

    try:

        # ----------------------------------------------------
        # Convert request telemetry to DataFrame
        # ----------------------------------------------------

        telemetry_df = pd.DataFrame(
            request.telemetry
        )

        if telemetry_df.empty:

            raise HTTPException(
                status_code=400,
                detail="Telemetry data cannot be empty.",
            )

        # ----------------------------------------------------
        # Run inference
        # ----------------------------------------------------

        result = predict_security_event(
            telemetry=telemetry_df,
            criticality=request.criticality,
            endpoint=request.endpoint,
        )

        # ----------------------------------------------------
        # Save prediction to PostgreSQL audit log
        # ----------------------------------------------------

        create_audit_log(
            db=db,
            prediction=result,
        )

        # ----------------------------------------------------
        # Return prediction
        # ----------------------------------------------------

        return result

    except HTTPException:
        raise

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Security inference failed: "
                f"{str(exc)}"
            ),
        )


# ============================================================
# AUDIT - RECENT EVENTS
# ============================================================

@app.get(
    "/api/v1/audit/recent",
    tags=["Audit"],
)
def get_recent_audit_logs(
    limit: int = 20,
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """
    Return the most recent security audit events.
    """

    if limit < 1 or limit > 100:

        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 100.",
        )

    records = (
        db.query(AuditLog)
        .order_by(AuditLog.timestamp.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "id": record.id,
            "timestamp": record.timestamp,
            "endpoint": record.endpoint,

            "attack_probability": (
                record.attack_probability
            ),

            "attack_prediction": (
                record.attack_prediction
            ),

            "anomaly_score": (
                record.anomaly_score
            ),

            "anomaly_prediction": (
                record.anomaly_prediction
            ),

            "criticality": record.criticality,

            "risk_score": record.risk_score,

            "risk_level": record.risk_level,

            "recommended_action": (
                record.recommended_action
            ),

            "operator_confirmation_required": (
                record.operator_confirmation_required
            ),

            "logical_isolation": (
                record.logical_isolation
            ),

            "alert": record.alert,

            "allow_automated_control": (
                record.allow_automated_control
            ),

            "risk_reason": record.risk_reason,

            "safety_reason": record.safety_reason,

            "physical_breaker_control": (
                record.physical_breaker_control
            ),

            # SHAP explainability
            "shap_explanation": (
                record.shap_explanation
            ),
        }
        for record in records
    ]


# ============================================================
# AUDIT - SUMMARY
# ============================================================

@app.get(
    "/api/v1/audit/summary",
    tags=["Audit"],
)
def get_audit_summary(
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Return aggregated security audit statistics.
    """

    total_events = (
        db.query(func.count(AuditLog.id))
        .scalar()
        or 0
    )

    attack_events = (
        db.query(func.count(AuditLog.id))
        .filter(
            AuditLog.attack_prediction == 1
        )
        .scalar()
        or 0
    )

    anomaly_events = (
        db.query(func.count(AuditLog.id))
        .filter(
            AuditLog.anomaly_prediction == 1
        )
        .scalar()
        or 0
    )

    alert_events = (
        db.query(func.count(AuditLog.id))
        .filter(
            AuditLog.alert.is_(True)
        )
        .scalar()
        or 0
    )

    high_risk_events = (
        db.query(func.count(AuditLog.id))
        .filter(
            AuditLog.risk_level == "HIGH"
        )
        .scalar()
        or 0
    )

    critical_risk_events = (
        db.query(func.count(AuditLog.id))
        .filter(
            AuditLog.risk_level == "CRITICAL"
        )
        .scalar()
        or 0
    )

    medium_risk_events = (
        db.query(func.count(AuditLog.id))
        .filter(
            AuditLog.risk_level == "MEDIUM"
        )
        .scalar()
        or 0
    )

    low_risk_events = (
        db.query(func.count(AuditLog.id))
        .filter(
            AuditLog.risk_level == "LOW"
        )
        .scalar()
        or 0
    )

    average_risk = (
        db.query(func.avg(AuditLog.risk_score))
        .scalar()
    )

    return {
        "total_events": total_events,

        "attack_events": attack_events,

        "anomaly_events": anomaly_events,

        "alert_events": alert_events,

        "average_risk_score": (
            round(float(average_risk), 2)
            if average_risk is not None
            else 0.0
        ),

        "risk_distribution": {
            "LOW": low_risk_events,
            "MEDIUM": medium_risk_events,
            "HIGH": high_risk_events,
            "CRITICAL": critical_risk_events,
        },

        "physical_breaker_control": False,
    }


# ============================================================
# API ROOT
# ============================================================

@app.get(
    "/",
    tags=["System"],
)
def root() -> Dict[str, Any]:
    """
    API information endpoint.
    """

    return {
        "service": "AI-CyberShield",

        "description": (
            "Cybersecurity-enabled smart controller "
            "inference API for grid-connected microgrids."
        ),

        "version": "1.0.0",

        "status": "running",

        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "prediction": "/api/v1/predict",
            "recent_audit": "/api/v1/audit/recent",
            "audit_summary": "/api/v1/audit/summary",
            "documentation": "/docs",
        },

        "database": {
            "audit_logging": True,
            "audit_table": "security_audit_logs",
            "shap_storage": True,
        },

        "security": {
            "api_key_authentication": True,
        },

        "safety": {
            "physical_breaker_control": False,
            "automated_physical_actuation": False,
        },
    }