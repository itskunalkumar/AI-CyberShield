"""FastAPI inference service for AI-CyberShield."""
import os
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from src.predict import CyberShieldPredictor

app = FastAPI(
    title="AI-CyberShield API",
    version="1.0.0",
    description="Cybersecurity ML inference for a grid-connected microgrid prototype.",
)

_predictor: Optional[CyberShieldPredictor] = None


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(default_factory=dict)
    criticality: float = Field(default=0.5, ge=0.0, le=1.0)


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Enforce an API key only when API_KEY is configured (useful for deployment)."""
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def get_predictor() -> CyberShieldPredictor:
    global _predictor
    if _predictor is None:
        try:
            _predictor = CyberShieldPredictor()
        except Exception as exc:  # model files missing or corrupt
            raise HTTPException(status_code=503, detail=f"Models unavailable: {exc}")
    return _predictor


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.get("/model-info", dependencies=[Depends(require_api_key)])
def model_info() -> dict:
    predictor = get_predictor()
    return {
        "detector_features": len(predictor.features),
        "threshold": predictor.threshold,
        "classifier_available": predictor.classifier is not None,
    }


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(req: PredictRequest) -> dict:
    return get_predictor().predict(req.features, req.criticality)
