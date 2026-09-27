"""Inference: combines detector, anomaly model, family classifier and policy engine."""
import joblib
import numpy as np
import pandas as pd

from .config import DEFAULT_THRESHOLD, MODEL_DIR
from .preprocessing import clean_frame
from .risk_engine import risk_level, risk_score
from .safety_engine import safety_action


class CyberShieldPredictor:
    def __init__(self, model_dir=MODEL_DIR):
        model_dir = MODEL_DIR if model_dir is None else model_dir
        det = joblib.load(model_dir / "detector.joblib")
        self.detector = det["model"]
        self.features = det["features"]
        self.medians = pd.Series(det["medians"])
        self.threshold = float(det.get("threshold", DEFAULT_THRESHOLD))

        ano = joblib.load(model_dir / "anomaly.joblib")
        self.anomaly = ano["model"]
        self.anomaly_features = ano["features"]
        self.anomaly_medians = pd.Series(ano["medians"])

        clf_path = model_dir / "attack_classifier.joblib"
        self.classifier = joblib.load(clf_path) if clf_path.exists() else None

    def _frame(self, payload: dict, features, medians) -> pd.DataFrame:
        return clean_frame(pd.DataFrame([payload]), features, medians)

    def predict(self, payload: dict, criticality: float = 0.5) -> dict:
        p = float(self.detector.predict_proba(self._frame(payload, self.features, self.medians))[0, 1])
        attack_detected = p >= self.threshold

        raw = float(self.anomaly.decision_function(
            self._frame(payload, self.anomaly_features, self.anomaly_medians))[0])
        anomaly = float(np.clip(0.5 - raw, 0, 1))  # larger = more anomalous

        family, family_prob = None, None
        if attack_detected and self.classifier is not None:
            probs = self.classifier["model"].predict_proba(
                self._frame(payload, self.classifier["features"], pd.Series(self.classifier["medians"])))[0]
            idx = int(np.argmax(probs))
            family, family_prob = self.classifier["classes"][idx], float(probs[idx])

        score = risk_score(p, anomaly, criticality)
        return {
            "attack_probability": round(p, 6),
            "attack_detected": attack_detected,
            "threshold": self.threshold,
            "anomaly_score": round(anomaly, 6),
            "attack_family": family,
            "attack_family_probability": family_prob,
            "risk_score": score,
            "risk_level": risk_level(score),
            "safety": safety_action(score, attack_detected),
        }
