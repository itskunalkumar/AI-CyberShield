import pytest

from src.risk_engine import RiskEngine


@pytest.fixture
def risk_engine():
    return RiskEngine()


@pytest.mark.parametrize(
    "attack_probability,anomaly_score,criticality",
    [(0, 0, 0), (1, 1, 1), (0.5, 0.2, 0.9)],
)
def test_risk_score_is_bounded(
    risk_engine,
    attack_probability,
    anomaly_score,
    criticality,
):
    score = risk_engine.calculate_risk_score(
        attack_probability,
        anomaly_score,
        criticality,
    )

    assert 0 <= score <= 100


@pytest.mark.parametrize(
    "score,expected_level",
    [(10, "LOW"), (40, "MEDIUM"), (70, "HIGH"), (90, "CRITICAL")],
)
def test_risk_level(score, expected_level):
    assert RiskEngine.get_risk_level(score) == expected_level


def test_risk_score_clips_out_of_range_inputs(risk_engine):
    assert risk_engine.calculate_risk_score(5, -3, 9) == (
        risk_engine.calculate_risk_score(1, 0, 1)
    )
