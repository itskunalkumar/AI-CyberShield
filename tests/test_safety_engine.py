import pytest

from src.risk_engine import RiskEngine
from src.safety_engine import SafetyEngine


@pytest.fixture
def safety_engine():
    return SafetyEngine()


def assessment_for_score(score):
    normalized_score = score / 100
    return RiskEngine().assess(
        attack_probability=normalized_score,
        anomaly_score=normalized_score,
        criticality=normalized_score,
    )


def test_low_risk_allows_normal_operation(safety_engine):
    decision = safety_engine.evaluate(assessment_for_score(10))

    assert decision.action == "NORMAL_OPERATION"
    assert decision.allow_control is True
    assert decision.operator_confirmation_required is False
    assert decision.logical_isolation is False


@pytest.mark.parametrize(
    "score,expected_action,requires_confirmation",
    [
        (40, "MONITOR", False),
        (70, "PROTECTIVE_MODE", True),
        (90, "SAFE_STATE_REVIEW", True),
    ],
)
def test_safety_escalates_with_risk(
    safety_engine,
    score,
    expected_action,
    requires_confirmation,
):
    decision = safety_engine.evaluate(assessment_for_score(score))

    assert decision.action == expected_action
    assert decision.operator_confirmation_required is requires_confirmation
    assert decision.alert is True


def test_critical_risk_blocks_automated_control(safety_engine):
    decision = safety_engine.evaluate(assessment_for_score(95))

    assert decision.action == "SAFE_STATE_REVIEW"
    assert decision.allow_control is False
    assert decision.operator_confirmation_required is True
    assert decision.logical_isolation is True
