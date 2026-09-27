import pytest

from src.risk_engine import risk_level, risk_score


@pytest.mark.parametrize("p,a,c", [(0, 0, 0), (1, 1, 1), (0.5, 0.2, 0.9)])
def test_risk_score_bounded(p, a, c):
    assert 0 <= risk_score(p, a, c) <= 100


@pytest.mark.parametrize("score,level", [(10, "LOW"), (40, "MEDIUM"), (70, "HIGH"), (90, "CRITICAL")])
def test_risk_level(score, level):
    assert risk_level(score) == level


def test_risk_score_clips_out_of_range_inputs():
    assert risk_score(5, -3, 9) == risk_score(1, 0, 1)
