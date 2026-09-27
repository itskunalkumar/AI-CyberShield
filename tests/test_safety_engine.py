from src.safety_engine import safety_action


def test_no_attack_is_normal():
    assert safety_action(10, False)["mode"] == "NORMAL"


def test_escalation_levels():
    assert safety_action(40, True)["mode"] == "MONITOR"
    assert safety_action(70, True)["mode"] == "PROTECTIVE"
    assert safety_action(90, True)["mode"] == "SAFE_MODE"


def test_safe_mode_requires_operator_confirmation():
    assert "operator_confirmation_required" in safety_action(95, True)["actions"]
