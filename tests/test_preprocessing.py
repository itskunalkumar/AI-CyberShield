import numpy as np
import pandas as pd

from src.features import LEAKAGE_COLUMNS, select_features
from src.preprocessing import clean_frame, fit_medians


def test_leakage_columns_excluded():
    df = pd.DataFrame({
        "sensor": [1.0, 2.0, 3.0],
        "attack_active": [0, 1, 0],
        "timestamp": [1, 2, 3],
    })
    assert select_features(df) == ["sensor"]
    assert "attack_active" in LEAKAGE_COLUMNS


def test_constant_columns_removed():
    df = pd.DataFrame({"a": [1.0, 2.0], "const": [5.0, 5.0]})
    assert select_features(df) == ["a"]


def test_clean_frame_fills_missing_and_inf_with_medians():
    train = pd.DataFrame({"a": [1.0, 3.0, 5.0]})
    medians = fit_medians(train, ["a"])
    new = pd.DataFrame({"a": [np.inf, np.nan]})
    out = clean_frame(new, ["a"], medians)
    assert out["a"].tolist() == [3.0, 3.0]
