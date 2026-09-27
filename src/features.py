"""Leakage-aware numeric feature selection."""
import pandas as pd

# Columns that reveal the label or attacker annotations must never be model inputs.
LEAKAGE_COLUMNS = {
    "timestamp", "attack_active", "attack_phase", "attack_phase_all",
    "scenario", "attack_family", "Attacker.event", "Attacker.old_value",
    "Attacker.new_value", "Profile.timestamp", "MQTT-Price-Signal.event",
}


def select_features(train_df: pd.DataFrame) -> list[str]:
    """Numeric, non-leaking columns that vary within the training data."""
    cols = [
        c for c in train_df.columns
        if c not in LEAKAGE_COLUMNS and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    return [c for c in cols if train_df[c].nunique(dropna=False) > 1]
