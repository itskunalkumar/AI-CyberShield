"""Load Data4Cyber scenario CSV files into a single DataFrame."""
import json

import pandas as pd

from .config import DATA_ROOT, PRIMARY_SCENARIOS


def load_scenario(scenario: str) -> pd.DataFrame:
    path = DATA_ROOT / scenario / "dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/prepare_data.py or set DATA_ROOT."
        )
    df = pd.read_csv(path)
    df["scenario"] = scenario

    meta_path = DATA_ROOT / scenario / "scenario_definition.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        df["attack_family"] = meta.get("attack_family", PRIMARY_SCENARIOS[scenario])
    else:
        df["attack_family"] = PRIMARY_SCENARIOS[scenario]
    return df


def load_primary() -> pd.DataFrame:
    frames = [load_scenario(s) for s in PRIMARY_SCENARIOS]
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["attack_active"] = df["attack_active"].astype(bool)
    return df
