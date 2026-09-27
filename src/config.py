"""Central configuration: paths, scenario mapping and split definitions."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv("DATA_ROOT", PROJECT_ROOT / "data" / "raw" / "data4cyber_dataset"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
REPORT_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.5

PRIMARY_SCENARIOS = {
    "S0_benign_baseline": "benign",
    "S1_industroyer_pv": "industroyer_modbus",
    "S2_industroyer_bss": "industroyer_modbus",
    "S3_arp_spoof_bss_meter_half_values": "mitm_false_data_injection",
    "S4_arp_spoof_loads_pv_two_phase": "mitm_false_data_injection",
    "S5_arp_spoof_loads_pv_bss_two_phase": "mitm_false_data_injection",
    "S6_mqtt_supply_chain_compromise": "mqtt_price_signal_manipulation",
}

# Scenario-level holdout: no adjacent timestamps of the same scenario cross the split.
DETECTOR_TEST_SCENARIOS = {
    "S2_industroyer_bss",
    "S5_arp_spoof_loads_pv_bss_two_phase",
    "S6_mqtt_supply_chain_compromise",
}
# S4 is used as validation for threshold selection (still inside the detector training pool).
DETECTOR_VALIDATION_SCENARIO = "S4_arp_spoof_loads_pv_two_phase"

CLASSIFIER_TEST_SCENARIOS = {
    "S2_industroyer_bss",
    "S5_arp_spoof_loads_pv_bss_two_phase",
}
# S6 family is unseen during training, so it is excluded from classifier training.
UNSEEN_FAMILY_SCENARIOS = {"S6_mqtt_supply_chain_compromise"}
