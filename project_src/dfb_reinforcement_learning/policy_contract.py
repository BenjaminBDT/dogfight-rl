from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "dfb_reinforcement_learning"
    / "part3_policy_contract_v1.json"
)


def _load_policy_contract() -> tuple[bytes, dict[str, Any]]:
    try:
        raw = POLICY_CONTRACT_PATH.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"unable to read policy contract: {POLICY_CONTRACT_PATH}") from exc
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid policy contract JSON: {POLICY_CONTRACT_PATH}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("policy contract root must be an object")
    return raw, payload


POLICY_CONTRACT_BYTES, POLICY_CONTRACT = _load_policy_contract()
POLICY_CONTRACT_SHA256 = hashlib.sha256(POLICY_CONTRACT_BYTES).hexdigest()

POLICY_CONTRACT_ID = str(POLICY_CONTRACT["policy_contract_id"])
OBSERVATION_SCHEMA_ID = str(POLICY_CONTRACT["observation"]["schema_id"])
ACTION_SCHEMA_ID = str(POLICY_CONTRACT["action"]["schema_id"])
NORMALIZER_SCHEMA_ID = str(POLICY_CONTRACT["normalizer_schema_id"])
DATASET_SCHEMA_ID = str(POLICY_CONTRACT["dataset_schema_id"])
CHECKPOINT_SCHEMA_ID = str(POLICY_CONTRACT["checkpoint_schema_id"])
MODEL_FAMILY_ID = str(POLICY_CONTRACT["model_family_id"])
OBS_DIM = int(POLICY_CONTRACT["observation"]["dim"])
ACTION_CONT_DIM = len(POLICY_CONTRACT["action"]["continuous"])
ACTION_BIN_DIM = len(POLICY_CONTRACT["action"]["binary"])
