from __future__ import annotations

import hashlib

from dfb_reinforcement_learning.obs.policy_schema import POLICY_OBSERVATION_SCHEMA
from dfb_reinforcement_learning.policy_contract import (
    POLICY_CONTRACT,
    POLICY_CONTRACT_BYTES,
    POLICY_CONTRACT_ID,
    POLICY_CONTRACT_PATH,
    POLICY_CONTRACT_SHA256,
)


def test_policy_contract_has_stable_identity_and_file_hash() -> None:
    assert POLICY_CONTRACT_ID == "dfb_part3_policy_contract_v1"
    assert POLICY_CONTRACT_PATH.name == "part3_policy_contract_v1.json"
    assert POLICY_CONTRACT_BYTES.endswith(b"\n")
    assert not POLICY_CONTRACT_BYTES.startswith(b"\xef\xbb\xbf")
    assert POLICY_CONTRACT_SHA256 == hashlib.sha256(POLICY_CONTRACT_BYTES).hexdigest()


def test_policy_observation_schema_is_contiguous_and_unique() -> None:
    schema = POLICY_OBSERVATION_SCHEMA
    assert schema.schema_id == "dfb_part3_policy_observation_v1"
    assert schema.dim == 69
    assert len(schema.field_names) == len(set(schema.field_names))
    assert [field.offset for field in schema.fields] == [
        sum(previous.size for previous in schema.fields[:index])
        for index in range(len(schema.fields))
    ]
    assert schema.fields[-1].offset + schema.fields[-1].size == schema.dim
    assert schema.binary_indices == (22, 24, 26, 27, 29, 45, 59, 61, 62, 64)


def test_policy_contract_action_layout_is_explicit() -> None:
    action = POLICY_CONTRACT["action"]
    assert action["schema_id"] == "dfb_part3_policy_action_v1"
    assert action["continuous"] == ["throttle_delta", "pitch", "roll", "yaw"]
    assert action["binary"] == ["brake", "fire_gun", "repair"]
