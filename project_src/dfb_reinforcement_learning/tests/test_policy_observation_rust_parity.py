from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np

from dfb_reinforcement_learning.obs.policy_adapter import PolicyObservationAdapter
from dfb_reinforcement_learning.obs.policy_schema import POLICY_OBSERVATION_SCHEMA
from dfb_reinforcement_learning.policy_contract import (
    POLICY_CONTRACT_ID,
    POLICY_CONTRACT_SHA256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = (
    PROJECT_ROOT
    / "config/dfb_reinforcement_learning/fixtures/part3_policy_observation_v1_cases.json"
)


def test_python_and_rust_policy_observations_match_shared_fixture() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--bin",
            "dfb_tool_dataset",
            "--",
            "part3-policy-observation-fixture",
            "--fixture",
            str(FIXTURE_PATH),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    rust = json.loads(completed.stdout)

    assert rust["policy_contract_id"] == POLICY_CONTRACT_ID
    assert rust["observation_schema_id"] == POLICY_OBSERVATION_SCHEMA.schema_id
    assert rust["contract_sha256"] == POLICY_CONTRACT_SHA256
    assert [case["name"] for case in rust["cases"]] == [
        "rotation",
        "head_on",
        "tail_chase",
        "crossing",
    ]

    rust_by_name = {case["name"]: case["roles"] for case in rust["cases"]}
    adapter = PolicyObservationAdapter()
    for case in fixture["cases"]:
        for role in ("fighter1", "fighter2"):
            python_vector = adapter.build(
                case["state"],
                role,
                episode_start_sim_time_seconds=case["episode_start_sim_time_seconds"],
            )["vector"]
            rust_vector = np.asarray(rust_by_name[case["name"]][role], dtype=np.float32)
            np.testing.assert_allclose(
                python_vector,
                rust_vector,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"case={case['name']} role={role}",
            )
