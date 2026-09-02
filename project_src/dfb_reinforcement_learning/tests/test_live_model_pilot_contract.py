from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest
import torch

from dfb_reinforcement_learning.data import ObservationNormalizer
from dfb_reinforcement_learning.eval.rollout_bc_policy import _policy_action
from dfb_reinforcement_learning.live.model_pilot_stdio import (
    _load_model,
    _predict_action_response,
)
from dfb_reinforcement_learning.models import StatelessHybridActorCritic
from dfb_reinforcement_learning.obs.policy_adapter import PolicyObservationAdapter
from dfb_reinforcement_learning.policy_assets import (
    PolicyDatasetContract,
    checkpoint_contract_metadata,
    checkpoint_model_hyperparameters,
    validate_policy_checkpoint_initialization_payload,
    validate_policy_checkpoint_payload,
)
from dfb_reinforcement_learning.policy_contract import (
    ACTION_SCHEMA_ID,
    DATASET_SCHEMA_ID,
    NORMALIZER_SCHEMA_ID,
    OBSERVATION_SCHEMA_ID,
    OBS_DIM,
    POLICY_CONTRACT_ID,
    POLICY_CONTRACT_SHA256,
)
from dfb_reinforcement_learning.policy_inference import (
    deterministic_policy_output,
    normalize_policy_observation,
    policy_output_batch,
)
from dfb_reinforcement_learning.train.train_ppo import _normalize_obs as _normalize_train_obs


def _dataset_contract(tmp_path: Path) -> PolicyDatasetContract:
    return PolicyDatasetContract(
        root=tmp_path,
        dataset_id="dataset-fixture",
        dataset_schema_id=DATASET_SCHEMA_ID,
        policy_contract_id=POLICY_CONTRACT_ID,
        observation_schema_id=OBSERVATION_SCHEMA_ID,
        action_schema_id=ACTION_SCHEMA_ID,
        normalizer_schema_id=NORMALIZER_SCHEMA_ID,
        contract_sha256=POLICY_CONTRACT_SHA256,
        obs_dim=OBS_DIM,
        normalizer_path=tmp_path / "obs_normalizer.json",
    )


def _normalizer() -> ObservationNormalizer:
    mean = np.zeros((OBS_DIM,), dtype=np.float32)
    std = np.ones((OBS_DIM,), dtype=np.float32)
    mean[:4] = np.asarray([0.25, -0.5, 0.75, -1.0], dtype=np.float32)
    std[:4] = np.asarray([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    return ObservationNormalizer(
        normalizer_schema_id=NORMALIZER_SCHEMA_ID,
        policy_contract_id=POLICY_CONTRACT_ID,
        observation_schema_id=OBSERVATION_SCHEMA_ID,
        contract_sha256=POLICY_CONTRACT_SHA256,
        obs_dim=OBS_DIM,
        epsilon=1e-6,
        mean=mean,
        std=std,
        train_row_count=1,
        source_dataset_id="dataset-fixture",
    )


def _checkpoint_payload(dataset: PolicyDatasetContract) -> dict[str, object]:
    model = StatelessHybridActorCritic(obs_dim=OBS_DIM, hidden_dim=64, num_layers=2)
    return {
        **checkpoint_contract_metadata(dataset),
        "model_hyperparameters": checkpoint_model_hyperparameters(
            hidden_dim=64,
            num_layers=2,
            dropout=0.0,
            continuous_action_std=None,
            popart_beta=None,
            popart_min_std=None,
        ),
        "training_stage": "test",
        "update_index": 0,
        "global_step": 0,
        "args": {"hidden_dim": 64, "num_layers": 2, "dropout": 0.0},
        "model_state_dict": model.state_dict(),
    }


def _aircraft(role: str, z_position: float, z_velocity: float) -> dict[str, object]:
    return {
        "role": role,
        "position": [0.0, 300.0, z_position],
        "orientation_quat": [0.0, 0.0, 0.0, 1.0],
        "linear_velocity": [0.0, 0.0, z_velocity],
        "angular_velocity_deg": [0.0, 0.0, 0.0],
        "hit_points": 100.0,
        "throttle": 0.5,
        "brake": False,
        "stall_factor": 0.0,
        "gun_overheated": False,
        "gun_heat": 0.0,
        "is_firing": False,
        "repairing": False,
        "repair_elapsed_seconds": 0.0,
        "out_of_bounds_seconds": 0.0,
        "subsystems": [
            {"name": name, "hit_points": 20.0, "max_hit_points": 20.0}
            for name in ("LeftWing", "RightWing", "PitchTail", "YawTail", "Engine")
        ],
    }


def _state() -> dict[str, object]:
    return {
        "tick": 120,
        "sim_time_seconds": 12.0,
        "scene_name": "contract_parity",
        "arena": {"arena_radius": 5000.0},
        "aircraft": [
            _aircraft("fighter1", 0.0, 60.0),
            _aircraft("fighter2", 300.0, -60.0),
        ],
    }


def test_live_model_loads_matching_asset_set(tmp_path: Path) -> None:
    dataset = _dataset_contract(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(_checkpoint_payload(dataset), checkpoint_path)
    loaded = _load_model(
        checkpoint_path,
        _normalizer(),
        dataset,
        torch.device("cpu"),
    )
    assert loaded.shared_stem[0].weight.shape[1] == OBS_DIM


def test_live_model_loads_identity_extended_checkpoint(tmp_path: Path) -> None:
    dataset = _dataset_contract(tmp_path)
    model = StatelessHybridActorCritic(
        obs_dim=OBS_DIM,
        hidden_dim=64,
        num_layers=2,
        shared_extension_blocks=1,
        actor_extension_blocks=2,
    )
    payload = _checkpoint_payload(dataset)
    payload["model_hyperparameters"] = checkpoint_model_hyperparameters(
        hidden_dim=64,
        num_layers=2,
        dropout=0.0,
        shared_extension_blocks=1,
        actor_extension_blocks=2,
        critic_extension_blocks=0,
        continuous_action_std=None,
        popart_beta=None,
        popart_min_std=None,
    )
    payload["model_state_dict"] = model.state_dict()
    checkpoint_path = tmp_path / "expanded.pt"
    torch.save(payload, checkpoint_path)

    loaded = _load_model(
        checkpoint_path,
        _normalizer(),
        dataset,
        torch.device("cpu"),
    )

    assert len(loaded.shared_extension_tower) == 1
    assert len(loaded.actor_extension_tower) == 2
    assert len(loaded.critic_extension_tower) == 0


def test_live_model_rejects_same_shape_with_different_contract_hash(tmp_path: Path) -> None:
    dataset = _dataset_contract(tmp_path)
    payload = _checkpoint_payload(dataset)
    payload["contract_sha256"] = "0" * 64
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint_path)
    with pytest.raises(ValueError, match="contract_sha256"):
        _load_model(
            checkpoint_path,
            _normalizer(),
            dataset,
            torch.device("cpu"),
        )


def test_checkpoint_initialization_accepts_explicit_parent_dataset_only(
    tmp_path: Path,
) -> None:
    parent = _dataset_contract(tmp_path)
    child = replace(
        parent,
        dataset_id="dataset-dagger-child",
        initialization_parent_dataset_ids=(parent.dataset_id,),
    )
    payload = _checkpoint_payload(parent)

    validate_policy_checkpoint_initialization_payload(payload, dataset=child)
    with pytest.raises(ValueError, match="dataset_id"):
        validate_policy_checkpoint_payload(payload, dataset=child)

    unrelated = {**payload, "dataset_id": "dataset-unrelated"}
    with pytest.raises(ValueError, match="not an allowed initialization source"):
        validate_policy_checkpoint_initialization_payload(unrelated, dataset=child)


def test_train_eval_and_live_use_identical_deterministic_policy_path() -> None:
    torch.manual_seed(7)
    model = StatelessHybridActorCritic(obs_dim=OBS_DIM, hidden_dim=64, num_layers=2)
    model.eval()
    normalizer = _normalizer()
    adapter = PolicyObservationAdapter()
    state = _state()
    episode_start = 10.0
    obs = adapter.build(
        state,
        "fighter1",
        episode_start_sim_time_seconds=episode_start,
    )["vector"]
    shared_normalized = normalize_policy_observation(normalizer, obs)
    train_normalized = _normalize_train_obs(normalizer, obs)
    shared_output = deterministic_policy_output(
        model=model,
        normalizer=normalizer,
        obs=obs,
        device=torch.device("cpu"),
    )

    eval_cont, _, eval_bin_prob = _policy_action(
        model,
        normalizer,
        obs,
        torch.device("cpu"),
    )
    live = _predict_action_response(
        model=model,
        normalizer=normalizer,
        obs_adapter=adapter,
        state=state,
        role="fighter1",
        episode_start_sim_time_seconds=episode_start,
        device=torch.device("cpu"),
        binary_threshold=0.5,
    )

    np.testing.assert_array_equal(train_normalized, shared_normalized)
    np.testing.assert_array_equal(eval_cont, shared_output.action_cont)
    np.testing.assert_array_equal(eval_bin_prob, shared_output.action_bin_prob)
    np.testing.assert_array_equal(np.asarray(live["action_cont"]), eval_cont)
    np.testing.assert_array_equal(
        np.asarray(live["action_bin_prob"]),
        eval_bin_prob,
    )


def test_policy_inference_rejects_wrong_observation_shape() -> None:
    with pytest.raises(ValueError, match="last dimension"):
        normalize_policy_observation(
            _normalizer(),
            np.zeros((OBS_DIM - 1,), dtype=np.float32),
        )


def test_batch_deterministic_policy_matches_single_policy_output() -> None:
    torch.manual_seed(9)
    model = StatelessHybridActorCritic(obs_dim=OBS_DIM, hidden_dim=64, num_layers=2)
    model.eval()
    normalizer = _normalizer()
    obs = np.stack(
        [
            np.linspace(-0.5, 0.5, OBS_DIM, dtype=np.float32),
            np.linspace(0.25, -0.25, OBS_DIM, dtype=np.float32),
        ]
    )
    batch = policy_output_batch(
        model=model,
        normalizer=normalizer,
        obs=obs,
        device=torch.device("cpu"),
        mode="deterministic",
    )
    for index in range(obs.shape[0]):
        single = deterministic_policy_output(
            model=model,
            normalizer=normalizer,
            obs=obs[index],
            device=torch.device("cpu"),
        )
        np.testing.assert_allclose(batch.action_cont[index], single.action_cont, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(batch.action_bin_prob[index], single.action_bin_prob, rtol=0.0, atol=1e-6)


def test_batch_sampled_policy_is_reproducible_and_bounded() -> None:
    torch.manual_seed(11)
    model = StatelessHybridActorCritic(obs_dim=OBS_DIM, hidden_dim=64, num_layers=2)
    model.eval()
    normalizer = _normalizer()
    obs = np.zeros((3, OBS_DIM), dtype=np.float32)
    generator_a = torch.Generator(device="cpu").manual_seed(123)
    generator_b = torch.Generator(device="cpu").manual_seed(123)
    output_a = policy_output_batch(
        model=model,
        normalizer=normalizer,
        obs=obs,
        device=torch.device("cpu"),
        mode="sampled",
        continuous_std=0.25,
        generator=generator_a,
    )
    output_b = policy_output_batch(
        model=model,
        normalizer=normalizer,
        obs=obs,
        device=torch.device("cpu"),
        mode="sampled",
        continuous_std=0.25,
        generator=generator_b,
    )
    np.testing.assert_array_equal(output_a.action_cont, output_b.action_cont)
    np.testing.assert_array_equal(output_a.action_bin, output_b.action_bin)
    assert np.all(np.abs(output_a.action_cont) <= 1.0)
