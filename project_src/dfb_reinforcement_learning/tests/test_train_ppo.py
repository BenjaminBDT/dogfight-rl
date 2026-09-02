from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dfb_reinforcement_learning.data import ObservationNormalizer
from dfb_reinforcement_learning.models import StatelessHybridActorCritic
from dfb_reinforcement_learning.policy_assets import (
    PolicyDatasetContract,
    checkpoint_contract_metadata,
    checkpoint_model_hyperparameters,
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
from dfb_reinforcement_learning.rewards import PolicyRewardConfig
from dfb_reinforcement_learning.rewards.reward_diagnostics import (
    RewardDiagnosticsAccumulator,
)
from dfb_reinforcement_learning.train.train_ppo import (
    EpisodeRecord,
    EpisodeAggregateMetrics,
    FrozenSceneAsset,
    PpoEvalMetrics,
    PpoTrainMetrics,
    _apply_optimizer_hparams,
    _apply_checkpoint_arch_args,
    _aggregate_episode_records,
    _assert_rollout_log_prob_consistency,
    _build_optimizer,
    _build_rollout_policy_observation_batch,
    _build_training_semantics,
    _clipped_normal_log_prob,
    _close_train_env_backend,
    _count_hits_from_events,
    _compute_gae,
    _compute_rollout_bootstrap_values,
    _episode_rate,
    _episode_sampling_diagnostics,
    _episode_uses_self_play,
    _evaluation_limit_reason,
    _finalize_episode_record,
    _is_new_best_eval,
    _load_warm_start,
    _mean_max_out_of_bounds_seconds,
    _materialize_base_scene_asset,
    _optimizer_learning_rates,
    _policy_reference_diagnostics,
    _new_reward_runtime,
    _reward_diagnostics_by_outcome,
    _reset_train_env_at,
    _reset_train_envs,
    _sample_policy,
    _sampled_opponent_requires_state,
    _save_checkpoint,
    _should_save_periodic_checkpoint,
    _step_train_envs,
    _target_kl_reached,
    _validate_output_dir_usage,
    _validate_resume_training_semantics,
    _validate_worker_reward_support,
)
from dfb_reinforcement_learning.train.truncation import (
    OpeningShotWindow,
    TacticalAdvantageTruncation,
)
from dfb_reinforcement_learning.envs import ResetRequest, StepRequest
from dfb_reinforcement_learning.opponents.opponent_pool import SampledOpponent


class _DummyAttackComponents:
    def __init__(self, *, shot_feasibility: float, attack_advantage: float = 0.0) -> None:
        self.shot_feasibility = shot_feasibility
        self.attack_advantage = attack_advantage


class _DummyRewardComposer:
    def __init__(self, shots: list[float]) -> None:
        self._shots = list(shots)

    def _compute_attack_advantage(self, *, attacker_state, defender_state):
        value = self._shots.pop(0)
        return _DummyAttackComponents(shot_feasibility=value, attack_advantage=value * 10.0)


class _DummyTacticalRewardComposer:
    def __init__(self, margins: list[float]) -> None:
        self._margins = list(margins)

    def build_attack_history_cache(self, info):
        del info
        margin = self._margins.pop(0)
        return {
            "self_attack": _DummyAttackComponents(
                shot_feasibility=0.0,
                attack_advantage=margin,
            ),
            "opponent_attack": _DummyAttackComponents(
                shot_feasibility=0.0,
                attack_advantage=0.0,
            ),
        }


class _FirstFeatureValueModel(torch.nn.Module):
    def forward(self, obs: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(value=obs[:, 0])


class _RewardRuntimeComposer:
    def reward_history_frame_length(self) -> int:
        return 4

    def compute(self, **kwargs) -> SimpleNamespace:
        del kwargs
        return SimpleNamespace(total=0.0)


def _test_normalizer() -> ObservationNormalizer:
    return ObservationNormalizer(
        normalizer_schema_id=NORMALIZER_SCHEMA_ID,
        policy_contract_id=POLICY_CONTRACT_ID,
        observation_schema_id=OBSERVATION_SCHEMA_ID,
        contract_sha256=POLICY_CONTRACT_SHA256,
        obs_dim=OBS_DIM,
        epsilon=1e-6,
        mean=np.full((OBS_DIM,), 2.0, dtype=np.float32),
        std=np.full((OBS_DIM,), 2.0, dtype=np.float32),
        train_row_count=1,
        source_dataset_id="test",
    )


def _dataset_contract(tmp_path: Path) -> PolicyDatasetContract:
    root = tmp_path / "dataset"
    return PolicyDatasetContract(
        root=root,
        dataset_id="dataset-fixture",
        dataset_schema_id=DATASET_SCHEMA_ID,
        policy_contract_id=POLICY_CONTRACT_ID,
        observation_schema_id=OBSERVATION_SCHEMA_ID,
        action_schema_id=ACTION_SCHEMA_ID,
        normalizer_schema_id=NORMALIZER_SCHEMA_ID,
        contract_sha256=POLICY_CONTRACT_SHA256,
        obs_dim=OBS_DIM,
        normalizer_path=root / "obs_normalizer.json",
    )


def _checkpoint_metadata(
    dataset_contract: PolicyDatasetContract,
    *,
    update_index: int = 0,
    global_step: int = 0,
) -> dict[str, object]:
    return {
        **checkpoint_contract_metadata(dataset_contract),
        "model_hyperparameters": checkpoint_model_hyperparameters(
            hidden_dim=64,
            num_layers=2,
            dropout=0.0,
            continuous_action_std=0.18,
            popart_beta=0.999,
            popart_min_std=1e-2,
        ),
        "training_stage": "test",
        "update_index": update_index,
        "global_step": global_step,
    }


def _dummy_train_metrics() -> PpoTrainMetrics:
    return PpoTrainMetrics(
        update=0,
        global_step=0,
        rollout_return_mean=0.0,
        rollout_length_mean=0.0,
        rollout_survival_seconds_mean=0.0,
        rollout_episode_duration_seconds_mean=0.0,
        rollout_self_destroy_rate=0.0,
        rollout_enemy_destroy_rate=0.0,
        rollout_mutual_destroy_rate=0.0,
        rollout_out_of_bounds_destroy_rate=0.0,
        rollout_truncated_rate=0.0,
        rollout_mean_max_out_of_bounds_seconds=0.0,
        rollout_self_mean_hit_count=0.0,
        rollout_enemy_mean_hit_count=0.0,
        rollout_episode_count=0,
        rollout_termination_reason_counts={},
        rollout_window=EpisodeAggregateMetrics(
            episode_count=0,
            mean_return=0.0,
            mean_length=0.0,
            mean_duration_seconds=0.0,
            self_destroy_rate=0.0,
            enemy_destroy_rate=0.0,
            mutual_destroy_rate=0.0,
            out_of_bounds_destroy_rate=0.0,
            truncated_rate=0.0,
            mean_max_out_of_bounds_seconds=0.0,
            self_mean_hit_count=0.0,
            enemy_mean_hit_count=0.0,
            termination_reason_counts={},
        ),
        policy_loss=0.0,
        value_loss=0.0,
        continuous_entropy=0.0,
        binary_entropy=0.0,
        approx_kl=0.0,
        max_approx_kl=0.0,
        clip_fraction=0.0,
        target_kl=0.03,
        target_kl_triggered=False,
        ppo_epochs_completed=0,
        ppo_minibatches_completed=0,
        shared_learning_rate=0.0,
        actor_learning_rate=0.0,
        critic_learning_rate=0.0,
        reward_mean=0.0,
        advantage_mean=0.0,
        action_diagnostics={},
        reward_diagnostics={},
        reward_diagnostics_by_outcome={},
        value_diagnostics={},
        policy_reference_diagnostics=None,
        rollout_collection_seconds=0.0,
        rollout_steps_per_second=0.0,
        policy_forward_seconds=0.0,
        opponent_action_seconds=0.0,
        env_step_seconds=0.0,
        reward_compute_seconds=0.0,
        diagnostics_seconds=0.0,
        env_reset_seconds=0.0,
        ppo_update_seconds=0.0,
        eval_seconds=0.0,
        checkpoint_seconds=0.0,
        update_compute_seconds=0.0,
        update_total_seconds=0.0,
        eval=None,
    )


def _semantic_args(**overrides) -> SimpleNamespace:
    payload = {
        "project_root": None,
        "scene_name": "open_ho",
        "scene_path": None,
        "scene_pool_json": None,
        "opponent_mode": "built_in_ai_imperfect",
        "opponent_pool_json": None,
        "ego_role": "fighter1",
        "ticks_per_step": 1,
        "opening_shot_window_mode": "none",
        "opening_shot_window_max_seconds": 2.5,
        "opening_shot_window_activate_threshold": 0.08,
        "opening_shot_window_keep_threshold": 0.04,
        "opening_shot_window_loss_seconds": 0.35,
        "max_episode_seconds": 0.0,
        "tactical_advantage_loss_threshold": None,
        "tactical_advantage_activate_threshold": 0.5,
        "tactical_advantage_ema_seconds": 0.35,
        "tactical_advantage_activate_seconds": 0.5,
        "tactical_advantage_loss_seconds": 0.75,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _frozen_scene_asset(tmp_path: Path) -> FrozenSceneAsset:
    scene_path = tmp_path / "open_ho.ron"
    scene_path.write_text("(test: true)\n", encoding="utf-8")
    manifest_path = tmp_path / "base_scene_manifest.json"
    manifest_path.write_text('{"scene_name": "open_ho"}\n', encoding="utf-8")
    return FrozenSceneAsset(
        scene_name="open_ho",
        scene_path=str(scene_path),
        manifest_path=str(manifest_path),
    )


def test_mixed_policy_observation_batch_only_adds_active_self_play_opponents() -> None:
    normalizer = _test_normalizer()
    ego_normalized = np.full((3, OBS_DIM), -0.5, dtype=np.float32)
    observations, slots = _build_rollout_policy_observation_batch(
        ego_obs_normalized=ego_normalized,
        self_play_env_indices=[1],
        dual_policy_slots=True,
        opponent_obs_raw_by_env=[
            None,
            np.full((OBS_DIM,), 4.0, dtype=np.float32),
            None,
        ],
        normalizer=normalizer,
    )

    np.testing.assert_array_equal(slots, np.asarray([0, 2, 4, 3]))
    np.testing.assert_allclose(observations[:3], -0.5)
    np.testing.assert_allclose(observations[3], 1.0)


def test_self_play_bootstrap_computes_each_role_value_independently() -> None:
    normalizer = _test_normalizer()
    ego_normalized = np.full((1, OBS_DIM), -0.5, dtype=np.float32)

    values = _compute_rollout_bootstrap_values(
        model=_FirstFeatureValueModel(),
        ego_obs_normalized=ego_normalized,
        device=torch.device("cpu"),
        self_play=True,
        opponent_obs_raw_by_env=[
            np.full((OBS_DIM,), 4.0, dtype=np.float32),
        ],
        normalizer=normalizer,
    )

    np.testing.assert_allclose(values, np.asarray([-0.5, 1.0], dtype=np.float32))


def test_reward_runtimes_keep_role_history_isolated() -> None:
    composer = _RewardRuntimeComposer()
    fighter1 = _new_reward_runtime(
        composer,
        initial_info={"ego_role": "fighter1", "enemy_role": "fighter2"},
    )
    fighter2 = _new_reward_runtime(
        composer,
        initial_info={"ego_role": "fighter2", "enemy_role": "fighter1"},
    )

    fighter1.advance(
        info={"ego_role": "fighter1", "enemy_role": "fighter2"},
        action_cont=np.ones((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
        terminated_or_truncated=False,
    )

    assert fighter1 is not fighter2
    assert len(fighter1.reward_history_frames) == 1
    assert fighter2.reward_history_frames == []
    assert fighter1.previous_info["ego_role"] == "fighter1"
    assert fighter2.previous_info["ego_role"] == "fighter2"


def test_load_warm_start_can_reset_critic_on_init(tmp_path: Path) -> None:
    dataset_contract = _dataset_contract(tmp_path)
    obs_dim = dataset_contract.obs_dim
    source = StatelessHybridActorCritic(obs_dim=obs_dim, hidden_dim=64, num_layers=2)
    with torch.no_grad():
        source.shared_stem[0].weight.fill_(0.125)
        source.value_head.weight.fill_(0.5)
        source.value_head.bias.fill_(0.25)
    checkpoint_path = tmp_path / "init.pt"
    torch.save(
        {
            **_checkpoint_metadata(dataset_contract),
            "model_state_dict": source.state_dict(),
        },
        checkpoint_path,
    )

    target = StatelessHybridActorCritic(obs_dim=obs_dim, hidden_dim=64, num_layers=2)
    optimizer = torch.optim.Adam(target.parameters(), lr=1e-3)

    _load_warm_start(
        target,
        optimizer,
        resume=None,
        init_checkpoint=str(checkpoint_path),
        reset_critic_on_init=True,
        device=torch.device("cpu"),
        dataset_contract=dataset_contract,
    )

    assert torch.allclose(target.shared_stem[0].weight, source.shared_stem[0].weight)
    assert torch.allclose(target.action_cont_head.weight, source.action_cont_head.weight)
    assert not torch.allclose(target.value_head.weight, source.value_head.weight)
    assert not torch.allclose(target.value_head.bias, source.value_head.bias)


def test_save_checkpoint_creates_missing_parent_directory(tmp_path: Path) -> None:
    dataset_contract = _dataset_contract(tmp_path)
    obs_dim = dataset_contract.obs_dim
    model = StatelessHybridActorCritic(obs_dim=obs_dim, hidden_dim=64, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "missing" / "checkpoints" / "latest.pt"

    _save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        args=type(
            "Args",
            (),
            {
                "output_dir": str(tmp_path),
                "hidden_dim": 64,
                    "num_layers": 2,
                    "shared_extension_blocks": 0,
                    "actor_extension_blocks": 0,
                    "critic_extension_blocks": 0,
                    "dropout": 0.0,
                "continuous_action_std": 0.18,
                "popart_beta": 0.999,
                "popart_min_std": 1e-2,
            },
        )(),
        update=7,
        global_step=1234,
        best_eval_return=12.5,
        metrics=_dummy_train_metrics(),
        dataset_contract=dataset_contract,
        training_semantics={"schema_id": "test", "sha256": "test"},
    )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["update"] == 7
    assert payload["global_step"] == 1234
    assert payload["best_eval_return"] == 12.5
    assert not list(checkpoint_path.parent.glob("*.tmp.*"))


def test_rollout_log_probs_recompute_exactly_before_update() -> None:
    torch.manual_seed(7)
    model = StatelessHybridActorCritic(obs_dim=OBS_DIM, hidden_dim=64, num_layers=2)
    observations = torch.randn((32, OBS_DIM))
    action_cont, _, action_bin, _, sampled_log_probs, _ = _sample_policy(
        model,
        observations,
        continuous_std=0.1,
    )

    max_error = _assert_rollout_log_prob_consistency(
        model=model,
        observations=observations,
        action_cont=action_cont,
        action_bin=action_bin,
        sampled_log_probs=sampled_log_probs,
        continuous_std=0.1,
    )

    assert max_error < 1e-5


def test_rollout_log_prob_guard_accepts_float32_rounding_noise() -> None:
    torch.manual_seed(9)
    model = StatelessHybridActorCritic(obs_dim=OBS_DIM, hidden_dim=64, num_layers=2)
    observations = torch.randn((32, OBS_DIM))
    action_cont, _, action_bin, _, sampled_log_probs, _ = _sample_policy(
        model,
        observations,
        continuous_std=0.1,
    )

    max_error = _assert_rollout_log_prob_consistency(
        model=model,
        observations=observations,
        action_cont=action_cont,
        action_bin=action_bin,
        sampled_log_probs=sampled_log_probs + 1.5e-4,
        continuous_std=0.1,
    )

    assert 1e-4 < max_error < 2e-4


def test_rollout_log_prob_guard_rejects_stale_policy() -> None:
    torch.manual_seed(11)
    model = StatelessHybridActorCritic(obs_dim=OBS_DIM, hidden_dim=64, num_layers=2)
    observations = torch.randn((16, OBS_DIM))
    action_cont, _, action_bin, _, sampled_log_probs, _ = _sample_policy(
        model,
        observations,
        continuous_std=0.1,
    )
    with torch.no_grad():
        model.action_cont_head.bias.add_(0.1)

    try:
        _assert_rollout_log_prob_consistency(
            model=model,
            observations=observations,
            action_cont=action_cont,
            action_bin=action_bin,
            sampled_log_probs=sampled_log_probs,
            continuous_std=0.1,
        )
    except RuntimeError as exc:
        assert "log-probability mismatch" in str(exc)
    else:
        raise AssertionError("expected stale rollout policy to be rejected")


def test_clipped_normal_log_prob_uses_boundary_probability_mass() -> None:
    mean = torch.zeros((1, 3))
    std = torch.ones((1, 3))
    action = torch.tensor([[-1.0, 0.0, 1.0]])

    actual = _clipped_normal_log_prob(action, mean, std)
    expected = (
        torch.special.log_ndtr(torch.tensor(-1.0))
        - 0.5 * torch.tensor(np.log(2.0 * np.pi), dtype=torch.float32)
        + torch.special.log_ndtr(torch.tensor(-1.0))
    )

    torch.testing.assert_close(actual.squeeze(0), expected)


def test_ppo_resume_requires_latest_checkpoint_and_clean_metrics(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    latest_path = checkpoints_dir / "latest.pt"
    torch.save({"update": 3}, latest_path)
    (output_dir / "metrics.jsonl").write_text(
        "\n".join(f'{{"update": {update}}}' for update in range(4)) + "\n",
        encoding="utf-8",
    )

    _validate_output_dir_usage(
        SimpleNamespace(resume=str(latest_path), init_checkpoint=None),
        output_dir,
    )

    periodic_path = checkpoints_dir / "update_0002.pt"
    torch.save({"update": 2}, periodic_path)
    try:
        _validate_output_dir_usage(
            SimpleNamespace(resume=str(periodic_path), init_checkpoint=None),
            output_dir,
        )
    except ValueError as exc:
        assert "checkpoints/latest.pt" in str(exc)
    else:
        raise AssertionError("expected periodic checkpoint resume to be rejected")


def test_ppo_resume_rejects_duplicate_metrics_updates(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    latest_path = checkpoints_dir / "latest.pt"
    torch.save({"update": 1}, latest_path)
    (output_dir / "metrics.jsonl").write_text(
        '{"update": 0}\n{"update": 1}\n{"update": 1}\n',
        encoding="utf-8",
    )

    try:
        _validate_output_dir_usage(
            SimpleNamespace(resume=str(latest_path), init_checkpoint=None),
            output_dir,
        )
    except ValueError as exc:
        assert "strictly increasing" in str(exc)
    else:
        raise AssertionError("expected duplicate PPO updates to be rejected")


def test_ppo_resume_rejects_changed_reward_semantics(tmp_path: Path) -> None:
    dataset_contract = _dataset_contract(tmp_path)
    checkpoint_path = tmp_path / "latest.pt"
    base_scene_asset = _frozen_scene_asset(tmp_path)
    saved = _build_training_semantics(
        args=_semantic_args(),
        dataset_contract=dataset_contract,
        reward_config=PolicyRewardConfig(),
        base_scene_asset=base_scene_asset,
        scene_pool_manifest_path=None,
    )
    torch.save({"ppo_training_semantics": saved}, checkpoint_path)
    changed = _build_training_semantics(
        args=_semantic_args(),
        dataset_contract=dataset_contract,
        reward_config=PolicyRewardConfig(fire_command_bonus_weight=3.0),
        base_scene_asset=base_scene_asset,
        scene_pool_manifest_path=None,
    )

    try:
        _validate_resume_training_semantics(
            checkpoint_path=checkpoint_path,
            expected=changed,
        )
    except ValueError as exc:
        assert "training semantics changed" in str(exc)
    else:
        raise AssertionError("expected changed reward semantics to be rejected")


def test_materialize_base_scene_reuses_frozen_snapshot_on_resume(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    scene_dir = project_root / "config" / "dfb_game" / "scenes"
    scene_dir.mkdir(parents=True)
    source_path = scene_dir / "open_ho.ron"
    source_path.write_text("(revision: 1)\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    args = _semantic_args(project_root=str(project_root), resume=None)

    frozen = _materialize_base_scene_asset(args, output_dir)
    frozen_path = Path(frozen.scene_path)
    assert frozen_path.read_text(encoding="utf-8") == "(revision: 1)\n"

    source_path.write_text("(revision: 2)\n", encoding="utf-8")
    resumed = _materialize_base_scene_asset(
        _semantic_args(project_root=str(project_root), resume="latest.pt"),
        output_dir,
    )

    assert resumed == frozen
    assert frozen_path.read_text(encoding="utf-8") == "(revision: 1)\n"


def test_load_warm_start_rejects_reset_critic_with_resume() -> None:
    obs_dim = OBS_DIM
    model = StatelessHybridActorCritic(obs_dim=obs_dim, hidden_dim=64, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset_contract = _dataset_contract(Path("."))

    try:
        _load_warm_start(
            model,
            optimizer,
            resume="dummy.pt",
            init_checkpoint=None,
            reset_critic_on_init=True,
            device=torch.device("cpu"),
            dataset_contract=dataset_contract,
        )
    except ValueError as exc:
        assert "--reset-critic-on-init" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_warm_start_rejects_checkpoint_without_contract(tmp_path: Path) -> None:
    dataset_contract = _dataset_contract(tmp_path)
    obs_dim = dataset_contract.obs_dim
    hidden_dim = 64
    checkpoint_path = tmp_path / "missing_contract_init.pt"
    torch.save(
        {
            "model_state_dict": StatelessHybridActorCritic(
                obs_dim=obs_dim,
                hidden_dim=hidden_dim,
                num_layers=4,
            ).state_dict(),
        },
        checkpoint_path,
    )

    target = StatelessHybridActorCritic(obs_dim=obs_dim, hidden_dim=hidden_dim, num_layers=4)
    optimizer = torch.optim.Adam(target.parameters(), lr=1e-3)

    try:
        _load_warm_start(
            target,
            optimizer,
            resume=None,
            init_checkpoint=str(checkpoint_path),
            reset_critic_on_init=False,
            device=torch.device("cpu"),
            dataset_contract=dataset_contract,
        )
    except ValueError as exc:
        assert "missing required field checkpoint_schema_id" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_warm_start_rejects_incompatible_model_state(tmp_path: Path) -> None:
    dataset_contract = _dataset_contract(tmp_path)
    obs_dim = dataset_contract.obs_dim
    hidden_dim = 64
    resume_path = tmp_path / "incompatible_resume.pt"
    source = StatelessHybridActorCritic(obs_dim=obs_dim, hidden_dim=hidden_dim, num_layers=2)
    incompatible_state = dict(source.state_dict())
    incompatible_state.pop("value_head.std")
    torch.save(
        {
            **_checkpoint_metadata(dataset_contract, update_index=0, global_step=0),
            "model_state_dict": incompatible_state,
            "optimizer_state_dict": torch.optim.Adam(source.parameters(), lr=1e-3).state_dict(),
        },
        resume_path,
    )
    model = StatelessHybridActorCritic(obs_dim=obs_dim, hidden_dim=hidden_dim, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    try:
        _load_warm_start(
            model,
            optimizer,
            resume=str(resume_path),
            init_checkpoint=None,
            reset_critic_on_init=False,
            device=torch.device("cpu"),
            dataset_contract=dataset_contract,
        )
    except RuntimeError as exc:
        assert "value_head.std" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_build_optimizer_uses_single_group_by_default() -> None:
    obs_dim = OBS_DIM
    model = StatelessHybridActorCritic(obs_dim=obs_dim, hidden_dim=64, num_layers=2)
    optimizer = _build_optimizer(
        model,
        learning_rate=3e-4,
        weight_decay=1e-5,
        shared_learning_rate=None,
        actor_learning_rate=None,
        critic_learning_rate=None,
    )
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 3e-4


def test_build_optimizer_supports_grouped_learning_rates() -> None:
    obs_dim = OBS_DIM
    model = StatelessHybridActorCritic(obs_dim=obs_dim, hidden_dim=64, num_layers=2)
    optimizer = _build_optimizer(
        model,
        learning_rate=3e-4,
        weight_decay=1e-5,
        shared_learning_rate=1e-4,
        actor_learning_rate=2e-4,
        critic_learning_rate=6e-4,
    )
    assert len(optimizer.param_groups) == 3
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert groups["shared"]["lr"] == 1e-4
    assert groups["actor"]["lr"] == 2e-4
    assert groups["critic"]["lr"] == 6e-4


def test_build_optimizer_assigns_extension_parameters_to_matching_groups() -> None:
    model = StatelessHybridActorCritic(
        obs_dim=OBS_DIM,
        hidden_dim=64,
        num_layers=2,
        shared_extension_blocks=1,
        actor_extension_blocks=1,
        critic_extension_blocks=1,
    )
    optimizer = _build_optimizer(
        model,
        learning_rate=3e-4,
        weight_decay=1e-5,
        shared_learning_rate=1e-4,
        actor_learning_rate=2e-4,
        critic_learning_rate=6e-4,
    )
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    parameter_ids_by_group = {
        name: {id(parameter) for parameter in group["params"]}
        for name, group in groups.items()
    }

    assert id(model.shared_extension_tower[0].gate) in parameter_ids_by_group["shared"]
    assert id(model.actor_extension_tower[0].gate) in parameter_ids_by_group["actor"]
    assert id(model.critic_extension_tower[0].gate) in parameter_ids_by_group["critic"]


def test_apply_checkpoint_arch_args_uses_explicit_extension_metadata(tmp_path: Path) -> None:
    dataset_contract = _dataset_contract(tmp_path)
    checkpoint_path = tmp_path / "expanded.pt"
    payload = {
        **_checkpoint_metadata(dataset_contract),
        "model_hyperparameters": checkpoint_model_hyperparameters(
            hidden_dim=128,
            num_layers=2,
            dropout=0.0,
            shared_extension_blocks=1,
            actor_extension_blocks=2,
            critic_extension_blocks=0,
            continuous_action_std=0.1,
            popart_beta=0.999,
            popart_min_std=1e-2,
        ),
        "model_state_dict": StatelessHybridActorCritic(
            obs_dim=OBS_DIM,
            hidden_dim=128,
            num_layers=2,
            shared_extension_blocks=1,
            actor_extension_blocks=2,
        ).state_dict(),
    }
    torch.save(payload, checkpoint_path)
    args = SimpleNamespace(
        resume=None,
        init_checkpoint=str(checkpoint_path),
        hidden_dim=64,
        num_layers=4,
        dropout=0.5,
        shared_extension_blocks=0,
        actor_extension_blocks=0,
        critic_extension_blocks=0,
    )

    _apply_checkpoint_arch_args(args, dataset_contract=dataset_contract)

    assert args.hidden_dim == 128
    assert args.num_layers == 2
    assert args.dropout == 0.0
    assert args.shared_extension_blocks == 1
    assert args.actor_extension_blocks == 2
    assert args.critic_extension_blocks == 0


def test_apply_optimizer_hparams_overrides_loaded_grouped_learning_rates() -> None:
    obs_dim = OBS_DIM
    model = StatelessHybridActorCritic(obs_dim=obs_dim, hidden_dim=64, num_layers=2)
    optimizer = _build_optimizer(
        model,
        learning_rate=3e-4,
        weight_decay=1e-5,
        shared_learning_rate=1e-10,
        actor_learning_rate=1e-10,
        critic_learning_rate=1e-4,
    )

    _apply_optimizer_hparams(
        optimizer,
        learning_rate=5e-5,
        weight_decay=2e-5,
        shared_learning_rate=1e-4,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-4,
    )

    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert groups["shared"]["lr"] == 1e-4
    assert groups["actor"]["lr"] == 1e-3
    assert groups["critic"]["lr"] == 1e-4
    assert all(group["weight_decay"] == 2e-5 for group in optimizer.param_groups)


def test_optimizer_learning_rates_reports_grouped_rates() -> None:
    obs_dim = OBS_DIM
    model = StatelessHybridActorCritic(obs_dim=obs_dim, hidden_dim=64, num_layers=2)
    optimizer = _build_optimizer(
        model,
        learning_rate=3e-4,
        weight_decay=1e-5,
        shared_learning_rate=1e-4,
        actor_learning_rate=2e-4,
        critic_learning_rate=6e-4,
    )

    assert _optimizer_learning_rates(optimizer) == (1e-4, 2e-4, 6e-4)


def test_opening_shot_window_truncates_after_window_loss_patience() -> None:
    info0 = {
        "sim_time_seconds": 0.0,
        "aircraft_by_role": {"fighter1": {}, "fighter2": {}},
    }
    composer = _DummyRewardComposer([0.10, 0.0, 0.0])
    policy = OpeningShotWindow(
        max_seconds=2.5,
        activate_threshold=0.08,
        keep_threshold=0.04,
        loss_seconds=0.35,
        reward_composer=composer,
        ego_role="fighter1",
    )
    runtime = policy.initial_runtime(info0)

    truncated, runtime = policy.check(
        info={"sim_time_seconds": 0.1, "aircraft_by_role": {"fighter1": {}, "fighter2": {}}},
        episode_stats={"start_sim_time_seconds": 0.0},
        runtime=runtime,
    )
    assert truncated is False
    assert runtime.window_seen is True
    assert runtime.window_lost_seconds == 0.0

    truncated, runtime = policy.check(
        info={"sim_time_seconds": 0.3, "aircraft_by_role": {"fighter1": {}, "fighter2": {}}},
        episode_stats={"start_sim_time_seconds": 0.0},
        runtime=runtime,
    )
    assert truncated is False
    assert abs(runtime.window_lost_seconds - 0.2) < 1e-6

    truncated, runtime = policy.check(
        info={"sim_time_seconds": 0.5, "aircraft_by_role": {"fighter1": {}, "fighter2": {}}},
        episode_stats={"start_sim_time_seconds": 0.0},
        runtime=runtime,
    )
    assert truncated is True
    assert abs(runtime.window_lost_seconds - 0.4) < 1e-6


def test_opening_shot_window_truncates_at_max_seconds_without_loss() -> None:
    composer = _DummyRewardComposer([0.10])
    policy = OpeningShotWindow(
        max_seconds=0.25,
        activate_threshold=0.08,
        keep_threshold=0.04,
        loss_seconds=0.35,
        reward_composer=composer,
        ego_role="fighter1",
    )
    runtime = policy.initial_runtime(
        {"sim_time_seconds": 0.0, "aircraft_by_role": {"fighter1": {}, "fighter2": {}}}
    )
    truncated, runtime = policy.check(
        info={"sim_time_seconds": 0.3, "aircraft_by_role": {"fighter1": {}, "fighter2": {}}},
        episode_stats={"start_sim_time_seconds": 0.0},
        runtime=runtime,
    )
    assert truncated is True
    assert runtime.window_seen is True


def test_opening_shot_window_reports_truncation_reason() -> None:
    composer = _DummyRewardComposer([0.10])
    policy = OpeningShotWindow(
        max_seconds=0.25,
        activate_threshold=0.08,
        keep_threshold=0.04,
        loss_seconds=0.35,
        reward_composer=composer,
        ego_role="fighter1",
    )
    runtime = policy.initial_runtime(
        {"sim_time_seconds": 0.0, "aircraft_by_role": {"fighter1": {}, "fighter2": {}}}
    )

    truncated, _, reason = policy.check_with_reason(
        info={"sim_time_seconds": 0.3, "aircraft_by_role": {"fighter1": {}, "fighter2": {}}},
        episode_stats={"start_sim_time_seconds": 0.0},
        runtime=runtime,
    )

    assert truncated
    assert reason == "opening_shot_window_time_limit"


def test_tactical_advantage_truncates_only_after_activation_and_sustained_loss() -> None:
    composer = _DummyTacticalRewardComposer([0.6, 0.7, -0.3, -0.4])
    policy = TacticalAdvantageTruncation(
        loss_threshold=-0.2,
        activate_threshold=0.5,
        ema_seconds=0.0,
        activate_seconds=0.2,
        loss_seconds=0.3,
        reward_composer=composer,
        ego_role="fighter1",
    )
    runtime = policy.initial_runtime({"sim_time_seconds": 0.0})
    episode_stats = {"start_sim_time_seconds": 0.0}

    for sim_time in (0.1, 0.2, 0.3):
        info = {"sim_time_seconds": sim_time}
        truncated, runtime, reason = policy.check_with_reason(
            info=info,
            episode_stats=episode_stats,
            runtime=runtime,
        )
        assert truncated is False
        assert reason is None

    assert runtime.armed is True
    assert np.isclose(runtime.loss_seconds, 0.1)

    info = {"sim_time_seconds": 0.5}
    truncated, runtime, reason = policy.check_with_reason(
        info=info,
        episode_stats=episode_stats,
        runtime=runtime,
    )

    assert truncated is True
    assert reason == "tactical_advantage_lost"
    assert np.isclose(runtime.loss_seconds, 0.3)
    assert info["tactical_advantage_window"]["raw_margin"] == -0.4
    assert info["tactical_advantage_window"]["armed"] is True


def test_tactical_advantage_hysteresis_resets_transient_loss() -> None:
    composer = _DummyTacticalRewardComposer([0.6, 0.7, -0.3, 0.0])
    policy = TacticalAdvantageTruncation(
        loss_threshold=-0.2,
        activate_threshold=0.5,
        ema_seconds=0.0,
        activate_seconds=0.2,
        loss_seconds=0.3,
        reward_composer=composer,
    )
    runtime = policy.initial_runtime({"sim_time_seconds": 0.0})
    episode_stats = {"start_sim_time_seconds": 0.0}

    for sim_time in (0.1, 0.2, 0.3, 0.4):
        truncated, runtime = policy.check(
            info={"sim_time_seconds": sim_time},
            episode_stats=episode_stats,
            runtime=runtime,
        )
        assert truncated is False

    assert runtime.armed is True
    assert runtime.loss_seconds == 0.0


def test_tactical_advantage_ema_uses_simulation_time() -> None:
    composer = _DummyTacticalRewardComposer([1.0, -1.0])
    policy = TacticalAdvantageTruncation(
        loss_threshold=-2.0,
        activate_threshold=2.0,
        ema_seconds=1.0,
        activate_seconds=0.5,
        loss_seconds=0.5,
        reward_composer=composer,
    )
    runtime = policy.initial_runtime({"sim_time_seconds": 0.0})
    episode_stats = {"start_sim_time_seconds": 0.0}
    _, runtime = policy.check(
        info={"sim_time_seconds": 0.1},
        episode_stats=episode_stats,
        runtime=runtime,
    )
    _, runtime = policy.check(
        info={"sim_time_seconds": 1.1},
        episode_stats=episode_stats,
        runtime=runtime,
    )

    expected = 1.0 + (1.0 - np.exp(-1.0)) * (-2.0)
    assert np.isclose(runtime.smoothed_margin, expected)


def test_gae_bootstraps_truncation_without_crossing_reset() -> None:
    rewards = np.asarray([[1.0], [100.0]], dtype=np.float32)
    values = np.asarray([[0.5], [20.0]], dtype=np.float32)
    dones = np.asarray([[1.0], [0.0]], dtype=np.float32)
    terminated = np.zeros_like(dones)
    truncated_bootstrap_values = np.asarray([[4.0], [0.0]], dtype=np.float32)

    advantages = _compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        terminated=terminated,
        truncated_bootstrap_values=truncated_bootstrap_values,
        next_values=np.asarray([30.0], dtype=np.float32),
        gamma=0.9,
        gae_lambda=0.8,
    )

    assert advantages[0, 0] == np.float32(1.0 + 0.9 * 4.0 - 0.5)
    assert advantages[1, 0] == np.float32(100.0 + 0.9 * 30.0 - 20.0)


def test_gae_does_not_bootstrap_true_terminal() -> None:
    advantages = _compute_gae(
        rewards=np.asarray([[1.0]], dtype=np.float32),
        values=np.asarray([[0.5]], dtype=np.float32),
        dones=np.asarray([[1.0]], dtype=np.float32),
        terminated=np.asarray([[1.0]], dtype=np.float32),
        truncated_bootstrap_values=np.asarray([[99.0]], dtype=np.float32),
        next_values=np.asarray([30.0], dtype=np.float32),
        gamma=0.9,
        gae_lambda=0.8,
    )

    assert advantages[0, 0] == np.float32(0.5)


def test_hit_metrics_match_reward_hit_event_semantics() -> None:
    self_hits, enemy_hits = _count_hits_from_events(
        [
            {"kind": "Hit", "subject": "fighter1"},
            {"kind": "Hit", "subject": "fighter2"},
            {"kind": "Hit", "subject": "fighter2:left_wing"},
            {"kind": "Damage", "subject": "fighter2"},
            {"kind": "SubsystemHit", "subject": "fighter2:left_wing"},
            {"kind": "Destroy", "subject": "fighter2"},
        ],
        self_role="fighter1",
        enemy_role="fighter2",
    )

    assert self_hits == 1
    assert enemy_hits == 2


def test_should_save_periodic_checkpoint_uses_1_based_interval_and_final_update() -> None:
    assert not _should_save_periodic_checkpoint(update=0, target_update=10, checkpoint_interval=5)
    assert _should_save_periodic_checkpoint(update=4, target_update=10, checkpoint_interval=5)
    assert not _should_save_periodic_checkpoint(update=8, target_update=10, checkpoint_interval=5)
    assert _should_save_periodic_checkpoint(update=9, target_update=10, checkpoint_interval=5)


def test_should_save_periodic_checkpoint_rejects_invalid_interval() -> None:
    try:
        _should_save_periodic_checkpoint(update=0, target_update=1, checkpoint_interval=0)
    except ValueError as exc:
        assert "checkpoint_interval" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_aggregate_episode_records_includes_survival_time() -> None:
    records = [
        EpisodeRecord(
            episode_return=10.0,
            episode_length=100,
            episode_survival_seconds=12.0,
            self_destroyed=False,
            enemy_destroyed=True,
            terminated=True,
            truncated=False,
            max_out_of_bounds_seconds=0.0,
        ),
        EpisodeRecord(
            episode_return=-5.0,
            episode_length=50,
            episode_survival_seconds=6.0,
            self_destroyed=True,
            enemy_destroyed=False,
            terminated=True,
            truncated=False,
            max_out_of_bounds_seconds=2.0,
        ),
    ]
    mean_return, mean_length, mean_survival_seconds = _aggregate_episode_records(records)
    assert mean_return == 2.5
    assert mean_length == 75.0
    assert mean_survival_seconds == 9.0
    assert _episode_rate(records, lambda item: item.enemy_destroyed) == 0.5
    assert _episode_rate(records, lambda item: item.self_destroyed) == 0.5
    assert _mean_max_out_of_bounds_seconds(records) == 1.0


def test_finalize_episode_record_uses_episode_elapsed_time() -> None:
    record = _finalize_episode_record(
        {
            "return": 1.0,
            "length": 12,
            "start_sim_time_seconds": 100.0,
            "max_out_of_bounds_seconds": 0.0,
        },
        {
            "sim_time_seconds": 145.5,
            "ego_role": "fighter1",
            "enemy_role": "fighter2",
            "aircraft_by_role": {
                "fighter1": {"destroyed": False},
                "fighter2": {"destroyed": True},
            },
        },
        terminated=True,
        truncated=False,
    )

    assert record.episode_survival_seconds == 45.5
    assert record.termination_reason == "enemy_destroyed"


def test_reward_diagnostics_by_outcome_uses_episode_component_sums() -> None:
    def payload(total: float, hit_bonus: float, sample_count: int) -> dict[str, Any]:
        accumulator = RewardDiagnosticsAccumulator()
        for _ in range(sample_count):
            accumulator.add(
                {
                    "total": total / sample_count,
                    "hit_enemy_bonus": hit_bonus / sample_count,
                }
            )
        return accumulator.raw_payload()

    records = [
        EpisodeRecord(
            episode_return=10.0,
            episode_length=2,
            episode_survival_seconds=1.0,
            self_destroyed=False,
            enemy_destroyed=True,
            terminated=True,
            truncated=False,
            max_out_of_bounds_seconds=0.0,
            termination_reason="enemy_destroyed",
            reward_diagnostics=payload(10.0, 4.0, 2),
        ),
        EpisodeRecord(
            episode_return=-6.0,
            episode_length=3,
            episode_survival_seconds=1.5,
            self_destroyed=True,
            enemy_destroyed=False,
            terminated=True,
            truncated=False,
            max_out_of_bounds_seconds=0.0,
            termination_reason="self_destroyed",
            reward_diagnostics=payload(-6.0, 0.0, 3),
        ),
    ]

    summary = _reward_diagnostics_by_outcome(records)

    assert summary["all"]["episode_count"] == 2
    assert summary["all"]["transition_count"] == 5
    assert summary["all"]["components"]["total"]["mean_episode_sum"] == 2.0
    assert summary["enemy_destroyed"]["components"]["hit_enemy_bonus"][
        "mean_episode_sum"
    ] == 4.0
    assert summary["self_destroyed"]["components"]["total"]["mean_step"] == -2.0


def test_evaluation_limit_reason_prefers_step_limit() -> None:
    assert (
        _evaluation_limit_reason(
            episode_steps=120,
            elapsed_sim_seconds=2.0,
            max_steps=120,
            max_seconds=1.0,
        )
        == "eval_step_limit"
    )
    assert (
        _evaluation_limit_reason(
            episode_steps=10,
            elapsed_sim_seconds=120.0,
            max_steps=0,
            max_seconds=120.0,
        )
        == "eval_time_limit"
    )
    assert (
        _evaluation_limit_reason(
            episode_steps=10,
            elapsed_sim_seconds=1.0,
            max_steps=0,
            max_seconds=0.0,
        )
        is None
    )


def test_best_checkpoint_requires_fixed_eval_result() -> None:
    assert not _is_new_best_eval(None, best_eval_return=-100.0)
    metrics = PpoEvalMetrics(
        mean_return=5.0,
        mean_length=10.0,
        mean_survival_seconds=1.0,
        self_destroy_rate=0.0,
        enemy_destroy_rate=1.0,
        mean_max_out_of_bounds_seconds=0.0,
        episode_count=1,
    )
    assert _is_new_best_eval(metrics, best_eval_return=4.0)
    assert not _is_new_best_eval(metrics, best_eval_return=5.0)


def test_target_kl_gate_can_be_disabled() -> None:
    assert _target_kl_reached(approximate_kl=0.03, target_kl=0.03)
    assert not _target_kl_reached(approximate_kl=0.029, target_kl=0.03)
    assert not _target_kl_reached(approximate_kl=100.0, target_kl=0.0)


def test_policy_reference_diagnostics_are_zero_for_identical_actors() -> None:
    model = StatelessHybridActorCritic(obs_dim=OBS_DIM, hidden_dim=64, num_layers=2)
    reference = StatelessHybridActorCritic(obs_dim=OBS_DIM, hidden_dim=64, num_layers=2)
    reference.load_state_dict(model.state_dict(), strict=True)

    diagnostics = _policy_reference_diagnostics(
        model=model,
        reference_model=reference,
        observations=torch.zeros((8, OBS_DIM), dtype=torch.float32),
        sample_size=4,
    )

    assert diagnostics["sample_count"] == 4
    assert diagnostics["continuous_mean_abs_drift"] == 0.0
    assert diagnostics["binary_probability_mean_abs_drift"] == 0.0
    assert diagnostics["binary_reference_to_current_kl"] == 0.0


class _FakeSerialEnv:
    def __init__(self, env_id: int) -> None:
        self.env_id = env_id
        self.shutdown_calls = 0
        self._state = {"env_id": env_id, "tick": 0}

    def reset(
        self,
        *,
        seed: int | None = None,
        scene_name: str | None = None,
        scene_path: str | None = None,
        opponent_mode: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        self._state = {
            "env_id": self.env_id,
            "seed": seed,
            "scene_name": scene_name,
            "scene_path": scene_path,
            "opponent_mode": opponent_mode,
            "tick": 0,
        }
        return torch.tensor([float(self.env_id)], dtype=torch.float32).numpy(), {"seed": seed}

    def step_arrays(
        self,
        continuous,
        binary,
        *,
        binary_threshold: float = 0.5,
        opponent_continuous=None,
        opponent_binary=None,
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, object]]:
        self._state = {
            "env_id": self.env_id,
            "tick": 1,
            "binary_threshold": binary_threshold,
            "has_opponent_action": opponent_continuous is not None and opponent_binary is not None,
        }
        return torch.tensor([float(self.env_id + 10)], dtype=torch.float32).numpy(), 2.0, False, False, {"ok": True}

    def latest_state(self) -> dict[str, object]:
        return dict(self._state)

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def test_serial_train_env_helpers_wrap_reset_step_and_close() -> None:
    envs = [_FakeSerialEnv(0), _FakeSerialEnv(1)]
    reset_results = _reset_train_envs(
        envs,
        [
            ResetRequest(seed=11, scene_name="scene_a", opponent_mode="external", include_state=True),
            ResetRequest(seed=22, scene_name="scene_b", opponent_mode="built_in_ai", include_state=False),
        ],
    )
    assert [result.info["seed"] for result in reset_results] == [11, 22]
    assert reset_results[0].state["scene_name"] == "scene_a"
    assert reset_results[1].state is None

    step_results = _step_train_envs(
        envs,
        [
            StepRequest(
                continuous=torch.zeros((4,), dtype=torch.float32).numpy(),
                binary=torch.zeros((3,), dtype=torch.float32).numpy(),
                include_state=False,
            ),
            StepRequest(
                continuous=torch.ones((4,), dtype=torch.float32).numpy(),
                binary=torch.ones((3,), dtype=torch.float32).numpy(),
                opponent_continuous=torch.ones((4,), dtype=torch.float32).numpy(),
                opponent_binary=torch.ones((3,), dtype=torch.float32).numpy(),
                include_state=True,
            ),
        ],
    )
    assert [result.reward for result in step_results] == [2.0, 2.0]
    assert step_results[0].state is None
    assert step_results[1].state["has_opponent_action"] is True

    reset_result = _reset_train_env_at(
        envs,
        index=1,
        request=ResetRequest(seed=33, scene_name="scene_c", opponent_mode="external", include_state=True),
    )
    assert reset_result.state["seed"] == 33
    assert reset_result.state["scene_name"] == "scene_c"

    _close_train_env_backend(envs)
    assert envs[0].shutdown_calls == 1
    assert envs[1].shutdown_calls == 1


def test_sampled_opponent_requires_state_only_for_checkpoint_policy() -> None:
    assert not _sampled_opponent_requires_state(None)
    assert not _sampled_opponent_requires_state(
        SampledOpponent(label="neutral", env_mode="external", runtime_kind="neutral")
    )
    assert not _sampled_opponent_requires_state(
        SampledOpponent(label="builtin", env_mode="built_in_ai", runtime_kind="built_in")
    )
    assert _sampled_opponent_requires_state(
        SampledOpponent(label="ckpt", env_mode="external", runtime_kind="checkpoint", checkpoint_path="a.pt")
    )
    self_play = SampledOpponent(
        label="current",
        env_mode="model",
        runtime_kind="self_play",
    )
    assert not _sampled_opponent_requires_state(self_play)
    assert _episode_uses_self_play(
        configured_opponent_mode="built_in_ai_imperfect",
        sampled_opponent=self_play,
    )
    assert not _episode_uses_self_play(
        configured_opponent_mode="self_play",
        sampled_opponent=SampledOpponent(
            label="builtin",
            env_mode="built_in_ai_imperfect",
            runtime_kind="built_in",
        ),
    )


def test_episode_sampling_diagnostics_reports_opponent_and_scene_mix() -> None:
    diagnostics = _episode_sampling_diagnostics(
        [
            EpisodeRecord(
                episode_return=0.0,
                episode_length=1,
                episode_survival_seconds=1.0,
                self_destroyed=False,
                enemy_destroyed=False,
                terminated=False,
                truncated=True,
                max_out_of_bounds_seconds=0.0,
                opponent_label="current",
                opponent_kind="self_play",
                scene_label="collision",
            ),
            EpisodeRecord(
                episode_return=0.0,
                episode_length=1,
                episode_survival_seconds=1.0,
                self_destroyed=False,
                enemy_destroyed=False,
                terminated=False,
                truncated=True,
                max_out_of_bounds_seconds=0.0,
                opponent_label="rule",
                opponent_kind="built_in",
                scene_label="head_on",
            ),
        ]
    )
    assert diagnostics["self_play_episode_fraction"] == 0.5
    assert diagnostics["opponent_kind_counts"] == {
        "self_play": 1,
        "built_in": 1,
    }
    assert diagnostics["scene_label_counts"] == {
        "collision": 1,
        "head_on": 1,
    }


def test_validate_worker_reward_support_rejects_unsupported_configurations() -> None:
    try:
        _validate_worker_reward_support(
            worker_reward_mode="worker",
            env_backend="serial",
            sampled_opponent=None,
        )
    except ValueError as exc:
        assert "--env-backend subproc" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    try:
        _validate_worker_reward_support(
            worker_reward_mode="worker",
            env_backend="subproc",
            sampled_opponent=SampledOpponent(
                label="ckpt",
                env_mode="external",
                runtime_kind="checkpoint",
                checkpoint_path="a.pt",
            ),
        )
    except ValueError as exc:
        assert "checkpoint opponents" in str(exc)
    else:
        raise AssertionError("expected ValueError")
