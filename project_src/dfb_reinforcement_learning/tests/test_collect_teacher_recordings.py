import json

from dfb_reinforcement_learning.scenes import MaterializedScene, PreparedScenePool
from dfb_reinforcement_learning.tools.collect_teacher_recordings import (
    _advance_until_running,
    build_collection_jobs,
    summarize_collection_jobs,
)


def _prepared_pool() -> PreparedScenePool:
    return PreparedScenePool(
        train_tactical_scenes=[
            MaterializedScene(label="tactical_a", weight=3.0, scene_name="tactical_a"),
            MaterializedScene(label="tactical_b", weight=1.0, scene_name="tactical_b"),
        ],
        train_recovery_scenes=[
            MaterializedScene(label="recovery_a", weight=1.0, scene_name="recovery_a"),
        ],
        train_tactical_ratio=0.75,
        train_recovery_ratio=0.25,
        eval_scenes=[],
        rng_seed=17,
    )


def test_build_collection_jobs_honors_category_ratio_and_is_reproducible() -> None:
    prepared = _prepared_pool()

    jobs = build_collection_jobs(prepared, episode_count=20, seed=100)
    repeated = build_collection_jobs(prepared, episode_count=20, seed=100)

    assert jobs == repeated
    assert [job.environment_seed for job in jobs] == list(range(100, 120))
    assert summarize_collection_jobs(jobs)["category_counts"] == {
        "recovery": 5,
        "tactical": 15,
    }
    assert sum(job.scene.label == "tactical_a" for job in jobs) > sum(
        job.scene.label == "tactical_b" for job in jobs
    )


def test_build_collection_jobs_uses_available_category_when_other_is_empty() -> None:
    prepared = PreparedScenePool(
        train_tactical_scenes=[],
        train_recovery_scenes=[
            MaterializedScene(label="recovery", weight=1.0, scene_name="recovery"),
        ],
        train_tactical_ratio=1.0,
        train_recovery_ratio=0.0,
        eval_scenes=[],
        rng_seed=17,
    )

    jobs = build_collection_jobs(prepared, episode_count=3, seed=12)

    assert [job.category for job in jobs] == ["recovery", "recovery", "recovery"]


def test_build_collection_jobs_exhaustive_uses_every_scene_once() -> None:
    prepared = _prepared_pool()

    jobs = build_collection_jobs(
        prepared,
        episode_count=3,
        seed=100,
        sampling_strategy="exhaustive",
    )

    assert {job.scene.label for job in jobs} == {
        "tactical_a",
        "tactical_b",
        "recovery_a",
    }
    assert all(job.repetition == 0 for job in jobs)


def test_build_collection_jobs_exhaustive_rejects_count_mismatch() -> None:
    prepared = _prepared_pool()

    try:
        build_collection_jobs(
            prepared,
            episode_count=2,
            seed=100,
            sampling_strategy="exhaustive",
        )
    except ValueError as exc:
        assert "materialized scene count (3)" in str(exc)
    else:
        raise AssertionError("expected exhaustive scene count mismatch")


def test_advance_until_running_flushes_stale_terminal_state() -> None:
    class FakeEnvironment:
        def __init__(self) -> None:
            self.steps = 0

        def episode_status_json(self) -> str:
            running = self.steps > 0
            return json.dumps(
                {
                    "match_phase": "Running" if running else "Finished",
                    "terminated": not running,
                    "truncated": False,
                }
            )

        def step_json(self, _action: object) -> None:
            self.steps += 1

    environment = FakeEnvironment()

    status = _advance_until_running(environment, object())

    assert status["match_phase"] == "Running"
    assert environment.steps == 1
