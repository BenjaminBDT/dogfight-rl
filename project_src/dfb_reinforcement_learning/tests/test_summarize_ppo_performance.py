from __future__ import annotations

import pytest

from dfb_reinforcement_learning.tools.summarize_ppo_performance import (
    TIMING_FIELDS,
    percentile,
    summarize_records,
)


def _record(update: int, *, sps: float, total: float, eval_seconds: float = 0.0) -> dict[str, float]:
    record = {
        "update": float(update),
        "rollout_steps_per_second": sps,
        "eval_seconds": eval_seconds,
    }
    record.update({field: total for field in TIMING_FIELDS})
    record["eval_seconds"] = eval_seconds
    return record


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.50) == pytest.approx(2.5)
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


def test_summary_excludes_eval_and_discards_warmup() -> None:
    summary = summarize_records(
        [
            _record(0, sps=100.0, total=10.0),
            _record(1, sps=200.0, total=8.0),
            _record(2, sps=1.0, total=100.0, eval_seconds=92.0),
            _record(3, sps=300.0, total=6.0),
        ],
        warmup_updates=1,
    )

    assert summary["sample_count"] == 2
    assert summary["first_update"] == 1
    assert summary["last_update"] == 3
    assert summary["excluded_eval_count"] == 1
    assert summary["metrics"]["rollout_steps_per_second"]["p50"] == pytest.approx(250.0)
    assert summary["metrics"]["update_total_seconds"]["p95"] == pytest.approx(7.9)


def test_summary_rejects_empty_selection() -> None:
    with pytest.raises(ValueError, match="no PPO updates"):
        summarize_records([_record(0, sps=100.0, total=10.0)], warmup_updates=1)


def test_summary_reports_fields_missing_from_legacy_records() -> None:
    record = _record(0, sps=100.0, total=10.0)
    del record["diagnostics_seconds"]

    summary = summarize_records([record], warmup_updates=0)

    assert summary["omitted_fields"] == ["diagnostics_seconds"]
    assert "diagnostics_seconds" not in summary["metrics"]
