#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from dfb_state_estimation.datasets import WindowDataset
from dfb_state_estimation.models.audio import SingleStepAudioModule
from dfb_state_estimation.models.evidence import SingleStepEvidenceModule
from dfb_state_estimation.models.vision import SingleStepVisionModule
from dfb_state_estimation.train.train import (
    _build_temporal_inputs_from_window_batch,
    _collate_belief_targets,
    _collate_temporal_targets,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a temporal teacher/cache dataset by freezing single-step front-ends, "
            "running them over authoritative window samples, and writing cached modality "
            "inputs plus temporal/belief supervision targets."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("runs/dfb_state_estimation/authoritative_multimodal_recording_pack_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/dfb_state_estimation/temporal_teacher_cache_v1"),
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--min-stride-steps", type=int, default=1)
    parser.add_argument("--max-stride-steps", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--limit-windows", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vision-checkpoint", type=Path, default=None)
    parser.add_argument("--audio-checkpoint", type=Path, default=None)
    parser.add_argument("--evidence-checkpoint", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _load_module_state(module: torch.nn.Module, checkpoint_path: Path, module_name: str) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    module_states = payload.get("modules") if isinstance(payload, dict) else None
    if isinstance(module_states, dict) and module_name in module_states:
        state_dict = module_states[module_name]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(f"unsupported checkpoint payload: {checkpoint_path}")
    incompatible = module.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"warning: partially loaded {module_name} from {checkpoint_path} "
            f"(missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)})"
        )


def _clone_detached_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_detached_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_detached_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_detached_cpu(item) for item in value)
    return value


def _belief_context_batch(samples: list[Any], *, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "context_relative_position": torch.tensor(
            [sample.core["gt_relative_position"] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        "context_relative_orientation": torch.tensor(
            [sample.core["gt_relative_orientation"] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        "context_position_confidence": torch.tensor(
            [sample.rule_targets["target_pos_conf"] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        "context_orientation_confidence": torch.tensor(
            [sample.rule_targets["target_ori_conf"] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        "linear_velocity": torch.tensor(
            [sample.core["gt_linear_velocity"] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        "angular_velocity": torch.tensor(
            [sample.core["gt_angular_velocity"] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        "dt_to_prev": torch.tensor(
            [sample.dt_to_prev for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        "time_from_now": torch.tensor(
            [sample.time_from_now for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
    }


def _sample_payloads(
    samples: list[Any],
    *,
    modality_inputs,
    temporal_targets,
    belief_context_batch,
    belief_targets,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        ref = sample.ref
        payloads.append(
            {
                "ref": {
                    "episode_id": ref.episode_id,
                    "observed_role": ref.observed_role,
                    "window_end_model_step_index": ref.window_end_model_step_index,
                    "window_end_simulation_step_index": ref.window_end_simulation_step_index,
                    "window_length_steps": ref.window_length_steps,
                    "max_steps": ref.max_steps,
                },
                "step_refs": [
                    {
                        "episode_id": step_ref.episode_id,
                        "observed_role": step_ref.observed_role,
                        "chunk_id": step_ref.chunk_id,
                        "chunk_index": step_ref.chunk_index,
                        "chunk_step_offset": step_ref.chunk_step_offset,
                        "global_model_step_index": step_ref.global_model_step_index,
                        "simulation_step_index": step_ref.simulation_step_index,
                    }
                    for step_ref in sample.step_refs
                ],
                "modality_inputs": {
                    field_name: _clone_detached_cpu(getattr(modality_inputs, field_name)[index])
                    for field_name in modality_inputs.__dataclass_fields__
                },
                "temporal_targets": {
                    field_name: _clone_detached_cpu(getattr(temporal_targets, field_name)[index])
                    for field_name in temporal_targets.__dataclass_fields__
                },
                "belief_context": {
                    key: _clone_detached_cpu(value[index]) for key, value in belief_context_batch.items()
                },
                "belief_targets": {
                    field_name: _clone_detached_cpu(getattr(belief_targets, field_name)[index])
                    for field_name in belief_targets.__dataclass_fields__
                },
            }
        )
    return payloads


def main() -> None:
    args = _parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise SystemExit(f"output dir already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = WindowDataset(
        dataset_root,
        max_steps=args.max_steps,
        min_stride_steps=args.min_stride_steps,
        max_stride_steps=args.max_stride_steps,
        random_seed=args.random_seed,
    )
    total_windows = len(dataset) if args.limit_windows <= 0 else min(len(dataset), args.limit_windows)
    manifest = {
        "source_dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "max_steps": args.max_steps,
        "min_stride_steps": args.min_stride_steps,
        "max_stride_steps": args.max_stride_steps,
        "random_seed": args.random_seed,
        "batch_size": args.batch_size,
        "shard_size": args.shard_size,
        "limit_windows": args.limit_windows,
        "entry_count": total_windows,
        "vision_checkpoint": str(args.vision_checkpoint.resolve()) if args.vision_checkpoint else None,
        "audio_checkpoint": str(args.audio_checkpoint.resolve()) if args.audio_checkpoint else None,
        "evidence_checkpoint": str(args.evidence_checkpoint.resolve()) if args.evidence_checkpoint else None,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    device = _resolve_device(args.device)
    vision = SingleStepVisionModule().to(device).eval()
    audio = SingleStepAudioModule().to(device).eval()
    evidence = SingleStepEvidenceModule().to(device).eval()
    if args.vision_checkpoint is not None:
        _load_module_state(vision, args.vision_checkpoint.resolve(), "vision")
    if args.audio_checkpoint is not None:
        _load_module_state(audio, args.audio_checkpoint.resolve(), "audio")
    if args.evidence_checkpoint is not None:
        _load_module_state(evidence, args.evidence_checkpoint.resolve(), "evidence")

    entries_path = output_dir / "entries.jsonl"
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_payloads: list[dict[str, Any]] = []
    shard_index = 0
    written_entries = 0

    def flush_shard() -> None:
        nonlocal shard_payloads, shard_index
        if not shard_payloads:
            return
        shard_name = f"shard_{shard_index:06d}.pt"
        torch.save({"samples": shard_payloads}, shard_dir / shard_name)
        shard_payloads = []
        shard_index += 1

    with torch.no_grad(), entries_path.open("w", encoding="utf-8") as entries_handle:
        batch: list[Any] = []
        for index in range(total_windows):
            batch.append(dataset[index])
            if len(batch) < args.batch_size and index + 1 < total_windows:
                continue
            modality_inputs = _build_temporal_inputs_from_window_batch(
                batch,
                vision=vision,
                audio=audio,
                evidence=evidence,
                device=device,
            )
            temporal_targets = _collate_temporal_targets(batch, device)
            belief_targets = _collate_belief_targets(batch, device)
            belief_context = _belief_context_batch(batch, device=device)
            payloads = _sample_payloads(
                batch,
                modality_inputs=modality_inputs,
                temporal_targets=temporal_targets,
                belief_context_batch=belief_context,
                belief_targets=belief_targets,
            )
            for payload in payloads:
                if len(shard_payloads) >= args.shard_size:
                    flush_shard()
                current_shard_name = f"shard_{shard_index:06d}.pt"
                current_shard_offset = len(shard_payloads)
                shard_payloads.append(payload)
                ref = payload["ref"]
                entries_handle.write(
                    json.dumps(
                        {
                            "episode_id": ref["episode_id"],
                            "observed_role": ref["observed_role"],
                            "window_end_model_step_index": ref["window_end_model_step_index"],
                            "window_end_simulation_step_index": ref["window_end_simulation_step_index"],
                            "window_length_steps": ref["window_length_steps"],
                            "max_steps": ref["max_steps"],
                            "shard_path": f"shards/{current_shard_name}",
                            "shard_index": current_shard_offset,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written_entries += 1
            print(f"cached windows: {written_entries}/{total_windows}")
            batch = []
        flush_shard()

    manifest["entry_count"] = written_entries
    manifest["shard_count"] = shard_index
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote temporal teacher cache: {output_dir}")


if __name__ == "__main__":
    main()
