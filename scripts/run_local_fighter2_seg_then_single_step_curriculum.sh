#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

JOB_DIR="runs/dfb_state_estimation/jobs"
GENERATED_CONFIG_DIR="$JOB_DIR/generated_configs"
mkdir -p "$JOB_DIR" "$GENERATED_CONFIG_DIR"

LOG_PATH="$JOB_DIR/local_fighter2_seg_then_single_step_curriculum.log"
rm -f "$LOG_PATH"
exec > >(tee -a "$LOG_PATH") 2>&1

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
export PYTHONPATH="$ROOT_DIR/project_src"

SEG_RESUME_CKPT="$ROOT_DIR/runs/dfb_state_estimation/checkpoints/fighter2_v6_cloud_b64_continue500_best_target_iou.pt"

resolve_output_root() {
  local config_path="$1"
  "$PYTHON_BIN" - <<'PY' "$config_path"
import json, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(config["output_root"])
PY
}

inject_init_vision_from() {
  local template_config="$1"
  local output_config="$2"
  local init_ckpt="$3"
  "$PYTHON_BIN" - <<'PY' "$template_config" "$output_config" "$init_ckpt"
import json, sys
from pathlib import Path
template_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
init_ckpt = sys.argv[3]
payload = json.loads(template_path.read_text(encoding="utf-8"))
payload["init_vision_from"] = init_ckpt
output_path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
}

extract_best_checkpoint() {
  local best_json="$1"
  "$PYTHON_BIN" - <<'PY' "$best_json"
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["checkpoint"])
PY
}

run_train() {
  local config_path="$1"
  local label="$2"
  local resume_path="${3:-}"
  local output_root
  output_root="$(resolve_output_root "$config_path")"

  echo
  echo "===== ${label} ====="
  echo "config: ${config_path}"
  echo "output_root: ${output_root}"
  rm -rf "$output_root"

  if [[ -n "$resume_path" ]]; then
    echo "resume: ${resume_path}"
    "$PYTHON_BIN" -m dfb_state_estimation.train.train \
      --config "$config_path" \
      --resume "$resume_path"
  else
    "$PYTHON_BIN" -m dfb_state_estimation.train.train \
      --config "$config_path"
  fi
}

run_if_missing() {
  local done_marker="$1"
  local config_path="$2"
  local label="$3"
  local resume_path="${4:-}"
  if [[ -f "$done_marker" ]]; then
    echo
    echo "===== ${label} ====="
    echo "skip: found done marker ${done_marker}"
    return
  fi
  run_train "$config_path" "$label" "$resume_path"
}

ensure_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
}

SEG_CONFIG="config/dfb_state_estimation/train/segmentation_only_synthetic_target_fighter2_single_view_v6_full10k_binary_ce_rebalanced_resnet18_fpn_target_patch64_local_to10000_eval100.json"
S032_TEMPLATE="config/dfb_state_estimation/train/single_step_synthetic_target_fighter2_v6_curriculum_overfit032_visual_only_resnet18.json"
S064_CONFIG="config/dfb_state_estimation/train/single_step_synthetic_target_fighter2_v6_curriculum_overfit064_visual_only_resnet18.json"
S128_CONFIG="config/dfb_state_estimation/train/single_step_synthetic_target_fighter2_v6_curriculum_overfit128_visual_only_resnet18.json"
S256_CONFIG="config/dfb_state_estimation/train/single_step_synthetic_target_fighter2_v6_curriculum_overfit256_visual_only_resnet18.json"
S512_CONFIG="config/dfb_state_estimation/train/single_step_synthetic_target_fighter2_v6_curriculum_overfit512_visual_only_resnet18.json"
SFULL_CONFIG="config/dfb_state_estimation/train/single_step_synthetic_target_fighter2_v6_curriculum_full10k_visual_only_resnet18.json"

ensure_file "$SEG_RESUME_CKPT"

SEG_OUTPUT_ROOT="$(resolve_output_root "$SEG_CONFIG")"
SEG_FINAL_CKPT="$SEG_OUTPUT_ROOT/checkpoints/step_010000.pt"
run_if_missing "$SEG_FINAL_CKPT" "$SEG_CONFIG" "segmentation_full10k_to_10000" "$SEG_RESUME_CKPT"

SEG_BEST_JSON="$SEG_OUTPUT_ROOT/eval/best_target_iou.json"
ensure_file "$SEG_BEST_JSON"
SEG_BEST_CKPT="$(extract_best_checkpoint "$SEG_BEST_JSON")"
ensure_file "$SEG_BEST_CKPT"

S032_RUNTIME_CONFIG="$GENERATED_CONFIG_DIR/single_step_synthetic_target_fighter2_v6_curriculum_overfit032_visual_only_resnet18.runtime.json"
inject_init_vision_from "$S032_TEMPLATE" "$S032_RUNTIME_CONFIG" "$SEG_BEST_CKPT"
S032_OUTPUT_ROOT="$ROOT_DIR/runs/dfb_state_estimation/train/single_step_synthetic_target_fighter2_v6_curriculum_overfit032_visual_only_resnet18"
run_if_missing "$S032_OUTPUT_ROOT/checkpoints/step_000400.pt" "$S032_RUNTIME_CONFIG" "single_step_overfit032_from_seg_best"

CKPT032="$S032_OUTPUT_ROOT/checkpoints/step_000400.pt"
ensure_file "$CKPT032"
S064_OUTPUT_ROOT="$ROOT_DIR/runs/dfb_state_estimation/train/single_step_synthetic_target_fighter2_v6_curriculum_overfit064_visual_only_resnet18"
run_if_missing "$S064_OUTPUT_ROOT/checkpoints/step_000600.pt" "$S064_CONFIG" "single_step_overfit064" "$CKPT032"

CKPT064="$S064_OUTPUT_ROOT/checkpoints/step_000600.pt"
ensure_file "$CKPT064"
S128_OUTPUT_ROOT="$ROOT_DIR/runs/dfb_state_estimation/train/single_step_synthetic_target_fighter2_v6_curriculum_overfit128_visual_only_resnet18"
run_if_missing "$S128_OUTPUT_ROOT/checkpoints/step_000800.pt" "$S128_CONFIG" "single_step_overfit128" "$CKPT064"

CKPT128="$S128_OUTPUT_ROOT/checkpoints/step_000800.pt"
ensure_file "$CKPT128"
S256_OUTPUT_ROOT="$ROOT_DIR/runs/dfb_state_estimation/train/single_step_synthetic_target_fighter2_v6_curriculum_overfit256_visual_only_resnet18"
run_if_missing "$S256_OUTPUT_ROOT/checkpoints/step_001000.pt" "$S256_CONFIG" "single_step_overfit256" "$CKPT128"

CKPT256="$S256_OUTPUT_ROOT/checkpoints/step_001000.pt"
ensure_file "$CKPT256"
S512_OUTPUT_ROOT="$ROOT_DIR/runs/dfb_state_estimation/train/single_step_synthetic_target_fighter2_v6_curriculum_overfit512_visual_only_resnet18"
run_if_missing "$S512_OUTPUT_ROOT/checkpoints/step_001200.pt" "$S512_CONFIG" "single_step_overfit512" "$CKPT256"

CKPT512="$S512_OUTPUT_ROOT/checkpoints/step_001200.pt"
ensure_file "$CKPT512"
SFULL_OUTPUT_ROOT="$ROOT_DIR/runs/dfb_state_estimation/train/single_step_synthetic_target_fighter2_v6_curriculum_full10k_visual_only_resnet18"
run_if_missing "$SFULL_OUTPUT_ROOT/checkpoints/step_004000.pt" "$SFULL_CONFIG" "single_step_full10k" "$CKPT512"

echo
echo "pipeline complete"
