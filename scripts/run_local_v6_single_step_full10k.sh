#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p runs/dfb_state_estimation/jobs

echo "building dfb_tool_dataset once..."
cargo +stable build --bin dfb_tool_dataset

BIN="target/debug/dfb_tool_dataset"

run_one() {
  local name="$1"
  local source_root="runs/dfb_state_estimation/synthetic_visual_dataset_fix_v6_20k_target_identity/target_fighter${name#fighter}/clean"
  local output_dir="runs/dfb_state_estimation/synthetic_single_step_v6_target_${name}_full10k"
  local log_path="runs/dfb_state_estimation/jobs/single_step_local_${name}_full10k.log"

  if [[ ! -d "$source_root" ]]; then
    echo "missing source root: $source_root" >&2
    exit 1
  fi

  rm -rf "$output_dir"
  rm -f "$log_path"

  echo "running synthetic-single-step for ${name}..."
  "$BIN" synthetic-single-step \
    --source-root "$source_root" \
    --output-dir "$output_dir" \
    --max-samples 10000 \
    --force \
    > "$log_path" 2>&1

  echo "finished ${name}: $output_dir"
}

run_one fighter2
run_one fighter1

echo "all done"
