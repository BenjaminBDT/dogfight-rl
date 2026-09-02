#!/usr/bin/env bash
set -euo pipefail

ROOT="/run/media/ayano/SharedProjects/general_projects/dfb"
PYTHON="${ROOT}/.venv/bin/python"
CFG="config/dfb_state_estimation/train/single_step_synthetic_target_fighter2_v6_curriculum_full10k_adaptive_seg_visual_only_resnet18.json"
CFG_OVERNIGHT="config/dfb_state_estimation/train/single_step_synthetic_target_fighter2_v6_curriculum_full10k_adaptive_seg_visual_only_resnet18_overnight.json"
RESUME="runs/dfb_state_estimation/train/single_step_synthetic_target_fighter2_v6_curriculum_full10k_adaptive_seg_visual_only_resnet18/checkpoints/latest.pt"

cd "${ROOT}"

if [[ -f "${RESUME}" ]]; then
  echo "resume single-step full10k from ${RESUME}"
  PYTHONPATH=project_src "${PYTHON}" -m dfb_state_estimation.train.train \
    --config "${CFG_OVERNIGHT}" \
    --resume "${RESUME}"
else
  echo "start single-step full10k from config init"
  PYTHONPATH=project_src "${PYTHON}" -m dfb_state_estimation.train.train \
    --config "${CFG_OVERNIGHT}"
fi
