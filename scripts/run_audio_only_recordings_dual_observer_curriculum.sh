#!/usr/bin/env bash
set -euo pipefail

ROOT="/run/media/ayano/SharedProjects/general_projects/dfb"
PYTHON="${ROOT}/.venv/bin/python"

OVERFIT_CFG="config/dfb_state_estimation/train/audio_only_recordings_dual_observer_min20s_overfit2048.json"
FULL_CFG="config/dfb_state_estimation/train/audio_only_recordings_dual_observer_min20s_full.json"

OVERFIT_BOOTSTRAP="runs/dfb_state_estimation/train/audio_only_recordings_dual_observer_min20s_overfit256/checkpoints/latest.pt"
OVERFIT_RESUME="runs/dfb_state_estimation/train/audio_only_recordings_dual_observer_min20s_overfit2048/checkpoints/latest.pt"
FULL_RESUME="runs/dfb_state_estimation/train/audio_only_recordings_dual_observer_min20s_full/checkpoints/latest.pt"

cd "${ROOT}"

if [[ -f "${OVERFIT_RESUME}" ]]; then
  echo "resume audio overfit2048 from ${OVERFIT_RESUME}"
  PYTHONPATH=project_src "${PYTHON}" -m dfb_state_estimation.train.train \
    --config "${OVERFIT_CFG}" \
    --resume "${OVERFIT_RESUME}"
elif [[ -f "${OVERFIT_BOOTSTRAP}" ]]; then
  echo "bootstrap audio overfit2048 from ${OVERFIT_BOOTSTRAP}"
  PYTHONPATH=project_src "${PYTHON}" -m dfb_state_estimation.train.train \
    --config "${OVERFIT_CFG}" \
    --resume "${OVERFIT_BOOTSTRAP}"
else
  echo "missing bootstrap checkpoint: ${OVERFIT_BOOTSTRAP}" >&2
  exit 1
fi

if [[ -f "${FULL_RESUME}" ]]; then
  echo "resume audio full from ${FULL_RESUME}"
  PYTHONPATH=project_src "${PYTHON}" -m dfb_state_estimation.train.train \
    --config "${FULL_CFG}" \
    --resume "${FULL_RESUME}"
else
  echo "bootstrap audio full from overfit2048 latest"
  PYTHONPATH=project_src "${PYTHON}" -m dfb_state_estimation.train.train \
    --config "${FULL_CFG}" \
    --resume "${OVERFIT_RESUME}"
fi
