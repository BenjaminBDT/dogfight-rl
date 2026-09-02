#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

python_bin="${DFB_PYTHON_BIN:-${repo_root}/.venv/bin/python}"
maturin_bin="${DFB_MATURIN_BIN:-${repo_root}/.venv/bin/maturin}"
dfb_cargo_target_dir="${DFB_CARGO_TARGET_DIR:-${repo_root}/.cache/dfb_cargo_target}"
dfb_cargo_build_jobs="${DFB_CARGO_BUILD_JOBS:-1}"
dfb_virtual_env="$(cd -- "$(dirname -- "${python_bin}")/.." && pwd)"

if [[ ! -x "${python_bin}" ]]; then
    echo "python not found or not executable: ${python_bin}" >&2
    exit 1
fi

if [[ ! -x "${maturin_bin}" ]]; then
    echo "maturin not found or not executable: ${maturin_bin}" >&2
    exit 1
fi

mkdir -p "${dfb_cargo_target_dir}"

echo "repo_root=${repo_root}"
echo "python=${python_bin}"
echo "maturin=${maturin_bin}"
echo "CARGO_TARGET_DIR=${dfb_cargo_target_dir}"
echo "CARGO_BUILD_JOBS=${dfb_cargo_build_jobs}"
echo "profile=release"

cd "${repo_root}"

CARGO_TARGET_DIR="${dfb_cargo_target_dir}" \
CARGO_BUILD_JOBS="${dfb_cargo_build_jobs}" \
VIRTUAL_ENV="${dfb_virtual_env}" \
"${maturin_bin}" develop \
    --release \
    --manifest-path project_src/dfb_game/Cargo.toml \
    --features python-bindings
