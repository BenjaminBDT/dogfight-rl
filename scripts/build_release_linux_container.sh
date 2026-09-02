#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_TAG="${DFB_RELEASE_IMAGE:-dfb-release-ubuntu2204:latest}"
DOCKERFILE_PATH="${DFB_RELEASE_DOCKERFILE:-${REPO_ROOT}/Dockerfile}"
CONTAINER_WORKDIR="/workspace"
LINUX_TARGET="x86_64-unknown-linux-gnu"
PACKAGE_BASENAME_PREFIX="dfb"
CONTAINER_CARGO_TARGET_DIR="/home/builder/release-target"
CARGO_BUILD_JOBS_DEFAULT="1"

BINS=(
    dfb_launcher
    dfb_client_gameplay
    dfb_client_observer
    dfb_server
    dfb_tool_keybindings
    dfb_tool_dataset
)

usage() {
    cat <<'EOF'
Usage:
  scripts/build_release_linux_container.sh

Environment:
  DFB_RELEASE_DOCKERFILE   Path to the Ubuntu 22.04 release Dockerfile. Defaults to ./Dockerfile.
  DFB_RELEASE_IMAGE        Docker image tag. Defaults to dfb-release-ubuntu2204:latest.
  DFB_RELEASE_CACHE_DIR    Host cache directory. Defaults to /tmp/dfb-release-cache-<uid>.
  DFB_SKIP_DOCKER_BUILD    Set to 1 to reuse an existing image.
  DFB_NO_PROXY_ARGS        Set to 1 to avoid forwarding proxy build args/env.
  DFB_DOCKER_NETWORK       Optional docker network mode, for example host.
  DFB_RELEASE_CARGO_BUILD_JOBS
                           Cargo parallel job count inside container. Defaults to 1.
  DFB_RELEASE_CARGO_TARGET_DIR
                           Container cargo target dir. Defaults to /home/builder/release-target.

Outputs:
  dist/linux/staging/dfb-x86_64-unknown-linux-gnu-<version>/
  dist/linux/package/dfb-x86_64-unknown-linux-gnu-<version>.zip
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "${DFB_RELEASE_INSIDE_CONTAINER:-0}" != "1" ]]; then
    if ! command -v docker >/dev/null 2>&1; then
        echo "docker is required on the host" >&2
        exit 1
    fi
    if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
        echo "release Dockerfile not found: ${DOCKERFILE_PATH}" >&2
        echo "set DFB_RELEASE_DOCKERFILE=/path/to/Dockerfile if it is stored elsewhere" >&2
        exit 1
    fi

    host_cache_dir="${DFB_RELEASE_CACHE_DIR:-/tmp/dfb-release-cache-$(id -u)}"
    mkdir -p \
        "${host_cache_dir}/cargo-registry" \
        "${host_cache_dir}/cargo-git" \
        "${host_cache_dir}/cargo-target"

    docker_build_args=(
        --build-arg "UID=$(id -u)"
        --build-arg "GID=$(id -g)"
    )
    docker_env_args=()
    docker_run_network_args=()
    saw_loopback_proxy=0
    if [[ "${DFB_NO_PROXY_ARGS:-0}" != "1" ]]; then
        for proxy_name in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do
            proxy_value="${!proxy_name:-}"
            if [[ -n "${proxy_value}" ]]; then
                docker_build_args+=(--build-arg "${proxy_name}=${proxy_value}")
                docker_env_args+=(-e "${proxy_name}=${proxy_value}")
                if [[ "${proxy_value}" == *"127.0.0.1"* || "${proxy_value}" == *"localhost"* ]]; then
                    saw_loopback_proxy=1
                fi
            fi
        done
    fi

    docker_network="${DFB_DOCKER_NETWORK:-}"
    if [[ -z "${docker_network}" && "${saw_loopback_proxy}" == "1" ]]; then
        docker_network="host"
    fi
    if [[ -n "${docker_network}" ]]; then
        docker_run_network_args+=(--network "${docker_network}")
    fi

    if [[ "${DFB_SKIP_DOCKER_BUILD:-0}" != "1" ]]; then
        docker build \
            -t "${IMAGE_TAG}" \
            -f "${DOCKERFILE_PATH}" \
            "${docker_build_args[@]}" \
            "$(dirname "${DOCKERFILE_PATH}")"
    fi

    docker run --rm \
        "${docker_run_network_args[@]}" \
        -e DFB_RELEASE_INSIDE_CONTAINER=1 \
        -e "DFB_RELEASE_CARGO_BUILD_JOBS=${DFB_RELEASE_CARGO_BUILD_JOBS:-${CARGO_BUILD_JOBS_DEFAULT}}" \
        -e "DFB_RELEASE_CARGO_TARGET_DIR=${DFB_RELEASE_CARGO_TARGET_DIR:-${CONTAINER_CARGO_TARGET_DIR}}" \
        "${docker_env_args[@]}" \
        -v "${REPO_ROOT}:${CONTAINER_WORKDIR}" \
        -v "${host_cache_dir}/cargo-registry:/home/builder/.cargo/registry" \
        -v "${host_cache_dir}/cargo-git:/home/builder/.cargo/git" \
        -v "${host_cache_dir}/cargo-target:${CONTAINER_CARGO_TARGET_DIR}" \
        -w "${CONTAINER_WORKDIR}" \
        "${IMAGE_TAG}" \
        bash "scripts/$(basename "${BASH_SOURCE[0]}")"
    exit 0
fi

cd "${CONTAINER_WORKDIR}"

export CARGO_BUILD_JOBS="${DFB_RELEASE_CARGO_BUILD_JOBS:-${CARGO_BUILD_JOBS_DEFAULT}}"
export CARGO_TARGET_DIR="${DFB_RELEASE_CARGO_TARGET_DIR:-${CONTAINER_CARGO_TARGET_DIR}}"
export CARGO_INCREMENTAL=0

version="$(python3 - <<'PY'
from pathlib import Path
for line in Path("project_src/dfb_game/Cargo.toml").read_text().splitlines():
    stripped = line.strip()
    if stripped.startswith("version"):
        print(stripped.split("=", 1)[1].strip().strip('"'))
        break
else:
    raise SystemExit("failed to read package version")
PY
)"

cargo_bin_args=()
for bin_name in "${BINS[@]}"; do
    cargo_bin_args+=(--bin "${bin_name}")
done

stage_common_assets() {
    local staging_dir="$1"
    rm -rf "${staging_dir}"
    mkdir -p "${staging_dir}/recordings"
    rsync -a --delete assets/ "${staging_dir}/assets/"
    rsync -a --delete config/ "${staging_dir}/config/"
}

copy_binaries() {
    local staging_dir="$1"
    local source_dir="${CARGO_TARGET_DIR}/${LINUX_TARGET}/release"
    for bin_name in "${BINS[@]}"; do
        cp "${source_dir}/${bin_name}" "${staging_dir}/${bin_name}"
    done
}

zip_staging_dir() {
    local staging_dir="$1"
    local package_dir="$2"
    local package_name="$3"
    mkdir -p "${package_dir}"
    rm -f "${package_dir}/${package_name}.zip"
    (
        cd "$(dirname "${staging_dir}")"
        zip -qr "${package_dir}/${package_name}.zip" "$(basename "${staging_dir}")"
    )
}

rustup target add "${LINUX_TARGET}"

package_name="${PACKAGE_BASENAME_PREFIX}-${LINUX_TARGET}-${version}"
staging_dir="dist/linux/staging/${package_name}"
package_dir="${CONTAINER_WORKDIR}/dist/linux/package"

cargo build --release --jobs "${CARGO_BUILD_JOBS}" --target "${LINUX_TARGET}" "${cargo_bin_args[@]}"
stage_common_assets "${staging_dir}"
copy_binaries "${staging_dir}"
zip_staging_dir "${CONTAINER_WORKDIR}/${staging_dir}" "${package_dir}" "${package_name}"
echo "wrote ${package_dir}/${package_name}.zip"
