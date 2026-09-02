#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WINDOWS_TARGET="x86_64-pc-windows-msvc"
PACKAGE_BASENAME_PREFIX="dfb"

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
  scripts/build_release_windows_host_xwin.sh

Environment:
  DFB_RELEASE_CACHE_DIR  Optional cache root. Defaults to /tmp/dfb-release-cache-<uid>.

Outputs:
  dist/windows/staging/dfb-x86_64-pc-windows-msvc-<version>/
  dist/windows/package/dfb-x86_64-pc-windows-msvc-<version>.zip
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

cd "${REPO_ROOT}"

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

find_command_candidate() {
    local name="$1"
    shift
    if command -v "${name}" >/dev/null 2>&1; then
        command -v "${name}"
        return
    fi
    for candidate in "$@"; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            command -v "${candidate}"
            return
        fi
    done
    return 1
}

ensure_target() {
    rustup target add "${WINDOWS_TARGET}"
}

ensure_host_xwin() {
    if ! command -v cargo-xwin >/dev/null 2>&1; then
        echo "missing cargo-xwin on host" >&2
        echo "Install it with: cargo install cargo-xwin --locked" >&2
        exit 1
    fi
    if ! command -v xwin >/dev/null 2>&1; then
        echo "missing xwin on host" >&2
        echo "Install it with: cargo install xwin --locked" >&2
        exit 1
    fi
}

ensure_windows_msvc_tools() {
    local shim_dir="${REPO_ROOT}/target/release-tool-shims"
    mkdir -p "${shim_dir}"

    local clang_cl
    local lld_link
    local llvm_lib
    clang_cl="$(find_command_candidate clang-cl clang-cl-18 clang-cl-17 clang-cl-16 clang-cl-15 clang-cl-14 clang clang-18 clang-17 clang-16 clang-15 clang-14)" || {
        echo "missing clang/clang-cl on host" >&2
        exit 1
    }
    lld_link="$(find_command_candidate lld-link lld-link-18 lld-link-17 lld-link-16 lld-link-15 lld-link-14)" || {
        echo "missing lld-link on host" >&2
        exit 1
    }
    llvm_lib="$(find_command_candidate llvm-lib llvm-lib-18 llvm-lib-17 llvm-lib-16 llvm-lib-15 llvm-lib-14)" || {
        echo "missing llvm-lib on host" >&2
        exit 1
    }

    ln -sf "${clang_cl}" "${shim_dir}/clang-cl"
    ln -sf "${lld_link}" "${shim_dir}/lld-link"
    ln -sf "${llvm_lib}" "${shim_dir}/llvm-lib"
    export PATH="${shim_dir}:${PATH}"
}

stage_common_assets() {
    local staging_dir="$1"
    rm -rf "${staging_dir}"
    mkdir -p "${staging_dir}/recordings"
    rsync -a --delete assets/ "${staging_dir}/assets/"
    rsync -a --delete config/ "${staging_dir}/config/"
}

copy_binaries() {
    local staging_dir="$1"
    local source_dir="target/${WINDOWS_TARGET}/release"
    for bin_name in "${BINS[@]}"; do
        cp "${source_dir}/${bin_name}.exe" "${staging_dir}/${bin_name}.exe"
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

ensure_target
ensure_host_xwin
ensure_windows_msvc_tools

host_cache_dir="${DFB_RELEASE_CACHE_DIR:-/tmp/dfb-release-cache-$(id -u)}"
mkdir -p "${host_cache_dir}/xwin-cache"
export XWIN_CACHE_DIR="${host_cache_dir}/xwin-cache"

package_name="${PACKAGE_BASENAME_PREFIX}-${WINDOWS_TARGET}-${version}"
staging_dir="dist/windows/staging/${package_name}"
package_dir="${REPO_ROOT}/dist/windows/package"

cargo xwin build --release --target "${WINDOWS_TARGET}" "${cargo_bin_args[@]}"
stage_common_assets "${staging_dir}"
copy_binaries "${staging_dir}"
zip_staging_dir "${REPO_ROOT}/${staging_dir}" "${package_dir}" "${package_name}"
echo "wrote ${package_dir}/${package_name}.zip"
