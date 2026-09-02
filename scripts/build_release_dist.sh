#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage:
  scripts/build_release_dist.sh [linux-gnu|windows-msvc|all]

Behavior:
  linux-gnu     Build Linux package in Ubuntu 22.04 container.
  windows-msvc  Build Windows MSVC package on the host with cargo-xwin/xwin.
  all           Build Linux in container and Windows on the host.
EOF
}

platform_arg="${1:-all}"
if [[ "${platform_arg}" == "-h" || "${platform_arg}" == "--help" ]]; then
    usage
    exit 0
fi

case "${platform_arg}" in
    linux-gnu)
        exec "${SCRIPT_DIR}/build_release_linux_container.sh"
        ;;
    windows-msvc)
        exec "${SCRIPT_DIR}/build_release_windows_host_xwin.sh"
        ;;
    all)
        "${SCRIPT_DIR}/build_release_linux_container.sh"
        "${SCRIPT_DIR}/build_release_windows_host_xwin.sh"
        ;;
    *)
        echo "unknown platform: ${platform_arg}" >&2
        usage >&2
        exit 2
        ;;
esac
