#!/usr/bin/env bash
# Select NovaNN's project-local vcpkg. Source it before invoking vcpkg or CMake:
#   source "$SCRIPT_DIR/lib/vcpkg-env.sh"
# The submodule is initialized and bootstrapped on demand, so a from-source
# build needs no separate setup script.

_vcpkg_env_lib_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_vcpkg_env_project_root="$(dirname -- "$(dirname -- "$_vcpkg_env_lib_dir")")"
VCPKG_ROOT="$_vcpkg_env_project_root/thirdParty/vcpkg"

if [[ ! -f "$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" ]]; then
    printf 'vcpkg: initializing the thirdParty/vcpkg submodule...\n' >&2
    git -C "$_vcpkg_env_project_root" submodule update --init -- thirdParty/vcpkg
fi

if [[ ! -x "$VCPKG_ROOT/vcpkg" ]]; then
    printf 'vcpkg: bootstrapping the local executable...\n' >&2
    "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
fi

if [[ ! -f "$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" || ! -x "$VCPKG_ROOT/vcpkg" ]]; then
    printf 'error: NovaNN vcpkg submodule is missing, uninitialized, or not bootstrapped: %s\n' "$VCPKG_ROOT" >&2
    printf '       initialize it with: git submodule update --init -- thirdParty/vcpkg\n' >&2
    printf '       or bootstrap it with: thirdParty/vcpkg/bootstrap-vcpkg.sh\n' >&2
    if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
        return 1
    fi
    exit 1
fi

export VCPKG_ROOT
# Put the project's executable ahead of any global vcpkg directory in PATH.
export PATH="$VCPKG_ROOT:$PATH"
unset _vcpkg_env_lib_dir _vcpkg_env_project_root
