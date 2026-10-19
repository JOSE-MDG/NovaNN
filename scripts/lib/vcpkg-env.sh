#!/usr/bin/env bash
## @file vcpkg-env.sh
## @brief Select NovaNN's project-local vcpkg for the current script.
##
## Source this file from NovaNN shell scripts before invoking `vcpkg` or CMake:
##   source "$SCRIPT_DIR/lib/vcpkg-env.sh"

_vcpkg_env_lib_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_vcpkg_env_project_root="$(dirname -- "$(dirname -- "$_vcpkg_env_lib_dir")")"
VCPKG_ROOT="$_vcpkg_env_project_root/thirdParty/vcpkg"

if [[ ! -f "$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" || ! -x "$VCPKG_ROOT/vcpkg" ]]; then
    printf 'error: NovaNN vcpkg submodule is missing, uninitialized, or not bootstrapped: %s\n' "$VCPKG_ROOT" >&2
    printf '       initialize/update it with: scripts/update-vcpkg.sh\n' >&2
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
