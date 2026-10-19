#!/usr/bin/env bash
## @file update-vcpkg.sh
## @brief Update NovaNN's vcpkg submodule and synchronize registry baselines.
##
## @details
## Updates the project-local vcpkg submodule, bootstraps the executable if its
## pinned source commit changed, and runs x-update-baseline using that local
## vcpkg instance. It never runs git pull in the global vcpkg checkout.
##
## @par Usage
##   scripts/update-vcpkg.sh [OPTIONS]
##
## @par Options
##   -h, --help    Show help and exit.
##
## @par Exit status
##   0 — update and baseline synchronization completed.
##   1 — an update step failed.
##   2 — usage error.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname -- "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
nova::install_err_trap

VCPKG_REL_PATH="thirdParty/vcpkg"
VCPKG_DIR="$PROJECT_ROOT/$VCPKG_REL_PATH"

usage() {
    cat <<EOF_USAGE
${C_BOLD}Usage:${C_RESET} $0 [OPTIONS]

Update NovaNN's project-local vcpkg submodule and synchronize the baseline
in vcpkg-configuration.json. The global VCPKG_ROOT checkout is not used.

${C_BOLD}Options:${C_RESET}
  -h, --help    Show this help and exit.

${C_BOLD}Effects:${C_RESET}
  1. Fetch and update the vcpkg submodule to the remote's default branch tip.
  2. Bootstrap vcpkg if the source commit changed or the executable is missing.
  3. Run the local vcpkg x-update-baseline command.

The script does not stage or commit changes. Review vcpkg and
vcpkg-configuration.json with git diff before committing.
EOF_USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage_error "unknown option: $1"
            ;;
    esac
done

if ! git -C "$PROJECT_ROOT" rev-parse --show-toplevel >/dev/null 2>&1; then
    die "NovaNN root is not inside a Git repository"
fi
if [[ ! -f "$PROJECT_ROOT/.gitmodules" ]]; then
    die "no .gitmodules file found; add vcpkg as a Git submodule first"
fi
if [[ ! -f "$PROJECT_ROOT/vcpkg-configuration.json" ]]; then
    die "vcpkg-configuration.json was not found in the NovaNN root"
fi

# The baseline update rewrites that file, so uncommitted work in it
# would mix with the generated baseline.
if [[ -n "$(git -C "$PROJECT_ROOT" status --porcelain -- vcpkg-configuration.json)" ]]; then
    die "vcpkg-configuration.json has uncommitted changes; commit or stash them before updating the baseline"
fi

# If initialized, refuse to overwrite local changes inside the submodule.
if git -C "$VCPKG_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if [[ -n "$(git -C "$VCPKG_DIR" status --porcelain --untracked-files=normal)" ]]; then
        die "the vcpkg submodule has local changes; commit or discard them before updating"
    fi
    OLD_COMMIT="$(git -C "$VCPKG_DIR" rev-parse HEAD)"
else
    OLD_COMMIT=""
fi

printf '\n%sNovaNN — update vcpkg%s\n' "$C_BOLD" "$C_RESET"
printf '%sSubmodule: %s%s\n' "$C_DIM" "$VCPKG_REL_PATH" "$C_RESET"

printf '\n%s[1/3]%s Updating the project-local submodule...\n' "$C_CYAN" "$C_RESET"
git submodule update --init --remote -- "$VCPKG_REL_PATH"
NEW_COMMIT="$(git -C "$VCPKG_DIR" rev-parse HEAD)"
printf '  %sPinned source:%s %s → %s\n' \
    "$C_DIM" "$C_RESET" "${OLD_COMMIT:-<uninitialized>}" "$NEW_COMMIT"

# Override the user's shell-level VCPKG_ROOT for this process and its children.
export VCPKG_ROOT="$VCPKG_DIR"
export PATH="$VCPKG_ROOT:$PATH"

if [[ "$OLD_COMMIT" != "$NEW_COMMIT" || ! -x "$VCPKG_ROOT/vcpkg" ]]; then
    printf '\n%s[2/3]%s Bootstrapping the local vcpkg executable...\n' "$C_CYAN" "$C_RESET"
    "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
else
    printf '\n%s[2/3]%s Source commit unchanged; reusing the existing executable.\n' "$C_CYAN" "$C_RESET"
fi

if [[ ! -x "$VCPKG_ROOT/vcpkg" ]]; then
    die "bootstrap completed without producing an executable at $VCPKG_ROOT/vcpkg"
fi

printf '\n%s[3/3]%s Updating configured registry baselines...\n' "$C_CYAN" "$C_RESET"
"$VCPKG_ROOT/vcpkg" x-update-baseline

printf '\n%s✔ vcpkg update completed.%s\n' "$C_GREEN" "$C_RESET"
printf '\n%sReview the resulting changes:%s\n' "$C_BOLD" "$C_RESET"
printf '  git status --short -- %s vcpkg-configuration.json\n' "$VCPKG_REL_PATH"
printf '  git diff -- vcpkg-configuration.json\n'
printf '  git diff --submodule=log -- %s\n' "$VCPKG_REL_PATH"
printf '\n%sThe submodule pointer and baseline are not staged or committed automatically.%s\n' \
    "$C_DIM" "$C_RESET"
