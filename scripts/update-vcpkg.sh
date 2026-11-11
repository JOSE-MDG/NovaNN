#!/usr/bin/env bash
# Initialize or move the project-local vcpkg submodule. Maintainer tool; the
# build scripts provision vcpkg on their own.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
common::init
common::install_err_trap

VCPKG_REL_PATH="thirdParty/vcpkg"
VCPKG_DIR="$PROJECT_ROOT/$VCPKG_REL_PATH"
REMOTE=0

usage() {
    cat <<EOF
${C_BOLD}Usage:${C_RESET} $0 [OPTIONS]

Ensure the project-local vcpkg submodule is initialized at the commit pinned
by this repository and that its executable is bootstrapped. The global
VCPKG_ROOT checkout is not used.

${C_BOLD}Options:${C_RESET}
  -r, --remote        Move the submodule to the remote's default branch tip
                      (maintainer bump of the vcpkg version).
  -h, --help          Show this help and exit.

The script does not stage or commit the submodule pointer. Review it with
git diff --submodule=log before committing. Exit status: 0 = ready,
1 = an update step failed, 2 = usage error.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -r|--remote) REMOTE=1 ;;
        *) usage_error "unknown option: $1" ;;
    esac
    shift
done

git -C "$PROJECT_ROOT" rev-parse --show-toplevel >/dev/null 2>&1 \
    || die "NovaNN root is not inside a Git repository"
[[ -f "$PROJECT_ROOT/.gitmodules" ]] || die "no .gitmodules file found; add vcpkg as a Git submodule first"

# If initialized, refuse to overwrite local changes inside the submodule.
OLD_COMMIT=""
if git -C "$VCPKG_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    [[ -z "$(git -C "$VCPKG_DIR" status --porcelain --untracked-files=normal)" ]] \
        || die "the vcpkg submodule has local changes; commit or discard them before updating"
    OLD_COMMIT="$(git -C "$VCPKG_DIR" rev-parse HEAD)"
fi

printf '\n%sNovaNN — vcpkg submodule%s\n' "$C_BOLD" "$C_RESET"
printf '%sPath: %s%s\n' "$C_DIM" "$VCPKG_REL_PATH" "$C_RESET"

printf '\n%s[1/2]%s Updating the project-local submodule...\n' "$C_CYAN" "$C_RESET"
if (( REMOTE )); then
    git submodule update --init --remote -- "$VCPKG_REL_PATH"
else
    git submodule update --init -- "$VCPKG_REL_PATH"
fi
NEW_COMMIT="$(git -C "$VCPKG_DIR" rev-parse HEAD)"
printf '  %sPinned source:%s %s → %s\n' \
    "$C_DIM" "$C_RESET" "${OLD_COMMIT:-<uninitialized>}" "$NEW_COMMIT"

# Override the user's shell-level VCPKG_ROOT for this process and its children.
export VCPKG_ROOT="$VCPKG_DIR"
export PATH="$VCPKG_ROOT:$PATH"

if [[ "$OLD_COMMIT" != "$NEW_COMMIT" || ! -x "$VCPKG_ROOT/vcpkg" ]]; then
    printf '\n%s[2/2]%s Bootstrapping the local vcpkg executable...\n' "$C_CYAN" "$C_RESET"
    "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
else
    printf '\n%s[2/2]%s Source commit unchanged; reusing the existing executable.\n' "$C_CYAN" "$C_RESET"
fi

[[ -x "$VCPKG_ROOT/vcpkg" ]] || die "bootstrap completed without producing an executable at $VCPKG_ROOT/vcpkg"

printf '\n%s✔ vcpkg submodule ready.%s\n' "$C_GREEN" "$C_RESET"
printf '\n%sReview the resulting changes:%s\n' "$C_BOLD" "$C_RESET"
printf '  git status --short -- %s\n' "$VCPKG_REL_PATH"
printf '  git diff --submodule=log -- %s\n' "$VCPKG_REL_PATH"
printf '\n%sThe submodule pointer is not staged or committed automatically.%s\n' "$C_DIM" "$C_RESET"
