#!/usr/bin/env bash
## @file clean.sh
## @brief Clean build artifacts and logs for the NovaNN workspace.
##
## @details
## Without @c --target the whole @c build/ directory is deleted (with an
## interactive confirmation when stdin is a TTY).  With a directory target the
## script performs the CMake-equivalent of a target clean —
## @c cmake --build <dir> --target clean — which removes build artifacts while
## keeping the configuration intact.  Log targets remove what their names say:
## @c logs deletes all of @c build/logs/, while @c test-logs and
## @c build-logs delete only their own subtree.
##
## A bare preset name may be passed as the target (e.g. @c cpu-asan-debug)
## and is resolved to @c build/@<preset@> when such a directory exists.
## Directory targets must contain a CMake cache or Ninja build file; the
## script refuses anything that does not look like a configured build tree.
##
## @par Usage
## @code
##   clean.sh [OPTIONS] [--target <path|logs|test-logs|build-logs>]
## @endcode
##
## @par Options
## @li @c -n, @c —dry-run — print actions without deleting or cleaning.
## @li @c -y, @c —yes — assume yes for prompts (whole-build deletion).
## @li @c -h, @c —help — show help and exit.
##
## @par Exit status
## @li 0 — cleaned, or nothing to do.
## @li 1 — refused or operation failed.
## @li 2 — usage error.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# clean.sh lives at the repository root, unlike the scripts/ family.
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

# shellcheck source=scripts/lib/common.sh
source "$SCRIPT_DIR/scripts/lib/common.sh"
nova::install_err_trap

DRY_RUN=0
ASSUME_YES=0
TARGET=""

usage() {
    cat <<EOF
${C_BOLD}Usage:${C_RESET} $0 [OPTIONS] [--target <path|logs|test-logs|build-logs>]

${C_BOLD}Targets:${C_RESET}
  <path>                CMake build directory: runs
                        'cmake --build <path> --target clean' (artifacts are
                        removed, configuration is kept).  A bare preset name
                        is resolved to build/<preset> when it exists.
  logs                  Delete build/logs entirely.
  test-logs             Delete build/logs/tests only.
  build-logs            Delete build/logs/*.log (configure/build logs),
                        keeping tests/.
  (no --target)         Delete the entire build/ directory.

${C_BOLD}Options:${C_RESET}
  -n, --dry-run         Print actions without touching anything.
  -y, --yes             Assume yes for prompts.
  -h, --help            Show this help and exit.

${C_BOLD}Examples:${C_RESET}
  $0                                    # wipe build/
  $0 --target cuda-asan-test-debug      # cmake-clean one preset
  $0 --dry-run --target logs

Exit status: 0 = cleaned or nothing to do, 1 = refused or failed,
2 = usage error.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -n|--dry-run)
            DRY_RUN=1
            ;;
        -y|--yes)
            ASSUME_YES=1
            ;;
        --target)
            [[ $# -ge 2 ]] || usage_error "--target requires a value"
            if [[ -n "$TARGET" ]]; then
                usage_error "--target given more than once"
            fi
            TARGET="$2"
            shift
            ;;
        *)
            usage_error "unknown option: $1"
            ;;
    esac
    shift
done

run_cmd() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '%s[dry-run]%s %s\n' "$C_CYAN" "$C_RESET" "$*"
        return 0
    fi
    "$@"
}

printf '\n%sNovaNN — clean%s\n' "$C_BOLD" "$C_RESET"

if [[ -z "$TARGET" ]]; then
    if [[ ! -d "build" ]]; then
        printf '%snothing to do:%s build/ does not exist.\n' "$C_YELLOW" "$C_RESET"
        exit 0
    fi
    if [[ "$ASSUME_YES" -ne 1 && "$DRY_RUN" -ne 1 ]]; then
        if [[ ! -t 0 ]]; then
            die "refusing to delete build/ non-interactively; pass --yes"
        fi
        read -r -p "$(printf '%sDelete entire build/ directory?%s [y/N] ' "$C_RED" "$C_RESET")" reply
        if [[ ! "$reply" =~ ^[Yy]$ ]]; then
            printf '%saborted.%s\n' "$C_YELLOW" "$C_RESET"
            exit 0
        fi
    fi
    run_cmd rm -rf build
    if [[ "$DRY_RUN" -ne 1 ]]; then
        printf '%s  ✔ deleted%s build/\n' "$C_GREEN" "$C_RESET"
    fi
    exit 0
fi

case "$TARGET" in
    logs)
        if [[ ! -d "build/logs" ]]; then
            printf '%snothing to do:%s build/logs does not exist.\n' "$C_YELLOW" "$C_RESET"
            exit 0
        fi
        run_cmd rm -rf build/logs
        [[ "$DRY_RUN" -ne 1 ]] && printf '%s  ✔ deleted%s build/logs\n' "$C_GREEN" "$C_RESET"
        ;;
    test-logs)
        if [[ ! -d "build/logs/tests" ]]; then
            printf '%snothing to do:%s build/logs/tests does not exist.\n' "$C_YELLOW" "$C_RESET"
            exit 0
        fi
        run_cmd rm -rf build/logs/tests
        [[ "$DRY_RUN" -ne 1 ]] && printf '%s  ✔ deleted%s build/logs/tests\n' "$C_GREEN" "$C_RESET"
        ;;
    build-logs)
        shopt -s nullglob
        matches=(build/logs/*.log)
        shopt -u nullglob
        if [[ ${#matches[@]} -eq 0 ]]; then
            printf '%snothing to do:%s no build/logs/*.log files.\n' "$C_YELLOW" "$C_RESET"
            exit 0
        fi
        for f in "${matches[@]}"; do
            run_cmd rm -f "$f"
        done
        if [[ "$DRY_RUN" -ne 1 ]]; then
            printf '%s  ✔ deleted%s %d log file(s) from build/logs\n' \
                "$C_GREEN" "$C_RESET" "${#matches[@]}"
        fi
        ;;
    *)
        dir="$TARGET"
        if [[ -d "$dir" ]]; then
            :
        elif [[ "$TARGET" != */* && -d "build/$TARGET" ]]; then
            dir="build/$TARGET"
        else
            die "target '$TARGET': directory does not exist"
        fi
        if [[ ! -f "$dir/CMakeCache.txt" && ! -f "$dir/build.ninja" ]]; then
            die "'$dir' is not a configured CMake build directory"
        fi
        if [[ "$DRY_RUN" -eq 1 ]]; then
            printf '%s[dry-run]%s cmake --build %s --target clean\n' "$C_CYAN" "$C_RESET" "$dir"
        else
            printf '  %s▸ cleaning%s %s\n' "$C_CYAN" "$C_RESET" "$dir"
            if cmake --build "$dir" --target clean; then
                printf '%s  ✔ cleaned%s %s\n' "$C_GREEN" "$C_RESET" "$dir"
            else
                die "cmake --build $dir --target clean failed"
            fi
        fi
        ;;
esac
