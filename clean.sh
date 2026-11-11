#!/usr/bin/env bash
# Clean build artifacts and logs. Without --target the whole build/ tree is
# removed; with one, that tree is cleaned and its source-tree codegen outputs
# are preserved.

set -Eeuo pipefail

# clean.sh lives at the repository root, unlike the scripts/ family.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# shellcheck source=scripts/lib/common.sh
source "$PROJECT_ROOT/scripts/lib/common.sh"
common::install_err_trap

DRY_RUN=0
ASSUME_YES=0
TARGET=""

usage() {
    cat <<EOF
${C_BOLD}Usage:${C_RESET} $0 [OPTIONS] [--target <path|logs|test-logs|build-logs>]

${C_BOLD}Targets:${C_RESET}
  <path>                CMake build directory: runs
                        'cmake --build <path> --target clean'. Artifacts are
                        removed, configuration is kept, and source-tree codegen
                        outputs are snapshotted and restored. A bare preset
                        name is resolved to build/<preset> when it exists.
  logs                  Delete build/logs entirely.
  test-logs             Delete build/logs/tests only.
  build-logs            Delete build/logs/*.log, keeping tests/.
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
        -h|--help) usage; exit 0 ;;
        -n|--dry-run) DRY_RUN=1 ;;
        -y|--yes) ASSUME_YES=1 ;;
        --target)
            [[ $# -ge 2 ]] || usage_error "--target requires a value"
            [[ -z "$TARGET" ]] || usage_error "--target given more than once"
            TARGET="$2"
            shift
            ;;
        *) usage_error "unknown option: $1" ;;
    esac
    shift
done

run_cmd() {
    if (( DRY_RUN )); then
        printf '%s[dry-run]%s %s\n' "$C_CYAN" "$C_RESET" "$*"
        return 0
    fi
    "$@"
}

# Codegen outputs live in the source tree, but Ninja's clean deletes them along
# with the object files. Snapshot them before a directory clean, restore them
# afterwards byte for byte.
backup_codegen_outputs() {
    local list backup_dir f
    list="$(uv run -q tools/codegen/generate.py gen --all --list-outputs 2>/dev/null)" || return 0
    [[ -n "$list" ]] || return 0

    backup_dir="$(mktemp -d "${TMPDIR:-/tmp}/nova-clean.XXXXXXXX")"
    while IFS= read -r f; do
        [[ -n "$f" && -f "$f" ]] || continue
        mkdir -p "$backup_dir/$(dirname -- "$f")"
        cp -p "$f" "$backup_dir/$f"
    done <<<"$list"
    printf '%s' "$backup_dir"
}

restore_codegen_outputs() {
    [[ -n "${1:-}" && -d "${1:-}" ]] || return 0
    cp -rp "$1/." "$PROJECT_ROOT/"
    rm -rf "$1"
}

printf '\n%sNovaNN — clean%s\n' "$C_BOLD" "$C_RESET"

if [[ -z "$TARGET" ]]; then
    if [[ ! -d "build" ]]; then
        printf '%snothing to do:%s build/ does not exist.\n' "$C_YELLOW" "$C_RESET"
        exit 0
    fi

    if (( ! ASSUME_YES && ! DRY_RUN )); then
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
    (( DRY_RUN )) || printf '%s  ✔ deleted%s build/\n' "$C_GREEN" "$C_RESET"
    exit 0
fi

case "$TARGET" in
    logs)
        if [[ ! -d "build/logs" ]]; then
            printf '%snothing to do:%s build/logs does not exist.\n' "$C_YELLOW" "$C_RESET"
            exit 0
        fi
        run_cmd rm -rf build/logs
        (( DRY_RUN )) || printf '%s  ✔ deleted%s build/logs\n' "$C_GREEN" "$C_RESET"
        ;;
    test-logs)
        if [[ ! -d "build/logs/tests" ]]; then
            printf '%snothing to do:%s build/logs/tests does not exist.\n' "$C_YELLOW" "$C_RESET"
            exit 0
        fi
        run_cmd rm -rf build/logs/tests
        (( DRY_RUN )) || printf '%s  ✔ deleted%s build/logs/tests\n' "$C_GREEN" "$C_RESET"
        ;;
    build-logs)
        shopt -s nullglob
        logs=(build/logs/*.log)
        shopt -u nullglob

        if (( ${#logs[@]} == 0 )); then
            printf '%snothing to do:%s no build/logs/*.log files.\n' "$C_YELLOW" "$C_RESET"
            exit 0
        fi
        for f in "${logs[@]}"; do
            run_cmd rm -f "$f"
        done
        (( DRY_RUN )) || printf '%s  ✔ deleted%s %d log file(s) from build/logs\n' \
            "$C_GREEN" "$C_RESET" "${#logs[@]}"
        ;;
    *)
        dir="$TARGET"
        if [[ ! -d "$dir" && "$TARGET" != */* && -d "build/$TARGET" ]]; then
            dir="build/$TARGET"
        fi
        [[ -d "$dir" ]] || die "target '$TARGET': directory does not exist"

        if [[ ! -f "$dir/CMakeCache.txt" && ! -f "$dir/build.ninja" ]]; then
            die "'$dir' is not a configured CMake build directory"
        fi

        if (( DRY_RUN )); then
            printf '%s[dry-run]%s cmake --build %s --target clean (codegen outputs preserved)\n' \
                "$C_CYAN" "$C_RESET" "$dir"
            exit 0
        fi

        printf '  %s▸ cleaning%s %s\n' "$C_CYAN" "$C_RESET" "$dir"
        backup_dir="$(backup_codegen_outputs)"
        clean_rc=0
        cmake --build "$dir" --target clean || clean_rc=$?
        restore_codegen_outputs "$backup_dir"
        (( clean_rc == 0 )) || die "cmake --build $dir --target clean failed"
        printf '%s  ✔ cleaned%s %s\n' "$C_GREEN" "$C_RESET" "$dir"
        ;;
esac
