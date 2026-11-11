#!/usr/bin/env bash
# Configure every CMake preset (or a filtered subset) into build/<preset>/.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
common::init
common::install_err_trap

LOG_DIR="build/logs"
CONTINUE_ON_ERROR=0
LIST_ONLY=0
FILTER=""

usage() {
    cat <<EOF
${C_BOLD}Usage:${C_RESET} $0 [OPTIONS] [FILTER]

Configure every CMake preset into build/<preset>, one at a time.

${C_BOLD}Arguments:${C_RESET}
  FILTER                Backend prefix (cpu, cuda, hip) or an exact preset
                        name (e.g. cpu-asan-debug).

${C_BOLD}Options:${C_RESET}
  -c, --continue        Keep going after a preset fails.
  -l, --list            Print the matching presets and exit.
  -h, --help            Show this help and exit.

${C_BOLD}Examples:${C_RESET}
  $0                     Configure all presets.
  $0 cpu                 Configure the cpu-* presets.
  $0 --continue cuda     Configure cuda-* presets, ignoring failures.

Full output is written to ${C_BOLD}build/logs/<preset>.log${C_RESET}. Exit status:
0 = all configured, 1 = at least one preset failed, 2 = usage error.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -c|--continue) CONTINUE_ON_ERROR=1 ;;
        -l|--list) LIST_ONLY=1 ;;
        -*) usage_error "unknown option: $1" ;;
        *)
            [[ -z "$FILTER" ]] || usage_error "only one FILTER is allowed"
            FILTER="$1"
            ;;
    esac
    shift
done

if [[ -n "$FILTER" ]]; then
    mapfile -t PRESETS < <(common::list_presets | common::select_presets "$FILTER")
else
    mapfile -t PRESETS < <(common::list_presets)
fi

if (( ${#PRESETS[@]} == 0 )); then
    die "no presets match '$FILTER' (see --list)"
fi

TOTAL=${#PRESETS[@]}

if (( LIST_ONLY )); then
    printf '%sMatching presets (%d):%s\n' "$C_BOLD" "$TOTAL" "$C_RESET"
    for preset in "${PRESETS[@]}"; do
        printf '  %s%s%s\n' "$C_CYAN" "$preset" "$C_RESET"
    done
    exit 0
fi

# Prefer NovaNN's pinned vcpkg for this build and all child processes.
# shellcheck source=lib/vcpkg-env.sh
source "$SCRIPT_DIR/lib/vcpkg-env.sh"

mkdir -p "$LOG_DIR"

printf '\n%sNovaNN — configure presets%s\n' "$C_BOLD" "$C_RESET"
printf '%s%d preset(s) → build/<preset>  ·  full logs: %s%s\n' \
    "$C_DIM" "$TOTAL" "$LOG_DIR" "$C_RESET"

FAILED=0

for i in "${!PRESETS[@]}"; do
    preset="${PRESETS[$i]}"
    n=$((i + 1))

    printf '\n  %s[%2d/%d]%s %s▸ %s%s\n' \
        "$C_BOLD" "$n" "$TOTAL" "$C_RESET" "$C_CYAN" "$preset" "$C_RESET"

    start=$(date +%s)

    if [[ -t 1 ]]; then
        cmake --preset "$preset" >"$LOG_DIR/$preset.log" 2>&1 &
        run_pid=$!
        common::spinner "$run_pid" "$LOG_DIR/$preset.log" "$preset" "$n" "$TOTAL" &
        spin_pid=$!
        wait "$run_pid" && rc=0 || rc=$?
        wait "$spin_pid" 2>/dev/null || true
    else
        cmake --preset "$preset" >"$LOG_DIR/$preset.log" 2>&1 && rc=0 || rc=$?
    fi

    elapsed=$(( $(date +%s) - start ))

    if (( rc == 0 )); then
        printf '%s  %s✔ configured%s  %s→%s %s  %s(%s)%s\n' \
            "$C_CLEAR" "$C_GREEN" "$C_RESET" "$C_DIM" "$C_RESET" "build/$preset" \
            "$C_DIM" "$(fmt_elapsed "$elapsed")" "$C_RESET"
    else
        FAILED=$((FAILED + 1))
        printf '%s  %s✘ FAILED%s  → build/%s  %s(%s)%s\n' \
            "$C_CLEAR" "$C_RED" "$C_RESET" "$preset" "$C_DIM" "$(fmt_elapsed "$elapsed")" "$C_RESET"
        printf '%s  ── last lines of %s/%s.log ──%s\n' \
            "$C_YELLOW" "$LOG_DIR" "$preset" "$C_RESET" >&2
        tail -n 15 "$LOG_DIR/$preset.log" | while IFS= read -r line; do
            printf '  %s%s%s\n' "$C_DIM" "$line" "$C_RESET" >&2
        done
        if (( ! CONTINUE_ON_ERROR )); then
            printf '\n%sAborting: first failure (use --continue to skip failures).%s\n' \
                "$C_RED" "$C_RESET"
            exit 1
        fi
    fi
done

printf '\n'
if (( FAILED > 0 )); then
    printf '%s✘ %d of %d preset(s) failed — see %s/%s\n' \
        "$C_RED" "$FAILED" "$TOTAL" "$LOG_DIR" "$C_RESET"
    exit 1
fi
printf '%s✔ All %d preset(s) configured successfully.%s\n' \
    "$C_GREEN" "$TOTAL" "$C_RESET"
