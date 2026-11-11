#!/usr/bin/env bash
# Build the CMake presets that are already configured under build/<preset>/.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
common::init
common::install_err_trap

LOG_DIR="build/logs"
CONTINUE_ON_ERROR=0
LIST_ONLY=0
CONFIG=""
JOBS=""
FILTERS=()

usage() {
    cat <<EOF
${C_BOLD}Usage:${C_RESET} $0 [OPTIONS] [FILTER...]

Build every CMake preset already configured in build/<preset>, one at a time.

${C_BOLD}Arguments:${C_RESET}
  FILTER...             Backend prefix (cpu, cuda, hip) or an exact preset
                        name (e.g. cpu-asan-debug). Repeat to select several.

${C_BOLD}Options:${C_RESET}
  -C, --config MODE     Build configuration: Release or Debug.
                        Default: derived from the preset name
                        (*-debug* → Debug, everything else → Release).
  -j, --jobs N          Run the build with N parallel jobs.
  -c, --continue        Keep going after a preset fails.
  -l, --list            Print the matching presets and exit.
  -h, --help            Show this help and exit.

${C_BOLD}Examples:${C_RESET}
  $0                     Build all presets.
  $0 cpu                 Build the cpu-* presets.
  $0 --config Debug cpu-debug
  $0 -j 16 cuda --continue

Full output is appended to ${C_BOLD}build/logs/<preset>.log${C_RESET}. Presets without
a configured build directory are skipped with a warning. Exit status:
0 = all built, 1 = at least one preset failed or nothing was built,
2 = usage error.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -c|--continue) CONTINUE_ON_ERROR=1 ;;
        -l|--list) LIST_ONLY=1 ;;
        -C|--config)
            [[ $# -ge 2 ]] || usage_error "option '$1' requires a value"
            shift
            case "$1" in
                Release|Debug) CONFIG="$1" ;;
                *) usage_error "invalid --config '$1' (expected Release or Debug)" ;;
            esac
            ;;
        -j|--jobs)
            [[ $# -ge 2 ]] || usage_error "option '$1' requires a value"
            shift
            [[ "$1" =~ ^[1-9][0-9]*$ ]] || usage_error "invalid --jobs '$1' (expected a positive integer)"
            JOBS="$1"
            ;;
        -*) usage_error "unknown option: $1" ;;
        *) FILTERS+=("$1") ;;
    esac
    shift
done

mapfile -t PRESETS < <(common::list_presets)
if (( ${#FILTERS[@]} > 0 )); then
    mapfile -t PRESETS < <(printf '%s\n' "${PRESETS[@]}" | common::select_presets "${FILTERS[@]}")
fi

if (( ${#PRESETS[@]} == 0 )); then
    die "no presets match '${FILTERS[*]}' (see --list)"
fi

TOTAL=${#PRESETS[@]}

if (( LIST_ONLY )); then
    printf '%sMatching presets (%d):%s\n' "$C_BOLD" "$TOTAL" "$C_RESET"
    for preset in "${PRESETS[@]}"; do
        printf '  %s%s%s\n' "$C_CYAN" "$preset" "$C_RESET"
    done
    exit 0
fi

# shellcheck source=lib/vcpkg-env.sh
source "$SCRIPT_DIR/lib/vcpkg-env.sh"

mkdir -p "$LOG_DIR"

printf '\n%sNovaNN — compile presets%s\n' "$C_BOLD" "$C_RESET"
printf '%s%d preset(s) → cmake --build build/<preset>  ·  full logs: %s%s\n' \
    "$C_DIM" "$TOTAL" "$LOG_DIR" "$C_RESET"
if [[ -n "$JOBS" ]]; then
    printf '%sparallel jobs: %s%s\n' "$C_DIM" "$JOBS" "$C_RESET"
fi

FAILED=0
SKIPPED=0
BUILT=0

for i in "${!PRESETS[@]}"; do
    preset="${PRESETS[$i]}"
    n=$((i + 1))

    config="$CONFIG"
    if [[ -z "$config" ]]; then
        case "$preset" in
            *-debug*) config="Debug" ;;
            *) config="Release" ;;
        esac
    fi

    printf '\n  %s[%2d/%d]%s %s▸ %s%s  %s(%s)%s\n' \
        "$C_BOLD" "$n" "$TOTAL" "$C_RESET" "$C_CYAN" "$preset" "$C_RESET" \
        "$C_DIM" "$config" "$C_RESET"

    if [[ ! -d "build/$preset" ]]; then
        printf '  %s⚠ not configured — run scripts/build-presets.sh %s first%s\n' \
            "$C_YELLOW" "$preset" "$C_RESET"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    cmd=(cmake --build "build/$preset" --config "$config")
    [[ -n "$JOBS" ]] && cmd+=(--parallel "$JOBS")

    start=$(date +%s)

    if [[ -t 1 ]]; then
        "${cmd[@]}" >>"$LOG_DIR/$preset.log" 2>&1 &
        run_pid=$!
        common::spinner "$run_pid" "$LOG_DIR/$preset.log" "$preset" "$n" "$TOTAL" &
        spin_pid=$!
        wait "$run_pid" && rc=0 || rc=$?
        wait "$spin_pid" 2>/dev/null || true
    else
        "${cmd[@]}" >>"$LOG_DIR/$preset.log" 2>&1 && rc=0 || rc=$?
    fi

    elapsed=$(( $(date +%s) - start ))

    if (( rc == 0 )); then
        BUILT=$((BUILT + 1))
        printf '%s  %s✔ built%s  %s→%s %s  %s(%s)%s\n' \
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
    printf '%s✘ %d failed, %d built, %d skipped — see %s/%s\n' \
        "$C_RED" "$FAILED" "$BUILT" "$SKIPPED" "$LOG_DIR" "$C_RESET"
    exit 1
elif (( BUILT == 0 && SKIPPED > 0 )); then
    printf '%s⚠ nothing built — %d preset(s) not configured yet%s\n' \
        "$C_YELLOW" "$SKIPPED" "$C_RESET"
    printf '%s  Run scripts/build-presets.sh to configure them first.%s\n' \
        "$C_DIM" "$C_RESET"
    exit 1
elif (( SKIPPED > 0 )); then
    printf '%s✔ %d built, %d skipped (not configured)%s\n' \
        "$C_GREEN" "$BUILT" "$SKIPPED" "$C_RESET"
else
    printf '%s✔ All %d preset(s) built successfully.%s\n' \
        "$C_GREEN" "$TOTAL" "$C_RESET"
fi
