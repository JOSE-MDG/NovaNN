#!/usr/bin/env bash
## @file compile-presets.sh
## @brief Build CMake presets that have already been configured.
##
## @details
## Iterates over CMake presets and runs @c cmake --build for each one that
## has a configured build directory under @c build/@<preset@>/.  Full compiler
## output is appended to @c build/logs/@<preset@>.log; the terminal shows a
## single summary line per preset with a live progress spinner when stdout
## is a TTY.
##
## Presets without a configured build directory are skipped with a warning.
## The build configuration (Release/Debug) is derived from the preset name
## by default (@c *-debug* → Debug, everything else → Release) but can be
## overridden with @c -C.
##
## @par Usage
## @code
##   scripts/compile-presets.sh [OPTIONS] [FILTER...]
## @endcode
##
## @par Options
## @li @c -C, @c --config @c MODE — build configuration: Release or Debug.
## @li @c -j, @c --jobs @c N — run with N parallel jobs.
## @li @c -c, @c --continue — keep going after a preset fails.
## @li @c -l, @c —list — print matching presets and exit.
## @li @c -h, @c —help — show help and exit.
##
## @par Exit status
## @li 0 — all presets built (some may have been skipped).
## @li 1 — at least one preset failed, or nothing was built.
## @li 2 — usage error.
##
## @see build-presets.sh
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname -- "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

NO_COLOR="${NO_COLOR:-}"
TERM="${TERM:-}"

if [[ -t 1 && -z "$NO_COLOR" && "$TERM" != "dumb" ]]; then
    C_RESET=$'\033[0m'
    C_BOLD=$'\033[1m'
    C_DIM=$'\033[2m'
    C_RED=$'\033[31m'
    C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'
    C_CYAN=$'\033[36m'
else
    C_RESET=""; C_BOLD=""; C_DIM=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_CYAN=""
fi

C_CLEAR=""
[[ -t 1 ]] && C_CLEAR=$'\r\033[K'

LOG_DIR="build/logs"
CONTINUE_ON_ERROR=0
LIST_ONLY=0
CONFIG=""
JOBS=""
FILTERS=()

## @brief Print a formatted error message and exit with status 1.
## @param[in] ...  Message parts concatenated and printed to stderr.
die() {
    printf '%sERROR:%s %s\n' "$C_RED" "$C_RESET" "$*" >&2
    exit 1
}

## @brief Print a usage error message and exit with status 2.
## @param[in] ...  Message parts concatenated and printed to stderr.
usage_error() {
    printf '%sUsage error:%s %s\n' "$C_RED" "$C_RESET" "$*" >&2
    printf 'Run %s--help%s for usage.\n' "$C_BOLD" "$C_RESET" >&2
    exit 2
}

trap 's=$?; printf "%s%s: line %d — %s%s\n" "$C_RED" "$0" "$LINENO" "$BASH_COMMAND" "$C_RESET" >&2; exit "$s"' ERR

## @brief Format elapsed seconds into a human-readable string.
## @param[in] s  Elapsed time in seconds.
## @return Formatted string such as @c 5s or @c 2m 03s.
fmt_elapsed() {
    local s="$1" m
    m=$((s / 60))
    s=$((s % 60))
    if [[ "$m" -gt 0 ]]; then
        printf '%dm %02ds' "$m" "$s"
    else
        printf '%ds' "$s"
    fi
}

## @brief Display a live progress spinner while a background process runs.
##
## Parses the build log for @c [current/total] progress markers and renders
## a percentage bar with a braille or ASCII spinner.  Falls back to a simple
## label when no progress fraction is found.  Automatically detects UTF-8
## locale for braille characters.
##
## @param[in] pid      PID of the background process to monitor.
## @param[in] logfile  Path to the log file the process writes to.
## @param[in] label    Fallback label shown when no progress fraction is available.
## @param[in] n        Current index (1-based) in the preset list.
## @param[in] total    Total number of presets being processed.
spinner() {
    set +e
    trap - ERR
    local pid="$1" logfile="$2" label="$3" n="$4" total="$5"
    local locale="${LC_ALL:-${LC_CTYPE:-${LANG:-}}}"
    local frames bar_full bar_empty
    if [[ "$locale" == *[Uu][Tt][Ff]-8* ]]; then
        frames=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
        bar_full='█'
        bar_empty='░'
    else
        frames=('|' '/' '-' '\')
        bar_full='#'
        bar_empty='-'
    fi
    local i=0 start elapsed frac pct target num denom filled bar j offset
    start=$(date +%s 2>/dev/null || printf '0')
    offset=0
    if [[ -f "$logfile" ]]; then
        offset=$(wc -c < "$logfile" 2>/dev/null || printf '0')
    fi
    while kill -0 "$pid" 2>/dev/null; do
        elapsed=$(( $(date +%s 2>/dev/null || printf '0') - start ))
        [[ "$elapsed" -lt 0 ]] && elapsed=0
        frac=$(tail -c +$((offset + 1)) "$logfile" 2>/dev/null | tail -c 8192 | grep -aoE '\[[0-9]+/[0-9]+\][^[:cntrl:]]*' | tail -n1) || true
        pct=-1
        target=""
        if [[ -n "$frac" ]]; then
            num=${frac#[}
            num=${num%%/*}
            denom=${frac#*/}
            denom=${denom%%]*}
            if [[ "$num" =~ ^[0-9]+$ && "$denom" =~ ^[0-9]+$ && "$denom" -gt 0 ]]; then
                pct=$((num * 100 / denom))
            fi
            if [[ "$frac" == *"] "* ]]; then
                target="${frac#*] }"
            fi
            [[ -n "$target" ]] || target="$label"
            [[ ${#target} -le 40 ]] || target="${target:0:37}..."
        fi
        if [[ "$pct" -ge 0 ]]; then
            filled=$((pct * 20 / 100))
            bar=""
            for ((j = 0; j < filled; j++)); do bar+="$bar_full"; done
            for ((j = filled; j < 20; j++)); do bar+="$bar_empty"; done
            printf '\r  %s %s%3d%%%s %s[%s]%s %s%s%s · %s[%d/%d]%s · %s%s%s' \
                "${frames[$i]}" "$C_CYAN" "$pct" "$C_RESET" \
                "$C_CYAN" "$bar" "$C_RESET" \
                "$C_DIM" "$target" "$C_RESET" \
                "$C_DIM" "$n" "$total" "$C_RESET" \
                "$C_DIM" "$(fmt_elapsed "$elapsed")" "$C_RESET"
        else
            printf '\r  %s %s%s%s · %s[%d/%d]%s · %s%s%s' \
                "${frames[$i]}" "$C_BOLD" "$label" "$C_RESET" \
                "$C_DIM" "$n" "$total" "$C_RESET" \
                "$C_DIM" "$(fmt_elapsed "$elapsed")" "$C_RESET"
        fi
        sleep 0.1
        i=$(( (i + 1) % ${#frames[@]} ))
    done
    return 0
}

## @brief Print usage information to stdout and exit.
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

Full output is appended to ${C_BOLD}build/logs/<preset>.log${C_RESET}; the terminal
only shows one summary line per preset. Presets without a configured build
directory are skipped with a warning. Exit status: 0 = all built,
1 = at least one preset failed or nothing was built, 2 = usage error.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -c|--continue)
            CONTINUE_ON_ERROR=1
            ;;
        -l|--list)
            LIST_ONLY=1
            ;;
        -C|--config)
            if [[ $# -lt 2 ]]; then
                usage_error "option '$1' requires a value"
            fi
            shift
            case "$1" in
                Release|Debug)
                    CONFIG="$1"
                    ;;
                *)
                    usage_error "invalid --config '$1' (expected Release or Debug)"
                    ;;
            esac
            ;;
        -j|--jobs)
            if [[ $# -lt 2 ]]; then
                usage_error "option '$1' requires a value"
            fi
            shift
            if [[ ! "$1" =~ ^[1-9][0-9]*$ ]]; then
                usage_error "invalid --jobs '$1' (expected a positive integer)"
            fi
            JOBS="$1"
            ;;
        -*)
            usage_error "unknown option: $1"
            ;;
        *)
            FILTERS+=("$1")
            ;;
    esac
    shift
done

PRESETS=()
while IFS= read -r preset; do
    PRESETS+=("$preset")
done < <(cmake --list-presets 2>/dev/null | sed -n 's/^  "\([^"]*\)".*/\1/p')

if [[ ${#PRESETS[@]} -eq 0 ]]; then
    die "no CMake presets found — is CMakePresets.json present?"
fi

if [[ ${#FILTERS[@]} -gt 0 ]]; then
    MATCHED=()
    for preset in "${PRESETS[@]}"; do
        for filter in "${FILTERS[@]}"; do
            if [[ "$preset" == "$filter" || "$preset" == "$filter-"* ]]; then
                if [[ " ${MATCHED[*]} " != *" $preset "* ]]; then
                    MATCHED+=("$preset")
                fi
                break
            fi
        done
    done
    if [[ ${#MATCHED[@]} -eq 0 ]]; then
        die "no presets match '${FILTERS[*]}' (see --list)"
    fi
    PRESETS=("${MATCHED[@]}")
fi

TOTAL=${#PRESETS[@]}

if [[ "$LIST_ONLY" -eq 1 ]]; then
    printf '%sMatching presets (%d):%s\n' "$C_BOLD" "$TOTAL" "$C_RESET"
    for preset in "${PRESETS[@]}"; do
        printf '  %s%s%s\n' "$C_CYAN" "$preset" "$C_RESET"
    done
    exit 0
fi

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
    if [[ -n "$JOBS" ]]; then
        cmd+=(--parallel "$JOBS")
    fi

    start=$(date +%s)

    if [[ -t 1 ]]; then
        "${cmd[@]}" >>"$LOG_DIR/$preset.log" 2>&1 &
        run_pid=$!
        spinner "$run_pid" "$LOG_DIR/$preset.log" "$preset" "$n" "$TOTAL" &
        spin_pid=$!
        if wait "$run_pid"; then
            rc=0
        else
            rc=$?
        fi
        wait "$spin_pid" 2>/dev/null || true
    else
        if "${cmd[@]}" >>"$LOG_DIR/$preset.log" 2>&1; then
            rc=0
        else
            rc=$?
        fi
    fi

    elapsed=$(($(date +%s) - start))

    if [[ "$rc" -eq 0 ]]; then
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
        if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
            printf '\n%sAborting: first failure (use --continue to skip failures).%s\n' \
                "$C_RED" "$C_RESET"
            exit 1
        fi
    fi
done

printf '\n'
if [[ "$FAILED" -gt 0 ]]; then
    printf '%s✘ %d failed, %d built, %d skipped — see %s/%s\n' \
        "$C_RED" "$FAILED" "$BUILT" "$SKIPPED" "$LOG_DIR" "$C_RESET"
    exit 1
elif [[ "$BUILT" -eq 0 && "$SKIPPED" -gt 0 ]]; then
    printf '%s⚠ nothing built — %d preset(s) not configured yet%s\n' \
        "$C_YELLOW" "$SKIPPED" "$C_RESET"
    printf '%s  Run scripts/build-presets.sh to configure them first.%s\n' \
        "$C_DIM" "$C_RESET"
    exit 1
elif [[ "$SKIPPED" -gt 0 ]]; then
    printf '%s✔ %d built, %d skipped (not configured)%s\n' \
        "$C_GREEN" "$BUILT" "$SKIPPED" "$C_RESET"
else
    printf '%s✔ All %d preset(s) built successfully.%s\n' \
        "$C_GREEN" "$TOTAL" "$C_RESET"
fi
