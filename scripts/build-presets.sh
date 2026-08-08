#!/usr/bin/env bash
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
FILTER=""

die() {
    printf '%sERROR:%s %s\n' "$C_RED" "$C_RESET" "$*" >&2
    exit 1
}

usage_error() {
    printf '%sUsage error:%s %s\n' "$C_RED" "$C_RESET" "$*" >&2
    printf 'Run %s--help%s for usage.\n' "$C_BOLD" "$C_RESET" >&2
    exit 2
}

trap 's=$?; printf "%s%s: line %d — %s%s\n" "$C_RED" "$0" "$LINENO" "$BASH_COMMAND" "$C_RESET" >&2; exit "$s"' ERR

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

Full output is written to ${C_BOLD}build/logs/<preset>.log${C_RESET}; the terminal
only shows one summary line per preset. Exit status: 0 = all configured,
1 = at least one preset failed, 2 = usage error.
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
        -*)
            usage_error "unknown option: $1"
            ;;
        *)
            if [[ -n "$FILTER" ]]; then
                usage_error "only one FILTER is allowed"
            fi
            FILTER="$1"
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

if [[ -n "$FILTER" ]]; then
    MATCHED=()
    for preset in "${PRESETS[@]}"; do
        if [[ "$preset" == "$FILTER" || "$preset" == "$FILTER-"* ]]; then
            MATCHED+=("$preset")
        fi
    done
    if [[ ${#MATCHED[@]} -eq 0 ]]; then
        die "no presets match '${FILTER}' (see --list)"
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
        spinner "$run_pid" "$LOG_DIR/$preset.log" "$preset" "$n" "$TOTAL" &
        spin_pid=$!
        if wait "$run_pid"; then
            rc=0
        else
            rc=$?
        fi
        wait "$spin_pid" 2>/dev/null || true
    else
        if cmake --preset "$preset" >"$LOG_DIR/$preset.log" 2>&1; then
            rc=0
        else
            rc=$?
        fi
    fi

    elapsed=$(($(date +%s) - start))

    if [[ "$rc" -eq 0 ]]; then
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
        if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
            printf '\n%sAborting: first failure (use --continue to skip failures).%s\n' \
                "$C_RED" "$C_RESET"
            exit 1
        fi
    fi
done

printf '\n'
if [[ "$FAILED" -eq 0 ]]; then
    printf '%s✔ All %d preset(s) configured successfully.%s\n' \
        "$C_GREEN" "$TOTAL" "$C_RESET"
else
    printf '%s✘ %d of %d preset(s) failed — see %s/%s\n' \
        "$C_RED" "$FAILED" "$TOTAL" "$LOG_DIR" "$C_RESET"
    exit 1
fi
