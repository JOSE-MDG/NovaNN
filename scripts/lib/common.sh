#!/usr/bin/env bash
# Shared helpers for the NovaNN scripts. Sourced, never executed.

# Colors, enabled only when stdout is a TTY.
if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" ]]; then
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

die() {
    printf '%sERROR:%s %s\n' "$C_RED" "$C_RESET" "$*" >&2
    exit 1
}

usage_error() {
    printf '%sUsage error:%s %s\n' "$C_RED" "$C_RESET" "$*" >&2
    printf 'Run %s--help%s for usage.\n' "$C_BOLD" "$C_RESET" >&2
    exit 2
}

# Print the failing command and line before exiting.
common::install_err_trap() {
    trap 's=$?; printf "%s%s: line %d: %s%s\n" "$C_RED" "$0" "$LINENO" "$BASH_COMMAND" "$C_RESET" >&2; exit "$s"' ERR
}

# Set PROJECT_ROOT from the calling script and move there.
common::init() {
    PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[1]}")/.." && pwd)"
    cd "$PROJECT_ROOT"
}

fmt_elapsed() {
    local minutes=$(( $1 / 60 )) seconds=$(( $1 % 60 ))
    if (( minutes > 0 )); then
        printf '%dm %02ds' "$minutes" "$seconds"
    else
        printf '%ds' "$seconds"
    fi
}

# Print one CMake preset name per line.
common::list_presets() {
    local presets
    presets=$(cmake --list-presets 2>/dev/null | sed -n 's/^  "\([^"]*\)".*/\1/p')
    [[ -n "$presets" ]] || die "no CMake presets found; is CMakePresets.json present?"
    printf '%s\n' "$presets"
}

# Filter preset names from stdin: exact match or "<filter>-*". With no
# filters, everything passes through.
common::select_presets() {
    local filters=("$@") preset f
    while IFS= read -r preset; do
        if (( ${#filters[@]} == 0 )); then
            printf '%s\n' "$preset"
            continue
        fi
        for f in "${filters[@]}"; do
            if [[ "$preset" == "$f" || "$preset" == "$f-"* ]]; then
                printf '%s\n' "$preset"
                break
            fi
        done
    done
}

# Spinner for a background process. Tails $logfile for "[n/m]" markers;
# $ere overrides the marker regex for tools with a different format.
common::spinner() {
    set +e
    trap - ERR

    local pid="$1" logfile="$2" label="$3" n="$4" total="$5"
    local ere="${6:-\[[0-9]+/[0-9]+\][^[:cntrl:]]*}"
    local frames=('|' '/' '-' '\') i=0 start elapsed raw token rest num denom pct target
    start=$(date +%s)

    while kill -0 "$pid" 2>/dev/null; do
        elapsed=$(( $(date +%s) - start ))
        raw=$(tail -c 8192 "$logfile" 2>/dev/null | grep -aoE "$ere" | tail -1)
        pct=-1
        target=""

        # Both marker shapes start with a whitespace-delimited num/denom token:
        # Ninja writes "[1/3] Building ...", ctest writes "1/2328 Test ...".
        token=${raw%%[[:space:]]*}
        rest=${raw#"$token"}
        if [[ "$token" =~ ^\[?([0-9]+)/([0-9]+)\]?$ ]]; then
            num=${BASH_REMATCH[1]}
            denom=${BASH_REMATCH[2]}
            if (( denom > 0 )); then
                pct=$(( num * 100 / denom ))
                target="${rest# }"
                [[ -n "$target" ]] || target="$label"
                (( ${#target} <= 60 )) || target="${target:0:57}..."
            fi
        fi

        printf '\r  %s ' "${frames[$i]}"
        if (( pct >= 0 )); then
            printf '%s%3d%%%s %s%s%s' "$C_CYAN" "$pct" "$C_RESET" "$C_DIM" "$target" "$C_RESET"
        else
            printf '%s%s%s' "$C_BOLD" "$label" "$C_RESET"
        fi
        printf ' %s· [%d/%d] · %s%s' "$C_DIM" "$n" "$total" "$(fmt_elapsed "$elapsed")" "$C_RESET"

        sleep 0.1
        i=$(( (i + 1) % 4 ))
    done
}
