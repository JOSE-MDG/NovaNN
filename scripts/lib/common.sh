#!/usr/bin/env bash
## @file common.sh
## @brief Shared helpers for the NovaNN script family.
##
## @details
## Sourced (never executed) by the scripts in @c scripts/.  Provides TTY-aware
## color variables, error helpers, elapsed-time formatting, CMake preset
## discovery, and a reusable progress spinner whose progress pattern can be
## adapted per tool (CMake/Ninja bracket markers, ctest counters, ...).
##
## The library assumes the consumer runs with @c set -Eeuo pipefail and calls
## @c nova::install_err_trap itself.
##
## @see ../build-presets.sh
## @see ../compile-presets.sh
## @see ../run-tests.sh
## @see ../../clean.sh

# ---------------------------------------------------------------------------
# Colors (resolved once, when the library is sourced).
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

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

## @brief Install the standard ERR trap used across the script family.
## @details
## Must be called by the consumer after @c set -Eeuo pipefail; the trap
## expands @c \$0 / @c \$LINENO / @c \$BASH_COMMAND in the failing context.
nova::install_err_trap() {
    trap 's=$?; printf "%s%s: line %d — %s%s\n" "$C_RED" "$0" "$LINENO" "$BASH_COMMAND" "$C_RESET" >&2; exit "$s"' ERR
}

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Preset discovery
# ---------------------------------------------------------------------------

## @brief Emit every CMake preset name, one per line.
## @details
## Parses @c cmake --list-presets output; dies when no presets are found.
common::list_presets() {
    local presets=()
    local line
    while IFS= read -r line; do
        presets+=("$line")
    done < <(cmake --list-presets 2>/dev/null | sed -n 's/^  "\([^"]*\)".*/\1/p')

    if [[ ${#presets[@]} -eq 0 ]]; then
        die "no CMake presets found — is CMakePresets.json present?"
    fi

    local p
    for p in "${presets[@]}"; do
        printf '%s\n' "$p"
    done
}

# ---------------------------------------------------------------------------
# Progress spinner
# ---------------------------------------------------------------------------

## @brief Display a live progress spinner while a background process runs.
##
## @param[in] pid      PID of the background process to monitor.
## @param[in] logfile  Log file the process writes to (tail-scanned).
## @param[in] label    Fallback label when no progress was parsed yet.
## @param[in] n        Current index (1-based) in the work list.
## @param[in] total    Total number of work items.
## @param[in] ere      Optional extended regex locating the newest progress
##                     marker in @p logfile. The match must contain a
##                     @c NUM/DENOM pair (optionally wrapped in brackets),
##                     optionally followed by a description. Defaults to the
##                     CMake/Ninja style @c \`[num/denom] text\` marker.
##
## Recognized marker shapes:
## @li @c \\[[0-9]+/[0-9]+\\][^[:cntrl:]]*  — CMake/Ninja (@c default )
## @li @c [0-9]+/[0-9]+[ ]+Test[^[:cntrl:]]* — ctest counters
common::spinner() {
    set +e
    trap - ERR

    local pid="$1" logfile="$2" label="$3" n="$4" total="$5"
    local ere="${6:-\[[0-9]+/[0-9]+\][^[:cntrl:]]*}"

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

    local i=0 start elapsed raw token rest t2 num denom frac_pct
    local filled bar j target
    start=$(date +%s 2>/dev/null || printf '0')
    local offset=0
    if [[ -f "$logfile" ]]; then
        offset=$(wc -c < "$logfile" 2>/dev/null || printf '0')
    fi

    while kill -0 "$pid" 2>/dev/null; do
        elapsed=$(( $(date +%s 2>/dev/null || printf '0') - start ))
        [[ "$elapsed" -lt 0 ]] && elapsed=0

        raw=$(tail -c +$((offset + 1)) "$logfile" 2>/dev/null | tail -c 8192 \
              | grep -aoE "$ere" | tail -n1) || true

        frac_pct=-1
        target=""
        if [[ -n "$raw" ]]; then
            token=${raw%%[[:space:]]*}
            rest=${raw#"$token"}
            rest=${rest# }
            t2=${token#\[}
            num=${t2%%/*}
            denom=${t2##*/}
            denom=${denom%\]}
            if [[ "$num" =~ ^[0-9]+$ && "$denom" =~ ^[0-9]+$ && "$denom" -gt 0 ]]; then
                frac_pct=$((num * 100 / denom))
                target="$rest"
                [[ -n "$target" ]] || target="$label"
                [[ ${#target} -le 40 ]] || target="${target:0:37}..."
            fi
        fi

        if [[ "$frac_pct" -ge 0 ]]; then
            filled=$((frac_pct * 20 / 100))
            bar=""
            for ((j = 0; j < filled; j++)); do bar+="$bar_full"; done
            for ((j = filled; j < 20; j++)); do bar+="$bar_empty"; done
            printf '\r  %s %s%3d%%%s %s[%s]%s %s%s%s · %s[%d/%d]%s · %s%s%s' \
                "${frames[$i]}" "$C_CYAN" "$frac_pct" "$C_RESET" \
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
