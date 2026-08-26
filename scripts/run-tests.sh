#!/usr/bin/env bash
## @file run-tests.sh
## @brief Run ctest suites for the NovaNN test presets.
##
## @details
## Discovers every @c *-test-* CMake preset (optionally restricted by FILTER,
## using the same prefix/exact matching as build-presets.sh) and runs
## @c ctest --preset for each configured one.  Sanitizer runtime options are
## managed automatically: ASan needs @c protect_shadow_gap=0 for CUDA device
## probes, and LeakSanitizer discards vendor-runtime internals through the
## repository-root @c suppr.txt.
##
## Without arguments the script sweeps **all** test presets and continues past
## failing presets, collecting a final report.  Pass
## @c --force-stop-on-failure to abort at the first red preset instead.
## Anything after a literal @c -- is forwarded verbatim to every ctest call.
##
## Full output per preset is written to @c build/logs/tests/@<preset@>.log;
## the terminal shows one summary line per preset with a live spinner when
## stdout is a TTY.
##
## @par Usage
## @code
##   scripts/run-tests.sh [OPTIONS] [FILTER] [-- CTEST_ARGS...]
## @endcode
##
## @par Options
## @li @c --force-stop-on-failure — abort at the first preset with failures.
## @li @c -l, @c —list — print matching test presets and exit.
## @li @c -h, @c —help — show help and exit.
##
## @par Exit status
## @li 0 — all executed presets passed (skips excluded).
## @li 1 — at least one preset reported failures.
## @li 2 — usage error.
##
## @see clean.sh
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname -- "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
nova::install_err_trap

LOG_DIR="build/logs/tests"
FORCE_STOP=0
LIST_ONLY=0
FILTER=""
CTEST_ARGS=()

usage() {
    cat <<EOF
${C_BOLD}Usage:${C_RESET} $0 [OPTIONS] [FILTER] [-- CTEST_ARGS...]

Run ctest for every *-test-* CMake preset (or the subset matched by FILTER).

${C_BOLD}Arguments:${C_RESET}
  FILTER                Backend prefix (cpu, cuda, hip) or an exact preset
                        name (e.g. cpu-asan-test-debug).

${C_BOLD}Options:${C_RESET}
      --force-stop-on-failure
                        Abort at the first preset reporting failures instead
                        of sweeping everything and summarizing.
  -l, --list            Print the matching test presets and exit.
  -h, --help            Show this help and exit.

${C_BOLD}ctest pass-through:${C_RESET}
  Everything after a literal '--' is forwarded verbatim to each ctest call:

    $0 hip -- -V                       # verbose
    $0 cuda-asan-test-release -- -j 8  # parallel
    $0 hip -- -R 'BitPatternIdentity.*' -V

${C_BOLD}Environment:${C_RESET} managed automatically from ${C_BOLD}suppr.txt${C_RESET}:
  ASAN_OPTIONS=protect_shadow_gap=0     (CUDA probes fail under ASan otherwise)
  LSAN_OPTIONS=suppressions=<root>/suppr.txt

Full output is written to ${C_BOLD}build/logs/tests/<preset>.log${C_RESET}.
Exit status: 0 = all executed presets passed, 1 = failures found,
2 = usage error.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -l|--list)
            LIST_ONLY=1
            ;;
        --force-stop-on-failure)
            FORCE_STOP=1
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                CTEST_ARGS+=("$1")
                shift
            done
            break
            ;;
        -*)
            usage_error "unknown option: $1 (forward ctest flags after '--')"
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

# ---------------------------------------------------------------------------
# Preset discovery: family filter, then keep test presets only.
# ---------------------------------------------------------------------------

ALL_PRESETS=()
while IFS= read -r p; do
    ALL_PRESETS+=("$p")
done < <(common::list_presets)

if [[ ${#ALL_PRESETS[@]} -eq 0 ]]; then
    die "no CMake presets found — is CMakePresets.json present?"
fi

if [[ -n "$FILTER" ]]; then
    MATCHED=()
    for preset in "${ALL_PRESETS[@]}"; do
        if [[ "$preset" == "$FILTER" || "$preset" == "$FILTER-"* ]]; then
            MATCHED+=("$preset")
        fi
    done
else
    MATCHED=("${ALL_PRESETS[@]}")
fi

PRESETS=()
for preset in "${MATCHED[@]}"; do
    if [[ "$preset" == *-test-* ]]; then
        PRESETS+=("$preset")
    fi
done

if [[ ${#PRESETS[@]} -eq 0 ]]; then
    die "no test presets match '${FILTER:-*}' (see --list)"
fi

TOTAL=${#PRESETS[@]}

if [[ "$LIST_ONLY" -eq 1 ]]; then
    printf '%sMatching test presets (%d):%s\n' "$C_BOLD" "$TOTAL" "$C_RESET"
    for preset in "${PRESETS[@]}"; do
        printf '  %s%s%s\n' "$C_CYAN" "$preset" "$C_RESET"
    done
    exit 0
fi

# ---------------------------------------------------------------------------
# Sanitizer environment (user-provided values are appended last, so they win).
# ---------------------------------------------------------------------------

SUPPR="${PROJECT_ROOT}/suppr.txt"
SAN_ENV=(ASAN_OPTIONS="protect_shadow_gap=0")
if [[ -f "$SUPPR" ]]; then
    SAN_ENV+=(LSAN_OPTIONS="suppressions=$SUPPR")
else
    printf '%sWARNING:%s %s not found; LSan suppressions inactive.\n' \
        "$C_YELLOW" "$C_RESET" "$SUPPR" >&2
fi
[[ -n "${ASAN_OPTIONS:-}" ]] && SAN_ENV[0]+=":${ASAN_OPTIONS}"
if [[ -n "${LSAN_OPTIONS:-}" ]]; then
    if [[ -f "$SUPPR" ]]; then
        SAN_ENV[1]+=":${LSAN_OPTIONS}"
    else
        SAN_ENV+=(LSAN_OPTIONS="${LSAN_OPTIONS}")
    fi
fi

mkdir -p "$LOG_DIR"

printf '\n%sNovaNN — run tests%s\n' "$C_BOLD" "$C_RESET"
printf '%s%d test preset(s) · full logs: %s%s\n' \
    "$C_DIM" "$TOTAL" "$LOG_DIR" "$C_RESET"

FAILED_PRESETS=()
SKIPPED=0

for i in "${!PRESETS[@]}"; do
    preset="${PRESETS[$i]}"
    n=$((i + 1))

    printf '\n  %s[%2d/%d]%s %s▸ %s%s\n' \
        "$C_BOLD" "$n" "$TOTAL" "$C_RESET" "$C_CYAN" "$preset" "$C_RESET"

    log="$LOG_DIR/$preset.log"
    start=$(date +%s)

    if [[ ! -f "build/$preset/CMakeCache.txt" ]]; then
        SKIPPED=$((SKIPPED + 1))
        printf '%s  %s⏭ skipped%s  not configured (run scripts/build-presets.sh %s)\n' \
            "$C_CLEAR" "$C_YELLOW" "$C_RESET" "$preset"
        continue
    fi

    total_tests=$(env "${SAN_ENV[@]}" ctest --preset "$preset" "${CTEST_ARGS[@]}" -N 2>/dev/null \
                  | sed -n 's/^Total Tests: //p' | head -n1)
    total_tests=${total_tests:-0}

    if [[ -t 1 ]]; then
        env "${SAN_ENV[@]}" ctest --preset "$preset" "${CTEST_ARGS[@]}" >"$log" 2>&1 &
        run_pid=$!
        common::spinner "$run_pid" "$log" "$preset" "$n" "$TOTAL" \
            '[0-9]+/[0-9]+[ ]+Test[^[:cntrl:]]*' &
        spin_pid=$!
        if wait "$run_pid"; then rc=0; else rc=$?; fi
        wait "$spin_pid" 2>/dev/null || true
    else
        if env "${SAN_ENV[@]}" ctest --preset "$preset" "${CTEST_ARGS[@]}" >"$log" 2>&1; then
            rc=0
        else
            rc=$?
        fi
    fi

    elapsed=$(($(date +%s) - start))

    failed_names=$(awk '/The following tests FAILED:/{f=1; next}
                        f && /^[[:space:]]*[0-9]+ - / {
                            sub(/^[[:space:]]*[0-9]+ - /, "");
                            sub(/ \(.*$/, "");
                            print "    " $0
                        }' "$log")
    n_failed=$(printf '%s' "$failed_names" | grep -c . || true)

    if [[ "$rc" -eq 0 && "$n_failed" -eq 0 ]]; then
        printf '%s  %s✔ passed%s  %s%d/%d%s  %s(%s)%s\n' \
            "$C_CLEAR" "$C_GREEN" "$C_RESET" \
            "$C_BOLD" "$total_tests" "$total_tests" "$C_RESET" \
            "$C_DIM" "$(fmt_elapsed "$elapsed")" "$C_RESET"
    else
        FAILED_PRESETS+=("$preset")
        printf '%s  %s✘ FAILED%s  %s%d/%d · %d failed%s  %s(%s)%s\n' \
            "$C_CLEAR" "$C_RED" "$C_RESET" \
            "$C_BOLD" "$((total_tests - n_failed))" "$total_tests" "$n_failed" "$C_RESET" \
            "$C_DIM" "$(fmt_elapsed "$elapsed")" "$C_RESET"
        printf '%s  ── failing tests (%s/%s.log) ──%s\n' \
            "$C_YELLOW" "$LOG_DIR" "$preset" "$C_RESET" >&2
        while IFS= read -r line; do
            printf '%s\n' "$line" >&2
        done <<<"$failed_names"

        if [[ "$FORCE_STOP" -eq 1 ]]; then
            printf '\n%sAborting: --force-stop-on-failure and %s reported failures.%s\n' \
                "$C_RED" "$preset" "$C_RESET"
            exit 1
        fi
    fi
done

printf '\n'
if [[ ${#FAILED_PRESETS[@]} -eq 0 ]]; then
    msg="✔ All $TOTAL test preset(s) passed"
    [[ "$SKIPPED" -gt 0 ]] && msg+=" ($SKIPPED skipped: not configured)"
    printf '%s%s%s\n' "$C_GREEN" "$msg.$C_RESET"
    exit 0
fi

printf '%s✘ %d of %d test preset(s) failed:%s\n' \
    "$C_RED" "${#FAILED_PRESETS[@]}" "$TOTAL" "$C_RESET"
for preset in "${FAILED_PRESETS[@]}"; do
    printf '  %s✘%s %s\n' "$C_RED" "$C_RESET" "$preset"
done
printf 'Logs: %s/\n' "$LOG_DIR"
exit 1
