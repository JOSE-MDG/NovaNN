#!/usr/bin/env bash
# Run ctest for every *-test-* preset (or the subset matched by FILTER).

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
common::init
common::install_err_trap

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

Full output is written to ${C_BOLD}build/logs/tests/<preset>.log${C_RESET}. Exit status:
0 = all executed presets passed, 1 = failures found, 2 = usage error.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -l|--list) LIST_ONLY=1 ;;
        --force-stop-on-failure) FORCE_STOP=1 ;;
        --)
            shift
            CTEST_ARGS=("$@")
            break
            ;;
        -*) usage_error "unknown option: $1 (forward ctest flags after '--')" ;;
        *)
            [[ -z "$FILTER" ]] || usage_error "only one FILTER is allowed"
            FILTER="$1"
            ;;
    esac
    shift
done

mapfile -t ALL_PRESETS < <(common::list_presets)
if [[ -n "$FILTER" ]]; then
    mapfile -t ALL_PRESETS < <(printf '%s\n' "${ALL_PRESETS[@]}" | common::select_presets "$FILTER")
fi
mapfile -t PRESETS < <(printf '%s\n' "${ALL_PRESETS[@]}" | grep -E -- '-test-')

if (( ${#PRESETS[@]} == 0 )); then
    die "no test presets match '${FILTER:-*}' (see --list)"
fi

TOTAL=${#PRESETS[@]}

if (( LIST_ONLY )); then
    printf '%sMatching test presets (%d):%s\n' "$C_BOLD" "$TOTAL" "$C_RESET"
    for preset in "${PRESETS[@]}"; do
        printf '  %s%s%s\n' "$C_CYAN" "$preset" "$C_RESET"
    done
    exit 0
fi

# Sanitizer environment. User-provided values are appended last, so they win.
SUPPR="${PROJECT_ROOT}/suppr.txt"
SAN_ENV=(ASAN_OPTIONS="protect_shadow_gap=0")
[[ -n "${ASAN_OPTIONS:-}" ]] && SAN_ENV[0]+=":${ASAN_OPTIONS}"

if [[ -f "$SUPPR" ]]; then
    SAN_ENV+=(LSAN_OPTIONS="suppressions=$SUPPR")
    [[ -n "${LSAN_OPTIONS:-}" ]] && SAN_ENV[1]+=":${LSAN_OPTIONS}"
else
    printf '%sWARNING:%s %s not found; LSan suppressions inactive.\n' \
        "$C_YELLOW" "$C_RESET" "$SUPPR" >&2
    [[ -n "${LSAN_OPTIONS:-}" ]] && SAN_ENV+=(LSAN_OPTIONS="$LSAN_OPTIONS")
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
        wait "$run_pid" && rc=0 || rc=$?
        wait "$spin_pid" 2>/dev/null || true
    else
        env "${SAN_ENV[@]}" ctest --preset "$preset" "${CTEST_ARGS[@]}" >"$log" 2>&1 && rc=0 || rc=$?
    fi

    elapsed=$(( $(date +%s) - start ))

    mapfile -t failed_names < <(awk '/The following tests FAILED:/{f=1; next}
                                     f && /^[[:space:]]*[0-9]+ - / {
                                         sub(/^[[:space:]]*[0-9]+ - /, "");
                                         sub(/ \(.*$/, "");
                                         print "    " $0
                                     }' "$log")
    n_failed=${#failed_names[@]}

    if (( rc == 0 && n_failed == 0 )); then
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
        for line in "${failed_names[@]}"; do
            printf '%s\n' "$line" >&2
        done

        if (( FORCE_STOP )); then
            printf '\n%sAborting: --force-stop-on-failure and %s reported failures.%s\n' \
                "$C_RED" "$preset" "$C_RESET"
            exit 1
        fi
    fi
done

printf '\n'
if (( ${#FAILED_PRESETS[@]} == 0 )); then
    msg="✔ All $TOTAL test preset(s) passed"
    (( SKIPPED > 0 )) && msg+=" ($SKIPPED skipped: not configured)"
    printf '%s%s%s\n' "$C_GREEN" "$msg" "$C_RESET"
    exit 0
fi

printf '%s✘ %d of %d test preset(s) failed:%s\n' \
    "$C_RED" "${#FAILED_PRESETS[@]}" "$TOTAL" "$C_RESET"
for preset in "${FAILED_PRESETS[@]}"; do
    printf '  %s✘%s %s\n' "$C_RED" "$C_RESET" "$preset"
done
printf 'Logs: %s/\n' "$LOG_DIR"
exit 1
