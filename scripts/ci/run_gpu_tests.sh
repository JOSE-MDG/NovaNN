#!/usr/bin/env bash
# Shared steps of the GPU simulator workflow (.github/workflows/gpu-sim.yml).
#
# Usage: scripts/ci/run_gpu_tests.sh <cuda|hip> <deps|test>
#
#   deps   Install the build dependencies (GCC 15, Ninja, GoogleTest, uv and
#          clang-format). Meant for a hosted ubuntu-24.04 runner.
#   test   Run the GPU tests of the build in ./build on the simulated GPU and
#          fail on a failing test, on a simulator error, or if a test did not
#          run on the device.

set -Eeuo pipefail

usage() {
    echo "Usage: $0 <cuda|hip> <deps|test>" >&2
    exit 2
}

[[ $# -eq 2 ]] || usage

BACKEND="$1"
ACTION="$2"

# The only per-backend difference: how a test reports a missing device.
case "$BACKEND" in
    cuda) NO_DEVICE="No CUDA device" ;;
    hip) NO_DEVICE="No HIP device" ;;
    *) usage ;;
esac

install_deps() {
    # The build needs GCC 15 (Ubuntu 24.04 ships 13), GoogleTest and Ninja.
    # The codegen step runs through uv and calls clang-format.
    sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
    sudo apt-get update
    sudo apt-get install -y gcc-15 g++-15 ninja-build libgtest-dev
    python3 -m venv "$RUNNER_TEMP/venv"
    "$RUNNER_TEMP/venv/bin/pip" install uv clang-format==23.1.5
    echo "$RUNNER_TEMP/venv/bin" >> "$GITHUB_PATH"
}

run_tests() {
    local rc=0

    # Only the tests that need a device. A failing ctest must not stop the
    # script before the guards below.
    vgpu run ctest --test-dir build --no-tests=error -V \
        -R '^(DeviceDetection|RngGpuParity|GpuDispatch|GpuSaturation|TensorTransfer)\.|^MemoryAllocator\.Device|Gpu(OracleEquivalence|SpecialValues)' \
        > ctest.log 2>&1 || rc=$?

    # Reporting only: none of these may end the script early.
    grep -E "\[  FAILED  \]|\*\*\*Failed" ctest.log | head -n 40 || true
    tail -n 5 ctest.log || true
    grep -E "tests passed|tests failed" ctest.log || true

    # The allocation limit test provokes an expected out-of-memory line.
    if grep "VirtualGPU error" ctest.log | grep -v "out-of-memory"; then
        echo "The simulator reported an error"
        exit 1
    fi
    if grep -E "$NO_DEVICE|GPU backend/device not available|no GPU backend compiled in" ctest.log; then
        echo "A device test did not run on the simulated GPU"
        exit 1
    fi
    exit "$rc"
}

case "$ACTION" in
    deps) install_deps ;;
    test) run_tests ;;
    *) usage ;;
esac
