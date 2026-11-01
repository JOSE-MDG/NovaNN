/**
 * @file Gpu.hpp
 * @brief GPU availability guard shared by the device test suites.
 */

#pragma once

#include <gtest/gtest.h>

#include <ncore/core/device.h>

namespace tests::common {

/**
 * @brief True when a GPU backend is compiled in and a device is present.
 *
 * Probing also warms the one-shot device-detection cache consulted by
 * the copy and transfer paths.
 */
inline bool gpuAvailable() {
#if defined(NOVA_HAS_CUDA) || defined(NOVA_HAS_HIP)
  return is_cuda_available() || is_hip_available();
#else
  return false;
#endif
}

} // namespace tests::common

/**
 * @brief Skips the current test when no usable GPU backend/device.
 *
 * Must be invoked as the first statement of a GPU-guarded test body so
 * `GTEST_SKIP` returns from the test itself.
 */
#define NOVA_TEST_REQUIRE_GPU()                                                \
  do {                                                                         \
    if (!tests::common::gpuAvailable()) {                                      \
      GTEST_SKIP() << "GPU backend/device not available";                      \
    }                                                                          \
  } while (0)
