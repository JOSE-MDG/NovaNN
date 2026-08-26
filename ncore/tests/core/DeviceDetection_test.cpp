/**
 * @file DeviceDetection_test.cpp
 * @brief Unit tests for GPU backend detection.
 *
 * The expectations follow is_device_available() in @ref device.h:
 * a backend not compiled in (NOVA_HAS_CUDA / NOVA_HAS_HIP) always
 * reports @c false without querying the runtime, while a compiled-in
 * backend performs a real runtime probe whose result is cached.
 *
 * A compiled-in probe depends on whether the executing machine actually
 * exposes a device, which is an environment property rather than a code
 * defect: those tests skip when no device answers instead of hard
 * failing on GPU-less runners (e.g. CI).
 */

#include <gtest/gtest.h>
#include <ncore/core/device.h>

#ifdef NOVA_HAS_CUDA
/**
 * @brief Verifies CUDA is detected in a NOVA_HAS_CUDA build.
 * @test is_cuda_available() probes the CUDA runtime via
 *       is_device_available(CUDA_DEVICE, false); a visible device must
 *       be found. Without one the test skips rather than fails.
 */
TEST(DeviceDetection, CudaDetectionWorks) {
  if (!is_cuda_available()) {
    GTEST_SKIP() << "No CUDA device visible to this runner";
  }
  SUCCEED();
}

/**
 * @brief Verifies HIP stays unavailable in a CUDA-only build.
 * @test HIP is not compiled in, so is_hip_available() returns @c false
 *       without querying the runtime (device.h); deterministic either
 *       way, so there is no skip path.
 */
TEST(DeviceDetection, HipUnavailableInCudaOnlyBuild) {
  EXPECT_EQ(is_hip_available(), false);
}
#elif defined(NOVA_HAS_HIP)
/**
 * @brief Verifies CUDA stays unavailable in a HIP-only build.
 * @test CUDA is not compiled in, so is_cuda_available() returns @c
 *       false without querying the runtime (device.h); deterministic
 *       either way, so there is no skip path.
 */
TEST(DeviceDetection, CudaUnavailableInHipOnlyBuild) {
  EXPECT_EQ(is_cuda_available(), false);
}

/**
 * @brief Verifies HIP is detected in a NOVA_HAS_HIP build.
 * @test is_hip_available() probes the HIP runtime via
 *       is_device_available(HIP_DEVICE, false); a visible device must
 *       be found. Without one the test skips rather than fails.
 */
TEST(DeviceDetection, HipDetectionWorks) {
  if (!is_hip_available()) {
    GTEST_SKIP() << "No HIP device visible to this runner";
  }
  SUCCEED();
}
#elif !defined(NOVA_HAS_CUDA) and !defined(NOVA_HAS_HIP)
/**
 * @brief Verifies CUDA is not detected without GPU backends.
 * @test CUDA is not compiled in, so is_device_available() returns
 *       @c false without querying the runtime (device.h).
 */
TEST(DeviceDetection, CudaUnavailableWithoutBackends) {
  EXPECT_EQ(is_cuda_available(), false);
}

/**
 * @brief Verifies HIP is not detected without GPU backends.
 * @test HIP is not compiled in, so is_device_available() returns
 *       @c false without querying the runtime (device.h).
 */
TEST(DeviceDetection, HipUnavailableWithoutBackends) {
  EXPECT_EQ(is_hip_available(), false);
}
#endif
