/**
 * @file PoolDestroyGuard_test.cpp
 * @brief Pool destroy guard gate.
 *
 * The gate pins the contract of the device pool destroy path: a pool that
 * still holds a live allocation is rejected with an error status instead of
 * being torn down under the caller's feet.
 */

#include <gtest/gtest.h>

#include <cstddef>

#include <ncore/memory/ffi.hpp>
#include <tests/common/Gpu.hpp>
#include <tests/memory/DeviceBuffer.hpp>
#include <tests/tensor/TensorUtils.hpp>

#if defined(NOVA_HAS_CUDA)
#include <cuda_runtime_api.h>
#elif defined(NOVA_HAS_HIP)
#include <hip/hip_runtime_api.h>
#endif

#if defined(NOVA_HAS_CUDA) || defined(NOVA_HAS_HIP)

using tests::memory::kTestDeviceBackend;
using tests::memory::ScopedDeviceBuffer;

TEST(PoolDestroyGuard, PendingAllocationRejected) {
  NOVA_TEST_REQUIRE_GPU();

  int deviceId = -1;
#ifdef NOVA_HAS_CUDA
  ASSERT_EQ(cudaGetDevice(&deviceId), cudaSuccess);
#else
  ASSERT_EQ(hipGetDevice(&deviceId), hipSuccess);
#endif

  DeviceMemoryPoolStats stats{};
  novaStatus_t status = devicePoolStats(kTestDeviceBackend, deviceId, &stats);
  if (status.err == novaMemoryPoolsNotNativelySupported) {
    GTEST_SKIP() << "Current device does not support native memory pools";
  }
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  ASSERT_TRUE(stats.native_pool_supported);

  constexpr std::size_t kBytes = 4096;
  ScopedDeviceBuffer buffer;
  status = buffer.reserve(kBytes);
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  ASSERT_EQ(buffer.deviceId(), deviceId);
  ASSERT_NE(buffer.get().ptr, nullptr);

  // The allocation runs on the engine stream, so the device is drained
  // before the destroy: the pool counter must reflect the live block.
#ifdef NOVA_HAS_CUDA
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
#else
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
#endif

  // The live allocation keeps the pool busy, so the destroy must be
  // rejected instead of pulling the pool out from under the buffer.
  const novaStatus_t destroyStatus =
      devicePoolDestroy(kTestDeviceBackend, deviceId);
  EXPECT_EQ(destroyStatus.err, novaResourceBusy) << destroyStatus.message;

  status = buffer.release();
  ASSERT_EQ(status.err, novaSuccess) << status.message;

  // The free is enqueued on the engine stream, so the pool only reports
  // itself idle once the device has drained.
#ifdef NOVA_HAS_CUDA
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
#else
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
#endif

  DeviceMemoryPoolStats after{};
  status = devicePoolStats(kTestDeviceBackend, deviceId, &after);
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  EXPECT_EQ(after.used_current, 0U);
}

#endif // NOVA_HAS_CUDA || NOVA_HAS_HIP
