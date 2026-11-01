/**
 * @file MemoryPoolTerminal_test.cpp
 * @brief Verifies that a destroyed device pool rejects later operations.
 */

#include <gtest/gtest.h>

#include <cstdint>

#include <ncore/core/status.h>

#include <ncore/memory/ffi.hpp>
#include <tests/common/Gpu.hpp>
#include <tests/memory/DeviceBuffer.hpp>
#include <tests/tensor/TensorUtils.hpp>

#if defined(NOVA_HAS_CUDA) || defined(NOVA_HAS_HIP)

using tests::memory::ScopedDeviceBuffer;
using tests::memory::kTestDeviceBackend;

TEST(MemoryPoolTerminal, DestroyedPoolRejectsSubsequentOperations) {
  NOVA_TEST_REQUIRE_GPU();

  ScopedDeviceBuffer buffer;
  novaStatus_t status = buffer.reserve(256);
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  ASSERT_NE(buffer.get().ptr, nullptr);

  // The reservation lands on the device the runtime has active, so the
  // test destroys that pool instead of assuming ordinal zero.
  const std::int32_t deviceId = buffer.deviceId();

  DeviceMemoryPoolStats stats{};
  status = devicePoolStats(kTestDeviceBackend, deviceId, &stats);
  if (status.err == novaMemoryPoolsNotNativelySupported) {
    GTEST_SKIP() << "Current device does not support native memory pools";
  }
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  ASSERT_TRUE(stats.native_pool_supported);

  status = devicePoolDestroy(kTestDeviceBackend, deviceId);
  EXPECT_EQ(status.err, novaResourceBusy) << status.message;

  status = buffer.release();
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  EXPECT_EQ(buffer.get().ptr, nullptr);

  status = devicePoolDestroy(kTestDeviceBackend, deviceId);
  ASSERT_EQ(status.err, novaSuccess) << status.message;

  DeviceMemoryPoolStats afterDestroy{};
  status = devicePoolStats(kTestDeviceBackend, deviceId, &afterDestroy);
  EXPECT_EQ(status.err, novaInvalidResourceHandle) << status.message;

  status = devicePoolSetReleaseThreshold(kTestDeviceBackend, deviceId, 4096);
  EXPECT_EQ(status.err, novaInvalidResourceHandle) << status.message;

  status = deviceEmptyCache(kTestDeviceBackend, deviceId, 0);
  EXPECT_EQ(status.err, novaInvalidResourceHandle) << status.message;

  status = devicePoolDestroy(kTestDeviceBackend, deviceId);
  EXPECT_EQ(status.err, novaInvalidResourceHandle) << status.message;

  deviceBuffer_t afterDestroyBuffer{};
  std::int32_t afterDestroyDeviceId = -1;
  status = deviceReserve(64, &afterDestroyBuffer, false, kTestDeviceBackend,
                         &afterDestroyDeviceId);
  EXPECT_EQ(status.err, novaInvalidResourceHandle) << status.message;
  EXPECT_EQ(afterDestroyBuffer.ptr, nullptr);
}

#endif
