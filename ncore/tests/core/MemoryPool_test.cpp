/**
 * @file MemoryPool_test.cpp
 * @brief Tests for per-device GPU pools and stream-aware tensor layouts.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/core/storage.h>
#include <ncore/tensor.h>

#include <ncore/memory/ffi.hpp>

#include <tests/common/Gpu.hpp>
#include <tests/gpu/Stream.hpp>
#include <tests/memory/DeviceBuffer.hpp>
#include <tests/memory/PoolStats.hpp>
#include <tests/tensor/TensorUtils.hpp>

namespace {

#if defined(NOVA_HAS_CUDA) || defined(NOVA_HAS_HIP)

using tests::memory::ScopedDeviceBuffer;
using tests::memory::getPoolStats;
using tests::memory::kTestDeviceBackend;

using namespace tests::gpu;

class ScopedTensor final {
public:
  explicit ScopedTensor(void *stream) : stream_(stream) {}
  ScopedTensor(const ScopedTensor &) = delete;
  ScopedTensor &operator=(const ScopedTensor &) = delete;

  ~ScopedTensor() {
    if (tensor_.storage != nullptr) {
      (void)collect(&tensor_);
    }
  }

  Tensor &get() { return tensor_; }
  [[nodiscard]] const Tensor &get() const { return tensor_; }

private:
  Tensor tensor_{};
  /// Retained to document that collection ignores the caller stream.
  [[maybe_unused]] void *stream_ = nullptr;
};

class ScopedThresholdRestore final {
public:
  ScopedThresholdRestore(std::int32_t device_id, std::uint64_t threshold)
      : device_id_(device_id), threshold_(threshold) {}
  ScopedThresholdRestore(const ScopedThresholdRestore &) = delete;
  ScopedThresholdRestore &operator=(const ScopedThresholdRestore &) = delete;

  ~ScopedThresholdRestore() {
    if (active_) {
      (void)devicePoolSetReleaseThreshold(kTestDeviceBackend, device_id_, threshold_);
    }
  }

  void activate() { active_ = true; }

private:
  std::int32_t device_id_;
  std::uint64_t threshold_;
  bool active_ = false;
};

#endif

} // namespace

#if defined(NOVA_HAS_CUDA) || defined(NOVA_HAS_HIP)

TEST(MemoryPool, ReserveResizeAndReleasePreserveBytesOnCallerStream) {
  NOVA_TEST_REQUIRE_GPU();
  ScopedStream stream;
  ASSERT_EQ(stream.create(), kRuntimeSuccess);

  int deviceId = -1;
  ASSERT_EQ(runtimeGetDevice(&deviceId), kRuntimeSuccess);

  DeviceMemoryPoolStats before{};
  novaStatus_t status = getPoolStats(deviceId, &before);
  if (status.err == novaMemoryPoolsNotNativelySupported) {
    GTEST_SKIP() << "Current device does not support native memory pools";
  }
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  ASSERT_TRUE(before.native_pool_supported);

  ScopedDeviceBuffer buffer(stream.opaque());
  constexpr std::array<std::uint8_t, 12> expected{
      0x11, 0x21, 0x31, 0x41, 0x51, 0x61, 0x71, 0x81, 0x91, 0xa1, 0xb1, 0xc1};
  std::array<std::uint8_t, expected.size()> actual{};

  status = buffer.reserve(expected.size());
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  ASSERT_EQ(buffer.deviceId(), deviceId);
  ASSERT_NE(buffer.get().ptr, nullptr);
  ASSERT_EQ(buffer.get().bytes, expected.size());

  status = deviceTransfer(expected.data(), buffer.get().ptr,
                          deviceMemcpyHostToDevice, expected.size(),
                          stream.opaque());
  ASSERT_EQ(status.err, novaSuccess) << status.message;

  status = buffer.resize(2 * expected.size());
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  ASSERT_EQ(buffer.get().bytes, 2 * expected.size());

  status =
      deviceTransfer(buffer.get().ptr, actual.data(), deviceMemcpyDeviceToHost,
                     actual.size(), stream.opaque());
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  ASSERT_EQ(runtimeStreamSynchronize(stream.get()), kRuntimeSuccess);
  EXPECT_EQ(actual, expected);

  DeviceMemoryPoolStats allocated{};
  status = getPoolStats(deviceId, &allocated);
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  EXPECT_GT(allocated.used_current, before.used_current);

  status = buffer.release();
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  ASSERT_EQ(runtimeStreamSynchronize(stream.get()), kRuntimeSuccess);

  DeviceMemoryPoolStats released{};
  status = getPoolStats(deviceId, &released);
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  EXPECT_EQ(released.used_current, before.used_current);
}

TEST(MemoryPool, PinnedHostAllocationStaysOutsideDevicePool) {
  NOVA_TEST_REQUIRE_GPU();
  int deviceId = -1;
  ASSERT_EQ(runtimeGetDevice(&deviceId), kRuntimeSuccess);

  DeviceMemoryPoolStats before{};
  const novaStatus_t beforeStatus = getPoolStats(deviceId, &before);
  const bool hasNativePool =
      beforeStatus.err == novaSuccess && before.native_pool_supported;
  ASSERT_TRUE(hasNativePool ||
              beforeStatus.err == novaMemoryPoolsNotNativelySupported)
      << beforeStatus.message;

  ScopedDeviceBuffer pinnedBuffer;
  constexpr std::array<std::uint8_t, 8> expected{0x08, 0x18, 0x28, 0x38,
                                                 0x48, 0x58, 0x68, 0x78};
  const novaStatus_t reserveStatus =
      pinnedBuffer.reserve(expected.size(), true);
  ASSERT_EQ(reserveStatus.err, novaSuccess) << reserveStatus.message;
  ASSERT_TRUE(pinnedBuffer.get().isPinned);
  ASSERT_NE(pinnedBuffer.get().ptr, nullptr);

  auto *hostBytes = static_cast<std::uint8_t *>(pinnedBuffer.get().ptr);
  std::copy(expected.begin(), expected.end(), hostBytes);
  EXPECT_TRUE(std::equal(expected.begin(), expected.end(), hostBytes));

  if (hasNativePool) {
    DeviceMemoryPoolStats allocated{};
    const novaStatus_t allocatedStatus = getPoolStats(deviceId, &allocated);
    ASSERT_EQ(allocatedStatus.err, novaSuccess) << allocatedStatus.message;
    EXPECT_EQ(allocated.used_current, before.used_current);
  }

  const novaStatus_t releaseStatus = pinnedBuffer.release();
  ASSERT_EQ(releaseStatus.err, novaSuccess) << releaseStatus.message;

  if (hasNativePool) {
    DeviceMemoryPoolStats released{};
    const novaStatus_t releasedStatus = getPoolStats(deviceId, &released);
    ASSERT_EQ(releasedStatus.err, novaSuccess) << releasedStatus.message;
    EXPECT_EQ(released.used_current, before.used_current);
  }
}

TEST(MemoryPool, ReleaseThresholdAndTrimPreserveLiveAllocations) {
  NOVA_TEST_REQUIRE_GPU();
  int deviceId = -1;
  ASSERT_EQ(runtimeGetDevice(&deviceId), kRuntimeSuccess);

  DeviceMemoryPoolStats before{};
  novaStatus_t status = getPoolStats(deviceId, &before);
  if (status.err == novaMemoryPoolsNotNativelySupported) {
    GTEST_SKIP() << "Current device does not support native memory pools";
  }
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  ASSERT_TRUE(before.native_pool_supported);

  ScopedThresholdRestore restore(deviceId, before.release_threshold);
  constexpr std::uint64_t threshold = 2U * 1024U * 1024U;
  status = devicePoolSetReleaseThreshold(kTestDeviceBackend, deviceId, threshold);
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  restore.activate();

  DeviceMemoryPoolStats configured{};
  status = getPoolStats(deviceId, &configured);
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  EXPECT_EQ(configured.release_threshold, threshold);

  ScopedDeviceBuffer buffer;
  constexpr std::array<std::uint8_t, 8> expected{0x12, 0x34, 0x56, 0x78,
                                                 0x9a, 0xbc, 0xde, 0xf0};
  std::array<std::uint8_t, expected.size()> actual{};
  status = buffer.reserve(4096);
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  ASSERT_EQ(buffer.deviceId(), deviceId);
  status = deviceTransfer(expected.data(), buffer.get().ptr,
                          deviceMemcpyHostToDevice, expected.size());
  ASSERT_EQ(status.err, novaSuccess) << status.message;

  status = deviceEmptyCache(kTestDeviceBackend, deviceId, 0);
  ASSERT_EQ(status.err, novaSuccess) << status.message;

  DeviceMemoryPoolStats live{};
  status = getPoolStats(deviceId, &live);
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  EXPECT_GT(live.used_current, before.used_current);
  EXPECT_EQ(live.release_threshold, threshold);

  status = deviceTransfer(buffer.get().ptr, actual.data(),
                          deviceMemcpyDeviceToHost, actual.size());
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  EXPECT_EQ(actual, expected);

  status = buffer.release();
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  status = deviceEmptyCache(kTestDeviceBackend, deviceId, 0);
  ASSERT_EQ(status.err, novaSuccess) << status.message;

  DeviceMemoryPoolStats idle{};
  status = getPoolStats(deviceId, &idle);
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  EXPECT_EQ(idle.used_current, before.used_current);
}

TEST(MemoryPool, DestroyRejectsLivePoolAllocations) {
  NOVA_TEST_REQUIRE_GPU();
  int deviceId = -1;
  ASSERT_EQ(runtimeGetDevice(&deviceId), kRuntimeSuccess);

  DeviceMemoryPoolStats stats{};
  novaStatus_t status = getPoolStats(deviceId, &stats);
  if (status.err == novaMemoryPoolsNotNativelySupported) {
    GTEST_SKIP() << "Current device does not support native memory pools";
  }
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  ASSERT_TRUE(stats.native_pool_supported);

  ScopedDeviceBuffer buffer;
  status = buffer.reserve(4096);
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  ASSERT_EQ(buffer.deviceId(), deviceId);

  status = devicePoolDestroy(kTestDeviceBackend, deviceId);
  EXPECT_EQ(status.err, novaResourceBusy) << status.message;

  const novaStatus_t releaseStatus = buffer.release();
  ASSERT_EQ(releaseStatus.err, novaSuccess) << releaseStatus.message;

  DeviceMemoryPoolStats afterRelease{};
  status = getPoolStats(deviceId, &afterRelease);
  EXPECT_EQ(status.err, novaSuccess) << status.message;
}

TEST(MemoryPool, UnsupportedNativePoolUsesDirectAllocationFallback) {
  NOVA_TEST_REQUIRE_GPU();
  int deviceId = -1;
  ASSERT_EQ(runtimeGetDevice(&deviceId), kRuntimeSuccess);

  DeviceMemoryPoolStats stats{};
  const novaStatus_t poolStatus = getPoolStats(deviceId, &stats);
  if (poolStatus.err == novaSuccess && stats.native_pool_supported) {
    GTEST_SKIP() << "Current device supports native pools; direct fallback is "
                    "not selectable on this device";
  }
  ASSERT_EQ(poolStatus.err, novaMemoryPoolsNotNativelySupported)
      << poolStatus.message;
  EXPECT_FALSE(stats.native_pool_supported);

  ScopedDeviceBuffer buffer;
  constexpr std::array<std::uint8_t, 8> expected{0x14, 0x25, 0x36, 0x47,
                                                 0x58, 0x69, 0x7a, 0x8b};
  std::array<std::uint8_t, expected.size()> actual{};
  novaStatus_t status = buffer.reserve(expected.size());
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  ASSERT_EQ(buffer.deviceId(), deviceId);

  status = deviceTransfer(expected.data(), buffer.get().ptr,
                          deviceMemcpyHostToDevice, expected.size());
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  status = buffer.resize(2 * expected.size());
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  status = deviceTransfer(buffer.get().ptr, actual.data(),
                          deviceMemcpyDeviceToHost, actual.size());
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  EXPECT_EQ(actual, expected);

  status = buffer.release();
  EXPECT_EQ(status.err, novaSuccess) << status.message;
}

TEST(MemoryPool, ContiguousOnEngineStreamDensifiesTransposedGpuTensor) {
  NOVA_TEST_REQUIRE_GPU();

  constexpr shape_t inputShape{2, 3};
  constexpr std::array<float32, 6> inputValues{1.0F, 2.0F, 3.0F,
                                               4.0F, 5.0F, 6.0F};
  constexpr std::array<float32, 6> expected{1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F};
  std::array<float32, expected.size()> actual{};

  ScopedTensor input(nullptr);
  novaStatus_t status{};
  input.get() =
      create_tensor(inputShape, Float32, DEVICE_GPU, false, false, 2, &status);
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  ASSERT_NE(input.get().data.v, nullptr);

  // The whole pipeline runs on the device engine stream: the tensor
  // allocations, the transpose kernel, the contiguous kernel, and both
  // transfers. A null stream selects the engine stream, whose path drains
  // before returning.
  status = deviceTransfer(inputValues.data(), input.get().data.v,
                          deviceMemcpyHostToDevice,
                          inputValues.size() * sizeof(float32), nullptr);
  ASSERT_EQ(status.err, novaSuccess) << status.message;

  ScopedTensor transposed(nullptr);
  transposed.get() = transpose(&input.get(), &status, 0, 1);
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  ASSERT_FALSE(is_contiguous(&transposed.get()));

  ScopedTensor dense(nullptr);
  dense.get() = contiguous(&transposed.get(), &status);
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  ASSERT_TRUE(is_contiguous(&dense.get()));
  ASSERT_EQ(dense.get().shape[0], 3U);
  ASSERT_EQ(dense.get().shape[1], 2U);
  ASSERT_NE(dense.get().data.v, nullptr);

  status = deviceTransfer(dense.get().data.v, actual.data(),
                          deviceMemcpyDeviceToHost,
                          actual.size() * sizeof(float32), nullptr);
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  EXPECT_EQ(actual, expected);
}

TEST(MemoryPool, EmptyCachePreservesLiveAllocations) {
  NOVA_TEST_REQUIRE_GPU();

  int deviceId = -1;
  ASSERT_EQ(runtimeGetDevice(&deviceId), kRuntimeSuccess);

  DeviceMemoryPoolStats before{};
  novaStatus_t status = getPoolStats(deviceId, &before);
  if (status.err == novaMemoryPoolsNotNativelySupported) {
    GTEST_SKIP() << "Current device does not support native memory pools";
  }
  ASSERT_EQ(status.err, novaSuccess) << status.message;

  constexpr std::size_t kBytes = 4096;
  ScopedDeviceBuffer buffer;
  status = buffer.reserve(kBytes);
  ASSERT_EQ(status.err, novaSuccess) << status.message;

  const std::vector<float> expected(kBytes / sizeof(float), 3.5F);
  status = deviceTransfer(expected.data(), buffer.get().ptr,
                          deviceMemcpyHostToDevice, kBytes, nullptr);
  ASSERT_EQ(status.err, novaSuccess) << status.message;

  // The trim only reclaims the backing no live allocation holds, so the
  // buffer keeps its contents.
  const novaStatus_t trim = empty_cache(deviceId, 0);
  EXPECT_EQ(trim.err, novaSuccess) << trim.message;

  std::vector<float> actual(expected.size(), 0.0F);
  status = deviceTransfer(buffer.get().ptr, actual.data(),
                          deviceMemcpyDeviceToHost, kBytes, nullptr);
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  EXPECT_EQ(actual, expected);

  status = buffer.release();
  EXPECT_EQ(status.err, novaSuccess) << status.message;
}

#endif
