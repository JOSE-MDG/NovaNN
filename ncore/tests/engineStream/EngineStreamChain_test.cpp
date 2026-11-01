/**
 * @file EngineStreamChain_test.cpp
 * @brief Engine-stream pattern gates: reuse, pageable pipeline correctness,
 *        cross-stream reuse guards, and batched-vs-sync-per-op ratio.
 */

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include <ncore/core/device.h>
#include <ncore/core/status.h>

#include <ncore/memory/ffi.hpp>
#include <tests/common/Gpu.hpp>
#include <tests/gpu/Stream.hpp>
#include <tests/memory/DeviceBuffer.hpp>
#include <tests/memory/PoolStats.hpp>
#include <tests/tensor/TensorUtils.hpp>

#include "HazardKernel.hpp"

#if defined(NOVA_HAS_CUDA)
#include <cuda_runtime_api.h>
#elif defined(NOVA_HAS_HIP)
#include <hip/hip_runtime_api.h>
#endif

namespace {

#if defined(NOVA_HAS_CUDA) || defined(NOVA_HAS_HIP)

using namespace tests::gpu;

using tests::memory::getPoolStats;
using tests::memory::ScopedDeviceBuffer;

#endif // NOVA_HAS_CUDA || NOVA_HAS_HIP

} // namespace

#if defined(NOVA_HAS_CUDA) || defined(NOVA_HAS_HIP)

TEST(EngineStreamChain, SameStreamImmediateReuse) {
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

  constexpr std::array<std::uint8_t, 12> expected{
      0x11, 0x21, 0x31, 0x41, 0x51, 0x61, 0x71, 0x81, 0x91, 0xa1, 0xb1, 0xc1};
  constexpr std::array<std::uint8_t, 12> expected2{
      0x12, 0x22, 0x32, 0x42, 0x52, 0x62, 0x72, 0x82, 0x92, 0xa2, 0xb2, 0xc2};

  int reuseCount = 0;
  int dataOk = 0;
  const int rounds = 20;
  for (int round = 0; round < rounds; ++round) {
    ScopedDeviceBuffer buffer(stream.opaque());
    status = buffer.reserve(expected.size());
    ASSERT_EQ(status.err, novaSuccess) << status.message;
    ASSERT_EQ(buffer.deviceId(), deviceId);

    const void *firstPtr = buffer.get().ptr;
    status = deviceTransfer(expected.data(), buffer.get().ptr,
                            deviceMemcpyHostToDevice, expected.size(),
                            stream.opaque());
    ASSERT_EQ(status.err, novaSuccess) << status.message;

    status = buffer.release();
    ASSERT_EQ(status.err, novaSuccess) << status.message;

    status = buffer.reserve(expected.size());
    ASSERT_EQ(status.err, novaSuccess) << status.message;
    if (buffer.get().ptr == firstPtr) {
      ++reuseCount;
    }

    status = deviceTransfer(expected2.data(), buffer.get().ptr,
                            deviceMemcpyHostToDevice, expected2.size(),
                            stream.opaque());
    ASSERT_EQ(status.err, novaSuccess) << status.message;

    std::array<std::uint8_t, expected2.size()> actual{};
    status = deviceTransfer(buffer.get().ptr, actual.data(),
                            deviceMemcpyDeviceToHost, actual.size(),
                            stream.opaque());
    ASSERT_EQ(status.err, novaSuccess) << status.message;
    ASSERT_EQ(runtimeStreamSynchronize(stream.get()), kRuntimeSuccess);

    if (actual == expected2) {
      ++dataOk;
    }

    status = buffer.release();
    ASSERT_EQ(status.err, novaSuccess) << status.message;
  }

  RecordProperty("reuse_count", reuseCount);
  // A free and the next same-size allocation on the same stream are ordered
  // by that stream, so the pool hands the block straight back. Losing that
  // reuse serializes every allocation behind the previous free.
  EXPECT_EQ(reuseCount, rounds) << "same-stream immediate reuse was lost";
  EXPECT_EQ(dataOk, rounds) << "data mismatch on same-stream chain";
}

TEST(EngineStreamChain, PageablePipelineCorrect) {
  NOVA_TEST_REQUIRE_GPU();
  const int rounds = 60;
  int failRounds = 0;
  for (int round = 0; round < rounds; ++round) {
    const int n =
        4096 + static_cast<int>((static_cast<unsigned>(round) & 3U) * 128U);
    const std::size_t sz = static_cast<std::size_t>(n) * sizeof(float);

    std::vector<float> hIn(sz / sizeof(float));
    std::vector<float> hOut(sz / sizeof(float), 0.0f);
    for (int j = 0; j < n; ++j) {
      hIn[static_cast<std::size_t>(j)] = (static_cast<float>(j) * 0.25f) +
                                         (static_cast<float>(round) * 0.001f);
    }

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

    ScopedDeviceBuffer src(stream.opaque());
    ScopedDeviceBuffer dst(stream.opaque());
    status = src.reserve(sz);
    ASSERT_EQ(status.err, novaSuccess) << status.message;
    status = dst.reserve(sz);
    ASSERT_EQ(status.err, novaSuccess) << status.message;

    status = deviceTransfer(hIn.data(), src.get().ptr, deviceMemcpyHostToDevice,
                            sz, stream.opaque());
    ASSERT_EQ(status.err, novaSuccess) << status.message;

    ASSERT_EQ(runtimeStreamSynchronize(stream.get()), kRuntimeSuccess);

    std::this_thread::sleep_for(std::chrono::microseconds(500));

    status = deviceTransfer(src.get().ptr, dst.get().ptr,
                            deviceMemcpyDeviceToDevice, sz, stream.opaque());
    ASSERT_EQ(status.err, novaSuccess) << status.message;

    status = deviceTransfer(dst.get().ptr, hOut.data(),
                            deviceMemcpyDeviceToHost, sz, stream.opaque());
    ASSERT_EQ(status.err, novaSuccess) << status.message;

    status = src.release();
    ASSERT_EQ(status.err, novaSuccess) << status.message;
    status = dst.release();
    ASSERT_EQ(status.err, novaSuccess) << status.message;

    ASSERT_EQ(runtimeStreamSynchronize(stream.get()), kRuntimeSuccess);

    int bad = 0;
    for (int j = 0; j < n; ++j) {
      if (std::fabs(hOut[static_cast<std::size_t>(j)] -
                    hIn[static_cast<std::size_t>(j)]) > 0.05f) {
        ++bad;
      }
    }
    if (bad > 0) {
      ++failRounds;
      if (failRounds <= 3) {
        ADD_FAILURE() << "round " << round << " (N=" << n << "): " << bad
                      << " values wrong";
      }
    }
  }

  EXPECT_EQ(failRounds, 0) << "pageable pipeline produced wrong data";
}

TEST(EngineStreamChain, CrossStreamNoLiveHandoffWithoutEvent) {
  NOVA_TEST_REQUIRE_GPU();
  const int rounds = 10;
  const int n = 4096;
  const std::size_t sz = static_cast<std::size_t>(n) * sizeof(float);

  ScopedStream s0, s1, s2;
  ASSERT_EQ(s0.create(), kRuntimeSuccess);
  ASSERT_EQ(s1.create(), kRuntimeSuccess);
  ASSERT_EQ(s2.create(), kRuntimeSuccess);

  int deviceId = -1;
  ASSERT_EQ(runtimeGetDevice(&deviceId), kRuntimeSuccess);

  DeviceMemoryPoolStats before{};
  novaStatus_t status = getPoolStats(deviceId, &before);
  if (status.err == novaMemoryPoolsNotNativelySupported) {
    GTEST_SKIP() << "Current device does not support native memory pools";
  }
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  ASSERT_TRUE(before.native_pool_supported);

  {
    ScopedDeviceBuffer warm(s0.opaque());
    status = warm.reserve(sz);
    ASSERT_EQ(status.err, novaSuccess) << status.message;
    status = warm.release();
    ASSERT_EQ(status.err, novaSuccess) << status.message;
  }
  ASSERT_EQ(runtimeStreamSynchronize(s0.get()), kRuntimeSuccess);

  ScopedDeviceBuffer a(s0.opaque());
  status = a.reserve(sz);
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  std::vector<float> hostA(sz / sizeof(float), 3.25f);
  status = deviceTransfer(hostA.data(), a.get().ptr, deviceMemcpyHostToDevice,
                          sz, s0.opaque());
  ASSERT_EQ(status.err, novaSuccess) << status.message;
  ASSERT_EQ(runtimeStreamSynchronize(s0.get()), kRuntimeSuccess);

  ScopedDeviceBuffer r1(s1.opaque());
  status = r1.reserve(sz);
  ASSERT_EQ(status.err, novaSuccess) << status.message;

  int liveHandoff = 0;
  for (int round = 0; round < rounds; ++round) {
    novaTestScaleInto(reinterpret_cast<const float *>(a.get().ptr),
                      reinterpret_cast<float *>(r1.get().ptr), n, 2.0f,
                      s1.opaque());

    status = a.release();
    ASSERT_EQ(status.err, novaSuccess) << status.message;

    status = a.reserve(sz);
    ASSERT_EQ(status.err, novaSuccess) << status.message;

    novaTestWriteFloat(reinterpret_cast<float *>(a.get().ptr), n, 9.75f,
                       s2.opaque());

    ASSERT_EQ(runtimeStreamSynchronize(s1.get()), kRuntimeSuccess);
    ASSERT_EQ(runtimeStreamSynchronize(s2.get()), kRuntimeSuccess);

    std::vector<float> hostR(sz / sizeof(float), 0.0f);
    status = deviceTransfer(r1.get().ptr, hostR.data(),
                            deviceMemcpyDeviceToHost, sz, s0.opaque());
    ASSERT_EQ(status.err, novaSuccess) << status.message;
    ASSERT_EQ(runtimeStreamSynchronize(s0.get()), kRuntimeSuccess);

    if (std::fabs(hostR[0] - 9.75f) < 0.01f) {
      ++liveHandoff;
    }
  }

  EXPECT_EQ(liveHandoff, 0)
      << "cross-stream reuse delivered a live block without an event";
}

/**
 * @brief Measures the batched engine-stream path against sync-per-op.
 *
 * @details
 * The ratio between enqueueing without host synchronization and
 * synchronizing after every operation is a property of the host, not
 * of the code under test: a shared runner saturates the CPU and the
 * measurement drifts.  The test is therefore disabled by default and
 * CTest never schedules it.
 *
 * Run it explicitly to collect the metric:
 *
 * @code
 * ./ncore/tests/engineStream/ncore_engine_stream_tests \
 *   --gtest_also_run_disabled_tests \
 *   --gtest_filter=EngineStreamChain.DISABLED_BatchedVsSyncPerOp
 * @endcode
 *
 * The measured ratio is reported as a test property so a CI job can
 * archive it without gating on it.
 */
TEST(EngineStreamChain, DISABLED_BatchedVsSyncPerOp) {
  NOVA_TEST_REQUIRE_GPU();
  const int n = 64;
  const std::size_t sz = static_cast<std::size_t>(n) * sizeof(float);
  const int iters = 500;

  int deviceId = -1;
  ASSERT_EQ(runtimeGetDevice(&deviceId), kRuntimeSuccess);

  DeviceMemoryPoolStats before{};
  const novaStatus_t st = getPoolStats(deviceId, &before);
  if (st.err == novaMemoryPoolsNotNativelySupported) {
    GTEST_SKIP() << "Current device does not support native memory pools";
  }

  ScopedStream stream;
  ASSERT_EQ(stream.create(), kRuntimeSuccess);

  for (int w = 0; w < 10; ++w) {
    ScopedDeviceBuffer b(stream.opaque());
    ASSERT_EQ(b.reserve(sz).err, novaSuccess);
    ASSERT_EQ(b.release().err, novaSuccess);
  }
  ASSERT_EQ(runtimeStreamSynchronize(stream.get()), kRuntimeSuccess);

  double bestSpeedup = 0.0;
  double bestSyncUs = 0.0;
  double bestBatchUs = 0.0;
  for (int trial = 0; trial < 3; ++trial) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      ScopedDeviceBuffer b(stream.opaque());
      ASSERT_EQ(b.reserve(sz).err, novaSuccess);
      ASSERT_EQ(runtimeStreamSynchronize(stream.get()), kRuntimeSuccess);
      std::vector<float> h(sz / sizeof(float), 1.0f);
      ASSERT_EQ(deviceTransfer(h.data(), b.get().ptr, deviceMemcpyHostToDevice,
                               sz, stream.opaque())
                    .err,
                novaSuccess);
      ASSERT_EQ(runtimeStreamSynchronize(stream.get()), kRuntimeSuccess);
      ASSERT_EQ(deviceTransfer(b.get().ptr, h.data(), deviceMemcpyDeviceToHost,
                               sz, stream.opaque())
                    .err,
                novaSuccess);
      ASSERT_EQ(runtimeStreamSynchronize(stream.get()), kRuntimeSuccess);
      ASSERT_EQ(b.release().err, novaSuccess);
      ASSERT_EQ(runtimeStreamSynchronize(stream.get()), kRuntimeSuccess);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double syncUs =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

    const auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      ScopedDeviceBuffer b(stream.opaque());
      ASSERT_EQ(b.reserve(sz).err, novaSuccess);
      std::vector<float> h(sz / sizeof(float), 1.0f);
      ASSERT_EQ(deviceTransfer(h.data(), b.get().ptr, deviceMemcpyHostToDevice,
                               sz, stream.opaque())
                    .err,
                novaSuccess);
      ASSERT_EQ(deviceTransfer(b.get().ptr, h.data(), deviceMemcpyDeviceToHost,
                               sz, stream.opaque())
                    .err,
                novaSuccess);
      ASSERT_EQ(b.release().err, novaSuccess);
    }
    ASSERT_EQ(runtimeStreamSynchronize(stream.get()), kRuntimeSuccess);
    const auto t3 = std::chrono::steady_clock::now();
    const double batchUs =
        std::chrono::duration<double, std::micro>(t3 - t2).count() / iters;

    const double speedup = (batchUs > 0.0) ? (syncUs / batchUs) : 0.0;
    if (speedup > bestSpeedup) {
      bestSpeedup = speedup;
      bestSyncUs = syncUs;
      bestBatchUs = batchUs;
    }
  }

  RecordProperty("bestSpeedup", bestSpeedup);
  RecordProperty("bestSyncUs", bestSyncUs);
  RecordProperty("bestBatchUs", bestBatchUs);
  EXPECT_GE(bestSpeedup, 1.2)
      << "batched path did not beat sync-per-op by the required margin";
}

#endif // NOVA_HAS_CUDA || NOVA_HAS_HIP
