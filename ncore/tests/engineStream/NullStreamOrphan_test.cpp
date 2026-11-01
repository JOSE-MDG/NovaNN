/**
 * @file NullStreamOrphan_test.cpp
 * @brief Rocclr null-stream orphan behavior.
 *
 * These gates pin the observed backend behavior of an operation enqueued
 * on the null stream followed by a host delay and a stream creation. They
 * call the runtime directly (not the NovaNN pool layer) so they stay
 * stable across the engine-stream API migration.
 *
 * The expectations are backend-specific because the two runtimes differ:
 * the legacy CUDA null stream keeps the operation, rocclr orphans it. When
 * a future rocclr release fixes the orphaning, these gates fail on
 * purpose and the HIP branch has to be inverted to expect the operation
 * to survive.
 */

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <print>
#include <thread>

#include "HazardKernel.hpp"
#include <tests/common/Gpu.hpp>
#include <tests/tensor/TensorUtils.hpp>

#if defined(NOVA_HAS_CUDA)
#include <cuda_runtime_api.h>
#define RTAPI(name) cuda##name
using rt_prop_t = cudaDeviceProp;
#elifdef NOVA_HAS_HIP
#include <hip/hip_runtime.h>
#define RTAPI(name) hip##name
using rt_prop_t = hipDeviceProp_t;
#endif

#if defined(NOVA_HAS_CUDA) || defined(NOVA_HAS_HIP)

using rt_t = RTAPI(Error_t);
using rt_stream_t = RTAPI(Stream_t);
using rt_memcpy_kind_t = RTAPI(MemcpyKind);

static const rt_t RT_SUCCESS = RTAPI(Success);
static const unsigned int RT_NONBLOCK = RTAPI(StreamNonBlocking);
static const rt_memcpy_kind_t RT_H2D = RTAPI(MemcpyHostToDevice);
static const rt_memcpy_kind_t RT_D2H = RTAPI(MemcpyDeviceToHost);

namespace {

class ScopedDevMem final {
public:
  ScopedDevMem() = default;
  ScopedDevMem(const ScopedDevMem &) = delete;
  ScopedDevMem &operator=(const ScopedDevMem &) = delete;

  ~ScopedDevMem() {
    if (ptr_ != nullptr) {
      (void)RTAPI(Free)(ptr_);
    }
  }

  [[nodiscard]] void **out() { return &ptr_; }
  [[nodiscard]] void *get() const { return ptr_; }

private:
  void *ptr_ = nullptr;
};

} // namespace

#endif // NOVA_HAS_CUDA || NOVA_HAS_HIP

#if defined(NOVA_HAS_CUDA) || defined(NOVA_HAS_HIP)

TEST(NullStreamOrphan, DelayedOpLost) {
  NOVA_TEST_REQUIRE_GPU();

  ScopedDevMem dResult;
  ASSERT_EQ(RTAPI(Malloc)(dResult.out(), sizeof(int)), RT_SUCCESS);

  int zero = 0;
  ASSERT_EQ(RTAPI(Memcpy)(dResult.get(), &zero, sizeof(int), RT_H2D),
            RT_SUCCESS);
  ASSERT_EQ(RTAPI(DeviceSynchronize)(), RT_SUCCESS);

  // Enqueue the op on the null stream, wait on the host, then create a stream.
  novaTestNullWrite(static_cast<int *>(dResult.get()), 777, nullptr);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));

  rt_stream_t s = nullptr;
  ASSERT_EQ(RTAPI(StreamCreateWithFlags)(&s, RT_NONBLOCK), RT_SUCCESS);

  ASSERT_EQ(RTAPI(StreamSynchronize)(nullptr), RT_SUCCESS);

  int hResult = 0;
  ASSERT_EQ(RTAPI(Memcpy)(&hResult, dResult.get(), sizeof(int), RT_D2H),
            RT_SUCCESS);

  (void)RTAPI(StreamDestroy)(s);
  (void)RTAPI(DeviceSynchronize)();

#ifdef NOVA_HAS_HIP
  // rocclr: the null-stream op is orphaned by the subsequent stream
  // creation. Invert this expectation when upstream fixes the orphaning.
  EXPECT_NE(hResult, 777) << "expected LOST on HIP, got KEPT";
#else
  // CUDA: the legacy null stream keeps the op.
  EXPECT_EQ(hResult, 777) << "expected KEPT on CUDA, got LOST";
#endif
}

TEST(NullStreamOrphan, CreateSweepAllDelaysLost) {
  NOVA_TEST_REQUIRE_GPU();

  ScopedDevMem dResult;
  ASSERT_EQ(RTAPI(Malloc)(dResult.out(), sizeof(int)), RT_SUCCESS);

  static const std::array<double, 8> delays = {0.0,  0.05, 0.10, 0.20,
                                               0.50, 1.0,  2.0,  5.0};
  [[maybe_unused]] bool positiveDelaysAllLost = true;
  bool anyLost = false;
  for (const double delayMs : delays) {
    int lost = 0;
    const int tries = 8;
    for (int r = 0; r < tries; ++r) {
      int zero = 0;
      ASSERT_EQ(RTAPI(Memcpy)(dResult.get(), &zero, sizeof(int), RT_H2D),
                RT_SUCCESS);
      ASSERT_EQ(RTAPI(DeviceSynchronize)(), RT_SUCCESS);

      novaTestNullWrite(static_cast<int *>(dResult.get()), 777, nullptr);
      std::this_thread::sleep_for(
          std::chrono::duration<double, std::milli>(delayMs));

      rt_stream_t s = nullptr;
      ASSERT_EQ(RTAPI(StreamCreateWithFlags)(&s, RT_NONBLOCK), RT_SUCCESS);

      ASSERT_EQ(RTAPI(StreamSynchronize)(nullptr), RT_SUCCESS);

      int hResult = 0;
      ASSERT_EQ(RTAPI(Memcpy)(&hResult, dResult.get(), sizeof(int), RT_D2H),
                RT_SUCCESS);

      if (hResult != 777) {
        ++lost;
      }
      ASSERT_EQ(RTAPI(StreamDestroy)(s), RT_SUCCESS);
      ASSERT_EQ(RTAPI(DeviceSynchronize)(), RT_SUCCESS);
    }
    std::println("  delay={:.2f} ms -> lost={}/{}", delayMs, lost, tries);
    if (lost > 0) {
      anyLost = true;
    }
    // Without a host delay the op can complete before the stream is
    // created, so only the positive delays pin the defect.
    if (delayMs > 0.0 && lost != tries) {
      positiveDelaysAllLost = false;
    }
  }

#ifdef NOVA_HAS_HIP
  // rocclr: every positive delay loses the op, and the zero delay
  // reproduces the defect at least once. Invert these expectations when
  // upstream fixes the orphaning.
  EXPECT_TRUE(positiveDelaysAllLost)
      << "expected every positive delay LOST on HIP";
  EXPECT_TRUE(anyLost) << "expected at least one LOST on HIP";
#else
  // CUDA: no delay loses the op.
  EXPECT_FALSE(anyLost) << "expected zero LOST on CUDA";
#endif
}

#endif // NOVA_HAS_CUDA || NOVA_HAS_HIP
