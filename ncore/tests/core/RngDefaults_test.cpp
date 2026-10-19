/**
 * @file RngDefaults_test.cpp
 * @brief Process-default fills advance the shared stream.
 *
 * Two successive default fills must hand out disjoint values: the
 * first one matches an identically seeded private handle, the second
 * one continues like a 2n fill would. Copies taken through
 * default_generator() stay frozen snapshots, which is why the
 * advancing entry points exist.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/wrappers/tensor.hh>
#include <ncore/rand/generator.h>

#include "utils/TensorUtils.hpp"

namespace {

using ncore::wrappers::TensorCXX;

/// Asserts @p st holds the success code (message included on failure).
void expectOk(const novaStatus_t st, const char *ctx) {
  ASSERT_EQ(st.err, novaSuccess)
      << ctx << (st.message != nullptr ? st.message : "");
}

TEST(RngDefaults, CpuDefaultFillAdvancesTheSharedStream) {
  constexpr size_t kN = 8;
  constexpr uint64_t kSeed = 77;

  expectOk(manual_seed(kSeed), "seed");

  Generator_ ref = {};
  expectOk(generator_init(&ref, Xoshiro256, DEVICE_CPU, kSeed, 0), "ref init");
  uint32_t refWhole[kN * 2] = {};
  expectOk(generator_fill_u32(&ref, refWhole, kN * 2), "ref fill");

  uint32_t first[kN] = {};
  expectOk(default_fill_u32(DEVICE_CPU, first, kN), "first default fill");
  EXPECT_EQ(0, std::memcmp(first, refWhole, sizeof first))
      << "first default draw must match a fresh identically seeded handle";

  uint32_t second[kN] = {};
  expectOk(default_fill_u32(DEVICE_CPU, second, kN), "second default fill");
  EXPECT_EQ(0, std::memcmp(second, refWhole + kN, sizeof second))
      << "second default draw must continue the stream";
  EXPECT_NE(0, std::memcmp(first, second, sizeof first))
      << "successive default fills must hand out disjoint values";
}

TEST(RngDefaults, DefaultGeneratorSnapshotsStayFrozen) {
  constexpr size_t kN = 4;
  constexpr uint64_t kSeed = 91;

  expectOk(manual_seed(kSeed), "seed");

  Generator_ a = {};
  Generator_ b = {};
  expectOk(default_generator(DEVICE_CPU, &a), "copy a");
  expectOk(default_generator(DEVICE_CPU, &b), "copy b");

  uint32_t fa[kN] = {};
  uint32_t fb[kN] = {};
  expectOk(generator_fill_u32(&a, fa, kN), "fill a");
  expectOk(generator_fill_u32(&b, fb, kN), "fill b");
  EXPECT_EQ(0, std::memcmp(fa, fb, sizeof fa))
      << "snapshots are private streams and must stay reproducible";
}

TEST(RngDefaults, DefaultFillGuardsPlacementAndEmptyDemands) {
  uint32_t unused[4] = {};

  const novaStatus_t meta = default_fill_u32(DEVICE_META, unused, 4);
  EXPECT_EQ(meta.err, novaInvalidDevice)
      << (meta.message != nullptr ? meta.message : "");

  const novaStatus_t unknown =
      default_fill_u32(static_cast<Device_>(77), unused, 4);
  EXPECT_EQ(unknown.err, novaInvalidDevice)
      << (unknown.message != nullptr ? unknown.message : "");

  const novaStatus_t empty = default_fill_u32(DEVICE_CPU, nullptr, 0);
  expectOk(empty, "zero demand is inert");
}

TEST(RngDefaults, GpuDefaultFillAdvancesTheSharedStream) {
  NOVA_TEST_REQUIRE_GPU();
  constexpr size_t kN = 8;
  constexpr uint64_t kSeed = 88;

  expectOk(manual_seed(kSeed), "seed");

  Generator_ ref = {};
  expectOk(generator_init(&ref, Philox4x32, DEVICE_CPU, kSeed, 0), "ref init");
  uint32_t refWhole[kN * 2] = {};
  expectOk(generator_fill_u32(&ref, refWhole, kN * 2), "ref fill");

  const std::vector<size_t> shape{kN};
  novaStatus_t st{};
  TensorCXX dev(shape, DType_::UnSigned32, DEVICE_GPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess) << (st.message != nullptr ? st.message : "");
  TensorCXX home(shape, DType_::UnSigned32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess) << (st.message != nullptr ? st.message : "");

  uint32_t *devPtr = dev.mutableCTensor().data.u32;

  expectOk(default_fill_u32(DEVICE_GPU, devPtr, kN), "first default fill");
  {
    Tensor devTen = dev.mutableCTensor();
    Tensor homeTen = home.mutableCTensor();
    expectOk(transf_tensor_from_device(&devTen, &homeTen), "fetch first");
    const uint32_t *homePtr = homeTen.data.u32;
    EXPECT_EQ(0, std::memcmp(homePtr, refWhole, kN * sizeof(uint32_t)))
        << "first device draw must reproduce the host reference";
  }

  expectOk(default_fill_u32(DEVICE_GPU, devPtr, kN), "second default fill");
  {
    Tensor devTen = dev.mutableCTensor();
    Tensor homeTen = home.mutableCTensor();
    expectOk(transf_tensor_from_device(&devTen, &homeTen), "fetch second");
    const uint32_t *homePtr = homeTen.data.u32;
    EXPECT_EQ(0, std::memcmp(homePtr, refWhole + kN, kN * sizeof(uint32_t)))
        << "second device draw must continue the stream";
  }
}

} // namespace
