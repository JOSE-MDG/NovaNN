/**
 * @file RngGpuParity_test.cpp
 * @brief Bitwise host/device parity for the Philox fills.
 *
 * The GPU kernels receive the same key and base counter the host uses,
 * so a device fill must reproduce the host fill bit for bit, including
 * full blocks, masked tails, and counters whose block addition crosses
 * a 32-bit word of the 128-bit counter. The carry unit tests pin
 * @ref philox_ctr_add() on every word boundary of that counter. The
 * launchers must also accept a zero demand with success and no effect.
 *
 * Tests needing a device call NOVA_TEST_REQUIRE_GPU() first; the
 * counter-carry cases run unconditionally.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <ncore/headeronly/wrappers/tensor.hh>
#include <ncore/native/kernels/random/rng.h>
#include <ncore/rand/generator.h>
#include <ncore/rand/philox.h>

#include <tests/common/Gpu.hpp>
#include <tests/common/Status.hpp>
#include <tests/tensor/DeviceMirror.hpp>
#include <tests/tensor/TensorUtils.hpp>

namespace {

using tests::common::expectOk;
using tests::tensor::DeviceMirror;

constexpr uint64_t kSeed = 123;
constexpr uint64_t kSubsequence = 5;

/** Host Philox handle seeded once; fills read its key and counter. */
Generator_ makePhiloxHost(uint64_t seed, uint64_t subsequence) {
  Generator_ gen = {};
  const novaStatus_t st =
      generator_init(&gen, Philox4x32, DEVICE_CPU, seed, subsequence);
  EXPECT_EQ(st.err, novaSuccess) << (st.message != nullptr ? st.message : "");
  return gen;
}

/** Snapshots the stream location before any fill advances the handle. */
void captureBase(const Generator_ &gen, uint32_t key[2], uint32_t base[4]) {
  key[0] = gen.state.philox.key[0];
  key[1] = gen.state.philox.key[1];
  for (size_t w = 0; w < 4; ++w) {
    base[w] = gen.state.philox.ctr[w];
  }
}

/** Non-fitting expected counter for the carry assertions. */
void expectCtr(const uint32_t actual[4], const uint32_t expected[4]) {
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(actual[i], expected[i]) << "lane " << i;
  }
}

TEST(RngPhiloxCounter, CtrAddCarriesOnEveryWordBoundary) {
  const uint32_t baseWord1[4] = {0xFFFFFFFFU, 0, 0, 0};
  const uint32_t baseWord2[4] = {0xFFFFFFFFU, 0xFFFFFFFFU, 0, 0};
  const uint32_t baseWord3[4] = {0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0};
  const uint32_t expectWord1[4] = {0, 1, 0, 0};
  const uint32_t expectWord2[4] = {0, 0, 1, 0};
  const uint32_t expectWord3[4] = {0, 0, 0, 1};
  const uint32_t expectSpread[4] = {0, 2, 0, 0};
  uint32_t out[4] = {};

  philox_ctr_add(baseWord1, 1, out);
  expectCtr(out, expectWord1);

  philox_ctr_add(baseWord2, 1, out);
  expectCtr(out, expectWord2);

  philox_ctr_add(baseWord3, 1, out);
  expectCtr(out, expectWord3);

  const uint32_t zeroSpread[4] = {0, 0, 0, 0};
  philox_ctr_add(zeroSpread, 1ULL << 33, out);
  expectCtr(out, expectSpread);
}

TEST(RngGpuParity, FillU32MatchesHostBitwise) {
  NOVA_TEST_REQUIRE_GPU();
  constexpr size_t kN = 64;

  Generator_ gen = makePhiloxHost(kSeed, kSubsequence);
  uint32_t key[2] = {};
  uint32_t base[4] = {};
  captureBase(gen, key, base);
  uint32_t host[kN] = {};
  expectOk(generator_fill_u32(&gen, host, kN), "host");

  DeviceMirror mirror(kN, DType_::UnSigned32);
  ASSERT_EQ(mirror.st.err, novaSuccess)
      << (mirror.st.message != nullptr ? mirror.st.message : "");
  expectOk(launchRngFillU32(key, base, mirror.devTen().data.u32, kN), "device");
  mirror.fetch();

  const uint32_t *home = mirror.homeTen().data.u32;
  EXPECT_EQ(0, std::memcmp(host, home, sizeof host))
      << "device Philox must reproduce the host stream bit for bit";
}

TEST(RngGpuParity, FillU64MatchesHostBitwise) {
  NOVA_TEST_REQUIRE_GPU();
  constexpr size_t kN = 64;

  Generator_ gen = makePhiloxHost(kSeed, kSubsequence);
  uint32_t key[2] = {};
  uint32_t base[4] = {};
  captureBase(gen, key, base);
  uint64_t host[kN] = {};
  expectOk(generator_fill_u64(&gen, host, kN), "host");

  DeviceMirror mirror(kN, DType_::UnSigned64);
  ASSERT_EQ(mirror.st.err, novaSuccess)
      << (mirror.st.message != nullptr ? mirror.st.message : "");
  expectOk(launchRngFillU64(key, base, mirror.devTen().data.u64, kN), "device");
  mirror.fetch();

  const uint64_t *home = mirror.homeTen().data.u64;
  EXPECT_EQ(0, std::memcmp(host, home, sizeof host))
      << "device Philox must reproduce the host stream bit for bit";
}

TEST(RngGpuParity, PartialTailMatchesHostBitwise) {
  NOVA_TEST_REQUIRE_GPU();
  constexpr size_t kN = 5;

  Generator_ gen = makePhiloxHost(kSeed, kSubsequence);
  uint32_t key[2] = {};
  uint32_t base[4] = {};
  captureBase(gen, key, base);
  uint32_t host[kN] = {};
  expectOk(generator_fill_u32(&gen, host, kN), "host");

  DeviceMirror mirror(kN, DType_::UnSigned32);
  ASSERT_EQ(mirror.st.err, novaSuccess)
      << (mirror.st.message != nullptr ? mirror.st.message : "");
  expectOk(launchRngFillU32(key, base, mirror.devTen().data.u32, kN), "device");
  mirror.fetch();

  const uint32_t *home = mirror.homeTen().data.u32;
  EXPECT_EQ(0, std::memcmp(host, home, sizeof host))
      << "the masked tail must discard the same lanes on both sides";
}

TEST(RngGpuParity, CounterCrossesCountsAcross32Bit) {
  NOVA_TEST_REQUIRE_GPU();
  constexpr size_t kN = 8;
  const uint32_t boundary[4] = {0xFFFFFFFFU, 0, 0, 0};

  Generator_ gen = makePhiloxHost(kSeed, kSubsequence);
  expectOk(philox_set_state(&gen.state.philox, boundary, gen.state.philox.key),
           "state boundary");
  uint32_t host[kN] = {};
  expectOk(generator_fill_u32(&gen, host, kN), "host");

  DeviceMirror mirror(kN, DType_::UnSigned32);
  ASSERT_EQ(mirror.st.err, novaSuccess)
      << (mirror.st.message != nullptr ? mirror.st.message : "");
  expectOk(launchRngFillU32(gen.state.philox.key, boundary,
                            mirror.devTen().data.u32, kN),
           "device");
  mirror.fetch();

  const uint32_t *home = mirror.homeTen().data.u32;
  EXPECT_EQ(0, std::memcmp(host, home, sizeof host))
      << "a carry across counter word 0 must match on host and device";
}

TEST(RngGpuParity, CounterCrossesCountsAcross64Bit) {
  NOVA_TEST_REQUIRE_GPU();
  constexpr size_t kN = 8;
  const uint32_t boundary[4] = {0xFFFFFFFFU, 0xFFFFFFFFU, 0, 0};

  Generator_ gen = makePhiloxHost(kSeed, kSubsequence);
  expectOk(philox_set_state(&gen.state.philox, boundary, gen.state.philox.key),
           "state boundary");
  uint32_t host[kN] = {};
  expectOk(generator_fill_u32(&gen, host, kN), "host");

  DeviceMirror mirror(kN, DType_::UnSigned32);
  ASSERT_EQ(mirror.st.err, novaSuccess)
      << (mirror.st.message != nullptr ? mirror.st.message : "");
  expectOk(launchRngFillU32(gen.state.philox.key, boundary,
                            mirror.devTen().data.u32, kN),
           "device");
  mirror.fetch();

  const uint32_t *home = mirror.homeTen().data.u32;
  EXPECT_EQ(0, std::memcmp(host, home, sizeof host))
      << "a carry across counter words 0-1 (2^64) must match on both sides";
}

TEST(RngGpuParity, ZeroDemandSucceedsAndLeavesTheBufferAlone) {
  NOVA_TEST_REQUIRE_GPU();
  constexpr size_t kN = 8;

  Generator_ gen = makePhiloxHost(kSeed, kSubsequence);
  DeviceMirror mirror(kN, DType_::UnSigned32);
  ASSERT_EQ(mirror.st.err, novaSuccess)
      << (mirror.st.message != nullptr ? mirror.st.message : "");

  expectOk(launchRngFillU32(gen.state.philox.key, gen.state.philox.ctr,
                            mirror.devTen().data.u32, kN),
           "pattern launch");
  mirror.fetch();
  const std::vector<uint32_t> pattern(mirror.homeTen().data.u32,
                                      mirror.homeTen().data.u32 + kN);

  const novaStatus_t zero = launchRngFillU32(
      gen.state.philox.key, gen.state.philox.ctr, mirror.devTen().data.u32, 0);
  EXPECT_EQ(zero.err, novaSuccess)
      << "an empty demand must not look like an overflow"
      << (zero.message != nullptr ? zero.message : "");

  mirror.fetch();
  EXPECT_EQ(0, std::memcmp(pattern.data(), mirror.homeTen().data.u32,
                           kN * sizeof(uint32_t)))
      << "a zero demand must not touch the destination";
}

} // namespace
