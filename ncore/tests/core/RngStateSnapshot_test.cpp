/**
 * @file RngStateSnapshot_test.cpp
 * @brief Snapshot capacity, offset coherence, and restoration guards.
 *
 * The snapshot travels in a caller-owned buffer: the in/out count must
 * honor the passed capacity and report the number of entries written.
 * A snapshot must carry the handle offset so a restored handle keeps a
 * coherent counter of emitted values, and out-of-range entries are
 * rejected instead of silently truncated.
 */

#include <gtest/gtest.h>

#include <cstdint>

#include <ncore/core/status.h>
#include <ncore/rand/generator.h>

namespace {

constexpr uint64_t kSeed = 7;

/// Maximum snapshot entries any engine emits (state + offset + up to 4 spare).
constexpr size_t kMaxSnapshot = 16;

/// Asserts that restoring a non-fitting copy fails loudly.
TEST(RngStateSnapshot, SnapshotWritesNothingPastThePassedCapacity) {
  Generator_ philox = {};
  ASSERT_EQ(philox_init(&philox.state.philox, kSeed, 0).err, novaSuccess);
  philox.device = DEVICE_CPU;
  philox.engine = Philox4x32;

  size_t capacity = 2;
  uint64_t snap[kMaxSnapshot] = {};
  const novaStatus_t st = generator_get_state(&philox, snap, &capacity);

  EXPECT_EQ(st.err, novaInvalidValue)
      << "an undersized snapshot count must fail, not silently overrun"
      << (st.message != nullptr ? st.message : "");
  EXPECT_EQ(capacity, 7U)
      << "the out count must report how many entries Philox needs";
}

TEST(RngStateSnapshot, XoshiroSnapshotReportsFiveEntries) {
  Generator_ xoshiro = {};
  ASSERT_EQ(xoshiro256_init(&xoshiro.state.xoshiro, kSeed).err, novaSuccess);
  xoshiro.device = DEVICE_CPU;
  xoshiro.engine = Xoshiro256;

  size_t capacity = kMaxSnapshot;
  uint64_t snap[kMaxSnapshot] = {};
  const novaStatus_t st = generator_get_state(&xoshiro, snap, &capacity);

  ASSERT_EQ(st.err, novaSuccess) << (st.message != nullptr ? st.message : "");
  EXPECT_EQ(capacity, 5U) << "four state entries plus the offset slot";
}

TEST(RngStateSnapshot, PhiloxSnapshotReportsSevenEntries) {
  Generator_ philox = {};
  ASSERT_EQ(philox_init(&philox.state.philox, kSeed, 0).err, novaSuccess);
  philox.device = DEVICE_CPU;
  philox.engine = Philox4x32;

  size_t capacity = kMaxSnapshot;
  uint64_t snap[kMaxSnapshot] = {};
  const novaStatus_t st = generator_get_state(&philox, snap, &capacity);

  ASSERT_EQ(st.err, novaSuccess) << (st.message != nullptr ? st.message : "");
  EXPECT_EQ(capacity, 7U) << "counter then key then offset";
}

TEST(RngStateSnapshot, SnapshotCarriesTheEmittedOffset) {
  Generator_ src = {};
  ASSERT_EQ(generator_init(&src, Philox4x32, DEVICE_CPU, kSeed, 0).err,
            novaSuccess);
  uint32_t scratch[6] = {};
  ASSERT_EQ(generator_fill_u32(&src, scratch, 6).err, novaSuccess);
  ASSERT_EQ(src.offset, 6U);

  uint64_t snap[kMaxSnapshot] = {};
  size_t capacity = kMaxSnapshot;
  ASSERT_EQ(generator_get_state(&src, snap, &capacity).err, novaSuccess);
  ASSERT_EQ(capacity, 7U);

  Generator_ dst = {};
  ASSERT_EQ(philox_init(&dst.state.philox, 0, 0).err, novaSuccess);
  dst.device = DEVICE_CPU;
  dst.engine = Philox4x32;
  ASSERT_EQ(generator_set_state(&dst, snap, capacity).err, novaSuccess);
  EXPECT_EQ(dst.offset, 6U)
      << "a restored handle must keep the count of emitted values";
}

TEST(RngStateSnapshot, SetStateRejectsEntriesBeyond32Bits) {
  uint64_t snap[kMaxSnapshot] = {0, 0, 0, 0, 5, 7, 0};
  snap[0] = (uint64_t(1) << 32) | 5; ///< out of counter range.

  Generator_ dst = {};
  ASSERT_EQ(philox_init(&dst.state.philox, 0, 0).err, novaSuccess);
  dst.device = DEVICE_CPU;
  dst.engine = Philox4x32;
  const novaStatus_t st = generator_set_state(&dst, snap, 7);

  EXPECT_EQ(st.err, novaInvalidValue)
      << "silently truncating a 64-bit entry would incoherent the counter"
      << (st.message != nullptr ? st.message : "");
}

} // namespace
