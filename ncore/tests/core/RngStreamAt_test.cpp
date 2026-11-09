/**
 * @file RngStreamAt_test.cpp
 * @brief Stream cursors split one handle into byte-identical views.
 *
 * generator_u32_stream_at() and generator_u64_stream_at() must seat a
 * cursor so that filling it continues the source stream exactly where
 * the requested width left off: parked Xoshiro halves, parked Philox
 * lanes, and the mixed width rule (only a whole parked pair feeds a
 * u64 fill, a lone u32 lane survives forward) included. On a source
 * holding parked lanes the cursor must behave like a clone of that
 * source after really consuming the skipped values of the positioned
 * width.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/rand/generator.h>

namespace {

constexpr uint64_t kSeed = 42;

/// Continuation length every cursor fills after positioning.
constexpr size_t kTail = 64;

/// Skip fills inside the parked lane reach of a source.
constexpr size_t kNearOffsets[] = {0, 1, 2, 3, 5};

/// Offsets around lane, block and sample boundaries up to 2^16 + 1.
const size_t kOffsets[] = {
    0,    1,    2,    3,     4,     5,     6,     7,     8,
    9,    15,   17,   31,    33,    63,    65,    127,   129,
    255,  257,  511,  513,   1023,  1025,  2047,  2049,  4095,
    4097, 8191, 8193, 16383, 16385, 32767, 32769, 65535, 65537,
};

/** Requires the u32 cursor continuation to match the serial fill from
    every listed offset, with the source left unconsumed. */
void expectU32CursorsContinueTheSerialFill(RngEngine engine) {
  constexpr size_t kLast = 65537;
  std::vector<uint32_t> reference(kLast + kTail);
  Generator_ serial = {};
  ASSERT_EQ(generator_init(&serial, engine, DEVICE_CPU, kSeed, 0).err,
            novaSuccess);
  ASSERT_EQ(generator_fill_u32(&serial, reference.data(), reference.size()).err,
            novaSuccess);

  for (size_t offset : kOffsets) {
    SCOPED_TRACE(::testing::Message() << "offset=" << offset);
    Generator_ source = {};
    Generator_ cursor = {};
    ASSERT_EQ(generator_init(&source, engine, DEVICE_CPU, kSeed, 0).err,
              novaSuccess);
    ASSERT_EQ(generator_u32_stream_at(&source, offset, &cursor).err,
              novaSuccess);
    ASSERT_EQ(source.offset, 0U) << "positioning must not consume the source";

    std::vector<uint32_t> tail(kTail);
    ASSERT_EQ(generator_fill_u32(&cursor, tail.data(), kTail).err, novaSuccess);
    EXPECT_EQ(0, std::memcmp(reference.data() + offset, tail.data(),
                             kTail * sizeof(uint32_t)))
        << "the u32 cursor left the serial stream";
    EXPECT_EQ(cursor.offset, offset + kTail);
  }
}

/** u64 flavor of the serial continuation check. */
void expectU64CursorsContinueTheSerialFill(RngEngine engine) {
  constexpr size_t kLast = 65537;
  std::vector<uint64_t> reference(kLast + kTail);
  Generator_ serial = {};
  ASSERT_EQ(generator_init(&serial, engine, DEVICE_CPU, kSeed, 0).err,
            novaSuccess);
  ASSERT_EQ(generator_fill_u64(&serial, reference.data(), reference.size()).err,
            novaSuccess);

  for (size_t offset : kOffsets) {
    SCOPED_TRACE(::testing::Message() << "offset=" << offset);
    Generator_ source = {};
    Generator_ cursor = {};
    ASSERT_EQ(generator_init(&source, engine, DEVICE_CPU, kSeed, 0).err,
              novaSuccess);
    ASSERT_EQ(generator_u64_stream_at(&source, offset, &cursor).err,
              novaSuccess);
    ASSERT_EQ(source.offset, 0U) << "positioning must not consume the source";

    std::vector<uint64_t> tail(kTail);
    ASSERT_EQ(generator_fill_u64(&cursor, tail.data(), kTail).err, novaSuccess);
    EXPECT_EQ(0, std::memcmp(reference.data() + offset, tail.data(),
                             kTail * sizeof(uint64_t)))
        << "the u64 cursor left the serial stream";
    EXPECT_EQ(cursor.offset, offset + kTail);
  }
}

/** Requires the cursor to behave like a clone of @p source after
    really consuming @p offset values of the positioned width, for
    later fills of both widths. */
void expectCursorEquivalence(const Generator_ &source, bool wide,
                             size_t offset) {
  Generator_ clone = {};
  Generator_ cursor = {};
  ASSERT_EQ(generator_clone(&source, &clone).err, novaSuccess);

  std::vector<uint32_t> skipNarrow(offset);
  std::vector<uint64_t> skipWide(offset);
  if (wide) {
    ASSERT_EQ(generator_u64_stream_at(&source, offset, &cursor).err,
              novaSuccess);
    ASSERT_EQ(generator_fill_u64(&clone, skipWide.data(), offset).err,
              novaSuccess);
  } else {
    ASSERT_EQ(generator_u32_stream_at(&source, offset, &cursor).err,
              novaSuccess);
    ASSERT_EQ(generator_fill_u32(&clone, skipNarrow.data(), offset).err,
              novaSuccess);
  }
  EXPECT_EQ(cursor.offset, clone.offset);

  constexpr size_t kWide = 9;
  constexpr size_t kNarrow = 11;
  std::vector<uint64_t> cursorWide(kWide);
  std::vector<uint32_t> cursorNarrow(kNarrow);
  std::vector<uint64_t> cloneWide(kWide);
  std::vector<uint32_t> cloneNarrow(kNarrow);
  ASSERT_EQ(generator_fill_u64(&cursor, cursorWide.data(), kWide).err,
            novaSuccess);
  ASSERT_EQ(generator_fill_u32(&cursor, cursorNarrow.data(), kNarrow).err,
            novaSuccess);
  ASSERT_EQ(generator_fill_u64(&clone, cloneWide.data(), kWide).err,
            novaSuccess);
  ASSERT_EQ(generator_fill_u32(&clone, cloneNarrow.data(), kNarrow).err,
            novaSuccess);

  EXPECT_EQ(0, std::memcmp(cursorWide.data(), cloneWide.data(),
                           kWide * sizeof(uint64_t)))
      << "the u64 continuation diverged after positioning";
  EXPECT_EQ(0, std::memcmp(cursorNarrow.data(), cloneNarrow.data(),
                           kNarrow * sizeof(uint32_t)))
      << "the u32 continuation diverged after positioning";
}

/** Builds a Philox source holding parked lanes: a 5 lane u32 demand
    parks 3 lanes, 6 parks 2, 7 parks 1, and an odd u64 demand parks a
    whole pair. */
Generator_ parkedPhiloxSource(size_t demand, bool wide) {
  Generator_ source = {};
  EXPECT_EQ(generator_init(&source, Philox4x32, DEVICE_CPU, kSeed, 0).err,
            novaSuccess);
  if (wide) {
    std::vector<uint64_t> scratch(demand);
    EXPECT_EQ(generator_fill_u64(&source, scratch.data(), demand).err,
              novaSuccess);
  } else {
    std::vector<uint32_t> scratch(demand);
    EXPECT_EQ(generator_fill_u32(&source, scratch.data(), demand).err,
              novaSuccess);
  }
  return source;
}

/** Builds a Xoshiro source holding one parked half lane via an odd
    u32 demand. */
Generator_ parkedXoshiroSource(size_t demand) {
  Generator_ source = {};
  EXPECT_EQ(generator_init(&source, Xoshiro256, DEVICE_CPU, kSeed, 0).err,
            novaSuccess);
  std::vector<uint32_t> scratch(demand);
  EXPECT_EQ(generator_fill_u32(&source, scratch.data(), demand).err,
            novaSuccess);
  return source;
}

TEST(RngStreamAt, PhiloxU32CursorContinuesTheSerialFill) {
  expectU32CursorsContinueTheSerialFill(Philox4x32);
}

TEST(RngStreamAt, PhiloxU64CursorContinuesTheSerialFill) {
  expectU64CursorsContinueTheSerialFill(Philox4x32);
}

TEST(RngStreamAt, XoshiroU32CursorContinuesTheSerialFill) {
  expectU32CursorsContinueTheSerialFill(Xoshiro256);
}

TEST(RngStreamAt, XoshiroU64CursorContinuesTheSerialFill) {
  expectU64CursorsContinueTheSerialFill(Xoshiro256);
}

TEST(RngStreamAt, PhiloxU64CursorMatchesTheSourceAfterTheSkipFill) {
  const std::pair<size_t, bool> kShapes[] = {
      {5, false},
      {6, false},
      {7, false},
      {5, true},
  };
  for (const auto &shape : kShapes) {
    const Generator_ source = parkedPhiloxSource(shape.first, shape.second);
    for (size_t offset : kNearOffsets) {
      SCOPED_TRACE(::testing::Message()
                   << "demand=" << shape.first << " wide=" << shape.second
                   << " offset=" << offset);
      expectCursorEquivalence(source, true, offset);
    }
  }
}

TEST(RngStreamAt, PhiloxU32CursorMatchesTheSourceAfterTheSkipFill) {
  const std::pair<size_t, bool> kShapes[] = {
      {5, false},
      {6, false},
      {7, false},
      {5, true},
  };
  for (const auto &shape : kShapes) {
    const Generator_ source = parkedPhiloxSource(shape.first, shape.second);
    for (size_t offset : kNearOffsets) {
      SCOPED_TRACE(::testing::Message()
                   << "demand=" << shape.first << " wide=" << shape.second
                   << " offset=" << offset);
      expectCursorEquivalence(source, false, offset);
    }
  }
}

TEST(RngStreamAt, XoshiroU64CursorMatchesTheSourceAfterTheSkipFill) {
  const Generator_ source = parkedXoshiroSource(5);
  for (size_t offset : kNearOffsets) {
    SCOPED_TRACE(::testing::Message() << "offset=" << offset);
    expectCursorEquivalence(source, true, offset);
  }
}

TEST(RngStreamAt, XoshiroU32CursorMatchesTheSourceAfterTheSkipFill) {
  const Generator_ source = parkedXoshiroSource(5);
  for (size_t offset : kNearOffsets) {
    SCOPED_TRACE(::testing::Message() << "offset=" << offset);
    expectCursorEquivalence(source, false, offset);
  }
}

/** A lone u32 lane parked by an odd u32 fill must survive a u64
    cursor: the u64 continuation stays on the pure stream and the
    survivor leads the next u32 fill. */
TEST(RngStreamAt, PhiloxLoneLaneSurvivesAU64Cursor) {
  Generator_ observer = parkedPhiloxSource(7, false);
  uint32_t lone = 0;
  ASSERT_EQ(generator_fill_u32(&observer, &lone, 1).err, novaSuccess);

  Generator_ probe = parkedPhiloxSource(7, false);
  Generator_ mirror = {};
  ASSERT_EQ(generator_init(&mirror, Philox4x32, DEVICE_CPU, kSeed, 0).err,
            novaSuccess);
  std::vector<uint64_t> spent(4);
  ASSERT_EQ(generator_fill_u64(&mirror, spent.data(), 4).err, novaSuccess);

  Generator_ source = parkedPhiloxSource(7, false);
  Generator_ cursor = {};
  ASSERT_EQ(generator_u64_stream_at(&source, 0, &cursor).err, novaSuccess);

  std::vector<uint64_t> tail(kTail);
  std::vector<uint64_t> mirrorTail(kTail);
  ASSERT_EQ(generator_fill_u64(&cursor, tail.data(), kTail).err, novaSuccess);
  ASSERT_EQ(generator_fill_u64(&mirror, mirrorTail.data(), kTail).err,
            novaSuccess);
  EXPECT_EQ(
      0, std::memcmp(tail.data(), mirrorTail.data(), kTail * sizeof(uint64_t)))
      << "the u64 cursor must continue the pure stream, lone lane aside";

  std::vector<uint64_t> probeSpend(kTail);
  ASSERT_EQ(generator_fill_u64(&probe, probeSpend.data(), kTail).err,
            novaSuccess);

  std::vector<uint32_t> narrow(11);
  std::vector<uint32_t> probeNarrow(11);
  ASSERT_EQ(generator_fill_u32(&cursor, narrow.data(), 11).err, novaSuccess);
  ASSERT_EQ(generator_fill_u32(&probe, probeNarrow.data(), 11).err,
            novaSuccess);
  EXPECT_EQ(narrow[0], lone) << "the surviving lane must lead the u32 fill";
  EXPECT_EQ(
      0, std::memcmp(narrow.data(), probeNarrow.data(), 11 * sizeof(uint32_t)))
      << "the u32 continuation after the u64 cursor diverged";
}

/** A parked Xoshiro half lane must survive a u64 cursor: the u64
    continuation stays on the pure sample stream and the survivor
    leads the next u32 fill. */
TEST(RngStreamAt, XoshiroHalfLaneSurvivesAU64Cursor) {
  Generator_ observer = parkedXoshiroSource(5);
  uint32_t lone = 0;
  ASSERT_EQ(generator_fill_u32(&observer, &lone, 1).err, novaSuccess);

  Generator_ probe = parkedXoshiroSource(5);
  Generator_ mirror = {};
  ASSERT_EQ(generator_init(&mirror, Xoshiro256, DEVICE_CPU, kSeed, 0).err,
            novaSuccess);
  std::vector<uint64_t> spent(3);
  ASSERT_EQ(generator_fill_u64(&mirror, spent.data(), 3).err, novaSuccess);

  Generator_ source = parkedXoshiroSource(5);
  Generator_ cursor = {};
  ASSERT_EQ(generator_u64_stream_at(&source, 0, &cursor).err, novaSuccess);

  std::vector<uint64_t> tail(kTail);
  std::vector<uint64_t> mirrorTail(kTail);
  ASSERT_EQ(generator_fill_u64(&cursor, tail.data(), kTail).err, novaSuccess);
  ASSERT_EQ(generator_fill_u64(&mirror, mirrorTail.data(), kTail).err,
            novaSuccess);
  EXPECT_EQ(
      0, std::memcmp(tail.data(), mirrorTail.data(), kTail * sizeof(uint64_t)))
      << "the u64 cursor must continue the pure sample stream";

  std::vector<uint64_t> probeSpend(kTail);
  ASSERT_EQ(generator_fill_u64(&probe, probeSpend.data(), kTail).err,
            novaSuccess);

  std::vector<uint32_t> narrow(11);
  std::vector<uint32_t> probeNarrow(11);
  ASSERT_EQ(generator_fill_u32(&cursor, narrow.data(), 11).err, novaSuccess);
  ASSERT_EQ(generator_fill_u32(&probe, probeNarrow.data(), 11).err,
            novaSuccess);
  EXPECT_EQ(narrow[0], lone) << "the surviving half lane must lead the fill";
  EXPECT_EQ(
      0, std::memcmp(narrow.data(), probeNarrow.data(), 11 * sizeof(uint32_t)))
      << "the u32 continuation after the u64 cursor diverged";
}

/** Two sequential positionings must compose into one deeper one. */
TEST(RngStreamAt, CursorPositioningsCompose) {
  constexpr size_t kFirst = 5;
  constexpr size_t kSecond = 3;

  for (RngEngine engine : {Philox4x32, Xoshiro256}) {
    SCOPED_TRACE(::testing::Message()
                 << "engine=" << (engine == Philox4x32 ? "philox" : "xoshiro"));
    std::vector<uint32_t> reference(kFirst + kSecond + kTail);
    Generator_ serial = {};
    ASSERT_EQ(generator_init(&serial, engine, DEVICE_CPU, kSeed, 0).err,
              novaSuccess);
    ASSERT_EQ(
        generator_fill_u32(&serial, reference.data(), reference.size()).err,
        novaSuccess);

    Generator_ source = {};
    Generator_ mid = {};
    Generator_ deep = {};
    ASSERT_EQ(generator_init(&source, engine, DEVICE_CPU, kSeed, 0).err,
              novaSuccess);
    ASSERT_EQ(generator_u32_stream_at(&source, kFirst, &mid).err, novaSuccess);
    ASSERT_EQ(generator_u32_stream_at(&mid, kSecond, &deep).err, novaSuccess);

    std::vector<uint32_t> tail(kTail);
    ASSERT_EQ(generator_fill_u32(&deep, tail.data(), kTail).err, novaSuccess);
    EXPECT_EQ(0, std::memcmp(reference.data() + kFirst + kSecond, tail.data(),
                             kTail * sizeof(uint32_t)))
        << "chained positionings must compose into one deeper offset";
  }
}

/** Null sources, null cursors, non CPU sources, and offsets
    overflowing the emitted value counter must be rejected. */
TEST(RngStreamAt, CursorGuardsRejectNullsDevicesAndOverflow) {
  Generator_ cpu = {};
  ASSERT_EQ(generator_init(&cpu, Philox4x32, DEVICE_CPU, kSeed, 0).err,
            novaSuccess);
  Generator_ cursor = {};

  EXPECT_EQ(generator_u32_stream_at(nullptr, 0, &cursor).err,
            novaInvalidPointer);
  EXPECT_EQ(generator_u64_stream_at(nullptr, 0, &cursor).err,
            novaInvalidPointer);
  EXPECT_EQ(generator_u32_stream_at(&cpu, 0, nullptr).err, novaInvalidPointer);
  EXPECT_EQ(generator_u64_stream_at(&cpu, 0, nullptr).err, novaInvalidPointer);

  Generator_ gpu = {};
  ASSERT_EQ(generator_init(&gpu, Philox4x32, DEVICE_GPU, kSeed, 0).err,
            novaSuccess)
      << "init performs no device probe, the pair must construct";
  EXPECT_EQ(generator_u32_stream_at(&gpu, 0, &cursor).err, novaInvalidDevice);
  EXPECT_EQ(generator_u64_stream_at(&gpu, 0, &cursor).err, novaInvalidDevice);

  cpu.offset = UINT64_MAX;
  EXPECT_EQ(generator_u32_stream_at(&cpu, 1, &cursor).err, novaInvalidValue);
  EXPECT_EQ(generator_u64_stream_at(&cpu, 1, &cursor).err, novaInvalidValue);
}

} // namespace
