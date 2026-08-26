/**
 * @file Float4_e2m1fn_x2_test.cpp
 * @brief Exhaustive bit-pattern identity tests for pair-packed FP4 E2M1FN.
 *
 * Covers both domains exhaustively: all 16 nibble patterns as lane values
 * and all 256 packed bytes as lane pairs. Verifies:
 * @li lane decode against an independently declared magnitude table,
 * @li the low-nibble-first lane order end to end,
 * @li pack/unpack round-trip identity over the full byte domain,
 * @li the fp4e2m1x2_to_f32_bits / from_f32_bits decomposition contract,
 * @li the finite-only predicate contract (isnan/isinf always false).
 */

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include <ncore/core/dtype.h>
#include <ncore/core/fp_utils.h>
#include <ncore/headeronly/dtypes/fp4_e2m1fn_x2.hh>

#include "utils/FloatingPointClassification.hpp"

namespace {

using tests::fpc::referenceDecodeFp4Nibble;

/// Number of distinct FP4 nibble patterns.
constexpr uint32_t kFp4Nibbles = 0x10U;

/// Number of distinct packed-byte patterns.
constexpr uint32_t kFp4PackedBytes = 0x100U;

} // namespace

/**
 * @brief Verifies scalar-lane decode against the independent magnitude
 *        table over all 16 nibbles.
 */
TEST(BitPatternIdentityFloat4_e2m1fn_x2, AllNibblesDecodeToTableMagnitudes) {
  for (uint32_t n = 0; n < kFp4Nibbles; ++n) {
    const auto nibble = static_cast<uint8_t>(n);
    const float want = referenceDecodeFp4Nibble(nibble);
    const ncore::dtypes::Float4_e2m1fn lane{
        nibble, ncore::dtypes::Float4_e2m1fn::from_bits()};
    EXPECT_FLOAT_EQ(static_cast<float>(lane), want)
        << "nibble=0x" << std::hex << n;
  }
}

/**
 * @brief Verifies that all 256 packed bytes decode into two independent
 *        lanes matching the reference.
 */
TEST(BitPatternIdentityFloat4_e2m1fn_x2,
     AllPackedBytesDecodeLanesIndependently) {
  for (uint32_t b = 0; b < kFp4PackedBytes; ++b) {
    const auto byte = static_cast<float4_e2m1fn_x2>(b);
    float lo = 0.0F;
    float hi = 0.0F;
    fp4e2m1x2_to_floats(byte, &lo, &hi);
    EXPECT_FLOAT_EQ(lo, referenceDecodeFp4Nibble(b & 0xFU))
        << "byte=0x" << std::hex << b << " (low lane)";
    EXPECT_FLOAT_EQ(hi, referenceDecodeFp4Nibble(b >> 4))
        << "byte=0x" << std::hex << b << " (high lane)";
  }
}

/**
 * @brief Pins the lane order with sentinel bytes: the low nibble feeds the
 *        first output (and even tensor indices), the high nibble the second.
 */
TEST(BitPatternIdentityFloat4_e2m1fn_x2, LaneOrderIsLowNibbleFirst) {
  // 0x21: low nibble 1 -> +0.5, high nibble 2 -> +1.0.
  float lo = 0.0F;
  float hi = 0.0F;
  fp4e2m1x2_to_floats(static_cast<float4_e2m1fn_x2>(0x21), &lo, &hi);
  EXPECT_FLOAT_EQ(lo, 0.5F);
  EXPECT_FLOAT_EQ(hi, 1.0F);

  // 0xF1: low nibble 1 -> +0.5, high nibble F -> -6.0 (sign lives per lane).
  fp4e2m1x2_to_floats(static_cast<float4_e2m1fn_x2>(0xF1), &lo, &hi);
  EXPECT_FLOAT_EQ(lo, 0.5F);
  EXPECT_FLOAT_EQ(hi, -6.0F);
}

/**
 * @brief Verifies pack/unpack round-trip identity over all 256 bytes.
 */
TEST(BitPatternIdentityFloat4_e2m1fn_x2, PackUnpackRoundTripsAllBytes) {
  for (uint32_t b = 0; b < kFp4PackedBytes; ++b) {
    const auto byte = static_cast<float4_e2m1fn_x2>(b);
    float lo = 0.0F;
    float hi = 0.0F;
    fp4e2m1x2_to_floats(byte, &lo, &hi);
    const auto repacked = fp4e2m1x2_from_floats(lo, hi);
    EXPECT_EQ(static_cast<uint8_t>(repacked), static_cast<uint8_t>(b))
        << "byte=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies the documented decomposition contract: lo/hi hold the raw
 *        nibbles and val reproduces the original byte as (hi << 4) | lo.
 */
TEST(BitPatternIdentityFloat4_e2m1fn_x2, DecompositionApiSplitsAndRestores) {
  for (uint32_t b = 0; b < kFp4PackedBytes; ++b) {
    const auto byte = static_cast<float4_e2m1fn_x2>(b);
    const fp4e2m1x2Result_t parts = fp4e2m1x2_to_f32_bits(byte);

    EXPECT_EQ(parts.lo, b & 0xFU)
        << "byte=0x" << std::hex << b << " (lo field)";
    EXPECT_EQ(parts.hi, (b >> 4) & 0xFU)
        << "byte=0x" << std::hex << b << " (hi field)";
    EXPECT_EQ(parts.val, byte)
        << "byte=0x" << std::hex << b << " (val round trip)";

    EXPECT_EQ(fp4e2m1x2_from_f32_bits(&parts), byte)
        << "byte=0x" << std::hex << b << " (from_f32_bits restore)";
  }
}

/**
 * @brief Verifies the finite-only contract: isnan()/isinf() are false for
 *        every nibble and every lane of every packed byte.
 */
TEST(BitPatternIdentityFloat4_e2m1fn_x2, StructPredicatesAlwaysFalse) {
  for (uint32_t n = 0; n < kFp4Nibbles; ++n) {
    const auto nibble = static_cast<uint8_t>(n);
    const ncore::dtypes::Float4_e2m1fn lane{
        nibble, ncore::dtypes::Float4_e2m1fn::from_bits()};
    EXPECT_FALSE(lane.isnan()) << "nibble=0x" << std::hex << n;
    EXPECT_FALSE(lane.isinf()) << "nibble=0x" << std::hex << n;
  }
  for (uint32_t b = 0; b < kFp4PackedBytes; ++b) {
    const ncore::dtypes::Float4_e2m1fn_x2 packed(static_cast<uint8_t>(b));
    EXPECT_FALSE(packed.low().isnan()) << "byte=0x" << std::hex << b;
    EXPECT_FALSE(packed.low().isinf()) << "byte=0x" << std::hex << b;
    EXPECT_FALSE(packed.high().isnan()) << "byte=0x" << std::hex << b;
    EXPECT_FALSE(packed.high().isinf()) << "byte=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies agreement between the scalar struct conversion, the
 *        packed-struct lane accessors, and the independent reference.
 */
TEST(BitPatternIdentityFloat4_e2m1fn_x2, ScalarStructAgreesWithPackedApi) {
  for (uint32_t n = 0; n < kFp4Nibbles; ++n) {
    const auto nibble = static_cast<uint8_t>(n);
    const float want = referenceDecodeFp4Nibble(nibble);

    const ncore::dtypes::Float4_e2m1fn lane{
        nibble, ncore::dtypes::Float4_e2m1fn::from_bits()};
    EXPECT_FLOAT_EQ(static_cast<float>(lane), want)
        << "nibble=0x" << std::hex << n;

    const ncore::dtypes::Float4_e2m1fn_x2 packed =
        ncore::dtypes::Float4_e2m1fn_x2(lane, lane);
    EXPECT_FLOAT_EQ(static_cast<float>(packed.low()), want)
        << "nibble=0x" << std::hex << n << " (low)";
    EXPECT_FLOAT_EQ(static_cast<float>(packed.high()), want)
        << "nibble=0x" << std::hex << n << " (high)";
  }
}
