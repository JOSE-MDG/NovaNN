/**
 * @file Float8_e4m3fn_test.cpp
 * @brief Exhaustive bit-pattern identity tests for FP8 E4M3FN.
 *
 * Sweeps all 256 storage patterns and verifies that:
 * @li numeric decode matches an independently derived reference value,
 * @li decode/re-encode round-trips every pattern (including both NaN
 *     patterns: E4M3FN has no infinity, and its NaN encoding survives the
 *     encode side's sign handling),
 * @li the bit-level API agrees with the numeric API,
 * @li the struct predicates match the format definition: NaN exactly at
 *     0x7F|sign, and isinf() false everywhere (finite-only format).
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>

#include <ncore/core/dtype.h>
#include <ncore/core/fp_utils.h>
#include <ncore/headeronly/dtypes/fp8_e4m3fn.hh>

#include "utils/FloatingPointClassification.hpp"

namespace {

using tests::fpc::f32BitsOf;
using tests::fpc::isNaNF32Bits;
using tests::fpc::isNaNPatternE4M3;
using tests::fpc::referenceDecodeFp8E4M3;
using tests::fpc::signBitF32Bits;

/// Total number of E4M3FN storage patterns.
constexpr uint32_t kFp8Patterns = 0x100U;

} // namespace

/**
 * @brief Verifies numeric decode against the independent reference over the
 *        full 8-bit domain. NaN classification is decided on the raw
 *        pattern bits.
 */
TEST(BitPatternIdentityFloat8_e4m3fn, AllPatternsDecodeToReferenceValue) {
  for (uint32_t b = 0; b < kFp8Patterns; ++b) {
    const auto bits = static_cast<uint8_t>(b);
    const float got = fp8e4m3fn_to_float(bits);
    const uint32_t gotBits = f32BitsOf(got);
    if (isNaNPatternE4M3(bits)) {
      EXPECT_TRUE(isNaNF32Bits(gotBits)) << "pattern=0x" << std::hex << b;
      EXPECT_EQ(signBitF32Bits(gotBits), (bits >> 7) != 0)
          << "NaN sign mismatch, pattern=0x" << std::hex << b;
    } else {
      const double want = referenceDecodeFp8E4M3(bits);
      EXPECT_FLOAT_EQ(got, static_cast<float>(want))
          << "pattern=0x" << std::hex << b;
    }
  }
}

/**
 * @brief Verifies decode/re-encode round-trip identity for every pattern,
 *        including both NaN encodings.
 */
TEST(BitPatternIdentityFloat8_e4m3fn, DecodeReencodeRoundTripsAllPatterns) {
  for (uint32_t b = 0; b < kFp8Patterns; ++b) {
    const auto bits = static_cast<uint8_t>(b);
    EXPECT_EQ(fp8e4m3fn_from_float(fp8e4m3fn_to_float(bits)), bits)
        << "pattern=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies that the bit-level widening API equals the numeric decode
 *        over the full domain.
 */
TEST(BitPatternIdentityFloat8_e4m3fn, ToF32BitsMatchesNumericDecode) {
  for (uint32_t b = 0; b < kFp8Patterns; ++b) {
    const auto bits = static_cast<uint8_t>(b);
    const uint32_t viaBitApi = fp8e4m3fn_to_f32_bits(bits);
    const uint32_t viaNumeric = f32BitsOf(fp8e4m3fn_to_float(bits));
    EXPECT_EQ(viaBitApi, viaNumeric) << "pattern=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies that fp8e4m3fn_from_f32_bits inverts fp8e4m3fn_to_f32_bits
 *        over the full domain.
 */
TEST(BitPatternIdentityFloat8_e4m3fn, FromF32BitsInvertsToF32Bits) {
  for (uint32_t b = 0; b < kFp8Patterns; ++b) {
    const auto bits = static_cast<uint8_t>(b);
    const uint32_t wide = fp8e4m3fn_to_f32_bits(bits);
    const uint32_t roundTrip =
        fp8e4m3fn_to_f32_bits(fp8e4m3fn_from_f32_bits(wide));
    EXPECT_EQ(roundTrip, wide) << "pattern=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies that the struct isnan() predicate matches the independent
 *        classifier over the full domain.
 */
TEST(BitPatternIdentityFloat8_e4m3fn, StructIsnanMatchesFormatDefinition) {
  for (uint32_t b = 0; b < kFp8Patterns; ++b) {
    const auto bits = static_cast<uint8_t>(b);
    const ncore::dtypes::Float8_e4m3fn v{
        bits, ncore::dtypes::Float8_e4m3fn::from_bits()};
    EXPECT_EQ(v.isnan(), isNaNPatternE4M3(bits))
        << "pattern=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies the finite-only contract: isinf() is false for every
 *        one of the 256 patterns.
 */
TEST(BitPatternIdentityFloat8_e4m3fn, StructIsinfAlwaysFalse) {
  for (uint32_t b = 0; b < kFp8Patterns; ++b) {
    const auto bits = static_cast<uint8_t>(b);
    const ncore::dtypes::Float8_e4m3fn v{
        bits, ncore::dtypes::Float8_e4m3fn::from_bits()};
    EXPECT_FALSE(v.isinf()) << "pattern=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies bitwise agreement between the struct conversion and the
 *        extern C API over the full domain.
 */
TEST(BitPatternIdentityFloat8_e4m3fn, StructConversionAgreesWithCApi) {
  for (uint32_t b = 0; b < kFp8Patterns; ++b) {
    const auto bits = static_cast<uint8_t>(b);
    const ncore::dtypes::Float8_e4m3fn v{
        bits, ncore::dtypes::Float8_e4m3fn::from_bits()};
    const uint32_t viaStruct = f32BitsOf(static_cast<float>(v));
    const uint32_t viaCApi = f32BitsOf(fp8e4m3fn_to_float(bits));
    EXPECT_EQ(viaStruct, viaCApi) << "pattern=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies textbook sentinel values as human-readable anchors.
 */
TEST(BitPatternIdentityFloat8_e4m3fn, KnownSpecialsDecodeExactly) {
  struct Sentinel {
    uint8_t bits;
    double want;
    bool isNan;
    const char *name;
  };
  constexpr std::array<Sentinel, 10> kSentinels{{
      {.bits = 0x00, .want = 0.0, .isNan = false, .name = "+0.0"},
      {.bits = 0x80, .want = -0.0, .isNan = false, .name = "-0.0"},
      {.bits = 0x38, .want = 1.0, .isNan = false, .name = "1.0"},
      {.bits = 0x08,
       .want = 0.015625,
       .isNan = false,
       .name = "min normal 2^-6"},
      {.bits = 0x01,
       .want = 0.001953125,
       .isNan = false,
       .name = "min subnormal 2^-9"},
      {.bits = 0x07,
       .want = 0.013671875,
       .isNan = false,
       .name = "max subnormal 7*2^-9"},
      {.bits = 0x40, .want = 2.0, .isNan = false, .name = "2.0"},
      {.bits = 0x7E, .want = 448.0, .isNan = false, .name = "max finite +448"},
      {.bits = 0xFE, .want = -448.0, .isNan = false, .name = "max finite -448"},
      {.bits = 0x7F, .want = HUGE_VAL, .isNan = true, .name = "canonical NaN"},
  }};
  for (const auto &s : kSentinels) {
    const double got = referenceDecodeFp8E4M3(s.bits);
    if (s.isNan) {
      EXPECT_TRUE(std::isnan(got)) << s.name;
    } else {
      ASSERT_FALSE(std::isnan(got)) << s.name;
      EXPECT_DOUBLE_EQ(got, s.want) << s.name;
    }

    const float viaKernel = fp8e4m3fn_to_float(s.bits);
    if (s.isNan) {
      EXPECT_TRUE(std::isnan(viaKernel)) << s.name;
    } else {
      EXPECT_FLOAT_EQ(viaKernel, static_cast<float>(s.want)) << s.name;
    }
  }
}
