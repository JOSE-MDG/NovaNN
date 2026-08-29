/**
 * @file Float8_e5m2_test.cpp
 * @brief Exhaustive bit-pattern identity tests for FP8 E5M2.
 *
 * Sweeps all 256 storage patterns and verifies that:
 * @li numeric decode matches an independently derived reference value,
 * @li decode/re-encode round-trips every finite and infinity pattern (NaN
 *     payloads are canonicalized to 0x7F|sign by the encoder, so they get
 *     their own class/sign test),
 * @li the bit-level API agrees with the numeric API,
 * @li the struct predicates partition the domain exactly: inf at 0x7C|sign,
 *     NaN for anything above, disjoint and complete.
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>

#include <ncore/core/dtype.h>
#include <ncore/core/fp_utils.h>
#include <ncore/headeronly/dtypes/fp8_e5m2.hh>

#include "utils/FloatingPointClassification.hpp"

namespace {

using tests::fpc::f32BitsOf;
using tests::fpc::isInfF32Bits;
using tests::fpc::isInfPatternE5M2;
using tests::fpc::isNaNF32Bits;
using tests::fpc::isNaNPatternE5M2;
using tests::fpc::referenceDecodeFp8E5M2;
using tests::fpc::signBitF32Bits;

/// Total number of E5M2 storage patterns.
constexpr uint32_t kFp8Patterns = 0x100U;

} // namespace

/**
 * @brief Verifies numeric decode against the independent reference over the
 *        full 8-bit domain. NaN classification is decided on the raw
 *        pattern bits.
 */
TEST(BitPatternIdentityFloat8_e5m2, AllPatternsDecodeToReferenceValue) {
  for (uint32_t b = 0; b < kFp8Patterns; ++b) {
    const auto bits = static_cast<uint8_t>(b);
    const float got = fp8e5m2_to_float(bits);
    const uint32_t gotBits = f32BitsOf(got);
    if (isNaNPatternE5M2(bits)) {
      EXPECT_TRUE(isNaNF32Bits(gotBits)) << "pattern=0x" << std::hex << b;
      EXPECT_EQ(signBitF32Bits(gotBits), (bits >> 7) != 0)
          << "NaN sign mismatch, pattern=0x" << std::hex << b;
    } else {
      const double want = referenceDecodeFp8E5M2(bits);
      EXPECT_FLOAT_EQ(got, static_cast<float>(want))
          << "pattern=0x" << std::hex << b;
    }
  }
}

/**
 * @brief Verifies decode/re-encode round-trip identity for every finite and
 *        infinity pattern. NaN patterns are excluded deliberately: the
 *        encoder canonicalizes NaN payloads to 0x7F|sign by design.
 */
TEST(BitPatternIdentityFloat8_e5m2,
     DecodeReencodeRoundTripsFiniteAndInfPatterns) {
  for (uint32_t b = 0; b < kFp8Patterns; ++b) {
    const auto bits = static_cast<uint8_t>(b);
    if (isNaNPatternE5M2(bits)) {
      continue;
    }
    EXPECT_EQ(fp8e5m2_from_float(fp8e5m2_to_float(bits)), bits)
        << "pattern=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies that every NaN pattern re-encodes to the canonical NaN
 *        with its sign preserved, and decodes back as a NaN of that sign.
 */
TEST(BitPatternIdentityFloat8_e5m2, NanPatternsCanonicalizeOnReencode) {
  for (uint32_t b = 0; b < kFp8Patterns; ++b) {
    const auto bits = static_cast<uint8_t>(b);
    if (!isNaNPatternE5M2(bits)) {
      continue;
    }
    const uint8_t reencoded = fp8e5m2_from_float(fp8e5m2_to_float(bits));
    EXPECT_EQ(reencoded, static_cast<uint8_t>((bits & 0x80U) | 0x7FU))
        << "pattern=0x" << std::hex << b;
    EXPECT_TRUE(isNaNPatternE5M2(reencoded)) << "pattern=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies that the bit-level widening API equals the numeric decode
 *        over the full domain (also validates the FP16-reuse decode trick).
 */
TEST(BitPatternIdentityFloat8_e5m2, ToF32BitsMatchesNumericDecode) {
  for (uint32_t b = 0; b < kFp8Patterns; ++b) {
    const auto bits = static_cast<uint8_t>(b);
    const uint32_t viaBitApi = fp8e5m2_to_f32_bits(bits);
    const uint32_t viaNumeric = f32BitsOf(fp8e5m2_to_float(bits));
    EXPECT_EQ(viaBitApi, viaNumeric) << "pattern=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies that fp8e5m2_from_f32_bits inverts fp8e5m2_to_f32_bits
 *        for every non-NaN pattern. NaN patterns are excluded: the encoder
 *        canonicalizes NaN payloads, so the inverse property cannot hold
 *        for them (0x7D widens to a payload-carrying NaN but re-encodes to
 *        the canonical 0x7F).
 */
TEST(BitPatternIdentityFloat8_e5m2, FromF32BitsInvertsToF32Bits) {
  for (uint32_t b = 0; b < kFp8Patterns; ++b) {
    const auto bits = static_cast<uint8_t>(b);
    if (isNaNPatternE5M2(bits)) {
      continue;
    }
    const uint32_t wide = fp8e5m2_to_f32_bits(bits);
    const uint32_t roundTrip = fp8e5m2_to_f32_bits(fp8e5m2_from_f32_bits(wide));
    EXPECT_EQ(roundTrip, wide) << "pattern=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies that the struct predicates partition the domain exactly:
 *        inf set {0x7C, 0xFC}, NaN set {0x7D..0x7F} plus sign variants,
 *        disjoint from each other and from the finite domain.
 */
TEST(BitPatternIdentityFloat8_e5m2, StructPredicatesMatchFormatDefinition) {
  for (uint32_t b = 0; b < kFp8Patterns; ++b) {
    const auto bits = static_cast<uint8_t>(b);
    const ncore::dtypes::Float8_e5m2 v{bits,
                                       ncore::dtypes::Float8_e5m2::from_bits()};
    EXPECT_EQ(v.isnan(), isNaNPatternE5M2(bits))
        << "pattern=0x" << std::hex << b;
    EXPECT_EQ(v.isinf(), isInfPatternE5M2(bits))
        << "pattern=0x" << std::hex << b;
    EXPECT_FALSE(v.isnan() && v.isinf())
        << "predicates overlap at pattern=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies bitwise agreement between the struct conversion and the
 *        extern C API over the full domain.
 */
TEST(BitPatternIdentityFloat8_e5m2, StructConversionAgreesWithCApi) {
  for (uint32_t b = 0; b < kFp8Patterns; ++b) {
    const auto bits = static_cast<uint8_t>(b);
    const ncore::dtypes::Float8_e5m2 v{bits,
                                       ncore::dtypes::Float8_e5m2::from_bits()};
    const uint32_t viaStruct = f32BitsOf(static_cast<float>(v));
    const uint32_t viaCApi = f32BitsOf(fp8e5m2_to_float(bits));
    EXPECT_EQ(viaStruct, viaCApi) << "pattern=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies textbook sentinel values as human-readable anchors.
 */
TEST(BitPatternIdentityFloat8_e5m2, KnownSpecialsDecodeExactly) {
  struct Sentinel {
    uint8_t bits;
    double want;
    bool isNan;
    bool isInf;
    const char *name;
  };
  constexpr std::array<Sentinel, 11> kSentinels{{
      {.bits = 0x00,
       .want = 0.0,
       .isNan = false,
       .isInf = false,
       .name = "+0.0"},
      {.bits = 0x80,
       .want = -0.0,
       .isNan = false,
       .isInf = false,
       .name = "-0.0"},
      {.bits = 0x3C,
       .want = 1.0,
       .isNan = false,
       .isInf = false,
       .name = "1.0"},
      {.bits = 0x04,
       .want = 6.103515625e-05,
       .isNan = false,
       .isInf = false,
       .name = "min normal 2^-14"},
      {.bits = 0x01,
       .want = 1.52587890625e-05,
       .isNan = false,
       .isInf = false,
       .name = "min subnormal 2^-16"},
      {.bits = 0x03,
       .want = 4.57763671875e-05,
       .isNan = false,
       .isInf = false,
       .name = "max subnormal"},
      {.bits = 0x42,
       .want = 3.0,
       .isNan = false,
       .isInf = false,
       .name = "3.0"},
      {.bits = 0x7B,
       .want = 57344.0,
       .isNan = false,
       .isInf = false,
       .name = "max finite +57344"},
      {.bits = 0xFB,
       .want = -57344.0,
       .isNan = false,
       .isInf = false,
       .name = "max finite -57344"},
      {.bits = 0x7C,
       .want = HUGE_VAL,
       .isNan = false,
       .isInf = true,
       .name = "+inf"},
      {.bits = 0x7F,
       .want = HUGE_VAL,
       .isNan = true,
       .isInf = false,
       .name = "canonical NaN"},
  }};
  for (const auto &s : kSentinels) {
    const double got = referenceDecodeFp8E5M2(s.bits);
    if (s.isNan) {
      EXPECT_TRUE(std::isnan(got)) << s.name;
    } else if (s.isInf) {
      EXPECT_TRUE(std::isinf(got)) << s.name;
      EXPECT_DOUBLE_EQ(got, s.want) << s.name;
    } else {
      ASSERT_FALSE(std::isnan(got)) << s.name;
      EXPECT_DOUBLE_EQ(got, s.want) << s.name;
    }

    // Pattern-space classification on raw bits (no FP round-trip).
    const uint32_t viaBits = f32BitsOf(fp8e5m2_to_float(s.bits));
    if (s.isNan) {
      EXPECT_TRUE(isNaNF32Bits(viaBits)) << s.name;
    } else if (s.isInf) {
      EXPECT_TRUE(isInfF32Bits(viaBits)) << s.name;
    } else {
      EXPECT_FLOAT_EQ(fp8e5m2_to_float(s.bits), static_cast<float>(s.want))
          << s.name;
    }
  }
}
