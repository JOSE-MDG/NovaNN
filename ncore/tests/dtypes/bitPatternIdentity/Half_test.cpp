/**
 * @file Half_test.cpp
 * @brief Exhaustive bit-pattern identity tests for IEEE 754 binary16 (FP16).
 *
 * Sweeps all 65536 storage patterns and verifies that:
 * @li numeric decode matches an independently derived reference value,
 * @li decoding an exact value and re-encoding it reproduces the original
 *     bits (finite patterns; NaN payloads are canonicalized by design on
 *     some conversion paths, so they get their own class/sign test),
 * @li the bit-level API agrees with the numeric API,
 * @li the Half struct conversion agrees with the extern C API.
 *
 * The conversion takes different paths per build (native _Float16
 * static_cast, F16C intrinsics, or software bit manipulation); exhaustive
 * identity is the strongest portable check that all of them agree with the
 * format.
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <ncore/core/dtype.h>
#include <ncore/core/fp_utils.h>
#include <ncore/headeronly/dtypes/half.hh>

#include "utils/FloatingPointClassification.hpp"

namespace {

using tests::fpc::f32BitsOf;
using tests::fpc::isNaNF32Bits;
using tests::fpc::referenceDecodeFp16;
using tests::fpc::signBitF32Bits;

/// Total number of FP16 storage patterns.
constexpr uint32_t kFp16Patterns = 0x10000U;

/// Reinterprets a raw 16-bit pattern as the float16 storage alias.
inline float16 bitsToFp16(uint16_t bits) {
  // float16 is _Float16 on GCC/Clang: a 16-bit trivially copyable type.
  float16 out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

/**
 * @brief True when a pattern's exponent field is all ones with nonzero
 *        mantissa (the IEEE NaN domain).
 */
constexpr bool isNanPattern(uint16_t bits) {
  return (bits & 0x7C00U) == 0x7C00U && (bits & 0x03FFU) != 0U;
}

} // namespace

/**
 * @brief Verifies numeric decode against the independent reference over the
 *        full 16-bit domain.
 *
 * NaN classification is decided on raw bits: the input is a 16-bit
 * pattern, not a float object.
 */
TEST(BitPatternIdentityHalf, AllPatternsDecodeToReferenceValue) {
  for (uint32_t b = 0; b < kFp16Patterns; ++b) {
    const auto bits = static_cast<uint16_t>(b);
    const float got = fp16_to_float(bitsToFp16(bits));
    const uint32_t gotBits = f32BitsOf(got);
    if (isNanPattern(bits)) {
      EXPECT_TRUE(isNaNF32Bits(gotBits)) << "pattern=0x" << std::hex << b;
      EXPECT_EQ(signBitF32Bits(gotBits), (bits >> 15) != 0)
          << "NaN sign mismatch, pattern=0x" << std::hex << b;
    } else {
      const double want = referenceDecodeFp16(bits);
      EXPECT_FLOAT_EQ(got, static_cast<float>(want))
          << "pattern=0x" << std::hex << b;
    }
  }
}

/**
 * @brief Verifies decode/re-encode round-trip identity for every non-NaN
 *        pattern. NaN patterns are excluded deliberately: the software
 *        encoder canonicalizes NaN payloads to 0x7E00 by design, so payload
 *        round-trips are not part of the contract.
 */
TEST(BitPatternIdentityHalf, DecodeReencodeRoundTripsFinitePatterns) {
  for (uint32_t b = 0; b < kFp16Patterns; ++b) {
    const auto bits = static_cast<uint16_t>(b);
    if (isNanPattern(bits)) {
      continue;
    }
    const float16 h = bitsToFp16(bits);
    EXPECT_EQ(fp16_from_float(fp16_to_float(h)), h)
        << "pattern=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies that every NaN pattern decodes to a NaN carrying the same
 *        sign as the source pattern (bit-level classification).
 */
TEST(BitPatternIdentityHalf, NanDecodesPreserveClassAndSign) {
  for (uint32_t b = 0; b < kFp16Patterns; ++b) {
    const auto bits = static_cast<uint16_t>(b);
    if (!isNanPattern(bits)) {
      continue;
    }
    const uint32_t gotBits = f32BitsOf(fp16_to_float(bitsToFp16(bits)));
    EXPECT_TRUE(isNaNF32Bits(gotBits)) << "pattern=0x" << std::hex << b;
    EXPECT_EQ(signBitF32Bits(gotBits), (bits >> 15) != 0)
        << "NaN sign mismatch, pattern=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies that the bit-level widening API equals the numeric decode
 *        over the full domain.
 *
 * For NaN patterns only class and sign are required to agree: the native
 * _Float16 widening quiets signaling NaNs, while the software bit API
 * preserves payloads, so bitwise equality is not part of the contract there.
 */
TEST(BitPatternIdentityHalf, ToF32BitsMatchesNumericDecode) {
  for (uint32_t b = 0; b < kFp16Patterns; ++b) {
    const auto bits = static_cast<uint16_t>(b);
    const uint32_t viaBitApi = fp16_to_f32_bits(bitsToFp16(bits));
    const uint32_t viaNumeric = f32BitsOf(fp16_to_float(bitsToFp16(bits)));
    if (isNanPattern(bits)) {
      EXPECT_TRUE(isNaNF32Bits(viaBitApi)) << "pattern=0x" << std::hex << b;
      EXPECT_TRUE(isNaNF32Bits(viaNumeric)) << "pattern=0x" << std::hex << b;
      EXPECT_EQ(signBitF32Bits(viaBitApi), signBitF32Bits(viaNumeric))
          << "pattern=0x" << std::hex << b;
    } else {
      EXPECT_EQ(viaBitApi, viaNumeric) << "pattern=0x" << std::hex << b;
    }
  }
}

/**
 * @brief Verifies that fp16_from_f32_bits inverts fp16_to_f32_bits for
 *        every non-NaN pattern.
 */
TEST(BitPatternIdentityHalf, FromF32BitsInvertsToF32Bits) {
  for (uint32_t b = 0; b < kFp16Patterns; ++b) {
    const auto bits = static_cast<uint16_t>(b);
    if (isNanPattern(bits)) {
      continue;
    }
    const uint32_t wide = fp16_to_f32_bits(bitsToFp16(bits));
    const uint32_t roundTrip = fp16_to_f32_bits(fp16_from_f32_bits(wide));
    EXPECT_EQ(roundTrip, wide) << "pattern=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies that fp16_to_bits is the identity over the full domain.
 */
TEST(BitPatternIdentityHalf, StorageBitsAreIdentity) {
  for (uint32_t b = 0; b < kFp16Patterns; ++b) {
    const auto bits = static_cast<uint16_t>(b);
    const ncore::dtypes::Half half{bits, ncore::dtypes::Half::from_bits()};
    EXPECT_EQ(fp16_to_bits(bitsToFp16(half.x)), bits)
        << "pattern=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies bitwise agreement between the Half struct conversion and
 *        the extern C API over the full domain.
 */
TEST(BitPatternIdentityHalf, StructConversionAgreesWithCApi) {
  for (uint32_t b = 0; b < kFp16Patterns; ++b) {
    const auto bits = static_cast<uint16_t>(b);
    const ncore::dtypes::Half half{bits, ncore::dtypes::Half::from_bits()};
    const uint32_t viaStruct = f32BitsOf(static_cast<float>(half));
    const uint32_t viaCApi = f32BitsOf(fp16_to_float(bitsToFp16(bits)));
    EXPECT_EQ(viaStruct, viaCApi) << "pattern=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies textbook sentinel values as human-readable anchors.
 */
TEST(BitPatternIdentityHalf, KnownSpecialsDecodeExactly) {
  struct Sentinel {
    uint16_t bits;
    double want;
    const char *name;
  };
  constexpr std::array<Sentinel, 10> kSentinels{{
      {.bits = 0x0000, .want = 0.0, .name = "+0.0"},
      {.bits = 0x8000, .want = -0.0, .name = "-0.0"},
      {.bits = 0x3C00, .want = 1.0, .name = "1.0"},
      {.bits = 0x3555, .want = 0.333251953125, .name = "0.333251953125"},
      {.bits = 0x7C00, .want = HUGE_VAL, .name = "+inf"},
      {.bits = 0xFC00, .want = -HUGE_VAL, .name = "-inf"},
      {.bits = 0x0400, .want = 6.103515625e-05, .name = "min normal 2^-14"},
      {.bits = 0x0001,
       .want = 5.9604644775390625e-08,
       .name = "min subnormal 2^-24"},
      {.bits = 0x03FF, .want = 6.097555160522461e-05, .name = "max subnormal"},
      {.bits = 0x7BFF, .want = 65504.0, .name = "max finite"},
  }};
  for (const auto &s : kSentinels) {
    const double got = referenceDecodeFp16(s.bits);
    if (std::isnan(s.want)) {
      EXPECT_TRUE(std::isnan(got)) << s.name;
      continue;
    }
    EXPECT_DOUBLE_EQ(got, s.want) << s.name;

    const float viaKernel = fp16_to_float(bitsToFp16(s.bits));
    EXPECT_FLOAT_EQ(viaKernel, static_cast<float>(s.want)) << s.name;
  }
}
