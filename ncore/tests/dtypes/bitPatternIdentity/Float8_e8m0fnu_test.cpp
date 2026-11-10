/**
 * @file Float8_e8m0fnu_test.cpp
 * @brief Exhaustive bit-pattern tests for the E8M0 scale factor.
 *
 * Sweeps all 256 storage patterns and pins:
 * @li numeric decode matches the OCP MX formula (0x00 widens to the float32
 *     subnormal 2^-127, 0xFF widens to a quiet NaN, every other pattern is
 *     the byte shifted into the exponent field),
 * @li decode/re-encode round-trips every pattern (0xFF included: the single
 *     NaN pattern canonicalizes onto itself),
 * @li the exponent RNE encode matches an independently derived frexp-based
 *     reference over curated corners plus a deterministic random sweep, and
 *     treats float32 NaN/inf and the overflow leg as documented (0xFF),
 * @li the cuBLASLt round-up derivation keeps the exponent on exact powers of
 *     two, rounds one step up otherwise, and clamps at 0xFE.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <ncore/core/dtype.h>
#include <ncore/core/fp_utils.h>
#include <ncore/headeronly/cast.h>
#include <ncore/headeronly/dtypes/fp8_e8m0fnu.hh>
#include <ncore/headeronly/wrappers/tensor.hh>

#include <tests/oracle/CastingOracle.hpp>

namespace {

using ncore::dtypes::Float8_e8m0fnu;

/// Total number of E8M0 storage patterns.
constexpr uint32_t kE8M0Patterns = 0x100U;

/// Reinterprets a float as its 32-bit pattern.
uint32_t f32BitsOf(float v) {
  uint32_t w = 0;
  std::memcpy(&w, &v, sizeof(w));
  return w;
}

/// Reinterprets a 32-bit pattern as a float.
float f32OfBits(uint32_t w) {
  float v = 0.0F;
  std::memcpy(&v, &w, sizeof(v));
  return v;
}

/// True for any float32 NaN payload.
bool isNaNF32(float v) {
  const uint32_t w = f32BitsOf(v);
  return (w & UINT32_C(0x7F800000)) == UINT32_C(0x7F800000) &&
         (w & UINT32_C(0x007FFFFF)) != 0U;
}

/// True for either float32 infinity.
bool isInfF32(float v) {
  const uint32_t w = f32BitsOf(v);
  return (w & UINT32_C(0x7FFFFFFF)) == UINT32_C(0x7F800000);
}

/// Independent decode reference: exact float32 pattern for an E8M0 byte.
uint32_t referenceDecodeBitsE8M0(uint8_t bits) {
  if (bits == UINT8_C(0x00)) {
    return UINT32_C(0x00400000); // The f32 subnormal 2^-127.
  }
  if (bits == UINT8_C(0xFF)) {
    return UINT32_C(0x7F800001); // Quiet NaN.
  }
  return static_cast<uint32_t>(bits) << 23;
}

/**
 * @brief Independent RNE encode reference for E8M0, derived through exact dyadic arithmetic
 *
 * @details
 * Independent RNE encode reference, derived through exact dyadic arithmetic
 * in double instead of the kernel's g/r/s bit fields, replicating the
 * semantics of the PyTorch reference implementation
 * (torch/headeronly/util/Float8_e8m0fnu.h):
 *
 * @li NaN/infinity map to 0xFF; float32 zero keeps its zero exponent field
 *     (scale 0x00).
 * @li Normal magnitudes k = e2 - 1 from frexp: below the halfway fraction
 *     (m < 1.5) the input's own exponent stands, the exact tie and above
 *     round one step up because the implied mantissa bit (the format's
 *     least significant bit) is 1 for normals.
 * @li Subnormal magnitudes follow the reference's mantissa cut: any
 *     magnitude strictly above the smallest scale 2^-127 bumps to the next
 *     scale 0x01 (the reference cuts the raw mantissa at bit 22, so every
 *     round/sticky bit or the round bit itself under the cut forces the
 *     bump, the geometric tie included), while magnitudes at or below
 *     2^-127 stay at scale 0x00 because denormals carry lsb 0.
 * @li No upper clamp: normal inputs at or above 2^127 * 1.5 legitimately
 *     overflow the format and land on 0xFF, the OCP answer for it.
 */
uint8_t referenceEncodeRneE8M0(float v) {
  if (isNaNF32(v) || isInfF32(v)) {
    return UINT8_C(0xFF);
  }
  const double w = static_cast<double>(std::fabs(v));
  if (w == 0.0) {
    return UINT8_C(0x00); // Zero keeps exponent field 0 -> scale 0x00.
  }
  if (w < std::ldexp(1.0, -126)) {
    // Subnormal band: strictly above 2^-127 takes the next scale 0x01.
    return w > std::ldexp(1.0, -127) ? UINT8_C(0x01) : UINT8_C(0x00);
  }
  int e2 = 0;
  const double frac = std::frexp(w, &e2); // w = frac * 2^e2, frac in [0.5, 1).
  const double m = 2.0 * frac;            // In [1, 2).
  long long k = e2 - 1;
  if (m >= 1.5) {
    ++k; // Normal inputs carry the implied 1 as their lsb.
  }
  return static_cast<uint8_t>(k + 127);
}

/**
 * @brief Independent round-up reference (cuBLASLt UE8M0 derivation): keep the
 *        exponent when the magnitude is an exact power of two, add one step
 *        otherwise, clamp to the largest finite pattern 0xFE.
 */
uint8_t referenceEncodeRoundupE8M0(float v) {
  if (isNaNF32(v) || isInfF32(v)) {
    return UINT8_C(0xFF);
  }
  const double w = static_cast<double>(std::fabs(v));
  if (w == 0.0) {
    return UINT8_C(0x00);
  }
  int e2 = 0;
  const double frac = std::frexp(w, &e2);
  const double m = 2.0 * frac;
  long long k = e2 - 1 + (m != 1.0 ? 1 : 0);
  k = std::max<long long>(k, -127);
  k = std::min<long long>(k, 127);
  return static_cast<uint8_t>(k + 127);
}

/// Deterministic xorshift32 stream (seed fixed, no globals).
uint32_t xorshift32(uint32_t state) {
  state ^= state << 13U;
  state ^= state >> 17U;
  state ^= state << 5U;
  return state;
}

/// Mantissa patterns covering every decision boundary of the encode.
constexpr std::array<uint32_t, 11> kMantissaCorners{{
    0x000000U, // Exact power of two.
    0x000001U, // One ulp above a power of two.
    0x1FFFFFU, // All low bits set, guard bit clear.
    0x200000U, // Round bit set only.
    0x3FFFFFU, // Guard and round set.
    0x400000U, // Exact tie (guard set, round and sticky clear).
    0x400001U, // One ulp past the exact tie.
    0x600000U, // Upper halfway fraction (m = 1.5).
    0x600001U, // Just above the upper halfway fraction.
    0x7FFFFEU, // Two ulps below the next binade boundary.
    0x7FFFFFU, // Largest mantissa of the binade.
}};

} // namespace

/**
 * @brief Verifies numeric decode against the independent bit formula over
 *        the full 8-bit domain, through both the C API and the struct.
 */
TEST(BitPatternIdentityFloat8_e8m0fnu, AllPatternsDecodeToReferenceValue) {
  for (uint32_t b = 0; b < kE8M0Patterns; ++b) {
    const auto bits = static_cast<uint8_t>(b);
    const uint32_t want = referenceDecodeBitsE8M0(bits);
    EXPECT_EQ(fp8e8m0fnu_to_f32_bits(bits), want)
        << "pattern=0x" << std::hex << b;
    EXPECT_EQ(f32BitsOf(ncore::dtypes::detail::fp8e8m0fnu_to_fp32_value(bits)),
              want)
        << "pattern=0x" << std::hex << b;
    const Float8_e8m0fnu wrapped{bits, Float8_e8m0fnu::from_bits()};
    EXPECT_EQ(f32BitsOf(static_cast<float>(wrapped)), want)
        << "pattern=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies the struct round-trips every pattern through its float
 *        conversion: every decoded magnitude is an exact power of two (and
 *        0xFF widens to a NaN that re-encodes to 0xFF regardless of
 *        payload), so re-encoding is the identity for all 256 cells.
 */
TEST(BitPatternIdentityFloat8_e8m0fnu,
     StructDecodeReencodeIsIdentityForAllPatterns) {
  for (uint32_t b = 0; b < kE8M0Patterns; ++b) {
    const auto bits = static_cast<uint8_t>(b);
    const Float8_e8m0fnu value{bits, Float8_e8m0fnu::from_bits()};
    const float widened = static_cast<float>(value);
    const Float8_e8m0fnu back = widened;
    EXPECT_EQ(back.x, bits) << "pattern=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies the C API bit-level widening equals the numeric decode
 *        over the full domain.
 */
TEST(BitPatternIdentityFloat8_e8m0fnu, ToF32BitsMatchesNumericDecode) {
  for (uint32_t b = 0; b < kE8M0Patterns; ++b) {
    const auto bits = static_cast<uint8_t>(b);
    const uint32_t viaBitApi = fp8e8m0fnu_to_f32_bits(bits);
    const uint32_t viaNumeric = f32BitsOf(fp8e8m0fnu_to_float(bits));
    EXPECT_EQ(viaBitApi, viaNumeric) << "pattern=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies the encode inverts the decode for every pattern through
 *        both the numeric and the bit-level C API: each decoded value is an
 *        exact scale power (2^-127 through 2^126, or the NaN canon), so the
 *        RNE encode returns the original byte unchanged.
 */
TEST(BitPatternIdentityFloat8_e8m0fnu, FromF32BitsInvertsToF32Bits) {
  for (uint32_t b = 0; b < kE8M0Patterns; ++b) {
    const auto bits = static_cast<uint8_t>(b);
    EXPECT_EQ(fp8e8m0fnu_from_f32_bits(fp8e8m0fnu_to_f32_bits(bits)), bits)
        << "pattern=0x" << std::hex << b;
    EXPECT_EQ(fp8e8m0fnu_from_float(fp8e8m0fnu_to_float(bits)), bits)
        << "pattern=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies the RNE encode against the frexp-based reference over
 *        curated special values, exponent/mantissa corners, and a
 *        deterministic random sweep.
 */
TEST(BitPatternIdentityFloat8_e8m0fnu, FromFloatMatchesReferenceEncodeRne) {
  const std::array<uint32_t, 18> cornerBits{{
      UINT32_C(0x7FC00000), // qNaN.
      UINT32_C(0x7F800000), // +inf.
      UINT32_C(0xFF800000), // -inf (sign ignored).
      UINT32_C(0x00000000), // +0 -> lowest scale.
      UINT32_C(0x80000000), // -0 -> lowest scale, sign ignored.
      UINT32_C(0x00000001), // Min subnormal, stays at the bottom scale.
      UINT32_C(0x00200000), // 2^-128, below the smallest scale.
      UINT32_C(0x00400000), // Exactly 2^-127, the smallest scale itself.
      UINT32_C(0x00600000), // 1.5 * 2^-127: subnormal above 2^-127 bumps.
      UINT32_C(0x00800000), // Exactly 2^-126, first normal exponent field.
      UINT32_C(0x3F800000), // 1.0.
      UINT32_C(0x3FC00000), // 1.5: tie on a normal, rounds up.
      UINT32_C(0x40400000), // 3.0: halfway between 2 and 4, rounds up.
      UINT32_C(0x43000000), // 128.0.
      UINT32_C(0x43000001), // One ulp past 128.
      UINT32_C(0x7F000000), // 2^127.
      UINT32_C(0x7F7FFFFF), // Float32 max overflows the format to 0xFF.
      UINT32_C(0x7F000001), // Tiny ulp above 2^127 stays at 0xFE.
  }};
  for (const uint32_t bits : cornerBits) {
    const float v = f32OfBits(bits);
    EXPECT_EQ(fp8e8m0fnu_from_float(v), referenceEncodeRneE8M0(v))
        << "float32 bits=0x" << std::hex << bits;
  }

  // Every exponent field crossed with the decision-boundary mantissas.
  for (uint32_t exp = 0; exp < 0xFFU; ++exp) {
    for (const uint32_t mant : kMantissaCorners) {
      const uint32_t bits = (exp << 23) | mant;
      const float v = f32OfBits(bits);
      EXPECT_EQ(fp8e8m0fnu_from_float(v), referenceEncodeRneE8M0(v))
          << "float32 bits=0x" << std::hex << bits;
    }
  }

  // Deterministic random sweep over the full bit space.
  uint32_t state = UINT32_C(0x9E3779B9);
  for (int i = 0; i < 4096; ++i) {
    state = xorshift32(state);
    const float v = f32OfBits(state);
    EXPECT_EQ(fp8e8m0fnu_from_float(v), referenceEncodeRneE8M0(v))
        << "float32 bits=0x" << std::hex << state << " at " << i;
  }
}

/**
 * @brief Verifies the round-up derivation against its reference over the
 *        same sweep spaces. Exact powers of two keep their exponent, any
 *        other magnitude rounds one step up, and the clamp holds at 0xFE.
 */
TEST(BitPatternIdentityFloat8_e8m0fnu, RoundupMatchesCublasltDerivation) {
  // Absolute expectations for the documented behavior.
  struct Case {
    uint32_t bits;
    uint8_t want;
    const char *what;
  };
  const std::array<Case, 11> cases{{
      {.bits = UINT32_C(0x7FC00000), .want = UINT8_C(0xFF), .what = "qNaN"},
      {.bits = UINT32_C(0x7F800000), .want = UINT8_C(0xFF), .what = "+inf"},
      {.bits = UINT32_C(0x00000000), .want = UINT8_C(0x00), .what = "zero"},
      {.bits = UINT32_C(0x00400000),
       .want = UINT8_C(0x00),
       .what = "2^-127 exact"},
      {.bits = UINT32_C(0x00800000),
       .want = UINT8_C(0x01),
       .what = "2^-126 exact keeps exponent"},
      {.bits = UINT32_C(0x00080000),
       .want = UINT8_C(0x00),
       .what = "tiny subnormal clamps low"},
      {.bits = UINT32_C(0x3F800000),
       .want = UINT8_C(0x7F),
       .what = "1.0 exact keeps exponent"},
      {.bits = UINT32_C(0x3F8CCCCD),
       .want = UINT8_C(0x80),
       .what = "1.1 rounds up to 2"},
      {.bits = UINT32_C(0x405FFFFF),
       .want = UINT8_C(0x81),
       .what = "non-power rounds up a binade"},
      {.bits = UINT32_C(0x7F000000),
       .want = UINT8_C(0xFE),
       .what = "2^127 exact"},
      {.bits = UINT32_C(0x7F7FFFFF),
       .want = UINT8_C(0xFE),
       .what = "f32 max clamps, never NaN"},
  }};
  for (const Case &c : cases) {
    const uint8_t got = fp8e8m0fnu_roundup_from_float(f32OfBits(c.bits));
    EXPECT_EQ(got, c.want) << c.what << " (float32 bits=0x" << std::hex
                           << c.bits << ")";
  }

  // Reference agreement over exponents, corner mantissas, and random bits.
  for (uint32_t exp = 0; exp < 0xFFU; ++exp) {
    for (const uint32_t mant : kMantissaCorners) {
      const uint32_t bits = (exp << 23) | mant;
      const float v = f32OfBits(bits);
      EXPECT_EQ(fp8e8m0fnu_roundup_from_float(v), referenceEncodeRoundupE8M0(v))
          << "float32 bits=0x" << std::hex << bits;
    }
  }
  uint32_t state = UINT32_C(0x853CA45B);
  for (int i = 0; i < 4096; ++i) {
    state = xorshift32(state);
    const float v = f32OfBits(state);
    EXPECT_EQ(fp8e8m0fnu_roundup_from_float(v), referenceEncodeRoundupE8M0(v))
        << "float32 bits=0x" << std::hex << state << " at " << i;
  }
}

/**
 * @brief Verifies the E8M0 pair registry shape: only f32<->E8M0 resolve to
 *        a kernel, every other E8M0 row/column entry stays null, and no
 *        null pair is ever invoked.
 */
TEST(BitPatternIdentityFloat8_e8m0fnu, PairRegistryResolvesExactlyTwoKernels) {
  EXPECT_NE(get_dispatched_cast_func(DType_::Float32, DType_::Float8E8M0fnu),
            nullptr);
  EXPECT_NE(get_dispatched_cast_func(DType_::Float8E8M0fnu, DType_::Float32),
            nullptr);

  for (size_t i = 0; i < NUM_DTYPES; ++i) {
    const auto d = static_cast<DType_>(i);
    if (d == DType_::Float32) {
      continue;
    }
    EXPECT_EQ(get_dispatched_cast_func(d, DType_::Float8E8M0fnu), nullptr)
        << "dtype index=" << i;
    EXPECT_EQ(get_dispatched_cast_func(DType_::Float8E8M0fnu, d), nullptr)
        << "dtype index=" << i;
  }
  EXPECT_EQ(
      get_dispatched_cast_func(DType_::Float8E8M0fnu, DType_::Float8E8M0fnu),
      nullptr);
}

/**
 * @brief Verifies the classification truth on a real tensor: E8M0 counts
 *        as a floating format (exponent-only encoding) and reports false
 *        for every other classification slot.
 */
TEST(BitPatternIdentityFloat8_e8m0fnu, TensorClassificationMatchesFormat) {
  novaStatus_t st{};
  ncore::wrappers::TensorCXX ten(std::vector<size_t>{4}, DType_::Float8E8M0fnu,
                                 DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor view = ten.getCTensor();

  EXPECT_TRUE(is_floating(&view));
  EXPECT_FALSE(is_integer(&view));
  EXPECT_FALSE(is_signed_integer(&view));
  EXPECT_FALSE(is_unsigned_integer(&view));
  EXPECT_FALSE(is_quantized_signed_integer(&view));
  EXPECT_FALSE(is_quantized_unsigned_integer(&view));
}

/**
 * @brief Verifies the end-to-end path: public cast() and the generated
 *        cast.h wrapper both write exactly what the scalar kernel writes,
 *        and casting back reconstructs the f32 decode bit-exactly.
 */
TEST(BitPatternIdentityFloat8_e8m0fnu, CastEndToEndMatchesScalarKernel) {
  const std::array<uint32_t, 10> inputBits{{
      UINT32_C(0x00000000), // 0.0 -> 0x00.
      UINT32_C(0x80000000), // -0.0, sign ignored -> 0x00.
      UINT32_C(0x3F800000), // 1.0 -> 0x7F (scale 2^0).
      UINT32_C(0x3F8CCCCD), // 1.1 -> 0x7F (below the halfway fraction).
      UINT32_C(0x00400000), // 2^-127 -> 0x00.
      UINT32_C(0x40400000), // 3.0 -> 0x81 (halfway tie rounds up to 4.0).
      UINT32_C(0x43000000), // 128.0 -> 0x86 (exact scale 2^7).
      UINT32_C(0x7F000000), // 2^127 -> 0xFE.
      UINT32_C(0x7F7FFFFF), // f32 max -> 0xFF (overflow leg).
      UINT32_C(0x7FC00000), // qNaN -> 0xFF.
  }};
  constexpr size_t kCount = 10;

  auto dispatchPair = tests::casting::makePair(
      DType_::Float32, DType_::Float8E8M0fnu, kCount, 1U);
  auto wrapperPair = tests::casting::makePair(
      DType_::Float32, DType_::Float8E8M0fnu, kCount, 1U);
  ASSERT_TRUE(dispatchPair.ok);
  ASSERT_TRUE(wrapperPair.ok);
  for (size_t i = 0; i < kCount; ++i) {
    tests::casting::writeRaw(dispatchPair.src.mutableCTensor(), i,
                             inputBits[i]);
    tests::casting::writeRaw(wrapperPair.src.mutableCTensor(), i, inputBits[i]);
  }

  const novaStatus_t dispatchSt =
      cast(&dispatchPair.src.mutableCTensor(),
           &dispatchPair.dst.mutableCTensor(), DType_::Float8E8M0fnu);
  ASSERT_EQ(dispatchSt.err, novaSuccess);
  novaStatus_t wrapperSt;
  tf32_to_fp8e8m0(&wrapperPair.src.mutableCTensor(),
                  &wrapperPair.dst.mutableCTensor(), &wrapperSt);
  ASSERT_EQ(wrapperSt.err, novaSuccess);

  for (size_t i = 0; i < kCount; ++i) {
    const float v = f32OfBits(inputBits[i]);
    const uint8_t scalar = fp8e8m0fnu_from_float(v);
    const uint8_t viaCast = static_cast<uint8_t>(
        tests::casting::readRaw(dispatchPair.dst.getCTensor(), i));
    const uint8_t viaWrapper = static_cast<uint8_t>(
        tests::casting::readRaw(wrapperPair.dst.getCTensor(), i));
    EXPECT_EQ(viaCast, scalar)
        << "idx=" << i << " bits=0x" << std::hex << inputBits[i];
    EXPECT_EQ(viaWrapper, scalar)
        << "idx=" << i << " bits=0x" << std::hex << inputBits[i];
  }

  // Cast back to f32: the decode must be bit-exactly the scalar kernel's.
  const novaStatus_t backSt =
      cast(&dispatchPair.dst.mutableCTensor(),
           &dispatchPair.src.mutableCTensor(), DType_::Float32);
  ASSERT_EQ(backSt.err, novaSuccess);
  for (size_t i = 0; i < kCount; ++i) {
    const uint32_t reconstituted = static_cast<uint32_t>(
        tests::casting::readRaw(dispatchPair.src.getCTensor(), i));
    const uint32_t want =
        referenceDecodeBitsE8M0(fp8e8m0fnu_from_float(f32OfBits(inputBits[i])));
    EXPECT_EQ(reconstituted, want)
        << "idx=" << i << " bits=0x" << std::hex << inputBits[i];
  }
}
