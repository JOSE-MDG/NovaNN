/**
 * @file RangeAndDensity_test.cpp
 * @brief Numeric-range and spacing (density) tests for the reduced formats.
 *
 * Per format (FP16, BF16, FP8 E4M3FN, FP8 E5M2, FP4 E2M1FN):
 * @li max finite / min normal / min subnormal match the bias model exactly,
 * @li consecutive patterns inside a binade differ by exactly one ULP,
 * @li subnormal ranges form arithmetic sequences,
 * @li pattern walks are strictly monotonic,
 * @li std::numeric_limits specializations agree with the reference model,
 * @li encode saturates per policy just beyond the representable range.
 *
 * All expectations are exact dyadic doubles; no tolerances are used.
 */

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <string>

#include <ncore/core/dtype.h>
#include <ncore/core/fp_utils.h>
#include <ncore/headeronly/dtypes/bfloat16.hh>
#include <ncore/headeronly/dtypes/fp4_e2m1fn_x2.hh>
#include <ncore/headeronly/dtypes/fp8_e4m3fn.hh>
#include <ncore/headeronly/dtypes/fp8_e5m2.hh>
#include <ncore/headeronly/dtypes/half.hh>

#include "../dtypeCasting/utils/Oracle.hpp"
#include "utils/FloatingPointClassification.hpp"

namespace {

using tests::casting::FormatSpec;
using tests::casting::kBf16;
using tests::casting::kFp16;
using tests::casting::kFp4E2M1;
using tests::casting::kFp8E4M3;
using tests::casting::kFp8E5M2;
using tests::casting::maxFiniteMagnitude;
using tests::casting::minNormalMagnitude;
using tests::fpc::f32BitsOf;
using tests::fpc::isNaNF32Bits;

/// One reduced format wired to its conversion entry points.
struct FormatUnderTest {
  const char *label;
  FormatSpec spec;
  std::function<uint64_t(double)> encodeBits; ///< float -> raw pattern.
  std::function<double(uint64_t)> decodeBits; ///< raw pattern -> double.
};

/// Builds the per-format parameter set.
std::array<FormatUnderTest, 5> buildFormats() {
  return {{
      {.label = "fp16",
       .spec = kFp16,
       .encodeBits =
           [](double v) {
             float16 h = fp16_from_float(static_cast<float>(v));
             uint16_t bits = 0;
             std::memcpy(&bits, &h, sizeof(bits));
             return static_cast<uint64_t>(bits);
           },
       .decodeBits =
           [](uint64_t p) {
             float16 h;
             const auto bits = static_cast<uint16_t>(p);
             std::memcpy(&h, &bits, sizeof(h));
             return static_cast<double>(fp16_to_float(h));
           }},
      {.label = "bf16",
       .spec = kBf16,
       .encodeBits =
           [](double v) {
#if defined(__clang__)
             // Clang __bf16 ABI limitation: see
             // utils/BFloat16ClangLimitations.md.
             return static_cast<uint64_t>(
                 ncore::dtypes::detail::round_to_nearest_even(
                     static_cast<float>(v)));
#else
             bfloat16 b = bf16_from_float(static_cast<float>(v));
             uint16_t bits = 0;
             std::memcpy(&bits, &b, sizeof(bits));
             return static_cast<uint64_t>(bits);
#endif
           },
       .decodeBits =
           [](uint64_t p) {
#if defined(__clang__)
             return static_cast<double>(ncore::dtypes::detail::f32_from_bits(
                 static_cast<uint16_t>(p)));
#else
             bfloat16 b;
             const auto bits = static_cast<uint16_t>(p);
             std::memcpy(&b, &bits, sizeof(b));
             return static_cast<double>(bf16_to_float(b));
#endif
           }},
      {.label = "fp8e4m3",
       .spec = kFp8E4M3,
       .encodeBits =
           [](double v) { return fp8e4m3fn_from_float(static_cast<float>(v)); },
       .decodeBits =
           [](uint64_t p) {
             return static_cast<double>(
                 fp8e4m3fn_to_float(static_cast<float8_e4m3fn>(p)));
           }},
      {.label = "fp8e5m2",
       .spec = kFp8E5M2,
       .encodeBits =
           [](double v) { return fp8e5m2_from_float(static_cast<float>(v)); },
       .decodeBits =
           [](uint64_t p) {
             return static_cast<double>(
                 fp8e5m2_to_float(static_cast<float8_e5m2>(p)));
           }},
      {.label = "fp4e2m1",
       .spec = kFp4E2M1,
       .encodeBits =
           [](double v) {
             const ncore::dtypes::Float4_e2m1fn lane(static_cast<float>(v));
             return static_cast<uint64_t>(lane.x);
           },
       .decodeBits =
           [](uint64_t p) {
             const ncore::dtypes::Float4_e2m1fn lane{
                 static_cast<uint8_t>(p),
                 ncore::dtypes::Float4_e2m1fn::from_bits()};
             return static_cast<double>(static_cast<float>(lane));
           }},
  }};
}

/// Documented max-finite bound per format (exact dyadic doubles).
double documentedMaxFinite(const FormatSpec &spec) {
  switch (spec.bias) {
  case 15: // FP16 and E5M2 share the bias; distinguish by mantissa width.
    return spec.manBits == 10 ? 65504.0 : 57344.0;
  case 127:
    // BF16: (2 - 2^-7) * 2^127.
    return std::ldexp(2.0 - std::ldexp(1.0, -7), 127);
  case 7:
    return 448.0;
  case 1:
    return 6.0;
  default:
    ADD_FAILURE() << "unknown format";
    return 0.0;
  }
}

} // namespace

/**
 * @brief Verifies decode(max finite) equals the documented numeric bound.
 */
TEST(RangeAndDensity, MaxFiniteMatchesDocumentedBound) {
  auto formats = buildFormats();
  for (const FormatUnderTest &fmt : formats) {
    SCOPED_TRACE(fmt.label);
    const double got =
        fmt.decodeBits(maxFiniteMagnitude(fmt.spec) |
                       ((1ULL << (tests::casting::totalBits(fmt.spec) - 1)) *
                        0)); // Positive magnitude.
    EXPECT_DOUBLE_EQ(got, documentedMaxFinite(fmt.spec)) << fmt.label;

    const double neg =
        fmt.decodeBits(maxFiniteMagnitude(fmt.spec) |
                       (1ULL << (tests::casting::totalBits(fmt.spec) - 1)));
    EXPECT_DOUBLE_EQ(neg, -documentedMaxFinite(fmt.spec)) << fmt.label;
  }
}

/**
 * @brief Verifies min normal / min subnormal against the bias model:
 *        2^(1-bias) and 2^(1-bias-manBits).
 */
TEST(RangeAndDensity, MinNormalAndMinSubnormalMatchBiasModel) {
  auto formats = buildFormats();
  for (const FormatUnderTest &fmt : formats) {
    SCOPED_TRACE(fmt.label);
    const FormatSpec &spec = fmt.spec;

    const double minNormal = fmt.decodeBits(minNormalMagnitude(spec));
    EXPECT_DOUBLE_EQ(minNormal, std::ldexp(1.0, 1 - spec.bias)) << fmt.label;

    const double minSub =
        fmt.decodeBits(tests::casting::minSubnormalMagnitude(spec));
    EXPECT_DOUBLE_EQ(minSub, std::ldexp(1.0, 1 - spec.bias - spec.manBits))
        << fmt.label;
  }
}

/**
 * @brief Verifies that consecutive patterns inside a normal binade differ
 *        by exactly one ULP, near the smallest normal and one binade up.
 */
TEST(RangeAndDensity, NormalBinadeSpacingEqualsUlp) {
  auto formats = buildFormats();
  for (const FormatUnderTest &fmt : formats) {
    SCOPED_TRACE(fmt.label);
    const FormatSpec &spec = fmt.spec;
    const uint64_t maxMag = maxFiniteMagnitude(spec);

    for (const int expField : {1, 2}) {
      const uint64_t binadeBase = static_cast<uint64_t>(expField)
                                  << spec.manBits;
      if (binadeBase > maxMag) {
        continue; // FP4's domain ends inside the first normal binade.
      }
      const double ulp = std::ldexp(1.0, expField - spec.bias - spec.manBits);
      // Stay strictly inside this binade: the ULP doubles in the next one.
      const uint64_t binadeEnd =
          ((static_cast<uint64_t>(expField) + 1U) << spec.manBits) - 1U;
      const uint64_t lastPattern =
          std::min(binadeBase + 14U, std::min(binadeEnd - 1U, maxMag));
      for (uint64_t p = binadeBase; p <= lastPattern; ++p) {
        const double v0 = fmt.decodeBits(p);
        const double v1 = fmt.decodeBits(p + 1);
        EXPECT_DOUBLE_EQ(v1 - v0, ulp)
            << fmt.label << " pattern=0x" << std::hex << p;
      }
    }
  }
}

/**
 * @brief Verifies the full subnormal range forms an arithmetic sequence
 *        with step equal to the minimum subnormal.
 */
TEST(RangeAndDensity, SubnormalSpacingIsConstant) {
  auto formats = buildFormats();
  for (const FormatUnderTest &fmt : formats) {
    SCOPED_TRACE(fmt.label);
    const FormatSpec &spec = fmt.spec;
    const double step = std::ldexp(1.0, 1 - spec.bias - spec.manBits);
    const uint64_t count = (1ULL << spec.manBits);

    double previous = 0.0;
    for (uint64_t man = 1; man < count; ++man) {
      const double value = fmt.decodeBits(man);
      EXPECT_DOUBLE_EQ(value, static_cast<double>(man) * step)
          << fmt.label << " man=" << man;
      EXPECT_GT(value, previous) << fmt.label << " man=" << man;
      previous = value;
    }
  }
}

/**
 * @brief Verifies strict monotonicity of decode across the whole positive
 *        finite magnitude domain.
 */
TEST(RangeAndDensity, PatternWalkIsStrictlyMonotonic) {
  auto formats = buildFormats();
  for (const FormatUnderTest &fmt : formats) {
    SCOPED_TRACE(fmt.label);
    const uint64_t maxMag = maxFiniteMagnitude(fmt.spec);

    double previous = -1.0;
    for (uint64_t mag = 0; mag <= maxMag; ++mag) {
      // mag <= maxFiniteMagnitude excludes the reserved exponent fields, so
      // decode cannot produce NaN here.
      const double value = fmt.decodeBits(mag);
      EXPECT_GT(value, previous)
          << fmt.label << " pattern=0x" << std::hex << mag;
      previous = value;
    }
  }
}

/**
 * @brief Verifies std::numeric_limits<Half> against the reference model.
 */
TEST(RangeAndDensity, NumericLimitsHalfAgreeWithReferenceModel) {
  using L = std::numeric_limits<ncore::dtypes::Half>;
  auto decode = [](ncore::dtypes::Half h) {
    return static_cast<double>(static_cast<float>(h));
  };

  EXPECT_TRUE(L::has_infinity);
  EXPECT_EQ(L::digits, 11);
  EXPECT_DOUBLE_EQ(decode(L::max()), 65504.0);
  EXPECT_DOUBLE_EQ(decode(L::lowest()), -65504.0);
  EXPECT_DOUBLE_EQ(decode(L::min()), std::ldexp(1.0, -14));
  EXPECT_DOUBLE_EQ(decode(L::denorm_min()), std::ldexp(1.0, -24));
  EXPECT_DOUBLE_EQ(decode(L::epsilon()), std::ldexp(1.0, -10));

  const uint32_t infBits = f32BitsOf(static_cast<float>(decode(L::infinity())));
  EXPECT_TRUE(tests::fpc::isInfF32Bits(infBits));
  const uint32_t nanBits =
      f32BitsOf(static_cast<float>(decode(L::quiet_NaN())));
  EXPECT_TRUE(isNaNF32Bits(nanBits));
}

/**
 * @brief Verifies std::numeric_limits<BFloat16> against the reference model.
 */
TEST(RangeAndDensity, NumericLimitsBFloat16AgreeWithReferenceModel) {
  using L = std::numeric_limits<ncore::dtypes::BFloat16>;
  auto decode = [](ncore::dtypes::BFloat16 b) {
    return static_cast<double>(static_cast<float>(b));
  };

  EXPECT_TRUE(L::has_infinity);
  EXPECT_EQ(L::digits, 8);
  EXPECT_DOUBLE_EQ(decode(L::max()),
                   std::ldexp(2.0 - std::ldexp(1.0, -7), 127));
  EXPECT_DOUBLE_EQ(decode(L::lowest()),
                   -std::ldexp(2.0 - std::ldexp(1.0, -7), 127));
  EXPECT_DOUBLE_EQ(decode(L::min()), std::ldexp(1.0, -126));
  EXPECT_DOUBLE_EQ(decode(L::denorm_min()), std::ldexp(1.0, -133));
  EXPECT_DOUBLE_EQ(decode(L::epsilon()), std::ldexp(1.0, -7));

  const uint32_t infBits = f32BitsOf(static_cast<float>(decode(L::infinity())));
  EXPECT_TRUE(tests::fpc::isInfF32Bits(infBits));
  const uint32_t nanBits =
      f32BitsOf(static_cast<float>(decode(L::quiet_NaN())));
  EXPECT_TRUE(isNaNF32Bits(nanBits));
}

/**
 * @brief Verifies std::numeric_limits<Float8_e4m3fn> against the reference
 *        model (finite-only: has_infinity is false).
 */
TEST(RangeAndDensity, NumericLimitsFp8E4M3AgreeWithReferenceModel) {
  using L = std::numeric_limits<ncore::dtypes::Float8_e4m3fn>;
  auto decode = [](ncore::dtypes::Float8_e4m3fn v) {
    return static_cast<double>(static_cast<float>(v));
  };

  EXPECT_FALSE(L::has_infinity);
  EXPECT_EQ(L::digits, 4);
  EXPECT_DOUBLE_EQ(decode(L::max()), 448.0);
  EXPECT_DOUBLE_EQ(decode(L::lowest()), -448.0);
  EXPECT_DOUBLE_EQ(decode(L::min()), std::ldexp(1.0, -6));
  EXPECT_DOUBLE_EQ(decode(L::denorm_min()), std::ldexp(1.0, -9));
  EXPECT_DOUBLE_EQ(decode(L::epsilon()), std::ldexp(1.0, -3));

  const uint32_t nanBits =
      f32BitsOf(static_cast<float>(decode(L::quiet_NaN())));
  EXPECT_TRUE(isNaNF32Bits(nanBits));
}

/**
 * @brief Verifies std::numeric_limits<Float8_e5m2> against the reference
 *        model.
 */
TEST(RangeAndDensity, NumericLimitsFp8E5M2AgreeWithReferenceModel) {
  using L = std::numeric_limits<ncore::dtypes::Float8_e5m2>;
  auto decode = [](ncore::dtypes::Float8_e5m2 v) {
    return static_cast<double>(static_cast<float>(v));
  };

  EXPECT_TRUE(L::has_infinity);
  EXPECT_EQ(L::digits, 3);
  EXPECT_DOUBLE_EQ(decode(L::max()), 57344.0);
  EXPECT_DOUBLE_EQ(decode(L::lowest()), -57344.0);
  EXPECT_DOUBLE_EQ(decode(L::min()), std::ldexp(1.0, -14));
  EXPECT_DOUBLE_EQ(decode(L::denorm_min()), std::ldexp(1.0, -16));
  EXPECT_DOUBLE_EQ(decode(L::epsilon()), std::ldexp(1.0, -2));

  const uint32_t infBits = f32BitsOf(static_cast<float>(decode(L::infinity())));
  EXPECT_TRUE(tests::fpc::isInfF32Bits(infBits));
  const uint32_t nanBits =
      f32BitsOf(static_cast<float>(decode(L::quiet_NaN())));
  EXPECT_TRUE(isNaNF32Bits(nanBits));
}

/**
 * @brief Verifies std::numeric_limits<Float4_e2m1fn> against the reference
 *        model (finite-only; epsilon follows the ULP(1) convention).
 */
TEST(RangeAndDensity, NumericLimitsFp4AgreeWithReferenceModel) {
  using L = std::numeric_limits<ncore::dtypes::Float4_e2m1fn>;
  auto decode = [](ncore::dtypes::Float4_e2m1fn v) {
    return static_cast<double>(static_cast<float>(v));
  };

  EXPECT_FALSE(L::has_infinity);
  EXPECT_EQ(L::digits, 2);
  EXPECT_DOUBLE_EQ(decode(L::max()), 6.0);
  EXPECT_DOUBLE_EQ(decode(L::lowest()), -6.0);
  EXPECT_DOUBLE_EQ(decode(L::min()), 1.0);
  EXPECT_DOUBLE_EQ(decode(L::denorm_min()), 0.5);
  // Conventional machine epsilon: the gap between 1.0 and the next
  // representable value (1.5), i.e. 0.5.
  EXPECT_DOUBLE_EQ(decode(L::epsilon()), 0.5);
}

/**
 * @brief Pins each format's saturation policy at one boundary pair just
 *        beyond the representable range.
 */
TEST(RangeAndDensity, EncodeJustBeyondMaxSaturatesPerPolicy) {
  auto formats = buildFormats();

  EXPECT_EQ(formats[0].encodeBits(65520.0), UINT64_C(0x7C00)); // inf
  EXPECT_EQ(formats[1].encodeBits(tests::fpc::f32FromBits(0x7F800000U)),
            UINT64_C(0x7F80));                               // inf
  EXPECT_EQ(formats[2].encodeBits(1000.0), UINT64_C(0x7E));  // clamp 448
  EXPECT_EQ(formats[3].encodeBits(65535.0), UINT64_C(0x7C)); // inf
  EXPECT_EQ(formats[4].encodeBits(100.0), UINT64_C(0x7));    // clamp 6.0
}
