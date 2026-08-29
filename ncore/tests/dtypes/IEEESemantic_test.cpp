/**
 * @file IEEESemantic_test.cpp
 * @brief Semantic property tests for the reduced-precision conversions.
 *
 * Beyond per-pattern identity (covered by the bitPatternIdentity suites),
 * this file pins the rounding *semantics* that make training numerics
 * predictable:
 * @li round-to-nearest-even tie behavior at exactly representable midpoints,
 * @li monotonicity of the quantization maps,
 * @li idempotence (quantization is a projection),
 * @li sign symmetry,
 * @li the three distinct overflow policies (inf / clamp-at-max /
 *     saturate-at-max).
 *
 * Expectations are hardcoded literals derived from the format definitions,
 * never from another kernel.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include <ncore/core/dtype.h>
#include <ncore/core/fp_utils.h>
#include <ncore/headeronly/dtypes/bfloat16.hh>
#include <ncore/headeronly/dtypes/fp4_e2m1fn_x2.hh>

#include "../dtypeCasting/utils/Oracle.hpp"
#include "utils/FloatingPointClassification.hpp"

namespace {

using tests::casting::kDefaultSeed;
using tests::fpc::f32BitsOf;
using tests::fpc::f32FromBits;
using tests::fpc::isInfF32Bits;

/// One reduced format wired to its conversion entry points.
struct FormatUnderTest {
  const char *label;
  tests::casting::FormatSpec spec;
  std::function<uint64_t(double)> encodeBits; ///< float -> raw pattern.
  std::function<double(uint64_t)> decodeBits; ///< raw pattern -> double.
};

/// Builds the per-format parameter set.
std::array<FormatUnderTest, 5> buildFormats() {
  return {{
      {.label = "fp16",
       .spec = tests::casting::kFp16,
       .encodeBits =
           [](double v) {
             const float16 h = fp16_from_float(static_cast<float>(v));
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
       .spec = tests::casting::kBf16,
       .encodeBits =
           [](double v) {
#if defined(__clang__)
             // Clang __bf16 ABI limitation: see
             // utils/BFloat16ClangLimitations.md.
             return static_cast<uint64_t>(
                 ncore::dtypes::detail::round_to_nearest_even(
                     static_cast<float>(v)));
#else
             const bfloat16 b = bf16_from_float(static_cast<float>(v));
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
       .spec = tests::casting::kFp8E4M3,
       .encodeBits =
           [](double v) { return fp8e4m3fn_from_float(static_cast<float>(v)); },
       .decodeBits =
           [](uint64_t p) {
             return static_cast<double>(
                 fp8e4m3fn_to_float(static_cast<float8_e4m3fn>(p)));
           }},
      {.label = "fp8e5m2",
       .spec = tests::casting::kFp8E5M2,
       .encodeBits =
           [](double v) { return fp8e5m2_from_float(static_cast<float>(v)); },
       .decodeBits =
           [](uint64_t p) {
             return static_cast<double>(
                 fp8e5m2_to_float(static_cast<float8_e5m2>(p)));
           }},
      {.label = "fp4e2m1",
       .spec = tests::casting::kFp4E2M1,
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

/// One RNE tie point: value plus expected even/lower/upper patterns.
struct TiePoint {
  double value;
  uint64_t expectedEven;
  uint64_t lowerNeighbor;
  uint64_t upperNeighbor;
};

/// Tie tables per format index (matches buildFormats() order).
constexpr std::array<std::array<TiePoint, 2>, 5> kTieTables{{
    /* fp16 */ {{
        {.value = 1.00048828125,
         .expectedEven = 0x3C00,
         .lowerNeighbor = 0x3C00,
         .upperNeighbor = 0x3C01}, // midpoint(1.0, 1+2^-10)
        {.value = 1.00146484375,
         .expectedEven = 0x3C02,
         .lowerNeighbor = 0x3C01,
         .upperNeighbor = 0x3C02}, // midpoint(1+2^-10, +2ulp)
    }},
    /* bf16 */
    {{
        {.value = 1.00390625,
         .expectedEven = 0x3F80,
         .lowerNeighbor = 0x3F80,
         .upperNeighbor = 0x3F81}, // midpoint(1.0, 1+2^-7)
        {.value = 1.01171875,
         .expectedEven = 0x3F82,
         .lowerNeighbor = 0x3F81,
         .upperNeighbor = 0x3F82},
    }},
    /* fp8e4m3 */
    {{
        {.value = 1.0625,
         .expectedEven = 0x38,
         .lowerNeighbor = 0x38,
         .upperNeighbor = 0x39}, // midpoint(1.0, 1.125)
        {.value = 17.0,
         .expectedEven = 0x58,
         .lowerNeighbor = 0x58,
         .upperNeighbor = 0x59}, // midpoint(16.0, 18.0)
    }},
    /* fp8e5m2 */
    {{
        // 2-bit mantissa: ULP(1) = 0.25, so midpoints sit at x.125 steps.
        {.value = 1.125,
         .expectedEven = 0x3C,
         .lowerNeighbor = 0x3C,
         .upperNeighbor = 0x3D}, // midpoint(1.0, 1.25); even q=4
        {.value = 1.375,
         .expectedEven = 0x3E,
         .lowerNeighbor = 0x3D,
         .upperNeighbor = 0x3E}, // midpoint(1.25, 1.5); even q=6
    }},
    /* fp4e2m1 */
    {{
        {.value = 0.25,
         .expectedEven = 0x0,
         .lowerNeighbor = 0x0,
         .upperNeighbor = 0x1},
        {.value = 0.75,
         .expectedEven = 0x2,
         .lowerNeighbor = 0x1,
         .upperNeighbor = 0x2},
    }},
}};

/// Extra FP4 ties beyond the two-slot table above.
constexpr std::array<TiePoint, 5> kFp4ExtraTies{{
    {.value = 1.25,
     .expectedEven = 0x2,
     .lowerNeighbor = 0x2,
     .upperNeighbor = 0x3},
    {.value = 1.75,
     .expectedEven = 0x4,
     .lowerNeighbor = 0x3,
     .upperNeighbor = 0x4},
    {.value = 2.5,
     .expectedEven = 0x4,
     .lowerNeighbor = 0x4,
     .upperNeighbor = 0x5},
    {.value = 3.5,
     .expectedEven = 0x6,
     .lowerNeighbor = 0x5,
     .upperNeighbor = 0x6},
    {.value = 5.0,
     .expectedEven = 0x6,
     .lowerNeighbor = 0x6,
     .upperNeighbor = 0x7}, // Ties to even: 4.0 wins over 6.0.
}};

/// Sign-magnitude ordering rank of a raw pattern within one format.
int64_t patternRank(uint64_t pattern, const tests::casting::FormatSpec &spec) {
  const uint64_t total = static_cast<uint64_t>(tests::casting::totalBits(spec));
  const uint64_t magnitude = pattern & ((1ULL << (total - 1)) - 1ULL);
  const bool neg = (pattern >> (total - 1)) != 0ULL;
  return neg ? -static_cast<int64_t>(magnitude)
             : static_cast<int64_t>(magnitude);
}

/// Deterministic LCG state for sweeps.
struct Lcg {
  uint32_t state;
  explicit Lcg(uint32_t seed) : state(seed) {}
  uint32_t next() {
    state = (state * 1664525U) + 1013904223U;
    return state;
  }
};

} // namespace

/**
 * @brief Verifies that exact midpoints round to the even-mantissa neighbor
 *        and that the just-below/just-above inputs round to the correct
 *        adjacent values.
 */
TEST(IEEESemantic, MidpointsRoundToEvenMantissa) {
  auto formats = buildFormats();
  for (size_t f = 0; f < formats.size(); ++f) {
    const FormatUnderTest &fmt = formats[f];
    SCOPED_TRACE(fmt.label);

    for (const TiePoint &tie : kTieTables[f]) {
      EXPECT_EQ(fmt.encodeBits(tie.value), tie.expectedEven)
          << fmt.label << " tie=" << tie.value;

      // Neighbors are taken in float space: the encoders consume floats,
      // and a double-space nextafter would round straight back to the tie.
      const float tieF = static_cast<float>(tie.value);
      const double below = static_cast<double>(std::nextafterf(tieF, 0.0F));
      const double above = static_cast<double>(
          std::nextafterf(tieF, std::numeric_limits<float>::infinity()));
      EXPECT_EQ(fmt.encodeBits(below), tie.lowerNeighbor)
          << fmt.label << " below=" << below;
      EXPECT_EQ(fmt.encodeBits(above), tie.upperNeighbor)
          << fmt.label << " above=" << above;
    }
  }

  // FP4 has seven documented midpoints; five more live in kFp4ExtraTies.
  const FormatUnderTest &fp4 = formats[4];
  for (const TiePoint &tie : kFp4ExtraTies) {
    EXPECT_EQ(fp4.encodeBits(tie.value), tie.expectedEven)
        << "fp4 tie=" << tie.value;
    const float tieF = static_cast<float>(tie.value);
    const double below = static_cast<double>(std::nextafterf(tieF, 0.0F));
    const double above = static_cast<double>(
        std::nextafterf(tieF, std::numeric_limits<float>::infinity()));
    EXPECT_EQ(fp4.encodeBits(below), tie.lowerNeighbor)
        << "fp4 below=" << below;
    EXPECT_EQ(fp4.encodeBits(above), tie.upperNeighbor)
        << "fp4 above=" << above;
  }
}

/**
 * @brief Verifies monotonicity of every quantization map: sorted inputs
 *        must produce non-decreasing pattern ranks.
 */
TEST(IEEESemantic, QuantizationIsMonotonic) {
  auto formats = buildFormats();
  for (const FormatUnderTest &fmt : formats) {
    SCOPED_TRACE(fmt.label);

    std::vector<double> values;
    values.reserve(2048);

    // Geometric ladder spanning several binades around 1.0.
    for (int e = -24; e <= 16; ++e) {
      const double base = std::ldexp(1.0, e);
      values.push_back(std::nextafter(base, 0.0));
      values.push_back(base);
      values.push_back(
          std::nextafter(base, std::numeric_limits<double>::infinity()));
      values.push_back(base * 1.25);
      values.push_back(base * 1.5);
    }

    // Seeded pseudo-random mantissas inside [1, 2) scaled across binades.
    Lcg rng(kDefaultSeed);
    for (int i = 0; i < 1024; ++i) {
      const double frac =
          1.0 + (static_cast<double>(rng.next() >> 8) / 16777216.0);
      const int e = -20 + static_cast<int>(rng.next() % 32U);
      values.push_back(frac * std::ldexp(1.0, e));
    }

    std::sort(values.begin(), values.end());

    int64_t previousRank = std::numeric_limits<int64_t>::min();
    for (const double v : values) {
      const int64_t rank = patternRank(fmt.encodeBits(v), fmt.spec);
      EXPECT_GE(rank, previousRank) << fmt.label << " non-monotonic at v=" << v;
      previousRank = rank;
    }
  }
}

/**
 * @brief Verifies idempotence: encode(decode(encode(x))) == encode(x).
 *        Sweeps the full finite domain of each format.
 *
 * NaN patterns are skipped: their payloads are canonicalized on encode, so
 * they are not fixed points of the quantization projection by design.
 */
TEST(IEEESemantic, QuantizationIsIdempotent) {
  auto formats = buildFormats();
  for (const FormatUnderTest &fmt : formats) {
    SCOPED_TRACE(fmt.label);
    const tests::casting::FormatSpec &spec = fmt.spec;
    const uint64_t maxFinite = tests::casting::maxFiniteMagnitude(spec);

    const auto isNaNPattern = [&spec](uint64_t p) {
      if (!spec.hasNan) {
        return false;
      }
      const uint64_t maxExpField = (1ULL << spec.expBits) - 1ULL;
      const uint64_t expField = (p >> spec.manBits) & maxExpField;
      const uint64_t man = p & ((1ULL << spec.manBits) - 1ULL);
      return expField == maxExpField && man != 0ULL;
    };

    for (uint64_t magnitude = 0; magnitude <= maxFinite; ++magnitude) {
      for (const uint64_t sign : {uint64_t{0}, uint64_t{1}}) {
        const uint64_t pattern =
            (sign << (tests::casting::totalBits(spec) - 1)) | magnitude;
        if (isNaNPattern(pattern)) {
          continue;
        }
        const double decoded = fmt.decodeBits(pattern);
        if (isInfF32Bits(f32BitsOf(static_cast<float>(decoded)))) {
          continue; // Overflow policy target; not part of the projection.
        }
        const uint64_t once = fmt.encodeBits(decoded);
        ASSERT_EQ(once, pattern)
            << fmt.label << " first encode drifted, pattern=0x" << std::hex
            << pattern;
        const uint64_t twice = fmt.encodeBits(fmt.decodeBits(once));
        EXPECT_EQ(twice, once) << fmt.label << " not idempotent, pattern=0x"
                               << std::hex << pattern;
      }
    }
  }
}

/**
 * @brief Verifies sign symmetry: encode(-x) == encode(x) | signbit for the
 *        sampled sweep values.
 */
TEST(IEEESemantic, SignSymmetryHolds) {
  auto formats = buildFormats();
  for (const FormatUnderTest &fmt : formats) {
    SCOPED_TRACE(fmt.label);
    const uint64_t signBit = 1ULL << (tests::casting::totalBits(fmt.spec) - 1);

    Lcg rng(kDefaultSeed ^ 0x5EEDU);
    for (int i = 0; i < 512; ++i) {
      const double frac =
          1.0 + (static_cast<double>(rng.next() >> 8) / 16777216.0);
      const int e = -12 + static_cast<int>(rng.next() % 20U);
      const double v = frac * std::ldexp(1.0, e);

      const uint64_t positive = fmt.encodeBits(v);
      const uint64_t negative = fmt.encodeBits(-v);
      EXPECT_EQ(negative, positive | signBit)
          << fmt.label << " asymmetry at v=" << v;
    }
    // Signed zero included explicitly.
    EXPECT_EQ(fmt.encodeBits(-0.0), fmt.encodeBits(0.0) | signBit);
  }
}

/**
 * @brief Pins each format's overflow policy at one boundary pair: the
 *        largest finite-representable input stays put and the next
 *        representable input crosses to inf / max-finite per policy.
 */
TEST(IEEESemantic, OverflowBoundaryRoundsPerPolicy) {
  auto formats = buildFormats();

  // FP16: 65519 rounds to max finite; 65520 (the overflow midpoint) -> inf.
  EXPECT_EQ(formats[0].encodeBits(65519.0), UINT64_C(0x7BFF));
  EXPECT_EQ(formats[0].encodeBits(65520.0), UINT64_C(0x7C00));

  // BF16: f32 bit patterns bracketing the overflow midpoint
  // (midpoint between max finite and the hypothetical next value).
  EXPECT_EQ(formats[1].encodeBits(f32FromBits(0x7F7F7FFFU)), UINT64_C(0x7F7F));
  EXPECT_EQ(formats[1].encodeBits(f32FromBits(0x7F7F8000U)), UINT64_C(0x7F80));

  // E4M3FN: saturates to max finite; no infinity exists.
  EXPECT_EQ(formats[2].encodeBits(460.0), UINT64_C(0x7E));
  EXPECT_EQ(formats[2].encodeBits(464.0), UINT64_C(0x7E));
  EXPECT_EQ(formats[2].encodeBits(480.0), UINT64_C(0x7E));

  // E5M2: overflows to infinity past the last normal binade midpoint.
  EXPECT_EQ(formats[3].encodeBits(60000.0), UINT64_C(0x7B));
  EXPECT_EQ(formats[3].encodeBits(65535.0), UINT64_C(0x7C));

  // FP4: saturates at 6.0; the 5.0 tie rounds DOWN to 4.0 (ties-to-even).
  EXPECT_EQ(formats[4].encodeBits(5.0), UINT64_C(0x6));
  EXPECT_EQ(formats[4].encodeBits(5.5), UINT64_C(0x7));
  EXPECT_EQ(formats[4].encodeBits(100.0), UINT64_C(0x7));
}
