/**
 * @file SpecialValuesClassification_test.cpp
 * @brief Exhaustive special-value classification tests for the reduced
 *        formats.
 *
 * For every raw pattern of FP16, BF16, FP8 E4M3FN, and FP8 E5M2 (and all 16
 * FP4 nibbles), verifies that:
 * @li the independent field-level classifier partitions the domain exactly,
 * @li the class survives decoding into float32 (with one documented,
 *     unavoidable exception: storage subnormals widen into float32 normals),
 * @li struct predicates (isnan/isinf) agree with the partition where the
 *     structs expose them,
 * @li negating a pattern flips only the sign bit and preserves the class.
 */

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>

#include <ncore/core/dtype.h>
#include <ncore/core/fp_utils.h>
#include <ncore/headeronly/dtypes/bfloat16.hh>
#include <ncore/headeronly/dtypes/fp4_e2m1fn_x2.hh>
#include <ncore/headeronly/dtypes/fp8_e4m3fn.hh>
#include <ncore/headeronly/dtypes/fp8_e5m2.hh>
#include <ncore/headeronly/dtypes/half.hh>

#include "utils/FloatingPointClassification.hpp"

namespace {

using tests::fpc::f32BitsOf;
using tests::fpc::FpClass;
using tests::fpc::isInfF32Bits;
using tests::fpc::isNaNF32Bits;
using tests::fpc::signBitF32Bits;

/// One format wired to its decode entry point.
struct FormatUnderTest {
  const char *label;
  tests::fpc::FormatModel model;
  std::function<float(uint32_t)> decodeBits; ///< raw pattern -> float.
};

/// Builds the parameter set for the four IEEE-style / hybrid formats.
std::array<FormatUnderTest, 4> buildFormats() {
  return {{
      {.label = "fp16",
       .model = tests::fpc::kFp16Model,
       .decodeBits =
           [](uint32_t p) {
             float16 h;
             const auto bits = static_cast<uint16_t>(p);
             std::memcpy(&h, &bits, sizeof(h));
             return fp16_to_float(h);
           }},
      {.label = "bf16",
       .model = tests::fpc::kBf16Model,
       .decodeBits =
           [](uint32_t p) {
#if defined(__clang__)
             // Clang __bf16 ABI limitation: see
             // utils/BFloat16ClangLimitations.md.
             return ncore::dtypes::detail::f32_from_bits(
                 static_cast<uint16_t>(p));
#else
             bfloat16 b;
             const auto bits = static_cast<uint16_t>(p);
             std::memcpy(&b, &bits, sizeof(b));
             return bf16_to_float(b);
#endif
           }},
      {.label = "fp8e4m3",
       .model = tests::fpc::kFp8E4M3Model,
       .decodeBits =
           [](uint32_t p) {
             return fp8e4m3fn_to_float(static_cast<float8_e4m3fn>(p));
           }},
      {.label = "fp8e5m2",
       .model = tests::fpc::kFp8E5M2Model,
       .decodeBits =
           [](uint32_t p) {
             return fp8e5m2_to_float(static_cast<float8_e5m2>(p));
           }},
  }};
}

/**
 * @brief Classifies a decoded float32 purely from its bits (no FP
 *        round-trip).
 *
 * Storage subnormals widen into float32 normals for every reduced format
 * (their smallest subnormal is far above the binary32 minimum), so kSubnormal
 * never appears in the decoded domain; callers handle that mapping.
 */
FpClass classifyDecodedF32(uint32_t bits) {
  if (isNaNF32Bits(bits)) {
    return FpClass::kNaN;
  }
  if (isInfF32Bits(bits)) {
    return FpClass::kInf;
  }
  if ((bits & 0x7FFFFFFFU) == 0U) {
    return FpClass::kZero;
  }
  return FpClass::kNormal;
}

} // namespace

/**
 * @brief Exhaustive partition sweep: the independent classifier's verdict
 *        must survive decoding, with subnormals documented as widening into
 *        float32 normals.
 */
TEST(SpecialValuesClassification, PartitionSweepMatchesDecodedClasses) {
  auto formats = buildFormats();
  for (const FormatUnderTest &fmt : formats) {
    SCOPED_TRACE(fmt.label);
    const uint32_t domainSize =
        1U << static_cast<uint32_t>(1 + fmt.model.expBits + fmt.model.manBits);

    uint64_t counts[5] = {0, 0, 0, 0, 0};
    for (uint32_t p = 0; p < domainSize; ++p) {
      const FpClass storageClass = tests::fpc::classify(p, fmt.model);
      ++counts[static_cast<size_t>(storageClass)];

      const FpClass decodedClass =
          classifyDecodedF32(f32BitsOf(fmt.decodeBits(p)));
      FpClass wantDecoded = storageClass;
      if (storageClass == FpClass::kSubnormal) {
        // Documented widening: reduced-format subnormals land inside the
        // float32 normal range by construction.
        wantDecoded = FpClass::kNormal;
      }
      ASSERT_EQ(decodedClass, wantDecoded)
          << fmt.label << " pattern=0x" << std::hex << p;

      // Sign must survive decoding for every non-NaN pattern (NaN payload
      // handling differs across paths; sign is still checked).
      EXPECT_EQ(signBitF32Bits(f32BitsOf(fmt.decodeBits(p))),
                tests::fpc::signBit(p, fmt.model))
          << fmt.label << " sign drift, pattern=0x" << std::hex << p;
    }

    // Sanity: every class count is nonzero except where the format forbids
    // it (E4M3FN has no inf).
    EXPECT_GT(counts[static_cast<size_t>(FpClass::kZero)], UINT64_C(0))
        << fmt.label;
    EXPECT_GT(counts[static_cast<size_t>(FpClass::kNormal)], UINT64_C(0))
        << fmt.label;
    if (!fmt.model.hasInf) {
      EXPECT_EQ(counts[static_cast<size_t>(FpClass::kInf)], UINT64_C(0))
          << fmt.label;
    } else {
      // Exactly two infinity patterns per sign-bearing format.
      EXPECT_EQ(counts[static_cast<size_t>(FpClass::kInf)], UINT64_C(2))
          << fmt.label;
    }
  }
}

/**
 * @brief Verifies struct predicates agree with the independent partition
 *        over the full E4M3FN and E5M2 domains.
 */
TEST(SpecialValuesClassification, StructPredicatesAgreeWithPartition) {
  {
    SCOPED_TRACE("fp8e4m3");
    for (uint32_t p = 0; p < 256U; ++p) {
      const auto bits = static_cast<uint8_t>(p);
      const ncore::dtypes::Float8_e4m3fn v{
          bits, ncore::dtypes::Float8_e4m3fn::from_bits()};
      const FpClass cls = tests::fpc::classify(p, tests::fpc::kFp8E4M3Model);
      EXPECT_EQ(v.isnan(), cls == FpClass::kNaN)
          << "pattern=0x" << std::hex << p;
      EXPECT_FALSE(v.isinf()) << "pattern=0x" << std::hex << p;
    }
  }
  {
    SCOPED_TRACE("fp8e5m2");
    for (uint32_t p = 0; p < 256U; ++p) {
      const auto bits = static_cast<uint8_t>(p);
      const ncore::dtypes::Float8_e5m2 v{
          bits, ncore::dtypes::Float8_e5m2::from_bits()};
      const FpClass cls = tests::fpc::classify(p, tests::fpc::kFp8E5M2Model);
      EXPECT_EQ(v.isnan(), cls == FpClass::kNaN)
          << "pattern=0x" << std::hex << p;
      EXPECT_EQ(v.isinf(), cls == FpClass::kInf)
          << "pattern=0x" << std::hex << p;
    }
  }
}

/**
 * @brief Verifies all 16 FP4 nibbles are finite: no NaN, no inf, decoded
 *        magnitudes strictly inside the finite range.
 */
TEST(SpecialValuesClassification, Fp4AllNibblesAreFinite) {
  using ncore::dtypes::Float4_e2m1fn;
  for (uint32_t n = 0; n < 16U; ++n) {
    const auto nibble = static_cast<uint8_t>(n);
    const Float4_e2m1fn lane{nibble, Float4_e2m1fn::from_bits()};

    EXPECT_FALSE(lane.isnan()) << "nibble=0x" << std::hex << n;
    EXPECT_FALSE(lane.isinf()) << "nibble=0x" << std::hex << n;

    const float value = static_cast<float>(lane);
    EXPECT_GE(std::fabs(value), 0.0F) << "nibble=0x" << std::hex << n;
    EXPECT_LE(std::fabs(value), 6.0F) << "nibble=0x" << std::hex << n;

    // Class via the decoded bits: zero or normal only (0.5 is the lone
    // subnormal magnitude but widens to a normal float32).
    const FpClass cls = classifyDecodedF32(f32BitsOf(value));
    EXPECT_TRUE(cls == FpClass::kZero || cls == FpClass::kNormal)
        << "nibble=0x" << std::hex << n;
  }
}

/**
 * @brief Verifies that negating a pattern flips only the sign bit and
 *        preserves the value class, over sampled patterns per format.
 */
TEST(SpecialValuesClassification, NegationPreservesClassFlipsSign) {
  auto formats = buildFormats();
  for (const FormatUnderTest &fmt : formats) {
    SCOPED_TRACE(fmt.label);
    const uint32_t totalBits =
        static_cast<uint32_t>(1 + fmt.model.expBits + fmt.model.manBits);
    const uint32_t signBit = 1U << (totalBits - 1);
    const uint32_t maxExpField = (1U << fmt.model.expBits) - 1U;
    const uint32_t maxMag =
        (maxExpField << fmt.model.manBits) | ((1U << fmt.model.manBits) - 1U);

    // Sample: zero, min/max subnormal, first/last normal, specials.
    std::array<uint32_t, 6> samples{{
        0U,
        1U,
        (1U << fmt.model.manBits) - 1U,
        (1U << fmt.model.manBits),
        maxMag,
        (maxExpField << fmt.model.manBits) | 1U, // canonical NaN slot
    }};

    for (const uint32_t p : samples) {
      if (p > maxMag) {
        continue;
      }
      const FpClass positive = tests::fpc::classify(p, fmt.model);
      const FpClass negative = tests::fpc::classify(p | signBit, fmt.model);
      EXPECT_EQ(negative, positive)
          << fmt.label << " pattern=0x" << std::hex << p;
      EXPECT_TRUE(tests::fpc::signBit(p | signBit, fmt.model))
          << fmt.label << " pattern=0x" << std::hex << p;
    }
  }
}

/**
 * @brief Verifies human-readable sentinel classifications, including the
 *        finite-only trap: E4M3FN's 0x7E is a NORMAL value, not special.
 */
TEST(SpecialValuesClassification, CanonicalPatternsClassifyAsDocumented) {
  struct Entry {
    const char *label;
    uint32_t pattern;
    tests::fpc::FormatModel model;
    FpClass want;
  };
  constexpr std::array<Entry, 12> kSentinels{{
      {"fp16 qNaN", 0x7E00, tests::fpc::kFp16Model, FpClass::kNaN},
      {"fp16 inf", 0x7C00, tests::fpc::kFp16Model, FpClass::kInf},
      {"fp16 max sub", 0x03FF, tests::fpc::kFp16Model, FpClass::kSubnormal},
      {"fp16 min normal", 0x0400, tests::fpc::kFp16Model, FpClass::kNormal},
      {"bf16 qNaN", 0x7FC0, tests::fpc::kBf16Model, FpClass::kNaN},
      {"bf16 inf", 0x7F80, tests::fpc::kBf16Model, FpClass::kInf},
      {"e4m3 max finite", 0x7E, tests::fpc::kFp8E4M3Model, FpClass::kNormal},
      {"e4m3 NaN", 0x7F, tests::fpc::kFp8E4M3Model, FpClass::kNaN},
      {"e5m2 max finite", 0x7B, tests::fpc::kFp8E5M2Model, FpClass::kNormal},
      {"e5m2 inf", 0x7C, tests::fpc::kFp8E5M2Model, FpClass::kInf},
      {"e5m2 NaN", 0x7D, tests::fpc::kFp8E5M2Model, FpClass::kNaN},
      {"e5m2 zero", 0x00, tests::fpc::kFp8E5M2Model, FpClass::kZero},
  }};
  for (const auto &s : kSentinels) {
    EXPECT_EQ(tests::fpc::classify(s.pattern, s.model), s.want) << s.label;
  }
}
