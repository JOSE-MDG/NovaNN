/**
 * @file SpecialValueInvariant_test.cpp
 * @brief NaN/Inf/negative-zero/subnormal propagation through every
 *        float-involving cast pair.
 *
 * The reduced formats have deliberately asymmetric special-value policies:
 * E4M3FN and FP4 are finite-only substitutions, and BF16 canonicalizes NaN
 * to 0x7FC0 dropping the sign. Expectations therefore come from a
 * declarative policy table here, not from assumed IEEE behavior. Every case
 * also verifies that the live dispatch wrapper agrees with the portable
 * scalar kernel (bitwise except NaN payloads, where only the class contract
 * is required), which is what catches ISA variants mishandling specials.
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/headeronly/cast.h>
#include <ncore/headeronly/macros.h>

#include "utils/Oracle.hpp"

namespace {

using tests::casting::allPairs;
using tests::casting::f32BitsOf;
using tests::casting::FormatSpec;
using tests::casting::isFloatDtype;
using tests::casting::isNaNPattern;
using tests::casting::kDefaultSeed;
using tests::casting::makePair;
using tests::casting::readAs;
using tests::casting::readRaw;
using tests::casting::referenceDecode;
using tests::casting::referenceEncode;
using tests::casting::referenceSaturate;
using tests::casting::specialValuesOf;
using tests::casting::specOf;
using tests::casting::writeFloatValue;
using tests::casting::writeRaw;

struct SviParam {
  DType_ src;
  DType_ dst;
  std::string label;
};

std::vector<SviParam> buildParams() {
  std::vector<SviParam> params;
  for (const auto &p : allPairs()) {
    if (isFloatDtype(p.src)) {
      params.push_back(SviParam{.src = p.src, .dst = p.dst, .label = p.label});
    }
  }
  return params;
}

enum class NanPolicy : std::uint8_t { CanonicalNan, SaturateMaxFinite };
enum class InfPolicy : std::uint8_t { PreserveInfinity, SaturateMaxFinite };

InfPolicy infPolicyOf(DType_ dst) {
  switch (dst) {
  case DType_::Float16:
  case DType_::BFloat16:
  case DType_::Float8E5M2:
  case DType_::Float32:
  case DType_::Float64:
    return InfPolicy::PreserveInfinity;
  default:
    return InfPolicy::SaturateMaxFinite;
  }
}

/// Sign bit of a reduced-format raw pattern.
bool signRaw(DType_ dtype, uint64_t pattern) {
  const uint64_t total =
      static_cast<uint64_t>(tests::casting::totalBits(specOf(dtype)));
  return ((pattern >> (total - 1)) & 1ULL) != 0ULL;
}

/// Checks destination element @p idx against the policy for a NaN source.
void expectNanResult(const SviParam &p, bool srcNegative, const Tensor &dst,
                     size_t idx) {
  if (p.dst == DType_::Float4E2M1fn) {
    // Finite-only: saturates to max magnitude with the source sign.
    const uint64_t want =
        (srcNegative ? UINT64_C(0x8) : UINT64_C(0x0)) | UINT64_C(0x7);
    EXPECT_EQ(readRaw(dst, idx), want) << p.label;
    return;
  }
  if (p.dst == DType_::BFloat16) {
    // Documented: bf16_from_float canonicalizes to 0x7FC0, dropping sign.
    EXPECT_EQ(readRaw(dst, idx), UINT64_C(0x7FC0)) << p.label;
    return;
  }
  // IEEE-style destinations: exponent field all ones, sign preserved.
  const FormatSpec spec = specOf(p.dst);
  const uint64_t maxExp = (1ULL << spec.expBits) - 1ULL;
  const uint64_t got = readRaw(dst, idx);
  EXPECT_EQ((got >> spec.manBits) & maxExp, maxExp) << p.label;
  EXPECT_EQ(signRaw(p.dst, got), srcNegative) << p.label;
}

/// Checks destination element @p idx against the policy for an Inf source.
void expectInfResult(const SviParam &p, bool srcNegative, const Tensor &dst,
                     size_t idx) {
  const FormatSpec spec = specOf(p.dst);
  const uint64_t total = static_cast<uint64_t>(tests::casting::totalBits(spec));
  const uint64_t sign = srcNegative ? (1ULL << (total - 1)) : 0ULL;

  if (infPolicyOf(p.dst) == InfPolicy::PreserveInfinity) {
    const uint64_t want =
        sign | (((1ULL << spec.expBits) - 1ULL) << spec.manBits);
    EXPECT_EQ(readRaw(dst, idx), want) << p.label;
  } else {
    EXPECT_EQ(readRaw(dst, idx),
              sign | tests::casting::maxFiniteMagnitude(spec))
        << p.label;
  }
}

/// Parameterized fixture over every float-source cast pair.
class SpecialValueInvariant : public ::testing::TestWithParam<SviParam> {};
} // namespace

/**
 * @brief Verifies NaN propagation follows each destination's documented
 *        policy, with a known-good canary element guarding against smear.
 */
TEST_P(SpecialValueInvariant, NanPropagatesPerDestinationPolicy) {
  const SviParam &p = GetParam();
  if (!isFloatDtype(p.dst)) {
    GTEST_SKIP() << "float-destination case";
  }

  for (const auto &sv : specialValuesOf(p.src)) {
    const std::string name(sv.name);
    if (name != "qnan_pos" && name != "qnan_neg") {
      continue;
    }
    SCOPED_TRACE(::testing::Message() << p.label << " " << sv.name);

    auto pair = makePair(p.src, p.dst, 2, kDefaultSeed);
    ASSERT_TRUE(pair.ok);
    writeRaw(pair.src.mutableCTensor(), 0, sv.pattern);
    writeRaw(pair.src.mutableCTensor(), 1,
             tests::casting::minNormalMagnitude(specOf(p.src)));
    runScalar(pair);

    expectNanResult(p, name == "qnan_neg", pair.dst.getCTensor(), 0);

    // Canary element must convert normally, proving no lane/element smear.
    const double canaryDecoded = referenceDecode(
        specOf(p.src), tests::casting::minNormalMagnitude(specOf(p.src)));
    EXPECT_EQ(readRaw(pair.dst.getCTensor(), 1),
              tests::casting::referenceEncode(specOf(p.dst), canaryDecoded))
        << p.label << " canary drifted";
  }
}

/**
 * @brief Verifies infinity maps to infinity or to the finite-only
 *        saturation target, per destination format.
 */
TEST_P(SpecialValueInvariant, InfinityMapsToInfOrSaturates) {
  const SviParam &p = GetParam();
  if (!isFloatDtype(p.dst)) {
    GTEST_SKIP() << "float-destination case";
  }
  if (!specOf(p.src).hasInf) {
    // E4M3FN and FP4 have no infinity encoding; their "inf" registry
    // entries hold the saturation target (max finite), which is an
    // ordinary value, not a special.
    GTEST_SKIP() << "source format has no infinity";
  }

  for (const auto &sv : specialValuesOf(p.src)) {
    const std::string name(sv.name);
    if (name != "inf_pos" && name != "inf_neg") {
      continue;
    }
    SCOPED_TRACE(::testing::Message() << p.label << " " << sv.name);
    const bool negative = name == "inf_neg";

    auto pair = makePair(p.src, p.dst, 1, kDefaultSeed);
    ASSERT_TRUE(pair.ok);
    writeRaw(pair.src.mutableCTensor(), 0, sv.pattern);
    runScalar(pair);

    expectInfResult(p, negative, pair.dst.getCTensor(), 0);
  }
}

/**
 * @brief Verifies negative zero survives casts bit-exactly across all
 *        float-destination pairs.
 */
TEST_P(SpecialValueInvariant, NegativeZeroIsPreservedBitwise) {
  const SviParam &p = GetParam();
  if (!isFloatDtype(p.dst)) {
    GTEST_SKIP() << "float-destination case";
  }

  for (const auto &sv : specialValuesOf(p.src)) {
    const std::string name(sv.name);
    if (name != "zero_pos" && name != "zero_neg") {
      continue;
    }
    SCOPED_TRACE(::testing::Message() << p.label << " " << sv.name);

    auto pair = makePair(p.src, p.dst, 1, kDefaultSeed);
    ASSERT_TRUE(pair.ok);
    writeRaw(pair.src.mutableCTensor(), 0, sv.pattern);
    runScalar(pair);

    const FormatSpec dstSpec = specOf(p.dst);
    const uint64_t wantSign =
        name == "zero_neg" ? (1ULL << (tests::casting::totalBits(dstSpec) - 1))
                           : 0ULL;
    EXPECT_EQ(readRaw(pair.dst.getCTensor(), 0), wantSign) << p.label;
  }
}

/**
 * @brief Verifies subnormal sources round or flush to the exact
 *        oracle-composed destination pattern.
 */
TEST_P(SpecialValueInvariant, SubnormalsRoundOrFlushCorrectly) {
  const SviParam &p = GetParam();
  if (!isFloatDtype(p.dst)) {
    GTEST_SKIP() << "float-destination case";
  }
#if defined(__clang__)
  // Kernel bodies call the bf16 C API, whose __bf16 ABI edges are
  // re-narrowed by Clang; see
  // ncore/tests/dtypes/utils/BFloat16ClangLimitations.md.
  if (p.src == DType_::BFloat16 || p.dst == DType_::BFloat16) {
    GTEST_SKIP() << "bf16 pair under Clang __bf16 ABI limitation";
  }
#endif

  for (const auto &sv : specialValuesOf(p.src)) {
    const std::string name(sv.name);
    if (name != "min_sub" && name != "max_sub") {
      continue;
    }
    SCOPED_TRACE(::testing::Message() << p.label << " " << sv.name);

    auto pair = makePair(p.src, p.dst, 1, kDefaultSeed);
    ASSERT_TRUE(pair.ok);
    writeRaw(pair.src.mutableCTensor(), 0, sv.pattern);
    runScalar(pair);

    const double decoded = referenceDecode(specOf(p.src), sv.pattern);
    EXPECT_EQ(readRaw(pair.dst.getCTensor(), 0),
              referenceEncode(specOf(p.dst), decoded))
        << p.label << " " << sv.name;
  }
}

/**
 * @brief Verifies the defined FP->INT special-value results: NaN -> 0,
 *        +/-inf -> clamp bounds, -0.0 -> 0.
 */
TEST_P(SpecialValueInvariant, FpToIntSpecialsAreDefined) {
  const SviParam &p = GetParam();
  if (isFloatDtype(p.dst)) {
    GTEST_SKIP() << "int-destination case";
  }
  if (p.src == DType_::Float4E2M1fn) {
    // FP4 has no NaN or infinity encoding: there is no special to inject.
    GTEST_SKIP() << "finite-only source format";
  }

  struct Case {
    const char *srcName;
    double value;
  };
  // E4M3FN has no infinity: its would-be "inf" inputs are ordinary large
  // finite values, covered by the saturation suites instead.
  const bool srcHasInf = specOf(p.src).hasInf;
  // Drive through values rather than patterns: writeFloatValue quantizes
  // into the source format, and +/-inf survive as themselves there. NaN is
  // written as an explicit pattern instead (see below).
  const std::array<Case, 3> cases{{{.srcName = "pinf", .value = HUGE_VAL},
                                   {.srcName = "ninf", .value = -HUGE_VAL},
                                   {.srcName = "nzero", .value = -0.0}}};
  for (const auto &c : cases) {
    if (std::string(c.srcName) != "nzero" && !srcHasInf) {
      continue;
    }
    SCOPED_TRACE(::testing::Message() << p.label << " " << c.srcName);
    auto pair = makePair(p.src, p.dst, 1, kDefaultSeed);
    ASSERT_TRUE(pair.ok);

    writeFloatValue(pair.src.mutableCTensor(), 0, c.value);
    runScalar(pair);

    tests::casting::withIntType(p.dst, [&]<typename D>() {
      D want{};
      if (std::string(c.srcName) == "pinf") {
        want = referenceSaturate<D>(HUGE_VAL, p.src);
      } else if (std::string(c.srcName) == "ninf") {
        want = referenceSaturate<D>(-HUGE_VAL, p.src);
      } else {
        want = D{0};
      }
      EXPECT_EQ(readAs<D>(pair.dst.getCTensor(), 0), want) << c.srcName;
    });
  }

  // NaN via explicit source patterns.
  for (const auto &sv : specialValuesOf(p.src)) {
    const std::string name(sv.name);
    if (name != "qnan_pos" && name != "qnan_neg") {
      continue;
    }
    SCOPED_TRACE(::testing::Message() << p.label << " " << name);
    auto pair = makePair(p.src, p.dst, 1, kDefaultSeed);
    ASSERT_TRUE(pair.ok);
    writeRaw(pair.src.mutableCTensor(), 0, sv.pattern);
    runScalar(pair);

    tests::casting::withIntType(p.dst, [&]<typename D>() {
      EXPECT_EQ(readAs<D>(pair.dst.getCTensor(), 0), D{0}) << name;
    });
  }
}

/**
 * @brief Verifies the live dispatch wrapper agrees with the scalar kernel
 *        across the entire special-value matrix: bitwise for every non-NaN
 *        source element, and per the documented class contract for NaN
 *        sources (payload handling legitimately differs across hardware
 *        and software conversion paths).
 */
TEST_P(SpecialValueInvariant, DispatchAgreesWithScalarOnSpecials) {
  const SviParam &p = GetParam();
#if defined(__clang__)
  // Kernel bodies call the bf16 C API, whose __bf16 ABI edges are
  // re-narrowed by Clang; see
  // ncore/tests/dtypes/utils/BFloat16ClangLimitations.md.
  if (p.src == DType_::BFloat16 || p.dst == DType_::BFloat16) {
    GTEST_SKIP() << "bf16 pair under Clang __bf16 ABI limitation";
  }
#endif

  const tests::casting::PairInfo *info =
      tests::casting::findPairFor(p.src, p.dst);
  ASSERT_NE(info, nullptr) << p.label;

  for (const auto &sv : specialValuesOf(p.src)) {
    SCOPED_TRACE(::testing::Message() << p.label << " " << sv.name);
    auto scalarPair = makePair(p.src, p.dst, 2, kDefaultSeed);
    auto wrapperPair = makePair(p.src, p.dst, 2, kDefaultSeed);
    ASSERT_TRUE(scalarPair.ok);
    ASSERT_TRUE(wrapperPair.ok);

    for (auto *t : {&scalarPair, &wrapperPair}) {
      writeRaw(t->src.mutableCTensor(), 0, sv.pattern);
      writeRaw(t->src.mutableCTensor(), 1,
               tests::casting::minNormalMagnitude(specOf(p.src)));
    }

    info->scalar(&scalarPair.src.mutableCTensor(),
                 &scalarPair.dst.mutableCTensor());
    info->wrapper(&wrapperPair.src.mutableCTensor(),
                  &wrapperPair.dst.mutableCTensor());

    const Tensor &srcView = scalarPair.src.getCTensor();
    const Tensor &scalarDst = scalarPair.dst.getCTensor();
    const Tensor &wrapperDst = wrapperPair.dst.getCTensor();
    for (size_t i = 0; i < 2; ++i) {
      if (!isNaNPattern(p.src, readRaw(srcView, i))) {
        EXPECT_EQ(readRaw(scalarDst, i), readRaw(wrapperDst, i))
            << "element=" << i;
        continue;
      }
      // NaN source: only the class contract is guaranteed across paths.
      // FP4 has no NaN encoding and saturates instead; integer
      // destinations map NaN to 0 deterministically.
      if (p.dst == DType_::Float4E2M1fn) {
        expectNanResult(p, std::string(sv.name) == "qnan_neg", wrapperDst, i);
      } else if (isFloatDtype(p.dst)) {
        EXPECT_TRUE(isNaNPattern(p.dst, readRaw(wrapperDst, i)))
            << "element=" << i << " is not a NaN";
      } else {
        EXPECT_EQ(readRaw(wrapperDst, i), UINT64_C(0))
            << "element=" << i << " NaN did not map to 0";
      }
    }
  }
}

/**
 * @brief Verifies packed FP4 lanes carry specials independently through the
 *        unpacking kernels.
 */
TEST(SpecialValueInvariant, PackedFp4LanesCarrySpecialsIndependently) {
  auto pair = makePair(DType_::Float4E2M1fn, DType_::Float32, 2, kDefaultSeed);
  ASSERT_TRUE(pair.ok);

  // Byte 0x07: low lane +6.0 (max), high lane +0.0.
  writeRaw(pair.src.mutableCTensor(), 0, 0x7);
  writeRaw(pair.src.mutableCTensor(), 1, 0x0);
  runScalar(pair);

  EXPECT_FLOAT_EQ(tests::casting::readAsF32(pair.dst.getCTensor(), 0), 6.0F);
  EXPECT_FLOAT_EQ(tests::casting::readAsF32(pair.dst.getCTensor(), 1), 0.0F);
  EXPECT_FALSE(signRaw(DType_::Float32, f32BitsOf(tests::casting::readAsF32(
                                            pair.dst.getCTensor(), 1))));

  // Byte 0x87: low lane -6.0, high lane -0.0; signs stay per-lane.
  writeRaw(pair.src.mutableCTensor(), 0, 0xF);
  writeRaw(pair.src.mutableCTensor(), 1, 0x8);
  runScalar(pair);

  EXPECT_FLOAT_EQ(tests::casting::readAsF32(pair.dst.getCTensor(), 0), -6.0F);
  const uint32_t negZeroBits =
      f32BitsOf(tests::casting::readAsF32(pair.dst.getCTensor(), 1));
  EXPECT_EQ(negZeroBits, UINT32_C(0x80000000));
}

INSTANTIATE_TEST_SUITE_P(AllPairs, SpecialValueInvariant,
                         ::testing::ValuesIn(buildParams()),
                         [](const ::testing::TestParamInfo<SviParam> &info) {
                           std::string name;
                           for (const char c : info.param.label) {
                             name.push_back((c == '-' || c == '>') ? '_' : c);
                           }
                           return name;
                         });
