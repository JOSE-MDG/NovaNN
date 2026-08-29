/**
 * @file SaturationInvariant_test.cpp
 * @brief Saturation-policy tests for float->int and narrowing int->int
 *        cast kernels.
 *
 * Every float->int kernel clamps out-of-range inputs in the SOURCE float
 * precision and maps NaN to 0 (defined behavior for *valid* float inputs;
 * NaN/inf/overflow are values, not precondition violations). Integer
 * width/sign changes follow the kernels' per-pair policy: clamping into
 * the destination range for the documented 8-bit-target set, modular
 * truncation elsewhere (see kKernelClamps in utils/Oracle.hpp). Tail
 * loops of the SIMD paths must saturate exactly like the vectorized bodies.
 *
 * All kernels are driven through their portable _scalar entry points plus,
 * where a tail test is involved, the live dispatch wrapper so the selected
 * SIMD variant's tails are exercised too.
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/headeronly/cast.h>
#include <ncore/headeronly/macros.h>

#include "utils/Oracle.hpp"

namespace {

using tests::casting::allPairs;
using tests::casting::kDefaultSeed;
using tests::casting::makePair;
using tests::casting::payloadEqual;
using tests::casting::quantizeToSource;
using tests::casting::readAs;
using tests::casting::referenceSaturate;
using tests::casting::specialValuesOf;
using tests::casting::writeFloatValue;
using tests::casting::writeRaw;

/// One parameterized cast pair.
struct SatParam {
  DType_ src;
  DType_ dst;
  std::string label;
};

/// Builds the float->int and int->int parameter sets from the registry.
std::vector<SatParam> buildParams() {
  std::vector<SatParam> params;
  for (const auto &p : allPairs()) {
    const bool srcFloat = tests::casting::isFloatDtype(p.src);
    const bool dstFloat = tests::casting::isFloatDtype(p.dst);
    if (srcFloat && !dstFloat) {
      params.push_back(SatParam{.src = p.src, .dst = p.dst, .label = p.label});
    } else if (!srcFloat && !dstFloat) {
      params.push_back(SatParam{.src = p.src, .dst = p.dst, .label = p.label});
    }
  }
  return params;
}

/// Parameterized fixture over the float->int and int->int pair set.
class SaturationInvariant : public ::testing::TestWithParam<SatParam> {};
} // namespace

/**
 * @brief Verifies overflow above the destination maximum clamps to max.
 */
TEST_P(SaturationInvariant, FloatOverflowClampsToMax) {
  const SatParam &p = GetParam();
  if (!tests::casting::isFloatDtype(p.src)) {
    GTEST_SKIP() << "float-only case";
  }

  for (const double v : {1e30, HUGE_VAL}) {
    SCOPED_TRACE(::testing::Message() << p.label << " v=" << v);
    auto pair = makePair(p.src, p.dst, 4, kDefaultSeed);
    ASSERT_TRUE(pair.ok);

    for (size_t i = 0; i < 4; ++i) {
      writeFloatValue(pair.src.mutableCTensor(), i, v);
    }
    runScalar(pair);

    tests::casting::withIntType(p.dst, [&]<typename D>() {
      for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(readAs<D>(pair.dst.getCTensor(), i),
                  referenceSaturate<D>(quantizeToSource(p.src, v), p.src))
            << "element=" << i;
      }
    });
  }
}

/**
 * @brief Verifies underflow below the destination minimum clamps to min
 *        (0 for unsigned destinations).
 */
TEST_P(SaturationInvariant, FloatUnderflowClampsToMin) {
  const SatParam &p = GetParam();
  if (!tests::casting::isFloatDtype(p.src)) {
    GTEST_SKIP() << "float-only case";
  }

  for (const double v : {-1e30, -HUGE_VAL}) {
    SCOPED_TRACE(::testing::Message() << p.label << " v=" << v);
    auto pair = makePair(p.src, p.dst, 4, kDefaultSeed);
    ASSERT_TRUE(pair.ok);

    for (size_t i = 0; i < 4; ++i) {
      writeFloatValue(pair.src.mutableCTensor(), i, v);
    }
    runScalar(pair);

    tests::casting::withIntType(p.dst, [&]<typename D>() {
      for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(readAs<D>(pair.dst.getCTensor(), i),
                  referenceSaturate<D>(quantizeToSource(p.src, v), p.src))
            << "element=" << i;
      }
    });
  }
}

/**
 * @brief Verifies NaN inputs map to 0 in every integer destination.
 */
TEST_P(SaturationInvariant, NanMapsToZero) {
  const SatParam &p = GetParam();
  if (!tests::casting::isFloatDtype(p.src)) {
    GTEST_SKIP() << "float-only case";
  }

  for (const auto &sv : specialValuesOf(p.src)) {
    // Only canonical quiet-NaN patterns of this source format.
    const std::string name(sv.name);
    if (name != "qnan_pos" && name != "qnan_neg") {
      continue;
    }
    SCOPED_TRACE(::testing::Message() << p.label << " " << sv.name);

    auto pair = makePair(p.src, p.dst, 2, kDefaultSeed);
    ASSERT_TRUE(pair.ok);
    writeRaw(pair.src.mutableCTensor(), 0, sv.pattern);
    runScalar(pair);

    tests::casting::withIntType(p.dst, [&]<typename D>() {
      EXPECT_EQ(readAs<D>(pair.dst.getCTensor(), 0), D{0}) << sv.name;
    });
  }
}

/**
 * @brief Verifies strictly in-range values truncate toward zero unchanged
 *        by any clamping logic.
 */
TEST_P(SaturationInvariant, InRangeValuesUnaffected) {
  const SatParam &p = GetParam();
  if (!tests::casting::isFloatDtype(p.src)) {
    GTEST_SKIP() << "float-only case";
  }

  for (const double v : {-2.9, -0.5, 0.5, 2.9}) {
    SCOPED_TRACE(::testing::Message() << p.label << " v=" << v);
    auto pair = makePair(p.src, p.dst, 1, kDefaultSeed);
    ASSERT_TRUE(pair.ok);
    writeFloatValue(pair.src.mutableCTensor(), 0, v);
    runScalar(pair);

    tests::casting::withIntType(p.dst, [&]<typename D>() {
      EXPECT_EQ(readAs<D>(pair.dst.getCTensor(), 0),
                referenceSaturate<D>(quantizeToSource(p.src, v), p.src))
          << "v=" << v;
    });
  }
}

/**
 * @brief Pins exact boundary conversions: every destination-boundary
 *        input converts exactly as the kernel-faithful oracle
 *        predicts for THIS source format, including the float-precision
 *        clamp constants (which differ between f32- and f64-sourced
 *        kernels for wide destinations).
 */
TEST_P(SaturationInvariant, ExactBoundariesMapToExtremes) {
  const SatParam &p = GetParam();
  if (!tests::casting::isFloatDtype(p.src)) {
    GTEST_SKIP() << "float-only case";
  }

  struct Pin {
    double input;
    const char *name;
  };

  std::vector<Pin> pins;
  switch (p.dst) {
  case DType_::Signed8:
    pins = {{.input = 127.0, .name = "s8 max"},
            {.input = -128.0, .name = "s8 min"},
            {.input = 255.0, .name = "s8 clamp-high"}};
    break;
  case DType_::UnSigned8:
    pins = {{.input = 255.0, .name = "u8 max"},
            {.input = 0.0, .name = "u8 zero"},
            {.input = -1.0, .name = "u8 clamp-low"}};
    break;
  case DType_::Signed16:
    pins = {{.input = 32767.0, .name = "s16 max"},
            {.input = -32768.0, .name = "s16 min"}};
    break;
  case DType_::UnSigned16:
    pins = {{.input = 65535.0, .name = "u16 max"}};
    break;
  case DType_::Signed32:
    // Largest f32 below 2^31: the kernel's clamp constant.
    pins = {{.input = 2147483520.0, .name = "f32 s32 hi pin"},
            {.input = -2147483648.0, .name = "f32 s32 lo pin"}};
    break;
  case DType_::UnSigned32:
    pins = {{.input = 4294967040.0, .name = "f32 u32 hi pin"}};
    break;
  case DType_::Signed64:
    pins = {{.input = 9223371487098961920.0, .name = "f32 s64 hi pin"},
            {.input = -9223372036854775808.0, .name = "f32 s64 lo pin"}};
    break;
  case DType_::UnSigned64:
    pins = {{.input = 18446744073709549568.0, .name = "f64 u64 hi pin"}};
    break;
  default:
    GTEST_SKIP() << "non-integer destination";
  }

  for (const Pin &pin : pins) {
    SCOPED_TRACE(::testing::Message() << p.label << " " << pin.name);
    auto pair = makePair(p.src, p.dst, 1, kDefaultSeed);
    ASSERT_TRUE(pair.ok);
    writeFloatValue(pair.src.mutableCTensor(), 0, pin.input);
    runScalar(pair);

    tests::casting::withIntType(p.dst, [&]<typename D>() {
      EXPECT_EQ(readAs<D>(pair.dst.getCTensor(), 0),
                referenceSaturate<D>(quantizeToSource(p.src, pin.input), p.src))
          << pin.name;
    });
  }
}

/**
 * @brief Verifies integer width/sign conversions match each kernel's
 *        per-pair policy: clamping into the destination range for the
 *        documented 8-bit-target set (plus u8 -> s8), modular
 *        two's-complement truncation everywhere else.
 */
TEST_P(SaturationInvariant, IntConversionsMatchKernelPolicy) {
  const SatParam &p = GetParam();
  if (tests::casting::isFloatDtype(p.src) ||
      tests::casting::isFloatDtype(p.dst)) {
    GTEST_SKIP() << "int-only case";
  }

  tests::casting::withIntType(p.src, [&]<typename S>() {
    tests::casting::withIntType(p.dst, [&]<typename D>() {
      const std::array<S, 3> values{std::numeric_limits<S>::min(),
                                    std::numeric_limits<S>::max(),
                                    static_cast<S>(1)};
      for (const S v : values) {
        SCOPED_TRACE(::testing::Message()
                     << p.label << " v=" << static_cast<int64_t>(v));
        auto pair = makePair(p.src, p.dst, 1, kDefaultSeed);
        ASSERT_TRUE(pair.ok);
        writeRaw(pair.src.mutableCTensor(), 0,
                 static_cast<uint64_t>(static_cast<int64_t>(v)));
        runScalar(pair);
        EXPECT_EQ(readAs<D>(pair.dst.getCTensor(), 0),
                  (tests::casting::referenceNarrow<S, D>)(v))
            << p.label;
      }
    });
  });
}

/**
 * @brief Verifies saturation holds in the SIMD tail region: out-of-range
 *        values placed only past the vectorized body still clamp, and the
 *        live dispatch wrapper agrees with the scalar kernel bitwise.
 */
TEST_P(SaturationInvariant, TailElementsAlsoSaturate) {
  const SatParam &p = GetParam();
  if (!tests::casting::isFloatDtype(p.src)) {
    GTEST_SKIP() << "float-only case";
  }

  // Sizes chosen so every implemented SIMD step width (2/4/8/16 lanes)
  // sees a non-empty remainder; 16 would be fully consumed by a 16-lane
  // AVX-512 body and exercise no tail at all.
  for (const size_t count : {size_t{15}, size_t{17}, size_t{31}, size_t{33}}) {
    SCOPED_TRACE(::testing::Message() << p.label << " count=" << count);
    const size_t bodyLen = count >= 16 ? count - 3 : count - 1;

    auto scalarPair = makePair(p.src, p.dst, count, kDefaultSeed);
    auto wrapperPair = makePair(p.src, p.dst, count, kDefaultSeed);
    ASSERT_TRUE(scalarPair.ok);
    ASSERT_TRUE(wrapperPair.ok);

    for (size_t i = 0; i < count; ++i) {
      const double v = i < bodyLen ? 1.0 : 1e30;
      writeFloatValue(scalarPair.src.mutableCTensor(), i, v);
      writeFloatValue(wrapperPair.src.mutableCTensor(), i, v);
    }

    runScalar(scalarPair);
    const tests::casting::PairInfo *info =
        tests::casting::findPairFor(p.src, p.dst);
    ASSERT_NE(info, nullptr);
    info->wrapper(&wrapperPair.src.mutableCTensor(),
                  &wrapperPair.dst.mutableCTensor());

    size_t mismatch = 0;
    EXPECT_TRUE(payloadEqual(scalarPair.dst.getCTensor(),
                             wrapperPair.dst.getCTensor(), count, &mismatch))
        << "count=" << count << " first mismatch at " << mismatch;

    // The last element sits in the tail and must be saturated to max.
    tests::casting::withIntType(p.dst, [&]<typename D>() {
      EXPECT_EQ(readAs<D>(scalarPair.dst.getCTensor(), count - 1),
                referenceSaturate<D>(quantizeToSource(p.src, 1e30), p.src))
          << "tail not saturated, count=" << count;
    });
  }
}

/**
 * @brief Verifies every FP4 magnitude converts exactly into every integer
 *        destination: FP4's bounded domain never triggers clamping.
 */
TEST(SaturationInvariant, Fp4SourcesNeverTriggerUb) {
  constexpr std::array<DType_, 8> kIntDsts{{
      DType_::Signed8,
      DType_::UnSigned8,
      DType_::Signed16,
      DType_::UnSigned16,
      DType_::Signed32,
      DType_::UnSigned32,
      DType_::Signed64,
      DType_::UnSigned64,
  }};

  for (const DType_ dst : kIntDsts) {
    SCOPED_TRACE(testing::Message() << "dst=" << static_cast<int>(dst));
    auto pair = makePair(DType_::Float4E2M1fn, dst, 16, kDefaultSeed);
    ASSERT_TRUE(pair.ok);

    for (uint32_t nibble = 0; nibble < 16U; ++nibble) {
      writeRaw(pair.src.mutableCTensor(), nibble, nibble);
    }
    runScalar(pair);

    tests::casting::withIntType(dst, [&]<typename D>() {
      for (uint32_t nibble = 0; nibble < 16U; ++nibble) {
        const double decoded =
            tests::casting::referenceDecode(tests::casting::kFp4E2M1, nibble);
        EXPECT_EQ(readAs<D>(pair.dst.getCTensor(), nibble),
                  referenceSaturate<D>(decoded, DType_::Float4E2M1fn))
            << "nibble=0x" << std::hex << nibble;
      }
    });
  }
}

INSTANTIATE_TEST_SUITE_P(AllPairs, SaturationInvariant,
                         ::testing::ValuesIn(buildParams()),
                         [](const ::testing::TestParamInfo<SatParam> &info) {
                           std::string name;
                           for (const char c : info.param.label) {
                             name.push_back((c == '-' || c == '>') ? '_' : c);
                           }
                           return name;
                         });
