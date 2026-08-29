/**
 * @file ScalarOracle_test.cpp
 * @brief Verifies every portable scalar cast kernel against an independent
 *        double-precision oracle.
 *
 * The scalar fallbacks are the semantic bedrock of the casting subsystem:
 * SIMD variants are only ever measured relative to them, so this suite
 * anchors them to a reference implemented from the format definitions alone
 * (referenceDecode/referenceEncode in utils/Oracle.hpp), never another
 * kernel. Whole-tensor sweeps use curated candidate lists plus
 * deterministic LCG fills; no unseeded randomness.
 *
 * Int -> reduced-float expectations deliberately mirror the kernels'
 * narrowing chain (int64/32 values pass through float before the reduced
 * encode), so the oracle checks the documented pipeline rather than an
 * idealized single-rounding model.
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/headeronly/cast.h>
#include <ncore/headeronly/macros.h>

#include "utils/Oracle.hpp"

namespace {

using tests::casting::allPairs;
using tests::casting::f32BitsOf;
using tests::casting::f64BitsOf;
using tests::casting::FormatSpec;
using tests::casting::isFloatDtype;
using tests::casting::isNaNPattern;
using tests::casting::kBf16;
using tests::casting::kDefaultSeed;
using tests::casting::kFp16;
using tests::casting::kFp4E2M1;
using tests::casting::kFp8E4M3;
using tests::casting::kFp8E5M2;
using tests::casting::makePair;
using tests::casting::quantizeToSource;
using tests::casting::readAs;
using tests::casting::readRaw;
using tests::casting::referenceDecode;
using tests::casting::referenceEncode;
using tests::casting::referenceSaturate;
using tests::casting::specOf;
using tests::casting::writeFloatValue;
using tests::casting::writeRaw;

enum class Family : uint8_t { FpToFp, FpToInt, IntToFp, IntToInt };

Family familyOf(const tests::casting::PairInfo &p) {
  const bool srcFloat = isFloatDtype(p.src);
  const bool dstFloat = isFloatDtype(p.dst);
  if (srcFloat && dstFloat) {
    return Family::FpToFp;
  }
  if (srcFloat) {
    return Family::FpToInt;
  }
  if (dstFloat) {
    return Family::IntToFp;
  }
  return Family::IntToInt;
}

/// Expected raw destination pattern for a floating destination.
uint64_t expectedFloatPattern(DType_ dst, double decoded) {
  switch (dst) {
  case DType_::Float32:
    return f32BitsOf(static_cast<float>(decoded));
  case DType_::Float64:
    return f64BitsOf(decoded);
  default:
    return referenceEncode(specOf(dst), decoded);
  }
}

/// Deterministic finite candidate values for one floating source dtype.
std::vector<double> interestingValues(DType_ dtype) {
  std::vector<double> vals;
  auto pushSigned = [&](double v) {
    vals.push_back(v);
    vals.push_back(-v);
  };

  if (dtype == DType_::Float32 || dtype == DType_::Float64) {
    for (const double mag : {0.25, 0.5, 1.0, 1.5, 2.0, 3.0}) {
      for (int e = -20; e <= 20; e += 4) {
        pushSigned(mag * std::ldexp(1.0, e));
      }
    }
    vals.push_back(0.0);
    return vals;
  }

  const FormatSpec spec = specOf(dtype);
  const int emin = 1 - spec.bias;
  // Finite-only formats keep their whole top exponent binade normal, so the
  // maximum unbiased exponent extends past the IEEE bias (E4M3FN: 8, FP4: 2).
  const int emax = spec.hasInf
                       ? spec.bias
                       : static_cast<int>((1 << spec.expBits) - 1) - spec.bias;
  const int minSubExp = emin - spec.manBits;

  for (const double mag : {0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0}) {
    for (int e = emin; e <= emax; ++e) {
      pushSigned(mag * std::ldexp(1.0, e));
    }
  }
  if (!spec.hasInf) {
    // Explicit sweep of every mantissa pattern at the maximum exponent so
    // no top-binade value is skipped by the multiplier grid above (E4M3FN:
    // 256..416; the 448 edge is pushed below; FP4: 2..6).
    const uint64_t maxExpField = (1ULL << spec.expBits) - 1ULL;
    const uint64_t manMask = (1ULL << spec.manBits) - 1ULL;
    for (uint64_t man = 0; man <= manMask; ++man) {
      const uint64_t pattern = (maxExpField << spec.manBits) | man;
      if (isNaNPattern(dtype, pattern)) {
        continue;
      }
      pushSigned(referenceDecode(spec, pattern));
    }
  }
  // Subnormal samples and range edges (overflow handled by the oracle).
  pushSigned(std::ldexp(1.0, minSubExp));
  pushSigned(
      std::ldexp(static_cast<double>((1 << spec.manBits) - 1), minSubExp));
  pushSigned(std::ldexp(1.0, emin));
  pushSigned(referenceDecode(spec, tests::casting::maxFiniteMagnitude(spec)));
  vals.push_back(0.0);
  return vals;
}

/// Deterministic integer candidates for one integer source dtype.
template <typename T> std::vector<T> intCandidates() {
  return {T{0},
          T{1},
          static_cast<T>(-1),
          T{7},
          static_cast<T>(-7),
          static_cast<T>(31),
          std::numeric_limits<T>::min(),
          static_cast<T>(std::numeric_limits<T>::max() / 2),
          std::numeric_limits<T>::max()};
}

} // namespace

/**
 * @brief Harness self-check: the generic oracle decodes textbook sentinel
 *        patterns to their exact documented values.
 */
TEST(ScalarOracle, OracleSelfCheckDecodeMatchesKnownValues) {
  struct Sentinel {
    DType_ dtype;
    uint64_t pattern;
    double want;
    const char *name;
  };
  constexpr std::array<Sentinel, 12> kSentinels{{
      {.dtype = DType_::Float16,
       .pattern = 0x3C00,
       .want = 1.0,
       .name = "fp16 one"},
      {.dtype = DType_::Float16,
       .pattern = 0x0400,
       .want = 6.103515625e-05,
       .name = "fp16 min normal"},
      {.dtype = DType_::Float16,
       .pattern = 0x0001,
       .want = 5.9604644775390625e-08,
       .name = "fp16 min sub"},
      {.dtype = DType_::BFloat16,
       .pattern = 0x3F80,
       .want = 1.0,
       .name = "bf16 one"},
      {.dtype = DType_::BFloat16,
       .pattern = 0x0080,
       .want = 1.1754943508222875e-38,
       .name = "bf16 min normal"},
      {.dtype = DType_::Float8E4M3fn,
       .pattern = 0x38,
       .want = 1.0,
       .name = "e4m3 one"},
      {.dtype = DType_::Float8E4M3fn,
       .pattern = 0x08,
       .want = 0.015625,
       .name = "e4m3 min normal"},
      {.dtype = DType_::Float8E4M3fn,
       .pattern = 0x7E,
       .want = 448.0,
       .name = "e4m3 max finite"},
      {.dtype = DType_::Float8E5M2,
       .pattern = 0x3C,
       .want = 1.0,
       .name = "e5m2 one"},
      {.dtype = DType_::Float8E5M2,
       .pattern = 0x04,
       .want = 6.103515625e-05,
       .name = "e5m2 min normal"},
      {.dtype = DType_::Float8E5M2,
       .pattern = 0x7B,
       .want = 57344.0,
       .name = "e5m2 max finite"},
      {.dtype = DType_::Float4E2M1fn,
       .pattern = 0x07,
       .want = 6.0,
       .name = "fp4 max magnitude"},
  }};
  for (const auto &s : kSentinels) {
    EXPECT_DOUBLE_EQ(referenceDecode(specOf(s.dtype), s.pattern), s.want)
        << s.name;
  }
}

/**
 * @brief Harness self-check: the oracle encodes powers of two to the
 *        field-assembled pattern and round-trips its own finite domain.
 */
TEST(ScalarOracle, OracleSelfCheckEncodePowersOfTwoAndDomain) {
  const std::array<std::pair<FormatSpec, DType_>, 5> formats{{
      {kFp16, DType_::Float16},
      {kBf16, DType_::BFloat16},
      {kFp8E4M3, DType_::Float8E4M3fn},
      {kFp8E5M2, DType_::Float8E5M2},
      {kFp4E2M1, DType_::Float4E2M1fn},
  }};

  for (const auto &[spec, dtype] : formats) {
    SCOPED_TRACE(dtype);

    const int emin = 1 - spec.bias;
    // Finite-only formats keep their top binade normal (E4M3FN: 2^8 = 256
    // is representable), so powers of two extend past the IEEE bias.
    const int emax =
        spec.hasInf ? spec.bias
                    : static_cast<int>((1 << spec.expBits) - 1) - spec.bias;
    for (int e = emin; e <= emax; ++e) {
      const uint64_t want = static_cast<uint64_t>(e + spec.bias)
                            << spec.manBits;
      EXPECT_EQ(referenceEncode(spec, std::ldexp(1.0, e)), want)
          << "power=" << e;
    }

    // Full positive finite domain: decode then re-encode is the identity
    // (NaN slots excluded: their payloads canonicalize on encode).
    const uint64_t maxFinite = tests::casting::maxFiniteMagnitude(spec);
    for (uint64_t mag = 0; mag <= maxFinite; ++mag) {
      if (isNaNPattern(dtype, mag)) {
        continue;
      }
      const double decoded = referenceDecode(spec, mag);
      EXPECT_EQ(referenceEncode(spec, decoded), mag)
          << "magnitude=0x" << std::hex << mag;
    }
  }
}

/**
 * @brief Pins the compiler-flag contract the oracle depends on.
 *
 * The Release configuration builds with ``-ffast-math`` followed by
 * per-flag overrides: ``-fno-finite-math-only`` keeps NaN/Inf
 * classification live, and ``-fsigned-zeros`` keeps the sign of a zero
 * meaningful. If anyone reorders, dedups, or drops those CMake flags,
 * this test fails at the cause instead of letting oracle semantics drift
 * silently.
 */
TEST(ScalarOracle, OracleSelfCheckFlagContract) {
  // Classification must be live: bare -ffast-math folds these to false.
  EXPECT_TRUE(std::isnan(std::numeric_limits<double>::quiet_NaN()));
  EXPECT_TRUE(std::isinf(HUGE_VAL));
  EXPECT_FALSE(std::isnan(1.0));
  EXPECT_FALSE(std::isinf(1.0));

  // Signed zeros must be honored: bare -ffast-math folds this to false.
  const double negZero = -0.0;
  EXPECT_TRUE(std::signbit(negZero));
  EXPECT_FALSE(std::signbit(0.0));

  // Observable oracle contract: negative zero encodes with its sign.
  EXPECT_EQ(referenceEncode(tests::casting::kFp16, -0.0), UINT64_C(0x8000));
  EXPECT_EQ(referenceEncode(tests::casting::kFp8E4M3, -0.0), UINT64_C(0x80));
}

/**
 * @brief Verifies all FP<->FP scalar kernels against the oracle over the
 *        curated candidate lists.
 */
TEST(ScalarOracle, FpToFpMatchesOracle) {
  for (const auto &pair : allPairs()) {
    if (familyOf(pair) != Family::FpToFp) {
      continue;
    }
#if defined(__clang__)
    // Kernel bodies call the bf16 C API, whose __bf16 ABI edges are
    // re-narrowed by Clang; see
    // ncore/tests/dtypes/utils/BFloat16ClangLimitations.md.
    if (pair.src == DType_::BFloat16 || pair.dst == DType_::BFloat16) {
      continue;
    }
#endif
    SCOPED_TRACE(pair.label);

    const std::vector<double> candidates = interestingValues(pair.src);
    auto pairFix =
        makePair(pair.src, pair.dst, candidates.size(), kDefaultSeed);
    ASSERT_TRUE(pairFix.ok) << pair.label;

    for (size_t i = 0; i < candidates.size(); ++i) {
      writeFloatValue(pairFix.src.mutableCTensor(), i, candidates[i]);
    }
    pair.scalar(&pairFix.src.mutableCTensor(), &pairFix.dst.mutableCTensor());

    for (size_t i = 0; i < candidates.size(); ++i) {
      const double decoded = quantizeToSource(pair.src, candidates[i]);
      EXPECT_EQ(readRaw(pairFix.dst.getCTensor(), i),
                expectedFloatPattern(pair.dst, decoded))
          << pair.label << " candidate[" << i << "]=" << candidates[i];
    }
  }
}

/**
 * @brief Verifies all FP->INT scalar kernels implement the documented
 *        truncate-and-saturate policy over the curated candidate lists.
 */
TEST(ScalarOracle, FpToIntMatchesOracle) {
  for (const auto &pair : allPairs()) {
    if (familyOf(pair) != Family::FpToInt) {
      continue;
    }
    SCOPED_TRACE(pair.label);

    const std::vector<double> candidates = interestingValues(pair.src);
    auto pairFix =
        makePair(pair.src, pair.dst, candidates.size(), kDefaultSeed);
    ASSERT_TRUE(pairFix.ok) << pair.label;

    for (size_t i = 0; i < candidates.size(); ++i) {
      writeFloatValue(pairFix.src.mutableCTensor(), i, candidates[i]);
    }
    pair.scalar(&pairFix.src.mutableCTensor(), &pairFix.dst.mutableCTensor());

    tests::casting::withIntType(pair.dst, [&]<typename D>() {
      for (size_t i = 0; i < candidates.size(); ++i) {
        const double decoded = quantizeToSource(pair.src, candidates[i]);
        EXPECT_EQ(readAs<D>(pairFix.dst.getCTensor(), i),
                  referenceSaturate<D>(decoded, pair.src))
            << pair.label << " candidate[" << i << "]=" << candidates[i];
      }
    });
  }
}

/**
 * @brief Verifies all INT->FP scalar kernels, mirroring the
 *        float-narrowing chain for reduced destinations.
 */
TEST(ScalarOracle, IntToFpMatchesOracle) {
  for (const auto &pair : allPairs()) {
    if (familyOf(pair) != Family::IntToFp) {
      continue;
    }
    SCOPED_TRACE(pair.label);

    tests::casting::withIntType(pair.src, [&]<typename S>() {
      const auto candidates = intCandidates<S>();
      auto pairFix =
          makePair(pair.src, pair.dst, candidates.size(), kDefaultSeed);
      ASSERT_TRUE(pairFix.ok) << pair.label;

      for (size_t i = 0; i < candidates.size(); ++i) {
        const uint64_t bits =
            std::is_signed_v<S>
                ? static_cast<uint64_t>(static_cast<int64_t>(candidates[i]))
                : static_cast<uint64_t>(candidates[i]);
        writeRaw(pairFix.src.mutableCTensor(), i, bits);
      }
      pair.scalar(&pairFix.src.mutableCTensor(), &pairFix.dst.mutableCTensor());

      for (size_t i = 0; i < candidates.size(); ++i) {
        // Kernels narrow through float before reduced encodes.
        const double narrowed = static_cast<double>(
            static_cast<float>(static_cast<double>(candidates[i])));
        switch (pair.dst) {
        case DType_::Float32:
          EXPECT_EQ(
              readRaw(pairFix.dst.getCTensor(), i),
              f32BitsOf(static_cast<float>(static_cast<double>(candidates[i]))))
              << pair.label << " candidate[" << i << "]";
          break;
        case DType_::Float64:
          EXPECT_EQ(readRaw(pairFix.dst.getCTensor(), i),
                    f64BitsOf(static_cast<double>(candidates[i])))
              << pair.label << " candidate[" << i << "]";
          break;
        default:
          EXPECT_EQ(readRaw(pairFix.dst.getCTensor(), i),
                    referenceEncode(specOf(pair.dst), narrowed))
              << pair.label << " candidate[" << i << "]";
          break;
        }
      }
    });
  }
}

/**
 * @brief Verifies all INT->INT scalar kernels match the per-pair
 *        policy (clamping for the documented 8-bit-target set, modular
 *        truncation elsewhere).
 */
TEST(ScalarOracle, IntToIntMatchesOracle) {
  for (const auto &pair : allPairs()) {
    if (familyOf(pair) != Family::IntToInt) {
      continue;
    }
    SCOPED_TRACE(pair.label);

    tests::casting::withIntType(pair.src, [&]<typename S>() {
      tests::casting::withIntType(pair.dst, [&]<typename D>() {
        const auto candidates = intCandidates<S>();
        auto pairFix =
            makePair(pair.src, pair.dst, candidates.size(), kDefaultSeed);
        ASSERT_TRUE(pairFix.ok) << pair.label;

        for (size_t i = 0; i < candidates.size(); ++i) {
          const uint64_t bits =
              std::is_signed_v<S>
                  ? static_cast<uint64_t>(static_cast<int64_t>(candidates[i]))
                  : static_cast<uint64_t>(candidates[i]);
          writeRaw(pairFix.src.mutableCTensor(), i, bits);
        }
        pair.scalar(&pairFix.src.mutableCTensor(),
                    &pairFix.dst.mutableCTensor());

        for (size_t i = 0; i < candidates.size(); ++i) {
          EXPECT_EQ(readAs<D>(pairFix.dst.getCTensor(), i),
                    (tests::casting::referenceNarrow<S, D>)(candidates[i]))
              << pair.label << " candidate[" << i << "]";
        }
      });
    });
  }
}

/**
 * @brief Verifies LCG-filled raw patterns (excluding NaN slots) also match
 *        the oracle composition: catches pattern-space blind spots in the
 *        curated lists.
 */
TEST(ScalarOracle, LcgPatternsMatchOracleForFpToFp) {
  for (const auto &pair : allPairs()) {
    if (familyOf(pair) != Family::FpToFp) {
      continue;
    }
    if (pair.src == DType_::Float32 || pair.src == DType_::Float64) {
      continue; // Reduced-source pattern sweep; full-IEEE covered above.
    }
    SCOPED_TRACE(pair.label);

    constexpr size_t kCount = 16;
    auto pairFix = makePair(pair.src, pair.dst, kCount, 0xDEADBEEFU);
    ASSERT_TRUE(pairFix.ok) << pair.label;

    pair.scalar(&pairFix.src.mutableCTensor(), &pairFix.dst.mutableCTensor());

    for (size_t i = 0; i < kCount; ++i) {
      const uint64_t raw = readRaw(pairFix.src.getCTensor(), i);
      if (isNaNPattern(pair.src, raw)) {
        continue; // NaN payload policies are SpecialValueInvariant's job.
      }
      const double decoded = referenceDecode(specOf(pair.src), raw);
      EXPECT_EQ(readRaw(pairFix.dst.getCTensor(), i),
                expectedFloatPattern(pair.dst, decoded))
          << pair.label << " element=" << i << " src=0x" << std::hex << raw;
    }
  }
}

/**
 * @brief Pins the packed lane order end to end through tensor kernels:
 *        low nibble <-> even index, high nibble <-> odd index.
 */
TEST(ScalarOracle, Fp4LaneOrderIsLowNibbleFirst) {
  {
    SCOPED_TRACE("fp4->f32");
    auto pair =
        makePair(DType_::Float4E2M1fn, DType_::Float32, 2, kDefaultSeed);
    ASSERT_TRUE(pair.ok);
    writeRaw(pair.src.mutableCTensor(), 0,
             0xB); // lo nibble: -1.5 (E2M1 sign|1.5)
    writeRaw(pair.src.mutableCTensor(), 1, 0x1); // hi nibble: +0.5
    runScalar(pair);

    EXPECT_FLOAT_EQ(tests::casting::readAsF32(pair.dst.getCTensor(), 0), -1.5F);
    EXPECT_FLOAT_EQ(tests::casting::readAsF32(pair.dst.getCTensor(), 1), 0.5F);
  }
  {
    SCOPED_TRACE("f32->fp4");
    auto pair =
        makePair(DType_::Float32, DType_::Float4E2M1fn, 2, kDefaultSeed);
    ASSERT_TRUE(pair.ok);
    tests::casting::writeFloatValue(pair.src.mutableCTensor(), 0, -1.5);
    tests::casting::writeFloatValue(pair.src.mutableCTensor(), 1, 0.5);
    runScalar(pair);

    EXPECT_EQ(readRaw(pair.dst.getCTensor(), 0), UINT64_C(0xB));
    EXPECT_EQ(readRaw(pair.dst.getCTensor(), 1), UINT64_C(0x1));
  }
}

/**
 * @brief Guard-band fixture: destination prefilled with a poison byte must
 *        be fully overwritten across every logical element, proving the
 *        kernel touches exactly the expected extent. Out-of-allocation
 *        overruns are additionally caught by the ASan presets.
 */
TEST(ScalarOracle, KernelTouchesExactlyTheExpectedElements) {
  auto run = [](DType_ src, DType_ dst, size_t count, double fill) {
    SCOPED_TRACE(testing::Message()
                 << static_cast<int>(src) << "->" << static_cast<int>(dst));
    auto pair = makePair(src, dst, count, kDefaultSeed);
    EXPECT_TRUE(pair.ok);
    if (!pair.ok) {
      return;
    }

    Tensor dstView = pair.dst.mutableCTensor();
    std::memset(dstView.data.data, 0xA5, dstView.size * dstView.item_size);
    for (size_t i = 0; i < count; ++i) {
      writeFloatValue(pair.src.mutableCTensor(), i, fill);
    }
    runScalar(pair);

    uint64_t poison = 0;
    for (size_t b = 0; b < dstView.item_size; ++b) {
      poison = (poison << 8) | UINT64_C(0xA5);
    }
    for (size_t i = 0; i < count; ++i) {
      EXPECT_NE(readRaw(dstView, i), poison) << "element=" << i << " untouched";
    }
  };

  run(DType_::Float4E2M1fn, DType_::Float32, 8, 1.0); // Packed edge.
  run(DType_::Float32, DType_::Float16, 9, 1.0);      // Unpacked edge.
}
