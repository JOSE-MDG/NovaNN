/**
 * @file ISAEquivalence_test.cpp
 * @brief SIMD-variant equivalence and dispatch-ladder tests.
 *
 * get_simd_capabilities() is a call_once-cached singleton, so synthetic
 * capabilities cannot be injected into the live dispatch. Equivalence
 * is therefore established two ways:
 *
 * @li Wrapper-vs-scalar: the live dispatch wrapper must agree bitwise with
 *     the portable scalar kernel for every one of the 210 pairs, exercising
 *     whichever SIMD variant the host ladder selects, across sizes that
 *     straddle every vector step (tail loops included).
 * @li Direct variant invocation: for a curated set of pairs whose SIMD
 *     symbols have stable names, each variant is invoked directly (gated on
 *     real host support via isaSupported()) and compared against both the
 *     scalar kernel and the wrapper.
 */

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/headeronly/cast.h>
#include <ncore/headeronly/macros.h>

#ifdef _GNUC_CLANG_
#include <ncore/native/cpu/dtype/casting.h>
#endif

#include "utils/Oracle.hpp"

namespace {

using tests::casting::allPairs;
using tests::casting::expectEqualSkippingSrcNan;
using tests::casting::findPairFor;
using tests::casting::hostCaps;
using tests::casting::isaSupported;
using tests::casting::kDefaultSeed;
using tests::casting::makePair;

struct IseParam {
  DType_ src;
  DType_ dst;
  std::string label;
};

std::vector<IseParam> buildParams() {
  std::vector<IseParam> params;
  params.reserve(allPairs().size());
  for (const auto &p : allPairs()) {
    params.push_back(IseParam{.src = p.src, .dst = p.dst, .label = p.label});
  }
  return params;
}

std::string sanitize(std::string label) {
  for (char &c : label) {
    if (c == '-' || c == '>') {
      c = '_';
    }
  }
  return label;
}

/// Runs scalar and wrapper on identical LCG-filled inputs and requires
/// equal destinations over @p count logical elements (source-NaN elements
/// are exempt from bitwise equality; see expectEqualSkippingSrcNan).
void expectWrapperMatchesScalar(DType_ src, DType_ dst, size_t count,
                                uint32_t seed, const char *label) {
  const tests::casting::PairInfo *pair = findPairFor(src, dst);
  ASSERT_NE(pair, nullptr) << label;

  auto scalarPair = makePair(src, dst, count, seed);
  auto wrapperPair = makePair(src, dst, count, seed);
  ASSERT_TRUE(scalarPair.ok) << label;
  ASSERT_TRUE(wrapperPair.ok) << label;

  pair->scalar(&scalarPair.src.mutableCTensor(),
               &scalarPair.dst.mutableCTensor());
  pair->wrapper(&wrapperPair.src.mutableCTensor(),
                &wrapperPair.dst.mutableCTensor());

  expectEqualSkippingSrcNan(scalarPair.src.getCTensor(),
                            scalarPair.dst.getCTensor(),
                            wrapperPair.dst.getCTensor(), count, label);
}

/// Parameterized fixture over the full 210-pair registry.
class ISAEquivalence : public ::testing::TestWithParam<IseParam> {};

} // namespace

/**
 * @brief Verifies the live dispatch path agrees with the portable scalar
 *        kernel for every supported pair over deterministic inputs
 *        (bitwise except source-NaN elements, whose payload handling is
 *        not part of the contract).
 */
TEST_P(ISAEquivalence, DispatchMatchesScalarOnDeterministicInputs) {
  const IseParam &p = GetParam();
  expectWrapperMatchesScalar(p.src, p.dst, 33, kDefaultSeed, p.label.c_str());
}

/**
 * @brief Verifies tail handling: sizes straddling every vector width
 *        (128/256/512-bit lanes) produce identical results, one pair per
 *        X-macro family.
 */
TEST(ISAEquivalence, TailRemainderHandledByAllVariants) {
  const std::array<std::array<DType_, 2>, 7> kRepresentatives{{
      {DType_::Float16, DType_::Float32},      // Float -> Float
      {DType_::Float32, DType_::Signed8},      // Float -> Int
      {DType_::Signed8, DType_::Float32},      // Int -> Float
      {DType_::Signed8, DType_::Signed32},     // Signed -> Signed
      {DType_::UnSigned8, DType_::UnSigned32}, // Unsigned -> Unsigned
      {DType_::Signed16, DType_::UnSigned8},   // Signed -> Unsigned
      {DType_::UnSigned64, DType_::Signed8},   // Unsigned -> Signed
  }};

  for (const auto &rep : kRepresentatives) {
    SCOPED_TRACE(testing::Message() << static_cast<int>(rep[0]) << "->"
                                    << static_cast<int>(rep[1]));
    for (const size_t count :
         {size_t{1}, size_t{3}, size_t{7}, size_t{15}, size_t{16}, size_t{17},
          size_t{31}, size_t{33}}) {
      expectWrapperMatchesScalar(rep[0], rep[1], count, kDefaultSeed, "tail");
    }
  }
}

/**
 * @brief Verifies pairs whose tables hold only the portable fallback:
 *        the wrapper must behave exactly like the scalar kernel. All FP4-
 *        and FP8-involving pairs are single-entry today.
 */
TEST(ISAEquivalence, SingleEntryPairsDelegateToScalar) {
  for (const auto &p : allPairs()) {
    const bool reduced =
        p.src == DType_::Float4E2M1fn || p.dst == DType_::Float4E2M1fn ||
        p.src == DType_::Float8E4M3fn || p.dst == DType_::Float8E4M3fn ||
        p.src == DType_::Float8E5M2 || p.dst == DType_::Float8E5M2;
    if (!reduced) {
      continue;
    }
    expectWrapperMatchesScalar(p.src, p.dst, 9, kDefaultSeed, p.label.c_str());
  }
}

#if defined(_GNUC_CLANG_)

namespace {

/// One directly-invocable SIMD variant plus its cast_tables.h requirement.
///
/// The owning registry pair is identified by DTYPE PAIR, not by wrapper
/// pointer: cast.h's dispatchers are static inline (internal linkage), so
/// every translation unit holds its own copy and their addresses are only
/// comparable within one TU, while the allPairs() singleton may have
/// been materialized in any of them. The scalar reference and dispatch
/// wrapper under comparison are taken from the registry entry itself, so a
/// typo in this table cannot pit kernels of different pairs against each
/// other.
struct LadderEntry {
  const char *label;        ///< Diagnostic tag for failure messages.
  const char *requirements; ///< ISA requirement string per cast_tables.h.
  DType_ src;               ///< Source dtype of the owning registry pair.
  DType_ dst;               ///< Destination dtype of the owning pair.
  CastFn variant;           ///< Directly-invocable SIMD variant.
};

/// Curated ladders with stable symbol names, spanning every ISA family.
std::vector<LadderEntry> buildLadderTable() {
  return {
      {.label = "fp16->f32 avx512",
       .requirements = "AVX512F",
       .src = DType_::Float16,
       .dst = DType_::Float32,
       .variant = &tfp16_to_f32_avx512},
      {.label = "fp16->f32 avx_fp16c",
       .requirements = "AVX/AVX2,F16C",
       .src = DType_::Float16,
       .dst = DType_::Float32,
       .variant = &tfp16_to_f32_avx_avx2_fp16c},
      {.label = "fp16->f64 avx512",
       .requirements = "AVX512F,AVX512FP16",
       .src = DType_::Float16,
       .dst = DType_::Float64,
       .variant = &tfp16_to_f64_avx512},
      {.label = "fp16->f64 f16c",
       .requirements = "F16C",
       .src = DType_::Float16,
       .dst = DType_::Float64,
       .variant = &tfp16_to_f64_avx_avx2_fp16c},
      {.label = "fp16->bf16 avx512bf16",
       .requirements = "AVX512F,AVX512BF16",
       .src = DType_::Float16,
       .dst = DType_::BFloat16,
       .variant = &tfp16_to_bf16_avx512bf16},
      {.label = "f32->fp16 avx512fp16",
       .requirements = "AVX512F,AVX512FP16",
       .src = DType_::Float32,
       .dst = DType_::Float16,
       .variant = &tf32_to_fp16_avx512fp16},
      {.label = "f32->fp16 f16c",
       .requirements = "F16C",
       .src = DType_::Float32,
       .dst = DType_::Float16,
       .variant = &tf32_to_fp16_avx_avx2_f16c},
      {.label = "f32->f64 avx512",
       .requirements = "AVX512F",
       .src = DType_::Float32,
       .dst = DType_::Float64,
       .variant = &tf32_to_f64_avx512},
      {.label = "f32->f64 avx",
       .requirements = "AVX/AVX2",
       .src = DType_::Float32,
       .dst = DType_::Float64,
       .variant = &tf32_to_f64_avx_avx2},
      {.label = "f32->f64 sse4.2",
       .requirements = "SSE4.2",
       .src = DType_::Float32,
       .dst = DType_::Float64,
       .variant = &tf32_to_f64_sse4_2},
      {.label = "f32->bf16 avx512bf16",
       .requirements = "AVX512F,AVX512BF16,AVX512BW,AVX512VL",
       .src = DType_::Float32,
       .dst = DType_::BFloat16,
       .variant = &tf32_to_bf16_avx512bf16},
      {.label = "bf16->f32 avx512bf16",
       .requirements = "AVX512BF16,AVX512F",
       .src = DType_::BFloat16,
       .dst = DType_::Float32,
       .variant = &tbf16_to_f32_avx512bf16},
      {.label = "bf16->fp16 avx512",
       .requirements = "AVX512F,AVX512BF16,AVX512FP16",
       .src = DType_::BFloat16,
       .dst = DType_::Float16,
       .variant = &tbf16_to_fp16_avx512bf16_fp16},
      {.label = "bf16->f64 avx512bf16",
       .requirements = "AVX512F,AVX512BF16,AVX512VL",
       .src = DType_::BFloat16,
       .dst = DType_::Float64,
       .variant = &tbf16_to_f64_avx512bf16},
      {.label = "f64->fp16 avx512fp16",
       .requirements = "AVX512F,AVX512FP16",
       .src = DType_::Float64,
       .dst = DType_::Float16,
       .variant = &tf64_to_fp16_avx512fp16},
      {.label = "f64->f32 avx512",
       .requirements = "AVX512F",
       .src = DType_::Float64,
       .dst = DType_::Float32,
       .variant = &tf64_to_f32_avx512},
      {.label = "f64->f32 avx",
       .requirements = "AVX/AVX2",
       .src = DType_::Float64,
       .dst = DType_::Float32,
       .variant = &tf64_to_f32_avx_avx2},
      {.label = "f64->f32 sse4.2",
       .requirements = "SSE4.2",
       .src = DType_::Float64,
       .dst = DType_::Float32,
       .variant = &tf64_to_f32_sse4_2},
      {.label = "f64->bf16 avx512bf16",
       .requirements = "AVX512F,AVX512BF16,AVX512VL",
       .src = DType_::Float64,
       .dst = DType_::BFloat16,
       .variant = &tf64_to_bf16_avx512bf16},
      {.label = "f32->s32 avx512",
       .requirements = "AVX512F",
       .src = DType_::Float32,
       .dst = DType_::Signed32,
       .variant = &tf32_to_s32_avx512},
      {.label = "f32->s32 avx2",
       .requirements = "AVX2",
       .src = DType_::Float32,
       .dst = DType_::Signed32,
       .variant = &tf32_to_s32_avx2},
      {.label = "f32->s32 sse4.2",
       .requirements = "SSE4.2",
       .src = DType_::Float32,
       .dst = DType_::Signed32,
       .variant = &tf32_to_s32_sse4_2},
      {.label = "f32->s8 avx512",
       .requirements = "AVX512F,AVX512BW",
       .src = DType_::Float32,
       .dst = DType_::Signed8,
       .variant = &tf32_to_s8_avx512},
      {.label = "f32->s8 avx2",
       .requirements = "AVX2",
       .src = DType_::Float32,
       .dst = DType_::Signed8,
       .variant = &tf32_to_s8_avx2},
      {.label = "s8->f32 avx512",
       .requirements = "AVX512F",
       .src = DType_::Signed8,
       .dst = DType_::Float32,
       .variant = &ts8_to_f32_avx512},
      {.label = "s8->f32 avx2",
       .requirements = "AVX2",
       .src = DType_::Signed8,
       .dst = DType_::Float32,
       .variant = &ts8_to_f32_avx2},
      {.label = "s8->s32 avx512",
       .requirements = "AVX512F",
       .src = DType_::Signed8,
       .dst = DType_::Signed32,
       .variant = &ts8_to_s32_avx512},
      {.label = "s8->s32 avx2",
       .requirements = "AVX2",
       .src = DType_::Signed8,
       .dst = DType_::Signed32,
       .variant = &ts8_to_s32_avx2},
      {.label = "s8->s32 sse4.2",
       .requirements = "SSE4.2",
       .src = DType_::Signed8,
       .dst = DType_::Signed32,
       .variant = &ts8_to_s32_sse4_2},
      {.label = "u8->s8 avx512",
       .requirements = "AVX512F,AVX512BW",
       .src = DType_::UnSigned8,
       .dst = DType_::Signed8,
       .variant = &tu8_to_s8_avx512},
      {.label = "u8->s8 avx2",
       .requirements = "AVX2",
       .src = DType_::UnSigned8,
       .dst = DType_::Signed8,
       .variant = &tu8_to_s8_avx2},
      {.label = "u8->s8 sse4.2",
       .requirements = "SSE4.2",
       .src = DType_::UnSigned8,
       .dst = DType_::Signed8,
       .variant = &tu8_to_s8_sse4_2},
  };
}

} // namespace

/**
 * @brief Invokes each curated SIMD variant directly (when the host supports
 *        its requirements) and requires bitwise agreement with both the
 *        scalar kernel and the live dispatch wrapper.
 */
TEST(ISAEquivalence, DirectVariantLadderMatchesScalarAndWrapper) {
  constexpr size_t kCount = 33; // Straddles vector steps.

  size_t executed = 0;
  size_t skipped = 0;
  for (const LadderEntry &entry : buildLadderTable()) {
    SCOPED_TRACE(entry.label);
    if (!isaSupported(entry.requirements, hostCaps())) {
      ++skipped;
      continue;
    }
    ++executed;

    // Identify the owning pair by dtypes: wrapper addresses are
    // TU-local (static inline dispatchers), so pointer comparison against
    // the registry singleton cannot work across translation units. The
    // scalar/wrapper references come from the registry entry itself.
    const tests::casting::PairInfo *owner = findPairFor(entry.src, entry.dst);
    ASSERT_NE(owner, nullptr) << entry.label;

    auto variantPair = makePair(owner->src, owner->dst, kCount, kDefaultSeed);
    auto scalarPair = makePair(owner->src, owner->dst, kCount, kDefaultSeed);
    auto wrapperPair = makePair(owner->src, owner->dst, kCount, kDefaultSeed);
    ASSERT_TRUE(variantPair.ok);
    ASSERT_TRUE(scalarPair.ok);
    ASSERT_TRUE(wrapperPair.ok);

    entry.variant(&variantPair.src.mutableCTensor(),
                  &variantPair.dst.mutableCTensor());
    owner->scalar(&scalarPair.src.mutableCTensor(),
                  &scalarPair.dst.mutableCTensor());
    owner->wrapper(&wrapperPair.src.mutableCTensor(),
                   &wrapperPair.dst.mutableCTensor());

    expectEqualSkippingSrcNan(variantPair.src.getCTensor(),
                              variantPair.dst.getCTensor(),
                              scalarPair.dst.getCTensor(), kCount, entry.label);
    expectEqualSkippingSrcNan(
        variantPair.src.getCTensor(), variantPair.dst.getCTensor(),
        wrapperPair.dst.getCTensor(), kCount, entry.label);
  }

  if (executed == 0) {
    GTEST_SKIP() << "no curated SIMD variant supported on this host ("
                 << skipped << " skipped)";
  }
  SUCCEED() << executed << " variants executed, " << skipped
            << " skipped (ISA unavailable on this host)";
}

#else

/**
 * @brief Non-GCC/Clang builds collapse every table to the scalar fallback;
 *        the degenerate contract is wrapper == scalar everywhere.
 */
TEST(ISAEquivalence, NonGnuClangDegeneratesToScalar) {
  for (const auto &p : allPairs()) {
    expectWrapperMatchesScalar(p.src, p.dst, 4, kDefaultSeed, p.label.c_str());
  }
}

#endif

INSTANTIATE_TEST_SUITE_P(AllPairs, ISAEquivalence,
                         ::testing::ValuesIn(buildParams()),
                         [](const ::testing::TestParamInfo<IseParam> &info) {
                           return sanitize(info.param.label);
                         });
