/**
 * @file FloatingPointClassification.hpp
 * @brief Independent pattern classifiers and reference decoders for the
 *        reduced-precision dtype suites.
 *
 * This header is the dtypes-side oracle. Like
 * @c ncore/tests/dtypeCasting/utils/Oracle.hpp it is deliberately implemented
 * without calling into the conversion code under test: classify() inspects raw
 * bit fields against a declarative FormatModel, and the referenceDecode*
 * functions compute exact dyadic values in @c double from the format
 * definitions. A wrong constant on either side fails a test instead of
 * cancelling out.
 *
 * The decoders here are intentionally separate from the casting-side oracle:
 * the two test trees must be able to fail independently, so a shared-bug
 * blind spot cannot hide across suites.
 */

#pragma once

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace tests::fpc {

/* ==========================================================================
 * Bit reinterpretation helpers (memcpy-based, strict-aliasing safe)
 * ========================================================================== */

/**
 * @brief Reinterprets a float as its 32-bit pattern.
 *
 * @param[in] v Input value.
 * @return Raw IEEE 754 binary32 bits.
 */
inline uint32_t f32BitsOf(float v) {
  uint32_t w = 0;
  std::memcpy(&w, &v, sizeof(w));
  return w;
}

/**
 * @brief Reinterprets a 32-bit pattern as a float.
 *
 * @param[in] w Raw IEEE 754 binary32 bits.
 * @return Corresponding float value.
 */
inline float f32FromBits(uint32_t w) {
  float v = 0.0F;
  std::memcpy(&v, &w, sizeof(v));
  return v;
}

/**
 * @brief True when raw binary32 bits encode a NaN.
 *
 * Bit-level on purpose: the input is a storage pattern, not a float
 * object, so classification must not round-trip through the FP type.
 *
 * @param[in] w Raw IEEE 754 binary32 bits.
 * @return NaN classification.
 */
inline bool isNaNF32Bits(uint32_t w) {
  return (w & 0x7F800000U) == 0x7F800000U && (w & 0x007FFFFFU) != 0U;
}

/**
 * @brief True when raw binary32 bits encode infinity.
 *
 * @param[in] w Raw IEEE 754 binary32 bits.
 * @return Infinity classification.
 */
inline bool isInfF32Bits(uint32_t w) {
  return (w & 0x7FFFFFFFU) == 0x7F800000U;
}

/**
 * @brief Extracts the sign bit from raw binary32 bits.
 *
 * @param[in] w Raw IEEE 754 binary32 bits.
 * @return True when negative (sign bit set).
 */
inline bool signBitF32Bits(uint32_t w) { return (w >> 31) != 0U; }

/* ==========================================================================
 * Value classes and format models
 * ========================================================================== */

/**
 * @enum FpClass
 * @brief Value class of one raw storage pattern.
 */
enum class FpClass {
  kZero,      ///< Zero (positive or negative).
  kSubnormal, ///< Subnormal (nonzero, exponent field zero).
  kNormal,    ///< Normal finite value.
  kInf,       ///< Infinity.
  kNaN        ///< Not a Number.
};

/**
 * @struct FormatModel
 * @brief Declarative field layout of one reduced-precision format.
 */
struct FormatModel {
  int expBits; ///< Width of the biased exponent field.
  int manBits; ///< Width of the explicit mantissa field.
  bool hasInf; ///< Whether an infinity encoding exists.
  bool hasNan; ///< Whether a NaN encoding exists.
};

inline constexpr FormatModel kFp16Model{
    .expBits = 5, .manBits = 10, .hasInf = true, .hasNan = true};
inline constexpr FormatModel kBf16Model{
    .expBits = 8, .manBits = 7, .hasInf = true, .hasNan = true};
inline constexpr FormatModel kFp8E4M3Model{
    .expBits = 4, .manBits = 3, .hasInf = false, .hasNan = true};
inline constexpr FormatModel kFp8E5M2Model{
    .expBits = 5, .manBits = 2, .hasInf = true, .hasNan = true};

/**
 * @brief Classifies a raw storage pattern against a format model.
 *
 * Rules (IEEE-style unless the format is finite-only):
 * @li Exponent field zero with zero mantissa -> kZero.
 * @li Exponent field zero otherwise -> kSubnormal.
 * @li Maximum exponent field:
 *     - IEEE-style (has_inf): mantissa zero -> kInf, otherwise kNaN.
 *     - Finite-only-with-NaN (E4M3FN): only the all-ones mantissa slot is
 *       kNaN; every other pattern in that binade is kNormal.
 *     - Fully finite (FP4): the top binade is normal like any other.
 * @li Everything else -> kNormal.
 *
 * @param[in] pattern Raw storage pattern (zero-extended to 32 bits).
 * @param[in] m       Format model describing the field layout.
 * @return Value class of @p pattern.
 */
inline FpClass classify(uint32_t pattern, FormatModel m) {
  const uint32_t maxExpField = (1U << m.expBits) - 1U;
  const uint32_t expField = (pattern >> m.manBits) & ((1U << m.expBits) - 1U);
  const uint32_t man = pattern & ((1U << m.manBits) - 1U);

  if (expField == 0) {
    return man == 0 ? FpClass::kZero : FpClass::kSubnormal;
  }
  if (expField == maxExpField) {
    if (!m.hasInf && !m.hasNan) {
      return FpClass::kNormal; // FP4: no reserved binade.
    }
    if (m.hasInf) {
      return man == 0 ? FpClass::kInf : FpClass::kNaN;
    }
    // E4M3FN-style: single NaN slot at the all-ones mantissa.
    const uint32_t manMask = (1U << m.manBits) - 1U;
    return man == manMask ? FpClass::kNaN : FpClass::kNormal;
  }
  return FpClass::kNormal;
}

/**
 * @brief Extracts the sign bit of a raw storage pattern.
 *
 * @param[in] pattern Raw storage pattern.
 * @param[in] m       Format model describing the field layout.
 * @return True when the sign bit is set (negative).
 */
inline bool signBit(uint32_t pattern, FormatModel m) {
  const uint32_t total = static_cast<uint32_t>(1 + m.expBits + m.manBits);
  return ((pattern >> (total - 1)) & 1U) != 0U;
}

/* ==========================================================================
 * Pattern predicates for the FP8 variants
 * ========================================================================== */

/**
 * @brief True when an E4M3FN byte encodes NaN (all exponent+mantissa bits
 *        set). Declared independently of Float8_e4m3fn::isnan().
 *
 * @param[in] bits Raw E4M3FN byte.
 * @return NaN classification.
 */
inline bool isNaNPatternE4M3(uint8_t bits) { return (bits & 0x7FU) == 0x7FU; }

/**
 * @brief True when an E5M2 byte encodes NaN (exponent all ones, nonzero
 *        mantissa). Declared independently of Float8_e5m2::isnan().
 *
 * @param[in] bits Raw E5M2 byte.
 * @return NaN classification.
 */
inline bool isNaNPatternE5M2(uint8_t bits) { return (bits & 0x7FU) > 0x7CU; }

/**
 * @brief True when an E5M2 byte encodes infinity (exponent all ones,
 *        mantissa zero). Declared independently of Float8_e5m2::isinf().
 *
 * @param[in] bits Raw E5M2 byte.
 * @return Infinity classification.
 */
inline bool isInfPatternE5M2(uint8_t bits) { return (bits & 0x7FU) == 0x7CU; }

/* ==========================================================================
 * Reference decoders: exact dyadic arithmetic in double
 * ========================================================================== */

/**
 * @brief Exact binary16 value of a raw half-precision pattern.
 *
 * @param[in] bits Raw 16-bit IEEE 754 half-precision pattern.
 * @return Exact numeric value as double.
 */
inline double referenceDecodeFp16(uint16_t bits) {
  const bool neg = (bits >> 15) != 0;
  const uint32_t expField = (bits >> 10) & 0x1FU;
  const uint32_t man = bits & 0x03FFU;

  double out = 0.0;
  if (expField == 0) {
    out = std::ldexp(static_cast<double>(man), -24);
  } else if (expField == 0x1FU) {
    out = man == 0 ? HUGE_VAL : std::numeric_limits<double>::quiet_NaN();
  } else {
    out = std::ldexp(1.0 + (static_cast<double>(man) / 1024.0),
                     static_cast<int>(expField) - 15);
  }
  return neg ? -out : out;
}

/**
 * @brief Exact bfloat16 value of a raw pattern (top half of an f32 word).
 *
 * @param[in] bits Raw 16-bit bfloat16 pattern.
 * @return Exact numeric value as double.
 */
inline double referenceDecodeBf16(uint16_t bits) {
  const bool neg = (bits >> 15) != 0;
  const uint32_t expField = (bits >> 7) & 0xFFU;
  const uint32_t man = bits & 0x007FU;

  double out = 0.0;
  if (expField == 0) {
    out = std::ldexp(static_cast<double>(man), -126 - 7);
  } else if (expField == 0xFFU) {
    out = man == 0 ? HUGE_VAL : std::numeric_limits<double>::quiet_NaN();
  } else {
    out = std::ldexp(1.0 + (static_cast<double>(man) / 128.0),
                     static_cast<int>(expField) - 127);
  }
  return neg ? -out : out;
}

/**
 * @brief Exact E4M3FN value of a raw byte (finite-only format, NaN at
 *        0x7F|sign).
 *
 * @param[in] bits Raw E4M3FN byte.
 * @return Exact numeric value as double.
 */
inline double referenceDecodeFp8E4M3(uint8_t bits) {
  const bool neg = (bits >> 7) != 0;
  const uint32_t expField = (bits >> 3) & 0x0FU;
  const uint32_t man = bits & 0x07U;

  double out = 0.0;
  if (expField == 0) {
    out = std::ldexp(static_cast<double>(man), -9);
  } else if (expField == 0x0FU && man == 0x07U) {
    out = std::numeric_limits<double>::quiet_NaN();
  } else {
    out = std::ldexp(1.0 + (static_cast<double>(man) / 8.0),
                     static_cast<int>(expField) - 7);
  }
  return neg ? -out : out;
}

/**
 * @brief Exact E5M2 value of a raw byte (IEEE-style specials).
 *
 * @param[in] bits Raw E5M2 byte.
 * @return Exact numeric value as double.
 */
inline double referenceDecodeFp8E5M2(uint8_t bits) {
  const bool neg = (bits >> 7) != 0;
  const uint32_t expField = (bits >> 2) & 0x1FU;
  const uint32_t man = bits & 0x03U;

  double out = 0.0;
  if (expField == 0) {
    out = std::ldexp(static_cast<double>(man), -16);
  } else if (expField == 0x1FU) {
    out = man == 0 ? HUGE_VAL : std::numeric_limits<double>::quiet_NaN();
  } else {
    out = std::ldexp(1.0 + (static_cast<double>(man) / 4.0),
                     static_cast<int>(expField) - 15);
  }
  return neg ? -out : out;
}

/**
 * @brief The eight E2M1FN magnitudes, declared independently of the
 *        decode path under test so a wrong constant fails on either side.
 *
 * Indexed by the 3-bit magnitude field (2 exponent bits << 1 | 1 mantissa
 * bit); the sign bit is handled separately.
 */
inline constexpr std::array<float, 8> kFp4Magnitudes{0.0F, 0.5F, 1.0F, 1.5F,
                                                     2.0F, 3.0F, 4.0F, 6.0F};

/**
 * @brief Exact E2M1FN lane value of a raw nibble.
 *
 * All 16 nibble patterns are finite by definition (no inf/NaN encoding);
 * only the low nibble is read.
 *
 * @param[in] nibble Raw nibble (high bits ignored).
 * @return Exact numeric value as float.
 */
inline float referenceDecodeFp4Nibble(uint8_t nibble) {
  const uint8_t sign = static_cast<uint8_t>((nibble >> 3) & 0x1U);
  const uint8_t mag = nibble & 0x7U;
  const float magnitude = kFp4Magnitudes[mag];
  return sign != 0 ? -magnitude : magnitude;
}

} // namespace tests::fpc
