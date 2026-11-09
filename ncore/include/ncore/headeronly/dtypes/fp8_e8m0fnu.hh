/**
 * @file fp8_e8m0fnu.hh
 * @brief 8-bit scale-factor floating-point (FP8 E8M0) data type implementation.
 *
 * @details
 * Defines the @ref Float8_e8m0fnu type from the OCP Microscaling (MX)
 * specification, Section 5.4.1. The layout is eight exponent bits with no
 * sign, no mantissa, and bias 127:
 *  @li @c 0x00 to @c 0xFE encode @f$2^{k-127}@f$ for the stored exponent
 *      @f$k@f$, so the representable range is @f$2^{-127} .. 2^{127}@f$.
 *  @li @c 0xFF encodes the single NaN pattern; it is also what round-to-
 *      nearest-even produces when an input past the largest finite exponent
 *      rounds up.
 * There is no infinity and no zero: the smallest magnitude @c 0x00 is
 * @f$2^{-127}@f$, not zero.
 *
 * The type is a scale factor carrier, not an arithmetic value. OCP MX
 * formats multiply each block element by the shared E8M0 scale, and NVIDIA
 * documents the matching CUDA type under "scale factor data type". The only
 * defined conversions are float32 to and from E8M0; no arithmetic operators
 * are provided.
 *
 * Two rounding modes are implemented for the float32 to E8M0 direction:
 *  @li Round-to-nearest-even applied to the exponent, the OCP suggested
 *      semantics referenced by the checked-in conversion of PyTorch. The
 *      guard, round and sticky bits come from the float32 mantissa and the
 *      least significant bit of the rounding position is the implied mantissa
 *      bit (0 for float32 subnormals, 1 for normals), so exact ties on
 *      normals round up. NaN and +/-inf inputs map to @c 0xFF.
 *  @li Exponent round-up: the cuBLASLt "1D block quantization" recipe derives
 *      UE8M0 scale factors as @f$S = \text{amax}/\text{Amax(DType)}@f$ with
 *      the exponent rounded up and clamped to at most 254 (explicitly noted
 *      by NVIDIA as different from the OCP suggested rounding scheme). Block
 *      scale generators use it; the plain cast keeps nearest-even.
 *
 * References:
 * @li OCP Microscaling (MX) Specification v1.0, Section 5.4.1:
 *     https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
 * @li "Microscaling Data Formats for Deep Learning": https://arxiv.org/abs/2310.10537
 * @li NVIDIA CUDA Math API, @c __nv_fp8_e8m0 scale factor struct, default
 *     rounding @c cudaRoundPosInf:
 *     https://docs.nvidia.com/cuda/cuda-math-api/
 * @li cuBLASLt block scaling: @c CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0,
 *     32-element blocks, scale storage @c CUDA_R_8F_UE8M0:
 *     https://docs.nvidia.com/cuda/cublas/
 *
 * @see half.hh  Shared fp32_from_bits/fp32_to_bits bit-cast helpers.
 */

#pragma once

#include <cstdint>
#include <ostream>

#include <config.h>
#include <ncore/headeronly/macros.h>

#include "half.hh"

#ifdef _GNUC_CLANG_
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#endif

namespace ncore::dtypes {

/**
 * @struct Float8_e8m0fnu
 * @brief Representation of the 8-bit E8M0 scale factor from the OCP MX spec.
 *
 * @details
 * The storage is one unsigned byte: eight exponent bits with bias 127, no
 * sign and no mantissa. The format has no infinity and no zero; @c 0xFF is
 * the only special pattern (NaN). A float32 zero converts to @c 0x00, the
 * lowest scale @f$2^{-127}@f$. Because the value is an exponent carrier used
 * multiplicatively by block-quantized GEMM kernels, no arithmetic operators
 * are defined; rely on the implicit conversion to float instead.
 */
struct alignas(1) Float8_e8m0fnu {
  uint8_t x; ///< The 8-bit exponent representation of the scale factor.

  /**
   * @struct from_bits_t
   * @brief Tag type used to construct a @ref Float8_e8m0fnu directly from its
   * raw 8-bit representation.
   */
  struct from_bits_t {};

  /// Returns a default-constructed @ref from_bits_t.
  NCORE_HOST_DEVICE static constexpr from_bits_t from_bits() { return {}; }

  /**
   * @brief Default constructor. Left uninitialized for performance.
   */
  Float8_e8m0fnu() = default;

  /**
   * @brief Constructs a @ref Float8_e8m0fnu directly from a raw 8-bit integer
   * representation.
   * @param[in] bits The raw 8-bit exponent representation.
   * @param[in] unused Tag parameter to disambiguate from float conversions.
   */
  constexpr NCORE_HOST_DEVICE Float8_e8m0fnu(uint8_t bits,
                                             from_bits_t /*unused*/) noexcept
      : x(bits) {}

  /**
   * @brief Implicit constructor from a single-precision float, rounding the
   * exponent to nearest even.
   * @param[in] value The float value to convert.
   */
  inline NCORE_HOST_DEVICE Float8_e8m0fnu(float value);

  /**
   * @brief Implicit conversion operator to single-precision float.
   * @return The float value of the scale, or NaN for the @c 0xFF pattern.
   */
  inline NCORE_HOST_DEVICE operator float() const;

  /**
   * @brief Checks whether this value encodes NaN (pattern @c 0xFF).
   */
  [[nodiscard]] inline NCORE_HOST_DEVICE bool isnan() const;
};

/**
 * @brief Stream output operator for @ref Float8_e8m0fnu.
 *
 * Promotes to float before writing to the stream; the @c 0xFF pattern prints
 * as nan. Repr formatting that renders the @f$2^k@f$ form lives in the repr
 * pipeline.
 *
 * @param[in,out] out The output stream.
 * @param[in]     value The @ref Float8_e8m0fnu value to write.
 * @return Reference to the output stream.
 */
inline std::ostream &operator<<(std::ostream &out,
                                const Float8_e8m0fnu &value) {
  out << static_cast<float>(value);
  return out;
}

/**
 * @namespace ncore::dtypes::detail
 * @brief Internal low-level bit-manipulation and conversion helpers.
 */
namespace detail {

/**
 * @brief Converts a 32-bit IEEE single-precision float to the 8-bit E8M0
 * scale factor, with the exponent rounded to nearest even.
 *
 * @details
 * Integer-only bit manipulation; no floating-point traps can fire. The sign
 * of @p f is ignored (the scale is a magnitude). NaN and both infinities map
 * to @c 0xFF. A float32 that rounds past the largest finite exponent lands
 * on @c 0xFF as required by the OCP overflow rule for E8M0.
 *
 * @param[in] f The single-precision float value to convert.
 * @return The 8-bit E8M0 scale factor.
 *
 * @see fp8e8m0fnu_roundup_from_fp32_value()  Exponent round-up variant.
 */
NCORE_HOST_DEVICE inline uint8_t fp8e8m0fnu_from_fp32_value(float f) {
  const uint32_t fBits = ncore::dtypes::detail::fp32_to_bits(f);

  /* Extract the eight biased exponent bits. */
  const uint32_t exponent = (fBits >> 23) & UINT32_C(0x000000FF);

  /* f32 NaN and +/-inf map to the single e8m0 NaN pattern. */
  if (exponent == UINT32_C(0xFF)) {
    return UINT8_C(0xFF);
  }

  /*
   * Round-to-nearest-even applied to the exponent, with guard, round and
   * sticky bits taken from the float32 mantissa:
   *  @li g = mantissa bit 22 (the 0.5 position toward the upper grid point)
   *  @li r = mantissa bit 21
   *  @li s = any mantissa bit 20 down to 0
   * The least significant bit of the rounding position is the implied
   * mantissa bit: 0 when the float32 is subnormal, 1 when it is normal, so an
   * exact tie on normal inputs rounds up.
   */
  const uint8_t g = (fBits & UINT32_C(0x00400000)) > 0;
  const uint8_t r = (fBits & UINT32_C(0x00200000)) > 0;
  const uint8_t s = (fBits & UINT32_C(0x001FFFFF)) > 0;
  const uint8_t lsb = exponent > 0;

  const bool roundUp = g != 0 && ((r != 0) || (s != 0) || (lsb != 0));

  /*
   * Only 0x00 to 0xFE reach this point, so the increment stays inside the
   * byte. Whether the result lands on 0xFF is the OCP answer for overflow:
   * the pattern is NaN, and the cast is described exactly like that.
   */
  const uint8_t bits =
      static_cast<uint8_t>(exponent) + (roundUp ? UINT8_C(1) : UINT8_C(0));
  return bits;
}

/**
 * @brief Converts a 32-bit IEEE single-precision float to the 8-bit E8M0
 * scale factor with the exponent rounded up toward +inf in the log2 domain.
 *
 * @details
 * This is the derivation prescribed by the cuBLASLt "1D block quantization"
 * recipe for UE8M0 scale factors: keep the exponent when the float32 is an
 * exact power of two, add one otherwise, and clamp to the largest valid
 * pattern 0xFE so the result can never be NaN. NVIDIA documents the clamp
 * explicitly (the scheme "differs from the OCP MXFP suggested rounding
 * scheme"). Zero input yields @c 0x00, the lowest representable scale.
 *
 * @param[in] f The single-precision float value to convert.
 * @return The 8-bit E8M0 scale factor, at most @c 0xFE.
 */
NCORE_HOST_DEVICE inline uint8_t fp8e8m0fnu_roundup_from_fp32_value(float f) {
  const uint32_t fBits = ncore::dtypes::detail::fp32_to_bits(f);
  const uint32_t exponent = (fBits >> 23) & UINT32_C(0x000000FF);

  if (exponent == UINT32_C(0xFF)) {
    return UINT8_C(0xFF);
  }

  /*
   * Subnormal magnitudes: everything at or below 2^-127 clamps to the lowest
   * scale 0x00, everything strictly above 2^-127 is covered exactly by the
   * next scale 0x01 (2^-126).
   */
  if (exponent == UINT32_C(0x00)) {
    return (fBits & UINT32_C(0x007FFFFF)) > UINT32_C(0x00400000)
               ? UINT8_C(0x01)
               : UINT8_C(0x00);
  }

  /*
   * A normal float32 value is (1 + m) 2^(E - 127); m == 0 is an exact power
   * of two and keeps the exponent, any other fraction rounds one step up.
   */
  const uint32_t bits =
      exponent +
      ((fBits & UINT32_C(0x007FFFFF)) != 0 ? UINT32_C(1) : UINT32_C(0));
  return bits > UINT32_C(0xFE) ? UINT8_C(0xFE) : static_cast<uint8_t>(bits);
}

/**
 * @brief Converts an 8-bit E8M0 scale factor to a 32-bit IEEE
 * single-precision float.
 *
 * @details
 * For every pattern except the two specials the result is the exponent field
 * shifted into place: float32 shares the E8M0 bias, so @c bits << 23 is the
 * identity. @c 0x00 must widen to the float32 subnormal @f$2^{-127}@f$ rather
 * than collapsing to zero, and @c 0xFF widens to a quiet NaN.
 *
 * @param[in] input The 8-bit E8M0 scale factor.
 * @return The single-precision float value.
 */
NCORE_HOST_DEVICE inline float fp8e8m0fnu_to_fp32_value(uint8_t input) {
  if (input == UINT8_C(0x00)) {
    /* 2^-127 exists in float32 only as the subnormal pattern 0x00400000. */
    return ncore::dtypes::detail::fp32_from_bits(UINT32_C(0x00400000));
  }
  if (input == UINT8_C(0xFF)) {
    return ncore::dtypes::detail::fp32_from_bits(UINT32_C(0x7F800001));
  }
  return ncore::dtypes::detail::fp32_from_bits(static_cast<uint32_t>(input)
                                               << 23);
}

} // namespace detail

// ============================================================
// Constructors and conversion operators — inline definitions
// ============================================================

inline NCORE_HOST_DEVICE Float8_e8m0fnu::Float8_e8m0fnu(float value)
    : x(detail::fp8e8m0fnu_from_fp32_value(value)) {}

inline NCORE_HOST_DEVICE Float8_e8m0fnu::operator float() const {
  return detail::fp8e8m0fnu_to_fp32_value(x);
}

inline NCORE_HOST_DEVICE bool Float8_e8m0fnu::isnan() const {
  return x == UINT8_C(0xFF);
}

/// @note Arithmetic and comparison operators are intentionally not defined:
/// the E8M0 factor is combined with block elements by quantized GEMM kernels,
/// not through element-wise dtype arithmetic.

} // namespace ncore::dtypes

// ============================================================
// std::numeric_limits specialization
// ============================================================
namespace std {

/**
 * @class numeric_limits<ncore::dtypes::Float8_e8m0fnu>
 * @brief Specialization of std::numeric_limits for the custom @ref
 * ncore::dtypes::Float8_e8m0fnu scale factor type.
 *
 * @details
 * Follows the same patterns as the PyTorch reference specialization: the
 * type is unsigned and bounded, has a quiet NaN and no infinity, and every
 * representable value is an exact power of two.
 */
template <> class numeric_limits<ncore::dtypes::Float8_e8m0fnu> {
  using Float8_e8m0fnu = ncore::dtypes::Float8_e8m0fnu;

public:
  static constexpr bool is_specialized = true; ///< Has numeric limits.
  static constexpr bool is_signed = false;     ///< E8M0 is unsigned.
  static constexpr bool is_integer = false;    ///< E8M0 is floating-point.
  static constexpr bool is_exact =
      false; ///< Value semantics of a scale factor are not exact.
  static constexpr bool has_infinity =
      false; ///< E8M0 has no infinity representation.
  static constexpr bool has_quiet_NaN =
      true; ///< E8M0 has a quiet NaN representation.
  static constexpr bool has_signaling_NaN =
      false; ///< E8M0 has no signaling NaN representation.
  static constexpr auto has_denorm =
      false; ///< Every E8M0 value is a normalized power of two.
  static constexpr auto has_denorm_loss =
      false; ///< Denormalization loss is impossible.
  static constexpr auto round_style =
      numeric_limits<float>::round_style; ///< Inherits float32 rounding style.
  static constexpr bool is_iec559 =
      false; ///< Does not conform to IEC 60559 (IEEE 754).
  static constexpr bool is_bounded = true; ///< Values are bounded.
  static constexpr bool is_modulo =
      false;                       ///< E8M0 arithmetic does not modulo wrap.
  static constexpr int digits = 1; ///< One effective significand bit.
  static constexpr int digits10 =
      0; ///< No decimal digit is reliably preserved.
  static constexpr int max_digits10 =
      1; ///< Decimal digits required to uniquely represent values.
  static constexpr int radix = 2; ///< Base of the exponent.
  static constexpr int min_exponent =
      -126; ///< radix^(min-1) is the smallest normal (2^-127).
  static constexpr int min_exponent10 = -38; ///< Minimum negative power of 10.
  static constexpr int max_exponent =
      128; ///< radix^(max-1) is exactly the largest finite (2^127).
  static constexpr int max_exponent10 = 38; ///< Maximum positive power of 10.
  static constexpr auto traps =
      numeric_limits<float>::traps; ///< Inherits float32 trap behavior.
  static constexpr auto tinyness_before =
      false; ///< Tinyness is not tested before rounding.

  /**
   * @brief Smallest positive value.
   * @details 0x00 → @f$2^{-127}@f$, the lowest representable scale.
   */
  static constexpr Float8_e8m0fnu min() {
    return {UINT8_C(0x00), Float8_e8m0fnu::from_bits()};
  }

  /**
   * @brief Smallest representable value.
   * @details Identical to @ref min(): the format is unsigned.
   */
  static constexpr Float8_e8m0fnu lowest() {
    return {UINT8_C(0x00), Float8_e8m0fnu::from_bits()};
  }

  /**
   * @brief Largest finite value.
   * @details 0xFE → @f$2^{127}@f$; 0xFF is the NaN pattern.
   */
  static constexpr Float8_e8m0fnu max() {
    return {UINT8_C(0xFE), Float8_e8m0fnu::from_bits()};
  }

  /**
   * @brief Machine epsilon.
   * @details 0x7F → @f$2^{0}@f$: the gap between adjacent scales at 1.0 is 1.
   */
  static constexpr Float8_e8m0fnu epsilon() {
    return {UINT8_C(0x7F), Float8_e8m0fnu::from_bits()};
  }

  /**
   * @brief Maximum rounding error.
   * @details 0x7E → @f$2^{-1}@f$ (0.5).
   */
  static constexpr Float8_e8m0fnu round_error() {
    return {UINT8_C(0x7E), Float8_e8m0fnu::from_bits()};
  }

  /**
   * @brief Quiet NaN.
   * @details 0xFF is the only NaN pattern of the format.
   */
  static constexpr Float8_e8m0fnu quiet_NaN() {
    return {UINT8_C(0xFF), Float8_e8m0fnu::from_bits()};
  }
};

} // namespace std

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif

#ifdef _GNUC_CLANG_
#pragma GCC diagnostic pop
#endif
