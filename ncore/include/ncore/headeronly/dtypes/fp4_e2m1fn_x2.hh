/**
 * @file fp4_e2m1fn_x2.hh
 * @brief 4-bit floating-point (FP4 E2M1FN) data type implementation,
 * including a packed-pair storage type.
 *
 * @details
 * Defines @ref Float4_e2m1fn, a single 4-bit floating-point scalar (stored
 * in the low nibble of a byte), and @ref Float4_e2m1fn_x2, a byte holding
 * two such values packed together — the FP4 dtype from the OCP MX format
 * spec
 * (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf,
 * Section 5.3.3).
 *
 * Binary layout of a single lane, MSB to LSB: `s ee m`
 *  - 1 sign bit
 *  - 2 exponent bits (bias = 1)
 *  - 1 mantissa bit
 *
 * The "fn" suffix denotes "finite": E2M1FN has no infinity or NaN encoding
 * at all — every one of the 16 possible 4-bit patterns maps to a finite
 * value. The representable magnitudes are exactly:
 * `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`. Values that would overflow (including
 * +/-inf and NaN inputs, which have no lossless representation in this
 * format) saturate to the maximum finite magnitude (6.0).
 *
 * Given two high precision values val0 and val1, here is the binary
 * configuration of their packed @ref Float4_e2m1fn_x2 representation, from
 * MSB to LSB:
 * @code
 *   original value             | val1 : val0
 *   ========================================
 *   bit index (MSB==7, LSB==0) | 7654 : 3210
 *   sign/exponent/mantissa     | seem : seem
 * @endcode
 *
 * Arithmetic on @ref Float4_e2m1fn_x2 is defined lane-wise: both packed
 * values are unpacked to float32, the operation is performed there
 * independently on each lane, and the results are re-packed. Mixed
 * arithmetic with a scalar (float/double/int/int64_t) broadcasts the
 * scalar to both lanes, matching common SIMD-lane semantics.
 *
 * @see half.hh          Shared fp32_from_bits/fp32_to_bits bit-cast helpers.
 * @see fp8_e4m3fn.hh    Sibling "fn" (finite, no-inf) reduced-precision format.
 */

#pragma once

#include <cstdlib>

#include <cmath>
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

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#endif

namespace ncore::dtypes {

// ============================================================
// Float4_e2m1fn — single 4-bit scalar (low nibble of a byte)
// ============================================================

/**
 * @struct Float4_e2m1fn
 * @brief Representation of a single 4-bit floating-point number in E2M1FN
 * format, stored in the low nibble of a byte (high nibble always zero).
 *
 * @details
 * Binary layout, MSB to LSB: `s ee m`
 *  - 1 sign bit
 *  - 2 exponent bits (bias = 1)
 *  - 1 mantissa bit
 *
 * This format has neither infinity nor NaN: all 16 bit patterns are finite.
 * The maximum finite magnitude is 6.0; @ref isnan and @ref isinf are kept
 * for API parity with the other reduced-precision types in this project and
 * always return @c false.
 */
struct alignas(1) Float4_e2m1fn {
  uint8_t x; ///< The 4-bit value in the low nibble; high nibble is zero.

  /**
   * @struct from_bits_t
   * @brief Tag type used to construct a @ref Float4_e2m1fn directly from its
   * raw 4-bit representation.
   */
  struct from_bits_t {};

  /**
   * @brief Returns a tag instance of @ref from_bits_t.
   * @return A default-constructed @ref from_bits_t.
   */
  NCORE_HOST_DEVICE static constexpr from_bits_t from_bits() { return {}; }

  /**
   * @brief Default constructor. Left uninitialized for performance.
   */
  Float4_e2m1fn() = default;

  /**
   * @brief Constructs a @ref Float4_e2m1fn directly from a raw 4-bit integer
   * representation.
   * @param[in] bits The raw 4-bit binary representation (low nibble; high
   * nibble bits are masked off).
   * @param[in] unused Tag parameter to disambiguate from float conversions.
   */
  constexpr NCORE_HOST_DEVICE Float4_e2m1fn(uint8_t bits,
                                            from_bits_t /*unused*/) noexcept
      : x(static_cast<uint8_t>(bits & 0x0F)) {}

  /**
   * @brief Implicit constructor from a single-precision float.
   * @details Values outside the representable range (including +/-inf and
   * NaN) saturate to the maximum finite magnitude (6.0), since E2M1FN has
   * no infinity or NaN encoding.
   * @param[in] value The float value to convert.
   */
  inline NCORE_HOST_DEVICE Float4_e2m1fn(float value);

  /**
   * @brief Implicit conversion operator to single-precision float.
   * @return The float representation of the fp4 value.
   */
  inline NCORE_HOST_DEVICE operator float() const;

  /**
   * @brief Checks whether this value encodes NaN.
   * @details E2M1FN has no NaN representation; always returns @c false.
   * @return Always @c false.
   */
  [[nodiscard]] inline NCORE_HOST_DEVICE bool isnan() const;

  /**
   * @brief Checks whether this value encodes infinity.
   * @details E2M1FN has no infinity representation; always returns
   * @c false.
   * @return Always @c false.
   */
  [[nodiscard]] inline NCORE_HOST_DEVICE bool isinf() const;
};

/**
 * @brief Stream output operator for @ref Float4_e2m1fn.
 * @details Promotes the fp4 value to float before writing to the stream.
 * @param[in,out] out The output stream.
 * @param[in]     value The @ref Float4_e2m1fn value to write.
 * @return Reference to the output stream.
 */
inline std::ostream &operator<<(std::ostream &out, const Float4_e2m1fn &value) {
  out << static_cast<float>(value);
  return out;
}

/**
 * @namespace ncore::dtypes::detail
 * @brief Internal low-level bit-manipulation and conversion helpers.
 */
namespace detail {

/**
 * @brief Converts a 4-bit E2M1FN bit pattern to a 32-bit IEEE
 * single-precision float value.
 * @details
 * With only 8 possible finite magnitudes, a direct lookup table is the
 * simplest and most auditable implementation — there is no meaningful
 * "bit-trick" formula to derive for a magnitude space this small, unlike
 * the wider FP8 formats.
 * @param[in] nibble The 4-bit E2M1FN bit pattern (only the low nibble is
 * read; any bits above bit 3 are ignored).
 * @return The single-precision float value.
 */
NCORE_HOST_DEVICE inline float fp4e2m1fn_to_fp32_value(uint8_t nibble) {
  // Indexed by the 3-bit magnitude (2 exponent bits << 1 | 1 mantissa bit).
  constexpr std::array<float, 8> kMagnitudes = {0.0f, 0.5f, 1.0f, 1.5f,
                                                2.0f, 3.0f, 4.0f, 6.0f};

  const uint8_t sign = (nibble >> 3) & 0x1;
  const uint8_t mag = nibble & 0x7;
  const float magnitude = kMagnitudes[mag];
  return (sign != 0u) ? -magnitude : magnitude;
}

/**
 * @brief Converts a 32-bit IEEE single-precision float to a 4-bit E2M1FN
 * bit pattern.
 * @details
 * Since E2M1FN has no infinity or NaN encoding, NaN input and any magnitude
 * at or beyond the round-to-even midpoint of the top two representable
 * values (5.0) saturate to the maximum finite magnitude (6.0). Ties between
 * adjacent representable magnitudes are resolved with round-to-nearest-even
 * (the midpoints 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0 are all exactly
 * representable in float32, so no precision is lost comparing against
 * them).
 * @param[in] f The single-precision float value to convert.
 * @return The 4-bit E2M1FN bit pattern (high nibble always zero).
 */
NCORE_HOST_DEVICE inline uint8_t fp4e2m1fn_from_fp32_value(float f) {
  const uint32_t bits = ncore::dtypes::detail::fp32_to_bits(f);
  const uint8_t sign = static_cast<uint8_t>((bits >> 31) & 0x1);

  uint8_t mag;
  if (std::isnan(f)) {
    // No NaN encoding available; saturate to the largest finite magnitude.
    mag = 0b111;
  } else {
    const float absVal = std::fabs(f);
    if (absVal > 5.0f) {
      mag = 0b111; // 6.0 (also covers +/-inf and any other overflow).
    } else if (absVal >= 3.5f) {
      mag = 0b110; // 4.0
    } else if (absVal > 2.5f) {
      mag = 0b101; // 3.0
    } else if (absVal >= 1.75f) {
      mag = 0b100; // 2.0
    } else if (absVal > 1.25f) {
      mag = 0b011; // 1.5
    } else if (absVal >= 0.75f) {
      mag = 0b010; // 1.0
    } else if (absVal > 0.25f) {
      mag = 0b001; // 0.5
    } else {
      mag = 0b000; // 0.0
    }
  }

  return static_cast<uint8_t>((sign << 3) | mag);
}

} // namespace detail

// ============================================================
// Float4_e2m1fn — constructors and conversion operators
// ============================================================

inline NCORE_HOST_DEVICE Float4_e2m1fn::Float4_e2m1fn(float value)
    : x(detail::fp4e2m1fn_from_fp32_value(value)) {}

inline NCORE_HOST_DEVICE Float4_e2m1fn::operator float() const {
  return detail::fp4e2m1fn_to_fp32_value(x);
}

inline NCORE_HOST_DEVICE bool Float4_e2m1fn::isnan() const { return false; }

inline NCORE_HOST_DEVICE bool Float4_e2m1fn::isinf() const { return false; }

// ============================================================
// Float4_e2m1fn — arithmetic operators
// ============================================================

/// @name Arithmetic Operators (Float4_e2m1fn & Float4_e2m1fn)
/// @{

/**
 * @brief Addition operator for two @ref Float4_e2m1fn values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The sum as a @ref Float4_e2m1fn.
 */
inline NCORE_HOST_DEVICE Float4_e2m1fn operator+(const Float4_e2m1fn &a,
                                                 const Float4_e2m1fn &b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

/**
 * @brief Subtraction operator for two @ref Float4_e2m1fn values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The difference as a @ref Float4_e2m1fn.
 */
inline NCORE_HOST_DEVICE Float4_e2m1fn operator-(const Float4_e2m1fn &a,
                                                 const Float4_e2m1fn &b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

/**
 * @brief Multiplication operator for two @ref Float4_e2m1fn values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The product as a @ref Float4_e2m1fn.
 */
inline NCORE_HOST_DEVICE Float4_e2m1fn operator*(const Float4_e2m1fn &a,
                                                 const Float4_e2m1fn &b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

/**
 * @brief Division operator for two @ref Float4_e2m1fn values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The quotient as a @ref Float4_e2m1fn.
 */
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float4_e2m1fn operator/(const Float4_e2m1fn &a,
                                                 const Float4_e2m1fn &b) {
  return static_cast<float>(a) / static_cast<float>(b);
}

/**
 * @brief Unary minus operator for @ref Float4_e2m1fn.
 * @param[in] a The operand.
 * @return The negated value as a @ref Float4_e2m1fn.
 */
inline NCORE_HOST_DEVICE Float4_e2m1fn operator-(const Float4_e2m1fn &a) {
  return -static_cast<float>(a);
}

/**
 * @brief Addition assignment operator.
 * @param[in,out] a The destination operand.
 * @param[in]     b The operand to add.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE Float4_e2m1fn &operator+=(Float4_e2m1fn &a,
                                                   const Float4_e2m1fn &b) {
  a = a + b;
  return a;
}

/**
 * @brief Subtraction assignment operator.
 * @param[in,out] a The destination operand.
 * @param[in]     b The operand to subtract.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE Float4_e2m1fn &operator-=(Float4_e2m1fn &a,
                                                   const Float4_e2m1fn &b) {
  a = a - b;
  return a;
}

/**
 * @brief Multiplication assignment operator.
 * @param[in,out] a The destination operand.
 * @param[in]     b The operand to multiply.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE Float4_e2m1fn &operator*=(Float4_e2m1fn &a,
                                                   const Float4_e2m1fn &b) {
  a = a * b;
  return a;
}

/**
 * @brief Division assignment operator.
 * @param[in,out] a The destination operand.
 * @param[in]     b The divisor.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE Float4_e2m1fn &operator/=(Float4_e2m1fn &a,
                                                   const Float4_e2m1fn &b) {
  a = a / b;
  return a;
}

/// @}

/// @name Mixed-type Arithmetic (Float4_e2m1fn & float)
/// @{

inline NCORE_HOST_DEVICE float operator+(Float4_e2m1fn a, float b) {
  return static_cast<float>(a) + b;
}
inline NCORE_HOST_DEVICE float operator-(Float4_e2m1fn a, float b) {
  return static_cast<float>(a) - b;
}
inline NCORE_HOST_DEVICE float operator*(Float4_e2m1fn a, float b) {
  return static_cast<float>(a) * b;
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE float operator/(Float4_e2m1fn a, float b) {
  return static_cast<float>(a) / b;
}

inline NCORE_HOST_DEVICE float operator+(float a, Float4_e2m1fn b) {
  return a + static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float operator-(float a, Float4_e2m1fn b) {
  return a - static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float operator*(float a, Float4_e2m1fn b) {
  return a * static_cast<float>(b);
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE float operator/(float a, Float4_e2m1fn b) {
  return a / static_cast<float>(b);
}

inline NCORE_HOST_DEVICE float &operator+=(float &a, const Float4_e2m1fn &b) {
  return a += static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float &operator-=(float &a, const Float4_e2m1fn &b) {
  return a -= static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float &operator*=(float &a, const Float4_e2m1fn &b) {
  return a *= static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float &operator/=(float &a, const Float4_e2m1fn &b) {
  return a /= static_cast<float>(b);
}

/// @}

/// @name Mixed-type Arithmetic (Float4_e2m1fn & double)
/// @{

inline NCORE_HOST_DEVICE double operator+(Float4_e2m1fn a, double b) {
  return static_cast<double>(a) + b;
}
inline NCORE_HOST_DEVICE double operator-(Float4_e2m1fn a, double b) {
  return static_cast<double>(a) - b;
}
inline NCORE_HOST_DEVICE double operator*(Float4_e2m1fn a, double b) {
  return static_cast<double>(a) * b;
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE double operator/(Float4_e2m1fn a, double b) {
  return static_cast<double>(a) / b;
}

inline NCORE_HOST_DEVICE double operator+(double a, Float4_e2m1fn b) {
  return a + static_cast<double>(b);
}
inline NCORE_HOST_DEVICE double operator-(double a, Float4_e2m1fn b) {
  return a - static_cast<double>(b);
}
inline NCORE_HOST_DEVICE double operator*(double a, Float4_e2m1fn b) {
  return a * static_cast<double>(b);
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE double operator/(double a, Float4_e2m1fn b) {
  return a / static_cast<double>(b);
}

/// @}

/// @name Mixed-type Arithmetic (Float4_e2m1fn & int)
/// @{

inline NCORE_HOST_DEVICE Float4_e2m1fn operator+(Float4_e2m1fn a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a + static_cast<Float4_e2m1fn>(b);
}
inline NCORE_HOST_DEVICE Float4_e2m1fn operator-(Float4_e2m1fn a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a - static_cast<Float4_e2m1fn>(b);
}
inline NCORE_HOST_DEVICE Float4_e2m1fn operator*(Float4_e2m1fn a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a * static_cast<Float4_e2m1fn>(b);
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float4_e2m1fn operator/(Float4_e2m1fn a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a / static_cast<Float4_e2m1fn>(b);
}

inline NCORE_HOST_DEVICE Float4_e2m1fn operator+(int a, Float4_e2m1fn b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Float4_e2m1fn>(a) + b;
}
inline NCORE_HOST_DEVICE Float4_e2m1fn operator-(int a, Float4_e2m1fn b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Float4_e2m1fn>(a) - b;
}
inline NCORE_HOST_DEVICE Float4_e2m1fn operator*(int a, Float4_e2m1fn b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Float4_e2m1fn>(a) * b;
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float4_e2m1fn operator/(int a, Float4_e2m1fn b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Float4_e2m1fn>(a) / b;
}

/// @}

/// @name Mixed-type Arithmetic (Float4_e2m1fn & int64_t)
/// @{

inline NCORE_HOST_DEVICE Float4_e2m1fn operator+(Float4_e2m1fn a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a + static_cast<Float4_e2m1fn>(b);
}
inline NCORE_HOST_DEVICE Float4_e2m1fn operator-(Float4_e2m1fn a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a - static_cast<Float4_e2m1fn>(b);
}
inline NCORE_HOST_DEVICE Float4_e2m1fn operator*(Float4_e2m1fn a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a * static_cast<Float4_e2m1fn>(b);
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float4_e2m1fn operator/(Float4_e2m1fn a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a / static_cast<Float4_e2m1fn>(b);
}

inline NCORE_HOST_DEVICE Float4_e2m1fn operator+(int64_t a, Float4_e2m1fn b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Float4_e2m1fn>(a) + b;
}
inline NCORE_HOST_DEVICE Float4_e2m1fn operator-(int64_t a, Float4_e2m1fn b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Float4_e2m1fn>(a) - b;
}
inline NCORE_HOST_DEVICE Float4_e2m1fn operator*(int64_t a, Float4_e2m1fn b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Float4_e2m1fn>(a) * b;
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float4_e2m1fn operator/(int64_t a, Float4_e2m1fn b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Float4_e2m1fn>(a) / b;
}

/// @}

/// @note Comparison operators are not defined for Float4_e2m1fn; rely on
/// the implicit conversion to float.

} // namespace ncore::dtypes

// ============================================================
// std::numeric_limits<Float4_e2m1fn> specialisation
// ============================================================
namespace std {

/**
 * @class numeric_limits<ncore::dtypes::Float4_e2m1fn>
 * @brief Specialization of std::numeric_limits for the custom @ref
 * ncore::dtypes::Float4_e2m1fn type.
 */
template <> class numeric_limits<ncore::dtypes::Float4_e2m1fn> {
  using Float4_e2m1fn = ncore::dtypes::Float4_e2m1fn;

public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity =
      false; ///< E2M1FN has no infinity representation.
  static constexpr bool has_quiet_NaN =
      false; ///< E2M1FN has no NaN representation.
  static constexpr bool has_signaling_NaN = false;
  static constexpr auto has_denorm = true; ///< 0.5 is a subnormal value.
  static constexpr auto has_denorm_loss = true;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 2; ///< 1 mantissa bit + 1 implicit bit.
  static constexpr int digits10 = 0;
  static constexpr int max_digits10 = 2;
  static constexpr int radix = 2;
  static constexpr int min_exponent = 0; ///< Smallest normal: 1.0 = 2**0.
  static constexpr int min_exponent10 = 0;
  static constexpr int max_exponent = 3; ///< Largest normal: 4.0 = 2**2.
  static constexpr int max_exponent10 = 0;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before = false;

  /**
   * @brief Smallest positive normalized value.
   * @details 0b0010 → 1.0.
   */
  static constexpr Float4_e2m1fn min() {
    return {0b0010, Float4_e2m1fn::from_bits()};
  }

  /**
   * @brief Largest finite negative value.
   * @details 0b1111 → -6.0.
   */
  static constexpr Float4_e2m1fn lowest() {
    return {0b1111, Float4_e2m1fn::from_bits()};
  }

  /**
   * @brief Largest finite positive value.
   * @details 0b0111 → 6.0.
   */
  static constexpr Float4_e2m1fn max() {
    return {0b0111, Float4_e2m1fn::from_bits()};
  }

  /**
   * @brief Machine epsilon.
   * @details 0b0011 → 1.5; the gap between 1.0 and the next representable
   * value is itself 0.5, i.e. 2**(-1).
   */
  static constexpr Float4_e2m1fn epsilon() {
    return {0b0011, Float4_e2m1fn::from_bits()};
  }

  /**
   * @brief Maximum rounding error.
   * @details 0b0001 → 0.5.
   */
  static constexpr Float4_e2m1fn round_error() {
    return {0b0001, Float4_e2m1fn::from_bits()};
  }

  /**
   * @brief Smallest positive subnormal value.
   * @details 0b0001 → 0.5 (the only subnormal magnitude in this format).
   */
  static constexpr Float4_e2m1fn denorm_min() {
    return {0b0001, Float4_e2m1fn::from_bits()};
  }
};

} // namespace std

namespace ncore::dtypes {

// ============================================================
// Float4_e2m1fn_x2 — two 4-bit lanes packed into one byte
// ============================================================

/**
 * @struct Float4_e2m1fn_x2
 * @brief Storage type for two 4-bit E2M1FN floating-point values packed into
 * a single byte.
 *
 * @details
 * Unlike @ref Float4_e2m1fn, this type represents two independent lanes.
 * All arithmetic and conversion below operates lane-wise: each of the two
 * packed values is unpacked, the float32 operation is applied
 * independently, and the results are re-packed. A scalar operand
 * (float/double/int/int64_t) broadcasts to both lanes.
 */
struct alignas(1) Float4_e2m1fn_x2 {
  uint8_t val_; ///< Packed byte: high nibble = val1, low nibble = val0.

  /**
   * @brief Default constructor. Left uninitialized for performance.
   */
  Float4_e2m1fn_x2() = default;

  /**
   * @brief Constructs a @ref Float4_e2m1fn_x2 directly from a raw packed
   * byte.
   * @param[in] val The packed byte, high nibble = val1, low nibble = val0.
   */
  NCORE_HOST_DEVICE explicit constexpr Float4_e2m1fn_x2(uint8_t val) noexcept
      : val_(val) {}

  /**
   * @brief Constructs a @ref Float4_e2m1fn_x2 by packing two lane values.
   * @param[in] lo The value to store in the low nibble (val0).
   * @param[in] hi The value to store in the high nibble (val1).
   */
  inline NCORE_HOST_DEVICE Float4_e2m1fn_x2(Float4_e2m1fn lo, Float4_e2m1fn hi);

  /**
   * @brief Constructs a @ref Float4_e2m1fn_x2 by converting and packing two
   * float32 values.
   * @param[in] lo The value to convert and store in the low nibble (val0).
   * @param[in] hi The value to convert and store in the high nibble (val1).
   */
  inline NCORE_HOST_DEVICE Float4_e2m1fn_x2(float lo, float hi);

  /**
   * @brief Extracts and decodes the low nibble (val0).
   * @return The low-lane value as a @ref Float4_e2m1fn.
   */
  [[nodiscard]] inline NCORE_HOST_DEVICE Float4_e2m1fn low() const;

  /**
   * @brief Extracts and decodes the high nibble (val1).
   * @return The high-lane value as a @ref Float4_e2m1fn.
   */
  [[nodiscard]] inline NCORE_HOST_DEVICE Float4_e2m1fn high() const;

  /**
   * @brief Checks whether either lane encodes NaN.
   * @details E2M1FN has no NaN representation; always returns @c false.
   * @return Always @c false.
   */
  [[nodiscard]] inline NCORE_HOST_DEVICE bool isnan() const { return false; }

  /**
   * @brief Checks whether either lane encodes infinity.
   * @details E2M1FN has no infinity representation; always returns
   * @c false.
   * @return Always @c false.
   */
  [[nodiscard]] inline NCORE_HOST_DEVICE bool isinf() const { return false; }
};

/**
 * @brief Stream output operator for @ref Float4_e2m1fn_x2.
 * @details Prints both decoded lanes as an ordered pair `(val0, val1)`.
 * @param[in,out] out The output stream.
 * @param[in]     value The @ref Float4_e2m1fn_x2 value to write.
 * @return Reference to the output stream.
 */
inline std::ostream &operator<<(std::ostream &out,
                                const Float4_e2m1fn_x2 &value) {
  out << '(' << static_cast<float>(value.low()) << ", "
      << static_cast<float>(value.high()) << ')';
  return out;
}

// ============================================================
// Float4_e2m1fn_x2 — pack/unpack inline definitions
// ============================================================

inline NCORE_HOST_DEVICE Float4_e2m1fn_x2::Float4_e2m1fn_x2(Float4_e2m1fn lo,
                                                            Float4_e2m1fn hi)
    : val_(static_cast<uint8_t>((hi.x << 4) | (lo.x & 0x0F))) {}

inline NCORE_HOST_DEVICE Float4_e2m1fn_x2::Float4_e2m1fn_x2(float lo, float hi)
    : Float4_e2m1fn_x2(Float4_e2m1fn(lo), Float4_e2m1fn(hi)) {}

inline NCORE_HOST_DEVICE Float4_e2m1fn Float4_e2m1fn_x2::low() const {
  return {static_cast<uint8_t>(val_ & 0x0F), Float4_e2m1fn::from_bits()};
}

inline NCORE_HOST_DEVICE Float4_e2m1fn Float4_e2m1fn_x2::high() const {
  return {static_cast<uint8_t>((val_ >> 4) & 0x0F), Float4_e2m1fn::from_bits()};
}

// ============================================================
// Float4_e2m1fn_x2 — comparison operators (raw-byte based)
// ============================================================

/// @name Comparison Operators
/// @{

/**
 * @brief Equality comparison, based on the raw packed byte.
 * @note This compares bit patterns, not decoded values: +0.0 and -0.0 in a
 * given lane compare unequal here even though they are numerically equal,
 * matching the original packed-storage semantics of this type.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return @c true if both packed bytes are bitwise identical.
 */
inline NCORE_HOST_DEVICE bool operator==(const Float4_e2m1fn_x2 &a,
                                         const Float4_e2m1fn_x2 &b) {
  return a.val_ == b.val_;
}

/**
 * @brief Inequality comparison, based on the raw packed byte.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return @c true if the packed bytes differ.
 */
inline NCORE_HOST_DEVICE bool operator!=(const Float4_e2m1fn_x2 &a,
                                         const Float4_e2m1fn_x2 &b) {
  return a.val_ != b.val_;
}

/// @}

// ============================================================
// Float4_e2m1fn_x2 — lane-wise arithmetic operators
// ============================================================

/// @name Arithmetic Operators (Float4_e2m1fn_x2 & Float4_e2m1fn_x2, lane-wise)
/// @{

/**
 * @brief Lane-wise addition operator for two @ref Float4_e2m1fn_x2 values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The lane-wise sum as a @ref Float4_e2m1fn_x2.
 */
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator+(const Float4_e2m1fn_x2 &a,
                                                    const Float4_e2m1fn_x2 &b) {
  return {static_cast<float>(a.low()) + static_cast<float>(b.low()),
          static_cast<float>(a.high()) + static_cast<float>(b.high())};
}

/**
 * @brief Lane-wise subtraction operator for two @ref Float4_e2m1fn_x2
 * values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The lane-wise difference as a @ref Float4_e2m1fn_x2.
 */
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator-(const Float4_e2m1fn_x2 &a,
                                                    const Float4_e2m1fn_x2 &b) {
  return {static_cast<float>(a.low()) - static_cast<float>(b.low()),
          static_cast<float>(a.high()) - static_cast<float>(b.high())};
}

/**
 * @brief Lane-wise multiplication operator for two @ref Float4_e2m1fn_x2
 * values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The lane-wise product as a @ref Float4_e2m1fn_x2.
 */
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator*(const Float4_e2m1fn_x2 &a,
                                                    const Float4_e2m1fn_x2 &b) {
  return {static_cast<float>(a.low()) * static_cast<float>(b.low()),
          static_cast<float>(a.high()) * static_cast<float>(b.high())};
}

/**
 * @brief Lane-wise division operator for two @ref Float4_e2m1fn_x2 values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The lane-wise quotient as a @ref Float4_e2m1fn_x2.
 */
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator/(const Float4_e2m1fn_x2 &a,
                                                    const Float4_e2m1fn_x2 &b) {
  return {static_cast<float>(a.low()) / static_cast<float>(b.low()),
          static_cast<float>(a.high()) / static_cast<float>(b.high())};
}

/**
 * @brief Lane-wise unary minus operator for @ref Float4_e2m1fn_x2.
 * @param[in] a The operand.
 * @return The lane-wise negated value as a @ref Float4_e2m1fn_x2.
 */
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator-(const Float4_e2m1fn_x2 &a) {
  return {-static_cast<float>(a.low()), -static_cast<float>(a.high())};
}

/**
 * @brief Addition assignment operator (lane-wise).
 * @param[in,out] a The destination operand.
 * @param[in]     b The operand to add.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 &
operator+=(Float4_e2m1fn_x2 &a, const Float4_e2m1fn_x2 &b) {
  a = a + b;
  return a;
}

/**
 * @brief Subtraction assignment operator (lane-wise).
 * @param[in,out] a The destination operand.
 * @param[in]     b The operand to subtract.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 &
operator-=(Float4_e2m1fn_x2 &a, const Float4_e2m1fn_x2 &b) {
  a = a - b;
  return a;
}

/**
 * @brief Multiplication assignment operator (lane-wise).
 * @param[in,out] a The destination operand.
 * @param[in]     b The operand to multiply.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 &
operator*=(Float4_e2m1fn_x2 &a, const Float4_e2m1fn_x2 &b) {
  a = a * b;
  return a;
}

/**
 * @brief Division assignment operator (lane-wise).
 * @param[in,out] a The destination operand.
 * @param[in]     b The divisor.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 &
operator/=(Float4_e2m1fn_x2 &a, const Float4_e2m1fn_x2 &b) {
  a = a / b;
  return a;
}

/// @}

/// @name Mixed-type Arithmetic (Float4_e2m1fn_x2 & float, scalar broadcast)
/// @{

inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator+(const Float4_e2m1fn_x2 &a,
                                                    float b) {
  return {static_cast<float>(a.low()) + b, static_cast<float>(a.high()) + b};
}
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator-(const Float4_e2m1fn_x2 &a,
                                                    float b) {
  return {static_cast<float>(a.low()) - b, static_cast<float>(a.high()) - b};
}
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator*(const Float4_e2m1fn_x2 &a,
                                                    float b) {
  return {static_cast<float>(a.low()) * b, static_cast<float>(a.high()) * b};
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator/(const Float4_e2m1fn_x2 &a,
                                                    float b) {
  return {static_cast<float>(a.low()) / b, static_cast<float>(a.high()) / b};
}

inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator+(float a,
                                                    const Float4_e2m1fn_x2 &b) {
  return {a + static_cast<float>(b.low()), a + static_cast<float>(b.high())};
}
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator-(float a,
                                                    const Float4_e2m1fn_x2 &b) {
  return {a - static_cast<float>(b.low()), a - static_cast<float>(b.high())};
}
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator*(float a,
                                                    const Float4_e2m1fn_x2 &b) {
  return {a * static_cast<float>(b.low()), a * static_cast<float>(b.high())};
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator/(float a,
                                                    const Float4_e2m1fn_x2 &b) {
  return {a / static_cast<float>(b.low()), a / static_cast<float>(b.high())};
}

/// @}

/// @name Mixed-type Arithmetic (Float4_e2m1fn_x2 & double, scalar broadcast)
/// @{

inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator+(const Float4_e2m1fn_x2 &a,
                                                    double b) {
  return {static_cast<float>(static_cast<double>(a.low()) + b),
          static_cast<float>(static_cast<double>(a.high()) + b)};
}
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator-(const Float4_e2m1fn_x2 &a,
                                                    double b) {
  return {static_cast<float>(static_cast<double>(a.low()) - b),
          static_cast<float>(static_cast<double>(a.high()) - b)};
}
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator*(const Float4_e2m1fn_x2 &a,
                                                    double b) {
  return {static_cast<float>(static_cast<double>(a.low()) * b),
          static_cast<float>(static_cast<double>(a.high()) * b)};
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator/(const Float4_e2m1fn_x2 &a,
                                                    double b) {
  return {static_cast<float>(static_cast<double>(a.low()) / b),
          static_cast<float>(static_cast<double>(a.high()) / b)};
}

inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator+(double a,
                                                    const Float4_e2m1fn_x2 &b) {
  return {static_cast<float>(a + static_cast<double>(b.low())),
          static_cast<float>(a + static_cast<double>(b.high()))};
}
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator-(double a,
                                                    const Float4_e2m1fn_x2 &b) {
  return {static_cast<float>(a - static_cast<double>(b.low())),
          static_cast<float>(a - static_cast<double>(b.high()))};
}
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator*(double a,
                                                    const Float4_e2m1fn_x2 &b) {
  return {static_cast<float>(a * static_cast<double>(b.low())),
          static_cast<float>(a * static_cast<double>(b.high()))};
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator/(double a,
                                                    const Float4_e2m1fn_x2 &b) {
  return {static_cast<float>(a / static_cast<double>(b.low())),
          static_cast<float>(a / static_cast<double>(b.high()))};
}

/// @}

/// @name Mixed-type Arithmetic (Float4_e2m1fn_x2 & int, scalar broadcast)
/// @{

inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator+(const Float4_e2m1fn_x2 &a,
                                                    int b) {
  return a + static_cast<float>(b);
}
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator-(const Float4_e2m1fn_x2 &a,
                                                    int b) {
  return a - static_cast<float>(b);
}
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator*(const Float4_e2m1fn_x2 &a,
                                                    int b) {
  return a * static_cast<float>(b);
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator/(const Float4_e2m1fn_x2 &a,
                                                    int b) {
  return a / static_cast<float>(b);
}

inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator+(int a,
                                                    const Float4_e2m1fn_x2 &b) {
  return static_cast<float>(a) + b;
}
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator-(int a,
                                                    const Float4_e2m1fn_x2 &b) {
  return static_cast<float>(a) - b;
}
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator*(int a,
                                                    const Float4_e2m1fn_x2 &b) {
  return static_cast<float>(a) * b;
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator/(int a,
                                                    const Float4_e2m1fn_x2 &b) {
  return static_cast<float>(a) / b;
}

/// @}

/// @name Mixed-type Arithmetic (Float4_e2m1fn_x2 & int64_t, scalar broadcast)
/// @{

inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator+(const Float4_e2m1fn_x2 &a,
                                                    int64_t b) {
  return a + static_cast<float>(b);
}
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator-(const Float4_e2m1fn_x2 &a,
                                                    int64_t b) {
  return a - static_cast<float>(b);
}
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator*(const Float4_e2m1fn_x2 &a,
                                                    int64_t b) {
  return a * static_cast<float>(b);
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator/(const Float4_e2m1fn_x2 &a,
                                                    int64_t b) {
  return a / static_cast<float>(b);
}

inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator+(int64_t a,
                                                    const Float4_e2m1fn_x2 &b) {
  return static_cast<float>(a) + b;
}
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator-(int64_t a,
                                                    const Float4_e2m1fn_x2 &b) {
  return static_cast<float>(a) - b;
}
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator*(int64_t a,
                                                    const Float4_e2m1fn_x2 &b) {
  return static_cast<float>(a) * b;
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float4_e2m1fn_x2 operator/(int64_t a,
                                                    const Float4_e2m1fn_x2 &b) {
  return static_cast<float>(a) / b;
}

/// @}

} // namespace ncore::dtypes

// ============================================================
// std::numeric_limits<Float4_e2m1fn_x2> specialisation
// ============================================================
namespace std {

/**
 * @class numeric_limits<ncore::dtypes::Float4_e2m1fn_x2>
 * @brief Specialization of std::numeric_limits for the custom @ref
 * ncore::dtypes::Float4_e2m1fn_x2 type.
 * @details Since both packed lanes share identical format properties, the
 * scalar-valued members below mirror @ref
 * numeric_limits<ncore::dtypes::Float4_e2m1fn> exactly; the value-returning
 * members pack that same scalar limit into both lanes.
 */
template <> class numeric_limits<ncore::dtypes::Float4_e2m1fn_x2> {
  using Float4_e2m1fn = ncore::dtypes::Float4_e2m1fn;
  using Float4_e2m1fn_x2 = ncore::dtypes::Float4_e2m1fn_x2;
  using lane_limits = numeric_limits<Float4_e2m1fn>;

public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = false;
  static constexpr bool has_quiet_NaN = false;
  static constexpr bool has_signaling_NaN = false;
  static constexpr auto has_denorm = lane_limits::has_denorm;
  static constexpr auto has_denorm_loss = lane_limits::has_denorm_loss;
  static constexpr auto round_style = lane_limits::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = lane_limits::digits;
  static constexpr int digits10 = lane_limits::digits10;
  static constexpr int max_digits10 = lane_limits::max_digits10;
  static constexpr int radix = lane_limits::radix;
  static constexpr int min_exponent = lane_limits::min_exponent;
  static constexpr int min_exponent10 = lane_limits::min_exponent10;
  static constexpr int max_exponent = lane_limits::max_exponent;
  static constexpr int max_exponent10 = lane_limits::max_exponent10;
  static constexpr auto traps = lane_limits::traps;
  static constexpr auto tinyness_before = lane_limits::tinyness_before;

  /// @brief Both lanes set to the smallest positive normalized value.
  static constexpr Float4_e2m1fn_x2 min() {
    return {lane_limits::min(), lane_limits::min()};
  }
  /// @brief Both lanes set to the largest finite negative value.
  static constexpr Float4_e2m1fn_x2 lowest() {
    return {lane_limits::lowest(), lane_limits::lowest()};
  }
  /// @brief Both lanes set to the largest finite positive value.
  static constexpr Float4_e2m1fn_x2 max() {
    return {lane_limits::max(), lane_limits::max()};
  }
  /// @brief Both lanes set to the machine epsilon.
  static constexpr Float4_e2m1fn_x2 epsilon() {
    return {lane_limits::epsilon(), lane_limits::epsilon()};
  }
  /// @brief Both lanes set to the maximum rounding error.
  static constexpr Float4_e2m1fn_x2 round_error() {
    return {lane_limits::round_error(), lane_limits::round_error()};
  }
  /// @brief Both lanes set to the smallest positive subnormal value.
  static constexpr Float4_e2m1fn_x2 denorm_min() {
    return {lane_limits::denorm_min(), lane_limits::denorm_min()};
  }
};

} // namespace std

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#ifdef _GNUC_CLANG_
#pragma GCC diagnostic pop
#endif
