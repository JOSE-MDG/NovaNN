/**
 * @file fp8_e5m2.hh
 * @brief 8-bit floating-point (FP8 E5M2) data type implementation.
 *
 * @details
 * Defines the @ref Float8_e5m2 type: an 8-bit floating-point format with
 * 1 sign bit, 5 exponent bits, 2 mantissa bits and bias = 15. Because it
 * shares its exponent width and bias with IEEE 754 half-precision (FP16),
 * conversion to/from float32 is implemented by re-using this project's own
 * @ref ncore::dtypes::detail::fp16_ieee_to_fp32_value helper after a simple
 * bit-shift, exactly as PyTorch's own reference implementation does. This
 * includes conversions to and from standard C++ types (float, double, int,
 * etc.) and basic arithmetic operations.
 *
 * Arithmetic operations are implemented by converting to float32 and
 * performing the operation there, as most operations are memory-bound.
 *
 * Implementation based on https://arxiv.org/pdf/2209.05433.pdf and modeled
 * after this project's own @ref ncore::dtypes::Half conversion helpers.
 *
 * @see Half.hpp  Provides fp16_ieee_to_fp32_value and fp32_to_bits/from_bits.
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

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#endif

namespace ncore::dtypes {

/**
 * @struct Float8_e5m2
 * @brief Representation of an 8-bit floating-point number in E5M2 format.
 *
 * @details
 * Binary layout, MSB to LSB: `s eeeee mm`
 *  - 1 sign bit
 *  - 5 exponent bits (bias = 15, same as FP16)
 *  - 2 mantissa bits
 *
 * Unlike @ref Float8_e4m3fn, E5M2 has a conventional infinity encoding
 * (exponent all-ones, mantissa zero), matching the IEEE 754 convention.
 */
struct alignas(1) Float8_e5m2 {
  uint8_t x; ///< The 8-bit binary representation of the FP8 value.

  /**
   * @struct from_bits_t
   * @brief Tag type used to construct a @ref Float8_e5m2 directly from its
   * raw 8-bit representation.
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
  Float8_e5m2() = default;

  /**
   * @brief Constructs a @ref Float8_e5m2 directly from a raw 8-bit integer
   * representation.
   * @param[in] bits The raw 8-bit binary representation.
   * @param[in] unused Tag parameter to disambiguate from float conversions.
   */
  constexpr NCORE_HOST_DEVICE Float8_e5m2(uint8_t bits,
                                          from_bits_t /*unused*/) noexcept
      : x(bits) {}

  /**
   * @brief Implicit constructor from a single-precision float.
   * @param[in] value The float value to convert.
   */
  inline NCORE_HOST_DEVICE Float8_e5m2(float value);

  /**
   * @brief Implicit conversion operator to single-precision float.
   * @return The float representation of the fp8 value.
   */
  inline NCORE_HOST_DEVICE operator float() const;

  /**
   * @brief Checks whether this value encodes NaN.
   * @return @c true if this is a NaN pattern.
   */
  inline NCORE_HOST_DEVICE bool isnan() const;

  /**
   * @brief Checks whether this value encodes infinity.
   * @return @c true if this is +/-infinity.
   */
  inline NCORE_HOST_DEVICE bool isinf() const;
};

/**
 * @brief Stream output operator for @ref Float8_e5m2.
 * @details Promotes the fp8 value to float before writing to the stream.
 * @param[in,out] out The output stream.
 * @param[in]     value The @ref Float8_e5m2 value to write.
 * @return Reference to the output stream.
 */
inline std::ostream &operator<<(std::ostream &out, const Float8_e5m2 &value) {
  out << static_cast<float>(value);
  return out;
}

/**
 * @namespace ncore::dtypes::detail
 * @brief Internal low-level bit-manipulation and conversion helpers.
 */
namespace detail {

/**
 * @brief Converts an 8-bit E5M2 bit pattern to a 32-bit IEEE
 * single-precision float value.
 * @details
 * Zero-extends the fp8 pattern into the upper 8 bits of a 16-bit word:
 * @code
 *      +---+-----+--+-----------------------------+
 *      | S |EEEEE|MM|0000 0000 0000 0000 0000 0000|
 *      +---+-----+--+-----------------------------+
 * Bits  31 26-30 24-25          0-23
 * @endcode
 * then reuses @ref ncore::dtypes::detail::fp16_ieee_to_fp32_value, since
 * E5M2 shares FP16's exponent width and bias exactly.
 * @param[in] input The 8-bit E5M2 bit pattern.
 * @return The single-precision float value.
 */
NCORE_HOST_DEVICE inline float fp8e5m2_to_fp32_value(uint8_t input) {
  uint16_t halfRepresentation = input;
  halfRepresentation = static_cast<uint16_t>(halfRepresentation << 8);
  return ncore::dtypes::detail::fp16_ieee_to_fp32_value(halfRepresentation);
}

/**
 * @brief Converts a 32-bit IEEE single-precision float to an 8-bit E5M2
 * bit pattern.
 * @param[in] f The single-precision float value to convert.
 * @return The 8-bit E5M2 bit pattern.
 */
NCORE_HOST_DEVICE inline uint8_t fp8e5m2_from_fp32_value(float f) {
  /*
   * Binary representation of fp32 infinity: 0 11111111 0...0
   */
  constexpr uint32_t fp32Inf = UINT32_C(255) << 23;

  /*
   * Binary representation of 65536.0f, the first value not representable in
   * E5M2 range:
   *   0 11111 00 - fp8 E5M2
   *   0 10001111 0...0 - fp32
   */
  constexpr uint32_t fp8Max = UINT32_C(143) << 23;

  /*
   * Mask for converting fp32 numbers below E5M2's normal range into
   * denormal representation. Magic number: (127 - 15) + (23 - 2) + 1.
   */
  constexpr uint32_t denormMask = UINT32_C(134) << 23;

  uint32_t fBits = ncore::dtypes::detail::fp32_to_bits(f);
  uint8_t result = 0;

  const uint32_t sign = fBits & UINT32_C(0x80000000);
  fBits ^= sign;

  if (fBits >= fp8Max) {
    // NaN — all exponent and mantissa bits set to 1.
    result = fBits > fp32Inf ? UINT8_C(0x7F) : UINT8_C(0x7C);
  } else {
    if (fBits < (UINT32_C(113) << 23)) {
      // Input smaller than 2^(-14), the smallest E5M2 normal number.
      fBits = ncore::dtypes::detail::fp32_to_bits(
          ncore::dtypes::detail::fp32_from_bits(fBits) +
          ncore::dtypes::detail::fp32_from_bits(denormMask));
      result = static_cast<uint8_t>(fBits - denormMask);
    } else {
      // Resulting mantissa is odd.
      const uint32_t mantOdd = (fBits >> 21) & 1;

      // Update exponent, rounding bias part 1.
      fBits += (static_cast<uint32_t>(15 - 127) << 23) + 0xFFFFF;

      // Rounding bias part 2.
      fBits += mantOdd;

      // Take the bits!
      result = static_cast<uint8_t>(fBits >> 21);
    }
  }

  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}

} // namespace detail

// ============================================================
// Constructors and conversion operators — inline definitions
// ============================================================

inline NCORE_HOST_DEVICE Float8_e5m2::Float8_e5m2(float value)
    : x(detail::fp8e5m2_from_fp32_value(value)) {}

inline NCORE_HOST_DEVICE Float8_e5m2::operator float() const {
  return detail::fp8e5m2_to_fp32_value(x);
}

inline NCORE_HOST_DEVICE bool Float8_e5m2::isnan() const {
  return (x & 0b0111'1111) > 0b0111'1100;
}

inline NCORE_HOST_DEVICE bool Float8_e5m2::isinf() const {
  return (x & 0b0111'1111) == 0b0111'1100;
}

// ============================================================
// Arithmetic operators
// ============================================================

/// @name Arithmetic Operators (Float8_e5m2 & Float8_e5m2)
/// @{

/**
 * @brief Addition operator for two @ref Float8_e5m2 values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The sum as a @ref Float8_e5m2.
 */
inline NCORE_HOST_DEVICE Float8_e5m2 operator+(const Float8_e5m2 &a,
                                               const Float8_e5m2 &b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

/**
 * @brief Subtraction operator for two @ref Float8_e5m2 values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The difference as a @ref Float8_e5m2.
 */
inline NCORE_HOST_DEVICE Float8_e5m2 operator-(const Float8_e5m2 &a,
                                               const Float8_e5m2 &b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

/**
 * @brief Multiplication operator for two @ref Float8_e5m2 values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The product as a @ref Float8_e5m2.
 */
inline NCORE_HOST_DEVICE Float8_e5m2 operator*(const Float8_e5m2 &a,
                                               const Float8_e5m2 &b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

/**
 * @brief Division operator for two @ref Float8_e5m2 values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The quotient as a @ref Float8_e5m2.
 */
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float8_e5m2 operator/(const Float8_e5m2 &a,
                                               const Float8_e5m2 &b) {
  return static_cast<float>(a) / static_cast<float>(b);
}

/**
 * @brief Unary minus operator for @ref Float8_e5m2.
 * @param[in] a The operand.
 * @return The negated value as a @ref Float8_e5m2.
 */
inline NCORE_HOST_DEVICE Float8_e5m2 operator-(const Float8_e5m2 &a) {
  return -static_cast<float>(a);
}

/**
 * @brief Addition assignment operator.
 * @param[in,out] a The destination operand.
 * @param[in]     b The operand to add.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE Float8_e5m2 &operator+=(Float8_e5m2 &a,
                                                 const Float8_e5m2 &b) {
  a = a + b;
  return a;
}

/**
 * @brief Subtraction assignment operator.
 * @param[in,out] a The destination operand.
 * @param[in]     b The operand to subtract.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE Float8_e5m2 &operator-=(Float8_e5m2 &a,
                                                 const Float8_e5m2 &b) {
  a = a - b;
  return a;
}

/**
 * @brief Multiplication assignment operator.
 * @param[in,out] a The destination operand.
 * @param[in]     b The operand to multiply.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE Float8_e5m2 &operator*=(Float8_e5m2 &a,
                                                 const Float8_e5m2 &b) {
  a = a * b;
  return a;
}

/**
 * @brief Division assignment operator.
 * @param[in,out] a The destination operand.
 * @param[in]     b The divisor.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE Float8_e5m2 &operator/=(Float8_e5m2 &a,
                                                 const Float8_e5m2 &b) {
  a = a / b;
  return a;
}

/// @}

/// @name Mixed-type Arithmetic (Float8_e5m2 & float)
/// @{

inline NCORE_HOST_DEVICE float operator+(Float8_e5m2 a, float b) {
  return static_cast<float>(a) + b;
}
inline NCORE_HOST_DEVICE float operator-(Float8_e5m2 a, float b) {
  return static_cast<float>(a) - b;
}
inline NCORE_HOST_DEVICE float operator*(Float8_e5m2 a, float b) {
  return static_cast<float>(a) * b;
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE float operator/(Float8_e5m2 a, float b) {
  return static_cast<float>(a) / b;
}

inline NCORE_HOST_DEVICE float operator+(float a, Float8_e5m2 b) {
  return a + static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float operator-(float a, Float8_e5m2 b) {
  return a - static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float operator*(float a, Float8_e5m2 b) {
  return a * static_cast<float>(b);
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE float operator/(float a, Float8_e5m2 b) {
  return a / static_cast<float>(b);
}

inline NCORE_HOST_DEVICE float &operator+=(float &a, const Float8_e5m2 &b) {
  return a += static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float &operator-=(float &a, const Float8_e5m2 &b) {
  return a -= static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float &operator*=(float &a, const Float8_e5m2 &b) {
  return a *= static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float &operator/=(float &a, const Float8_e5m2 &b) {
  return a /= static_cast<float>(b);
}

/// @}

/// @name Mixed-type Arithmetic (Float8_e5m2 & double)
/// @{

inline NCORE_HOST_DEVICE double operator+(Float8_e5m2 a, double b) {
  return static_cast<double>(a) + b;
}
inline NCORE_HOST_DEVICE double operator-(Float8_e5m2 a, double b) {
  return static_cast<double>(a) - b;
}
inline NCORE_HOST_DEVICE double operator*(Float8_e5m2 a, double b) {
  return static_cast<double>(a) * b;
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE double operator/(Float8_e5m2 a, double b) {
  return static_cast<double>(a) / b;
}

inline NCORE_HOST_DEVICE double operator+(double a, Float8_e5m2 b) {
  return a + static_cast<double>(b);
}
inline NCORE_HOST_DEVICE double operator-(double a, Float8_e5m2 b) {
  return a - static_cast<double>(b);
}
inline NCORE_HOST_DEVICE double operator*(double a, Float8_e5m2 b) {
  return a * static_cast<double>(b);
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE double operator/(double a, Float8_e5m2 b) {
  return a / static_cast<double>(b);
}

/// @}

/// @name Mixed-type Arithmetic (Float8_e5m2 & int)
/// @{

inline NCORE_HOST_DEVICE Float8_e5m2 operator+(Float8_e5m2 a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a + static_cast<Float8_e5m2>(b);
}
inline NCORE_HOST_DEVICE Float8_e5m2 operator-(Float8_e5m2 a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a - static_cast<Float8_e5m2>(b);
}
inline NCORE_HOST_DEVICE Float8_e5m2 operator*(Float8_e5m2 a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a * static_cast<Float8_e5m2>(b);
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float8_e5m2 operator/(Float8_e5m2 a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a / static_cast<Float8_e5m2>(b);
}

inline NCORE_HOST_DEVICE Float8_e5m2 operator+(int a, Float8_e5m2 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<Float8_e5m2>(a) + b;
}
inline NCORE_HOST_DEVICE Float8_e5m2 operator-(int a, Float8_e5m2 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<Float8_e5m2>(a) - b;
}
inline NCORE_HOST_DEVICE Float8_e5m2 operator*(int a, Float8_e5m2 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<Float8_e5m2>(a) * b;
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float8_e5m2 operator/(int a, Float8_e5m2 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<Float8_e5m2>(a) / b;
}

/// @}

/// @name Mixed-type Arithmetic (Float8_e5m2 & int64_t)
/// @{

inline NCORE_HOST_DEVICE Float8_e5m2 operator+(Float8_e5m2 a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a + static_cast<Float8_e5m2>(b);
}
inline NCORE_HOST_DEVICE Float8_e5m2 operator-(Float8_e5m2 a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a - static_cast<Float8_e5m2>(b);
}
inline NCORE_HOST_DEVICE Float8_e5m2 operator*(Float8_e5m2 a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a * static_cast<Float8_e5m2>(b);
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float8_e5m2 operator/(Float8_e5m2 a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a / static_cast<Float8_e5m2>(b);
}

inline NCORE_HOST_DEVICE Float8_e5m2 operator+(int64_t a, Float8_e5m2 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<Float8_e5m2>(a) + b;
}
inline NCORE_HOST_DEVICE Float8_e5m2 operator-(int64_t a, Float8_e5m2 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<Float8_e5m2>(a) - b;
}
inline NCORE_HOST_DEVICE Float8_e5m2 operator*(int64_t a, Float8_e5m2 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<Float8_e5m2>(a) * b;
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float8_e5m2 operator/(int64_t a, Float8_e5m2 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<Float8_e5m2>(a) / b;
}

/// @}

/// @note Comparison operators are not defined here; rely on the implicit
/// conversion from ncore::dtypes::Float8_e5m2 to float.

} // namespace ncore::dtypes

// ============================================================
// std::numeric_limits specialisation
// ============================================================
namespace std {

/**
 * @class numeric_limits<ncore::dtypes::Float8_e5m2>
 * @brief Specialization of std::numeric_limits for the custom @ref
 * ncore::dtypes::Float8_e5m2 type.
 */
template <> class numeric_limits<ncore::dtypes::Float8_e5m2> {
  using Float8_e5m2 = ncore::dtypes::Float8_e5m2;

public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = false;
  static constexpr auto has_denorm = true;
  static constexpr auto has_denorm_loss = true;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 3;
  static constexpr int digits10 = 0;
  static constexpr int max_digits10 = 2;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before;

  /**
   * @brief Smallest positive normalized value.
   * @details 0x04 → 2**(-14).
   */
  static constexpr Float8_e5m2 min() { return {0x4, Float8_e5m2::from_bits()}; }

  /**
   * @brief Largest finite positive value.
   * @details 0x7B → 57344.
   */
  static constexpr Float8_e5m2 max() {
    return {0x7B, Float8_e5m2::from_bits()};
  }

  /**
   * @brief Largest finite negative value.
   * @details 0xFB → -57344.
   */
  static constexpr Float8_e5m2 lowest() {
    return {0xFB, Float8_e5m2::from_bits()};
  }

  /**
   * @brief Machine epsilon.
   * @details 0x34 → 2**(-2).
   */
  static constexpr Float8_e5m2 epsilon() {
    return {0x34, Float8_e5m2::from_bits()};
  }

  /**
   * @brief Maximum rounding error.
   * @details 0x38 → 0.5.
   */
  static constexpr Float8_e5m2 round_error() {
    return {0x38, Float8_e5m2::from_bits()};
  }

  /**
   * @brief Positive infinity.
   * @details 0x7C.
   */
  static constexpr Float8_e5m2 infinity() {
    return {0x7C, Float8_e5m2::from_bits()};
  }

  /**
   * @brief Quiet NaN.
   * @details 0x7F.
   */
  static constexpr Float8_e5m2 quiet_NaN() {
    return {0x7F, Float8_e5m2::from_bits()};
  }

  /**
   * @brief Smallest positive subnormal value.
   * @details 0x01 → 2**(-16).
   */
  static constexpr Float8_e5m2 denorm_min() {
    return {0x01, Float8_e5m2::from_bits()};
  }
};

} // namespace std

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#ifdef _GNUC_CLANG_
#pragma GCC diagnostic pop
#endif
