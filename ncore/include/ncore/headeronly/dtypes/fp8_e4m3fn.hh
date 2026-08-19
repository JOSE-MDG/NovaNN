/**
 * @file fp8_e4m3fn.hh
 * @brief 8-bit floating-point (FP8 E4M3FN) data type implementation.
 *
 * @details
 * Defines the @ref Float8_e4m3fn type: an 8-bit floating-point format with
 * 1 sign bit, 4 exponent bits, 3 mantissa bits and bias = 7. The "fn" suffix
 * denotes "finite" — this format has no representation for infinity and
 * reserves the all-ones-exponent/all-ones-mantissa pattern exclusively for
 * NaN. This includes conversions to and from standard C++ types (float,
 * double, int, etc.) and basic arithmetic operations.
 *
 * Arithmetic operations are implemented by converting to float32 and
 * performing the operation there, as most operations are memory-bound.
 *
 * Implementation based on https://arxiv.org/pdf/2209.05433.pdf and modeled
 * after this project's own @ref ncore::dtypes::Half conversion helpers.
 *
 * @see half.hh  Shared fp32_from_bits/fp32_to_bits bit-cast helpers.
 */

#pragma once

#include <bit>
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
 * @struct Float8_e4m3fn
 * @brief Representation of an 8-bit floating-point number in E4M3FN format.
 *
 * @details
 * Binary layout, MSB to LSB: @c s eeee mmm
 *  @li 1 sign bit
 *  @li 4 exponent bits (bias = 7)
 *  @li 3 mantissa bits
 *
 * This format has no infinities: the maximum finite magnitude is 448, and
 * the bit pattern that would otherwise encode +/-inf instead saturates to
 * the maximum finite value or resolves to NaN, depending on the exact
 * pattern (see @ref isnan / @ref isinf).
 */
struct alignas(1) Float8_e4m3fn {
  uint8_t x; ///< The 8-bit binary representation of the FP8 value.

  /**
   * @struct from_bits_t
   * @brief Tag type used to construct a @ref Float8_e4m3fn directly from its
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
  Float8_e4m3fn() = default;

  /**
   * @brief Constructs a @ref Float8_e4m3fn directly from a raw 8-bit integer
   * representation.
   * @param[in] bits The raw 8-bit binary representation.
   * @param[in] unused Tag parameter to disambiguate from float conversions.
   */
  constexpr NCORE_HOST_DEVICE Float8_e4m3fn(uint8_t bits,
                                            from_bits_t /*unused*/) noexcept
      : x(bits) {}

  /**
   * @brief Implicit constructor from a single-precision float.
   * @param[in] value The float value to convert.
   */
  inline NCORE_HOST_DEVICE Float8_e4m3fn(float value);

  /**
   * @brief Implicit conversion operator to single-precision float.
   * @return The float representation of the fp8 value.
   */
  inline NCORE_HOST_DEVICE operator float() const;

  /**
   * @brief Checks whether this value encodes NaN.
   * @return @c true if this is a NaN pattern.
   */
  [[nodiscard]] inline NCORE_HOST_DEVICE bool isnan() const;

  /**
   * @brief Checks whether this value encodes infinity.
   * @details E4M3FN has no infinity representation; always returns @c false.
   * @return Always @c false.
   */
  [[nodiscard]] inline NCORE_HOST_DEVICE bool isinf() const;
};

/**
 * @brief Stream output operator for @ref Float8_e4m3fn.
 * @details Promotes the fp8 value to float before writing to the stream.
 * @param[in,out] out The output stream.
 * @param[in]     value The @ref Float8_e4m3fn value to write.
 * @return Reference to the output stream.
 */
inline std::ostream &operator<<(std::ostream &out, const Float8_e4m3fn &value) {
  out << static_cast<float>(value);
  return out;
}

/**
 * @namespace ncore::dtypes::detail
 * @brief Internal low-level bit-manipulation and conversion helpers.
 */
namespace detail {

/**
 * @brief Converts an 8-bit E4M3FN bit pattern to a 32-bit IEEE
 * single-precision float value.
 * @details The implementation is purely integer-based; no floating-point
 * exceptions are triggered by intermediate steps.
 * @param[in] input The 8-bit E4M3FN bit pattern.
 * @return The single-precision float value.
 */
NCORE_HOST_DEVICE inline float fp8e4m3fn_to_fp32_value(uint8_t input) {
  /*
   * Extend the fp8 E4M3FN number to 32 bits and shift to the upper part of
   * the 32-bit word:
   *      +---+----+---+-----------------------------+
   *      | S |EEEE|MMM|0000 0000 0000 0000 0000 0000|
   *      +---+----+---+-----------------------------+
   * Bits  31 27-30 24-26          0-23
   *
   * S - sign bit, E - biased exponent bits, M - mantissa bits, 0 - zero bits.
   */
  const uint32_t w = static_cast<uint32_t>(input) << 24;
  /*
   * Extract the sign into the high bit of the 32-bit word.
   */
  const uint32_t sign = w & UINT32_C(0x80000000);
  /*
   * Strip the sign to get the unsigned magnitude (exponent + mantissa).
   */
  const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
  /*
   * renormShift: number of bits to shift mantissa left to normalize a
   * denormal. For a normalized input, one of the high 5 bits (sign == 0 plus
   * 4-bit exponent) is set, so renormShift == 0. For denormals renormShift
   * > 0; shifting the mantissa left by that amount moves the unit bit into
   * the exponent field, turning the biased exponent into 1 and yielding a
   * normalized mantissa (implicit leading 1 removed).
   *
   * This function is NCORE_HOST_DEVICE, so it is compiled for both the host
   * and the CUDA/HIP device pass. @c std::countl_zero is only used on host:
   * NovaNN's compiler floor (GCC 14+ / Clang 17+, see
   * CheckCompilerVersion.cmake — MSVC is rejected outright) guarantees a
   * real <bit> implementation there. On device, nvcc/hipcc are not
   * guaranteed to lower <bit> through to a device-valid intrinsic, so the
   * compiler-provided __clz/__builtin_clz is used instead, matching the
   * dispatch NovaNN's own CUDA/HIP backends rely on elsewhere.
   */
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  uint32_t renormShift = static_cast<uint32_t>(__clz(nonsign));
#elif defined(_MSC_VER) && !defined(__clang__)
  unsigned long nonsignBsr;
  _BitScanReverse(&nonsignBsr, static_cast<unsigned long>(nonsign));
  uint32_t renormShift = static_cast<uint32_t>(nonsignBsr) ^ 31;
#else
  uint32_t renormShift =
      static_cast<uint32_t>(std::countl_zero<uint32_t>(nonsign));
#endif
  renormShift = renormShift > 4 ? renormShift - 4 : 0;
  /*
   * If the fp8 exponent and mantissa bits are all set to 1 (NaN pattern),
   * the addition below overflows into bit 31, and the subsequent shift
   * fills the high 9 bits with 1s -> infNanMask == 0x7F800000. Otherwise
   * infNanMask == 0x00000000.
   */
  const int32_t infNanMask =
      (static_cast<int32_t>(nonsign + 0x01000000) >> 8) & INT32_C(0x7F800000);
  /*
   * If nonsign == 0 (+/-0.0), nonsign - 1 underflows to 0xFFFFFFFF, setting
   * bit 31, and the arithmetic right shift by 31 broadcasts that bit across
   * zeroMask. Otherwise zeroMask == 0x00000000.
   */
  const int32_t zeroMask = static_cast<int32_t>(nonsign - 1) >> 31;
  /*
   * Steps performed in one expression:
   *  1. Shift nonsign left by renormShift (normalize denormals).
   *  2. Shift right by 4 (expand 4-bit exponent to 8-bit; 3-bit mantissa
   *     moves into the high 3 bits of the 23-bit FP32 mantissa).
   *  3. Add 0x78 << 23 to the exponent to compensate the bias difference
   *     (0x7F for FP32 minus 0x07 for E4M3FN = 0x78), combined with step 4.
   *  4. Subtract renormShift from the exponent to account for renorm.
   *  5. OR with infNanMask to force 0xFF exponent for NaN.
   *  6. ANDNOT with zeroMask to zero mantissa/exponent for +/-0.
   *  7. OR with sign.
   */
  const uint32_t result =
      sign | ((((nonsign << renormShift >> 4) + ((0x78 - renormShift) << 23)) |
               static_cast<uint32_t>(infNanMask)) &
              ~static_cast<uint32_t>(zeroMask));
  return ncore::dtypes::detail::fp32_from_bits(result);
}

/**
 * @brief Converts a 32-bit IEEE single-precision float to an 8-bit E4M3FN
 * bit pattern.
 * @param[in] f The single-precision float value to convert.
 * @return The 8-bit E4M3FN bit pattern.
 */
NCORE_HOST_DEVICE inline uint8_t fp8e4m3fn_from_fp32_value(float f) {
  /*
   * Binary representation of 480.0f, the first value not representable in
   * E4M3FN range:
   *   0 1111 111 - fp8 E4M3FN
   *   0 10000111 1110...0 - fp32
   */
  constexpr uint32_t fp8Max = UINT32_C(1087) << 20;

  /*
   * Mask for converting fp32 numbers below E4M3FN's normal range into
   * denormal representation. Magic number: (127 - 7) + (23 - 3) + 1.
   */
  constexpr uint32_t denormMask = UINT32_C(141) << 23;

  uint32_t fBits = ncore::dtypes::detail::fp32_to_bits(f);
  uint8_t result = 0;

  const uint32_t sign = fBits & UINT32_C(0x80000000);
  fBits ^= sign;

  if (fBits >= fp8Max) {
    if (fBits > UINT32_C(0x7F800000)) {
      // NaN input -> NaN output.
      result = 0x7F;
    } else {
      // Finite overflow or +/-inf -> saturate to the max finite value.
      result = 0x7E;
    }
  } else {
    if (fBits < (UINT32_C(121) << 23)) {
      // Input smaller than 2^(-6), the smallest E4M3FN normal number.
      fBits = ncore::dtypes::detail::fp32_to_bits(
          ncore::dtypes::detail::fp32_from_bits(fBits) +
          ncore::dtypes::detail::fp32_from_bits(denormMask));
      result = static_cast<uint8_t>(fBits - denormMask);
    } else {
      // Resulting mantissa is odd.
      const uint8_t mantOdd = static_cast<uint8_t>((fBits >> 20) & 1);

      // Update exponent, rounding bias part 1.
      fBits += (static_cast<uint32_t>(7 - 127) << 23) + 0x7FFFF;

      // Rounding bias part 2.
      fBits += mantOdd;

      // Take the bits!
      result = static_cast<uint8_t>(fBits >> 20);

      // Rounding may carry into the NaN bit pattern (0x7F); saturate.
      if (result == 0x7F) {
        result = 0x7E;
      }
    }
  }

  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}

} // namespace detail

// ============================================================
// Constructors and conversion operators — inline definitions
// ============================================================

inline NCORE_HOST_DEVICE Float8_e4m3fn::Float8_e4m3fn(float value)
    : x(detail::fp8e4m3fn_from_fp32_value(value)) {}

inline NCORE_HOST_DEVICE Float8_e4m3fn::operator float() const {
  return detail::fp8e4m3fn_to_fp32_value(x);
}

inline NCORE_HOST_DEVICE bool Float8_e4m3fn::isnan() const {
  return (x & 0b0111'1111) == 0b0111'1111;
}

inline NCORE_HOST_DEVICE bool Float8_e4m3fn::isinf() const {
  // E4M3FN has no infinity representation.
  return false;
}

// ============================================================
// Arithmetic operators
// ============================================================

/// @name Arithmetic Operators (Float8_e4m3fn & Float8_e4m3fn)
/// @{

/**
 * @brief Addition operator for two @ref Float8_e4m3fn values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The sum as a @ref Float8_e4m3fn.
 */
inline NCORE_HOST_DEVICE Float8_e4m3fn operator+(const Float8_e4m3fn &a,
                                                 const Float8_e4m3fn &b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

/**
 * @brief Subtraction operator for two @ref Float8_e4m3fn values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The difference as a @ref Float8_e4m3fn.
 */
inline NCORE_HOST_DEVICE Float8_e4m3fn operator-(const Float8_e4m3fn &a,
                                                 const Float8_e4m3fn &b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

/**
 * @brief Multiplication operator for two @ref Float8_e4m3fn values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The product as a @ref Float8_e4m3fn.
 */
inline NCORE_HOST_DEVICE Float8_e4m3fn operator*(const Float8_e4m3fn &a,
                                                 const Float8_e4m3fn &b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

/**
 * @brief Division operator for two @ref Float8_e4m3fn values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The quotient as a @ref Float8_e4m3fn.
 */
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float8_e4m3fn operator/(const Float8_e4m3fn &a,
                                                 const Float8_e4m3fn &b) {
  return static_cast<float>(a) / static_cast<float>(b);
}

/**
 * @brief Unary minus operator for @ref Float8_e4m3fn.
 * @param[in] a The operand.
 * @return The negated value as a @ref Float8_e4m3fn.
 */
inline NCORE_HOST_DEVICE Float8_e4m3fn operator-(const Float8_e4m3fn &a) {
  return -static_cast<float>(a);
}

/**
 * @brief Addition assignment operator.
 * @param[in,out] a The destination operand.
 * @param[in]     b The operand to add.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE Float8_e4m3fn &operator+=(Float8_e4m3fn &a,
                                                   const Float8_e4m3fn &b) {
  a = a + b;
  return a;
}

/**
 * @brief Subtraction assignment operator.
 * @param[in,out] a The destination operand.
 * @param[in]     b The operand to subtract.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE Float8_e4m3fn &operator-=(Float8_e4m3fn &a,
                                                   const Float8_e4m3fn &b) {
  a = a - b;
  return a;
}

/**
 * @brief Multiplication assignment operator.
 * @param[in,out] a The destination operand.
 * @param[in]     b The operand to multiply.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE Float8_e4m3fn &operator*=(Float8_e4m3fn &a,
                                                   const Float8_e4m3fn &b) {
  a = a * b;
  return a;
}

/**
 * @brief Division assignment operator.
 * @param[in,out] a The destination operand.
 * @param[in]     b The divisor.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE Float8_e4m3fn &operator/=(Float8_e4m3fn &a,
                                                   const Float8_e4m3fn &b) {
  a = a / b;
  return a;
}

/// @}

/// @name Mixed-type Arithmetic (Float8_e4m3fn & float)
/// @{

inline NCORE_HOST_DEVICE float operator+(Float8_e4m3fn a, float b) {
  return static_cast<float>(a) + b;
}
inline NCORE_HOST_DEVICE float operator-(Float8_e4m3fn a, float b) {
  return static_cast<float>(a) - b;
}
inline NCORE_HOST_DEVICE float operator*(Float8_e4m3fn a, float b) {
  return static_cast<float>(a) * b;
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE float operator/(Float8_e4m3fn a, float b) {
  return static_cast<float>(a) / b;
}

inline NCORE_HOST_DEVICE float operator+(float a, Float8_e4m3fn b) {
  return a + static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float operator-(float a, Float8_e4m3fn b) {
  return a - static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float operator*(float a, Float8_e4m3fn b) {
  return a * static_cast<float>(b);
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE float operator/(float a, Float8_e4m3fn b) {
  return a / static_cast<float>(b);
}

inline NCORE_HOST_DEVICE float &operator+=(float &a, const Float8_e4m3fn &b) {
  return a += static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float &operator-=(float &a, const Float8_e4m3fn &b) {
  return a -= static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float &operator*=(float &a, const Float8_e4m3fn &b) {
  return a *= static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float &operator/=(float &a, const Float8_e4m3fn &b) {
  return a /= static_cast<float>(b);
}

/// @}

/// @name Mixed-type Arithmetic (Float8_e4m3fn & double)
/// @{

inline NCORE_HOST_DEVICE double operator+(Float8_e4m3fn a, double b) {
  return static_cast<double>(a) + b;
}
inline NCORE_HOST_DEVICE double operator-(Float8_e4m3fn a, double b) {
  return static_cast<double>(a) - b;
}
inline NCORE_HOST_DEVICE double operator*(Float8_e4m3fn a, double b) {
  return static_cast<double>(a) * b;
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE double operator/(Float8_e4m3fn a, double b) {
  return static_cast<double>(a) / b;
}

inline NCORE_HOST_DEVICE double operator+(double a, Float8_e4m3fn b) {
  return a + static_cast<double>(b);
}
inline NCORE_HOST_DEVICE double operator-(double a, Float8_e4m3fn b) {
  return a - static_cast<double>(b);
}
inline NCORE_HOST_DEVICE double operator*(double a, Float8_e4m3fn b) {
  return a * static_cast<double>(b);
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE double operator/(double a, Float8_e4m3fn b) {
  return a / static_cast<double>(b);
}

/// @}

/// @name Mixed-type Arithmetic (Float8_e4m3fn & int)
/// @{

inline NCORE_HOST_DEVICE Float8_e4m3fn operator+(Float8_e4m3fn a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a + static_cast<Float8_e4m3fn>(b);
}
inline NCORE_HOST_DEVICE Float8_e4m3fn operator-(Float8_e4m3fn a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a - static_cast<Float8_e4m3fn>(b);
}
inline NCORE_HOST_DEVICE Float8_e4m3fn operator*(Float8_e4m3fn a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a * static_cast<Float8_e4m3fn>(b);
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float8_e4m3fn operator/(Float8_e4m3fn a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a / static_cast<Float8_e4m3fn>(b);
}

inline NCORE_HOST_DEVICE Float8_e4m3fn operator+(int a, Float8_e4m3fn b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Float8_e4m3fn>(a) + b;
}
inline NCORE_HOST_DEVICE Float8_e4m3fn operator-(int a, Float8_e4m3fn b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Float8_e4m3fn>(a) - b;
}
inline NCORE_HOST_DEVICE Float8_e4m3fn operator*(int a, Float8_e4m3fn b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Float8_e4m3fn>(a) * b;
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float8_e4m3fn operator/(int a, Float8_e4m3fn b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Float8_e4m3fn>(a) / b;
}

/// @}

/// @name Mixed-type Arithmetic (Float8_e4m3fn & int64_t)
/// @{

inline NCORE_HOST_DEVICE Float8_e4m3fn operator+(Float8_e4m3fn a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a + static_cast<Float8_e4m3fn>(b);
}
inline NCORE_HOST_DEVICE Float8_e4m3fn operator-(Float8_e4m3fn a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a - static_cast<Float8_e4m3fn>(b);
}
inline NCORE_HOST_DEVICE Float8_e4m3fn operator*(Float8_e4m3fn a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a * static_cast<Float8_e4m3fn>(b);
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float8_e4m3fn operator/(Float8_e4m3fn a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a / static_cast<Float8_e4m3fn>(b);
}

inline NCORE_HOST_DEVICE Float8_e4m3fn operator+(int64_t a, Float8_e4m3fn b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Float8_e4m3fn>(a) + b;
}
inline NCORE_HOST_DEVICE Float8_e4m3fn operator-(int64_t a, Float8_e4m3fn b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Float8_e4m3fn>(a) - b;
}
inline NCORE_HOST_DEVICE Float8_e4m3fn operator*(int64_t a, Float8_e4m3fn b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Float8_e4m3fn>(a) * b;
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Float8_e4m3fn operator/(int64_t a, Float8_e4m3fn b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Float8_e4m3fn>(a) / b;
}

/// @}

/// @note Comparison operators are not defined here; rely on the implicit
/// conversion from ncore::dtypes::Float8_e4m3fn to float.

} // namespace ncore::dtypes

// ============================================================
// std::numeric_limits specialization
// ============================================================
namespace std {

/**
 * @class numeric_limits<ncore::dtypes::Float8_e4m3fn>
 * @brief Specialization of std::numeric_limits for the custom @ref
 * ncore::dtypes::Float8_e4m3fn type.
 */
template <> class numeric_limits<ncore::dtypes::Float8_e4m3fn> {
  using Float8_e4m3fn = ncore::dtypes::Float8_e4m3fn;

public:
  static constexpr bool is_specialized = true; ///< E4M3FN has numeric limits.
  static constexpr bool is_signed = true;      ///< E4M3FN is signed.
  static constexpr bool is_integer = false;    ///< E4M3FN is floating-point.
  static constexpr bool is_exact =
      false; ///< E4M3FN representation is not exact.
  static constexpr bool has_infinity =
      false; ///< E4M3FN has no infinity representation.
  static constexpr bool has_quiet_NaN =
      true; ///< E4M3FN has quiet NaN representation.
  static constexpr bool has_signaling_NaN =
      false; ///< E4M3FN has no signaling NaN representation.
  static constexpr auto has_denorm = true; ///< E4M3FN supports subnormals.
  static constexpr auto has_denorm_loss =
      true; ///< Denormalization loss is detected.
  static constexpr auto round_style =
      numeric_limits<float>::round_style; ///< Inherits float32 rounding style.
  static constexpr bool is_iec559 =
      false; ///< Does not conform to IEC 60559 (IEEE 754).
  static constexpr bool is_bounded = true; ///< Values are bounded.
  static constexpr bool is_modulo =
      false;                       ///< E4M3FN arithmetic does not modulo wrap.
  static constexpr int digits = 4; ///< Mantissa bits + 1 implicit bit.
  static constexpr int digits10 =
      0; ///< No decimal digit is reliably preserved.
  static constexpr int max_digits10 =
      3; ///< Decimal digits required to uniquely represent values.
  static constexpr int radix = 2; ///< Base of the exponent.
  static constexpr int min_exponent =
      -5; ///< Minimum negative power of 2 for a normal value.
  static constexpr int min_exponent10 = -1; ///< Minimum negative power of 10.
  static constexpr int max_exponent = 8;    ///< Maximum positive power of 2.
  static constexpr int max_exponent10 = 2;  ///< Maximum positive power of 10.
  static constexpr auto traps =
      numeric_limits<float>::traps; ///< Inherits float32 trap behaviour.
  static constexpr auto tinyness_before =
      false; ///< Tinyness is not tested before rounding.

  /**
   * @brief Smallest positive normalized value.
   * @details 0x08 → @f$2^{-6}@f$.
   */
  static constexpr Float8_e4m3fn min() {
    return {0x08, Float8_e4m3fn::from_bits()};
  }

  /**
   * @brief Largest finite negative value.
   * @details 0xFE → -448.
   */
  static constexpr Float8_e4m3fn lowest() {
    return {0xFE, Float8_e4m3fn::from_bits()};
  }

  /**
   * @brief Largest finite positive value.
   * @details 0x7E → 448.
   */
  static constexpr Float8_e4m3fn max() {
    return {0x7E, Float8_e4m3fn::from_bits()};
  }

  /**
   * @brief Machine epsilon.
   * @details 0x20 → @f$2^{-3}@f$.
   */
  static constexpr Float8_e4m3fn epsilon() {
    return {0x20, Float8_e4m3fn::from_bits()};
  }

  /**
   * @brief Maximum rounding error.
   * @details 0x30 → 0.5.
   */
  static constexpr Float8_e4m3fn round_error() {
    return {0x30, Float8_e4m3fn::from_bits()};
  }

  /**
   * @brief Quiet NaN.
   * @details 0x7F.
   */
  static constexpr Float8_e4m3fn quiet_NaN() {
    return {0x7F, Float8_e4m3fn::from_bits()};
  }

  /**
   * @brief Smallest positive subnormal value.
   * @details 0x01 → @f$2^{-9}@f$.
   */
  static constexpr Float8_e4m3fn denorm_min() {
    return {0x01, Float8_e4m3fn::from_bits()};
  }
};

} // namespace std

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif

#ifdef _GNUC_CLANG_
#pragma GCC diagnostic pop
#endif
