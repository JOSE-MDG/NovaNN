/**
 * @file bfloat16.hh
 * @brief Brain floating-point (BF16) data type implementation.
 *
 * @details
 * Defines the @ref BFloat16 type representing a 16-bit floating-point number
 * using 1 sign bit, 8 exponent bits and 7 mantissa bits (i.e. the same
 * exponent range as float32, truncated mantissa). This includes conversions
 * to and from standard C++ types (float, double, int, etc.), basic
 * arithmetic operations, and support for GPU compilation environments
 * (CUDA/HIP).
 *
 * Arithmetic operations are implemented by promoting to float32 rather than
 * using CUDA/HIP bf16 intrinsics directly on host, as most operations are
 * memory-bound.
 *
 * @see macros.h  NovaNN compiler and platform macros.
 * @see half.hh   IEEE 754 half-precision counterpart.
 */

#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ostream>

#include <config.h>
#include <ncore/headeronly/macros.h>

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

#if defined(__CUDACC__) && !defined(NOVA_HAS_HIP)
#include <cuda_bf16.h>
#endif

/**
 * @namespace ncore::dtypes
 * @brief Custom data types for operations.
 */
namespace ncore::dtypes {

/**
 * @struct BFloat16
 * @brief Representation of a "brain" 16-bit floating-point number (BF16).
 *
 * @details
 * Binary layout, MSB to LSB: @c s eeeeeeee mmmmmmm
 *  @li 1 sign bit
 *  @li 8 exponent bits (same bias/range as IEEE 754 float32: bias = 127)
 *  @li 7 mantissa bits
 *
 * The struct is aligned to 2 bytes to match the storage size of a 16-bit
 * word. It contains a single data member representing the bitwise storage.
 */
struct alignas(2) BFloat16 {
  unsigned short x; ///< The 16-bit binary representation of the BF16 value.

  /**
   * @struct from_bits_t
   * @brief Tag type used to construct a @ref BFloat16 directly from its raw
   * 16-bit representation.
   */
  struct from_bits_t {};

  /**
   * @brief Returns a tag instance of @ref from_bits_t.
   * @return A default-constructed @ref from_bits_t.
   */
  NCORE_HOST_DEVICE static constexpr from_bits_t from_bits() { return {}; }

#ifdef NOVA_HAS_HIP
  /**
   * @brief Default constructor. Left uninitialized for performance.
   */
  NCORE_HOST_DEVICE BFloat16() = default;
#else
  /**
   * @brief Default constructor. Left uninitialized for performance.
   */
  BFloat16() = default;
#endif

  /**
   * @brief Constructs a @ref BFloat16 directly from a raw 16-bit integer
   * representation.
   * @param[in] bits The raw 16-bit binary representation.
   * @param[in] unused Tag parameter to disambiguate from float conversions.
   */
  constexpr NCORE_HOST_DEVICE BFloat16(unsigned short bits,
                                       from_bits_t /*unused*/) noexcept
      : x(bits) {}

  /**
   * @brief Implicit constructor from a single-precision float.
   * @param[in] value The float value to convert.
   */
  inline NCORE_HOST_DEVICE BFloat16(float value);

  /**
   * @brief Implicit conversion operator to single-precision float.
   * @return The float representation of the bf16 value.
   */
  inline NCORE_HOST_DEVICE operator float() const;

#if defined(__CUDACC__) && !defined(NOVA_HAS_HIP)
  /**
   * @brief Constructor from a native CUDA @c __nv_bfloat16 representation.
   * @param[in] value The native __nv_bfloat16 value.
   */
  inline NCORE_HOST_DEVICE BFloat16(const __nv_bfloat16 &value);

  /**
   * @brief Implicit conversion operator to native CUDA @c __nv_bfloat16.
   * @return The native __nv_bfloat16 representation.
   */
  explicit inline NCORE_HOST_DEVICE operator __nv_bfloat16() const;
#endif
};

/**
 * @brief Stream output operator for @ref BFloat16.
 * @details Promotes the bf16 value to float before writing to the stream.
 * @param[in,out] out The output stream.
 * @param[in]     value The @ref BFloat16 value to write.
 * @return Reference to the output stream.
 */
inline std::ostream &operator<<(std::ostream &out, const BFloat16 &value) {
  out << static_cast<float>(value);
  return out;
}

/**
 * @namespace ncore::dtypes::detail
 * @brief Internal low-level bit-manipulation and conversion helpers.
 */
namespace detail {

/**
 * @brief Reinterpret the top 16 bits of a @c uint16_t bf16 pattern as a
 * 32-bit float.
 * @details
 * Since BF16 shares its exponent width and bias with float32, conversion is
 * a pure bit-shift with zero-extended mantissa; no special-case handling of
 * subnormals, infinities or NaNs is required.
 * @param[in] src The 16-bit bf16 bit pattern.
 * @return The single-precision float value.
 */
NCORE_HOST_DEVICE inline float f32_from_bits(uint16_t src) {
  float res = 0;
  uint32_t tmp = src;
  tmp <<= 16;

#if defined(NOVA_HAS_HIP) && defined(__HIPCC__)
  float *tempRes;

  // We should be using memcpy in order to respect the strict aliasing rule
  // but it fails in the HIP environment.
  tempRes = reinterpret_cast<float *>(&tmp);
  res = *tempRes;
#else
  std::memcpy(&res, &tmp, sizeof(tmp));
#endif

  return res;
}

/**
 * @brief Truncates a 32-bit float bit pattern down to its high 16 bits
 * (bf16), without rounding.
 * @param[in] src The single-precision float.
 * @return The truncated 16-bit bf16 bit pattern.
 */
NCORE_HOST_DEVICE inline uint16_t bits_from_f32(float src) {
  uint32_t res = 0;

#if defined(NOVA_HAS_HIP) && defined(__HIPCC__)
  // We should be using memcpy in order to respect the strict aliasing rule
  // but it fails in the HIP environment.
  uint32_t *tempRes = reinterpret_cast<uint32_t *>(&src);
  res = *tempRes;
#else
  std::memcpy(&res, &src, sizeof(res));
#endif

  return res >> 16;
}

/**
 * @brief Converts a 32-bit float to a 16-bit bf16 bit pattern using
 * round-to-nearest-even.
 * @param[in] src The single-precision float value to convert.
 * @return The 16-bit bf16 bit pattern.
 */
NCORE_HOST_DEVICE inline uint16_t round_to_nearest_even(float src) {
#if defined(NOVA_HAS_HIP) && defined(__HIPCC__)
  if (src != src) {
#elif defined(_MSC_VER) && !defined(__clang__)
  if (isnan(src)) {
#else
  if (std::isnan(src)) {
#endif
    return UINT16_C(0x7FC0);
  } else {
    const uint32_t u32 = std::bit_cast<uint32_t>(src);
    uint32_t roundingBias = ((u32 >> 16) & 1) + UINT32_C(0x7FFF);
    return static_cast<uint16_t>((u32 + roundingBias) >> 16);
  }
}

} // namespace detail

// ============================================================
// Constructors and conversion operators — inline definitions
// ============================================================

inline NCORE_HOST_DEVICE BFloat16::BFloat16(float value)
    :
#if !defined(NOVA_HAS_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
      x(__bfloat16_as_ushort(__float2bfloat16(value)))
#else
      // RNE (round to nearest even) by default.
      x(detail::round_to_nearest_even(value))
#endif
{
}

inline NCORE_HOST_DEVICE BFloat16::operator float() const {
#if defined(__CUDACC__) && !defined(NOVA_HAS_HIP)
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&x));
#else
  return detail::f32_from_bits(x);
#endif
}

#if defined(__CUDACC__) && !defined(NOVA_HAS_HIP)
inline NCORE_HOST_DEVICE BFloat16::BFloat16(const __nv_bfloat16 &value) {
  x = *reinterpret_cast<const unsigned short *>(&value);
}
inline NCORE_HOST_DEVICE BFloat16::operator __nv_bfloat16() const {
  return *reinterpret_cast<const __nv_bfloat16 *>(&x);
}
#endif

// ============================================================
// CUDA __ldg helper
// ============================================================
#if defined(__CUDACC__) || defined(__HIPCC__)
/**
 * @brief Load a @ref BFloat16 value from global memory using the CUDA
 * @c __ldg intrinsic.
 * @param[in] ptr Pointer to global memory.
 * @return The loaded @ref BFloat16 value.
 */
inline NCORE_DEVICE BFloat16 __ldg(const BFloat16 *ptr) {
#if !defined(NOVA_HAS_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __ldg(reinterpret_cast<const __nv_bfloat16 *>(ptr));
#else
  return *ptr;
#endif
}
#endif

// ============================================================
// Arithmetic operators
// ============================================================

/// @name Arithmetic Operators (BFloat16 & BFloat16)
/// @{

/**
 * @brief Addition operator for two @ref BFloat16 values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The sum as a @ref BFloat16.
 */
inline NCORE_HOST_DEVICE BFloat16 operator+(const BFloat16 &a,
                                            const BFloat16 &b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

/**
 * @brief Subtraction operator for two @ref BFloat16 values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The difference as a @ref BFloat16.
 */
inline NCORE_HOST_DEVICE BFloat16 operator-(const BFloat16 &a,
                                            const BFloat16 &b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

/**
 * @brief Multiplication operator for two @ref BFloat16 values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The product as a @ref BFloat16.
 */
inline NCORE_HOST_DEVICE BFloat16 operator*(const BFloat16 &a,
                                            const BFloat16 &b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

/**
 * @brief Division operator for two @ref BFloat16 values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The quotient as a @ref BFloat16.
 */
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE BFloat16 operator/(const BFloat16 &a,
                                            const BFloat16 &b) {
  return static_cast<float>(a) / static_cast<float>(b);
}

/**
 * @brief Unary minus operator for @ref BFloat16.
 * @param[in] a The operand.
 * @return The negated value as a @ref BFloat16.
 */
inline NCORE_HOST_DEVICE BFloat16 operator-(const BFloat16 &a) {
  return -static_cast<float>(a);
}

/**
 * @brief Addition assignment operator.
 * @param[in,out] a The destination operand.
 * @param[in]     b The operand to add.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE BFloat16 &operator+=(BFloat16 &a, const BFloat16 &b) {
  a = a + b;
  return a;
}

/**
 * @brief Subtraction assignment operator.
 * @param[in,out] a The destination operand.
 * @param[in]     b The operand to subtract.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE BFloat16 &operator-=(BFloat16 &a, const BFloat16 &b) {
  a = a - b;
  return a;
}

/**
 * @brief Multiplication assignment operator.
 * @param[in,out] a The destination operand.
 * @param[in]     b The operand to multiply.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE BFloat16 &operator*=(BFloat16 &a, const BFloat16 &b) {
  a = a * b;
  return a;
}

/**
 * @brief Division assignment operator.
 * @param[in,out] a The destination operand.
 * @param[in]     b The divisor.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE BFloat16 &operator/=(BFloat16 &a, const BFloat16 &b) {
  a = a / b;
  return a;
}

/**
 * @brief Bitwise OR, applied directly to the raw 16-bit storage.
 * @param[in,out] a The destination operand (mutated in place).
 * @param[in]     b The operand to OR with.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE BFloat16 &operator|(BFloat16 &a, const BFloat16 &b) {
  a.x = a.x | b.x;
  return a;
}

/**
 * @brief Bitwise XOR, applied directly to the raw 16-bit storage.
 * @param[in,out] a The destination operand (mutated in place).
 * @param[in]     b The operand to XOR with.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE BFloat16 &operator^(BFloat16 &a, const BFloat16 &b) {
  a.x = a.x ^ b.x;
  return a;
}

/**
 * @brief Bitwise AND, applied directly to the raw 16-bit storage.
 * @param[in,out] a The destination operand (mutated in place).
 * @param[in]     b The operand to AND with.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE BFloat16 &operator&(BFloat16 &a, const BFloat16 &b) {
  a.x = a.x & b.x;
  return a;
}

/// @}

/// @name Mixed-type Arithmetic (BFloat16 & float)
/// @{

inline NCORE_HOST_DEVICE float operator+(BFloat16 a, float b) {
  return static_cast<float>(a) + b;
}
inline NCORE_HOST_DEVICE float operator-(BFloat16 a, float b) {
  return static_cast<float>(a) - b;
}
inline NCORE_HOST_DEVICE float operator*(BFloat16 a, float b) {
  return static_cast<float>(a) * b;
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE float operator/(BFloat16 a, float b) {
  return static_cast<float>(a) / b;
}

inline NCORE_HOST_DEVICE float operator+(float a, BFloat16 b) {
  return a + static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float operator-(float a, BFloat16 b) {
  return a - static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float operator*(float a, BFloat16 b) {
  return a * static_cast<float>(b);
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE float operator/(float a, BFloat16 b) {
  return a / static_cast<float>(b);
}

inline NCORE_HOST_DEVICE float &operator+=(float &a, const BFloat16 &b) {
  return a += static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float &operator-=(float &a, const BFloat16 &b) {
  return a -= static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float &operator*=(float &a, const BFloat16 &b) {
  return a *= static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float &operator/=(float &a, const BFloat16 &b) {
  return a /= static_cast<float>(b);
}

/// @}

/// @name Mixed-type Arithmetic (BFloat16 & double)
/// @{

inline NCORE_HOST_DEVICE double operator+(BFloat16 a, double b) {
  return static_cast<double>(a) + b;
}
inline NCORE_HOST_DEVICE double operator-(BFloat16 a, double b) {
  return static_cast<double>(a) - b;
}
inline NCORE_HOST_DEVICE double operator*(BFloat16 a, double b) {
  return static_cast<double>(a) * b;
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE double operator/(BFloat16 a, double b) {
  return static_cast<double>(a) / b;
}

inline NCORE_HOST_DEVICE double operator+(double a, BFloat16 b) {
  return a + static_cast<double>(b);
}
inline NCORE_HOST_DEVICE double operator-(double a, BFloat16 b) {
  return a - static_cast<double>(b);
}
inline NCORE_HOST_DEVICE double operator*(double a, BFloat16 b) {
  return a * static_cast<double>(b);
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE double operator/(double a, BFloat16 b) {
  return a / static_cast<double>(b);
}

/// @}

/// @name Mixed-type Arithmetic (BFloat16 & int)
/// @{

inline NCORE_HOST_DEVICE BFloat16 operator+(BFloat16 a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a + static_cast<BFloat16>(b);
}
inline NCORE_HOST_DEVICE BFloat16 operator-(BFloat16 a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a - static_cast<BFloat16>(b);
}
inline NCORE_HOST_DEVICE BFloat16 operator*(BFloat16 a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a * static_cast<BFloat16>(b);
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE BFloat16 operator/(BFloat16 a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a / static_cast<BFloat16>(b);
}

inline NCORE_HOST_DEVICE BFloat16 operator+(int a, BFloat16 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<BFloat16>(a) + b;
}
inline NCORE_HOST_DEVICE BFloat16 operator-(int a, BFloat16 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<BFloat16>(a) - b;
}
inline NCORE_HOST_DEVICE BFloat16 operator*(int a, BFloat16 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<BFloat16>(a) * b;
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE BFloat16 operator/(int a, BFloat16 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<BFloat16>(a) / b;
}

/// @}

/// @name Mixed-type Arithmetic (BFloat16 & int64_t)
/// @{

inline NCORE_HOST_DEVICE BFloat16 operator+(BFloat16 a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a + static_cast<BFloat16>(b);
}
inline NCORE_HOST_DEVICE BFloat16 operator-(BFloat16 a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a - static_cast<BFloat16>(b);
}
inline NCORE_HOST_DEVICE BFloat16 operator*(BFloat16 a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a * static_cast<BFloat16>(b);
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE BFloat16 operator/(BFloat16 a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a / static_cast<BFloat16>(b);
}

inline NCORE_HOST_DEVICE BFloat16 operator+(int64_t a, BFloat16 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<BFloat16>(a) + b;
}
inline NCORE_HOST_DEVICE BFloat16 operator-(int64_t a, BFloat16 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<BFloat16>(a) - b;
}
inline NCORE_HOST_DEVICE BFloat16 operator*(int64_t a, BFloat16 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<BFloat16>(a) * b;
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE BFloat16 operator/(int64_t a, BFloat16 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<BFloat16>(a) / b;
}

/// @}

/**
 * @brief Greater-than comparison.
 * @details Explicitly overloaded (rather than relying solely on the implicit
 * float conversion) because @c std::max / @c std::min require an lvalue-ref
 * accepting @c operator>/@c operator< to resolve unambiguously once the
 * bitwise operators above are in overload resolution scope.
 * @param[in] lhs The left-hand operand.
 * @param[in] rhs The right-hand operand.
 * @return @c true if @p lhs is strictly greater than @p rhs.
 */
inline NCORE_HOST_DEVICE bool operator>(BFloat16 &lhs, BFloat16 &rhs) {
  return static_cast<float>(lhs) > static_cast<float>(rhs);
}

/**
 * @brief Less-than comparison.
 * @param[in] lhs The left-hand operand.
 * @param[in] rhs The right-hand operand.
 * @return @c true if @p lhs is strictly less than @p rhs.
 */
inline NCORE_HOST_DEVICE bool operator<(BFloat16 &lhs, BFloat16 &rhs) {
  return static_cast<float>(lhs) < static_cast<float>(rhs);
}

} // namespace ncore::dtypes

// ============================================================
// std::numeric_limits specialisation
// ============================================================
namespace std {

/**
 * @class numeric_limits<ncore::dtypes::BFloat16>
 * @brief Specialization of std::numeric_limits for the custom @ref
 * ncore::dtypes::BFloat16 type.
 */
template <> class numeric_limits<ncore::dtypes::BFloat16> {
  using BFloat16 = ncore::dtypes::BFloat16;

public:
  static constexpr bool is_specialized = true; ///< BFloat16 has numeric limits.
  static constexpr bool is_signed = true;      ///< BFloat16 is signed.
  static constexpr bool is_integer = false;    ///< BFloat16 is floating-point.
  static constexpr bool is_exact =
      false; ///< BFloat16 representation is not exact.
  static constexpr bool has_infinity =
      true; ///< BFloat16 has infinity representation.
  static constexpr bool has_quiet_NaN =
      true; ///< BFloat16 has quiet NaN representation.
  static constexpr bool has_signaling_NaN =
      true; ///< BFloat16 has signaling NaN representation.
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss =
      numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 =
      false; ///< Truncated mantissa breaks strict IEC 60559 conformance.
  static constexpr bool is_bounded = true; ///< Values are bounded.
  static constexpr bool is_modulo =
      false; ///< BFloat16 arithmetic does not modulo wrap.
  static constexpr int digits = 8;   ///< Mantissa bits + 1 implicit bit.
  static constexpr int digits10 = 2; ///< Decimal digits reliably preserved.
  static constexpr int max_digits10 =
      4; ///< Decimal digits required to uniquely represent values.
  static constexpr int radix = 2; ///< Base of the exponent.
  static constexpr int min_exponent =
      -125; ///< Minimum negative power of 2 for a normal value.
  static constexpr int min_exponent10 = -37; ///< Minimum negative power of 10.
  static constexpr int max_exponent = 128;   ///< Maximum positive power of 2.
  static constexpr int max_exponent10 = 38;  ///< Maximum positive power of 10.
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before;

  /**
   * @brief Smallest positive normalized value.
   * @details 0x0080 → 2**(-126).
   * @return Smallest normalized value.
   */
  static constexpr BFloat16 min() { return {0x0080, BFloat16::from_bits()}; }

  /**
   * @brief Largest finite negative value.
   * @details 0xFF7F.
   * @return Lowest negative value.
   */
  static constexpr BFloat16 lowest() { return {0xFF7F, BFloat16::from_bits()}; }

  /**
   * @brief Largest finite positive value.
   * @details 0x7F7F.
   * @return Maximum value.
   */
  static constexpr BFloat16 max() { return {0x7F7F, BFloat16::from_bits()}; }

  /**
   * @brief Machine epsilon.
   * @details 0x3C00 → 2**(-7).
   * @return Machine epsilon.
   */
  static constexpr BFloat16 epsilon() {
    return {0x3C00, BFloat16::from_bits()};
  }

  /**
   * @brief Maximum rounding error.
   * @details 0x3F00 → 0.5.
   * @return Rounding error.
   */
  static constexpr BFloat16 round_error() {
    return {0x3F00, BFloat16::from_bits()};
  }

  /**
   * @brief Positive infinity.
   * @details 0x7F80.
   * @return Infinity value.
   */
  static constexpr BFloat16 infinity() {
    return {0x7F80, BFloat16::from_bits()};
  }

  /**
   * @brief Quiet NaN.
   * @details 0x7FC0.
   * @return Quiet NaN value.
   */
  static constexpr BFloat16 quiet_NaN() {
    return {0x7FC0, BFloat16::from_bits()};
  }

  /**
   * @brief Signaling NaN.
   * @details 0x7F80 (shares the encoding with infinity; BF16 does not
   * distinguish a dedicated signaling pattern in this implementation).
   * @return Signaling NaN value.
   */
  static constexpr BFloat16 signaling_NaN() {
    return {0x7F80, BFloat16::from_bits()};
  }

  /**
   * @brief Smallest positive subnormal value.
   * @details 0x0001 → 2**(-133).
   * @return Subnormal minimum.
   */
  static constexpr BFloat16 denorm_min() {
    return {0x0001, BFloat16::from_bits()};
  }
};

} // namespace std

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif

#ifdef _GNUC_CLANG_
#pragma GCC diagnostic pop
#endif
