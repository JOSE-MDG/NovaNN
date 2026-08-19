/**
 * @file half.hh
 * @brief Half-precision floating-point (FP16) data type implementation.
 *
 * @details
 * Defines the @ref Half type representing a 16-bit half-precision
 * floating-point number conforming to the IEEE 754 standard. This includes
 * conversions to and from standard C++ types (float, double, int, etc.), basic
 * arithmetic operations, and support for GPU compilation environments
 * (CUDA/HIP).
 *
 * Arithmetic operations are implemented by promoting to float32 rather than
 * using CUDA/HIP half intrinsics directly on host, as most operations are
 * memory-bound.
 *
 * @see macros.h  NovaNN compiler and platform macros.
 */

#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
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

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

#ifdef __HIPCC__
#include <hip/hip_fp16.h>
#endif

#ifdef _GNUC_CLANG_
#if HAS_F16C && !(defined(__CUDA_ARCH__) || defined(__CUDACC__) ||             \
                  defined(__HIP_DEVICE_COMPILE__))
#define F16C_AVAILABLE HAS_F16C
#else
#define F16C_AVAILABLE 0
#endif // HAS_F16C && !(CUDA/HIP device)
#include <immintrin.h>
#endif // _GNUC_CLANG_

/**
 * @namespace ncore::dtypes
 * @brief Custom data types for operations.
 */
namespace ncore::dtypes {

/**
 * @struct Half
 * @brief Representation of an IEEE 754 half-precision (16-bit) floating-point
 * number.
 *
 * @details
 * The struct is aligned to 2 bytes to match the storage size of a 16-bit word.
 * It contains a single data member representing the bitwise storage.
 */
struct alignas(2) Half {
  unsigned short x; ///< The 16-bit binary representation of the FP16 value.

  /**
   * @struct from_bits_t
   * @brief Tag type used to construct a @ref Half directly from its raw 16-bit
   * representation.
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
  NCORE_HOST_DEVICE Half() = default;
#else
  /**
   * @brief Default constructor. Left uninitialized for performance.
   */
  Half() = default;
#endif

  /**
   * @brief Constructs a @ref Half directly from a raw 16-bit integer
   * representation.
   * @param[in] bits The raw 16-bit binary representation.
   * @param[in] unused Tag parameter to disambiguate from float conversions.
   */
  constexpr NCORE_HOST_DEVICE Half(unsigned short bits,
                                   from_bits_t /*unused*/) noexcept
      : x(bits) {}

  /**
   * @brief Implicit constructor from a single-precision float.
   * @param[in] value The float value to convert.
   */
  inline NCORE_HOST_DEVICE Half(float value);

  /**
   * @brief Implicit conversion operator to single-precision float.
   * @return The float representation of the half-precision value.
   */
  inline NCORE_HOST_DEVICE operator float() const;

#if defined(__CUDACC__) || defined(__HIPCC__)
  /**
   * @brief Constructor from a native CUDA/HIP @c __half representation.
   * @param[in] value The native __half value.
   */
  inline NCORE_HOST_DEVICE Half(const __half &value);

  /**
   * @brief Implicit conversion operator to native CUDA/HIP @c __half.
   * @return The native __half representation.
   */
  inline NCORE_HOST_DEVICE operator __half() const;
#endif
};

/**
 * @brief Stream output operator for @ref Half.
 * @details Promotes the half-precision value to float before writing to the
 * stream.
 * @param[in,out] out The output stream.
 * @param[in]     value The @ref Half value to write.
 * @return Reference to the output stream.
 */
inline std::ostream &operator<<(std::ostream &out, const Half &value) {
  out << static_cast<float>(value);
  return out;
}

/**
 * @namespace ncore::dtypes::detail
 * @brief Internal low-level bit-manipulation and conversion helpers.
 */
namespace detail {

/**
 * @brief Reinterpret a @c uint32_t bit-pattern as a float.
 * @param[in] w The 32-bit unsigned integer representing the bit pattern.
 * @return The float value corresponding to the bit pattern.
 */
NCORE_HOST_DEVICE inline float fp32_from_bits(uint32_t w) {
  float f;
#if defined(_MSC_VER) && defined(__clang__)
  __builtin_memcpy(&f, &w, sizeof(f));
#else
  std::memcpy(&f, &w, sizeof(f));
#endif
  return f;
}

/**
 * @brief Reinterpret a float as its @c uint32_t bit-pattern.
 * @param[in] f The single-precision float.
 * @return The 32-bit unsigned integer representing the float's bit pattern.
 */
NCORE_HOST_DEVICE inline uint32_t fp32_to_bits(float f) {
  uint32_t w;
#if defined(_MSC_VER) && defined(__clang__)
  __builtin_memcpy(&w, &f, sizeof(w));
#else
  std::memcpy(&w, &f, sizeof(w));
#endif
  return w;
}

/**
 * @brief Converts a 16-bit IEEE half-precision bit pattern to a 32-bit float
 * value.
 * @details
 * Uses hardware acceleration if F16C instruction set is available.
 * Otherwise fallback to software conversion handling normalized, denormalized,
 * and special values (zero, infinity, NaN).
 *
 * @param[in] h The 16-bit half-precision bit pattern.
 * @return The single-precision float value.
 */
NCORE_HOST_DEVICE inline float fp16_ieee_to_fp32_value(uint16_t h) {
#if F16C_AVAILABLE
  return _cvtsh_ss(h);
#else
  /*
   * Extend the half-precision floating-point number to 32 bits and shift to
   * the upper part of the 32-bit word:
   *      +---+-----+------------+-------------------+
   *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  31  26-30    16-25            0-15
   *
   * S - sign bit, E - biased exponent bits, M - mantissa bits, 0 - zero bits.
   */
  const uint32_t w = static_cast<uint32_t>(h) << 16;
  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-30
   */
  const uint32_t sign = w & UINT32_C(0x80000000);
  /*
   * Extract mantissa and biased exponent of the input number into the high bits
   * of the 32-bit word:
   *
   *      +-----+------------+---------------------+
   *      |EEEEE|MM MMMM MMMM|0 0000 0000 0000 0000|
   *      +-----+------------+---------------------+
   * Bits  27-31    17-26            0-16
   */
  const uint32_t twoW = w + w;

  /*
   * Shift mantissa and exponent into bits 23-28 and bits 13-22 so they become
   * mantissa and exponent of a single-precision floating-point number:
   *
   *       S|Exponent |          Mantissa
   *      +-+---+-----+------------+----------------+
   *      |0|000|EEEEE|MM MMMM MMMM|0 0000 0000 0000|
   *      +-+---+-----+------------+----------------+
   * Bits   | 23-31   |           0-22
   *
   * Exponent bias correction:
   *   - The exponent needs adjustment by the difference in exponent bias
   *     between single-precision and half-precision formats (0x7F - 0xF = 0x70)
   *   - Inf and NaN must remain Inf and NaN after conversion. We correct in
   *     two steps: adjust by 0xE0 (instead of 0x70), then multiply by 2**(-112)
   *     to reverse the extra 0x70 part. FP multiply hardware preserves Inf/NaN.
   *
   * Denormal inputs are NOT handled by the path below.
   */
  constexpr uint32_t expOffset = UINT32_C(0xE0) << 23;
  // exp_scale = 0x1.0p-112f represented as bits
  constexpr uint32_t scaleBits = static_cast<uint32_t>(15) << 23;
  float expScaleVal = 0;
  std::memcpy(&expScaleVal, &scaleBits, sizeof(expScaleVal));
  const float expScale = expScaleVal;
  const float normalizedValue =
      fp32_from_bits((twoW >> 4) + expOffset) * expScale;

  /*
   * Convert denormalized half-precision inputs into single-precision results
   * (always normalized). Zero inputs are also handled here.
   *
   * In a denormalized number the biased exponent is zero and the mantissa has
   * non-zero bits. We shift the mantissa into bits 0-9 of the 32-bit word:
   *
   *                  zeros           |  mantissa
   *      +---------------------------+------------+
   *      |0000 0000 0000 0000 0000 00|MM MMMM MMMM|
   *      +---------------------------+------------+
   * Bits             10-31                0-9
   *
   * Denormalized FP16 = mantissa * 2**(-24).
   * We construct a normalized FP32 with the same mantissa and biased exponent
   * 126, so a unit change in the mantissa changes the FP32 value by 2**(-24).
   * Finally we subtract 0.5 to remove the implicit leading 1 added by biased
   * exponent 126 (which contributes FP32 = 1 * 2**(126-127) = 0.5).
   */
  constexpr uint32_t magicMask = UINT32_C(126) << 23;
  constexpr float magicBias = 0.5f;
  const float denormalizedValue =
      fp32_from_bits((twoW >> 17) | magicMask) - magicBias;

  /*
   * Choose between normalized and denormalized result based on whether
   * two_w < 2**27 (i.e. the input was denormal or zero). Then combine with
   * the sign bit extracted above.
   */
  constexpr uint32_t denormalizedCutoff = UINT32_C(1) << 27;
  const uint32_t result =
      sign | (twoW < denormalizedCutoff ? fp32_to_bits(denormalizedValue)
                                        : fp32_to_bits(normalizedValue));
  return fp32_from_bits(result);
#endif // F16C_AVAILABLE
}

/**
 * @brief Converts a 32-bit float value to a 16-bit IEEE half-precision bit
 * pattern.
 * @details
 * Uses hardware acceleration if F16C instruction set is available.
 * Otherwise fallback to software conversion performing proper rounding to
 * nearest-even.
 *
 * @param[in] f The single-precision float value to convert.
 * @return The 16-bit half-precision bit pattern.
 */
inline uint16_t fp16_ieee_from_fp32_value(float f) {
#if F16C_AVAILABLE
  return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT);
#else
  // scale2inf  = 0x1.0p+112f
  // scale2zero = 0x1.0p-110f
  constexpr uint32_t scale2infBits = static_cast<uint32_t>(239) << 23;
  constexpr uint32_t scale2zeroBits = static_cast<uint32_t>(17) << 23;
  float scale2infVal = 0, scale2zeroVal = 0;
  std::memcpy(&scale2infVal, &scale2infBits, sizeof(scale2infVal));
  std::memcpy(&scale2zeroVal, &scale2zeroBits, sizeof(scale2zeroVal));
  const float scale2inf = scale2infVal;
  const float scale2zero = scale2zeroVal;

#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER == 1916
  float base = ((signbit(f) != 0 ? -f : f) * scale2inf) * scale2zero;
#else
  float base = (fabsf(f) * scale2inf) * scale2zero;
#endif

  const uint32_t w = fp32_to_bits(f);
  const uint32_t shl1W = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1W & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000)) {
    bias = UINT32_C(0x71000000);
  }

  base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = fp32_to_bits(base);
  const uint32_t expBits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissaBits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = expBits + mantissaBits;
  return static_cast<uint16_t>(
      (sign >> 16) |
      (shl1W > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
#endif // F16C_AVAILABLE
}

/**
 * @brief Converts a 16-bit half-precision bit pattern to a 32-bit float bit
 * pattern.
 * @details
 * This operation is purely integer-based and does not trigger floating-point
 * exceptions.
 *
 * @param[in] h The 16-bit half-precision bit pattern.
 * @return The 32-bit single-precision float bit pattern.
 */
inline uint32_t fp16_ieee_to_fp32_bits(uint16_t h) {
  /*
   * Extend the half-precision floating-point number to 32 bits and shift to
   * the upper part of the 32-bit word:
   *      +---+-----+------------+-------------------+
   *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  31  26-30    16-25            0-15
   */
  const uint32_t w = static_cast<uint32_t>(h) << 16;
  /*
   * Extract the sign into the high bit:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-30
   */
  const uint32_t sign = w & UINT32_C(0x80000000);
  /*
   * Strip the sign to get the unsigned magnitude (exponent + mantissa):
   *
   *      +---+-----+------------+-------------------+
   *      | 0 |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  30  27-31     17-26            0-16
   */
  const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
  /*
   * renormShift: number of bits to shift mantissa left to normalize a
   * denormal. For a normalized input one of its high 6 bits is 1, so
   * renormShift == 0. For denormals renormShift > 0; after shifting, the
   * leading mantissa bit moves into the exponent field (biased exp becomes 1)
   * making it normalized (implicit leading 1 removed).
   */
  uint32_t renormShift =
      static_cast<uint32_t>(std::countl_zero<uint32_t>(nonsign));
  renormShift = renormShift > 5 ? renormShift - 5 : 0;
  /*
   * If the half-precision exponent is 0x1F (max), the addition below
   * overflows into bit 31, and the subsequent shift fills the high 9 bits
   * with 1 → infNanMask == 0x7F800000.  Otherwise it is 0x00000000.
   */
  const int32_t infNanMask =
      (static_cast<int32_t>(nonsign + 0x04000000) >> 8) & INT32_C(0x7F800000);
  /*
   * If nonsign == 0 (±0.0h), nonsign - 1 underflows to 0xFFFFFFFF, making
   * bit 31 == 1, and the arithmetic right shift by 31 fills zeroMask with
   * all 1s. Otherwise zeroMask == 0x00000000.
   */
  const int32_t zeroMask = static_cast<int32_t>(nonsign - 1) >> 31;
  /*
   * Steps performed in one expression:
   *  1. Shift nonsign left by renormShift (normalize denormals).
   *  2. Shift right by 3 (expand 5-bit exponent to 8-bit, 10-bit mantissa
   *     into the high 10 bits of the 23-bit FP32 mantissa).
   *  3. Add 0x70 << 23 to the exponent to compensate the bias difference
   *     (0x7F for FP32 minus 0xF for FP16 = 0x70), combined with step 4.
   *  4. Subtract renormShift from the exponent to account for renorm.
   *  5. OR with infNanMask to force 0xFF exponent for NaN/Inf.
   *  6. ANDNOT with zeroMask to zero mantissa and exponent for ±0.
   *  7. OR with sign.
   */
  return sign |
         ((((nonsign << renormShift >> 3) + ((0x70 - renormShift) << 23)) |
           static_cast<uint32_t>(infNanMask)) &
          ~static_cast<uint32_t>(zeroMask));
}

#if F16C_AVAILABLE
#undef F16C_AVAILABLE
#endif

} // namespace detail

// ============================================================
// Constructors and conversion operators — inline definitions
// ============================================================

inline NCORE_HOST_DEVICE Half::Half(float value)
    :
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      x(__half_as_short(__float2half(value)))
#else
      x(detail::fp16_ieee_from_fp32_value(value))
#endif
{
}

inline NCORE_HOST_DEVICE Half::operator float() const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __half2float(*reinterpret_cast<const __half *>(&x));
#else
  return detail::fp16_ieee_to_fp32_value(x);
#endif
}

#if defined(__CUDACC__) || defined(__HIPCC__)
inline NCORE_HOST_DEVICE Half::Half(const __half &value) {
  x = *reinterpret_cast<const unsigned short *>(&value);
}
inline NCORE_HOST_DEVICE Half::operator __half() const {
  return *reinterpret_cast<const __half *>(&x);
}
#endif

// ============================================================
// CUDA __ldg helper
// ============================================================
#if (defined(__clang__) && defined(__CUDA__))
/**
 * @brief Load a @ref Half value from global memory using the CUDA @c __ldg
 * intrinsic.
 * @param[in] ptr Pointer to global memory.
 * @return The loaded @ref Half value.
 */
inline __device__ Half __ldg(const Half *ptr) {
  return __ldg(reinterpret_cast<const __half *>(ptr));
}
#endif

// ============================================================
// Arithmetic operators
// ============================================================

/// @name Arithmetic Operators (Half & Half)
/// @{

/**
 * @brief Addition operator for two @ref Half values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The sum as a @ref Half.
 */
inline NCORE_HOST_DEVICE Half operator+(const Half &a, const Half &b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

/**
 * @brief Subtraction operator for two @ref Half values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The difference as a @ref Half.
 */
inline NCORE_HOST_DEVICE Half operator-(const Half &a, const Half &b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

/**
 * @brief Multiplication operator for two @ref Half values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The product as a @ref Half.
 */
inline NCORE_HOST_DEVICE Half operator*(const Half &a, const Half &b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

/**
 * @brief Division operator for two @ref Half values.
 * @param[in] a The first operand.
 * @param[in] b The second operand.
 * @return The quotient as a @ref Half.
 */
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Half operator/(const Half &a, const Half &b) {
  return static_cast<float>(a) / static_cast<float>(b);
}

/**
 * @brief Unary minus operator for @ref Half.
 * @param[in] a The operand.
 * @return The negated value as a @ref Half.
 */
inline NCORE_HOST_DEVICE Half operator-(const Half &a) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530) ||                        \
    defined(__HIP_DEVICE_COMPILE__)
  return __hneg(a);
#else
  return -static_cast<float>(a);
#endif
}

/**
 * @brief Addition assignment operator.
 * @param[in,out] a The destination operand.
 * @param[in]     b The operand to add.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE Half &operator+=(Half &a, const Half &b) {
  a = a + b;
  return a;
}

/**
 * @brief Subtraction assignment operator.
 * @param[in,out] a The destination operand.
 * @param[in]     b The operand to subtract.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE Half &operator-=(Half &a, const Half &b) {
  a = a - b;
  return a;
}

/**
 * @brief Multiplication assignment operator.
 * @param[in,out] a The destination operand.
 * @param[in]     b The operand to multiply.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE Half &operator*=(Half &a, const Half &b) {
  a = a * b;
  return a;
}

/**
 * @brief Division assignment operator.
 * @param[in,out] a The destination operand.
 * @param[in]     b The divisor.
 * @return Reference to the updated destination operand.
 */
inline NCORE_HOST_DEVICE Half &operator/=(Half &a, const Half &b) {
  a = a / b;
  return a;
}

/// @}

/// @name Mixed-type Arithmetic (Half & float)
/// @{

inline NCORE_HOST_DEVICE float operator+(Half a, float b) {
  return static_cast<float>(a) + b;
}
inline NCORE_HOST_DEVICE float operator-(Half a, float b) {
  return static_cast<float>(a) - b;
}
inline NCORE_HOST_DEVICE float operator*(Half a, float b) {
  return static_cast<float>(a) * b;
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE float operator/(Half a, float b) {
  return static_cast<float>(a) / b;
}

inline NCORE_HOST_DEVICE float operator+(float a, Half b) {
  return a + static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float operator-(float a, Half b) {
  return a - static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float operator*(float a, Half b) {
  return a * static_cast<float>(b);
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE float operator/(float a, Half b) {
  return a / static_cast<float>(b);
}

inline NCORE_HOST_DEVICE float &operator+=(float &a, const Half &b) {
  return a += static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float &operator-=(float &a, const Half &b) {
  return a -= static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float &operator*=(float &a, const Half &b) {
  return a *= static_cast<float>(b);
}
inline NCORE_HOST_DEVICE float &operator/=(float &a, const Half &b) {
  return a /= static_cast<float>(b);
}

/// @}

/// @name Mixed-type Arithmetic (Half & double)
/// @{

inline NCORE_HOST_DEVICE double operator+(Half a, double b) {
  return static_cast<double>(a) + b;
}
inline NCORE_HOST_DEVICE double operator-(Half a, double b) {
  return static_cast<double>(a) - b;
}
inline NCORE_HOST_DEVICE double operator*(Half a, double b) {
  return static_cast<double>(a) * b;
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE double operator/(Half a, double b) {
  return static_cast<double>(a) / b;
}

inline NCORE_HOST_DEVICE double operator+(double a, Half b) {
  return a + static_cast<double>(b);
}
inline NCORE_HOST_DEVICE double operator-(double a, Half b) {
  return a - static_cast<double>(b);
}
inline NCORE_HOST_DEVICE double operator*(double a, Half b) {
  return a * static_cast<double>(b);
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE double operator/(double a, Half b) {
  return a / static_cast<double>(b);
}

/// @}

/// @name Mixed-type Arithmetic (Half & int)
/// @{

inline NCORE_HOST_DEVICE Half operator+(Half a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a + static_cast<Half>(b);
}
inline NCORE_HOST_DEVICE Half operator-(Half a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a - static_cast<Half>(b);
}
inline NCORE_HOST_DEVICE Half operator*(Half a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a * static_cast<Half>(b);
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Half operator/(Half a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a / static_cast<Half>(b);
}

inline NCORE_HOST_DEVICE Half operator+(int a, Half b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Half>(a) + b;
}
inline NCORE_HOST_DEVICE Half operator-(int a, Half b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Half>(a) - b;
}
inline NCORE_HOST_DEVICE Half operator*(int a, Half b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Half>(a) * b;
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Half operator/(int a, Half b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Half>(a) / b;
}

/// @}

/// @name Mixed-type Arithmetic (Half & int64_t)
/// @{

inline NCORE_HOST_DEVICE Half operator+(Half a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a + static_cast<Half>(b);
}
inline NCORE_HOST_DEVICE Half operator-(Half a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a - static_cast<Half>(b);
}
inline NCORE_HOST_DEVICE Half operator*(Half a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a * static_cast<Half>(b);
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Half operator/(Half a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return a / static_cast<Half>(b);
}

inline NCORE_HOST_DEVICE Half operator+(int64_t a, Half b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Half>(a) + b;
}
inline NCORE_HOST_DEVICE Half operator-(int64_t a, Half b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Half>(a) - b;
}
inline NCORE_HOST_DEVICE Half operator*(int64_t a, Half b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Half>(a) * b;
}
NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
inline NCORE_HOST_DEVICE Half operator/(int64_t a, Half b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
  return static_cast<Half>(a) / b;
}

/// @}

/// @note Comparison operators are not defined here; rely on the implicit
/// conversion from ncore::dtypes::Half to float.

} // namespace ncore::dtypes

// ============================================================
// std::numeric_limits specialization
// ============================================================
namespace std {

/**
 * @class numeric_limits<ncore::dtypes::Half>
 * @brief Specialization of std::numeric_limits for the custom @ref
 * ncore::dtypes::Half type.
 */
template <> class numeric_limits<ncore::dtypes::Half> {
  using Half = ncore::dtypes::Half;

public:
  static constexpr bool is_specialized = true; ///< Half has numeric limits.
  static constexpr bool is_signed = true;      ///< Half is signed.
  static constexpr bool is_integer = false;    ///< Half is floating-point.
  static constexpr bool is_exact = false; ///< Half representation is not exact.
  static constexpr bool has_infinity =
      true; ///< Half has infinity representation.
  static constexpr bool has_quiet_NaN =
      true; ///< Half has quiet NaN representation.
  static constexpr bool has_signaling_NaN =
      true; ///< Half has signaling NaN representation.
  static constexpr auto has_denorm = true; ///< Half supports subnormals.
  static constexpr auto has_denorm_loss =
      false; ///< No denormalization loss is detected.
  static constexpr auto round_style =
      numeric_limits<float>::round_style; ///< Inherits float32 rounding style.
  static constexpr bool is_iec559 = true; ///< Conforms to IEC 60559 (IEEE 754).
  static constexpr bool is_bounded = true; ///< Values are bounded.
  static constexpr bool is_modulo =
      false;                        ///< Half arithmetic does not modulo wrap.
  static constexpr int digits = 11; ///< Mantissa bits + 1 implicit bit.
  static constexpr int digits10 =
      3; ///< Number of decimal digits that can be represented.
  static constexpr int max_digits10 =
      5; ///< Number of decimal digits required to uniquely represent values.
  static constexpr int radix = 2; ///< Base of the exponent.
  static constexpr int min_exponent =
      -13; ///< Minimum negative power of 2 for a normal value.
  static constexpr int min_exponent10 = -4; ///< Minimum negative power of 10.
  static constexpr int max_exponent = 16;   ///< Maximum positive power of 2.
  static constexpr int max_exponent10 = 4;  ///< Maximum positive power of 10.
  static constexpr auto traps =
      numeric_limits<float>::traps; ///< Inherits float32 trap behaviour.
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before; ///< Inherits float32
                                              ///< tinyness-before semantics.

  /**
   * @brief Smallest positive normalized value.
   * @details 0x0400 → @f$2^{-14}@f$ ≈ 6.10e-5.
   * @return Smallest normalized value.
   */
  static constexpr Half min() { return {0x0400, Half::from_bits()}; }

  /**
   * @brief Largest finite negative value.
   * @details 0xFBFF → -65504.
   * @return Lowest negative value.
   */
  static constexpr Half lowest() { return {0xFBFF, Half::from_bits()}; }

  /**
   * @brief Largest finite positive value.
   * @details 0x7BFF → 65504.
   * @return Maximum value.
   */
  static constexpr Half max() { return {0x7BFF, Half::from_bits()}; }

  /**
   * @brief Machine epsilon.
   * @details 0x1400 → @f$2^{-10}@f$ ≈ 9.77e-4.
   * @return Machine epsilon.
   */
  static constexpr Half epsilon() { return {0x1400, Half::from_bits()}; }

  /**
   * @brief Maximum rounding error.
   * @details 0x3800 → 0.5.
   * @return Rounding error.
   */
  static constexpr Half round_error() { return {0x3800, Half::from_bits()}; }

  /**
   * @brief Positive infinity.
   * @details 0x7C00.
   * @return Infinity value.
   */
  static constexpr Half infinity() { return {0x7C00, Half::from_bits()}; }

  /**
   * @brief Quiet NaN.
   * @details 0x7E00.
   * @return Quiet NaN value.
   */
  static constexpr Half quiet_NaN() { return {0x7E00, Half::from_bits()}; }

  /**
   * @brief Signaling NaN.
   * @details 0x7D00.
   * @return Signaling NaN value.
   */
  static constexpr Half signaling_NaN() { return {0x7D00, Half::from_bits()}; }

  /**
   * @brief Smallest positive subnormal value.
   * @details 0x0001 → @f$2^{-24}@f$ ≈ 5.96e-8.
   * @return Subnormal minimum.
   */
  static constexpr Half denorm_min() { return {0x0001, Half::from_bits()}; }
};

} // namespace std

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif

#ifdef _GNUC_CLANG_
#pragma GCC diagnostic pop
#endif
