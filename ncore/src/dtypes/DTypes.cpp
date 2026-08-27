/**
 * @file DTypes.cpp
 * @brief Compile-time dispatch implementation for reduced-precision
 *        float conversions.
 *
 * Each (Except FP4 and FP8) conversion function selects between two strategies
 * at compile time:
 *
 *   @li Compiler builtin — when the platform provides a native type
 *     (@c _Float16 for FP16, @c __bf16 for BF16 on GCC/Clang) the conversion
 *     is a simple @c static_cast, producing optimal code.
 *   @li Software fallback — an integer-only bit-manipulation routine
 *     from the corresponding @c ncore/headeronly/dtypes/<file>.hh header.  The
 *     fallback is pulled in via a conditional @c #include on the matching
 *     @c *.hpp wrapper in this directory.
 *
 * FP8 formats (E5M2, E4M3FN) and FP4 always use the software path because
 * no mainstream compiler exposes a native 8-bit or 4-bit float type.
 *
 * The functions are declared with @c extern "C" linkage in DTypes.hpp and
 * exposed to user code through fp_utils.h.
 *
 * @see fp_utils.h       C API for reduced-precision float conversions.
 * @see DTypes.hpp       Declaration for reduced-precision float conversions.
 * @see half.hh          IEEE 754 half-precision (FP16).
 * @see bfloat16.hh      Brain Float 16 (BF16).
 * @see fp8_e5m2.hh      FP8 E5M2 (8-bit, 1.5.2 layout).
 * @see fp8_e4m3fn.hh    FP8 E4M3FN (8-bit, 1.4.3 layout).
 * @see fp4_e2m1fn_x2.hh FP4 E2M1FN pair-packed (4-bit, 1.2.1 layout).
 */

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ncore/core/dtype.h>
#include <ncore/headeronly/macros.h>

#include "BFloat16.hpp"
#include "DTypes.hpp"
#include "Float4_e2m1fn_x2.hpp"
#include "Float8_e4m3fn.hpp"
#include "Float8_e5m2.hpp"
#include "Half.hpp"
#include "ncore/headeronly/dtypes/half.hh"

/**
 * @brief Convert a single-precision float to IEEE 754 half-precision (FP16).
 *
 * @param[in] val  The single-precision float value to convert.
 * @return The FP16 representation of @p val.
 */
float16 fp16_from_float(float val) {
#if defined(_GNUC_CLANG_)
  return static_cast<float16>(val);
#else
  return ncore::dtypes::detail::fp16_ieee_from_fp32_value(val);
#endif
}

/**
 * @brief Convert an IEEE 754 half-precision (FP16) value to single-precision
 * float.
 *
 * @param[in] val  The FP16 value to convert.
 * @return The single-precision float equivalent of @p val.
 */
float fp16_to_float(float16 val) {
#if defined(_GNUC_CLANG_)
  return static_cast<float>(val);
#else
  return ncore::dtypes::detail::fp16_ieee_to_fp32_value(val);
#endif
}

/**
 * @brief Convert a single-precision float to bfloat16 (Brain Float 16).
 *
 * @param[in] val  The single-precision float value to convert.
 * @return The bfloat16 representation of @p val.
 */
bfloat16 bf16_from_float(float val) {
#if defined(_GNUC_CLANG_)
  // Contract (fp_utils.h): NaN maps to the canonical bf16 NaN 0x7FC0.  The
  // native __bf16 static_cast preserves the f32 NaN sign bit instead, so
  // canonicalize explicitly to stay consistent with the BFloat16 structure.
  if (std::isnan(val)) {
    return std::bit_cast<bfloat16>(static_cast<uint16_t>(UINT16_C(0x7FC0)));
  }
  return static_cast<bfloat16>(val);
#else
  return ncore::dtypes::detail::bits_from_f32(val);
#endif
}

/**
 * @brief Convert a bfloat16 (Brain Float 16) value to single-precision float.
 *
 * @param[in] val  The bfloat16 value to convert.
 * @return The single-precision float equivalent of @p val.
 */
float bf16_to_float(bfloat16 val) {
#if defined(_GNUC_CLANG_)
  return static_cast<float>(val);
#else
  return ncore::dtypes::detail::f32_from_bits(val);
#endif
}

/**
 * @brief Convert a single-precision float to FP8 E5M2 format.
 *
 * @param[in] val  The single-precision float value to convert.
 * @return The FP8 E5M2 representation of @p val.
 */
float8_e5m2 fp8e5m2_from_float(float val) {
  return ncore::dtypes::detail::fp8e5m2_from_fp32_value(val);
}

/**
 * @brief Convert an FP8 E5M2 value to single-precision float.
 *
 * @param[in] val  The FP8 E5M2 value to convert.
 * @return The single-precision float equivalent of @p val.
 */
float fp8e5m2_to_float(float8_e5m2 val) {
  return ncore::dtypes::detail::fp8e5m2_to_fp32_value(val);
}

/**
 * @brief Convert a single-precision float to FP8 E4M3FN format.
 *
 * @param[in] val  The single-precision float value to convert.
 * @return The FP8 E4M3FN representation of @p val.
 */
float8_e4m3fn fp8e4m3fn_from_float(float val) {
  return ncore::dtypes::detail::fp8e4m3fn_from_fp32_value(val);
}

/**
 * @brief Convert an FP8 E4M3FN value to single-precision float.
 *
 * @param[in] val  The FP8 E4M3FN value to convert.
 * @return The single-precision float equivalent of @p val.
 */
float fp8e4m3fn_to_float(float8_e4m3fn val) {
  return ncore::dtypes::detail::fp8e4m3fn_to_fp32_value(val);
}

/**
 * @brief Pack two single-precision floats into a single FP4 E2M1FN pair-packed
 * byte.
 *
 * @param[in] lo  The low-lane float value to pack.
 * @param[in] hi  The high-lane float value to pack.
 * @return The pair-packed FP4 byte.
 */
float4_e2m1fn_x2 fp4e2m1x2_from_floats(float lo, float hi) {
  const ncore::dtypes::Float4_e2m1fn_x2 packed(lo, hi);
  return static_cast<float4_e2m1fn_x2>(packed.val_);
}

/**
 * @brief Unpack a single FP4 E2M1FN pair-packed byte into two single-precision
 * floats.
 *
 * @param[in]  val  The pair-packed FP4 byte to unpack.
 * @param[out] lo   Receives the low-lane float value.
 * @param[out] hi   Receives the high-lane float value.
 */
void fp4e2m1x2_to_floats(float4_e2m1fn_x2 val, float *lo, float *hi) {
  const ncore::dtypes::Float4_e2m1fn_x2 packed(static_cast<uint8_t>(val));
  *lo = static_cast<float>(packed.low());
  *hi = static_cast<float>(packed.high());
}

/**
 * @brief Convert an IEEE 754 half-precision (FP16) value to the bit pattern
 *        of its single-precision float32 equivalent.
 *
 * @param[in] val  The FP16 value to convert.
 * @return The IEEE 754 float32 bit pattern of @p val.
 */
uint32_t fp16_to_f32_bits(float16 val) {
#if defined(_GNUC_CLANG_)
  return ncore::dtypes::detail::fp16_ieee_to_fp32_bits(
      std::bit_cast<uint16_t>(val));
#else
  return ncore::dtypes::detail::fp16_ieee_to_fp32_bits(val);
#endif
}

/**
 * @brief Convert the bit pattern of a single-precision float to an IEEE 754
 *        half-precision (FP16) value.
 *
 * @param[in] val  The IEEE 754 float32 bit pattern to convert.
 * @return The FP16 representation of the value described by @p val.
 */
float16 fp16_from_f32_bits(uint32_t val) {
#if defined(_GNUC_CLANG_)
  return std::bit_cast<float16>(
      ncore::dtypes::detail::fp16_ieee_from_fp32_value(
          std::bit_cast<float>(val)));
#else
  return ncore::dtypes::detail::fp16_ieee_from_fp32_value(
      std::bit_cast<float>(val));
#endif
}

/**
 * @brief Convert a bfloat16 (Brain Float 16) value to the bit pattern of its
 *        single-precision float32 equivalent.
 *
 * @param[in] val  The bfloat16 value to convert.
 * @return The float32 bit pattern of @p val (low 16 bits are zero).
 */
uint32_t bf16_to_f32_bits(bfloat16 val) {
#if defined(_GNUC_CLANG_)
  return static_cast<uint32_t>(std::bit_cast<uint16_t>(val)) << 16;
#else
  return static_cast<uint32_t>(val) << 16;
#endif
}

/**
 * @brief Convert the bit pattern of a single-precision float to a bfloat16
 *        (Brain Float 16) value.
 *
 * @param[in] val  The IEEE 754 float32 bit pattern to convert.
 * @return The bfloat16 representation of the value described by @p val.
 */
bfloat16 bf16_from_f32_bits(uint32_t val) {
#if defined(_GNUC_CLANG_)
  return std::bit_cast<bfloat16>(
      ncore::dtypes::detail::round_to_nearest_even(std::bit_cast<float>(val)));
#else
  return ncore::dtypes::detail::round_to_nearest_even(
      std::bit_cast<float>(val));
#endif
}

/**
 * @brief Convert an FP8 E5M2 value to the bit pattern of its
 *        single-precision float32 equivalent.
 *
 * @param[in] val  The FP8 E5M2 value to convert.
 * @return The float32 bit pattern of @p val.
 */
uint32_t fp8e5m2_to_f32_bits(float8_e5m2 val) {
  return ncore::dtypes::detail::fp32_to_bits(
      ncore::dtypes::detail::fp8e5m2_to_fp32_value(val));
}

/**
 * @brief Convert the bit pattern of a single-precision float to an FP8 E5M2
 *        value.
 *
 * @param[in] val  The IEEE 754 float32 bit pattern to convert.
 * @return The FP8 E5M2 representation of the value described by @p val.
 */
float8_e5m2 fp8e5m2_from_f32_bits(uint32_t val) {
  return ncore::dtypes::detail::fp8e5m2_from_fp32_value(
      std::bit_cast<float>(val));
}

/**
 * @brief Convert an FP8 E4M3FN value to the bit pattern of its
 *        single-precision float32 equivalent.
 *
 * @param[in] val  The FP8 E4M3FN value to convert.
 * @return The float32 bit pattern of @p val.
 */
uint32_t fp8e4m3fn_to_f32_bits(float8_e4m3fn val) {
  return ncore::dtypes::detail::fp32_to_bits(
      ncore::dtypes::detail::fp8e4m3fn_to_fp32_value(val));
}

/**
 * @brief Convert the bit pattern of a single-precision float to an FP8
 *        E4M3FN value.
 *
 * @param[in] val  The IEEE 754 float32 bit pattern to convert.
 * @return The FP8 E4M3FN representation of the value described by @p val.
 */
float8_e4m3fn fp8e4m3fn_from_f32_bits(uint32_t val) {
  return ncore::dtypes::detail::fp8e4m3fn_from_fp32_value(
      std::bit_cast<float>(val));
}

/**
 * @brief Decompose a pair-packed FP4 E2M1FN byte into its constituent
 *        nibbles.
 *
 * @param[in] val  The pair-packed FP4 byte to decompose.
 * @return The decomposed nibbles together with the original byte.
 */
fp4e2m1x2Result_t fp4e2m1x2_to_f32_bits(float4_e2m1fn_x2 val) {
  fp4e2m1x2Result_t r;
  r.lo = val & 0x0F;
  r.hi = (val >> 4) & 0x0F;
  r.val = static_cast<float4_e2m1fn_x2>((r.hi << 4) | r.lo);
  return r;
}

/**
 * @brief Reassemble a pair-packed FP4 E2M1FN byte from a decomposition
 *        produced by @ref fp4e2m1x2_to_f32_bits.
 *
 * @param[in] val  The decomposition to reassemble.  Must not be null.
 * @return The original pair-packed FP4 byte.
 */
float4_e2m1fn_x2 fp4e2m1x2_from_f32_bits(const fp4e2m1x2Result_t *val) {
  return val->val;
}

/**
 * @brief Return the raw storage bits of an IEEE 754 half-precision (FP16)
 *        value.
 *
 * @param[in] val  The FP16 value to inspect.
 * @return The raw 16-bit storage pattern of @p val.
 */
uint16_t fp16_to_bits(float16 val) {
#if defined(_GNUC_CLANG_)
  return std::bit_cast<uint16_t>(val);
#else
  return val;
#endif
}

/**
 * @brief Return the raw storage bits of a bfloat16 (Brain Float 16) value.
 *
 * @param[in] val  The bfloat16 value to inspect.
 * @return The raw 16-bit storage pattern of @p val.
 */
uint16_t bf16_to_bits(bfloat16 val) {
#if defined(_GNUC_CLANG_)
  return std::bit_cast<uint16_t>(val);
#else
  return val;
#endif
}
