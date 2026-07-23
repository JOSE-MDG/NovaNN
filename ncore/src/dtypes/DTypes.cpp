/**
 * @file DTypes.cpp
 * @brief Compile-time dispatch implementation for reduced-precision
 *        float conversions.
 *
 * Each (Except FP4 and FP8) conversion function selects between two strategies
 * at compile time:
 *
 *   - **Compiler builtin** — when the platform provides a native type
 *     (`_Float16` for FP16, `__bf16` for BF16 on GCC/Clang) the conversion
 *     is a simple `static_cast`, producing optimal code.
 *   - **Software fallback** — an integer-only bit-manipulation routine
 *     from the corresponding `ncore/headeronly/dtypes/<file>.hh` header.  The
 *     fallback is pulled in via a conditional `#include` on the matching
 *     `*.hpp` wrapper in this directory.
 *
 * FP8 formats (E5M2, E4M3FN) and FP4 always use the software path because
 * no mainstream compiler exposes a native 8-bit or 4-bit float type.
 *
 * The functions are declared with `extern "C"` linkage in DTypes.hpp and
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

#include <ncore/core/dtype.h>
#include <ncore/headeronly/macros.h>

#include "DTypes.hpp"
#include "Float4_e2m1fn_x2.hpp"
#include "Float8_e4m3fn.hpp"
#include "Float8_e5m2.hpp"

/**
 * @brief Convert a single-precision float to IEEE 754 half-precision (FP16).
 */
float16 fp16_from_float(float val) {
#ifdef _GNUC_CLANG_
  return static_cast<float16>(val);
#else
#include "Half.hpp"
  return ncore::dtypes::detail::fp16_ieee_from_fp32_value(val)
#endif
}

/**
 * @brief Convert an IEEE 754 half-precision (FP16) value to single-precision
 * float.
 */
float fp16_to_float(float16 val) {
#ifdef _GNUC_CLANG_
  return static_cast<float>(val);
#else
#include "Half.hpp"
  return ncore::dtypes::detail::fp16_ieee_to_fp32_value(val)
#endif
}

/**
 * @brief Convert a single-precision float to bfloat16 (Brain Float 16).
 */
bfloat16 bf16_from_float(float val) {
#ifdef _GNUC_CLANG_
  return static_cast<bfloat16>(val);
#else
#include "BFloat16.hpp"
  return ncore::dtypes::detail::bits_from_f32(val)
#endif
}

/**
 * @brief Convert a bfloat16 (Brain Float 16) value to single-precision float.
 */
float bf16_to_float(bfloat16 val) {
#ifdef _GNUC_CLANG_
  return static_cast<float>(val);
#else
#include "BFloat16.hpp"
  return ncore::dtypes::detail::f32_from_bits(val)
#endif
}

/**
 * @brief Convert a single-precision float to FP8 E5M2 format.
 */
float8_e5m2 fp8e5m2_from_float(float val) {
  return ncore::dtypes::detail::fp8e5m2_from_fp32_value(val);
}

/**
 * @brief Convert an FP8 E5M2 value to single-precision float.
 */
float fp8e5m2_to_float(float8_e5m2 val) {
  return ncore::dtypes::detail::fp8e5m2_to_fp32_value(val);
}

/**
 * @brief Convert a single-precision float to FP8 E4M3FN format.
 */
float8_e4m3fn fp8e4m3fn_from_float(float val) {
  return ncore::dtypes::detail::fp8e4m3fn_from_fp32_value(val);
}

/**
 * @brief Convert an FP8 E4M3FN value to single-precision float.
 */
float fp8e4m3fn_to_float(float8_e4m3fn val) {
  return ncore::dtypes::detail::fp8e4m3fn_to_fp32_value(val);
}

/**
 * @brief Pack two single-precision floats into a single FP4 E2M1FN pair-packed
 * byte.
 */
float4_e2m1fn_x2 fp4e2m1x2_from_floats(float lo, float hi) {
  const ncore::dtypes::Float4_e2m1fn_x2 packed(lo, hi);
  return static_cast<float4_e2m1fn_x2>(packed.val_);
}

/**
 * @brief Unpack a single FP4 E2M1FN pair-packed byte into two single-precision
 * floats.
 */
void fp4e2m1x2_to_floats(float4_e2m1fn_x2 val, float *lo, float *hi) {
  const ncore::dtypes::Float4_e2m1fn_x2 packed(static_cast<uint8_t>(val));
  *lo = static_cast<float>(packed.low());
  *hi = static_cast<float>(packed.high());
}
