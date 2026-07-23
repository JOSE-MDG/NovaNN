/**
 * @file DTypes.hpp
 * @brief Internal C-linkage declarations for reduced-precision float
 *        conversions.
 *
 * Every function declared here forwards to the implementation in DTypes.cpp,
 * which selects either a compiler builtin or a software fallback at compile
 * time.  The public-facing declarations with full Doxygen live in
 * fp_utils.h — this header exists solely as a compilation-unit boundary
 * between DTypes.cpp and its callers within ncore/src/dtypes/.
 *
 * @see fp_utils.h
 * @see DTypes.cpp
 */

#pragma once

#include <ncore/core/dtype.h>

extern "C" {

/**
 * @brief Convert a single-precision float to IEEE 754 half-precision (FP16).
 */
float16 fp16_from_float(float val);

/**
 * @brief Convert an IEEE 754 half-precision (FP16) value to single-precision
 * float.
 */
float fp16_to_float(float16 val);

/**
 * @brief Convert a single-precision float to bfloat16 (Brain Float 16).
 */
bfloat16 bf16_from_float(float val);

/**
 * @brief Convert a bfloat16 (Brain Float 16) value to single-precision float.
 */
float bf16_to_float(bfloat16 val);

/**
 * @brief Convert a single-precision float to FP8 E5M2 format.
 */
float8_e5m2 fp8e5m2_from_float(float val);

/**
 * @brief Convert an FP8 E5M2 value to single-precision float.
 */
float fp8e5m2_to_float(float8_e5m2 val);

/**
 * @brief Convert a single-precision float to FP8 E4M3FN format.
 */
float8_e4m3fn fp8e4m3fn_from_float(float val);

/**
 * @brief Convert an FP8 E4M3FN value to single-precision float.
 */
float fp8e4m3fn_to_float(float8_e4m3fn val);

/**
 * @brief Pack two single-precision floats into a single FP4 E2M1FN pair-packed
 * byte.
 */
float4_e2m1fn_x2 fp4e2m1x2_from_floats(float lo, float hi);

/**
 * @brief Unpack a single FP4 E2M1FN pair-packed byte into two single-precision
 * floats.
 */
void fp4e2m1x2_to_floats(float4_e2m1fn_x2 val, float *lo, float *hi);
}
