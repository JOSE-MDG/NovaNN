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
 * Two families of conversions are declared:
 *   @li Numeric conversions — between a reduced type and a @c float value
 *       (e.g. @ref fp16_from_float).  These are re-exported publicly in
 *       fp_utils.h.
 *   @li Bit-pattern conversions — between a reduced type and an integer
 *       representation of its bits.  Two flavors exist:
 *       @li @c *_to_bits (e.g. @ref fp16_to_bits) — the raw storage bits
 *           of the reduced type itself, without any widening.
 *       @li @c *_to_f32_bits / @c *_from_f32_bits (e.g.
 *           @ref fp16_to_f32_bits) — the bit pattern of the value's
 *           float32 equivalent.
 *       These are used internally for serialization, repr formatting and
 *       inspection, where a value is available as an integer bit pattern
 *       and materializing an intermediate @c float is undesirable.
 *
 * @see fp_utils.h
 * @see DTypes.cpp
 */

#pragma once

#include <ncore/core/dtype.h>

extern "C" {

/**
 * @struct fp4e2m1x2Result_t
 * @brief Result of decomposing a pair-packed FP4 E2M1FN byte into its
 *        constituent nibbles.
 *
 * @c lo and @c hi hold the raw 4-bit values of the low and high nibbles
 * respectively; @c val reproduces the original pair-packed byte as
 * @c (hi << 4) | lo.
 */
struct fp4e2m1x2Result_t {
  uint32_t lo;
  uint32_t hi;
  float4_e2m1fn_x2 val;
};

/**
 * @brief Convert a single-precision float to IEEE 754 half-precision (FP16).
 *
 * @param[in] val  The single-precision float value to convert.
 * @return The FP16 representation of @p val.
 */
float16 fp16_from_float(float val);

/**
 * @brief Convert an IEEE 754 half-precision (FP16) value to single-precision
 * float.
 *
 * @param[in] val  The FP16 value to convert.
 * @return The single-precision float equivalent of @p val.
 */
float fp16_to_float(float16 val);

/**
 * @brief Convert a single-precision float to bfloat16 (Brain Float 16).
 *
 * @param[in] val  The single-precision float value to convert.
 * @return The bfloat16 representation of @p val.
 */
bfloat16 bf16_from_float(float val);

/**
 * @brief Convert a bfloat16 (Brain Float 16) value to single-precision float.
 *
 * @param[in] val  The bfloat16 value to convert.
 * @return The single-precision float equivalent of @p val.
 */
float bf16_to_float(bfloat16 val);

/**
 * @brief Convert a single-precision float to FP8 E5M2 format.
 *
 * @param[in] val  The single-precision float value to convert.
 * @return The FP8 E5M2 representation of @p val.
 */
float8_e5m2 fp8e5m2_from_float(float val);

/**
 * @brief Convert an FP8 E5M2 value to single-precision float.
 *
 * @param[in] val  The FP8 E5M2 value to convert.
 * @return The single-precision float equivalent of @p val.
 */
float fp8e5m2_to_float(float8_e5m2 val);

/**
 * @brief Convert a single-precision float to FP8 E4M3FN format.
 *
 * @param[in] val  The single-precision float value to convert.
 * @return The FP8 E4M3FN representation of @p val.
 */
float8_e4m3fn fp8e4m3fn_from_float(float val);

/**
 * @brief Convert an FP8 E4M3FN value to single-precision float.
 *
 * @param[in] val  The FP8 E4M3FN value to convert.
 * @return The single-precision float equivalent of @p val.
 */
float fp8e4m3fn_to_float(float8_e4m3fn val);

/**
 * @brief Pack two single-precision floats into a single FP4 E2M1FN pair-packed
 * byte.
 *
 * @param[in] lo  The low-lane float value to pack.
 * @param[in] hi  The high-lane float value to pack.
 * @return The pair-packed FP4 byte.
 */
float4_e2m1fn_x2 fp4e2m1x2_from_floats(float lo, float hi);

/**
 * @brief Unpack a single FP4 E2M1FN pair-packed byte into two single-precision
 * floats.
 *
 * @param[in]  val  The pair-packed FP4 byte to unpack.
 * @param[out] lo   Receives the low-lane float value.
 * @param[out] hi   Receives the high-lane float value.
 */
void fp4e2m1x2_to_floats(float4_e2m1fn_x2 val, float *lo, float *hi);

/**
 * @brief Convert an IEEE 754 half-precision (FP16) value to the bit pattern
 *        of its single-precision float32 equivalent.
 *
 * The bit-level counterpart of @ref fp16_to_float: instead of returning a
 * @c float, the raw 32-bit pattern of the equivalent float32 value is
 * returned.  On GCC/Clang the conversion goes through the native @c _Float16
 * compiler type; otherwise it uses the software
 * @ref ncore::dtypes::detail::fp16_ieee_to_fp32_bits routine.
 *
 * @param[in] val  The FP16 value to convert.
 * @return The IEEE 754 float32 bit pattern of @p val.
 */
uint32_t fp16_to_f32_bits(float16 val);

/**
 * @brief Convert the bit pattern of a single-precision float to an IEEE 754
 *        half-precision (FP16) value.
 *
 * The bit-level counterpart of @ref fp16_from_float: the input is a raw
 * float32 bit pattern (e.g. read from serialized data) rather than a @c float
 * value.  Rounding follows round-to-nearest-even.
 *
 * @param[in] val  The IEEE 754 float32 bit pattern to convert.
 * @return The FP16 representation of the value described by @p val.
 */
float16 fp16_from_f32_bits(uint32_t val);

/**
 * @brief Return the raw storage bits of an IEEE 754 half-precision (FP16)
 *        value.
 *
 * Unlike @ref fp16_to_f32_bits, which returns the bit pattern of the
 * equivalent float32 value, this returns the 16-bit pattern of the FP16
 * value itself — the exact bytes a memory dump would contain.  On
 * GCC/Clang the pattern is obtained via @c std::bit_cast through the native
 * @c _Float16 type; otherwise the reduced type already stores the pattern
 * directly.
 *
 * @param[in] val  The FP16 value to inspect.
 * @return The raw 16-bit storage pattern of @p val.
 */
uint16_t fp16_to_bits(float16 val);

/**
 * @brief Convert a bfloat16 (Brain Float 16) value to the bit pattern of its
 *        single-precision float32 equivalent.
 *
 * BF16 shares the exponent width and bias of IEEE 754 float32, so the
 * conversion is a pure bit-shift: the 16-bit pattern is zero-extended into
 * the high 16 bits of the 32-bit pattern.
 *
 * @param[in] val  The bfloat16 value to convert.
 * @return The float32 bit pattern of @p val (low 16 bits are zero).
 */
uint32_t bf16_to_f32_bits(bfloat16 val);

/**
 * @brief Convert the bit pattern of a single-precision float to a bfloat16
 *        (Brain Float 16) value.
 *
 * The bit-level counterpart of @ref bf16_from_float.  The mantissa is
 * reduced to 7 bits with round-to-nearest-even semantics.
 *
 * @param[in] val  The IEEE 754 float32 bit pattern to convert.
 * @return The bfloat16 representation of the value described by @p val.
 */
bfloat16 bf16_from_f32_bits(uint32_t val);

/**
 * @brief Return the raw storage bits of a bfloat16 (Brain Float 16) value.
 *
 * Unlike @ref bf16_to_f32_bits, which zero-extends the pattern into a
 * 32-bit float pattern, this returns the 16-bit pattern of the BF16 value
 * itself — the exact bytes a memory dump would contain.
 *
 * @param[in] val  The bfloat16 value to inspect.
 * @return The raw 16-bit storage pattern of @p val.
 */
uint16_t bf16_to_bits(bfloat16 val);

/**
 * @brief Convert an FP8 E5M2 value to the bit pattern of its
 *        single-precision float32 equivalent.
 *
 * The value is widened to float32 (E5M2 shares FP16's exponent width and
 * bias) and the resulting bit pattern is returned as a @c uint32_t.
 *
 * @param[in] val  The FP8 E5M2 value to convert.
 * @return The float32 bit pattern of @p val.
 */
uint32_t fp8e5m2_to_f32_bits(float8_e5m2 val);

/**
 * @brief Convert the bit pattern of a single-precision float to an FP8 E5M2
 *        value.
 *
 * The bit-level counterpart of @ref fp8e5m2_from_float.  Values that
 * overflow the representable range saturate to +/-inf (the E5M2 infinity
 * encoding); NaN input maps to the canonical E5M2 NaN pattern (0x7F).
 *
 * @param[in] val  The IEEE 754 float32 bit pattern to convert.
 * @return The FP8 E5M2 representation of the value described by @p val.
 */
float8_e5m2 fp8e5m2_from_f32_bits(uint32_t val);

/**
 * @brief Convert an FP8 E4M3FN value to the bit pattern of its
 *        single-precision float32 equivalent.
 *
 * @param[in] val  The FP8 E4M3FN value to convert.
 * @return The float32 bit pattern of @p val.
 */
uint32_t fp8e4m3fn_to_f32_bits(float8_e4m3fn val);

/**
 * @brief Convert the bit pattern of a single-precision float to an FP8
 *        E4M3FN value.
 *
 * The bit-level counterpart of @ref fp8e4m3fn_from_float.  E4M3FN is a
 * finite-only format: NaN maps to the canonical NaN pattern (0x7F); other
 * out-of-range values (including +/-inf) saturate to the maximum finite
 * magnitude (448.0, pattern 0x7E).
 *
 * @param[in] val  The IEEE 754 float32 bit pattern to convert.
 * @return The FP8 E4M3FN representation of the value described by @p val.
 */
float8_e4m3fn fp8e4m3fn_from_f32_bits(uint32_t val);

/**
 * @brief Decompose a pair-packed FP4 E2M1FN byte into its constituent
 *        nibbles.
 *
 * Splits the packed byte into the raw low and high 4-bit values; @c val of
 * the result reproduces the original byte as @c (hi << 4) | lo.  The nibbles
 * are raw FP4 bit patterns (sign : exponent : mantissa), not decoded
 * magnitudes — use @ref fp4e2m1x2_to_floats to obtain numeric values.
 *
 * @param[in] val  The pair-packed FP4 byte to decompose.
 * @return The decomposed nibbles together with the original byte.
 */
fp4e2m1x2Result_t fp4e2m1x2_to_f32_bits(float4_e2m1fn_x2 val);

/**
 * @brief Reassemble a pair-packed FP4 E2M1FN byte from a decomposition
 *        produced by @ref fp4e2m1x2_to_f32_bits.
 *
 * Returns the @c val member of @p val — the original packed byte.
 *
 * @param[in] val  The decomposition to reassemble.  Must not be null.
 * @return The original pair-packed FP4 byte.
 */
float4_e2m1fn_x2 fp4e2m1x2_from_f32_bits(const fp4e2m1x2Result_t *val);
}
