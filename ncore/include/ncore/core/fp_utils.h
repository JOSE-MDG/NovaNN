/**
 * @file fp_utils.h
 * @brief C API for reduced-precision floating-point conversions.
 *
 * Covers FP16, BF16, FP8 E5M2, FP8 E4M3FN and FP4 E2M1FN (pair-packed).
 * Each conversion has a numeric form and a bit-pattern form. When the
 * native compiler type exists (@c _Float16, @c __bf16, CUDA/HIP
 * intrinsics) the hardware conversion is used, otherwise the software
 * fallbacks in @c ncore/include/ncore/headeronly/dtypes/ apply, with
 * round-to-nearest-even semantics where the format requires rounding.
 */

#pragma once

#include <ncore/core/dtype.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Convert a single-precision float to IEEE 754 half-precision (FP16).
 *
 * On GCC/Clang the conversion uses the native @c _Float16 compiler type;
 * otherwise it falls back to a software bit-manipulation routine with
 * round-to-nearest-even.
 *
 * @param[in] val  The float value to convert.  NaN, +/-inf, and values
 *                 outside the representable FP16 range are handled
 *                 according to IEEE 754 rules.
 * @return The half-precision representation of @p val.
 */
extern float16 fp16_from_float(float val);

/**
 * @brief Convert an IEEE 754 half-precision (FP16) value to
 *        single-precision float.
 *
 * On GCC/Clang the conversion uses the native @c _Float16 compiler type;
 * otherwise it falls back to a software bit-manipulation routine.
 *
 * @param[in] val  The FP16 value to convert.
 * @return The single-precision float representation of @p val.
 */
extern float fp16_to_float(float16 val);

/**
 * @brief Convert a single-precision float to bfloat16 (Brain Float 16).
 *
 * On GCC/Clang the conversion uses the native @c __bf16 compiler type;
 * otherwise it uses round-to-nearest-even via @ref
 * ncore::dtypes::detail::round_to_nearest_even.
 *
 * @param[in] val  The float value to convert.  NaN is mapped to the
 *                 canonical bf16 NaN pattern (0x7FC0).
 * @return The bfloat16 representation of @p val.
 */
extern bfloat16 bf16_from_float(float val);

/**
 * @brief Convert a bfloat16 (Brain Float 16) value to single-precision float.
 *
 * On GCC/Clang the conversion uses the native @c __bf16 compiler type;
 * otherwise it zero-extends the bf16 bit pattern into the high 16 bits of
 * a 32-bit float (pure bit-shift; no special-case handling is required
 * because BF16 shares the exponent width and bias of IEEE 754 float32).
 *
 * @param[in] val  The bfloat16 value to convert.
 * @return The single-precision float representation of @p val.
 */
extern float bf16_to_float(bfloat16 val);

/**
 * @brief Convert a single-precision float to FP8 E5M2 format.
 *
 * E5M2 has 1 sign bit, 5 exponent bits (bias = 15, same as FP16) and 2
 * mantissa bits.  Values that overflow the representable range saturate
 * to +/-inf (the E5M2 infinity encoding, 0x7C/0xFC).  NaN input is mapped
 * to the canonical E5M2 NaN pattern (0x7F).
 *
 * Implementation reuses the FP16 bit-manipulation helpers after a
 * bit-shift, the same approach as PyTorch's reference implementation.
 *
 * @param[in] val  The float value to convert.
 * @return The FP8 E5M2 representation of @p val.
 */
extern float8_e5m2 fp8e5m2_from_float(float val);

/**
 * @brief Convert an FP8 E5M2 value to single-precision float.
 *
 * Zero-extends the 8-bit pattern into the upper 8 bits of a 16-bit half
 * and reuses the IEEE 754 FP16-to-float conversion routine (E5M2 shares
 * FP16's exponent width and bias).
 *
 * @param[in] val  The FP8 E5M2 value to convert.
 * @return The single-precision float representation of @p val.
 */
extern float fp8e5m2_to_float(float8_e5m2 val);

/**
 * @brief Convert a single-precision float to FP8 E4M3FN format.
 *
 * E4M3FN has 1 sign bit, 4 exponent bits (bias = 7) and 3 mantissa bits.
 * The @c fn suffix is "finite": the format has no infinity encoding, so
 * overflowing values (including +/-inf and NaN inputs) saturate to the
 * maximum finite magnitude (448.0, pattern 0x7E).  NaN input maps to
 * 0x7F, the canonical E4M3FN NaN pattern.
 *
 * @param[in] val  The float value to convert.
 * @return The FP8 E4M3FN representation of @p val.
 */
extern float8_e4m3fn fp8e4m3fn_from_float(float val);

/**
 * @brief Convert an FP8 E4M3FN value to single-precision float.
 *
 * Integer-only bit manipulation that handles normals, denormals, zero
 * and NaN without raising floating-point exceptions.
 *
 * @param[in] val  The FP8 E4M3FN value to convert.
 * @return The single-precision float representation of @p val.
 */
extern float fp8e4m3fn_to_float(float8_e4m3fn val);

/**
 * @brief Pack two single-precision floats into a single FP4 E2M1FN
 *        pair-packed byte.
 *
 * E2M1FN has 1 sign bit, 2 exponent bits (bias = 1) and 1 mantissa bit
 * per lane. The format is finite only, so out-of-range inputs
 * (including +/-inf and NaN) saturate to the maximum finite magnitude
 * (6.0).
 *
 * @param[in] lo  The value to pack into the low nibble.
 * @param[in] hi  The value to pack into the high nibble.
 * @return The pair-packed byte, high nibble = @p hi, low nibble = @p lo.
 */
extern float4_e2m1fn_x2 fp4e2m1x2_from_floats(float lo, float hi);

/**
 * @brief Unpack a single FP4 E2M1FN pair-packed byte into two
 *        single-precision floats.
 *
 * Each nibble goes through a direct lookup table; with only 8 possible
 * finite magnitudes the table is the simplest implementation.
 *
 * @param[in]  val  The pair-packed FP4 byte to unpack.
 * @param[out] lo   Receives the value of the low nibble.
 * @param[out] hi   Receives the value of the high nibble.
 */
extern void fp4e2m1x2_to_floats(float4_e2m1fn_x2 val, float *lo, float *hi);

/**
 * @struct fp4e2m1x2Result_t
 * @brief Raw nibbles of a pair-packed FP4 E2M1FN byte.
 *
 * @c lo and @c hi hold the low and high 4-bit values. @c val
 * reproduces the original byte as @c (hi << 4) | lo.
 */
struct fp4e2m1x2Result_t {
  uint32_t lo;
  uint32_t hi;
  float4_e2m1fn_x2 val;
};

/**
 * @brief Convert an FP16 value to the bit pattern of its float32
 *        equivalent.
 *
 * Bit-level counterpart of @ref fp16_to_float.  On GCC/Clang the
 * conversion goes through the native @c _Float16 type, otherwise the
 * software @ref ncore::dtypes::detail::fp16_ieee_to_fp32_bits runs.
 *
 * @param[in] val  The FP16 value to convert.
 * @return The float32 bit pattern of @p val.
 */
extern uint32_t fp16_to_f32_bits(float16 val);

/**
 * @brief Convert a float32 bit pattern to an IEEE 754 half-precision
 *        (FP16) value.
 *
 * Bit-level counterpart of @ref fp16_from_float for patterns read from
 * serialized data.
 *
 * @param[in] val  The float32 bit pattern to convert.
 * @return The FP16 representation of the value described by @p val.
 */
extern float16 fp16_from_f32_bits(uint32_t val);

/**
 * @brief Return the raw 16-bit storage pattern of an FP16 value.
 *
 * Unlike @ref fp16_to_f32_bits (the widened float32 pattern) this is
 * what a memory dump would contain.
 *
 * @param[in] val  The FP16 value to inspect.
 * @return The raw 16-bit storage pattern of @p val.
 */
extern uint16_t fp16_to_bits(float16 val);

/**
 * @brief Convert a bfloat16 value to the bit pattern of its float32
 *        equivalent.
 *
 * A pure bit-shift: BF16 shares the exponent width and bias of float32,
 * so the pattern is zero-extended into the high 16 bits.
 *
 * @param[in] val  The bfloat16 value to convert.
 * @return The float32 bit pattern of @p val (low 16 bits are zero).
 */
extern uint32_t bf16_to_f32_bits(bfloat16 val);

/**
 * @brief Convert a float32 bit pattern to a bfloat16 value.
 *
 * The mantissa reduces to 7 bits with round-to-nearest-even.
 *
 * @param[in] val  The float32 bit pattern to convert.
 * @return The bfloat16 representation of the value described by @p val.
 */
extern bfloat16 bf16_from_f32_bits(uint32_t val);

/**
 * @brief Return the raw 16-bit storage pattern of a bfloat16 value.
 *
 * Counterpart of @ref bf16_to_f32_bits for the un-widened pattern.
 *
 * @param[in] val  The bfloat16 value to inspect.
 * @return The raw 16-bit storage pattern of @p val.
 */
extern uint16_t bf16_to_bits(bfloat16 val);

/**
 * @brief Convert an FP8 E5M2 value to the bit pattern of its float32
 *        equivalent.
 *
 * The value is widened to float32 (E5M2 shares FP16's exponent width and
 * bias) and the pattern returned as a @c uint32_t.
 *
 * @param[in] val  The FP8 E5M2 value to convert.
 * @return The float32 bit pattern of @p val.
 */
extern uint32_t fp8e5m2_to_f32_bits(float8_e5m2 val);

/**
 * @brief Convert a float32 bit pattern to an FP8 E5M2 value.
 *
 * Bit-level counterpart of @ref fp8e5m2_from_float, with the same
 * saturation semantics.
 *
 * @param[in] val  The float32 bit pattern to convert.
 * @return The FP8 E5M2 representation of the value described by @p val.
 */
extern float8_e5m2 fp8e5m2_from_f32_bits(uint32_t val);

/**
 * @brief Convert an FP8 E4M3FN value to the bit pattern of its float32
 *        equivalent.
 *
 * @param[in] val  The FP8 E4M3FN value to convert.
 * @return The float32 bit pattern of @p val.
 */
extern uint32_t fp8e4m3fn_to_f32_bits(float8_e4m3fn val);

/**
 * @brief Convert a float32 bit pattern to an FP8 E4M3FN value.
 *
 * Finite-only format: out-of-range values (including +/-inf and NaN
 * inputs) saturate to the maximum finite magnitude 448.0 (0x7E), NaN
 * input maps to the canonical pattern (0x7F).
 *
 * @param[in] val  The float32 bit pattern to convert.
 * @return The FP8 E4M3FN representation of the value described by @p val.
 */
extern float8_e4m3fn fp8e4m3fn_from_f32_bits(uint32_t val);

/**
 * @brief Decompose a pair-packed FP4 E2M1FN byte into its constituent
 *        nibbles.
 *
 * Nibbles are raw FP4 bit patterns, not decoded magnitudes (use
 * @ref fp4e2m1x2_to_floats for those). @c val of the result reproduces
 * the original byte as @c (hi << 4) | lo.
 *
 * @param[in] val  The pair-packed FP4 byte to decompose.
 * @return The decomposed nibbles together with the original byte.
 */
extern struct fp4e2m1x2Result_t fp4e2m1x2_to_f32_bits(float4_e2m1fn_x2 val);

/**
 * @brief Reassemble a pair-packed FP4 E2M1FN byte from a decomposition
 *        produced by @ref fp4e2m1x2_to_f32_bits.
 *
 * @param[in] val  The decomposition to reassemble.  Must not be null.
 * @return The @c val member of @p val.
 */
extern float4_e2m1fn_x2
fp4e2m1x2_from_f32_bits(const struct fp4e2m1x2Result_t *val);

#ifdef __cplusplus
}
#endif
