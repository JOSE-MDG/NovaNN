/**
 * @file fp_utils.h
 * @brief C-compatible public API for reduced-precision floating-point
 *        conversions.
 *
 * @details
 * Provides the canonical C extern functions for converting between IEEE 754
 * single-precision float32 and each of the reduced-precision formats
 * supported by NovaNN:
 * @li FP16       (IEEE 754 half-precision)
 * @li BF16       (Brain Float 16, 1.8.7 layout)
 * @li FP8 E5M2   (8-bit, 1.5.2 layout, bias = 15, matches FP16 exponent)
 * @li FP8 E4M3FN (8-bit, 1.4.3 layout, bias = 7, finite only, no inf/NaN)
 * @li FP4 E2M1FN (4-bit, 1.2.1 layout, bias = 1, finite, pair-packed ×2)
 *
 * Two families of conversions are provided:
 * @li <b>Numeric conversions</b> — between a reduced type and a @c float
 *     value (e.g. @ref fp16_from_float, @ref fp16_to_float).
 * @li <b>Bit-pattern conversions</b> — between a reduced type and an integer
 *     representation of its bits:
 *     @li @c *_to_bits / @c *_from_f32_bits / @c *_to_f32_bits operate on
 *         raw bit patterns without materializing an intermediate @c float,
 *         useful for tests.
 *
 * Each conversion is implemented via a compile-time dispatch:
 * @li When the native compiler type is available (e.g. @c _Float16 on
 *   GCC/Clang, @c __bf16 on GCC/Clang, or CUDA/HIP device intrinsics),
 *   the conversion uses the corresponding hardware instruction.
 * @li Otherwise, a software bit-manipulation fallback from
 *   @c ncore/include/ncore/headeronly/dtypes/ is used, with
 *   round-to-nearest-even semantics where applicable.
 *
 * The implementation lives in @c ncore/src/dtypes/DTypes.cpp and its
 * associated @c .hpp headers.
 *
 * @see DTypes.cpp       Implementation of these conversion functions.
 * @see half.hh          IEEE 754 half-precision (FP16).
 * @see bfloat16.hh      Brain Float 16 (BF16).
 * @see fp8_e5m2.hh      FP8 E5M2 (8-bit, 1.5.2 layout)
 * @see fp8_e4m3fn.hh    FP8 E4M3FN (8-bit, 1.4.3 layout).
 * @see fp4_e2m1fn_x2.hh FP4 E2M1FN pair-packed (4-bit, 1.2.1 layout)
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
 * The implementation reuses the FP16 bit-manipulation helpers after a
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
 * The @c fn suffix denotes @c finite — this format has no infinity encoding;
 * values that overflow (including +/-inf and NaN inputs) saturate to the
 * maximum finite magnitude (448.0, bit pattern 0x7E).  NaN is also
 * mapped to 0x7F, the canonical E4M3FN NaN pattern.
 *
 * @param[in] val  The float value to convert.
 * @return The FP8 E4M3FN representation of @p val.
 */
extern float8_e4m3fn fp8e4m3fn_from_float(float val);

/**
 * @brief Convert an FP8 E4M3FN value to single-precision float.
 *
 * Purely integer-based bit-manipulation routine that handles normals,
 * denormals, zero, and NaN without triggering floating-point exceptions.
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
 * per lane.  The @c fn suffix denotes @c finite — this format has no
 * infinity or NaN encoding.  Out-of-range inputs (including +/-inf and
 * NaN) saturate to the maximum finite magnitude (6.0).
 *
 * The packed byte layout (MSB to LSB): high nibble = hi, low nibble = lo.
 *
 * @param[in] lo  The value to pack into the low nibble.
 * @param[in] hi  The value to pack into the high nibble.
 * @return The pair-packed FP4 byte.
 */
extern float4_e2m1fn_x2 fp4e2m1x2_from_floats(float lo, float hi);

/**
 * @brief Unpack a single FP4 E2M1FN pair-packed byte into two
 *        single-precision floats.
 *
 * Each nibble is decoded independently via a direct lookup table
 * (only 8 possible finite magnitudes, so a table is the simplest
 * and most auditable implementation).
 *
 * @param[in]  val  The pair-packed FP4 byte to unpack.
 * @param[out] lo   Receives the decoded value of the low nibble.
 * @param[out] hi   Receives the decoded value of the high nibble.
 */
extern void fp4e2m1x2_to_floats(float4_e2m1fn_x2 val, float *lo, float *hi);

/**
 * @struct fp4e2m1x2Result_t
 * @brief Result of decomposing a pair-packed FP4 E2M1FN byte into its
 *        constituent nibbles.
 *
 * @c lo and @c hi hold the raw 4-bit values of the low and high nibbles
 * respectively; @c val reproduces the original pair-packed byte as
 * @c (hi << 4) | lo.
 */
struct fp4e2m1x2Result_t;

/**
 * @brief Convert an IEEE 754 half-precision (FP16) value to the bit pattern
 *        of its single-precision float32 equivalent.
 *
 * The bit-level counterpart of @ref fp16_to_float: instead of returning a
 * @c float, the raw 32-bit pattern of the equivalent float32 value is
 * returned.  On GCC/Clang the conversion goes through the native @c
 * _Float16 compiler type; otherwise it uses the software
 * @ref ncore::dtypes::detail::fp16_ieee_to_fp32_bits routine.
 *
 * @param[in] val  The FP16 value to convert.
 * @return The IEEE 754 float32 bit pattern of @p val.
 */
extern uint32_t fp16_to_f32_bits(float16 val);

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
extern float16 fp16_from_f32_bits(uint32_t val);

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
extern uint16_t fp16_to_bits(float16 val);

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
extern uint32_t bf16_to_f32_bits(bfloat16 val);

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
extern bfloat16 bf16_from_f32_bits(uint32_t val);

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
extern uint16_t bf16_to_bits(bfloat16 val);

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
extern uint32_t fp8e5m2_to_f32_bits(float8_e5m2 val);

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
extern float8_e5m2 fp8e5m2_from_f32_bits(uint32_t val);

/**
 * @brief Convert an FP8 E4M3FN value to the bit pattern of its
 *        single-precision float32 equivalent.
 *
 * @param[in] val  The FP8 E4M3FN value to convert.
 * @return The float32 bit pattern of @p val.
 */
extern uint32_t fp8e4m3fn_to_f32_bits(float8_e4m3fn val);

/**
 * @brief Convert the bit pattern of a single-precision float to an FP8
 *        E4M3FN value.
 *
 * The bit-level counterpart of @ref fp8e4m3fn_from_float.  E4M3FN is a
 * finite-only format: out-of-range values (including +/-inf and NaN inputs)
 * saturate to the maximum finite magnitude (448.0, pattern 0x7E); NaN is
 * mapped to the canonical E4M3FN NaN pattern (0x7F).
 *
 * @param[in] val  The IEEE 754 float32 bit pattern to convert.
 * @return The FP8 E4M3FN representation of the value described by @p val.
 */
extern float8_e4m3fn fp8e4m3fn_from_f32_bits(uint32_t val);

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
extern struct fp4e2m1x2Result_t fp4e2m1x2_to_f32_bits(float4_e2m1fn_x2 val);

/**
 * @brief Reassemble a pair-packed FP4 E2M1FN byte from a decomposition
 *        produced by @ref fp4e2m1x2_to_f32_bits.
 *
 * Returns the @c val member of @p val — the original packed byte.
 *
 * @param[in] val  The decomposition to reassemble.  Must not be null.
 * @return The original pair-packed FP4 byte.
 */
extern float4_e2m1fn_x2
fp4e2m1x2_from_f32_bits(const struct fp4e2m1x2Result_t *val);

#ifdef __cplusplus
}
#endif
