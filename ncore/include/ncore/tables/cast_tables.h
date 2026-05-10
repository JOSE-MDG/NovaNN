/**
 * @file cast_tables.h
 * @brief Tensor element-wise type cast dispatch tables.
 *
 * Declares lookup tables for tensor type conversions. Each table lists
 * implementations in descending capability order (index 0 = fastest SIMD,
 * last index = scalar fallback). The caller selects the first entry whose
 * required ISA is present at runtime.
 *
 */

#pragma once

#include <ncore/tensor.h>

/**
 * @brief Cast function pointer type.
 * @param src Source tensor (read-only)
 * @param dst Destination tensor (write-only, must be same size as src)
 */
typedef void (*castFn)(const Tensor *restrict, Tensor *restrict);

/* =========================================================================
 * Floating-point to floating-point conversions (12 tables)
 * ========================================================================= */

/**
 * @brief Dispatch table for fp16 to f32 conversions.
 *
 * Variants:
 * - Index 0: tfp16_to_f32_avx512_() — Requires: AVX512F
 * - Index 1: tfp16_to_f32_avx_avx2_fp16c_() — Requires: F16C
 * - Index 2: tfp16_to_f32_scalar_() — Portable fallback
 */
extern const castFn lookup_tfp16_to_f32_[];

/**
 * @brief Dispatch table for fp16 to f64 conversions.
 *
 * Variants:
 * - Index 0: tfp16_to_f64_avx512_() — Requires: AVX512F
 * - Index 1: tfp16_to_f64_avx_avx2_fp16c_() — Requires: F16C
 * - Index 2: tfp16_to_f64_scalar_() — Portable fallback
 */
extern const castFn lookup_tfp16_to_f64_[];

/**
 * @brief Dispatch table for fp16 to bf16 conversions.
 *
 * Variants:
 * - Index 0: tfp16_to_bf16_avx512bf16_() — Requires: AVX512F, AVX512BF16
 * - Index 1: tfp16_to_bf16_scalar_() — Portable fallback
 */
extern const castFn lookup_tfp16_to_bf16_[];

/**
 * @brief Dispatch table for f32 to fp16 conversions.
 *
 * Variants:
 * - Index 0: tf32_to_fp16_avx512fp16_() — Requires: AVX512F, AVX512FP16
 * - Index 1: tf32_to_fp16_avx_avx2_f16c_() — Requires: F16C
 * - Index 2: tf32_to_fp16_scalar_() — Portable fallback
 */
extern const castFn lookup_tf32_to_fp16_[];

/**
 * @brief Dispatch table for f32 to f64 conversions.
 *
 * Variants:
 * - Index 0: tf32_to_f64_avx512_() — Requires: AVX512F
 * - Index 1: tf32_to_f64_avx_avx2_() — Requires: AVX/AVX2
 * - Index 2: tf32_to_f64_sse4_2_() — Requires: SSE4.2
 * - Index 3: tf32_to_f64_scalar_() — Portable fallback
 */
extern const castFn lookup_tf32_to_f64_[];

/**
 * @brief Dispatch table for f32 to bf16 conversions.
 *
 * Variants:
 * - Index 0: tf32_to_bf16_avx512bf16_() — Requires: AVX512F, AVX512BF16
 * - Index 1: tf32_to_bf16_scalar_() — Portable fallback
 */
extern const castFn lookup_tf32_to_bf16_[];

/**
 * @brief Dispatch table for bf16 to fp16 conversions.
 *
 * Variants:
 * - Index 0: tbf16_to_fp16_avx512bf16_fp16_() — Requires: AVX512F, AVX512BF16,
 * AVX512FP16
 * - Index 1: tbf16_to_fp16_scalar_() — Portable fallback
 */
extern const castFn lookup_tbf16_to_fp16_[];

/**
 * @brief Dispatch table for bf16 to f32 conversions.
 *
 * Variants:
 * - Index 0: tbf16_to_f32_avx512bf16_() — Requires: AVX512F, AVX512BF16
 * - Index 1: tbf16_to_f32_scalar_() — Portable fallback
 */
extern const castFn lookup_tbf16_to_f32_[];

/**
 * @brief Dispatch table for bf16 to f64 conversions.
 *
 * Variants:
 * - Index 0: tbf16_to_f64_avx512bf16_() — Requires: AVX512F, AVX512BF16
 * - Index 1: tbf16_to_f64_scalar_() — Portable fallback
 */
extern const castFn lookup_tbf16_to_f64_[];

/**
 * @brief Dispatch table for f64 to fp16 conversions.
 *
 * Variants:
 * - Index 0: tf64_to_fp16_avx512fp16_() — Requires: AVX512F, AVX512FP16
 * - Index 1: tf64_to_fp16_scalar_() — Portable fallback
 */
extern const castFn lookup_tf64_to_fp16_[];

/**
 * @brief Dispatch table for f64 to f32 conversions.
 *
 * Variants:
 * - Index 0: tf64_to_f32_avx512_() — Requires: AVX512F
 * - Index 1: tf64_to_f32_avx_avx2_() — Requires: AVX/AVX2
 * - Index 2: tf64_to_f32_sse4_2_() — Requires: SSE4.2
 * - Index 3: tf64_to_f32_scalar_() — Portable fallback
 */
extern const castFn lookup_tf64_to_f32_[];

/**
 * @brief Dispatch table for f64 to bf16 conversions.
 *
 * Variants:
 * - Index 0: tf64_to_bf16_avx512bf16_() — Requires: AVX512F, AVX512BF16
 * - Index 1: tf64_to_bf16_scalar_() — Portable fallback
 */
extern const castFn lookup_tf64_to_bf16_[];

/* =========================================================================
 * Floating-point to integer conversions (18 tables)
 * ========================================================================= */

/**
 * @brief Dispatch table for fp16 to s8 conversions.
 *
 * Variants:
 * - Index 0: tfp16_to_s8_avx512fp16_() — Requires: AVX512F, AVX512FP16, AVX512BW
 * - Index 1: tfp16_to_s8_scalar_() — Portable fallback
 */
extern const castFn lookup_tfp16_to_s8_[];

/**
 * @brief Dispatch table for fp16 to s32 conversions.
 *
 * Variants:
 * - Index 0: tfp16_to_s32_avx512fp16_() — Requires: AVX512F, AVX512FP16
 * - Index 1: tfp16_to_s32_scalar_() — Portable fallback
 */
extern const castFn lookup_tfp16_to_s32_[];

/**
 * @brief Dispatch table for fp16 to s64 conversions.
 *
 * Variants:
 * - Index 0: tfp16_to_s64_avx512fp16_() — Requires: AVX512F, AVX512FP16
 * - Index 1: tfp16_to_s64_scalar_() — Portable fallback
 */
extern const castFn lookup_tfp16_to_s64_[];

/**
 * @brief Dispatch table for fp16 to u8 conversions.
 *
 * Variants:
 * - Index 0: tfp16_to_u8_avx512fp16_() — Requires: AVX512F, AVX512FP16, AVX512BW
 * - Index 1: tfp16_to_u8_scalar_() — Portable fallback
 */
extern const castFn lookup_tfp16_to_u8_[];

/**
 * @brief Dispatch table for fp16 to u32 conversions.
 *
 * Variants:
 * - Index 0: tfp16_to_u32_avx512fp16_() — Requires: AVX512F, AVX512FP16
 * - Index 1: tfp16_to_u32_scalar_() — Portable fallback
 */
extern const castFn lookup_tfp16_to_u32_[];

/**
 * @brief Dispatch table for fp16 to u64 conversions.
 *
 * Variants:
 * - Index 0: tfp16_to_u64_avx512fp16_() — Requires: AVX512F, AVX512FP16
 * - Index 1: tfp16_to_u64_scalar_() — Portable fallback
 */
extern const castFn lookup_tfp16_to_u64_[];

/**
 * @brief Dispatch table for bf16 to s8 conversions.
 *
 * Variants:
 * - Index 0: tbf16_to_s8_avx512bf16_() — Requires: AVX512F, AVX512BF16, AVX512BW
 * - Index 1: tbf16_to_s8_scalar_() — Portable fallback
 */
extern const castFn lookup_tbf16_to_s8_[];

/**
 * @brief Dispatch table for bf16 to s32 conversions.
 *
 * Variants:
 * - Index 0: tbf16_to_s32_avx512bf16_() — Requires: AVX512F, AVX512BF16
 * - Index 1: tbf16_to_s32_scalar_() — Portable fallback
 */
extern const castFn lookup_tbf16_to_s32_[];

/**
 * @brief Dispatch table for bf16 to s64 conversions.
 *
 * Variants:
 * - Index 0: tbf16_to_s64_avx512bf16_() — Requires: AVX512F, AVX512BF16, AVX512DQ, AVX512VL
 * - Index 1: tbf16_to_s64_scalar_() — Portable fallback
 */
extern const castFn lookup_tbf16_to_s64_[];

/**
 * @brief Dispatch table for bf16 to u8 conversions.
 *
 * Variants:
 * - Index 0: tbf16_to_u8_avx512bf16_() — Requires: AVX512F, AVX512BF16, AVX512BW
 * - Index 1: tbf16_to_u8_scalar_() — Portable fallback
 */
extern const castFn lookup_tbf16_to_u8_[];

/**
 * @brief Dispatch table for bf16 to u32 conversions.
 *
 * Variants:
 * - Index 0: tbf16_to_u32_avx512bf16_() — Requires: AVX512F, AVX512BF16
 * - Index 1: tbf16_to_u32_scalar_() — Portable fallback
 */
extern const castFn lookup_tbf16_to_u32_[];

/**
 * @brief Dispatch table for bf16 to u64 conversions.
 *
 * Variants:
 * - Index 0: tbf16_to_u64_avx512bf16_() — Requires: AVX512F, AVX512BF16, AVX512DQ, AVX512VL
 * - Index 1: tbf16_to_u64_scalar_() — Portable fallback
 */
extern const castFn lookup_tbf16_to_u64_[];

/**
 * @brief Dispatch table for f32 to s8 conversions.
 *
 * Variants:
 * - Index 0: tf32_to_s8_avx512_() — Requires: AVX512F, AVX512BW
 * - Index 1: tf32_to_s8_avx2_() — Requires: AVX2
 * - Index 2: tf32_to_s8_scalar_() — Portable fallback
 */
extern const castFn lookup_tf32_to_s8_[];

/**
 * @brief Dispatch table for f32 to s32 conversions.
 *
 * Variants:
 * - Index 0: tf32_to_s32_avx512_() — Requires: AVX512F
 * - Index 1: tf32_to_s32_avx_avx2_() — Requires: AVX/AVX2
 * - Index 2: tf32_to_s32_sse4_2_() — Requires: SSE4.2
 * - Index 3: tf32_to_s32_scalar_() — Portable fallback
 */
extern const castFn lookup_tf32_to_s32_[];

/**
 * @brief Dispatch table for f32 to s64 conversions.
 *
 * Variants:
 * - Index 0: tf32_to_s64_avx512_() — Requires: AVX512F, AVX512DQ
 * - Index 1: tf32_to_s64_scalar_() — Portable fallback
 */
extern const castFn lookup_tf32_to_s64_[];

/**
 * @brief Dispatch table for f32 to u8 conversions.
 *
 * Variants:
 * - Index 0: tf32_to_u8_avx512_() — Requires: AVX512F, AVX512BW
 * - Index 1: tf32_to_u8_avx2_() — Requires: AVX2
 * - Index 2: tf32_to_u8_scalar_() — Portable fallback
 */
extern const castFn lookup_tf32_to_u8_[];

/**
 * @brief Dispatch table for f32 to u32 conversions.
 *
 * Variants:
 * - Index 0: tf32_to_u32_avx512_() — Requires: AVX512F
 * - Index 1: tf32_to_u32_scalar_() — Portable fallback
 */
extern const castFn lookup_tf32_to_u32_[];

/**
 * @brief Dispatch table for f32 to u64 conversions.
 *
 * Variants:
 * - Index 0: tf32_to_u64_avx512_() — Requires: AVX512F, AVX512DQ
 * - Index 1: tf32_to_u64_scalar_() — Portable fallback
 */
extern const castFn lookup_tf32_to_u64_[];

/**
 * @brief Dispatch table for f64 to s8 conversions.
 *
 * Variants:
 * - Index 0: tf64_to_s8_avx512_() — Requires: AVX512F, AVX512BW
 * - Index 1: tf64_to_s8_scalar_() — Portable fallback
 */
extern const castFn lookup_tf64_to_s8_[];

/**
 * @brief Dispatch table for f64 to s32 conversions.
 *
 * Variants:
 * - Index 0: tf64_to_s32_avx512_() — Requires: AVX512F
 * - Index 1: tf64_to_s32_avx_avx2_() — Requires: AVX/AVX2
 * - Index 2: tf64_to_s32_sse4_2_() — Requires: SSE4.2
 * - Index 3: tf64_to_s32_scalar_() — Portable fallback
 */
extern const castFn lookup_tf64_to_s32_[];

/**
 * @brief Dispatch table for f64 to s64 conversions.
 *
 * Variants:
 * - Index 0: tf64_to_s64_avx512_() — Requires: AVX512F
 * - Index 1: tf64_to_s64_scalar_() — Portable fallback
 */
extern const castFn lookup_tf64_to_s64_[];

/**
 * @brief Dispatch table for f64 to u8 conversions.
 *
 * Variants:
 * - Index 0: tf64_to_u8_avx512_() — Requires: AVX512F, AVX512BW
 * - Index 1: tf64_to_u8_scalar_() — Portable fallback
 */
extern const castFn lookup_tf64_to_u8_[];

/**
 * @brief Dispatch table for f64 to u32 conversions.
 *
 * Variants:
 * - Index 0: tf64_to_u32_avx512_() — Requires: AVX512F
 * - Index 1: tf64_to_u32_scalar_() — Portable fallback
 */
extern const castFn lookup_tf64_to_u32_[];

/**
 * @brief Dispatch table for f64 to u64 conversions.
 *
 * Variants:
 * - Index 0: tf64_to_u64_avx512_() — Requires: AVX512F, AVX512DQ
 * - Index 1: tf64_to_u64_scalar_() — Portable fallback
 */
extern const castFn lookup_tf64_to_u64_[];

/* =========================================================================
 * Integer to floating-point conversions (24 tables)
 * ========================================================================= */

/**
 * @brief Dispatch table for s8 to fp16 conversions.
 *
 * Variants:
 * - Index 0: ts8_to_fp16_avx512_() — Requires: AVX512F
 * - Index 1: ts8_to_fp16_scalar_() — Portable fallback
 */
extern const castFn lookup_ts8_to_fp16_[];

/**
 * @brief Dispatch table for s32 to fp16 conversions.
 *
 * Variants:
 * - Index 0: ts32_to_fp16_avx512fp16_() — Requires: AVX512F, AVX512FP16
 * - Index 1: ts32_to_fp16_scalar_() — Portable fallback
 */
extern const castFn lookup_ts32_to_fp16_[];

/**
 * @brief Dispatch table for s64 to fp16 conversions.
 *
 * Variants:
 * - Index 0: ts64_to_fp16_avx512fp16_() — Requires: AVX512F, AVX512FP16
 * - Index 1: ts64_to_fp16_scalar_() — Portable fallback
 */
extern const castFn lookup_ts64_to_fp16_[];

/**
 * @brief Dispatch table for u8 to fp16 conversions.
 *
 * Variants:
 * - Index 0: tu8_to_fp16_avx512fp16_() — Requires: AVX512F, AVX512FP16
 * - Index 1: tu8_to_fp16_scalar_() — Portable fallback
 */
extern const castFn lookup_tu8_to_fp16_[];

/**
 * @brief Dispatch table for u32 to fp16 conversions.
 *
 * Variants:
 * - Index 0: tu32_to_fp16_avx512fp16_() — Requires: AVX512F, AVX512FP16
 * - Index 1: tu32_to_fp16_scalar_() — Portable fallback
 */
extern const castFn lookup_tu32_to_fp16_[];

/**
 * @brief Dispatch table for u64 to fp16 conversions.
 *
 * Variants:
 * - Index 0: tu64_to_fp16_avx512fp16_() — Requires: AVX512F, AVX512FP16
 * - Index 1: tu64_to_fp16_scalar_() — Portable fallback
 */
extern const castFn lookup_tu64_to_fp16_[];

/**
 * @brief Dispatch table for s8 to bf16 conversions.
 *
 * Variants:
 * - Index 0: ts8_to_bf16_avx512bf16_() — Requires: AVX512F, AVX512BF16
 * - Index 1: ts8_to_bf16_scalar_() — Portable fallback
 */
extern const castFn lookup_ts8_to_bf16_[];

/**
 * @brief Dispatch table for s32 to bf16 conversions.
 *
 * Variants:
 * - Index 0: ts32_to_bf16_avx512bf16_() — Requires: AVX512F, AVX512BF16
 * - Index 1: ts32_to_bf16_scalar_() — Portable fallback
 */
extern const castFn lookup_ts32_to_bf16_[];

/**
 * @brief Dispatch table for s64 to bf16 conversions.
 *
 * Variants:
 * - Index 0: ts64_to_bf16_avx512bf16_() — Requires: AVX512F, AVX512BF16, AVX512DQ, AVX512VL
 * - Index 1: ts64_to_bf16_scalar_() — Portable fallback
 */
extern const castFn lookup_ts64_to_bf16_[];

/**
 * @brief Dispatch table for u8 to bf16 conversions.
 *
 * Variants:
 * - Index 0: tu8_to_bf16_avx512bf16_() — Requires: AVX512F, AVX512BF16
 * - Index 1: tu8_to_bf16_scalar_() — Portable fallback
 */
extern const castFn lookup_tu8_to_bf16_[];

/**
 * @brief Dispatch table for u32 to bf16 conversions.
 *
 * Variants:
 * - Index 0: tu32_to_bf16_avx512bf16_() — Requires: AVX512F, AVX512BF16
 * - Index 1: tu32_to_bf16_scalar_() — Portable fallback
 */
extern const castFn lookup_tu32_to_bf16_[];

/**
 * @brief Dispatch table for u64 to bf16 conversions.
 *
 * Variants:
 * - Index 0: tu64_to_bf16_avx512bf16_() — Requires: AVX512F, AVX512BF16, AVX512DQ, AVX512VL
 * - Index 1: tu64_to_bf16_scalar_() — Portable fallback
 */
extern const castFn lookup_tu64_to_bf16_[];

/**
 * @brief Dispatch table for s8 to f32 conversions.
 *
 * Variants:
 * - Index 0: ts8_to_f32_avx512_() — Requires: AVX512F
 * - Index 1: ts8_to_f32_avx2_() — Requires: AVX2
 * - Index 2: ts8_to_f32_scalar_() — Portable fallback
 */
extern const castFn lookup_ts8_to_f32_[];

/**
 * @brief Dispatch table for s32 to f32 conversions.
 *
 * Variants:
 * - Index 0: ts32_to_f32_avx512_() — Requires: AVX512F
 * - Index 1: ts32_to_f32_avx_avx2_() — Requires: AVX/AVX2
 * - Index 2: ts32_to_f32_sse4_2_() — Requires: SSE2, SSE4.2
 * - Index 3: ts32_to_f32_scalar_() — Portable fallback
 */
extern const castFn lookup_ts32_to_f32_[];

/**
 * @brief Dispatch table for s64 to f32 conversions.
 *
 * Variants:
 * - Index 0: ts64_to_f32_avx512_() — Requires: AVX512F, AVX512DQ
 * - Index 1: ts64_to_f32_scalar_() — Portable fallback
 */
extern const castFn lookup_ts64_to_f32_[];

/**
 * @brief Dispatch table for u8 to f32 conversions.
 *
 * Variants:
 * - Index 0: tu8_to_f32_avx512_() — Requires: AVX512F
 * - Index 1: tu8_to_f32_avx2_() — Requires: AVX2
 * - Index 2: tu8_to_f32_scalar_() — Portable fallback
 */
extern const castFn lookup_tu8_to_f32_[];

/**
 * @brief Dispatch table for u32 to f32 conversions.
 *
 * Variants:
 * - Index 0: tu32_to_f32_avx512_() — Requires: AVX512F
 * - Index 1: tu32_to_f32_scalar_() — Portable fallback
 */
extern const castFn lookup_tu32_to_f32_[];

/**
 * @brief Dispatch table for u64 to f32 conversions.
 *
 * Variants:
 * - Index 0: tu64_to_f32_avx512_() — Requires: AVX512F, AVX512DQ
 * - Index 1: tu64_to_f32_scalar_() — Portable fallback
 */
extern const castFn lookup_tu64_to_f32_[];

/**
 * @brief Dispatch table for s8 to f64 conversions.
 *
 * Variants:
 * - Index 0: ts8_to_f64_avx512_() — Requires: AVX512F
 * - Index 1: ts8_to_f64_scalar_() — Portable fallback
 */
extern const castFn lookup_ts8_to_f64_[];

/**
 * @brief Dispatch table for s32 to f64 conversions.
 *
 * Variants:
 * - Index 0: ts32_to_f64_avx512_() — Requires: AVX512F
 * - Index 1: ts32_to_f64_avx_avx2_() — Requires: AVX/AVX2
 * - Index 2: ts32_to_f64_sse4_2_() — Requires: SSE4.2
 * - Index 3: ts32_to_f64_scalar_() — Portable fallback
 */
extern const castFn lookup_ts32_to_f64_[];

/**
 * @brief Dispatch table for s64 to f64 conversions.
 *
 * Variants:
 * - Index 0: ts64_to_f64_avx512_() — Requires: AVX512F
 * - Index 1: ts64_to_f64_scalar_() — Portable fallback
 */
extern const castFn lookup_ts64_to_f64_[];

/**
 * @brief Dispatch table for u8 to f64 conversions.
 *
 * Variants:
 * - Index 0: tu8_to_f64_avx512_() — Requires: AVX512F
 * - Index 1: tu8_to_f64_scalar_() — Portable fallback
 */
extern const castFn lookup_tu8_to_f64_[];

/**
 * @brief Dispatch table for u32 to f64 conversions.
 *
 * Variants:
 * - Index 0: tu32_to_f64_avx512_() — Requires: AVX512F
 * - Index 1: tu32_to_f64_scalar_() — Portable fallback
 */
extern const castFn lookup_tu32_to_f64_[];

/**
 * @brief Dispatch table for u64 to f64 conversions.
 *
 * Variants:
 * - Index 0: tu64_to_f64_avx512_() — Requires: AVX512F
 * - Index 1: tu64_to_f64_scalar_() — Portable fallback
 */
extern const castFn lookup_tu64_to_f64_[];

/* =========================================================================
 * Integer to integer conversions (37 tables)
 * ========================================================================= */

/**
 * @brief Dispatch table for s8 to s32 conversions.
 *
 * Variants:
 * - Index 0: ts8_to_s32_avx512_() — Requires: AVX512F
 * - Index 1: ts8_to_s32_avx2_() — Requires: AVX2
 * - Index 2: ts8_to_s32_sse4_2_() — Requires: SSE2, SSE4.2
 * - Index 3: ts8_to_s32_scalar_() — Portable fallback
 */
extern const castFn lookup_ts8_to_s32_[];

/**
 * @brief Dispatch table for s8 to s64 conversions.
 *
 * Variants:
 * - Index 0: ts8_to_s64_avx512_() — Requires: AVX512F
 * - Index 1: ts8_to_s64_avx2_() — Requires: AVX2
 * - Index 2: ts8_to_s64_sse4_2_() — Requires: SSE2, SSE4.2
 * - Index 3: ts8_to_s64_scalar_() — Portable fallback
 */
extern const castFn lookup_ts8_to_s64_[];

/**
 * @brief Dispatch table for s32 to s8 conversions.
 *
 * Variants:
 * - Index 0: ts32_to_s8_avx512_() — Requires: AVX512F
 * - Index 1: ts32_to_s8_scalar_() — Portable fallback
 */
extern const castFn lookup_ts32_to_s8_[];

/**
 * @brief Dispatch table for s32 to s64 conversions.
 *
 * Variants:
 * - Index 0: ts32_to_s64_avx512_() — Requires: AVX512F
 * - Index 1: ts32_to_s64_avx2_() — Requires: AVX2
 * - Index 2: ts32_to_s64_sse4_2_() — Requires: SSE2, SSE4.2
 * - Index 3: ts32_to_s64_scalar_() — Portable fallback
 */
extern const castFn lookup_ts32_to_s64_[];

/**
 * @brief Dispatch table for s64 to s8 conversions.
 *
 * Variants:
 * - Index 0: ts64_to_s8_avx512_() — Requires: AVX512F, AVX512BW
 * - Index 1: ts64_to_s8_scalar_() — Portable fallback
 */
extern const castFn lookup_ts64_to_s8_[];

/**
 * @brief Dispatch table for s64 to s32 conversions.
 *
 * Variants:
 * - Index 0: ts64_to_s32_avx512_() — Requires: AVX512F
 * - Index 1: ts64_to_s32_scalar_() — Portable fallback
 */
extern const castFn lookup_ts64_to_s32_[];

/**
 * @brief Dispatch table for u8 to u32 conversions.
 *
 * Variants:
 * - Index 0: tu8_to_u32_avx512_() — Requires: AVX512F
 * - Index 1: tu8_to_u32_avx2_() — Requires: AVX2
 * - Index 2: tu8_to_u32_scalar_() — Portable fallback
 */
extern const castFn lookup_tu8_to_u32_[];

/**
 * @brief Dispatch table for u8 to u64 conversions.
 *
 * Variants:
 * - Index 0: tu8_to_u64_avx512_() — Requires: AVX512F
 * - Index 1: tu8_to_u64_avx2_() — Requires: AVX2
 * - Index 2: tu8_to_u64_sse4_2_() — Requires: SSE4.2
 * - Index 3: tu8_to_u64_scalar_() — Portable fallback
 */
extern const castFn lookup_tu8_to_u64_[];

/**
 * @brief Dispatch table for u32 to u8 conversions.
 *
 * Variants:
 * - Index 0: tu32_to_u8_avx512_() — Requires: AVX512F
 * - Index 1: tu32_to_u8_scalar_() — Portable fallback
 */
extern const castFn lookup_tu32_to_u8_[];

/**
 * @brief Dispatch table for u32 to u64 conversions.
 *
 * Variants:
 * - Index 0: tu32_to_u64_avx512_() — Requires: AVX512F
 * - Index 1: tu32_to_u64_avx2_() — Requires: AVX2
 * - Index 2: tu32_to_u64_sse4_2_() — Requires: SSE4.2
 * - Index 3: tu32_to_u64_scalar_() — Portable fallback
 */
extern const castFn lookup_tu32_to_u64_[];

/**
 * @brief Dispatch table for u64 to u8 conversions.
 *
 * Variants:
 * - Index 0: tu64_to_u8_avx512_() — Requires: AVX512F, AVX512BW
 * - Index 1: tu64_to_u8_scalar_() — Portable fallback
 */
extern const castFn lookup_tu64_to_u8_[];

/**
 * @brief Dispatch table for u64 to u32 conversions.
 *
 * Variants:
 * - Index 0: tu64_to_u32_avx512_() — Requires: AVX512F
 * - Index 1: tu64_to_u32_scalar_() — Portable fallback
 */
extern const castFn lookup_tu64_to_u32_[];

/**
 * @brief Dispatch table for s8 to u8 conversions.
 *
 * Variants:
 * - Index 0: ts8_to_u8_avx512_() — Requires: AVX512F
 * - Index 1: ts8_to_u8_avx_avx2_() — Requires: AVX/AVX2
 * - Index 2: ts8_to_u8_sse4_2_() — Requires: SSE4.2
 * - Index 3: ts8_to_u8_scalar_() — Portable fallback
 */
extern const castFn lookup_ts8_to_u8_[];

/**
 * @brief Dispatch table for s8 to u32 conversions.
 *
 * Variants:
 * - Index 0: ts8_to_u32_avx512_() — Requires: AVX512F
 * - Index 1: ts8_to_u32_avx2_() — Requires: AVX2
 * - Index 2: ts8_to_u32_sse4_2_() — Requires: SSE4.2
 * - Index 3: ts8_to_u32_scalar_() — Portable fallback
 */
extern const castFn lookup_ts8_to_u32_[];

/**
 * @brief Dispatch table for s8 to u64 conversions.
 *
 * Variants:
 * - Index 0: ts8_to_u64_avx512_() — Requires: AVX512F
 * - Index 1: ts8_to_u64_scalar_() — Portable fallback
 */
extern const castFn lookup_ts8_to_u64_[];

/**
 * @brief Dispatch table for s32 to u8 conversions.
 *
 * Variants:
 * - Index 0: ts32_to_u8_avx512_() — Requires: AVX512F, AVX512BW
 * - Index 1: ts32_to_u8_avx2_() — Requires: AVX2
 * - Index 2: ts32_to_u8_sse4_2_() — Requires: SSE4.2
 * - Index 3: ts32_to_u8_scalar_() — Portable fallback
 */
extern const castFn lookup_ts32_to_u8_[];

/**
 * @brief Dispatch table for s32 to u32 conversions.
 *
 * Variants:
 * - Index 0: ts32_to_u32_avx512_() — Requires: AVX512F
 * - Index 1: ts32_to_u32_avx_avx2_() — Requires: AVX/AVX2
 * - Index 2: ts32_to_u32_sse4_2_() — Requires: SSE4.2
 * - Index 3: ts32_to_u32_scalar_() — Portable fallback
 */
extern const castFn lookup_ts32_to_u32_[];

/**
 * @brief Dispatch table for s32 to u64 conversions.
 *
 * Variants:
 * - Index 0: ts32_to_u64_avx512_() — Requires: AVX512F
 * - Index 1: ts32_to_u64_avx2_() — Requires: AVX2
 * - Index 2: ts32_to_u64_sse4_2_() — Requires: SSE4.2
 * - Index 3: ts32_to_u64_scalar_() — Portable fallback
 */
extern const castFn lookup_ts32_to_u64_[];

/**
 * @brief Dispatch table for s64 to u8 conversions.
 *
 * Variants:
 * - Index 0: ts64_to_u8_avx512_() — Requires: AVX512F, AVX512BW
 * - Index 1: ts64_to_u8_scalar_() — Portable fallback
 */
extern const castFn lookup_ts64_to_u8_[];

/**
 * @brief Dispatch table for s64 to u32 conversions.
 *
 * Variants:
 * - Index 0: ts64_to_u32_avx512_() — Requires: AVX512F
 * - Index 1: ts64_to_u32_scalar_() — Portable fallback
 */
extern const castFn lookup_ts64_to_u32_[];

/**
 * @brief Dispatch table for s64 to u64 conversions.
 *
 * Variants:
 * - Index 0: ts64_to_u64_avx512_() — Requires: AVX512F
 * - Index 1: ts64_to_u64_avx_avx2_() — Requires: AVX/AVX2
 * - Index 2: ts64_to_u64_sse4_2_() — Requires: SSE
 * - Index 3: ts64_to_u64_scalar_() — Portable fallback
 */
extern const castFn lookup_ts64_to_u64_[];

/**
 * @brief Dispatch table for u8 to s8 conversions.
 *
 * Variants:
 * - Index 0: tu8_to_s8_avx512_() — Requires: AVX512F
 * - Index 1: tu8_to_s8_avx_avx2_() — Requires: AVX/AVX2
 * - Index 2: tu8_to_s8_sse4_2_() — Requires: SSE4.2
 * - Index 3: tu8_to_s8_scalar_() — Portable fallback
 */
extern const castFn lookup_tu8_to_s8_[];

/**
 * @brief Dispatch table for u8 to s32 conversions.
 *
 * Variants:
 * - Index 0: tu8_to_s32_avx512_() — Requires: AVX512F
 * - Index 1: tu8_to_s32_avx2_() — Requires: AVX2
 * - Index 2: tu8_to_s32_sse4_2_() — Requires: SSE4.2
 * - Index 3: tu8_to_s32_scalar_() — Portable fallback
 */
extern const castFn lookup_tu8_to_s32_[];

/**
 * @brief Dispatch table for u8 to s64 conversions.
 *
 * Variants:
 * - Index 0: tu8_to_s64_avx512_() — Requires: AVX512F
 * - Index 1: tu8_to_s64_avx2_() — Requires: AVX2
 * - Index 2: tu8_to_s64_sse4_2_() — Requires: SSE4.2
 * - Index 3: tu8_to_s64_scalar_() — Portable fallback
 */
extern const castFn lookup_tu8_to_s64_[];

/**
 * @brief Dispatch table for u32 to s8 conversions.
 *
 * Variants:
 * - Index 0: tu32_to_s8_avx512_() — Requires: AVX512F, AVX512BW
 * - Index 1: tu32_to_s8_scalar_() — Portable fallback
 */
extern const castFn lookup_tu32_to_s8_[];

/**
 * @brief Dispatch table for u32 to s32 conversions.
 *
 * Variants:
 * - Index 0: tu32_to_s32_avx512_() — Requires: AVX512F
 * - Index 1: tu32_to_s32_avx_avx2_() — Requires: AVX/AVX2
 * - Index 2: tu32_to_s32_sse4_2_() — Requires: SSE
 * - Index 3: tu32_to_s32_scalar_() — Portable fallback
 */
extern const castFn lookup_tu32_to_s32_[];

/**
 * @brief Dispatch table for u32 to s64 conversions.
 *
 * Variants:
 * - Index 0: tu32_to_s64_avx512_() — Requires: AVX512F
 * - Index 1: tu32_to_s64_avx2_() — Requires: AVX2
 * - Index 2: tu32_to_s64_sse4_2_() — Requires: SSE4.2
 * - Index 3: tu32_to_s64_scalar_() — Portable fallback
 */
extern const castFn lookup_tu32_to_s64_[];

/**
 * @brief Dispatch table for u64 to s8 conversions.
 *
 * Variants:
 * - Index 0: tu64_to_s8_avx512_() — Requires: AVX512F, AVX512BW
 * - Index 1: tu64_to_s8_scalar_() — Portable fallback
 */
extern const castFn lookup_tu64_to_s8_[];

/**
 * @brief Dispatch table for u64 to s32 conversions.
 *
 * Variants:
 * - Index 0: tu64_to_s32_avx512_() — Requires: AVX512F
 * - Index 1: tu64_to_s32_scalar_() — Portable fallback
 */
extern const castFn lookup_tu64_to_s32_[];

/**
 * @brief Dispatch table for u64 to s64 conversions.
 *
 * Variants:
 * - Index 0: tu64_to_s64_avx512_() — Requires: AVX512F
 * - Index 1: tu64_to_s64_avx_avx2_() — Requires: AVX/AVX2
 * - Index 2: tu64_to_s64_sse4_2_() — Requires: SSE
 * - Index 3: tu64_to_s64_scalar_() — Portable fallback
 */
extern const castFn lookup_tu64_to_s64_[];
