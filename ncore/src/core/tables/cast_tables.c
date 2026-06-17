/**
 * @file cast_tables.c
 * @brief Tensor element-wise type cast kernels and dispatch tables.
 *
 * Provides scalar fallback implementations and SIMD-accelerated variants
 * (SSE4.2, AVX/AVX2, AVX-512F, AVX-512BF16, AVX-512FP16) for every
 * supported source/destination type combination.  Each conversion group
 * exposes a @c const @c CastFn[] lookup table that lists available
 * implementations in descending capability order; the caller selects the
 * first entry whose required ISA is present at runtime.
 *
 * Naming convention:
 *   @c t<src>_to_<dst>_<isa>()  – SIMD variant
 *   @c t<src>_to_<dst>_scalar() – portable fallback
 *   @c lookup_t<src>_to_<dst>[] – dispatch table
 *
 * Type abbreviations used in identifiers:
 *   @c fp16  = _Float16 (IEEE 754 half-precision)
 *   @c bf16  = __bf16   (Brain Float 16)
 *   @c f32   = float    (single-precision)
 *   @c f64   = double   (double-precision)
 *   @c s8/s32/s64 = int8 / int32 / int64
 *   @c u8/u32/u64 = uint8 / uint32 / uint64
 */

#include <ncore/headeronly/cast.h>
#include <ncore/tables/cast_tables.h>
#include <string.h>

/**
 * @def REMAINING(i, size, d, s, cvtype)
 * @brief Scalar tail-loop for SIMD functions.
 *
 * Converts remaining elements that do not fill a full SIMD vector after the
 * main vectorised loop.  Advances index @p i through @p size, writing
 * @c (cvtype)s[i] into @c d[i].
 *
 * @param i      Loop counter, must already point past the last complete vector.
 * @param size   Total number of elements in the tensor.
 * @param d      Pointer to the destination element array.
 * @param s      Pointer to the source element array.
 * @param cvtype C type to use for the element-wise cast.
 */
#define REMAINING(i, size, d, s, cvtype)                                       \
  {                                                                            \
    for (; i < size; i++)                                                      \
      d[i] = (cvtype)s[i];                                                     \
  }

/* =========================================================================
 * Scalar kernels
 * ========================================================================= */

/**
 * @brief Cast every element from _Float16 to float (scalar fallback).
 *
 * SIMD variants: tfp16_to_f32_avx512(), tfp16_to_f32_avx_avx2_fp16c()
 *
 * @param[in]  src  Source tensor with element type _Float16.
 * @param[out] dst  Destination tensor with element type float.
 */
static void tfp16_to_f32_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {
  const _Float16 *s = src->data.half;
  float *d = dst->data.f32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (float)s[i];
  }
}

/**
 * @brief Cast every element from _Float16 to double (scalar fallback).
 *
 * SIMD variants: tfp16_to_f64_avx512(), tfp16_to_f64_avx_avx2_fp16c()
 *
 * @param[in]  src  Source tensor with element type _Float16.
 * @param[out] dst  Destination tensor with element type double.
 */
static void tfp16_to_f64_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {
  const _Float16 *s = src->data.half;
  double *d = dst->data.f64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (double)s[i];
  }
}

/**
 * @brief Cast every element from _Float16 to __bf16 (scalar fallback).
 *
 * SIMD variants: tfp16_to_bf16_avx512bf16()
 *
 * @param[in]  src  Source tensor with element type _Float16.
 * @param[out] dst  Destination tensor with element type __bf16.
 */
static void tfp16_to_bf16_scalar(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  const _Float16 *s = src->data.half;
  __bf16 *d = dst->data.bf16;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (__bf16)s[i];
  }
}

/**
 * @brief Cast every element from float to _Float16 (scalar fallback).
 *
 * SIMD variants: tf32_to_fp16_avx512fp16(), tf32_to_fp16_avx_avx2_f16c()
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type _Float16.
 */
static void tf32_to_fp16_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {
  const float *s = src->data.f32;
  _Float16 *d = dst->data.half;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (_Float16)s[i];
  }
}

/**
 * @brief Cast every element from float to double (scalar fallback).
 *
 * SIMD variants: tf32_to_f64_avx512(), tf32_to_f64_avx_avx2(),
 * tf32_to_f64_sse4_2()
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type double.
 */
static void tf32_to_f64_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const float *s = src->data.f32;
  double *d = dst->data.f64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (double)s[i];
  }
}

/**
 * @brief Cast every element from float to __bf16 (scalar fallback).
 *
 * SIMD variants: tf32_to_bf16_avx512bf16()
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type __bf16.
 */
static void tf32_to_bf16_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {
  const float *s = src->data.f32;
  __bf16 *d = dst->data.bf16;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (__bf16)s[i];
  }
}

/**
 * @brief Cast every element from __bf16 to _Float16 (scalar fallback).
 *
 * SIMD variants: tbf16_to_fp16_avx512bf16_fp16()
 *
 * @param[in]  src  Source tensor with element type __bf16.
 * @param[out] dst  Destination tensor with element type _Float16.
 */
static void tbf16_to_fp16_scalar(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  const __bf16 *s = src->data.bf16;
  _Float16 *d = dst->data.half;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (_Float16)s[i];
  }
}

/**
 * @brief Cast every element from __bf16 to float (scalar fallback).
 *
 * SIMD variants: tbf16_to_f32_avx512bf16()
 *
 * @param[in]  src  Source tensor with element type __bf16.
 * @param[out] dst  Destination tensor with element type float.
 */
static void tbf16_to_f32_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {
  const __bf16 *s = src->data.bf16;
  float *d = dst->data.f32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (float)s[i];
  }
}

/**
 * @brief Cast every element from __bf16 to double (scalar fallback).
 *
 * SIMD variants: tbf16_to_f64_avx512bf16()
 *
 * @param[in]  src  Source tensor with element type __bf16.
 * @param[out] dst  Destination tensor with element type double.
 */
static void tbf16_to_f64_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {
  const __bf16 *s = src->data.bf16;
  double *d = dst->data.f64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (double)s[i];
  }
}

/**
 * @brief Cast every element from double to _Float16 (scalar fallback).
 *
 * SIMD variants: tf64_to_fp16_avx512fp16()
 *
 * @param[in]  src  Source tensor with element type double.
 * @param[out] dst  Destination tensor with element type _Float16.
 */
static void tf64_to_fp16_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {
  const double *s = src->data.f64;
  _Float16 *d = dst->data.half;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (_Float16)s[i];
  }
}

/**
 * @brief Cast every element from double to float (scalar fallback).
 *
 * SIMD variants: tf64_to_f32_avx512(), tf64_to_f32_avx_avx2(),
 * tf64_to_f32_sse4_2()
 *
 * @param[in]  src  Source tensor with element type double.
 * @param[out] dst  Destination tensor with element type float.
 */
static void tf64_to_f32_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const double *s = src->data.f64;
  float *d = dst->data.f32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (float)s[i];
  }
}

/**
 * @brief Cast every element from double to __bf16 (scalar fallback).
 *
 * SIMD variants: tf64_to_bf16_avx512bf16()
 *
 * @param[in]  src  Source tensor with element type double.
 * @param[out] dst  Destination tensor with element type __bf16.
 */
static void tf64_to_bf16_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {
  const double *s = src->data.f64;
  __bf16 *d = dst->data.bf16;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (__bf16)s[i];
  }
}
/* =========================================================================
 * Scalar kernels — floating-point to integer
 * ========================================================================= */

/**
 * @brief Cast every element from _Float16 to int8_t (scalar fallback).
 *
 * SIMD variants: tfp16_to_s8_avx512fp16()
 *
 * @param[in]  src  Source tensor with element type _Float16.
 * @param[out] dst  Destination tensor with element type int8_t.
 */
static void tfp16_to_s8_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const _Float16 *s = src->data.half;
  int8_t *d = dst->data.s8;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int8_t)s[i];
  }
}

/**
 * @brief Cast every element from _Float16 to int32_t (scalar fallback).
 *
 * SIMD variants: tfp16_to_s32_avx512fp16()
 *
 * @param[in]  src  Source tensor with element type _Float16.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
static void tfp16_to_s32_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {
  const _Float16 *s = src->data.half;
  int32_t *d = dst->data.s32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int32_t)s[i];
  }
}

/**
 * @brief Cast every element from _Float16 to int64_t (scalar fallback).
 *
 * SIMD variants: tfp16_to_s64_avx512fp16()
 *
 * @param[in]  src  Source tensor with element type _Float16.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
static void tfp16_to_s64_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {
  const _Float16 *s = src->data.half;
  int64_t *d = dst->data.s64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int64_t)s[i];
  }
}

/**
 * @brief Cast every element from _Float16 to uint8_t (scalar fallback).
 *
 * SIMD variants: tfp16_to_u8_avx512fp16()
 *
 * @param[in]  src  Source tensor with element type _Float16.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
static void tfp16_to_u8_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const _Float16 *s = src->data.half;
  uint8_t *d = dst->data.u8;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint8_t)s[i];
  }
}

/**
 * @brief Cast every element from _Float16 to uint32_t (scalar fallback).
 *
 * SIMD variants: tfp16_to_u32_avx512fp16()
 *
 * @param[in]  src  Source tensor with element type _Float16.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
static void tfp16_to_u32_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {
  const _Float16 *s = src->data.half;
  uint32_t *d = dst->data.u32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint32_t)s[i];
  }
}

/**
 * @brief Cast every element from _Float16 to uint64_t (scalar fallback).
 *
 * SIMD variants: tfp16_to_u64_avx512fp16()
 *
 * @param[in]  src  Source tensor with element type _Float16.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
static void tfp16_to_u64_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {
  const _Float16 *s = src->data.half;
  uint64_t *d = dst->data.u64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint64_t)s[i];
  }
}
/**
 * @brief Cast every element from __bf16 to int8_t (scalar fallback).
 *
 * SIMD variants: tbf16_to_s8_avx512bf16()
 *
 * @param[in]  src  Source tensor with element type __bf16.
 * @param[out] dst  Destination tensor with element type int8_t.
 */
static void tbf16_to_s8_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const __bf16 *s = src->data.bf16;
  int8_t *d = dst->data.s8;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int8_t)s[i];
  }
}

/**
 * @brief Cast every element from __bf16 to int32_t (scalar fallback).
 *
 * SIMD variants: tbf16_to_s32_avx512bf16()
 *
 * @param[in]  src  Source tensor with element type __bf16.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
static void tbf16_to_s32_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {
  const __bf16 *s = src->data.bf16;
  int32_t *d = dst->data.s32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int32_t)s[i];
  }
}

/**
 * @brief Cast every element from __bf16 to int64_t (scalar fallback).
 *
 * SIMD variants: tbf16_to_s64_avx512bf16()
 *
 * @param[in]  src  Source tensor with element type __bf16.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
static void tbf16_to_s64_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {
  const __bf16 *s = src->data.bf16;
  int64_t *d = dst->data.s64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int64_t)s[i];
  }
}

/**
 * @brief Cast every element from __bf16 to uint8_t (scalar fallback).
 *
 * SIMD variants: tbf16_to_u8_avx512bf16()
 *
 * @param[in]  src  Source tensor with element type __bf16.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
static void tbf16_to_u8_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const __bf16 *s = src->data.bf16;
  uint8_t *d = dst->data.u8;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint8_t)s[i];
  }
}

/**
 * @brief Cast every element from __bf16 to uint32_t (scalar fallback).
 *
 * SIMD variants: tbf16_to_u32_avx512bf16()
 *
 * @param[in]  src  Source tensor with element type __bf16.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
static void tbf16_to_u32_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {
  const __bf16 *s = src->data.bf16;
  uint32_t *d = dst->data.u32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint32_t)s[i];
  }
}

/**
 * @brief Cast every element from __bf16 to uint64_t (scalar fallback).
 *
 * SIMD variants: tbf16_to_u64_avx512bf16()
 *
 * @param[in]  src  Source tensor with element type __bf16.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
static void tbf16_to_u64_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {
  const __bf16 *s = src->data.bf16;
  uint64_t *d = dst->data.u64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint64_t)s[i];
  }
}
/**
 * @brief Cast every element from float to int8_t (scalar fallback).
 *
 * SIMD variants: tf32_to_s8_avx512(), tf32_to_s8_avx2()
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type int8_t.
 */
static void tf32_to_s8_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {

  const float *s = src->data.f32;
  int8_t *d = dst->data.s8;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int8_t)s[i];
  }
}
/**
 * @brief Cast every element from float to int32_t (scalar fallback).
 *
 * SIMD variants: tf32_to_s32_avx512(), tf32_to_s32_avx_avx2(),
 * tf32_to_s32_sse4_2()
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
static void tf32_to_s32_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {

  const float *s = src->data.f32;
  int32_t *d = dst->data.s32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int32_t)s[i];
  }
}
/**
 * @brief Cast every element from float to int64_t (scalar fallback).
 *
 * SIMD variants: tf32_to_s64_avx512()
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
static void tf32_to_s64_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {

  const float *s = src->data.f32;
  int64_t *d = dst->data.s64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int64_t)s[i];
  }
}
/**
 * @brief Cast every element from float to uint8_t (scalar fallback).
 *
 * SIMD variants: tf32_to_u8_avx512(), tf32_to_u8_avx2()
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
static void tf32_to_u8_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {

  const float *s = src->data.f32;
  uint8_t *d = dst->data.u8;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint8_t)s[i];
  }
}
/**
 * @brief Cast every element from float to uint32_t (scalar fallback).
 *
 * SIMD variants: tf32_to_u32_avx512()
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
static void tf32_to_u32_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {

  const float *s = src->data.f32;
  uint32_t *d = dst->data.u32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint32_t)s[i];
  }
}
/**
 * @brief Cast every element from float to uint64_t (scalar fallback).
 *
 * SIMD variants: tf32_to_u64_avx512()
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
static void tf32_to_u64_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {

  const float *s = src->data.f32;
  uint64_t *d = dst->data.u64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint64_t)s[i];
  }
}
/**
 * @brief Cast every element from double to int8_t (scalar fallback).
 *
 * SIMD variants: tf64_to_s8_avx512()
 *
 * @param[in]  src  Source tensor with element type double.
 * @param[out] dst  Destination tensor with element type int8_t.
 */
static void tf64_to_s8_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {

  const double *s = src->data.f64;
  int8_t *d = dst->data.s8;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int8_t)s[i];
  }
}
/**
 * @brief Cast every element from double to int32_t (scalar fallback).
 *
 * SIMD variants: tf64_to_s32_avx512(), tf64_to_s32_avx_avx2(),
 * tf64_to_s32_sse4_2()
 *
 * @param[in]  src  Source tensor with element type double.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
static void tf64_to_s32_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {

  const double *s = src->data.f64;
  int32_t *d = dst->data.s32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int32_t)s[i];
  }
}
/**
 * @brief Cast every element from double to int64_t (scalar fallback).
 *
 * SIMD variants: tf64_to_s64_avx512()
 *
 * @param[in]  src  Source tensor with element type double.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
static void tf64_to_s64_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {

  const double *s = src->data.f64;
  int64_t *d = dst->data.s64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int64_t)s[i];
  }
}
/**
 * @brief Cast every element from double to uint8_t (scalar fallback).
 *
 * SIMD variants: tf64_to_u8_avx512()
 *
 * @param[in]  src  Source tensor with element type double.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
static void tf64_to_u8_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {

  const double *s = src->data.f64;
  uint8_t *d = dst->data.u8;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint8_t)s[i];
  }
}
/**
 * @brief Cast every element from double to uint32_t (scalar fallback).
 *
 * SIMD variants: tf64_to_u32_avx512()
 *
 * @param[in]  src  Source tensor with element type double.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
static void tf64_to_u32_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {

  const double *s = src->data.f64;
  uint32_t *d = dst->data.u32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint32_t)s[i];
  }
}
/**
 * @brief Cast every element from double to uint64_t (scalar fallback).
 *
 * SIMD variants: tf64_to_u64_avx512()
 *
 * @param[in]  src  Source tensor with element type double.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
static void tf64_to_u64_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {

  const double *s = src->data.f64;
  uint64_t *d = dst->data.u64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint64_t)s[i];
  }
}
/**
 * @brief Cast every element from int8_t to _Float16 (scalar fallback).
 *
 * SIMD variants: ts8_to_fp16_avx512()
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type _Float16.
 */
static void ts8_to_fp16_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {

  const int8_t *s = src->data.s8;
  _Float16 *d = dst->data.half;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (_Float16)s[i];
  }
}
/**
 * @brief Cast every element from int32_t to _Float16 (scalar fallback).
 *
 * SIMD variants: ts32_to_fp16_avx512fp16()
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type _Float16.
 */
static void ts32_to_fp16_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {

  const int32_t *s = src->data.s32;
  _Float16 *d = dst->data.half;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (_Float16)s[i];
  }
}
/**
 * @brief Cast every element from int64_t to _Float16 (scalar fallback).
 *
 * SIMD variants: ts64_to_fp16_avx512fp16()
 *
 * @param[in]  src  Source tensor with element type int64_t.
 * @param[out] dst  Destination tensor with element type _Float16.
 */
static void ts64_to_fp16_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {

  const int64_t *s = src->data.s64;
  _Float16 *d = dst->data.half;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (_Float16)s[i];
  }
}
/**
 * @brief Cast every element from uint8_t to _Float16 (scalar fallback).
 *
 * SIMD variants: tu8_to_fp16_avx512fp16()
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type _Float16.
 */
static void tu8_to_fp16_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {

  const uint8_t *s = src->data.u8;
  _Float16 *d = dst->data.half;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (_Float16)s[i];
  }
}
/**
 * @brief Cast every element from uint32_t to _Float16 (scalar fallback).
 *
 * SIMD variants: tu32_to_fp16_avx512fp16()
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type _Float16.
 */
static void tu32_to_fp16_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {

  const uint32_t *s = src->data.u32;
  _Float16 *d = dst->data.half;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (_Float16)s[i];
  }
}
/**
 * @brief Cast every element from uint64_t to _Float16 (scalar fallback).
 *
 * SIMD variants: tu64_to_fp16_avx512fp16()
 *
 * @param[in]  src  Source tensor with element type uint64_t.
 * @param[out] dst  Destination tensor with element type _Float16.
 */
static void tu64_to_fp16_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {

  const uint64_t *s = src->data.u64;
  _Float16 *d = dst->data.half;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (_Float16)s[i];
  }
}
/**
 * @brief Cast every element from int8_t to __bf16 (scalar fallback).
 *
 * SIMD variants: ts8_to_bf16_avx512bf16()
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type __bf16.
 */
static void ts8_to_bf16_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {

  const int8_t *s = src->data.s8;
  __bf16 *d = dst->data.bf16;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (__bf16)s[i];
  }
}
/**
 * @brief Cast every element from int32_t to __bf16 (scalar fallback).
 *
 * SIMD variants: ts32_to_bf16_avx512bf16()
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type __bf16.
 */
static void ts32_to_bf16_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {

  const int32_t *s = src->data.s32;
  __bf16 *d = dst->data.bf16;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (__bf16)s[i];
  }
}
/**
 * @brief Cast every element from int64_t to __bf16 (scalar fallback).
 *
 * SIMD variants: ts64_to_bf16_avx512bf16()
 *
 * @param[in]  src  Source tensor with element type int64_t.
 * @param[out] dst  Destination tensor with element type __bf16.
 */
static void ts64_to_bf16_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {
  const int64_t *s = src->data.s64;
  __bf16 *d = dst->data.bf16;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (__bf16)s[i];
  }
}
/**
 * @brief Cast every element from uint8_t to __bf16 (scalar fallback).
 *
 * SIMD variants: tu8_to_bf16_avx512bf16()
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type __bf16.
 */
static void tu8_to_bf16_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const uint8_t *s = src->data.u8;
  __bf16 *d = dst->data.bf16;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (__bf16)s[i];
  }
}
/**
 * @brief Cast every element from uint32_t to __bf16 (scalar fallback).
 *
 * SIMD variants: tu32_to_bf16_avx512bf16()
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type __bf16.
 */
static void tu32_to_bf16_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {
  const uint32_t *s = src->data.u32;
  __bf16 *d = dst->data.bf16;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (__bf16)s[i];
  }
}
/**
 * @brief Cast every element from uint64_t to __bf16 (scalar fallback).
 *
 * SIMD variants: tu64_to_bf16_avx512bf16()
 *
 * @param[in]  src  Source tensor with element type uint64_t.
 * @param[out] dst  Destination tensor with element type __bf16.
 */
static void tu64_to_bf16_scalar(const Tensor *restrict src,
                                Tensor *restrict dst) {
  const uint64_t *s = src->data.u64;
  __bf16 *d = dst->data.bf16;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (__bf16)s[i];
  }
}
/**
 * @brief Cast every element from int8_t to float (scalar fallback).
 *
 * SIMD variants: ts8_to_f32_avx512(), ts8_to_f32_avx2()
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type float.
 */
static void ts8_to_f32_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {
  const int8_t *s = src->data.s8;
  float *d = dst->data.f32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (float)s[i];
  }
}
/**
 * @brief Cast every element from int32_t to float (scalar fallback).
 *
 * SIMD variants: ts32_to_f32_avx512(), ts32_to_f32_avx_avx2(),
 * ts32_to_f32_sse4_2()
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type float.
 */
static void ts32_to_f32_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const int32_t *s = src->data.s32;
  float *d = dst->data.f32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (float)s[i];
  }
}
/**
 * @brief Cast every element from int64_t to float (scalar fallback).
 *
 * SIMD variants: ts64_to_f32_avx512()
 *
 * @param[in]  src  Source tensor with element type int64_t.
 * @param[out] dst  Destination tensor with element type float.
 */
static void ts64_to_f32_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const int64_t *s = src->data.s64;
  float *d = dst->data.f32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (float)s[i];
  }
}
/**
 * @brief Cast every element from uint8_t to float (scalar fallback).
 *
 * SIMD variants: tu8_to_f32_avx512(), tu8_to_f32_avx2()
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type float.
 */
static void tu8_to_f32_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {
  const uint8_t *s = src->data.u8;
  float *d = dst->data.f32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (float)s[i];
  }
}
/**
 * @brief Cast every element from uint32_t to float (scalar fallback).
 *
 * SIMD variants: tu32_to_f32_avx512()
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type float.
 */
static void tu32_to_f32_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const uint32_t *s = src->data.u32;
  float *d = dst->data.f32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (float)s[i];
  }
}
/**
 * @brief Cast every element from uint64_t to float (scalar fallback).
 *
 * SIMD variants: tu64_to_f32_avx512()
 *
 * @param[in]  src  Source tensor with element type uint64_t.
 * @param[out] dst  Destination tensor with element type float.
 */
static void tu64_to_f32_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const uint64_t *s = src->data.u64;
  float *d = dst->data.f32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (float)s[i];
  }
}
/**
 * @brief Cast every element from int8_t to double (scalar fallback).
 *
 * SIMD variants: ts8_to_f64_avx512()
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type double.
 */
static void ts8_to_f64_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {
  const int8_t *s = src->data.s8;
  double *d = dst->data.f64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (double)s[i];
  }
}
/**
 * @brief Cast every element from int32_t to double (scalar fallback).
 *
 * SIMD variants: ts32_to_f64_avx512(), ts32_to_f64_avx_avx2(),
 * ts32_to_f64_sse4_2()
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type double.
 */
static void ts32_to_f64_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const int32_t *s = src->data.s32;
  double *d = dst->data.f64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (double)s[i];
  }
}
/**
 * @brief Cast every element from int64_t to double (scalar fallback).
 *
 * SIMD variants: ts64_to_f64_avx512()
 *
 * @param[in]  src  Source tensor with element type int64_t.
 * @param[out] dst  Destination tensor with element type double.
 */
static void ts64_to_f64_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const int64_t *s = src->data.s64;
  double *d = dst->data.f64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (double)s[i];
  }
}
/**
 * @brief Cast every element from uint8_t to double (scalar fallback).
 *
 * SIMD variants: tu8_to_f64_avx512()
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type double.
 */
static void tu8_to_f64_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {
  const uint8_t *s = src->data.u8;
  double *d = dst->data.f64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (double)s[i];
  }
}
/**
 * @brief Cast every element from uint32_t to double (scalar fallback).
 *
 * SIMD variants: tu32_to_f64_avx512()
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type double.
 */
static void tu32_to_f64_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const uint32_t *s = src->data.u32;
  double *d = dst->data.f64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (double)s[i];
  }
}
/**
 * @brief Cast every element from uint64_t to double (scalar fallback).
 *
 * SIMD variants: tu64_to_f64_avx512()
 *
 * @param[in]  src  Source tensor with element type uint64_t.
 * @param[out] dst  Destination tensor with element type double.
 */
static void tu64_to_f64_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const uint64_t *s = src->data.u64;
  double *d = dst->data.f64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (double)s[i];
  }
}
/**
 * @brief Cast every element from int8_t to int32_t (scalar fallback).
 *
 * SIMD variants: ts8_to_s32_avx512(), ts8_to_s32_avx2(), ts8_to_s32_sse4_2()
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
static void ts8_to_s32_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {
  const int8_t *s = src->data.s8;
  int32_t *d = dst->data.s32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int32_t)s[i];
  }
}
/**
 * @brief Cast every element from int8_t to int64_t (scalar fallback).
 *
 * SIMD variants: ts8_to_s64_avx512(), ts8_to_s64_avx2(), ts8_to_s64_sse4_2()
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
static void ts8_to_s64_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {
  const int8_t *s = src->data.s8;
  int64_t *d = dst->data.s64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int64_t)s[i];
  }
}
/**
 * @brief Cast every element from int32_t to int8_t (scalar fallback).
 *
 * SIMD variants: ts32_to_s8_avx512()
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type int8_t.
 */
static void ts32_to_s8_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {
  const int32_t *s = src->data.s32;
  int8_t *d = dst->data.s8;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int8_t)s[i];
  }
}
/**
 * @brief Cast every element from int32_t to int64_t (scalar fallback).
 *
 * SIMD variants: ts32_to_s64_avx512(), ts32_to_s64_avx2(),
 * ts32_to_s64_sse4_2()
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
static void ts32_to_s64_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const int32_t *s = src->data.s32;
  int64_t *d = dst->data.s64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int64_t)s[i];
  }
}
/**
 * @brief Cast every element from int64_t to int8_t (scalar fallback).
 *
 * SIMD variants: ts64_to_s8_avx512()
 *
 * @param[in]  src  Source tensor with element type int64_t.
 * @param[out] dst  Destination tensor with element type int8_t.
 */
static void ts64_to_s8_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {
  const int64_t *s = src->data.s64;
  int8_t *d = dst->data.s8;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int8_t)s[i];
  }
}
/**
 * @brief Cast every element from int64_t to int32_t (scalar fallback).
 *
 * SIMD variants: ts64_to_s32_avx512()
 *
 * @param[in]  src  Source tensor with element type int64_t.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
static void ts64_to_s32_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const int64_t *s = src->data.s64;
  int32_t *d = dst->data.s32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int32_t)s[i];
  }
}
/**
 * @brief Cast every element from uint8_t to uint32_t (scalar fallback).
 *
 * SIMD variants: tu8_to_u32_avx512(), tu8_to_u32_avx2()
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
static void tu8_to_u32_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {
  const uint8_t *s = src->data.u8;
  uint32_t *d = dst->data.u32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint32_t)s[i];
  }
}
/**
 * @brief Cast every element from uint8_t to uint64_t (scalar fallback).
 *
 * SIMD variants: tu8_to_u64_avx512(), tu8_to_u64_avx2(), tu8_to_u64_sse4_2()
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
static void tu8_to_u64_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {
  const uint8_t *s = src->data.u8;
  uint64_t *d = dst->data.u64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint64_t)s[i];
  }
}
/**
 * @brief Cast every element from uint32_t to uint8_t (scalar fallback).
 *
 * SIMD variants: tu32_to_u8_avx512()
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
static void tu32_to_u8_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {
  const uint32_t *s = src->data.u32;
  uint8_t *d = dst->data.u8;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint8_t)s[i];
  }
}
/**
 * @brief Cast every element from uint32_t to uint64_t (scalar fallback).
 *
 * SIMD variants: tu32_to_u64_avx512(), tu32_to_u64_avx2(),
 * tu32_to_u64_sse4_2()
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
static void tu32_to_u64_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const uint32_t *s = src->data.u32;
  uint64_t *d = dst->data.u64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint64_t)s[i];
  }
}
/**
 * @brief Cast every element from uint64_t to uint8_t (scalar fallback).
 *
 * SIMD variants: tu64_to_u8_avx512()
 *
 * @param[in]  src  Source tensor with element type uint64_t.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
static void tu64_to_u8_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {
  const uint64_t *s = src->data.u64;
  uint8_t *d = dst->data.u8;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint8_t)s[i];
  }
}
/**
 * @brief Cast every element from uint64_t to uint32_t (scalar fallback).
 *
 * SIMD variants: tu64_to_u32_avx512()
 *
 * @param[in]  src  Source tensor with element type uint64_t.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
static void tu64_to_u32_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const uint64_t *s = src->data.u64;
  uint32_t *d = dst->data.u32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint32_t)s[i];
  }
}
/**
 * @brief Cast every element from int8_t to uint8_t (scalar fallback).
 *
 * SIMD variants: ts8_to_u8_avx512(), ts8_to_u8_avx_avx2(),
 * ts8_to_u8_sse4_2()
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
static void ts8_to_u8_scalar(const Tensor *restrict src, Tensor *restrict dst) {
  const int8_t *s = src->data.s8;
  uint8_t *d = dst->data.u8;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint8_t)s[i];
  }
}
/**
 * @brief Cast every element from int8_t to uint32_t (scalar fallback).
 *
 * SIMD variants: ts8_to_u32_avx512(), ts8_to_u32_avx2(), ts8_to_u32_sse4_2()
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
static void ts8_to_u32_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {
  const int8_t *s = src->data.s8;
  uint32_t *d = dst->data.u32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint32_t)s[i];
  }
}
/**
 * @brief Cast every element from int8_t to uint64_t (scalar fallback).
 *
 * SIMD variants: ts8_to_u64_avx512()
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
static void ts8_to_u64_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {
  const int8_t *s = src->data.s8;
  uint64_t *d = dst->data.u64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint64_t)s[i];
  }
}
/**
 * @brief Cast every element from int32_t to uint8_t (scalar fallback).
 *
 * SIMD variants: ts32_to_u8_avx512(), ts32_to_u8_avx2(), ts32_to_u8_sse4_2()
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
static void ts32_to_u8_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {
  const int32_t *s = src->data.s32;
  uint8_t *d = dst->data.u8;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint8_t)s[i];
  }
}
/**
 * @brief Cast every element from int32_t to uint32_t (scalar fallback).
 *
 * SIMD variants: ts32_to_u32_avx512(), ts32_to_u32_avx_avx2(),
 * ts32_to_u32_sse4_2()
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
static void ts32_to_u32_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const int32_t *s = src->data.s32;
  uint32_t *d = dst->data.u32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint32_t)s[i];
  }
}
/**
 * @brief Cast every element from int32_t to uint64_t (scalar fallback).
 *
 * SIMD variants: ts32_to_u64_avx512(), ts32_to_u64_avx2(),
 * ts32_to_u64_sse4_2()
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
static void ts32_to_u64_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const int32_t *s = src->data.s32;
  uint64_t *d = dst->data.u64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint64_t)s[i];
  }
}
/**
 * @brief Cast every element from int64_t to uint8_t (scalar fallback).
 *
 * SIMD variants: ts64_to_u8_avx512()
 *
 * @param[in]  src  Source tensor with element type int64_t.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
static void ts64_to_u8_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {
  const int64_t *s = src->data.s64;
  uint8_t *d = dst->data.u8;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint8_t)s[i];
  }
}
/**
 * @brief Cast every element from int64_t to uint32_t (scalar fallback).
 *
 * SIMD variants: ts64_to_u32_avx512()
 *
 * @param[in]  src  Source tensor with element type int64_t.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
static void ts64_to_u32_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const int64_t *s = src->data.s64;
  uint32_t *d = dst->data.u32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint32_t)s[i];
  }
}
/**
 * @brief Cast every element from int64_t to uint64_t (scalar fallback).
 *
 * SIMD variants: ts64_to_u64_avx512(), ts64_to_u64_avx_avx2(),
 * ts64_to_u64_sse4_2()
 *
 * @param[in]  src  Source tensor with element type int64_t.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
static void ts64_to_u64_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const int64_t *s = src->data.s64;
  uint64_t *d = dst->data.u64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (uint64_t)s[i];
  }
}
/**
 * @brief Cast every element from uint8_t to int8_t (scalar fallback).
 *
 * SIMD variants: tu8_to_s8_avx512(), tu8_to_s8_avx_avx2(),
 * tu8_to_s8_sse4_2()
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type int8_t.
 */
static void tu8_to_s8_scalar(const Tensor *restrict src, Tensor *restrict dst) {
  const uint8_t *s = src->data.u8;
  int8_t *d = dst->data.s8;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int8_t)s[i];
  }
}
/**
 * @brief Cast every element from uint8_t to int32_t (scalar fallback).
 *
 * SIMD variants: tu8_to_s32_avx512(), tu8_to_s32_avx2(), tu8_to_s32_sse4_2()
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
static void tu8_to_s32_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {
  const uint8_t *s = src->data.u8;
  int32_t *d = dst->data.s32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int32_t)s[i];
  }
}
/**
 * @brief Cast every element from uint8_t to int64_t (scalar fallback).
 *
 * SIMD variants: tu8_to_s64_avx512(), tu8_to_s64_avx2(), tu8_to_s64_sse4_2()
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
static void tu8_to_s64_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {
  const uint8_t *s = src->data.u8;
  int64_t *d = dst->data.s64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int64_t)s[i];
  }
}
/**
 * @brief Cast every element from uint32_t to int8_t (scalar fallback).
 *
 * SIMD variants: tu32_to_s8_avx512()
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type int8_t.
 */
static void tu32_to_s8_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {
  const uint32_t *s = src->data.u32;
  int8_t *d = dst->data.s8;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int8_t)s[i];
  }
}
/**
 * @brief Cast every element from uint32_t to int32_t (scalar fallback).
 *
 * SIMD variants: tu32_to_s32_avx512(), tu32_to_s32_avx_avx2(),
 * tu32_to_s32_sse4_2()
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
static void tu32_to_s32_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const uint32_t *s = src->data.u32;
  int32_t *d = dst->data.s32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int32_t)s[i];
  }
}
/**
 * @brief Cast every element from uint32_t to int64_t (scalar fallback).
 *
 * SIMD variants: tu32_to_s64_avx512(), tu32_to_s64_avx2(),
 * tu32_to_s64_sse4_2()
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
static void tu32_to_s64_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const uint32_t *s = src->data.u32;
  int64_t *d = dst->data.s64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int64_t)s[i];
  }
}
/**
 * @brief Cast every element from uint64_t to int8_t (scalar fallback).
 *
 * SIMD variants: tu64_to_s8_avx512()
 *
 * @param[in]  src  Source tensor with element type uint64_t.
 * @param[out] dst  Destination tensor with element type int8_t.
 */
static void tu64_to_s8_scalar(const Tensor *restrict src,
                              Tensor *restrict dst) {
  const uint64_t *s = src->data.u64;
  int8_t *d = dst->data.s8;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int8_t)s[i];
  }
}
/**
 * @brief Cast every element from uint64_t to int32_t (scalar fallback).
 *
 * SIMD variants: tu64_to_s32_avx512()
 *
 * @param[in]  src  Source tensor with element type uint64_t.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
static void tu64_to_s32_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const uint64_t *s = src->data.u64;
  int32_t *d = dst->data.s32;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int32_t)s[i];
  }
}
/**
 * @brief Cast every element from uint64_t to int64_t (scalar fallback).
 *
 * SIMD variants: tu64_to_s64_avx512(), tu64_to_s64_avx_avx2(),
 * tu64_to_s64_sse4_2()
 *
 * @param[in]  src  Source tensor with element type uint64_t.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
static void tu64_to_s64_scalar(const Tensor *restrict src,
                               Tensor *restrict dst) {
  const uint64_t *s = src->data.u64;
  int64_t *d = dst->data.s64;
  for (size_t i = 0; i < src->size; i++) {
    d[i] = (int64_t)s[i];
  }
}

/* =========================================================================
 * SIMD kernels
 * ========================================================================= */

/**
 * @brief Cast every element from _Float16 to float using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type _Float16.
 * @param[out] dst  Destination tensor with element type float.
 */
__attribute__((target("avx512f"))) static inline void
tfp16_to_f32_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F32_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const _Float16 *s = src->data.half;
  float *d = dst->data.f32;
  for (; i < n; i += step) {
    __m256i v_src = _mm256_loadu_si256((const __m256i_u *)&s[i]);
    __m512 v_dst = _mm512_cvtph_ps(v_src);
    _mm512_storeu_ps(&d[i], v_dst);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, float)
  }
}
/**
 * @brief Cast every element from _Float16 to float using AVX/AVX2.
 *
 * Requires: F16C
 *
 * @param[in]  src  Source tensor with element type _Float16.
 * @param[out] dst  Destination tensor with element type float.
 */
__attribute__((target("f16c"))) static inline void
tfp16_to_f32_avx_avx2_fp16c(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F32_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const _Float16 *s = src->data.half;
  float *d = dst->data.f32;
  for (; i < n; i += step) {
    __m128i v_src = _mm_loadu_si128((const __m128i_u *)&s[i]);
    __m256 v_dst = _mm256_cvtph_ps(v_src);
    _mm256_storeu_ps(&d[i], v_dst);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, float)
  }
}
/**
 * @brief Cast every element from _Float16 to double using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type _Float16.
 * @param[out] dst  Destination tensor with element type double.
 */
__attribute__((target("avx512f,avx512fp16"))) static inline void
tfp16_to_f64_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const _Float16 *s = src->data.half;
  double *d = dst->data.f64;
  for (; i < n; i += step) {
    __m128h v_src = (__m128h)_mm_loadu_si128((const __m128i_u *)&s[i]);
    __m512d v_dst = _mm512_cvtph_pd(v_src);
    _mm512_storeu_pd(&d[i], v_dst);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, double)
  }
}
/**
 * @brief Cast every element from _Float16 to double using AVX/AVX2.
 *
 * Requires: F16C
 *
 * @param[in]  src  Source tensor with element type _Float16.
 * @param[out] dst  Destination tensor with element type double.
 */
__attribute__((target("f16c"))) static inline void
tfp16_to_f64_avx_avx2_fp16c(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const _Float16 *s = src->data.half;
  double *d = dst->data.f64;
  for (; i < n; i += step) {
    __m128i v = _mm_loadl_epi64((const __m128i_u *)&s[i]);
    __m128 f32 = _mm_cvtph_ps(v);
    __m256d r = _mm256_cvtps_pd(f32);
    _mm256_storeu_pd(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, double)
  }
}
/**
 * @brief Cast every element from _Float16 to __bf16 using AVX-512.
 *
 * Requires: AVX512F, AVX512BF16
 *
 * @param[in]  src  Source tensor with element type _Float16.
 * @param[out] dst  Destination tensor with element type __bf16.
 */
__attribute__((target("avx512f,avx512bf16"))) static inline void
tfp16_to_bf16_avx512bf16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = 16;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const _Float16 *s = src->data.half;
  __bf16 *d = dst->data.bf16;
  for (; i < n; i += step) {
    __m256i v_src = _mm256_loadu_si256((const __m256i_u *)&s[i]);
    __m512 v_inter = _mm512_cvtph_ps(v_src);
    __m256bh v_dst = _mm512_cvtneps_pbh(v_inter);
    memcpy(&d[i], &v_dst, sizeof(v_dst));
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, __bf16)
  }
}
/**
 * @brief Cast every element from float to _Float16 using AVX-512.
 *
 * Requires: AVX512F, AVX512FP16
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type _Float16.
 */
__attribute__((target("avx512f,avx512fp16"))) static inline void
tf32_to_fp16_avx512fp16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_FP16_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const float *s = src->data.f32;
  _Float16 *d = dst->data.half;
  for (; i < n; i += step) {
    __m512 v0 = _mm512_loadu_ps(&s[i]);
    __m512 v1 = _mm512_loadu_ps(&s[i + 16]);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
    __m256i r0 = (__m256i)_mm512_cvtps_ph(v0, _MM_FROUND_TO_NEAREST_INT |
                                                  _MM_FROUND_NO_EXC);
    __m256i r1 = (__m256i)_mm512_cvtps_ph(v1, _MM_FROUND_TO_NEAREST_INT |
                                                  _MM_FROUND_NO_EXC);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
    _mm256_storeu_si256((__m256i_u *)&d[i], r0);
    _mm256_storeu_si256((__m256i_u *)&d[i + 16], r1);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, _Float16)
  }
}
/**
 * @brief Cast every element from float to _Float16 using AVX/AVX2.
 *
 * Requires: F16C
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type _Float16.
 */
__attribute__((target("f16c"))) static inline void
tf32_to_fp16_avx_avx2_f16c(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_FP16_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const float *s = src->data.f32;
  _Float16 *d = dst->data.half;
  for (; i < n; i += step) {
    __m256 v0 = _mm256_loadu_ps(&s[i]);
    __m256 v1 = _mm256_loadu_ps(&s[i + 8]);
    __m128i r0 =
        _mm256_cvtps_ph(v0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m128i r1 =
        _mm256_cvtps_ph(v1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i_u *)&d[i], r0);
    _mm_storeu_si128((__m128i_u *)&d[i + 8], r1);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, _Float16)
  }
}
/**
 * @brief Cast every element from float to double using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type double.
 */
__attribute__((target("avx512f"))) static inline void
tf32_to_f64_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const float *s = src->data.f32;
  double *d = dst->data.f64;
  for (; i < n; i += step) {
    __m256 v = _mm256_loadu_ps(&s[i]);
    __m512d r = _mm512_cvtps_pd(v);
    _mm512_storeu_pd(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, double)
  }
}
/**
 * @brief Cast every element from float to double using AVX/AVX2.
 *
 * Requires: AVX/AVX2
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type double.
 */
__attribute__((target("avx,avx2"))) static inline void
tf32_to_f64_avx_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const float *s = src->data.f32;
  double *d = dst->data.f64;
  for (; i < n; i += step) {
    __m128 v = _mm_loadu_ps(&s[i]);
    __m256d r = _mm256_cvtps_pd(v);
    _mm256_storeu_pd(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, double)
  }
}
/**
 * @brief Cast every element from float to double using SSE4.2.
 *
 * Requires: SSE4.2
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type double.
 */
__attribute__((target("sse4.2"))) static inline void
tf32_to_f64_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_SSE;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const float *s = src->data.f32;
  double *d = dst->data.f64;
  for (; i < n; i += step) {
    __m128 v = _mm_loadu_ps(&s[i]);
    __m128d r = _mm_cvtps_pd(v);
    _mm_storeu_pd(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, double)
  }
}
/**
 * @brief Cast every element from float to __bf16 using AVX-512.
 *
 * Requires: AVX512F, AVX512BF16
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type __bf16.
 */
__attribute__((
    target("avx512f,avx512bf16,avx512bw,avx512vl"))) static inline void
tf32_to_bf16_avx512bf16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_BF16_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const float *s = src->data.f32;
  __bf16 *d = dst->data.bf16;
  for (; i < n; i += step) {
    __m512 v0 = _mm512_loadu_ps(&s[i]);
    __m512 v1 = _mm512_loadu_ps(&s[i + 16]);
    __m256bh r0 = _mm512_cvtneps_pbh(v0);
    __m256bh r1 = _mm512_cvtneps_pbh(v1);
    _mm256_storeu_epi16(&d[i], (__m256i)r0);
    _mm256_storeu_epi16(&d[i + 16], (__m256i)r1);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, __bf16)
  }
}
/**
 * @brief Cast every element from __bf16 to _Float16 using AVX-512.
 *
 * Requires: AVX512F, AVX512BF16, AVX512FP16
 *
 * @param[in]  src  Source tensor with element type __bf16.
 * @param[out] dst  Destination tensor with element type _Float16.
 */
__attribute__((target("avx512f,avx512bf16,avx512fp16"))) static inline void
tbf16_to_fp16_avx512bf16_fp16(const Tensor *restrict src,
                              Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_FP16_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const __bf16 *s = src->data.bf16;
  _Float16 *d = dst->data.half;
  for (; i < n; i += step) {
    __m256bh v0, v1;
    memcpy(&v0, &s[i], sizeof(v0));
    memcpy(&v1, &s[i + 16], sizeof(v1));
    __m512 f0 = _mm512_cvtpbh_ps(v0);
    __m512 f1 = _mm512_cvtpbh_ps(v1);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
    __m256i r0 = (__m256i)_mm512_cvtps_ph(f0, _MM_FROUND_TO_NEAREST_INT |
                                                  _MM_FROUND_NO_EXC);
    __m256i r1 = (__m256i)_mm512_cvtps_ph(f1, _MM_FROUND_TO_NEAREST_INT |
                                                  _MM_FROUND_NO_EXC);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
    _mm256_storeu_si256((__m256i_u *)&d[i], r0);
    _mm256_storeu_si256((__m256i_u *)&d[i + 16], r1);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, _Float16)
  }
}
/**
 * @brief Cast every element from __bf16 to float using AVX-512.
 *
 * Requires: AVX512BF16
 *
 * @param[in]  src  Source tensor with element type __bf16.
 * @param[out] dst  Destination tensor with element type float.
 */
__attribute__((target("avx512bf16"))) static inline void
tbf16_to_f32_avx512bf16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F32_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const __bf16 *s = src->data.bf16;
  float *d = dst->data.f32;
  for (; i < n; i += step) {
    __m256bh v;
    memcpy(&v, &s[i], sizeof(v));
    __m512 r = _mm512_cvtpbh_ps(v);
    _mm512_storeu_ps(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, float)
  }
}
/**
 * @brief Cast every element from __bf16 to double using AVX-512.
 *
 * Requires: AVX512F, AVX512BF16
 *
 * @param[in]  src  Source tensor with element type __bf16.
 * @param[out] dst  Destination tensor with element type double.
 */
__attribute__((target("avx512f,avx512bf16,avx512vl"))) static inline void
tbf16_to_f64_avx512bf16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const __bf16 *s = src->data.bf16;
  double *d = dst->data.f64;
  for (; i < n; i += step) {
    __m128bh v;
    memcpy(&v, &s[i], sizeof(v));
    __m256 f32 = _mm256_cvtpbh_ps(v);
    __m512d r = _mm512_cvtps_pd(f32);
    _mm512_storeu_pd(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, double)
  }
}
/**
 * @brief Cast every element from double to _Float16 using AVX-512.
 *
 * Requires: AVX512F, AVX512FP16
 *
 * @param[in]  src  Source tensor with element type double.
 * @param[out] dst  Destination tensor with element type _Float16.
 */
__attribute__((target("avx512f,avx512fp16"))) static inline void
tf64_to_fp16_avx512fp16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const double *s = src->data.f64;
  _Float16 *d = dst->data.half;
  for (; i < n; i += step) {
    __m512d v = _mm512_loadu_pd(&s[i]);
    __m128h r = _mm512_cvtpd_ph(v);
    _mm_storeu_si128((__m128i_u *)&d[i], (__m128i)r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, _Float16)
  }
}
/**
 * @brief Cast every element from double to float using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type double.
 * @param[out] dst  Destination tensor with element type float.
 */
__attribute__((target("avx512f"))) static inline void
tf64_to_f32_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const double *s = src->data.f64;
  float *d = dst->data.f32;
  for (; i < n; i += step) {
    __m512d v = _mm512_loadu_pd(&s[i]);
    __m256 r = _mm512_cvtpd_ps(v);
    _mm256_storeu_ps(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, float)
  }
}
/**
 * @brief Cast every element from double to float using AVX/AVX2.
 *
 * Requires: AVX/AVX2
 *
 * @param[in]  src  Source tensor with element type double.
 * @param[out] dst  Destination tensor with element type float.
 */
__attribute__((target("avx,avx2"))) static inline void
tf64_to_f32_avx_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const double *s = src->data.f64;
  float *d = dst->data.f32;
  for (; i < n; i += step) {
    __m256d v = _mm256_loadu_pd(&s[i]);
    __m128 r = _mm256_cvtpd_ps(v);
    _mm_storeu_ps(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, float)
  }
}
/**
 * @brief Cast every element from double to float using SSE4.2.
 *
 * Requires: SSE4.2
 *
 * @param[in]  src  Source tensor with element type double.
 * @param[out] dst  Destination tensor with element type float.
 */
__attribute__((target("sse4.2"))) static inline void
tf64_to_f32_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_SSE;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const double *s = src->data.f64;
  float *d = dst->data.f32;
  for (; i < n; i += step) {
    __m128d v = _mm_loadu_pd(&s[i]);
    __m128 r = _mm_cvtpd_ps(v);
    _mm_storeu_ps(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, float)
  }
}
/**
 * @brief Cast every element from double to __bf16 using AVX-512.
 *
 * Requires: AVX512F, AVX512BF16
 *
 * @param[in]  src  Source tensor with element type double.
 * @param[out] dst  Destination tensor with element type __bf16.
 */
__attribute__((target("avx512f,avx512bf16,avx512vl"))) static inline void
tf64_to_bf16_avx512bf16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const double *s = src->data.f64;
  __bf16 *d = dst->data.bf16;
  for (; i < n; i += step) {
    __m512d v = _mm512_loadu_pd(&s[i]);
    __m256 f32 = _mm512_cvtpd_ps(v);
    __m128bh r = _mm256_cvtneps_pbh(f32);
    memcpy(&d[i], &r, sizeof(r));
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, __bf16)
  }
}
/**
 * @brief Cast every element from _Float16 to int8_t using AVX-512.
 *
 * Requires: AVX512FP16, AVX512BW, AVX512F
 *
 * @param[in]  src  Source tensor with element type _Float16.
 * @param[out] dst  Destination tensor with element type int8_t.
 */
__attribute__((target("avx512fp16,avx512bw,avx512f"))) static inline void
tfp16_to_s8_avx512fp16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S8_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const _Float16 *s = src->data.half;
  int8_t *d = dst->data.s8;
  for (; i < n; i += step) {
    __m256h v0 = (__m256h)_mm256_loadu_si256((const __m256i_u *)&s[i]);
    __m256h v1 = (__m256h)_mm256_loadu_si256((const __m256i_u *)&s[i + 16]);
    __m256h v2 = (__m256h)_mm256_loadu_si256((const __m256i_u *)&s[i + 32]);
    __m256h v3 = (__m256h)_mm256_loadu_si256((const __m256i_u *)&s[i + 48]);
    __m512i i0 = _mm512_cvtph_epi32(v0);
    __m512i i1 = _mm512_cvtph_epi32(v1);
    __m512i i2 = _mm512_cvtph_epi32(v2);
    __m512i i3 = _mm512_cvtph_epi32(v3);
    __m512i p0 = _mm512_packs_epi32(i0, i1);
    __m512i p1 = _mm512_packs_epi32(i2, i3);
    __m512i r = _mm512_packs_epi16(p0, p1);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int8_t)
  }
}
/**
 * @brief Cast every element from _Float16 to int32_t using AVX-512.
 *
 * Requires: AVX512FP16, AVX512F
 *
 * @param[in]  src  Source tensor with element type _Float16.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
__attribute__((target("avx512fp16,avx512f"))) static inline void
tfp16_to_s32_avx512fp16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S32_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const _Float16 *s = src->data.half;
  int32_t *d = dst->data.s32;
  for (; i < n; i += step) {
    __m256h v = (__m256h)_mm256_loadu_si256((const __m256i_u *)&s[i]);
    __m512i r = _mm512_cvtph_epi32(v);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int32_t)
  }
}
/**
 * @brief Cast every element from _Float16 to int64_t using AVX-512.
 *
 * Requires: AVX512F, AVX512FP16
 *
 * @param[in]  src  Source tensor with element type _Float16.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
__attribute__((target("avx512f,avx512fp16"))) static inline void
tfp16_to_s64_avx512fp16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const _Float16 *s = src->data.half;
  int64_t *d = dst->data.s64;
  for (; i < n; i += step) {
    __m128h v = (__m128h)_mm_loadu_si128((const __m128i_u *)&s[i]);
    __m512i r = _mm512_cvtph_epi64(v);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int64_t)
  }
}
/**
 * @brief Cast every element from _Float16 to uint8_t using AVX-512.
 *
 * Requires: AVX512F, AVX512FP16, AVX512BW
 *
 * @param[in]  src  Source tensor with element type _Float16.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
__attribute__((target("avx512f,avx512fp16,avx512bw"))) static inline void
tfp16_to_u8_avx512fp16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U8_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const _Float16 *s = src->data.half;
  uint8_t *d = dst->data.u8;
  for (; i < n; i += step) {
    __m256h v0 = (__m256h)_mm256_loadu_si256((const __m256i_u *)&s[i]);
    __m256h v1 = (__m256h)_mm256_loadu_si256((const __m256i_u *)&s[i + 16]);
    __m256h v2 = (__m256h)_mm256_loadu_si256((const __m256i_u *)&s[i + 32]);
    __m256h v3 = (__m256h)_mm256_loadu_si256((const __m256i_u *)&s[i + 48]);
    __m512i i0 = _mm512_cvtph_epu32(v0);
    __m512i i1 = _mm512_cvtph_epu32(v1);
    __m512i i2 = _mm512_cvtph_epu32(v2);
    __m512i i3 = _mm512_cvtph_epu32(v3);
    __m512i p0 = _mm512_packus_epi32(i0, i1);
    __m512i p1 = _mm512_packus_epi32(i2, i3);
    __m512i r = _mm512_packus_epi16(p0, p1);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint8_t)
  }
}
/**
 * @brief Cast every element from _Float16 to uint32_t using AVX-512.
 *
 * Requires: AVX512F, AVX512FP16
 *
 * @param[in]  src  Source tensor with element type _Float16.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
__attribute__((target("avx512f,avx512fp16"))) static inline void
tfp16_to_u32_avx512fp16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U32_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const _Float16 *s = src->data.half;
  uint32_t *d = dst->data.u32;
  for (; i < n; i += step) {
    __m256h v = (__m256h)_mm256_loadu_si256((const __m256i_u *)&s[i]);
    __m512i r = _mm512_cvtph_epu32(v);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint32_t)
  }
}
/**
 * @brief Cast every element from _Float16 to uint64_t using AVX-512.
 *
 * Requires: AVX512F, AVX512FP16
 *
 * @param[in]  src  Source tensor with element type _Float16.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
__attribute__((target("avx512f,avx512fp16"))) static inline void
tfp16_to_u64_avx512fp16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const _Float16 *s = src->data.half;
  uint64_t *d = dst->data.u64;
  for (; i < n; i += step) {
    __m128h v = (__m128h)_mm_loadu_si128((const __m128i_u *)&s[i]);
    __m512i r = _mm512_cvtph_epu64(v);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint64_t)
  }
}
/**
 * @brief Cast every element from __bf16 to int8_t using AVX-512.
 *
 * Requires: AVX512BF16, AVX512BW, AVX512F
 *
 * @param[in]  src  Source tensor with element type __bf16.
 * @param[out] dst  Destination tensor with element type int8_t.
 */
__attribute__((target("avx512bf16,avx512bw,avx512f"))) static inline void
tbf16_to_s8_avx512bf16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S8_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const __bf16 *s = src->data.bf16;
  int8_t *d = dst->data.s8;
  for (; i < n; i += step) {
    __m256bh b0, b1, b2, b3;
    memcpy(&b0, &s[i], sizeof(b0));
    memcpy(&b1, &s[i + 16], sizeof(b1));
    memcpy(&b2, &s[i + 32], sizeof(b2));
    memcpy(&b3, &s[i + 48], sizeof(b3));
    __m512 f0 = _mm512_cvtpbh_ps(b0);
    __m512 f1 = _mm512_cvtpbh_ps(b1);
    __m512 f2 = _mm512_cvtpbh_ps(b2);
    __m512 f3 = _mm512_cvtpbh_ps(b3);
    __m512i i0 = _mm512_cvtps_epi32(f0);
    __m512i i1 = _mm512_cvtps_epi32(f1);
    __m512i i2 = _mm512_cvtps_epi32(f2);
    __m512i i3 = _mm512_cvtps_epi32(f3);
    __m512i s01 = _mm512_packs_epi32(i0, i1);
    __m512i s23 = _mm512_packs_epi32(i2, i3);
    __m512i r = _mm512_packus_epi16(s01, s23);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int8_t)
  }
}
/**
 * @brief Cast every element from __bf16 to int32_t using AVX-512.
 *
 * Requires: AVX512BF16, AVX512F
 *
 * @param[in]  src  Source tensor with element type __bf16.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
__attribute__((target("avx512bf16,avx512f"))) static inline void
tbf16_to_s32_avx512bf16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S32_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const __bf16 *s = src->data.bf16;
  int32_t *d = dst->data.s32;
  for (; i < n; i += step) {
    __m256bh v;
    memcpy(&v, &s[i], sizeof(v));
    __m512 f = _mm512_cvtpbh_ps(v);
    __m512i r = _mm512_cvtps_epi32(f);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int32_t)
  }
}
/**
 * @brief Cast every element from __bf16 to int64_t using AVX-512.
 *
 * Requires: AVX512F, AVX512BF16, AVX512VL, AVX512DQ
 *
 * @param[in]  src  Source tensor with element type __bf16.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
__attribute__((
    target("avx512f,avx512bf16,avx512vl,avx512dq"))) static inline void
tbf16_to_s64_avx512bf16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const __bf16 *s = src->data.bf16;
  int64_t *d = dst->data.s64;
  for (; i < n; i += step) {
    __m128bh v;
    memcpy(&v, &s[i], sizeof(v));
    __m256 f = _mm256_cvtpbh_ps(v);
    __m512i r = _mm512_cvtps_epi64(f);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int64_t)
  }
}
/**
 * @brief Cast every element from __bf16 to uint8_t using AVX-512.
 *
 * Requires: AVX512BF16, AVX512BW, AVX512F
 *
 * @param[in]  src  Source tensor with element type __bf16.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
__attribute__((target("avx512bf16,avx512bw,avx512f"))) static inline void
tbf16_to_u8_avx512bf16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U8_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const __bf16 *s = src->data.bf16;
  uint8_t *d = dst->data.u8;
  for (; i < n; i += step) {
    __m256bh b0, b1, b2, b3;
    memcpy(&b0, &s[i], sizeof(b0));
    memcpy(&b1, &s[i + 16], sizeof(b1));
    memcpy(&b2, &s[i + 32], sizeof(b2));
    memcpy(&b3, &s[i + 48], sizeof(b3));
    __m512 f0 = _mm512_cvtpbh_ps(b0);
    __m512 f1 = _mm512_cvtpbh_ps(b1);
    __m512 f2 = _mm512_cvtpbh_ps(b2);
    __m512 f3 = _mm512_cvtpbh_ps(b3);
    __m512i i0 = _mm512_cvtps_epu32(f0);
    __m512i i1 = _mm512_cvtps_epu32(f1);
    __m512i i2 = _mm512_cvtps_epu32(f2);
    __m512i i3 = _mm512_cvtps_epu32(f3);
    __m512i s01 = _mm512_packus_epi32(i0, i1);
    __m512i s23 = _mm512_packus_epi32(i2, i3);
    __m512i r = _mm512_packus_epi16(s01, s23);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint8_t)
  }
}
/**
 * @brief Cast every element from __bf16 to uint32_t using AVX-512.
 *
 * Requires: AVX512BF16, AVX512F
 *
 * @param[in]  src  Source tensor with element type __bf16.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
__attribute__((target("avx512bf16,avx512f"))) static inline void
tbf16_to_u32_avx512bf16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U32_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const __bf16 *s = src->data.bf16;
  uint32_t *d = dst->data.u32;
  for (; i < n; i += step) {
    __m256bh v;
    memcpy(&v, &s[i], sizeof(v));
    __m512 f = _mm512_cvtpbh_ps(v);
    __m512i r = _mm512_cvtps_epu32(f);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint32_t)
  }
}
/**
 * @brief Cast every element from __bf16 to uint64_t using AVX-512.
 *
 * Requires: AVX512F, AVX512BF16, AVX512VL, AVX512DQ
 *
 * @param[in]  src  Source tensor with element type __bf16.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
__attribute__((
    target("avx512f,avx512bf16,avx512vl,avx512dq"))) static inline void
tbf16_to_u64_avx512bf16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const __bf16 *s = src->data.bf16;
  uint64_t *d = dst->data.u64;
  for (; i < n; i += step) {
    __m128bh v;
    memcpy(&v, &s[i], sizeof(v));
    __m256 f = _mm256_cvtpbh_ps(v);
    __m512i r = _mm512_cvtps_epu64(f);
    _mm512_storeu_si512((__m512i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint64_t)
  }
}
/**
 * @brief Cast every element from float to int8_t using AVX-512.
 *
 * Requires: AVX512F, AVX512BW
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type int8_t.
 */
__attribute__((target("avx512f,avx512bw"))) static inline void
tf32_to_s8_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S8_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const float *s = src->data.f32;
  int8_t *d = dst->data.s8;
  for (; i < n; i += step) {
    __m512 v0 = _mm512_loadu_ps(&s[i]);
    __m512 v1 = _mm512_loadu_ps(&s[i + 16]);
    __m512 v2 = _mm512_loadu_ps(&s[i + 32]);
    __m512 v3 = _mm512_loadu_ps(&s[i + 48]);
    __m512i i0 = _mm512_cvtps_epi32(v0);
    __m512i i1 = _mm512_cvtps_epi32(v1);
    __m512i i2 = _mm512_cvtps_epi32(v2);
    __m512i i3 = _mm512_cvtps_epi32(v3);
    __m512i s01 = _mm512_packs_epi32(i0, i1);
    __m512i s23 = _mm512_packs_epi32(i2, i3);
    __m512i r = _mm512_packs_epi16(s01, s23);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int8_t)
  }
}
/**
 * @brief Cast every element from float to int8_t using AVX2.
 *
 * Requires: AVX2
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type int8_t.
 */
__attribute__((target("avx2"))) static inline void
tf32_to_s8_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S8_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const float *s = src->data.f32;
  int8_t *d = dst->data.s8;
  for (; i < n; i += step) {
    __m256 v0 = _mm256_loadu_ps(&s[i]);
    __m256 v1 = _mm256_loadu_ps(&s[i + 8]);
    __m256 v2 = _mm256_loadu_ps(&s[i + 16]);
    __m256 v3 = _mm256_loadu_ps(&s[i + 24]);
    __m256i i0 = _mm256_cvtps_epi32(v0);
    __m256i i1 = _mm256_cvtps_epi32(v1);
    __m256i i2 = _mm256_cvtps_epi32(v2);
    __m256i i3 = _mm256_cvtps_epi32(v3);
    __m256i s01 = _mm256_packs_epi32(i0, i1);
    __m256i s23 = _mm256_packs_epi32(i2, i3);
    __m256i r = _mm256_packs_epi16(s01, s23);
    _mm256_storeu_si256((__m256i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int8_t)
  }
}
/**
 * @brief Cast every element from float to int32_t using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
__attribute__((target("avx512f"))) static inline void
tf32_to_s32_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S32_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const float *s = src->data.f32;
  int32_t *d = dst->data.s32;
  for (; i < n; i += step) {
    __m512 v = _mm512_loadu_ps(&s[i]);
    __m512i r = _mm512_cvtps_epi32(v);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int32_t)
  }
}
/**
 * @brief Cast every element from float to int32_t using AVX/AVX2.
 *
 * Requires: AVX/AVX2
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
__attribute__((target("avx,avx2"))) static inline void
tf32_to_s32_avx_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S32_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const float *s = src->data.f32;
  int32_t *d = dst->data.s32;
  for (; i < n; i += step) {
    __m256 v = _mm256_loadu_ps(&s[i]);
    __m256i r = _mm256_cvtps_epi32(v);
    _mm256_storeu_si256((__m256i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int32_t)
  }
}
/**
 * @brief Cast every element from float to int32_t using SSE4.2.
 *
 * Requires: SSE4.2
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
__attribute__((target("sse4.2"))) static inline void
tf32_to_s32_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S32_WITH_SSE;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const float *s = src->data.f32;
  int32_t *d = dst->data.s32;
  for (; i < n; i += step) {
    __m128 v = _mm_loadu_ps(&s[i]);
    __m128i r = _mm_cvtps_epi32(v);
    _mm_storeu_si128((__m128i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int32_t)
  }
}
/**
 * @brief Cast every element from float to int64_t using AVX-512.
 *
 * Requires: AVX512F, AVX512DQ
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
__attribute__((target("avx512f,avx512dq"))) static inline void
tf32_to_s64_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const float *s = src->data.f32;
  int64_t *d = dst->data.s64;
  for (; i < n; i += step) {
    __m256 v = _mm256_loadu_ps(&s[i]);
    __m512i r = _mm512_cvtps_epi64(v);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int64_t)
  }
}
/**
 * @brief Cast every element from float to uint8_t using AVX-512.
 *
 * Requires: AVX512F, AVX512BW
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
__attribute__((target("avx512f,avx512bw"))) static inline void
tf32_to_u8_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U8_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const float *s = src->data.f32;
  uint8_t *d = dst->data.u8;
  for (; i < n; i += step) {
    __m512 v0 = _mm512_loadu_ps(&s[i]);
    __m512 v1 = _mm512_loadu_ps(&s[i + 16]);
    __m512 v2 = _mm512_loadu_ps(&s[i + 32]);
    __m512 v3 = _mm512_loadu_ps(&s[i + 48]);
    __m512i i0 = _mm512_cvtps_epu32(v0);
    __m512i i1 = _mm512_cvtps_epu32(v1);
    __m512i i2 = _mm512_cvtps_epu32(v2);
    __m512i i3 = _mm512_cvtps_epu32(v3);
    __m512i s01 = _mm512_packus_epi32(i0, i1);
    __m512i s23 = _mm512_packus_epi32(i2, i3);
    __m512i r = _mm512_packus_epi16(s01, s23);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint8_t)
  }
}
/**
 * @brief Cast every element from float to uint8_t using AVX2.
 *
 * Requires: AVX2
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
__attribute__((target("avx2"))) static inline void
tf32_to_u8_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U8_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const float *s = src->data.f32;
  uint8_t *d = dst->data.u8;
  for (; i < n; i += step) {
    __m256 v0 = _mm256_loadu_ps(&s[i]);
    __m256 v1 = _mm256_loadu_ps(&s[i + 8]);
    __m256 v2 = _mm256_loadu_ps(&s[i + 16]);
    __m256 v3 = _mm256_loadu_ps(&s[i + 24]);
    __m256i zero = _mm256_setzero_si256();
    __m256i i0 = _mm256_max_epi32(_mm256_cvtps_epi32(v0), zero);
    __m256i i1 = _mm256_max_epi32(_mm256_cvtps_epi32(v1), zero);
    __m256i i2 = _mm256_max_epi32(_mm256_cvtps_epi32(v2), zero);
    __m256i i3 = _mm256_max_epi32(_mm256_cvtps_epi32(v3), zero);
    __m256i s01 = _mm256_packus_epi32(i0, i1);
    __m256i s23 = _mm256_packus_epi32(i2, i3);
    __m256i r = _mm256_packus_epi16(s01, s23);
    _mm256_storeu_si256((__m256i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint8_t)
  }
}
/**
 * @brief Cast every element from float to uint32_t using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
__attribute__((target("avx512f"))) static inline void
tf32_to_u32_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U32_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const float *s = src->data.f32;
  uint32_t *d = dst->data.u32;
  for (; i < n; i += step) {
    __m512 v = _mm512_loadu_ps(&s[i]);
    __m512i r = _mm512_cvtps_epu32(v);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint32_t)
  }
}
/**
 * @brief Cast every element from float to uint64_t using AVX-512.
 *
 * Requires: AVX512F, AVX512DQ
 *
 * @param[in]  src  Source tensor with element type float.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
__attribute__((target("avx512f,avx512dq"))) static inline void
tf32_to_u64_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const float *s = src->data.f32;
  uint64_t *d = dst->data.u64;
  for (; i < n; i += step) {
    __m256 v = _mm256_loadu_ps(&s[i]);
    __m512i r = _mm512_cvtps_epu64(v);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint64_t)
  }
}
/**
 * @brief Cast every element from double to int8_t using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type double.
 * @param[out] dst  Destination tensor with element type int8_t.
 */
__attribute__((target("avx512f,avx2"))) static inline void
tf64_to_s8_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const double *s = src->data.f64;
  int8_t *d = dst->data.s8;
  for (; i < n; i += step) {
    __m512d v = _mm512_loadu_pd(&s[i]);
    __m256i s32 = _mm512_cvtpd_epi32(v);
    __m128i lo = _mm256_castsi256_si128(s32);
    __m128i hi = _mm256_extracti128_si256(s32, 1);
    __m128i s16 = _mm_packs_epi32(lo, hi);
    __m128i s8 = _mm_packs_epi16(s16, _mm_setzero_si128());
    _mm_storeu_si64(&d[i], s8);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int8_t)
  }
}
/**
 * @brief Cast every element from double to int32_t using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type double.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
__attribute__((target("avx512f"))) static inline void
tf64_to_s32_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const double *s = src->data.f64;
  int32_t *d = dst->data.s32;
  for (; i < n; i += step) {
    __m512d v = _mm512_loadu_pd(&s[i]);
    __m256i r = _mm512_cvtpd_epi32(v);
    _mm256_storeu_si256((__m256i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int32_t)
  }
}
/**
 * @brief Cast every element from double to int32_t using AVX/AVX2.
 *
 * Requires: AVX/AVX2
 *
 * @param[in]  src  Source tensor with element type double.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
__attribute__((target("avx,avx2"))) static inline void
tf64_to_s32_avx_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const double *s = src->data.f64;
  int32_t *d = dst->data.s32;
  for (; i < n; i += step) {
    __m256d v = _mm256_loadu_pd(&s[i]);
    __m128i r = _mm256_cvtpd_epi32(v);
    _mm_storeu_si128((__m128i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int32_t)
  }
}
/**
 * @brief Cast every element from double to int32_t using SSE4.2.
 *
 * Requires: SSE4.2
 *
 * @param[in]  src  Source tensor with element type double.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
__attribute__((target("sse4.2"))) static inline void
tf64_to_s32_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_SSE;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const double *s = src->data.f64;
  int32_t *d = dst->data.s32;
  for (; i < n; i += step) {
    __m128d v = _mm_loadu_pd(&s[i]);
    __m128i r = _mm_cvtpd_epi32(v);
    _mm_storeu_si64(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int32_t)
  }
}
/**
 * @brief Cast every element from double to int64_t using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type double.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
__attribute__((target("avx512f,avx512dq"))) static inline void
tf64_to_s64_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const double *s = src->data.f64;
  int64_t *d = dst->data.s64;
  for (; i < n; i += step) {
    __m512d v = _mm512_loadu_pd(&s[i]);
    __m512i r = _mm512_cvtpd_epi64(v);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int64_t)
  }
}
/**
 * @brief Cast every element from double to uint8_t using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type double.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
__attribute__((target("avx512f,avx2"))) static inline void
tf64_to_u8_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const double *s = src->data.f64;
  uint8_t *d = dst->data.u8;
  for (; i < n; i += step) {
    __m512d v = _mm512_loadu_pd(&s[i]);
    __m256i u32 = _mm512_cvtpd_epu32(v);
    __m128i lo = _mm256_castsi256_si128(u32);
    __m128i hi = _mm256_extracti128_si256(u32, 1);
    __m128i u16 = _mm_packus_epi32(lo, hi);
    __m128i u8 = _mm_packus_epi16(u16, _mm_setzero_si128());
    _mm_storeu_si64(&d[i], u8);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint8_t)
  }
}
/**
 * @brief Cast every element from double to uint32_t using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type double.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
__attribute__((target("avx512f"))) static inline void
tf64_to_u32_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const double *s = src->data.f64;
  uint32_t *d = dst->data.u32;
  for (; i < n; i += step) {
    __m512d v = _mm512_loadu_pd(&s[i]);
    __m256i r = _mm512_cvtpd_epu32(v);
    _mm256_storeu_si256((__m256i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint32_t)
  }
}
/**
 * @brief Cast every element from double to uint64_t using AVX-512.
 *
 * Requires: AVX512F, AVX512DQ
 *
 * @param[in]  src  Source tensor with element type double.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
__attribute__((target("avx512f,avx512dq"))) static inline void
tf64_to_u64_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const double *s = src->data.f64;
  uint64_t *d = dst->data.u64;
  for (; i < n; i += step) {
    __m512d v = _mm512_loadu_pd(&s[i]);
    __m512i r = _mm512_cvtpd_epu64(v);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint64_t)
  }
}
/**
 * @brief Cast every element from int8_t to _Float16 using AVX-512.
 *
 * Requires: AVX512F, AVX512FP16
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type _Float16.
 */
__attribute__((target("avx512f,avx512fp16"))) static inline void
ts8_to_fp16_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S8_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int8_t *s = src->data.s8;
  _Float16 *d = dst->data.half;
  for (; i < n; i += step) {
    __m128i b0 = _mm_loadu_si128((const __m128i_u *)&s[i]);
    __m128i b1 = _mm_loadu_si128((const __m128i_u *)&s[i + 16]);
    __m128i b2 = _mm_loadu_si128((const __m128i_u *)&s[i + 32]);
    __m128i b3 = _mm_loadu_si128((const __m128i_u *)&s[i + 48]);
    __m512i i0 = _mm512_cvtepi8_epi32(b0);
    __m512i i1 = _mm512_cvtepi8_epi32(b1);
    __m512i i2 = _mm512_cvtepi8_epi32(b2);
    __m512i i3 = _mm512_cvtepi8_epi32(b3);
    __m256h r0 = _mm512_cvtepi32_ph(i0);
    __m256h r1 = _mm512_cvtepi32_ph(i1);
    __m256h r2 = _mm512_cvtepi32_ph(i2);
    __m256h r3 = _mm512_cvtepi32_ph(i3);
    _mm256_storeu_si256((__m256i_u *)&d[i], (__m256i)r0);
    _mm256_storeu_si256((__m256i_u *)&d[i + 16], (__m256i)r1);
    _mm256_storeu_si256((__m256i_u *)&d[i + 32], (__m256i)r2);
    _mm256_storeu_si256((__m256i_u *)&d[i + 48], (__m256i)r3);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, _Float16)
  }
}
/**
 * @brief Cast every element from int32_t to _Float16 using AVX-512.
 *
 * Requires: AVX512F,  AVX512FP16
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type _Float16.
 */
__attribute__((target("avx512f,avx512fp16"))) static inline void
ts32_to_fp16_avx512fp16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_FP16_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int32_t *s = src->data.s32;
  _Float16 *d = dst->data.half;
  for (; i < n; i += step) {
    __m512i v0 = _mm512_loadu_si512(&s[i]);
    __m512i v1 = _mm512_loadu_si512(&s[i + 16]);
    __m256h r0 = _mm512_cvtepi32_ph(v0);
    __m256h r1 = _mm512_cvtepi32_ph(v1);
    _mm256_storeu_si256((__m256i_u *)&d[i], (__m256i)r0);
    _mm256_storeu_si256((__m256i_u *)&d[i + 16], (__m256i)r1);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, _Float16)
  }
}
/**
 * @brief Cast every element from int64_t to _Float16 using AVX-512.
 *
 * Requires: AVX512F, AVX512FP16
 *
 * @param[in]  src  Source tensor with element type int64_t.
 * @param[out] dst  Destination tensor with element type _Float16.
 */
__attribute__((target("avx512f,avx512fp16"))) static inline void
ts64_to_fp16_avx512fp16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int64_t *s = src->data.s64;
  _Float16 *d = dst->data.half;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    __m128h r = _mm512_cvtepi64_ph(v);
    _mm_storeu_si128((__m128i_u *)&d[i], (__m128i)r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, _Float16)
  }
}
/**
 * @brief Cast every element from uint8_t to _Float16 using AVX-512.
 *
 * Requires: AVX512F, AVX512FP16
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type _Float16.
 */
__attribute__((target("avx512f,avx512fp16"))) static inline void
tu8_to_fp16_avx512fp16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S8_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint8_t *s = src->data.u8;
  _Float16 *d = dst->data.half;
  for (; i < n; i += step) {
    __m128i b0 = _mm_loadu_si128((const __m128i_u *)&s[i]);
    __m128i b1 = _mm_loadu_si128((const __m128i_u *)&s[i + 16]);
    __m128i b2 = _mm_loadu_si128((const __m128i_u *)&s[i + 32]);
    __m128i b3 = _mm_loadu_si128((const __m128i_u *)&s[i + 48]);
    __m512i i0 = _mm512_cvtepu8_epi32(b0);
    __m512i i1 = _mm512_cvtepu8_epi32(b1);
    __m512i i2 = _mm512_cvtepu8_epi32(b2);
    __m512i i3 = _mm512_cvtepu8_epi32(b3);
    __m256h r0 = _mm512_cvtepi32_ph(i0);
    __m256h r1 = _mm512_cvtepi32_ph(i1);
    __m256h r2 = _mm512_cvtepi32_ph(i2);
    __m256h r3 = _mm512_cvtepi32_ph(i3);
    _mm256_storeu_si256((__m256i_u *)&d[i], (__m256i)r0);
    _mm256_storeu_si256((__m256i_u *)&d[i + 16], (__m256i)r1);
    _mm256_storeu_si256((__m256i_u *)&d[i + 32], (__m256i)r2);
    _mm256_storeu_si256((__m256i_u *)&d[i + 48], (__m256i)r3);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, _Float16)
  }
}
/**
 * @brief Cast every element from uint32_t to _Float16 using AVX-512.
 *
 * Requires: AVX512F, AVX512FP16
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type _Float16.
 */
__attribute__((target("avx512f,avx512fp16"))) static inline void
tu32_to_fp16_avx512fp16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_FP16_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint32_t *s = src->data.u32;
  _Float16 *d = dst->data.half;
  for (; i < n; i += step) {
    __m512i v0 = _mm512_loadu_si512(&s[i]);
    __m512i v1 = _mm512_loadu_si512(&s[i + 16]);
    __m256h r0 = _mm512_cvtepu32_ph(v0);
    __m256h r1 = _mm512_cvtepu32_ph(v1);
    _mm256_storeu_si256((__m256i_u *)&d[i], (__m256i)r0);
    _mm256_storeu_si256((__m256i_u *)&d[i + 16], (__m256i)r1);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, _Float16)
  }
}
/**
 * @brief Cast every element from uint64_t to _Float16 using AVX-512.
 *
 * Requires: AVX512F, AVX512FP16
 *
 * @param[in]  src  Source tensor with element type uint64_t.
 * @param[out] dst  Destination tensor with element type _Float16.
 */
__attribute__((target("avx512f,avx512fp16"))) static inline void
tu64_to_fp16_avx512fp16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint64_t *s = src->data.u64;
  _Float16 *d = dst->data.half;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    __m128h r = _mm512_cvtepu64_ph(v);
    _mm_storeu_si128((__m128i_u *)&d[i], (__m128i)r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, _Float16)
  }
}
/**
 * @brief Cast every element from int8_t to __bf16 using AVX-512.
 *
 * Requires: AVX512F, AVX512BF16
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type __bf16.
 */
__attribute__((target("avx512f,avx512bf16"))) static inline void
ts8_to_bf16_avx512bf16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S8_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int8_t *s = src->data.s8;
  __bf16 *d = dst->data.bf16;
  for (; i < n; i += step) {
    __m128i b0 = _mm_loadu_si128((const __m128i_u *)&s[i]);
    __m128i b1 = _mm_loadu_si128((const __m128i_u *)&s[i + 16]);
    __m128i b2 = _mm_loadu_si128((const __m128i_u *)&s[i + 32]);
    __m128i b3 = _mm_loadu_si128((const __m128i_u *)&s[i + 48]);
    __m256bh r0 =
        _mm512_cvtneps_pbh(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b0)));
    __m256bh r1 =
        _mm512_cvtneps_pbh(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b1)));
    __m256bh r2 =
        _mm512_cvtneps_pbh(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b2)));
    __m256bh r3 =
        _mm512_cvtneps_pbh(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b3)));
    memcpy(&d[i], &r0, sizeof(r0));
    memcpy(&d[i + 16], &r1, sizeof(r1));
    memcpy(&d[i + 32], &r2, sizeof(r2));
    memcpy(&d[i + 48], &r3, sizeof(r3));
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, __bf16)
  }
}
/**
 * @brief Cast every element from int32_t to __bf16 using AVX-512.
 *
 * Requires: AVX512F, AVX512BF16
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type __bf16.
 */
__attribute__((target("avx512f,avx512bf16"))) static inline void
ts32_to_bf16_avx512bf16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_BF16_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int32_t *s = src->data.s32;
  __bf16 *d = dst->data.bf16;
  for (; i < n; i += step) {
    __m512i v0 = _mm512_loadu_si512(&s[i]);
    __m512i v1 = _mm512_loadu_si512(&s[i + 16]);
    __m256bh r0 = _mm512_cvtneps_pbh(_mm512_cvtepi32_ps(v0));
    __m256bh r1 = _mm512_cvtneps_pbh(_mm512_cvtepi32_ps(v1));
    memcpy(&d[i], &r0, sizeof(r0));
    memcpy(&d[i + 16], &r1, sizeof(r1));
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, __bf16)
  }
}
/**
 * @brief Cast every element from int64_t to __bf16 using AVX-512.
 *
 * Requires: AVX512F, AVX512DQ, AVX512BF16, AVX512VL
 *
 * @param[in]  src  Source tensor with element type int64_t.
 * @param[out] dst  Destination tensor with element type __bf16.
 */
__attribute__((
    target("avx512f,avx512dq,avx512bf16,avx512vl"))) static inline void
ts64_to_bf16_avx512bf16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int64_t *s = src->data.s64;
  __bf16 *d = dst->data.bf16;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    __m256 f32 = _mm512_cvtepi64_ps(v);
    __m128bh r = _mm256_cvtneps_pbh(f32);
    memcpy(&d[i], &r, sizeof(r));
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, __bf16)
  }
}
/**
 * @brief Cast every element from uint8_t to __bf16 using AVX-512.
 *
 * Requires: AVX512F, AVX512BF16
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type __bf16.
 */
__attribute__((target("avx512f,avx512bf16"))) static inline void
tu8_to_bf16_avx512bf16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U8_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint8_t *s = src->data.u8;
  __bf16 *d = dst->data.bf16;
  for (; i < n; i += step) {
    __m128i b0 = _mm_loadu_si128((const __m128i_u *)&s[i]);
    __m128i b1 = _mm_loadu_si128((const __m128i_u *)&s[i + 16]);
    __m128i b2 = _mm_loadu_si128((const __m128i_u *)&s[i + 32]);
    __m128i b3 = _mm_loadu_si128((const __m128i_u *)&s[i + 48]);
    __m256bh r0 =
        _mm512_cvtneps_pbh(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(b0)));
    __m256bh r1 =
        _mm512_cvtneps_pbh(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(b1)));
    __m256bh r2 =
        _mm512_cvtneps_pbh(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(b2)));
    __m256bh r3 =
        _mm512_cvtneps_pbh(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(b3)));
    memcpy(&d[i], &r0, sizeof(r0));
    memcpy(&d[i + 16], &r1, sizeof(r1));
    memcpy(&d[i + 32], &r2, sizeof(r2));
    memcpy(&d[i + 48], &r3, sizeof(r3));
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, __bf16)
  }
}
/**
 * @brief Cast every element from uint32_t to __bf16 using AVX-512.
 *
 * Requires: AVX512F, AVX512BF16
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type __bf16.
 */
__attribute__((target("avx512f,avx512bf16"))) static inline void
tu32_to_bf16_avx512bf16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_BF16_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint32_t *s = src->data.u32;
  __bf16 *d = dst->data.bf16;
  for (; i < n; i += step) {
    __m512i v0 = _mm512_loadu_si512(&s[i]);
    __m512i v1 = _mm512_loadu_si512(&s[i + 16]);
    __m256bh r0 = _mm512_cvtneps_pbh(_mm512_cvtepu32_ps(v0));
    __m256bh r1 = _mm512_cvtneps_pbh(_mm512_cvtepu32_ps(v1));
    memcpy(&d[i], &r0, sizeof(r0));
    memcpy(&d[i + 16], &r1, sizeof(r1));
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, __bf16)
  }
}
/**
 * @brief Cast every element from uint64_t to __bf16 using AVX-512.
 *
 * Requires: AVX512F, AVX512DQ, AVX512BF16, AVX512VL
 *
 * @param[in]  src  Source tensor with element type uint64_t.
 * @param[out] dst  Destination tensor with element type __bf16.
 */
__attribute__((
    target("avx512f,avx512dq,avx512bf16,avx512vl"))) static inline void
tu64_to_bf16_avx512bf16(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint64_t *s = src->data.u64;
  __bf16 *d = dst->data.bf16;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    __m256 f32 = _mm512_cvtepu64_ps(v);
    __m128bh r = _mm256_cvtneps_pbh(f32);
    memcpy(&d[i], &r, sizeof(r));
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, __bf16)
  }
}
/**
 * @brief Cast every element from int8_t to float using AVX-512F.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type float.
 */
__attribute__((target("avx512f"))) static inline void
ts8_to_f32_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S8_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int8_t *s = src->data.s8;
  float *d = dst->data.f32;
  for (; i < n; i += step) {
    __m128i b0 = _mm_loadu_si128((const __m128i_u *)&s[i]);
    __m128i b1 = _mm_loadu_si128((const __m128i_u *)&s[i + 16]);
    __m128i b2 = _mm_loadu_si128((const __m128i_u *)&s[i + 32]);
    __m128i b3 = _mm_loadu_si128((const __m128i_u *)&s[i + 48]);
    _mm512_storeu_ps(&d[i], _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b0)));
    _mm512_storeu_ps(&d[i + 16], _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b1)));
    _mm512_storeu_ps(&d[i + 32], _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b2)));
    _mm512_storeu_ps(&d[i + 48], _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b3)));
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, float)
  }
}
/**
 * @brief Cast every element from int8_t to float using AVX2.
 *
 * Requires: AVX2
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type float.
 */
__attribute__((target("avx2"))) static inline void
ts8_to_f32_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S8_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int8_t *s = src->data.s8;
  float *d = dst->data.f32;
  for (; i < n; i += step) {
    __m256i i0 =
        _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i]));
    __m256i i1 =
        _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 8]));
    __m256i i2 =
        _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 16]));
    __m256i i3 =
        _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 24]));
    _mm256_storeu_ps(&d[i], _mm256_cvtepi32_ps(i0));
    _mm256_storeu_ps(&d[i + 8], _mm256_cvtepi32_ps(i1));
    _mm256_storeu_ps(&d[i + 16], _mm256_cvtepi32_ps(i2));
    _mm256_storeu_ps(&d[i + 24], _mm256_cvtepi32_ps(i3));
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, float)
  }
}
/**
 * @brief Cast every element from int32_t to float using AVX-512F.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type float.
 */
__attribute__((target("avx512f"))) static inline void
ts32_to_f32_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S32_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int32_t *s = src->data.s32;
  float *d = dst->data.f32;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    __m512 r = _mm512_cvtepi32_ps(v);
    _mm512_storeu_ps(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, float)
  }
}
/**
 * @brief Cast every element from int32_t to float using AVX/AVX2.
 *
 * Requires: AVX, AVX2
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type float.
 */
__attribute__((target("avx,avx2"))) static inline void
ts32_to_f32_avx_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S32_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int32_t *s = src->data.s32;
  float *d = dst->data.f32;
  for (; i < n; i += step) {
    __m256i v = _mm256_loadu_si256((const __m256i_u *)&s[i]);
    __m256 r = _mm256_cvtepi32_ps(v);
    _mm256_storeu_ps(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, float)
  }
}
/**
 * @brief Cast every element from int32_t to float using SSE4.2.
 *
 * Requires: SSE2, SSE4.2
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type float.
 */
__attribute__((target("sse4.2"))) static inline void
ts32_to_f32_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S32_WITH_SSE;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int32_t *s = src->data.s32;
  float *d = dst->data.f32;
  for (; i < n; i += step) {
    __m128i v = _mm_loadu_si128((const __m128i_u *)&s[i]);
    __m128 r = _mm_cvtepi32_ps(v);
    _mm_storeu_ps(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, float)
  }
}
/**
 * @brief Cast every element from int64_t to float using AVX-512F.
 *
 * Requires: AVX512F, AVX512DQ
 *
 * @param[in]  src  Source tensor with element type int64_t.
 * @param[out] dst  Destination tensor with element type float.
 */
__attribute__((target("avx512f,avx512dq"))) static inline void
ts64_to_f32_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int64_t *s = src->data.s64;
  float *d = dst->data.f32;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    __m256 r = _mm512_cvtepi64_ps(v);
    _mm256_storeu_ps(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, float)
  }
}
/**
 * @brief Cast every element from uint8_t to float using AVX-512F.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type float.
 */
__attribute__((target("avx512f"))) static inline void
tu8_to_f32_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U8_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint8_t *s = src->data.u8;
  float *d = dst->data.f32;
  for (; i < n; i += step) {
    __m128i b0 = _mm_loadu_si128((const __m128i_u *)&s[i]);
    __m128i b1 = _mm_loadu_si128((const __m128i_u *)&s[i + 16]);
    __m128i b2 = _mm_loadu_si128((const __m128i_u *)&s[i + 32]);
    __m128i b3 = _mm_loadu_si128((const __m128i_u *)&s[i + 48]);
    _mm512_storeu_ps(&d[i], _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(b0)));
    _mm512_storeu_ps(&d[i + 16], _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(b1)));
    _mm512_storeu_ps(&d[i + 32], _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(b2)));
    _mm512_storeu_ps(&d[i + 48], _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(b3)));
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, float)
  }
}
/**
 * @brief Cast every element from uint8_t to float using AVX2.
 *
 * Requires: AVX2
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type float.
 */
__attribute__((target("avx2"))) static inline void
tu8_to_f32_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U8_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint8_t *s = src->data.u8;
  float *d = dst->data.f32;
  for (; i < n; i += step) {
    __m256i i0 =
        _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i]));
    __m256i i1 =
        _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 8]));
    __m256i i2 =
        _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 16]));
    __m256i i3 =
        _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 24]));
    _mm256_storeu_ps(&d[i], _mm256_cvtepi32_ps(i0));
    _mm256_storeu_ps(&d[i + 8], _mm256_cvtepi32_ps(i1));
    _mm256_storeu_ps(&d[i + 16], _mm256_cvtepi32_ps(i2));
    _mm256_storeu_ps(&d[i + 24], _mm256_cvtepi32_ps(i3));
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, float)
  }
}
/**
 * @brief Cast every element from uint32_t to float using AVX-512F.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type float.
 */
__attribute__((target("avx512f"))) static inline void
tu32_to_f32_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U32_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint32_t *s = src->data.u32;
  float *d = dst->data.f32;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    __m512 r = _mm512_cvtepu32_ps(v);
    _mm512_storeu_ps(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, float)
  }
}
/**
 * @brief Cast every element from uint64_t to float using AVX-512F.
 *
 * Requires: AVX512F, AVX512DQ
 *
 * @param[in]  src  Source tensor with element type uint64_t.
 * @param[out] dst  Destination tensor with element type float.
 */
__attribute__((target("avx512f,avx512dq"))) static inline void
tu64_to_f32_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint64_t *s = src->data.u64;
  float *d = dst->data.f32;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    __m256 r = _mm512_cvtepu64_ps(v);
    _mm256_storeu_ps(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, float)
  }
}
/**
 * @brief Cast every element from int8_t to double using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type double.
 */
__attribute__((target("avx512f,avx2"))) static inline void
ts8_to_f64_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int8_t *s = src->data.s8;
  double *d = dst->data.f64;
  for (; i < n; i += step) {
    __m128i b = _mm_loadl_epi64((const __m128i_u *)&s[i]);
    __m256i i32 = _mm256_cvtepi8_epi32(b);
    __m512d r = _mm512_cvtepi32_pd(i32);
    _mm512_storeu_pd(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, double)
  }
}
/**
 * @brief Cast every element from int32_t to double using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type double.
 */
__attribute__((target("avx512f"))) static inline void
ts32_to_f64_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int32_t *s = src->data.s32;
  double *d = dst->data.f64;
  for (; i < n; i += step) {
    __m256i v = _mm256_loadu_si256((const __m256i_u *)&s[i]);
    __m512d r = _mm512_cvtepi32_pd(v);
    _mm512_storeu_pd(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, double)
  }
}
/**
 * @brief Cast every element from int32_t to double using AVX/AVX2.
 *
 * Requires: AVX/AVX2
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type double.
 */
__attribute__((target("avx,avx2"))) static inline void
ts32_to_f64_avx_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int32_t *s = src->data.s32;
  double *d = dst->data.f64;
  for (; i < n; i += step) {
    __m128i v = _mm_loadu_si128((const __m128i_u *)&s[i]);
    __m256d r = _mm256_cvtepi32_pd(v);
    _mm256_storeu_pd(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, double)
  }
}
/**
 * @brief Cast every element from int32_t to double using SSE4.2.
 *
 * Requires: SSE4.2
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type double.
 */
__attribute__((target("sse4.2"))) static inline void
ts32_to_f64_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_SSE;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int32_t *s = src->data.s32;
  double *d = dst->data.f64;
  for (; i < n; i += step) {
    __m128i v = _mm_loadu_si128((const __m128i_u *)&s[i]);
    __m128d r = _mm_cvtepi32_pd(v);
    _mm_storeu_pd(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, double)
  }
}
/**
 * @brief Cast every element from int64_t to double using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type int64_t.
 * @param[out] dst  Destination tensor with element type double.
 */
__attribute__((target("avx512f,avx512dq"))) static inline void
ts64_to_f64_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int64_t *s = src->data.s64;
  double *d = dst->data.f64;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    __m512d r = _mm512_cvtepi64_pd(v);
    _mm512_storeu_pd(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, double)
  }
}
/**
 * @brief Cast every element from uint8_t to double using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type double.
 */
__attribute__((target("avx512f,avx2"))) static inline void
tu8_to_f64_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint8_t *s = src->data.u8;
  double *d = dst->data.f64;
  for (; i < n; i += step) {
    __m128i b = _mm_loadl_epi64((const __m128i_u *)&s[i]);
    __m256i i32 = _mm256_cvtepu8_epi32(b);
    __m512d r = _mm512_cvtepi32_pd(i32);
    _mm512_storeu_pd(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, double)
  }
}
/**
 * @brief Cast every element from uint32_t to double using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type double.
 */
__attribute__((target("avx512f"))) static inline void
tu32_to_f64_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_F64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint32_t *s = src->data.u32;
  double *d = dst->data.f64;
  for (; i < n; i += step) {
    __m256i v = _mm256_loadu_si256((const __m256i_u *)&s[i]);
    __m512d r = _mm512_cvtepu32_pd(v);
    _mm512_storeu_pd(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, double)
  }
}
/**
 * @brief Cast every element from uint64_t to double using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type uint64_t.
 * @param[out] dst  Destination tensor with element type double.
 */
__attribute__((target("avx512f,avx512dq"))) static inline void
tu64_to_f64_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint64_t *s = src->data.u64;
  double *d = dst->data.f64;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    __m512d r = _mm512_cvtepu64_pd(v);
    _mm512_storeu_pd(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, double)
  }
}
/**
 * @brief Cast every element from int8_t to int32_t using AVX-512F.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
__attribute__((target("avx512f"))) static inline void
ts8_to_s32_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S8_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int8_t *s = src->data.s8;
  int32_t *d = dst->data.s32;
  for (; i < n; i += step) {
    __m128i b0 = _mm_loadu_si128((const __m128i_u *)&s[i]);
    __m128i b1 = _mm_loadu_si128((const __m128i_u *)&s[i + 16]);
    __m128i b2 = _mm_loadu_si128((const __m128i_u *)&s[i + 32]);
    __m128i b3 = _mm_loadu_si128((const __m128i_u *)&s[i + 48]);
    _mm512_storeu_si512(&d[i], _mm512_cvtepi8_epi32(b0));
    _mm512_storeu_si512(&d[i + 16], _mm512_cvtepi8_epi32(b1));
    _mm512_storeu_si512(&d[i + 32], _mm512_cvtepi8_epi32(b2));
    _mm512_storeu_si512(&d[i + 48], _mm512_cvtepi8_epi32(b3));
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int32_t)
  }
}
/**
 * @brief Cast every element from int8_t to int32_t using AVX2.
 *
 * Requires: AVX2
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
__attribute__((target("avx2"))) static inline void
ts8_to_s32_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S8_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int8_t *s = src->data.s8;
  int32_t *d = dst->data.s32;
  for (; i < n; i += step) {
    __m256i r0 =
        _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i]));
    __m256i r1 =
        _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 8]));
    __m256i r2 =
        _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 16]));
    __m256i r3 =
        _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 24]));
    _mm256_storeu_si256((__m256i_u *)&d[i], r0);
    _mm256_storeu_si256((__m256i_u *)&d[i + 8], r1);
    _mm256_storeu_si256((__m256i_u *)&d[i + 16], r2);
    _mm256_storeu_si256((__m256i_u *)&d[i + 24], r3);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int32_t)
  }
}
/**
 * @brief Cast every element from int8_t to int32_t using SSE2/SSE4.2.
 *
 * Requires: SSE2, SSE4.2
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
__attribute__((target("sse4.2"))) static inline void
ts8_to_s32_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S8_WITH_SSE;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int8_t *s = src->data.s8;
  int32_t *d = dst->data.s32;
  for (; i < n; i += step) {
    __m128i r0 = _mm_cvtepi8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i]));
    __m128i r1 =
        _mm_cvtepi8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 4]));
    __m128i r2 =
        _mm_cvtepi8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 8]));
    __m128i r3 =
        _mm_cvtepi8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 12]));
    _mm_storeu_si128((__m128i_u *)&d[i], r0);
    _mm_storeu_si128((__m128i_u *)&d[i + 4], r1);
    _mm_storeu_si128((__m128i_u *)&d[i + 8], r2);
    _mm_storeu_si128((__m128i_u *)&d[i + 12], r3);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int32_t)
  }
}
/**
 * @brief Cast every element from int8_t to int64_t using AVX-512F.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
__attribute__((target("avx512f"))) static inline void
ts8_to_s64_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int8_t *s = src->data.s8;
  int64_t *d = dst->data.s64;
  for (; i < n; i += step) {
    __m128i b = _mm_loadl_epi64((const __m128i_u *)&s[i]);
    __m512i r = _mm512_cvtepi8_epi64(b);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int64_t)
  }
}
/**
 * @brief Cast every element from int8_t to int64_t using AVX2.
 *
 * Requires: AVX2
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
__attribute__((target("avx2"))) static inline void
ts8_to_s64_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S64_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int8_t *s = src->data.s8;
  int64_t *d = dst->data.s64;
  for (; i < n; i += step) {
    __m128i b = _mm_loadl_epi64((const __m128i_u *)&s[i]);
    __m256i r = _mm256_cvtepi8_epi64(b);
    _mm256_storeu_si256((__m256i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int64_t)
  }
}
/**
 * @brief Cast every element from int8_t to int64_t using SSE2/SSE4.2.
 *
 * Requires: SSE2, SSE4.2
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
__attribute__((target("sse4.2"))) static inline void
ts8_to_s64_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S64_WITH_SSE;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int8_t *s = src->data.s8;
  int64_t *d = dst->data.s64;
  for (; i < n; i += step) {
    __m128i b = _mm_loadl_epi64((const __m128i_u *)&s[i]);
    __m128i r = _mm_cvtepi8_epi64(b);
    _mm_storeu_si128((__m128i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int64_t)
  }
}
/**
 * @brief Cast every element from int32_t to int8_t using AVX-512F and
 * AVX-512BW.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type int8_t.
 */
__attribute__((target("avx512f"))) static inline void
ts32_to_s8_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S8_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int32_t *s = src->data.s32;
  int8_t *d = dst->data.s8;
  for (; i < n; i += step) {
    __m512i v0 = _mm512_loadu_si512(&s[i]);
    __m512i v1 = _mm512_loadu_si512(&s[i + 16]);
    __m512i v2 = _mm512_loadu_si512(&s[i + 32]);
    __m512i v3 = _mm512_loadu_si512(&s[i + 48]);
    __m128i r0 = _mm512_cvtepi32_epi8(v0);
    __m128i r1 = _mm512_cvtepi32_epi8(v1);
    __m128i r2 = _mm512_cvtepi32_epi8(v2);
    __m128i r3 = _mm512_cvtepi32_epi8(v3);
    _mm_storeu_si128((__m128i_u *)&d[i], r0);
    _mm_storeu_si128((__m128i_u *)&d[i + 16], r1);
    _mm_storeu_si128((__m128i_u *)&d[i + 32], r2);
    _mm_storeu_si128((__m128i_u *)&d[i + 48], r3);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int8_t)
  }
}
/**
 * @brief Cast every element from int32_t to int64_t using AVX-512F.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
__attribute__((target("avx512f"))) static inline void
ts32_to_s64_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int32_t *s = src->data.s32;
  int64_t *d = dst->data.s64;
  for (; i < n; i += step) {
    __m256i v = _mm256_loadu_si256((const __m256i_u *)&s[i]);
    __m512i r = _mm512_cvtepi32_epi64(v);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int64_t)
  }
}
/**
 * @brief Cast every element from int32_t to int64_t using AVX2.
 *
 * Requires: AVX2
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
__attribute__((target("avx2"))) static inline void
ts32_to_s64_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S64_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int32_t *s = src->data.s32;
  int64_t *d = dst->data.s64;
  for (; i < n; i += step) {
    __m128i v = _mm_loadu_si128((const __m128i_u *)&s[i]);
    __m256i r = _mm256_cvtepi32_epi64(v);
    _mm256_storeu_si256((__m256i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int64_t)
  }
}
/**
 * @brief Cast every element from int32_t to int64_t using SSE2/SSE4.2.
 *
 * Requires: SSE2, SSE4.2
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
__attribute__((target("sse4.2"))) static inline void
ts32_to_s64_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S64_WITH_SSE;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int32_t *s = src->data.s32;
  int64_t *d = dst->data.s64;
  for (; i < n; i += step) {
    __m128i v = _mm_loadu_si128((const __m128i_u *)&s[i]);
    __m128i r = _mm_cvtepi32_epi64(v);
    _mm_storeu_si128((__m128i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int64_t)
  }
}
/**
 * @brief Cast every element from int64_t to int8_t using AVX-512.
 *
 * Requires: AVX512F, AVX512BW
 *
 * @param[in]  src  Source tensor with element type int64_t.
 * @param[out] dst  Destination tensor with element type int8_t.
 */
__attribute__((target("avx512f,avx512bw"))) static inline void
ts64_to_s8_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int64_t *s = src->data.s64;
  int8_t *d = dst->data.s8;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    __m128i r = _mm512_cvtepi64_epi8(v);
    _mm_storeu_si64(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int8_t)
  }
}
/**
 * @brief Cast every element from int64_t to int32_t using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type int64_t.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
__attribute__((target("avx512f"))) static inline void
ts64_to_s32_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int64_t *s = src->data.s64;
  int32_t *d = dst->data.s32;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    __m256i r = _mm512_cvtepi64_epi32(v);
    _mm256_storeu_si256((__m256i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int32_t)
  }
}
/**
 * @brief Cast every element from uint8_t to uint32_t using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
__attribute__((target("avx512f"))) static inline void
tu8_to_u32_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U8_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint8_t *s = src->data.u8;
  uint32_t *d = dst->data.u32;
  for (; i < n; i += step) {
    __m128i b0 = _mm_loadu_si128((const __m128i_u *)&s[i]);
    __m128i b1 = _mm_loadu_si128((const __m128i_u *)&s[i + 16]);
    __m128i b2 = _mm_loadu_si128((const __m128i_u *)&s[i + 32]);
    __m128i b3 = _mm_loadu_si128((const __m128i_u *)&s[i + 48]);
    _mm512_storeu_si512(&d[i], _mm512_cvtepu8_epi32(b0));
    _mm512_storeu_si512(&d[i + 16], _mm512_cvtepu8_epi32(b1));
    _mm512_storeu_si512(&d[i + 32], _mm512_cvtepu8_epi32(b2));
    _mm512_storeu_si512(&d[i + 48], _mm512_cvtepu8_epi32(b3));
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint32_t)
  }
}
/**
 * @brief Cast every element from uint8_t to uint32_t using AVX2.
 *
 * Requires: AVX2
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
__attribute__((target("avx2"))) static inline void
tu8_to_u32_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U8_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint8_t *s = src->data.u8;
  uint32_t *d = dst->data.u32;
  for (; i < n; i += step) {
    __m256i r0 =
        _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i]));
    __m256i r1 =
        _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 8]));
    __m256i r2 =
        _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 16]));
    __m256i r3 =
        _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 24]));
    _mm256_storeu_si256((__m256i_u *)&d[i], r0);
    _mm256_storeu_si256((__m256i_u *)&d[i + 8], r1);
    _mm256_storeu_si256((__m256i_u *)&d[i + 16], r2);
    _mm256_storeu_si256((__m256i_u *)&d[i + 24], r3);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint32_t)
  }
}
/**
 * @brief Cast every element from uint8_t to uint64_t using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
__attribute__((target("avx512f"))) static inline void
tu8_to_u64_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint8_t *s = src->data.u8;
  uint64_t *d = dst->data.u64;
  for (; i < n; i += step) {
    __m128i b = _mm_loadl_epi64((const __m128i_u *)&s[i]);
    __m512i r = _mm512_cvtepu8_epi64(b);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint64_t)
  }
}
/**
 * @brief Cast every element from uint8_t to uint64_t using AVX2.
 *
 * Requires: AVX2
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
__attribute__((target("avx2"))) static inline void
tu8_to_u64_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint8_t *s = src->data.u8;
  uint64_t *d = dst->data.u64;
  for (; i < n; i += step) {
    __m128i b = _mm_loadl_epi64((const __m128i_u *)&s[i]);
    __m256i r = _mm256_cvtepu8_epi64(b);
    _mm256_storeu_si256((__m256i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint64_t)
  }
}
/**
 * @brief Cast every element from uint8_t to uint64_t using SSE4.2.
 *
 * Requires: SSE4.2
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
__attribute__((target("sse4.2"))) static inline void
tu8_to_u64_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_SSE;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint8_t *s = src->data.u8;
  uint64_t *d = dst->data.u64;
  for (; i < n; i += step) {
    __m128i b = _mm_loadl_epi64((const __m128i_u *)&s[i]);
    __m128i r = _mm_cvtepu8_epi64(b);
    _mm_storeu_si128((__m128i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint64_t)
  }
}
/**
 * @brief Cast every element from uint32_t to uint8_t using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
__attribute__((target("avx512f"))) static inline void
tu32_to_u8_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U8_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint32_t *s = src->data.u32;
  uint8_t *d = dst->data.u8;
  for (; i < n; i += step) {
    __m512i v0 = _mm512_loadu_si512(&s[i]);
    __m512i v1 = _mm512_loadu_si512(&s[i + 16]);
    __m512i v2 = _mm512_loadu_si512(&s[i + 32]);
    __m512i v3 = _mm512_loadu_si512(&s[i + 48]);
    _mm_storeu_si128((__m128i_u *)&d[i], _mm512_cvtepi32_epi8(v0));
    _mm_storeu_si128((__m128i_u *)&d[i + 16], _mm512_cvtepi32_epi8(v1));
    _mm_storeu_si128((__m128i_u *)&d[i + 32], _mm512_cvtepi32_epi8(v2));
    _mm_storeu_si128((__m128i_u *)&d[i + 48], _mm512_cvtepi32_epi8(v3));
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint8_t)
  }
}
/**
 * @brief Cast every element from uint32_t to uint64_t using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
__attribute__((target("avx512f"))) static inline void
tu32_to_u64_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint32_t *s = src->data.u32;
  uint64_t *d = dst->data.u64;
  for (; i < n; i += step) {
    __m256i v = _mm256_loadu_si256((const __m256i_u *)&s[i]);
    __m512i r = _mm512_cvtepu32_epi64(v);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint64_t)
  }
}
/**
 * @brief Cast every element from uint32_t to uint64_t using AVX2.
 *
 * Requires: AVX2
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
__attribute__((target("avx2"))) static inline void
tu32_to_u64_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint32_t *s = src->data.u32;
  uint64_t *d = dst->data.u64;
  for (; i < n; i += step) {
    __m128i v = _mm_loadu_si128((const __m128i_u *)&s[i]);
    __m256i r = _mm256_cvtepu32_epi64(v);
    _mm256_storeu_si256((__m256i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint64_t)
  }
}
/**
 * @brief Cast every element from uint32_t to uint64_t using SSE4.2.
 *
 * Requires: SSE4.2
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
__attribute__((target("sse4.2"))) static inline void
tu32_to_u64_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_SSE;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint32_t *s = src->data.u32;
  uint64_t *d = dst->data.u64;
  for (; i < n; i += step) {
    __m128i v = _mm_loadu_si128((const __m128i_u *)&s[i]);
    __m128i r = _mm_cvtepu32_epi64(v);
    _mm_storeu_si128((__m128i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint64_t)
  }
}
/**
 * @brief Cast every element from uint64_t to uint8_t using AVX-512.
 *
 * Requires: AVX512F, AVX512BW
 *
 * @param[in]  src  Source tensor with element type uint64_t.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
__attribute__((target("avx512f,avx512bw"))) static inline void
tu64_to_u8_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint64_t *s = src->data.u64;
  uint8_t *d = dst->data.u8;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    __m128i r = _mm512_cvtepi64_epi8(v);
    _mm_storeu_si64(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint8_t)
  }
}
/**
 * @brief Cast every element from uint64_t to uint32_t using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type uint64_t.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
__attribute__((target("avx512f"))) static inline void
tu64_to_u32_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint64_t *s = src->data.u64;
  uint32_t *d = dst->data.u32;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    __m256i r = _mm512_cvtepi64_epi32(v);
    _mm256_storeu_si256((__m256i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint32_t)
  }
}
/**
 * @brief Cast every element from int8_t to uint8_t using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
__attribute__((target("avx512f"))) static inline void
ts8_to_u8_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U8_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int8_t *s = src->data.s8;
  uint8_t *d = dst->data.u8;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    _mm512_storeu_si512(&d[i], v);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint8_t)
  }
}
/**
 * @brief Cast every element from int8_t to uint8_t using AVX/AVX2.
 *
 * Requires: AVX/AVX2
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
__attribute__((target("avx,avx2"))) static inline void
ts8_to_u8_avx_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = 32;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int8_t *s = src->data.s8;
  uint8_t *d = dst->data.u8;
  for (; i < n; i += step) {
    __m256i v = _mm256_loadu_si256((const __m256i_u *)&s[i]);
    _mm256_storeu_si256((__m256i_u *)&d[i], v);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint8_t)
  }
}
/**
 * @brief Cast every element from int8_t to uint8_t using SSE4.2.
 *
 * Requires: SSE4.2
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
__attribute__((target("sse4.2"))) static inline void
ts8_to_u8_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = 16;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int8_t *s = src->data.s8;
  uint8_t *d = dst->data.u8;
  for (; i < n; i += step) {
    __m128i v = _mm_loadu_si128((const __m128i_u *)&s[i]);
    _mm_storeu_si128((__m128i_u *)&d[i], v);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint8_t)
  }
}
/**
 * @brief Cast every element from uint8_t to int8_t using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type int8_t.
 */
__attribute__((target("avx512f"))) static inline void
tu8_to_s8_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S8_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint8_t *s = src->data.u8;
  int8_t *d = dst->data.s8;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    _mm512_storeu_si512(&d[i], v);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int8_t)
  }
}
/**
 * @brief Cast every element from uint8_t to int8_t using AVX/AVX2.
 *
 * Requires: AVX/AVX2
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type int8_t.
 */
__attribute__((target("avx,avx2"))) static inline void
tu8_to_s8_avx_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = 32;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint8_t *s = src->data.u8;
  int8_t *d = dst->data.s8;
  for (; i < n; i += step) {
    __m256i v = _mm256_loadu_si256((const __m256i_u *)&s[i]);
    _mm256_storeu_si256((__m256i_u *)&d[i], v);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int8_t)
  }
}
/**
 * @brief Cast every element from uint8_t to int8_t using SSE4.2.
 *
 * Requires: SSE4.2
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type int8_t.
 */
__attribute__((target("sse4.2"))) static inline void
tu8_to_s8_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = 16;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint8_t *s = src->data.u8;
  int8_t *d = dst->data.s8;
  for (; i < n; i += step) {
    __m128i v = _mm_loadu_si128((const __m128i_u *)&s[i]);
    _mm_storeu_si128((__m128i_u *)&d[i], v);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int8_t)
  }
}
/**
 * @brief Cast every element from int8_t to uint32_t using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
__attribute__((target("avx512f"))) static inline void
ts8_to_u32_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S8_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int8_t *s = src->data.s8;
  uint32_t *d = dst->data.u32;
  for (; i < n; i += step) {
    __m128i b0 = _mm_loadu_si128((const __m128i_u *)&s[i]);
    __m128i b1 = _mm_loadu_si128((const __m128i_u *)&s[i + 16]);
    __m128i b2 = _mm_loadu_si128((const __m128i_u *)&s[i + 32]);
    __m128i b3 = _mm_loadu_si128((const __m128i_u *)&s[i + 48]);
    _mm512_storeu_si512(&d[i], _mm512_cvtepi8_epi32(b0));
    _mm512_storeu_si512(&d[i + 16], _mm512_cvtepi8_epi32(b1));
    _mm512_storeu_si512(&d[i + 32], _mm512_cvtepi8_epi32(b2));
    _mm512_storeu_si512(&d[i + 48], _mm512_cvtepi8_epi32(b3));
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint32_t)
  }
}
/**
 * @brief Cast every element from int8_t to uint32_t using AVX2.
 *
 * Requires: AVX2
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
__attribute__((target("avx2"))) static inline void
ts8_to_u32_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S8_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int8_t *s = src->data.s8;
  uint32_t *d = dst->data.u32;
  for (; i < n; i += step) {
    __m256i r0 =
        _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i]));
    __m256i r1 =
        _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 8]));
    __m256i r2 =
        _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 16]));
    __m256i r3 =
        _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 24]));
    _mm256_storeu_si256((__m256i_u *)&d[i], r0);
    _mm256_storeu_si256((__m256i_u *)&d[i + 8], r1);
    _mm256_storeu_si256((__m256i_u *)&d[i + 16], r2);
    _mm256_storeu_si256((__m256i_u *)&d[i + 24], r3);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint32_t)
  }
}
/**
 * @brief Cast every element from int8_t to uint32_t using SSE4.2.
 *
 * Requires: SSE4.2
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
__attribute__((target("sse4.2"))) static inline void
ts8_to_u32_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S8_WITH_SSE;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int8_t *s = src->data.s8;
  uint32_t *d = dst->data.u32;
  for (; i < n; i += step) {
    __m128i r0 = _mm_cvtepi8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i]));
    __m128i r1 =
        _mm_cvtepi8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 4]));
    __m128i r2 =
        _mm_cvtepi8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 8]));
    __m128i r3 =
        _mm_cvtepi8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 12]));
    _mm_storeu_si128((__m128i_u *)&d[i], r0);
    _mm_storeu_si128((__m128i_u *)&d[i + 4], r1);
    _mm_storeu_si128((__m128i_u *)&d[i + 8], r2);
    _mm_storeu_si128((__m128i_u *)&d[i + 12], r3);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint32_t)
  }
}
/**
 * @brief Cast every element from int8_t to uint64_t using AVX512F.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type int8_t.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
__attribute__((target("avx512f"))) static inline void
ts8_to_u64_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int8_t *s = src->data.s8;
  uint64_t *d = dst->data.u64;
  for (; i < n; i += step) {
    __m128i b = _mm_loadl_epi64((const __m128i_u *)&s[i]);
    __m512i r = _mm512_cvtepi8_epi64(b);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint64_t)
  }
}
/**
 * @brief Cast every element from int32_t to uint8_t using AVX512BW.
 *
 * Requires: AVX512F, AVX512BW
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
__attribute__((target("avx512f,avx512bw"))) static inline void
ts32_to_u8_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U8_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int32_t *s = src->data.s32;
  uint8_t *d = dst->data.u8;
  for (; i < n; i += step) {
    __m512i v0 = _mm512_loadu_si512(&s[i]);
    __m512i v1 = _mm512_loadu_si512(&s[i + 16]);
    __m512i v2 = _mm512_loadu_si512(&s[i + 32]);
    __m512i v3 = _mm512_loadu_si512(&s[i + 48]);
    __m512i s01 = _mm512_packs_epi32(v0, v1);
    __m512i s23 = _mm512_packs_epi32(v2, v3);
    __m512i r = _mm512_packus_epi16(s01, s23);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint8_t)
  }
}
/**
 * @brief Cast every element from int32_t to uint8_t using AVX2.
 *
 * Requires: AVX2
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
__attribute__((target("avx2"))) static inline void
ts32_to_u8_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U8_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int32_t *s = src->data.s32;
  uint8_t *d = dst->data.u8;
  for (; i < n; i += step) {
    __m256i v0 = _mm256_loadu_si256((const __m256i_u *)&s[i]);
    __m256i v1 = _mm256_loadu_si256((const __m256i_u *)&s[i + 8]);
    __m256i v2 = _mm256_loadu_si256((const __m256i_u *)&s[i + 16]);
    __m256i v3 = _mm256_loadu_si256((const __m256i_u *)&s[i + 24]);
    __m256i s01 = _mm256_packs_epi32(v0, v1);
    __m256i s23 = _mm256_packs_epi32(v2, v3);
    __m256i r = _mm256_packus_epi16(s01, s23);
    _mm256_storeu_si256((__m256i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint8_t)
  }
}
/**
 * @brief Cast every element from int32_t to uint8_t using SSE4.2.
 *
 * Requires: SSE4.2
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
__attribute__((target("sse4.2"))) static inline void
ts32_to_u8_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U8_WITH_SSE;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int32_t *s = src->data.s32;
  uint8_t *d = dst->data.u8;
  for (; i < n; i += step) {
    __m128i v0 = _mm_loadu_si128((__m128i_u *)&s[i]);
    __m128i v1 = _mm_loadu_si128((__m128i_u *)&s[i + 4]);
    __m128i v2 = _mm_loadu_si128((__m128i_u *)&s[i + 8]);
    __m128i v3 = _mm_loadu_si128((__m128i_u *)&s[i + 12]);
    __m128i s01 = _mm_packs_epi32(v0, v1);
    __m128i s23 = _mm_packs_epi32(v2, v3);
    __m128i r = _mm_packus_epi16(s01, s23);
    _mm_storeu_si128((__m128i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint8_t)
  }
}
/**
 * @brief Cast every element from int32_t to uint32_t using AVX512F.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
__attribute__((target("avx512f"))) static inline void
ts32_to_u32_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U32_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int32_t *s = src->data.s32;
  uint32_t *d = dst->data.u32;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    _mm512_storeu_si512(&d[i], v);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint32_t)
  }
}
/**
 * @brief Cast every element from int32_t to uint32_t using AVX/AVX2.
 *
 * Requires: AVX/AVX2
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
__attribute__((target("avx,avx2"))) static inline void
ts32_to_u32_avx_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = 8;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int32_t *s = src->data.s32;
  uint32_t *d = dst->data.u32;
  for (; i < n; i += step) {
    __m256i v = _mm256_loadu_si256((const __m256i_u *)&s[i]);
    _mm256_storeu_si256((__m256i_u *)&d[i], v);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint32_t)
  }
}
/**
 * @brief Cast every element from int32_t to uint32_t using SSE4.2.
 *
 * Requires: SSE4.2
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
__attribute__((target("sse4.2"))) static inline void
ts32_to_u32_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = 4;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int32_t *s = src->data.s32;
  uint32_t *d = dst->data.u32;
  for (; i < n; i += step) {
    __m128i v = _mm_loadu_si128((const __m128i_u *)&s[i]);
    _mm_storeu_si128((__m128i_u *)&d[i], v);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint32_t)
  }
}
/**
 * @brief Cast every element from int32_t to uint64_t using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
__attribute__((target("avx512f"))) static inline void
ts32_to_u64_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int32_t *s = src->data.s32;
  uint64_t *d = dst->data.u64;
  for (; i < n; i += step) {
    __m256i v = _mm256_loadu_si256((const __m256i_u *)&s[i]);
    __m512i r = _mm512_cvtepi32_epi64(v);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint64_t)
  }
}
/**
 * @brief Cast every element from int32_t to uint64_t using AVX2.
 *
 * Requires: AVX2
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
__attribute__((target("avx2"))) static inline void
ts32_to_u64_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int32_t *s = src->data.s32;
  uint64_t *d = dst->data.u64;
  for (; i < n; i += step) {
    __m128i v = _mm_loadu_si128((const __m128i_u *)&s[i]);
    __m256i r = _mm256_cvtepi32_epi64(v);
    _mm256_storeu_si256((__m256i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint64_t)
  }
}
/**
 * @brief Cast every element from int32_t to uint64_t using SSE4.2.
 *
 * Requires: SSE4.2
 *
 * @param[in]  src  Source tensor with element type int32_t.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
__attribute__((target("sse4.2"))) static inline void
ts32_to_u64_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_SSE;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int32_t *s = src->data.s32;
  uint64_t *d = dst->data.u64;
  for (; i < n; i += step) {
    __m128i v = _mm_loadu_si128((const __m128i_u *)&s[i]);
    __m128i r = _mm_cvtepi32_epi64(v);
    _mm_storeu_si128((__m128i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint64_t)
  }
}
/**
 * @brief Cast every element from int64_t to uint8_t using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type int64_t.
 * @param[out] dst  Destination tensor with element type uint8_t.
 */
__attribute__((target("avx512f"))) static inline void
ts64_to_u8_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int64_t *s = src->data.s64;
  uint8_t *d = dst->data.u8;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    __m128i r = _mm512_cvtepi64_epi8(v);
    _mm_storeu_si64(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint8_t)
  }
}
/**
 * @brief Cast every element from int64_t to uint32_t using AVX-512.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type int64_t.
 * @param[out] dst  Destination tensor with element type uint32_t.
 */
__attribute__((target("avx512f"))) static inline void
ts64_to_u32_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int64_t *s = src->data.s64;
  uint32_t *d = dst->data.u32;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    __m256i r = _mm512_cvtepi64_epi32(v);
    _mm256_storeu_si256((__m256i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint32_t)
  }
}
/**
 * @brief Cast every element from int64_t to uint64_t using AVX512F.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type int64_t.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
__attribute__((target("avx512f"))) static inline void
ts64_to_u64_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int64_t *s = src->data.s64;
  uint64_t *d = dst->data.u64;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    _mm512_storeu_si512(&d[i], v);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint64_t)
  }
}
/**
 * @brief Cast every element from int64_t to uint64_t using AVX/AVX2.
 *
 * Requires: AVX/AVX2
 *
 * @param[in]  src  Source tensor with element type int64_t.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
__attribute__((target("avx,avx2"))) static inline void
ts64_to_u64_avx_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = 4;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int64_t *s = src->data.s64;
  uint64_t *d = dst->data.u64;
  for (; i < n; i += step) {
    __m256i v = _mm256_loadu_si256((const __m256i_u *)&s[i]);
    _mm256_storeu_si256((__m256i_u *)&d[i], v);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint64_t)
  }
}
/**
 * @brief Cast every element from int64_t to uint64_t using SSE.
 *
 * Requires: SSE
 *
 * @param[in]  src  Source tensor with element type int64_t.
 * @param[out] dst  Destination tensor with element type uint64_t.
 */
__attribute__((target("sse4.2"))) static inline void
ts64_to_u64_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = 2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const int64_t *s = src->data.s64;
  uint64_t *d = dst->data.u64;
  for (; i < n; i += step) {
    __m128i v = _mm_loadu_si128((const __m128i_u *)&s[i]);
    _mm_storeu_si128((__m128i_u *)&d[i], v);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, uint64_t)
  }
}
/**
 * @brief Cast every element from uint8_t to int32_t using AVX512F.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
__attribute__((target("avx512f"))) static inline void
tu8_to_s32_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U8_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint8_t *s = src->data.u8;
  int32_t *d = dst->data.s32;
  for (; i < n; i += step) {
    __m128i b0 = _mm_loadu_si128((const __m128i_u *)&s[i]);
    __m128i b1 = _mm_loadu_si128((const __m128i_u *)&s[i + 16]);
    __m128i b2 = _mm_loadu_si128((const __m128i_u *)&s[i + 32]);
    __m128i b3 = _mm_loadu_si128((const __m128i_u *)&s[i + 48]);
    _mm512_storeu_si512(&d[i], _mm512_cvtepu8_epi32(b0));
    _mm512_storeu_si512(&d[i + 16], _mm512_cvtepu8_epi32(b1));
    _mm512_storeu_si512(&d[i + 32], _mm512_cvtepu8_epi32(b2));
    _mm512_storeu_si512(&d[i + 48], _mm512_cvtepu8_epi32(b3));
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int32_t)
  }
}
/**
 * @brief Cast every element from uint8_t to int32_t using AVX2.
 *
 * Requires: AVX2
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
__attribute__((target("avx2"))) static inline void
tu8_to_s32_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U8_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint8_t *s = src->data.u8;
  int32_t *d = dst->data.s32;
  for (; i < n; i += step) {
    __m256i r0 =
        _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i]));
    __m256i r1 =
        _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 8]));
    __m256i r2 =
        _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 16]));
    __m256i r3 =
        _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 24]));
    _mm256_storeu_si256((__m256i_u *)&d[i], r0);
    _mm256_storeu_si256((__m256i_u *)&d[i + 8], r1);
    _mm256_storeu_si256((__m256i_u *)&d[i + 16], r2);
    _mm256_storeu_si256((__m256i_u *)&d[i + 24], r3);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int32_t)
  }
}
/**
 * @brief Cast every element from uint8_t to int32_t using SSE2/SSE4.2.
 *
 * Requires: SSE2/SSE4.2
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
__attribute__((target("sse4.2"))) static inline void
tu8_to_s32_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U8_WITH_SSE;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint8_t *s = src->data.u8;
  int32_t *d = dst->data.s32;
  for (; i < n; i += step) {
    __m128i r0 = _mm_cvtepu8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i]));
    __m128i r1 =
        _mm_cvtepu8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 4]));
    __m128i r2 =
        _mm_cvtepu8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 8]));
    __m128i r3 =
        _mm_cvtepu8_epi32(_mm_loadl_epi64((const __m128i_u *)&s[i + 12]));
    _mm_storeu_si128((__m128i_u *)&d[i], r0);
    _mm_storeu_si128((__m128i_u *)&d[i + 4], r1);
    _mm_storeu_si128((__m128i_u *)&d[i + 8], r2);
    _mm_storeu_si128((__m128i_u *)&d[i + 12], r3);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int32_t)
  }
}
/**
 * @brief Cast every element from uint8_t to int64_t using AVX512F.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
__attribute__((target("avx512f"))) static inline void
tu8_to_s64_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint8_t *s = src->data.u8;
  int64_t *d = dst->data.s64;
  for (; i < n; i += step) {
    __m128i b = _mm_loadl_epi64((const __m128i_u *)&s[i]);
    __m512i r = _mm512_cvtepu8_epi64(b);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int64_t)
  }
}
/**
 * @brief Cast every element from uint8_t to int64_t using AVX2.
 *
 * Requires: AVX2
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
__attribute__((target("avx2"))) static inline void
tu8_to_s64_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S64_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint8_t *s = src->data.u8;
  int64_t *d = dst->data.s64;
  for (; i < n; i += step) {
    __m128i b = _mm_loadl_epi64((const __m128i_u *)&s[i]);
    __m256i r = _mm256_cvtepu8_epi64(b);
    _mm256_storeu_si256((__m256i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int64_t)
  }
}
/**
 * @brief Cast every element from uint8_t to int64_t using SSE2/SSE4.2.
 *
 * Requires: SSE2/SSE4.2
 *
 * @param[in]  src  Source tensor with element type uint8_t.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
__attribute__((target("sse4.2"))) static inline void
tu8_to_s64_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S64_WITH_SSE;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint8_t *s = src->data.u8;
  int64_t *d = dst->data.s64;
  for (; i < n; i += step) {
    __m128i b = _mm_loadl_epi64((const __m128i_u *)&s[i]);
    __m128i r = _mm_cvtepu8_epi64(b);
    _mm_storeu_si128((__m128i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int64_t)
  }
}
/**
 * @brief Cast every element from uint32_t to int8_t using AVX512F/AVX512BW.
 *
 * Requires: AVX512F, AVX512BW
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type int8_t.
 */
__attribute__((target("avx512f,avx512bw"))) static inline void
tu32_to_s8_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U8_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint32_t *s = src->data.u32;
  int8_t *d = dst->data.s8;
  for (; i < n; i += step) {
    __m512i v0 = _mm512_loadu_si512(&s[i]);
    __m512i v1 = _mm512_loadu_si512(&s[i + 16]);
    __m512i v2 = _mm512_loadu_si512(&s[i + 32]);
    __m512i v3 = _mm512_loadu_si512(&s[i + 48]);
    __m512i s01 = _mm512_packs_epi32(v0, v1);
    __m512i s23 = _mm512_packs_epi32(v2, v3);
    __m512i r = _mm512_packs_epi16(s01, s23);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int8_t)
  }
}
/**
 * @brief Cast every element from uint32_t to int32_t using AVX512F.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
__attribute__((target("avx512f"))) static inline void
tu32_to_s32_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S32_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint32_t *s = src->data.u32;
  int32_t *d = dst->data.s32;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    _mm512_storeu_si512(&d[i], v);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int32_t)
  }
}
/**
 * @brief Cast every element from uint32_t to int32_t using AVX/AVX2.
 *
 * Requires: AVX/AVX2
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
__attribute__((target("avx,avx2"))) static inline void
tu32_to_s32_avx_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = 8;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint32_t *s = src->data.u32;
  int32_t *d = dst->data.s32;
  for (; i < n; i += step) {
    __m256i v = _mm256_loadu_si256((const __m256i_u *)&s[i]);
    _mm256_storeu_si256((__m256i_u *)&d[i], v);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int32_t)
  }
}
/**
 * @brief Cast every element from uint32_t to int32_t using SSE4.2.
 *
 * Requires: SSE4.2
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
__attribute__((target("sse4.2"))) static inline void
tu32_to_s32_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = 4;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint32_t *s = src->data.u32;
  int32_t *d = dst->data.s32;
  for (; i < n; i += step) {
    __m128i v = _mm_loadu_si128((const __m128i_u *)&s[i]);
    _mm_storeu_si128((__m128i_u *)&d[i], v);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int32_t)
  }
}
/**
 * @brief Cast every element from uint32_t to int64_t using AVX512F.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
__attribute__((target("avx512f"))) static inline void
tu32_to_s64_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint32_t *s = src->data.u32;
  int64_t *d = dst->data.s64;
  for (; i < n; i += step) {
    __m256i v = _mm256_loadu_si256((const __m256i_u *)&s[i]);
    __m512i r = _mm512_cvtepu32_epi64(v);
    _mm512_storeu_si512(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int64_t)
  }
}
/**
 * @brief Cast every element from uint32_t to int64_t using AVX2.
 *
 * Requires: AVX2
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
__attribute__((target("avx2"))) static inline void
tu32_to_s64_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S64_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint32_t *s = src->data.u32;
  int64_t *d = dst->data.s64;
  for (; i < n; i += step) {
    __m128i v = _mm_loadu_si128((const __m128i_u *)&s[i]);
    __m256i r = _mm256_cvtepu32_epi64(v);
    _mm256_storeu_si256((__m256i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int64_t)
  }
}
/**
 * @brief Cast every element from uint32_t to int64_t using SSE2/SSE4.2.
 *
 * Requires: SSE2/SSE4.2
 *
 * @param[in]  src  Source tensor with element type uint32_t.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
__attribute__((target("sse4.2"))) static inline void
tu32_to_s64_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S64_WITH_SSE;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint32_t *s = src->data.u32;
  int64_t *d = dst->data.s64;
  for (; i < n; i += step) {
    __m128i v = _mm_loadu_si128((const __m128i_u *)&s[i]);
    __m128i r = _mm_cvtepu32_epi64(v);
    _mm_storeu_si128((__m128i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int64_t)
  }
}
/**
 * @brief Cast every element from uint64_t to int8_t using AVX512F.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type uint64_t.
 * @param[out] dst  Destination tensor with element type int8_t.
 */
__attribute__((target("avx512f"))) static inline void
tu64_to_s8_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint64_t *s = src->data.u64;
  int8_t *d = dst->data.s8;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    __m128i r = _mm512_cvtepi64_epi8(v);
    _mm_storeu_si64(&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int8_t)
  }
}
/**
 * @brief Cast every element from uint64_t to int32_t using AVX512F.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type uint64_t.
 * @param[out] dst  Destination tensor with element type int32_t.
 */
__attribute__((target("avx512f"))) static inline void
tu64_to_s32_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_U64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint64_t *s = src->data.u64;
  int32_t *d = dst->data.s32;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    __m256i r = _mm512_cvtepi64_epi32(v);
    _mm256_storeu_si256((__m256i_u *)&d[i], r);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int32_t)
  }
}
/**
 * @brief Cast every element from uint64_t to int64_t using AVX512F.
 *
 * Requires: AVX512F
 *
 * @param[in]  src  Source tensor with element type uint64_t.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
__attribute__((target("avx512f"))) static inline void
tu64_to_s64_avx512(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = NOVA_SIMD_S64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint64_t *s = src->data.u64;
  int64_t *d = dst->data.s64;
  for (; i < n; i += step) {
    __m512i v = _mm512_loadu_si512(&s[i]);
    _mm512_storeu_si512(&d[i], v);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int64_t)
  }
}
/**
 * @brief Cast every element from uint64_t to int64_t using AVX/AVX2.
 *
 * Requires: AVX/AVX2
 *
 * @param[in]  src  Source tensor with element type uint64_t.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
__attribute__((target("avx,avx2"))) static inline void
tu64_to_s64_avx_avx2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = 4;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint64_t *s = src->data.u64;
  int64_t *d = dst->data.s64;
  for (; i < n; i += step) {
    __m256i v = _mm256_loadu_si256((const __m256i_u *)&s[i]);
    _mm256_storeu_si256((__m256i_u *)&d[i], v);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int64_t)
  }
}

/**
 * @brief Cast every element from uint64_t to int64_t using SSE.
 *
 * Requires: SSE4.2
 *
 * @param[in]  src  Source tensor with element type uint64_t.
 * @param[out] dst  Destination tensor with element type int64_t.
 */
__attribute__((target("sse4.2"))) static inline void
tu64_to_s64_sse4_2(const Tensor *restrict src, Tensor *restrict dst) {
  const size_t step = 2;
  size_t i = 0;
  size_t rem = src->size % step;
  size_t n = src->size - rem;
  size_t size = src->size;
  const uint64_t *s = src->data.u64;
  int64_t *d = dst->data.s64;
  for (; i < n; i += step) {
    __m128i v = _mm_loadu_si128((const __m128i_u *)&s[i]);
    _mm_storeu_si128((__m128i_u *)&d[i], v);
  }
  if (rem > 0) {
    REMAINING(i, size, d, s, int64_t)
  }
}

/* =========================================================================
 * Dispatch tables — lists implementations in descending capability order
 * ========================================================================= */

/**
 * @brief Dispatch table for fp16 to f32 conversions.
 *
 * Variants:
 * - Index 0: tfp16_to_f32_avx512() — Requires: AVX512F
 * - Index 1: tfp16_to_f32_avx_avx2_fp16c() — Requires: F16C
 * - Index 2: tfp16_to_f32_scalar() — Portable fallback
 */
const CastFn lookup_tfp16_to_f32[] = {
    tfp16_to_f32_avx512, tfp16_to_f32_avx_avx2_fp16c, tfp16_to_f32_scalar};

/**
 * @brief Dispatch table for fp16 to f64 conversions.
 *
 * Variants:
 * - Index 0: tfp16_to_f64_avx512() — Requires: AVX512F
 * - Index 1: tfp16_to_f64_avx_avx2_fp16c() — Requires: F16C
 * - Index 2: tfp16_to_f64_scalar() — Portable fallback
 */
const CastFn lookup_tfp16_to_f64[] = {
    tfp16_to_f64_avx512, tfp16_to_f64_avx_avx2_fp16c, tfp16_to_f64_scalar};

/**
 * @brief Dispatch table for fp16 to bf16 conversions.
 *
 * Variants:
 * - Index 0: tfp16_to_bf16_avx512bf16() — Requires: AVX512F, AVX512BF16
 * - Index 1: tfp16_to_bf16_scalar() — Portable fallback
 */
const CastFn lookup_tfp16_to_bf16[] = {tfp16_to_bf16_avx512bf16,
                                       tfp16_to_bf16_scalar};

/**
 * @brief Dispatch table for f32 to fp16 conversions.
 *
 * Variants:
 * - Index 0: tf32_to_fp16_avx512fp16() — Requires: AVX512F, AVX512FP16
 * - Index 1: tf32_to_fp16_avx_avx2_f16c() — Requires: F16C
 * - Index 2: tf32_to_fp16_scalar() — Portable fallback
 */
const CastFn lookup_tf32_to_fp16[] = {
    tf32_to_fp16_avx512fp16, tf32_to_fp16_avx_avx2_f16c, tf32_to_fp16_scalar};

/**
 * @brief Dispatch table for f32 to f64 conversions.
 *
 * Variants:
 * - Index 0: tf32_to_f64_avx512() — Requires: AVX512F
 * - Index 1: tf32_to_f64_avx_avx2() — Requires: AVX/AVX2
 * - Index 2: tf32_to_f64_sse4_2() — Requires: SSE4.2
 * - Index 3: tf32_to_f64_scalar() — Portable fallback
 */
const CastFn lookup_tf32_to_f64[] = {tf32_to_f64_avx512, tf32_to_f64_avx_avx2,
                                     tf32_to_f64_sse4_2, tf32_to_f64_scalar};

/**
 * @brief Dispatch table for f32 to bf16 conversions.
 *
 * Variants:
 * - Index 0: tf32_to_bf16_avx512bf16() — Requires: AVX512F, AVX512BF16
 * - Index 1: tf32_to_bf16_scalar() — Portable fallback
 */
const CastFn lookup_tf32_to_bf16[] = {tf32_to_bf16_avx512bf16,
                                      tf32_to_bf16_scalar};

/**
 * @brief Dispatch table for bf16 to fp16 conversions.
 *
 * Variants:
 * - Index 0: tbf16_to_fp16_avx512bf16_fp16() — Requires: AVX512F, AVX512BF16,
 * AVX512FP16
 * - Index 1: tbf16_to_fp16_scalar() — Portable fallback
 */
const CastFn lookup_tbf16_to_fp16[] = {tbf16_to_fp16_avx512bf16_fp16,
                                       tbf16_to_fp16_scalar};

/**
 * @brief Dispatch table for bf16 to f32 conversions.
 *
 * Variants:
 * - Index 0: tbf16_to_f32_avx512bf16() — Requires: AVX512F, AVX512BF16
 * - Index 1: tbf16_to_f32_scalar() — Portable fallback
 */
const CastFn lookup_tbf16_to_f32[] = {tbf16_to_f32_avx512bf16,
                                      tbf16_to_f32_scalar};

/**
 * @brief Dispatch table for bf16 to f64 conversions.
 *
 * Variants:
 * - Index 0: tbf16_to_f64_avx512bf16() — Requires: AVX512F, AVX512BF16,
 * AVX512VL
 * - Index 1: tbf16_to_f64_scalar() — Portable fallback
 */
const CastFn lookup_tbf16_to_f64[] = {tbf16_to_f64_avx512bf16,
                                      tbf16_to_f64_scalar};

/**
 * @brief Dispatch table for f64 to fp16 conversions.
 *
 * Variants:
 * - Index 0: tf64_to_fp16_avx512fp16() — Requires: AVX512F, AVX512FP16
 * - Index 1: tf64_to_fp16_scalar() — Portable fallback
 */
const CastFn lookup_tf64_to_fp16[] = {tf64_to_fp16_avx512fp16,
                                      tf64_to_fp16_scalar};

/**
 * @brief Dispatch table for f64 to f32 conversions.
 *
 * Variants:
 * - Index 0: tf64_to_f32_avx512() — Requires: AVX512F
 * - Index 1: tf64_to_f32_avx_avx2() — Requires: AVX/AVX2
 * - Index 2: tf64_to_f32_sse4_2() — Requires: SSE4.2
 * - Index 3: tf64_to_f32_scalar() — Portable fallback
 */
const CastFn lookup_tf64_to_f32[] = {tf64_to_f32_avx512, tf64_to_f32_avx_avx2,
                                     tf64_to_f32_sse4_2, tf64_to_f32_scalar};

/**
 * @brief Dispatch table for f64 to bf16 conversions.
 *
 * Variants:
 * - Index 0: tf64_to_bf16_avx512bf16() — Requires: AVX512F, AVX512BF16
 * - Index 1: tf64_to_bf16_scalar() — Portable fallback
 */
const CastFn lookup_tf64_to_bf16[] = {tf64_to_bf16_avx512bf16,
                                      tf64_to_bf16_scalar};

/**
 * @brief Dispatch table for fp16 to s8 conversions.
 *
 * Variants:
 * - Index 0: tfp16_to_s8_avx512fp16() — Requires: AVX512FP16, AVX512BW,
 * AVX512F
 * - Index 1: tfp16_to_s8_scalar() — Portable fallback
 */
const CastFn lookup_tfp16_to_s8[] = {tfp16_to_s8_avx512fp16,
                                     tfp16_to_s8_scalar};

/**
 * @brief Dispatch table for fp16 to s32 conversions.
 *
 * Variants:
 * - Index 0: tfp16_to_s32_avx512fp16() — Requires: AVX512F, AVX512FP16
 * - Index 1: tfp16_to_s32_scalar() — Portable fallback
 */
const CastFn lookup_tfp16_to_s32[] = {tfp16_to_s32_avx512fp16,
                                      tfp16_to_s32_scalar};

/**
 * @brief Dispatch table for fp16 to s64 conversions.
 *
 * Variants:
 * - Index 0: tfp16_to_s64_avx512fp16() — Requires: AVX512F, AVX512FP16
 * - Index 1: tfp16_to_s64_scalar() — Portable fallback
 */
const CastFn lookup_tfp16_to_s64[] = {tfp16_to_s64_avx512fp16,
                                      tfp16_to_s64_scalar};

/**
 * @brief Dispatch table for fp16 to u8 conversions.
 *
 * Variants:
 * - Index 0: tfp16_to_u8_avx512fp16() — Requires: AVX512FP16, AVX512BW,
 * AVX512F
 * - Index 1: tfp16_to_u8_scalar() — Portable fallback
 */
const CastFn lookup_tfp16_to_u8[] = {tfp16_to_u8_avx512fp16,
                                     tfp16_to_u8_scalar};

/**
 * @brief Dispatch table for fp16 to u32 conversions.
 *
 * Variants:
 * - Index 0: tfp16_to_u32_avx512fp16() — Requires: AVX512F, AVX512FP16
 * - Index 1: tfp16_to_u32_scalar() — Portable fallback
 */
const CastFn lookup_tfp16_to_u32[] = {tfp16_to_u32_avx512fp16,
                                      tfp16_to_u32_scalar};

/**
 * @brief Dispatch table for fp16 to u64 conversions.
 *
 * Variants:
 * - Index 0: tfp16_to_u64_avx512fp16() — Requires: AVX512F, AVX512FP16
 * - Index 1: tfp16_to_u64_scalar() — Portable fallback
 */
const CastFn lookup_tfp16_to_u64[] = {tfp16_to_u64_avx512fp16,
                                      tfp16_to_u64_scalar};

/**
 * @brief Dispatch table for f32 to s8 conversions.
 *
 * Variants:
 * - Index 0: tf32_to_s8_avx512() — Requires: AVX512F, AVX512BW
 * - Index 1: tf32_to_s8_avx2() — Requires: AVX2
 * - Index 2: tf32_to_s8_scalar() — Portable fallback
 */
const CastFn lookup_tf32_to_s8[] = {tf32_to_s8_avx512, tf32_to_s8_avx2,
                                    tf32_to_s8_scalar};

/**
 * @brief Dispatch table for f32 to s32 conversions.
 *
 * Variants:
 * - Index 0: tf32_to_s32_avx512() — Requires: AVX512F
 * - Index 1: tf32_to_s32_avx_avx2() — Requires: AVX/AVX2
 * - Index 2: tf32_to_s32_sse4_2() — Requires: SSE4.2
 * - Index 3: tf32_to_s32_scalar() — Portable fallback
 */
const CastFn lookup_tf32_to_s32[] = {tf32_to_s32_avx512, tf32_to_s32_avx_avx2,
                                     tf32_to_s32_sse4_2, tf32_to_s32_scalar};

/**
 * @brief Dispatch table for f32 to s64 conversions.
 *
 * Variants:
 * - Index 0: tf32_to_s64_avx512() — Requires: AVX512F, AVX512DQ
 * - Index 1: tf32_to_s64_scalar() — Portable fallback
 */
const CastFn lookup_tf32_to_s64[] = {tf32_to_s64_avx512, tf32_to_s64_scalar};

/**
 * @brief Dispatch table for f32 to u8 conversions.
 *
 * Variants:
 * - Index 0: tf32_to_u8_avx512() — Requires: AVX512F, AVX512BW
 * - Index 1: tf32_to_u8_avx2() — Requires: AVX2
 * - Index 2: tf32_to_u8_scalar() — Portable fallback
 */
const CastFn lookup_tf32_to_u8[] = {tf32_to_u8_avx512, tf32_to_u8_avx2,
                                    tf32_to_u8_scalar};

/**
 * @brief Dispatch table for f32 to u32 conversions.
 *
 * Variants:
 * - Index 0: tf32_to_u32_avx512() — Requires: AVX512F
 * - Index 1: tf32_to_u32_scalar() — Portable fallback
 */
const CastFn lookup_tf32_to_u32[] = {tf32_to_u32_avx512, tf32_to_u32_scalar};

/**
 * @brief Dispatch table for f32 to u64 conversions.
 *
 * Variants:
 * - Index 0: tf32_to_u64_avx512() — Requires: AVX512F, AVX512DQ
 * - Index 1: tf32_to_u64_scalar() — Portable fallback
 */
const CastFn lookup_tf32_to_u64[] = {tf32_to_u64_avx512, tf32_to_u64_scalar};

/**
 * @brief Dispatch table for bf16 to s8 conversions.
 *
 * Variants:
 * - Index 0: tbf16_to_s8_avx512bf16() — Requires: AVX512BF16, AVX512BW,
 * AVX512F
 * - Index 1: tbf16_to_s8_scalar() — Portable fallback
 */
const CastFn lookup_tbf16_to_s8[] = {tbf16_to_s8_avx512bf16,
                                     tbf16_to_s8_scalar};

/**
 * @brief Dispatch table for bf16 to s32 conversions.
 *
 * Variants:
 * - Index 0: tbf16_to_s32_avx512bf16() — Requires: AVX512F, AVX512BF16
 * - Index 1: tbf16_to_s32_scalar() — Portable fallback
 */
const CastFn lookup_tbf16_to_s32[] = {tbf16_to_s32_avx512bf16,
                                      tbf16_to_s32_scalar};

/**
 * @brief Dispatch table for bf16 to s64 conversions.
 *
 * Variants:
 * - Index 0: tbf16_to_s64_avx512bf16() — Requires: AVX512F, AVX512BF16,
 * AVX512VL, AVX512DQ
 * - Index 1: tbf16_to_s64_scalar() — Portable fallback
 */
const CastFn lookup_tbf16_to_s64[] = {tbf16_to_s64_avx512bf16,
                                      tbf16_to_s64_scalar};

/**
 * @brief Dispatch table for bf16 to u8 conversions.
 *
 * Variants:
 * - Index 0: tbf16_to_u8_avx512bf16() — Requires: AVX512BF16, AVX512BW,
 * AVX512F
 * - Index 1: tbf16_to_u8_scalar() — Portable fallback
 */
const CastFn lookup_tbf16_to_u8[] = {tbf16_to_u8_avx512bf16,
                                     tbf16_to_u8_scalar};

/**
 * @brief Dispatch table for bf16 to u32 conversions.
 *
 * Variants:
 * - Index 0: tbf16_to_u32_avx512bf16() — Requires: AVX512F, AVX512BF16
 * - Index 1: tbf16_to_u32_scalar() — Portable fallback
 */
const CastFn lookup_tbf16_to_u32[] = {tbf16_to_u32_avx512bf16,
                                      tbf16_to_u32_scalar};

/**
 * @brief Dispatch table for bf16 to u64 conversions.
 *
 * Variants:
 * - Index 0: tbf16_to_u64_avx512bf16() — Requires: AVX512F, AVX512BF16,
 * AVX512VL, AVX512DQ
 * - Index 1: tbf16_to_u64_scalar() — Portable fallback
 */
const CastFn lookup_tbf16_to_u64[] = {tbf16_to_u64_avx512bf16,
                                      tbf16_to_u64_scalar};

/**
 * @brief Dispatch table for f64 to s8 conversions.
 *
 * Variants:
 * - Index 0: tf64_to_s8_avx512() — Requires: AVX512F, AVX2
 * - Index 1: tf64_to_s8_scalar() — Portable fallback
 */
const CastFn lookup_tf64_to_s8[] = {tf64_to_s8_avx512, tf64_to_s8_scalar};

/**
 * @brief Dispatch table for f64 to s32 conversions.
 *
 * Variants:
 * - Index 0: tf64_to_s32_avx512() — Requires: AVX512F
 * - Index 1: tf64_to_s32_avx_avx2() — Requires: AVX/AVX2
 * - Index 2: tf64_to_s32_sse4_2() — Requires: SSE4.2
 * - Index 3: tf64_to_s32_scalar() — Portable fallback
 */
const CastFn lookup_tf64_to_s32[] = {tf64_to_s32_avx512, tf64_to_s32_avx_avx2,
                                     tf64_to_s32_sse4_2, tf64_to_s32_scalar};

/**
 * @brief Dispatch table for f64 to s64 conversions.
 *
 * Variants:
 * - Index 0: tf64_to_s64_avx512() — Requires: AVX512F
 * - Index 1: tf64_to_s64_scalar() — Portable fallback
 */
const CastFn lookup_tf64_to_s64[] = {tf64_to_s64_avx512, tf64_to_s64_scalar};

/**
 * @brief Dispatch table for f64 to u8 conversions.
 *
 * Variants:
 * - Index 0: tf64_to_u8_avx512() — Requires: AVX512F, AVX2
 * - Index 1: tf64_to_u8_scalar() — Portable fallback
 */
const CastFn lookup_tf64_to_u8[] = {tf64_to_u8_avx512, tf64_to_u8_scalar};

/**
 * @brief Dispatch table for f64 to u32 conversions.
 *
 * Variants:
 * - Index 0: tf64_to_u32_avx512() — Requires: AVX512F
 * - Index 1: tf64_to_u32_scalar() — Portable fallback
 */
const CastFn lookup_tf64_to_u32[] = {tf64_to_u32_avx512, tf64_to_u32_scalar};

/**
 * @brief Dispatch table for f64 to u64 conversions.
 *
 * Variants:
 * - Index 0: tf64_to_u64_avx512() — Requires: AVX512F, AVX512DQ
 * - Index 1: tf64_to_u64_scalar() — Portable fallback
 */
const CastFn lookup_tf64_to_u64[] = {tf64_to_u64_avx512, tf64_to_u64_scalar};

/**
 * @brief Dispatch table for s8 to fp16 conversions.
 *
 * Variants:
 * - Index 0: ts8_to_fp16_avx512() — Requires: AVX512F, AVX512FP16
 * - Index 1: ts8_to_fp16_scalar() — Portable fallback
 */
const CastFn lookup_ts8_to_fp16[] = {ts8_to_fp16_avx512, ts8_to_fp16_scalar};

/**
 * @brief Dispatch table for s32 to fp16 conversions.
 *
 * Variants:
 * - Index 0: ts32_to_fp16_avx512fp16() — Requires: AVX512F, AVX512FP16
 * - Index 1: ts32_to_fp16_scalar() — Portable fallback
 */
const CastFn lookup_ts32_to_fp16[] = {ts32_to_fp16_avx512fp16,
                                      ts32_to_fp16_scalar};

/**
 * @brief Dispatch table for s64 to fp16 conversions.
 *
 * Variants:
 * - Index 0: ts64_to_fp16_avx512fp16() — Requires: AVX512F, AVX512FP16
 * - Index 1: ts64_to_fp16_scalar() — Portable fallback
 */
const CastFn lookup_ts64_to_fp16[] = {ts64_to_fp16_avx512fp16,
                                      ts64_to_fp16_scalar};

/**
 * @brief Dispatch table for u8 to fp16 conversions.
 *
 * Variants:
 * - Index 0: tu8_to_fp16_avx512fp16() — Requires: AVX512F, AVX512FP16
 * - Index 1: tu8_to_fp16_scalar() — Portable fallback
 */
const CastFn lookup_tu8_to_fp16[] = {tu8_to_fp16_avx512fp16,
                                     tu8_to_fp16_scalar};

/**
 * @brief Dispatch table for u32 to fp16 conversions.
 *
 * Variants:
 * - Index 0: tu32_to_fp16_avx512fp16() — Requires: AVX512F, AVX512FP16
 * - Index 1: tu32_to_fp16_scalar() — Portable fallback
 */
const CastFn lookup_tu32_to_fp16[] = {tu32_to_fp16_avx512fp16,
                                      tu32_to_fp16_scalar};

/**
 * @brief Dispatch table for u64 to fp16 conversions.
 *
 * Variants:
 * - Index 0: tu64_to_fp16_avx512fp16() — Requires: AVX512F, AVX512FP16
 * - Index 1: tu64_to_fp16_scalar() — Portable fallback
 */
const CastFn lookup_tu64_to_fp16[] = {tu64_to_fp16_avx512fp16,
                                      tu64_to_fp16_scalar};

/**
 * @brief Dispatch table for s8 to bf16 conversions.
 *
 * Variants:
 * - Index 0: ts8_to_bf16_avx512bf16() — Requires: AVX512F, AVX512BF16
 * - Index 1: ts8_to_bf16_scalar() — Portable fallback
 */
const CastFn lookup_ts8_to_bf16[] = {ts8_to_bf16_avx512bf16,
                                     ts8_to_bf16_scalar};

/**
 * @brief Dispatch table for s32 to bf16 conversions.
 *
 * Variants:
 * - Index 0: ts32_to_bf16_avx512bf16() — Requires: AVX512F, AVX512BF16
 * - Index 1: ts32_to_bf16_scalar() — Portable fallback
 */
const CastFn lookup_ts32_to_bf16[] = {ts32_to_bf16_avx512bf16,
                                      ts32_to_bf16_scalar};

/**
 * @brief Dispatch table for s64 to bf16 conversions.
 *
 * Variants:
 * - Index 0: ts64_to_bf16_avx512bf16() — Requires: AVX512F, AVX512DQ,
 * AVX512BF16, AVX512VL
 * - Index 1: ts64_to_bf16_scalar() — Portable fallback
 */
const CastFn lookup_ts64_to_bf16[] = {ts64_to_bf16_avx512bf16,
                                      ts64_to_bf16_scalar};

/**
 * @brief Dispatch table for u8 to bf16 conversions.
 *
 * Variants:
 * - Index 0: tu8_to_bf16_avx512bf16() — Requires: AVX512F, AVX512BF16
 * - Index 1: tu8_to_bf16_scalar() — Portable fallback
 */
const CastFn lookup_tu8_to_bf16[] = {tu8_to_bf16_avx512bf16,
                                     tu8_to_bf16_scalar};

/**
 * @brief Dispatch table for u32 to bf16 conversions.
 *
 * Variants:
 * - Index 0: tu32_to_bf16_avx512bf16() — Requires: AVX512F, AVX512BF16
 * - Index 1: tu32_to_bf16_scalar() — Portable fallback
 */
const CastFn lookup_tu32_to_bf16[] = {tu32_to_bf16_avx512bf16,
                                      tu32_to_bf16_scalar};

/**
 * @brief Dispatch table for u64 to bf16 conversions.
 *
 * Variants:
 * - Index 0: tu64_to_bf16_avx512bf16() — Requires: AVX512F, AVX512DQ,
 * AVX512BF16, AVX512VL
 * - Index 1: tu64_to_bf16_scalar() — Portable fallback
 */
const CastFn lookup_tu64_to_bf16[] = {tu64_to_bf16_avx512bf16,
                                      tu64_to_bf16_scalar};

/**
 * @brief Dispatch table for s8 to f32 conversions.
 *
 * Variants:
 * - Index 0: ts8_to_f32_avx512() — Requires: AVX512F
 * - Index 1: ts8_to_f32_avx2() — Requires: AVX2
 * - Index 2: ts8_to_f32_scalar() — Portable fallback
 */
const CastFn lookup_ts8_to_f32[] = {ts8_to_f32_avx512, ts8_to_f32_avx2,
                                    ts8_to_f32_scalar};

/**
 * @brief Dispatch table for s32 to f32 conversions.
 *
 * Variants:
 * - Index 0: ts32_to_f32_avx512() — Requires: AVX512F
 * - Index 1: ts32_to_f32_avx_avx2() — Requires: AVX/AVX2
 * - Index 2: ts32_to_f32_sse4_2() — Requires: SSE4.2
 * - Index 3: ts32_to_f32_scalar() — Portable fallback
 */
const CastFn lookup_ts32_to_f32[] = {ts32_to_f32_avx512, ts32_to_f32_avx_avx2,
                                     ts32_to_f32_sse4_2, ts32_to_f32_scalar};

/**
 * @brief Dispatch table for s64 to f32 conversions.
 *
 * Variants:
 * - Index 0: ts64_to_f32_avx512() — Requires: AVX512F, AVX512DQ
 * - Index 1: ts64_to_f32_scalar() — Portable fallback
 */
const CastFn lookup_ts64_to_f32[] = {ts64_to_f32_avx512, ts64_to_f32_scalar};

/**
 * @brief Dispatch table for u8 to f32 conversions.
 *
 * Variants:
 * - Index 0: tu8_to_f32_avx512() — Requires: AVX512F
 * - Index 1: tu8_to_f32_avx2() — Requires: AVX2
 * - Index 2: tu8_to_f32_scalar() — Portable fallback
 */
const CastFn lookup_tu8_to_f32[] = {tu8_to_f32_avx512, tu8_to_f32_avx2,
                                    tu8_to_f32_scalar};

/**
 * @brief Dispatch table for u32 to f32 conversions.
 *
 * Variants:
 * - Index 0: tu32_to_f32_avx512() — Requires: AVX512F
 * - Index 1: tu32_to_f32_scalar() — Portable fallback
 */
const CastFn lookup_tu32_to_f32[] = {tu32_to_f32_avx512, tu32_to_f32_scalar};

/**
 * @brief Dispatch table for u64 to f32 conversions.
 *
 * Variants:
 * - Index 0: tu64_to_f32_avx512() — Requires: AVX512F, AVX512DQ
 * - Index 1: tu64_to_f32_scalar() — Portable fallback
 */
const CastFn lookup_tu64_to_f32[] = {tu64_to_f32_avx512, tu64_to_f32_scalar};

/**
 * @brief Dispatch table for s8 to f64 conversions.
 *
 * Variants:
 * - Index 0: ts8_to_f64_avx512() — Requires: AVX512F, AVX2
 * - Index 1: ts8_to_f64_scalar() — Portable fallback
 */
const CastFn lookup_ts8_to_f64[] = {ts8_to_f64_avx512, ts8_to_f64_scalar};

/**
 * @brief Dispatch table for s32 to f64 conversions.
 *
 * Variants:
 * - Index 0: ts32_to_f64_avx512() — Requires: AVX512F
 * - Index 1: ts32_to_f64_avx_avx2() — Requires: AVX/AVX2
 * - Index 2: ts32_to_f64_sse4_2() — Requires: SSE4.2
 * - Index 3: ts32_to_f64_scalar() — Portable fallback
 */
const CastFn lookup_ts32_to_f64[] = {ts32_to_f64_avx512, ts32_to_f64_avx_avx2,
                                     ts32_to_f64_sse4_2, ts32_to_f64_scalar};

/**
 * @brief Dispatch table for s64 to f64 conversions.
 *
 * Variants:
 * - Index 0: ts64_to_f64_avx512() — Requires: AVX512F
 * - Index 1: ts64_to_f64_scalar() — Portable fallback
 */
const CastFn lookup_ts64_to_f64[] = {ts64_to_f64_avx512, ts64_to_f64_scalar};

/**
 * @brief Dispatch table for u8 to f64 conversions.
 *
 * Variants:
 * - Index 0: tu8_to_f64_avx512() — Requires: AVX512F, AVX2
 * - Index 1: tu8_to_f64_scalar() — Portable fallback
 */
const CastFn lookup_tu8_to_f64[] = {tu8_to_f64_avx512, tu8_to_f64_scalar};

/**
 * @brief Dispatch table for u32 to f64 conversions.
 *
 * Variants:
 * - Index 0: tu32_to_f64_avx512() — Requires: AVX512F
 * - Index 1: tu32_to_f64_scalar() — Portable fallback
 */
const CastFn lookup_tu32_to_f64[] = {tu32_to_f64_avx512, tu32_to_f64_scalar};

/**
 * @brief Dispatch table for u64 to f64 conversions.
 *
 * Variants:
 * - Index 0: tu64_to_f64_avx512() — Requires: AVX512F
 * - Index 1: tu64_to_f64_scalar() — Portable fallback
 */
const CastFn lookup_tu64_to_f64[] = {tu64_to_f64_avx512, tu64_to_f64_scalar};

/**
 * @brief Dispatch table for s8 to s32 conversions.
 *
 * Variants:
 * - Index 0: ts8_to_s32_avx512() — Requires: AVX512F
 * - Index 1: ts8_to_s32_avx2() — Requires: AVX2
 * - Index 2: ts8_to_s32_sse4_2() — Requires: SSE4.2
 * - Index 3: ts8_to_s32_scalar() — Portable fallback
 */
const CastFn lookup_ts8_to_s32[] = {ts8_to_s32_avx512, ts8_to_s32_avx2,
                                    ts8_to_s32_sse4_2, ts8_to_s32_scalar};

/**
 * @brief Dispatch table for s8 to s64 conversions.
 *
 * Variants:
 * - Index 0: ts8_to_s64_avx512() — Requires: AVX512F
 * - Index 1: ts8_to_s64_avx2() — Requires: AVX2
 * - Index 2: ts8_to_s64_sse4_2() — Requires: SSE4.2
 * - Index 3: ts8_to_s64_scalar() — Portable fallback
 */
const CastFn lookup_ts8_to_s64[] = {ts8_to_s64_avx512, ts8_to_s64_avx2,
                                    ts8_to_s64_sse4_2, ts8_to_s64_scalar};

/**
 * @brief Dispatch table for s32 to s8 conversions.
 *
 * Variants:
 * - Index 0: ts32_to_s8_avx512() — Requires: AVX512F
 * - Index 1: ts32_to_s8_scalar() — Portable fallback
 */
const CastFn lookup_ts32_to_s8[] = {ts32_to_s8_avx512, ts32_to_s8_scalar};

/**
 * @brief Dispatch table for s32 to s64 conversions.
 *
 * Variants:
 * - Index 0: ts32_to_s64_avx512() — Requires: AVX512F
 * - Index 1: ts32_to_s64_avx2() — Requires: AVX2
 * - Index 2: ts32_to_s64_sse4_2() — Requires: SSE4.2
 * - Index 3: ts32_to_s64_scalar() — Portable fallback
 */
const CastFn lookup_ts32_to_s64[] = {ts32_to_s64_avx512, ts32_to_s64_avx2,
                                     ts32_to_s64_sse4_2, ts32_to_s64_scalar};

/**
 * @brief Dispatch table for s64 to s8 conversions.
 *
 * Variants:
 * - Index 0: ts64_to_s8_avx512() — Requires: AVX512F, AVX512BW
 * - Index 1: ts64_to_s8_scalar() — Portable fallback
 */
const CastFn lookup_ts64_to_s8[] = {ts64_to_s8_avx512, ts64_to_s8_scalar};

/**
 * @brief Dispatch table for s64 to s32 conversions.
 *
 * Variants:
 * - Index 0: ts64_to_s32_avx512() — Requires: AVX512F
 * - Index 1: ts64_to_s32_scalar() — Portable fallback
 */
const CastFn lookup_ts64_to_s32[] = {ts64_to_s32_avx512, ts64_to_s32_scalar};

/**
 * @brief Dispatch table for u8 to u32 conversions.
 *
 * Variants:
 * - Index 0: tu8_to_u32_avx512() — Requires: AVX512F
 * - Index 1: tu8_to_u32_avx2() — Requires: AVX2
 * - Index 2: tu8_to_u32_scalar() — Portable fallback
 */
const CastFn lookup_tu8_to_u32[] = {tu8_to_u32_avx512, tu8_to_u32_avx2,
                                    tu8_to_u32_scalar};

/**
 * @brief Dispatch table for u8 to u64 conversions.
 *
 * Variants:
 * - Index 0: tu8_to_u64_avx512() — Requires: AVX512F
 * - Index 1: tu8_to_u64_avx2() — Requires: AVX2
 * - Index 2: tu8_to_u64_sse4_2() — Requires: SSE4.2
 * - Index 3: tu8_to_u64_scalar() —
 */
const CastFn lookup_tu8_to_u64[] = {tu8_to_u64_avx512, tu8_to_u64_avx2,
                                    tu8_to_u64_sse4_2, tu8_to_u64_scalar};

/**
 * @brief Dispatch table for u32 to u8 conversions.
 *
 * Variants:
 * - Index 0: tu32_to_u8_avx512() — Requires: AVX512F
 * - Index 1: tu32_to_u8_scalar() — Portable fallback
 */
const CastFn lookup_tu32_to_u8[] = {tu32_to_u8_avx512, tu32_to_u8_scalar};

/**
 * @brief Dispatch table for u32 to u64 conversions.
 *
 * Variants:
 * - Index 0: tu32_to_u64_avx512() — Requires: AVX512F
 * - Index 1: tu32_to_u64_avx2() — Requires: AVX2
 * - Index 2: tu32_to_u64_sse4_2() — Requires: SSE4.2
 * - Index 3: tu32_to_u64_scalar() — Portable fallback
 */
const CastFn lookup_tu32_to_u64[] = {tu32_to_u64_avx512, tu32_to_u64_avx2,
                                     tu32_to_u64_sse4_2, tu32_to_u64_scalar};

/**
 * @brief Dispatch table for u64 to u8 conversions.
 *
 * Variants:
 * - Index 0: tu64_to_u8_avx512() — Requires: AVX512F, AVX512BW
 * - Index 1: tu64_to_u8_scalar() — Portable fallback
 */
const CastFn lookup_tu64_to_u8[] = {tu64_to_u8_avx512, tu64_to_u8_scalar};

/**
 * @brief Dispatch table for u64 to u32 conversions.
 *
 * Variants:
 * - Index 0: tu64_to_u32_avx512() — Requires: AVX512F
 * - Index 1: tu64_to_u32_scalar() — Portable fallback
 */
const CastFn lookup_tu64_to_u32[] = {tu64_to_u32_avx512, tu64_to_u32_scalar};

/**
 * @brief Dispatch table for s8 to u8 conversions.
 *
 * Variants:
 * - Index 0: ts8_to_u8_avx512() — Requires: AVX512F
 * - Index 1: ts8_to_u8_avx_avx2() — Requires: AVX/AVX2
 * - Index 2: ts8_to_u8_sse4_2() — Requires: SSE4.2
 * - Index 3: ts8_to_u8_scalar() — Portable fallback
 */
const CastFn lookup_ts8_to_u8[] = {ts8_to_u8_avx512, ts8_to_u8_avx_avx2,
                                   ts8_to_u8_sse4_2, ts8_to_u8_scalar};

/**
 * @brief Dispatch table for s8 to u32 conversions.
 *
 * Variants:
 * - Index 0: ts8_to_u32_avx512() — Requires: AVX512F
 * - Index 1: ts8_to_u32_avx2() — Requires: AVX2
 * - Index 2: ts8_to_u32_sse4_2() — Requires: SSE4.2
 * - Index 3: ts8_to_u32_scalar() — Portable fallback
 */
const CastFn lookup_ts8_to_u32[] = {ts8_to_u32_avx512, ts8_to_u32_avx2,
                                    ts8_to_u32_sse4_2, ts8_to_u32_scalar};

/**
 * @brief Dispatch table for s8 to u64 conversions.
 *
 * Variants:
 * - Index 0: ts8_to_u64_avx512() — Requires: AVX512F
 * - Index 1: ts8_to_u64_scalar() — Portable fallback
 */
const CastFn lookup_ts8_to_u64[] = {ts8_to_u64_avx512, ts8_to_u64_scalar};

/**
 * @brief Dispatch table for s32 to u8 conversions.
 *
 * Variants:
 * - Index 0: ts32_to_u8_avx512() — Requires: AVX512F, AVX512BW
 * - Index 1: ts32_to_u8_avx2() — Requires: AVX2
 * - Index 2: ts32_to_u8_sse4_2() — Requires: SSE4.2
 * - Index 3: ts32_to_u8_scalar() — Portable fallback
 */
const CastFn lookup_ts32_to_u8[] = {ts32_to_u8_avx512, ts32_to_u8_avx2,
                                    ts32_to_u8_sse4_2, ts32_to_u8_scalar};

/**
 * @brief Dispatch table for s32 to u32 conversions.
 *
 * Variants:
 * - Index 0: ts32_to_u32_avx512() — Requires: AVX512F
 * - Index 1: ts32_to_u32_avx_avx2() — Requires: AVX/AVX2
 * - Index 2: ts32_to_u32_sse4_2() — Requires: SSE4.2
 * - Index 3: ts32_to_u32_scalar() — Portable fallback
 */
const CastFn lookup_ts32_to_u32[] = {ts32_to_u32_avx512, ts32_to_u32_avx_avx2,
                                     ts32_to_u32_sse4_2, ts32_to_u32_scalar};

/**
 * @brief Dispatch table for s32 to u64 conversions.
 *
 * Variants:
 * - Index 0: ts32_to_u64_avx512() — Requires: AVX512F
 * - Index 1: ts32_to_u64_avx2() — Requires: AVX2
 * - Index 2: ts32_to_u64_sse4_2() — Requires: SSE4.2
 * - Index 3: ts32_to_u64_scalar() — Portable fallback
 */
const CastFn lookup_ts32_to_u64[] = {ts32_to_u64_avx512, ts32_to_u64_avx2,
                                     ts32_to_u64_sse4_2, ts32_to_u64_scalar};

/**
 * @brief Dispatch table for s64 to u8 conversions.
 *
 * Variants:
 * - Index 0: ts64_to_u8_avx512() — Requires: AVX512F
 * - Index 1: ts64_to_u8_scalar() — Portable fallback
 */
const CastFn lookup_ts64_to_u8[] = {ts64_to_u8_avx512, ts64_to_u8_scalar};

/**
 * @brief Dispatch table for s64 to u32 conversions.
 *
 * Variants:
 * - Index 0: ts64_to_u32_avx512() — Requires: AVX512F
 * - Index 1: ts64_to_u32_scalar() — Portable fallback
 */
const CastFn lookup_ts64_to_u32[] = {ts64_to_u32_avx512, ts64_to_u32_scalar};

/**
 * @brief Dispatch table for s64 to u64 conversions.
 *
 * Variants:
 * - Index 0: ts64_to_u64_avx512() — Requires: AVX512F
 * - Index 1: ts64_to_u64_avx_avx2() — Requires: AVX/AVX2
 * - Index 2: ts64_to_u64_sse4_2() — Requires: SSE4.2
 * - Index 3: ts64_to_u64_scalar() — Portable fallback
 */
const CastFn lookup_ts64_to_u64[] = {ts64_to_u64_avx512, ts64_to_u64_avx_avx2,
                                     ts64_to_u64_sse4_2, ts64_to_u64_scalar};

/**
 * @brief Dispatch table for u8 to s8 conversions.
 *
 * Variants:
 * - Index 0: tu8_to_s8_avx512() — Requires: AVX512F
 * - Index 1: tu8_to_s8_avx_avx2() — Requires: AVX/AVX2
 * - Index 2: tu8_to_s8_sse4_2() — Requires: SSE4.2
 * - Index 3: tu8_to_s8_scalar() — Portable fallback
 */
const CastFn lookup_tu8_to_s8[] = {tu8_to_s8_avx512, tu8_to_s8_avx_avx2,
                                   tu8_to_s8_sse4_2, tu8_to_s8_scalar};

/**
 * @brief Dispatch table for u8 to s32 conversions.
 *
 * Variants:
 * - Index 0: tu8_to_s32_avx512() — Requires: AVX512F
 * - Index 1: tu8_to_s32_avx2() — Requires: AVX2
 * - Index 2: tu8_to_s32_sse4_2() — Requires: SSE4.2
 * - Index 3: tu8_to_s32_scalar() — Portable fallback
 */
const CastFn lookup_tu8_to_s32[] = {tu8_to_s32_avx512, tu8_to_s32_avx2,
                                    tu8_to_s32_sse4_2, tu8_to_s32_scalar};

/**
 * @brief Dispatch table for u8 to s64 conversions.
 *
 * Variants:
 * - Index 0: tu8_to_s64_avx512() — Requires: AVX512F
 * - Index 1: tu8_to_s64_avx2() — Requires: AVX2
 * - Index 2: tu8_to_s64_sse4_2() — Requires: SSE4.2
 * - Index 3: tu8_to_s64_scalar() — Portable fallback
 */
const CastFn lookup_tu8_to_s64[] = {tu8_to_s64_avx512, tu8_to_s64_avx2,
                                    tu8_to_s64_sse4_2, tu8_to_s64_scalar};

/**
 * @brief Dispatch table for u32 to s8 conversions.
 *
 * Variants:
 * - Index 0: tu32_to_s8_avx512() — Requires: AVX512F, AVX512BW
 * - Index 1: tu32_to_s8_scalar() — Portable fallback
 */
const CastFn lookup_tu32_to_s8[] = {tu32_to_s8_avx512, tu32_to_s8_scalar};

/**
 * @brief Dispatch table for u32 to s32 conversions.
 *
 * Variants:
 * - Index 0: tu32_to_s32_avx512() — Requires: AVX512F
 * - Index 1: tu32_to_s32_avx_avx2() — Requires: AVX/AVX2
 * - Index 2: tu32_to_s32_sse4_2() — Requires: SSE4.2
 * - Index 3: tu32_to_s32_scalar() — Portable fallback
 */
const CastFn lookup_tu32_to_s32[] = {tu32_to_s32_avx512, tu32_to_s32_avx_avx2,
                                     tu32_to_s32_sse4_2, tu32_to_s32_scalar};

/**
 * @brief Dispatch table for u32 to s64 conversions.
 *
 * Variants:
 * - Index 0: tu32_to_s64_avx512() — Requires: AVX512F
 * - Index 1: tu32_to_s64_avx2() — Requires: AVX2
 * - Index 2: tu32_to_s64_sse4_2() — Requires: SSE4.2
 * - Index 3: tu32_to_s64_scalar() — Portable fallback
 */
const CastFn lookup_tu32_to_s64[] = {tu32_to_s64_avx512, tu32_to_s64_avx2,
                                     tu32_to_s64_sse4_2, tu32_to_s64_scalar};

/**
 * @brief Dispatch table for u64 to s8 conversions.
 *
 * Variants:
 * - Index 0: tu64_to_s8_avx512() — Requires: AVX512F
 * - Index 1: tu64_to_s8_scalar() — Portable fallback
 */
const CastFn lookup_tu64_to_s8[] = {tu64_to_s8_avx512, tu64_to_s8_scalar};

/**
 * @brief Dispatch table for u64 to s32 conversions.
 *
 * Variants:
 * - Index 0: tu64_to_s32_avx512() — Requires: AVX512F
 * - Index 1: tu64_to_s32_scalar() — Portable fallback
 */
const CastFn lookup_tu64_to_s32[] = {tu64_to_s32_avx512, tu64_to_s32_scalar};

/**
 * @brief Dispatch table for u64 to s64 conversions.
 *
 * Variants:
 * - Index 0: tu64_to_s64_avx512() — Requires: AVX512F
 * - Index 1: tu64_to_s64_avx_avx2() — Requires: AVX/AVX2
 * - Index 2: tu64_to_s64_sse4_2() — Requires: SSE4.2
 * - Index 3: tu64_to_s64_scalar() — Portable fallback
 */
const CastFn lookup_tu64_to_s64[] = {tu64_to_s64_avx512, tu64_to_s64_avx_avx2,
                                     tu64_to_s64_sse4_2, tu64_to_s64_scalar};
