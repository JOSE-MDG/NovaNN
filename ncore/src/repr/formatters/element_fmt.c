/**
 * @file element_fmt.c
 * @brief Per-dtype element formatting dispatch table implementation.
 *
 * @details
 * Provides an O(1) dispatch mechanism for formatting individual tensor
 * elements into strings. Uses a static function pointer table indexed
 * by @ref DType_ to eliminate switch-statement overhead in layout
 * renderer inner loops.
 *
 * Each handler extracts raw bytes from the tensor's data pointer,
 * casts them to the appropriate C type, and delegates to a
 * specialized numeric formatter (@ref float_format_value,
 * @ref int_format_value, @ref uint_format_value, or
 * @ref qint_format_value).
 *
 * @section dispatch-table Dispatch Table
 *
 * @ref g_element_formatters is a @c NUM_DTYPES-sized array populated
 * via designated initializers. Every entry @c 0 .. @c NUM_DTYPES-1 is
 * explicitly filled, ensuring that @ref format_element() is always
 * safe to call for any valid @ref DType_.
 *
 * @see element_fmt.h          Dispatch table interface.
 * @see float_formatter.h      Floating-point formatting logic.
 * @see int_formatter.h        Integer formatting logic.
 * @see qint_formatter.h       Quantized formatting logic.
 */

#include <ncore/core/dtype.h>
#include <ncore/core/fp_utils.h>
#include <ncore/headeronly/macros.h>
#include <ncore/native/cpu/dtype/casting.h>
#include <ncore/tensor.h>

#include "element_fmt.h"
#include "float_formatter.h"
#include "int_formatter.h"
#include "qint_formatter.h"

/* ────────────────────────────────────────────────────────────────
 *  Floating-point formatters
 * ──────────────────────────────────────────────────────────────── */

/**
 * @brief Format a Float32 element.
 */
static inline int fmt_float32(char *buf, size_t cap, const void *ptr,
                              const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  return float_format_value(buf, cap, (double)*(const float *)ptr, ctx);
}

/**
 * @brief Format a Float64 element.
 */
static inline int fmt_float64(char *buf, size_t cap, const void *ptr,
                              const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  return float_format_value(buf, cap, *(const double *)ptr, ctx);
}

/**
 * @brief Format a Float16 element.
 */
static inline int fmt_float16(char *buf, size_t cap, const void *ptr,
                              const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
#ifdef _GNUC_CLANG_
  return float_format_value(buf, cap, (double)*(const float16 *)ptr, ctx);
#else
  return float_format_value(buf, cap,
                            (double)fp16_to_float(*(const float16 *)ptr), ctx);
#endif
}

/**
 * @brief Format a BFloat16 element.
 */
static inline int fmt_bfloat16(char *buf, size_t cap, const void *ptr,
                               const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
#ifdef _GNUC_CLANG_
  return float_format_value(buf, cap, (double)*(const bfloat16 *)ptr, ctx);
#else
  return float_format_value(buf, cap,
                            (double)bf16_to_float(*(const bfloat16 *)ptr), ctx);
#endif
}

/**
 * @brief Format a Float8 E4M3FN element.
 */
static inline int fmt_float8_e4m3fn(char *buf, size_t cap, const void *ptr,
                                    const Tensor *ten, const ReprContext *ctx) {

  (void)ten;
  return float_format_value(
      buf, cap, (double)fp8e4m3fn_to_float(*(const float8_e4m3fn *)ptr), ctx);
}

/**
 * @brief Format a Float8 E5M2 element.
 */
static inline int fmt_float8_e5m2(char *buf, size_t cap, const void *ptr,
                                  const Tensor *ten, const ReprContext *ctx) {

  (void)ten;
  return float_format_value(
      buf, cap, (double)fp8e5m2_to_float(*(const float8_e5m2 *)ptr), ctx);
}

/**
 * @brief Format a single Float4 E2M1FN element from a packed pair.
 */
static inline int fmt_float4_e2m1fn(char *buf, size_t cap, const void *ptr,
                                    const Tensor *ten, const ReprContext *ctx) {

  (void)ten;

  float lo;
  float hi;

  fp4e2m1x2_to_floats(*(const float4_e2m1fn_x2 *)ptr, &lo, &hi);

  double val = (ctx->sub_element_index == 0) ? (double)lo : (double)hi;
  return float_format_value(buf, cap, val, ctx);
}

/* ────────────────────────────────────────────────────────────────
 *  Signed integer formatters
 * ──────────────────────────────────────────────────────────────── */

/**
 * @brief Format a Signed8 element.
 */
static inline int fmt_signed8(char *buf, size_t cap, const void *ptr,
                              const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  (void)ctx;
  return int_format_value(buf, cap, (int64_t)*(const int8_t *)ptr);
}

/**
 * @brief Format a Signed16 element.
 */
static inline int fmt_signed16(char *buf, size_t cap, const void *ptr,
                               const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  (void)ctx;
  return int_format_value(buf, cap, (int64_t)*(const int16_t *)ptr);
}

/**
 * @brief Format a Signed32 element.
 */
static inline int fmt_signed32(char *buf, size_t cap, const void *ptr,
                               const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  (void)ctx;
  return int_format_value(buf, cap, (int64_t)*(const int32_t *)ptr);
}

/**
 * @brief Format a Signed64 element.
 */
static inline int fmt_signed64(char *buf, size_t cap, const void *ptr,
                               const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  (void)ctx;
  return int_format_value(buf, cap, *(const int64_t *)ptr);
}

/* ────────────────────────────────────────────────────────────────
 *  Unsigned integer formatters
 * ──────────────────────────────────────────────────────────────── */

/**
 * @brief Format an UnSigned8 element.
 */
static inline int fmt_unsigned8(char *buf, size_t cap, const void *ptr,
                                const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  return uint_format_value(buf, cap, (uint64_t)*(const uint8_t *)ptr,
                           ctx->is_bool);
}

/**
 * @brief Format an UnSigned16 element.
 */
static inline int fmt_unsigned16(char *buf, size_t cap, const void *ptr,
                                 const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  (void)ctx;
  return uint_format_value(buf, cap, (uint64_t)*(const uint16_t *)ptr, false);
}

/**
 * @brief Format an UnSigned32 element.
 */
static inline int fmt_unsigned32(char *buf, size_t cap, const void *ptr,
                                 const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  (void)ctx;
  return uint_format_value(buf, cap, (uint64_t)*(const uint32_t *)ptr, false);
}

/**
 * @brief Format an UnSigned64 element.
 */
static inline int fmt_unsigned64(char *buf, size_t cap, const void *ptr,
                                 const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  (void)ctx;
  return uint_format_value(buf, cap, *(const uint64_t *)ptr, false);
}

/* ────────────────────────────────────────────────────────────────
 *  Quantized formatters
 * ──────────────────────────────────────────────────────────────── */

/**
 * @brief Format a QSigned8 element.
 */
static inline int fmt_qsigned8(char *buf, size_t cap, const void *ptr,
                               const Tensor *ten, const ReprContext *ctx) {
  return qint_format_value(buf, cap, (int)*(const int8_t *)ptr, ten->scale_,
                           ten->zero_point_, ctx->options.show_dequantized);
}

/**
 * @brief Format a QUnSigned8 element.
 */
static inline int fmt_qunsigned8(char *buf, size_t cap, const void *ptr,
                                 const Tensor *ten, const ReprContext *ctx) {
  return qint_format_value(buf, cap, (int)*(const uint8_t *)ptr, ten->scale_,
                           ten->zero_point_, ctx->options.show_dequantized);
}

/**
 * @brief Format a QSigned16 element.
 */
static inline int fmt_qsigned16(char *buf, size_t cap, const void *ptr,
                                const Tensor *ten, const ReprContext *ctx) {
  return qint_format_value(buf, cap, (int)*(const int16_t *)ptr, ten->scale_,
                           ten->zero_point_, ctx->options.show_dequantized);
}

/**
 * @brief Format a QUnSigned16 element.
 */
static inline int fmt_qunsigned16(char *buf, size_t cap, const void *ptr,
                                  const Tensor *ten, const ReprContext *ctx) {
  return qint_format_value(buf, cap, (int)*(const uint16_t *)ptr, ten->scale_,
                           ten->zero_point_, ctx->options.show_dequantized);
}

/**
 * @brief Format a QSigned32 element.
 */
static inline int fmt_qsigned32(char *buf, size_t cap, const void *ptr,
                                const Tensor *ten, const ReprContext *ctx) {
  return qint_format_value(buf, cap, *(const int32_t *)ptr, ten->scale_,
                           ten->zero_point_, ctx->options.show_dequantized);
}

/**
 * @brief Format a QUnSigned32 element.
 */
static inline int fmt_qunsigned32(char *buf, size_t cap, const void *ptr,
                                  const Tensor *ten, const ReprContext *ctx) {
  return qint_format_value(buf, cap, (int)*(const uint32_t *)ptr, ten->scale_,
                           ten->zero_point_, ctx->options.show_dequantized);
}

/**
 * @brief Global dispatch table for element formatting.
 *
 * @details
 * Every entry @c 0 .. @c NUM_DTYPES-1 is explicitly populated via
 * designated initializers. This ensures that @ref format_element()
 * is always safe to call for any valid @ref DType_.
 *
 * @see format_element()
 * @see element_formatter_t
 */
element_formatter_t g_element_formatters[NUM_DTYPES] = {
    [Float32] = fmt_float32,
    [Float64] = fmt_float64,
    [Float16] = fmt_float16,
    [BFloat16] = fmt_bfloat16,
    [Float8E4M3fn] = fmt_float8_e4m3fn,
    [Float8E5M2] = fmt_float8_e5m2,
    [Float4E2M1fn] = fmt_float4_e2m1fn,
    [Signed8] = fmt_signed8,
    [UnSigned8] = fmt_unsigned8,
    [QSigned8] = fmt_qsigned8,
    [QUnSigned8] = fmt_qunsigned8,
    [Signed16] = fmt_signed16,
    [UnSigned16] = fmt_unsigned16,
    [QSigned16] = fmt_qsigned16,
    [QUnSigned16] = fmt_qunsigned16,
    [Signed32] = fmt_signed32,
    [UnSigned32] = fmt_unsigned32,
    [QSigned32] = fmt_qsigned32,
    [QUnSigned32] = fmt_qunsigned32,
    [Signed64] = fmt_signed64,
    [UnSigned64] = fmt_unsigned64,
};
