/**
 * @file element_fmt.c
 * @brief Per-dtype element formatting dispatch table.
 *
 * @details
 * Implements one formatter function per DType_ value and populates
 * g_element_formatters[] with designated initializers so that callers
 * can dispatch on ten->dtype in O(1) without a switch.
 *
 * Each handler casts elem_ptr to the appropriate C type, extracts the
 * value, and delegates to the type-specific formatter (float_formatter,
 * int_formatter, or qint_formatter).  Formatting parameters are read
 * from the ReprContext.
 */

#include "element_fmt.h"
#include "float_formatter.h"
#include "int_formatter.h"
#include "qint_formatter.h"
#include <ncore/dtype.h>
#include <ncore/tensor.h>

/**
 * @brief Format a Float32 element.
 *
 * @param[out] buf      Output buffer.
 * @param[in]  cap      Buffer capacity.
 * @param[in]  ptr      Pointer to the float value.
 * @param[in]  ten      Owning tensor (unused).
 * @param[in]  ctx      ReprContext with sci/precision settings.
 * @return Number of chars written (excl. null).
 */
static int fmt_float32(char *buf, size_t cap, const void *ptr,
                       const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  return float_format_value(buf, cap, *(const float *)ptr, ctx->use_sci,
                            ctx->effective_precision);
}

/**
 * @brief Format a Float64 element.
 *
 * @param[out] buf      Output buffer.
 * @param[in]  cap      Buffer capacity.
 * @param[in]  ptr      Pointer to the double value.
 * @param[in]  ten      Owning tensor (unused).
 * @param[in]  ctx      ReprContext with sci/precision settings.
 * @return Number of chars written (excl. null).
 */
static int fmt_float64(char *buf, size_t cap, const void *ptr,
                       const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  return float_format_value(buf, cap, *(const double *)ptr, ctx->use_sci,
                            ctx->effective_precision);
}

/**
 * @brief Format a Float16 element.
 *
 * @param[out] buf      Output buffer.
 * @param[in]  cap      Buffer capacity.
 * @param[in]  ptr      Pointer to the float16 value.
 * @param[in]  ten      Owning tensor (unused).
 * @param[in]  ctx      ReprContext with sci/precision settings.
 * @return Number of chars written (excl. null).
 */
static int fmt_float16(char *buf, size_t cap, const void *ptr,
                       const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  return float_format_value(buf, cap, (double)*(const float16 *)ptr,
                            ctx->use_sci, ctx->effective_precision);
}

/**
 * @brief Format a BFloat16 element.
 *
 * @param[out] buf      Output buffer.
 * @param[in]  cap      Buffer capacity.
 * @param[in]  ptr      Pointer to the bfloat16 value.
 * @param[in]  ten      Owning tensor (unused).
 * @param[in]  ctx      ReprContext with sci/precision settings.
 * @return Number of chars written (excl. null).
 */
static int fmt_bfloat16(char *buf, size_t cap, const void *ptr,
                        const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  return float_format_value(buf, cap, (double)*(const bfloat16 *)ptr,
                            ctx->use_sci, ctx->effective_precision);
}

/**
 * @brief Format a Signed8 element.
 *
 * @param[out] buf      Output buffer.
 * @param[in]  cap      Buffer capacity.
 * @param[in]  ptr      Pointer to the int8_t value.
 * @param[in]  ten      Owning tensor (unused).
 * @param[in]  ctx      ReprContext (unused).
 * @return Number of chars written (excl. null).
 */
static int fmt_signed8(char *buf, size_t cap, const void *ptr,
                       const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  (void)ctx;
  return int_format_value(buf, cap, (int64_t)*(const int8_t *)ptr);
}

/**
 * @brief Format an UnSigned8 element.
 *
 * @param[out] buf      Output buffer.
 * @param[in]  cap      Buffer capacity.
 * @param[in]  ptr      Pointer to the uint8_t value.
 * @param[in]  ten      Owning tensor (unused).
 * @param[in]  ctx      ReprContext (is_bool flag).
 * @return Number of chars written (excl. null).
 */
static int fmt_unsigned8(char *buf, size_t cap, const void *ptr,
                         const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  return uint_format_value(buf, cap, (uint64_t)*(const uint8_t *)ptr,
                           ctx->is_bool);
}

/**
 * @brief Format a Signed32 element.
 *
 * @param[out] buf      Output buffer.
 * @param[in]  cap      Buffer capacity.
 * @param[in]  ptr      Pointer to the int32_t value.
 * @param[in]  ten      Owning tensor (unused).
 * @param[in]  ctx      ReprContext (unused).
 * @return Number of chars written (excl. null).
 */
static int fmt_signed32(char *buf, size_t cap, const void *ptr,
                        const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  (void)ctx;
  return int_format_value(buf, cap, (int64_t)*(const int32_t *)ptr);
}

/**
 * @brief Format an UnSigned32 element.
 *
 * @param[out] buf      Output buffer.
 * @param[in]  cap      Buffer capacity.
 * @param[in]  ptr      Pointer to the uint32_t value.
 * @param[in]  ten      Owning tensor (unused).
 * @param[in]  ctx      ReprContext (unused).
 * @return Number of chars written (excl. null).
 */
static int fmt_unsigned32(char *buf, size_t cap, const void *ptr,
                          const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  (void)ctx;
  return uint_format_value(buf, cap, (uint64_t)*(const uint32_t *)ptr, false);
}

/**
 * @brief Format a Signed64 element.
 *
 * @param[out] buf      Output buffer.
 * @param[in]  cap      Buffer capacity.
 * @param[in]  ptr      Pointer to the int64_t value.
 * @param[in]  ten      Owning tensor (unused).
 * @param[in]  ctx      ReprContext (unused).
 * @return Number of chars written (excl. null).
 */
static int fmt_signed64(char *buf, size_t cap, const void *ptr,
                        const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  (void)ctx;
  return int_format_value(buf, cap, *(const int64_t *)ptr);
}

/**
 * @brief Format an UnSigned64 element.
 *
 * @param[out] buf      Output buffer.
 * @param[in]  cap      Buffer capacity.
 * @param[in]  ptr      Pointer to the uint64_t value.
 * @param[in]  ten      Owning tensor (unused).
 * @param[in]  ctx      ReprContext (unused).
 * @return Number of chars written (excl. null).
 */
static int fmt_unsigned64(char *buf, size_t cap, const void *ptr,
                          const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  (void)ctx;
  return uint_format_value(buf, cap, *(const uint64_t *)ptr, false);
}

/**
 * @brief Format a QSigned8 element.
 *
 * @param[out] buf      Output buffer.
 * @param[in]  cap      Buffer capacity.
 * @param[in]  ptr      Pointer to the int8_t value.
 * @param[in]  ten      Owning tensor (scale_/zero_point_).
 * @param[in]  ctx      ReprContext (show_dequantized flag).
 * @return Number of chars written (excl. null).
 */
static int fmt_qsigned8(char *buf, size_t cap, const void *ptr,
                        const Tensor *ten, const ReprContext *ctx) {
  return qint_format_value(buf, cap, (int)*(const int8_t *)ptr, ten->scale_,
                           ten->zero_point_, ctx->options.show_dequantized);
}

/**
 * @brief Format a QUnSigned8 element.
 *
 * @param[out] buf      Output buffer.
 * @param[in]  cap      Buffer capacity.
 * @param[in]  ptr      Pointer to the uint8_t value.
 * @param[in]  ten      Owning tensor (scale_/zero_point_).
 * @param[in]  ctx      ReprContext (show_dequantized flag).
 * @return Number of chars written (excl. null).
 */
static int fmt_qunsigned8(char *buf, size_t cap, const void *ptr,
                          const Tensor *ten, const ReprContext *ctx) {
  return qint_format_value(buf, cap, (int)*(const uint8_t *)ptr, ten->scale_,
                           ten->zero_point_, ctx->options.show_dequantized);
}

/**
 * @brief Dispatch table indexed by DType_.
 *
 * Every entry 0..11 is populated.  Float32 (index 0) is the default.
 */
element_formatter_t g_element_formatters[NUM_DTYPES] = {
    [Float32] = fmt_float32,   [Float64] = fmt_float64,
    [Float16] = fmt_float16,   [BFloat16] = fmt_bfloat16,
    [Signed8] = fmt_signed8,   [UnSigned8] = fmt_unsigned8,
    [QSigned8] = fmt_qsigned8, [QUnSigned8] = fmt_qunsigned8,
    [Signed32] = fmt_signed32, [UnSigned32] = fmt_unsigned32,
    [Signed64] = fmt_signed64, [UnSigned64] = fmt_unsigned64,
};
