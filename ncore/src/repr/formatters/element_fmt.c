/**
 * @file element_fmt.c
 * @brief Implementation of the per-dtype element formatting dispatch table.
 *
 * @details
 * This module provides a highly efficient O(1) dispatch mechanism for
 * formatting individual tensor elements into strings. By using a static
 * function pointer table indexed by @ref DType_, it eliminates the overhead
 * of large switch statements in the inner loops of layout renderers.
 *
 * Each formatter handles the extraction of raw bytes from the tensor's
 * data pointer, casts them to the appropriate C type, and delegates to
 * specialized numeric formatters (float, int, or quantized).
 *
 * ## Architecture
 * - **Dispatch Table**: `g_element_formatters` acts as the single entry
 *   point for element formatting.
 * - **Specialized Handlers**: Internal static functions (e.g., `fmt_float32`)
 *   bridge the gap between raw bytes and numeric formatting logic.
 * - **Context Awareness**: All formatters respect settings in @ref ReprContext,
 *   such as precision, scientific notation, and boolean interpretation.
 *
 * @see element_fmt.h Dispatch table interface.
 * @see float_formatter.h Floating-point formatting logic.
 * @see int_formatter.h Integer formatting logic.
 */

#include <ncore/core/dtype.h>
#include <ncore/tensor.h>

#include "element_fmt.h"
#include "float_formatter.h"
#include "int_formatter.h"
#include "qint_formatter.h"

/**
 * @brief Format a Float32 element.
 */
static int fmt_float32(char *buf, size_t cap, const void *ptr,
                       const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  return float_format_value(buf, cap, *(const float *)ptr, ctx->use_sci,
                            ctx->effective_precision);
}

/**
 * @brief Format a Float64 element.
 */
static int fmt_float64(char *buf, size_t cap, const void *ptr,
                       const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  return float_format_value(buf, cap, *(const double *)ptr, ctx->use_sci,
                            ctx->effective_precision);
}

/**
 * @brief Format a Float16 element.
 */
static int fmt_float16(char *buf, size_t cap, const void *ptr,
                       const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  return float_format_value(buf, cap, (double)*(const float16 *)ptr,
                            ctx->use_sci, ctx->effective_precision);
}

/**
 * @brief Format a BFloat16 element.
 */
static int fmt_bfloat16(char *buf, size_t cap, const void *ptr,
                        const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  return float_format_value(buf, cap, (double)*(const bfloat16 *)ptr,
                            ctx->use_sci, ctx->effective_precision);
}

/**
 * @brief Format a Signed8 element.
 */
static int fmt_signed8(char *buf, size_t cap, const void *ptr,
                       const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  (void)ctx;
  return int_format_value(buf, cap, (int64_t)*(const int8_t *)ptr);
}

/**
 * @brief Format an UnSigned8 element.
 */
static int fmt_unsigned8(char *buf, size_t cap, const void *ptr,
                         const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  return uint_format_value(buf, cap, (uint64_t)*(const uint8_t *)ptr,
                           ctx->is_bool);
}

/**
 * @brief Format a Signed32 element.
 */
static int fmt_signed32(char *buf, size_t cap, const void *ptr,
                        const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  (void)ctx;
  return int_format_value(buf, cap, (int64_t)*(const int32_t *)ptr);
}

/**
 * @brief Format an UnSigned32 element.
 */
static int fmt_unsigned32(char *buf, size_t cap, const void *ptr,
                          const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  (void)ctx;
  return uint_format_value(buf, cap, (uint64_t)*(const uint32_t *)ptr, false);
}

/**
 * @brief Format a Signed64 element.
 */
static int fmt_signed64(char *buf, size_t cap, const void *ptr,
                        const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  (void)ctx;
  return int_format_value(buf, cap, *(const int64_t *)ptr);
}

/**
 * @brief Format an UnSigned64 element.
 */
static int fmt_unsigned64(char *buf, size_t cap, const void *ptr,
                          const Tensor *ten, const ReprContext *ctx) {
  (void)ten;
  (void)ctx;
  return uint_format_value(buf, cap, *(const uint64_t *)ptr, false);
}

/**
 * @brief Format a QSigned8 element.
 */
static int fmt_qsigned8(char *buf, size_t cap, const void *ptr,
                        const Tensor *ten, const ReprContext *ctx) {
  return qint_format_value(buf, cap, (int)*(const int8_t *)ptr, ten->scale_,
                           ten->zero_point_, ctx->options.show_dequantized);
}

/**
 * @brief Format a QUnSigned8 element.
 */
static int fmt_qunsigned8(char *buf, size_t cap, const void *ptr,
                          const Tensor *ten, const ReprContext *ctx) {
  return qint_format_value(buf, cap, (int)*(const uint8_t *)ptr, ten->scale_,
                           ten->zero_point_, ctx->options.show_dequantized);
}

/**
 * @brief Global dispatch table for element formatting.
 *
 * @details
 * Every entry 0..NUM_DTYPES-1 is explicitly populated via designated
 * initializers. This ensures that @ref format_element() is always safe
 * to call for any valid @ref DType_.
 */
element_formatter_t g_element_formatters[NUM_DTYPES] = {
    [Float32] = fmt_float32,   [Float64] = fmt_float64,
    [Float16] = fmt_float16,   [BFloat16] = fmt_bfloat16,
    [Signed8] = fmt_signed8,   [UnSigned8] = fmt_unsigned8,
    [QSigned8] = fmt_qsigned8, [QUnSigned8] = fmt_qunsigned8,
    [Signed32] = fmt_signed32, [UnSigned32] = fmt_unsigned32,
    [Signed64] = fmt_signed64, [UnSigned64] = fmt_unsigned64,
};
