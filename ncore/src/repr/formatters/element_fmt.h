/**
 * @file element_fmt.h
 * @brief Dispatch table for per-dtype element formatting.
 *
 * @details
 * Replaces large switch statements with a function-pointer dispatch table
 * indexed by DType_.  Each handler formats one element into a caller-owned
 * buffer and returns the number of characters written (excluding the null
 * terminator).
 *
 * The table is initialised at file scope via designated initializers and
 * covers all 12 DType_ values.  Formatting parameters (precision, sci
 * mode, is_bool, show_dequantized) are read from the ReprContext.
 *
 * @see float_formatter.h Per-float-format internals
 * @see int_formatter.h   Per-integer-format internals
 * @see qint_formatter.h  Per-quantized-format internals
 */

#pragma once

#include <ncore/dtype.h>
#include <ncore/repr/repr_context.h>
#include <ncore/tensor.h>
#include <stddef.h>

/**
 * @brief Function pointer type for per-dtype element formatting.
 *
 * @param buf       Output buffer (caller-owned).
 * @param buf_size  Size of the output buffer.
 * @param elem_ptr  Pointer to the element value in tensor storage.
 * @param ten       The owning tensor (needed for scale_/zero_point_).
 * @param ctx       ReprContext with precision, sci, is_bool, etc.
 * @return Number of chars written (excl. null), or negative on error.
 */
typedef int (*element_formatter_t)(char *buf, size_t buf_size,
                                   const void *elem_ptr, const Tensor *ten,
                                   const ReprContext *ctx);

/**
 * @brief Dispatch table indexed by DType_.
 *
 * Initialised in element_fmt.c with one handler per dtype.  Safe to call
 * for any value 0..11; entries for all 12 DType_ values are populated.
 */
extern element_formatter_t g_element_formatters[NUM_DTYPES];

/**
 * @brief Convenience wrapper around the dispatch table.
 *
 * @param buf      Output buffer.
 * @param buf_size Output buffer size.
 * @param ptr      Pointer to the element value.
 * @param ten      Owning tensor.
 * @param ctx      ReprContext.
 * @return Number of chars written, or negative on error.
 */
static inline int format_element(char *buf, size_t buf_size, const void *ptr,
                                 const Tensor *ten, const ReprContext *ctx) {
  return g_element_formatters[ten->dtype](buf, buf_size, ptr, ten, ctx);
}
