/**
 * @file element_fmt.h
 * @brief Per-dtype element formatting dispatch table interface.
 *
 * @details
 * Defines the @ref element_formatter_t function pointer type, the
 * global dispatch table @ref g_element_formatters, and the inline
 * wrapper @ref format_element(). The dispatch mechanism provides
 * O(1) element formatting by indexing directly into the table with
 * the tensor's @ref DType_ value.
 *
 * @see element_fmt.c          Table initialization and handlers.
 * @see float_formatter.h      Floating-point formatting internals.
 * @see int_formatter.h        Integer formatting internals.
 * @see qint_formatter.h       Quantized formatting internals.
 */

#pragma once

#include <stddef.h>

#include <ncore/core/dtype.h>
#include <ncore/repr/repr_context.h>
#include <ncore/tensor.h>

/**
 * @brief Function pointer signature for element-wise string formatting.
 *
 * @param[out] buf       Output buffer where the string will be written.
 * @param[in]  buf_size  Capacity of the output buffer in bytes.
 * @param[in]  elem_ptr  Pointer to the element in tensor memory.
 * @param[in]  ten       Pointer to the parent tensor (for metadata).
 * @param[in]  ctx       Pointer to the current representation context.
 *
 * @return Number of characters written (excluding null-terminator), or
 *         a negative value on error.
 */
typedef int (*element_formatter_t)(char *buf, size_t buf_size,
                                   const void *elem_ptr, const Tensor *ten,
                                   const ReprContext *ctx);

/**
 * @var g_element_formatters
 * @brief Global dispatch table containing formatters for all DTypes.
 *
 * @details
 * A `NUM_DTYPES`-sized array of @ref element_formatter_t function
 * pointers, indexed directly by @ref DType_ values (`0` ..
 * `NUM_DTYPES-1`). Populated at compile time via designated
 * initializers in @ref element_fmt.c.
 *
 * @see format_element()
 * @see DType_
 */
extern element_formatter_t g_element_formatters[NUM_DTYPES];

/**
 * @brief Format a single tensor element into a string buffer.
 *
 * @details
 * Inline wrapper that looks up the correct formatter function for
 * the tensor's data type in @ref g_element_formatters and executes
 * it. This is the primary interface for layout renderers.
 *
 * @param[out] buf      Output string buffer. Must not be `nullptr`.
 * @param[in]  buf_size Buffer capacity in bytes.
 * @param[in]  ptr      Pointer to the element data.
 * @param[in]  ten      Parent tensor.
 * @param[in]  ctx      Active representation context.
 *
 * @return Number of characters written (excluding null-terminator).
 */
static inline int format_element(char *buf, size_t buf_size, const void *ptr,
                                 const Tensor *ten, const ReprContext *ctx) {
  return g_element_formatters[ten->dtype](buf, buf_size, ptr, ten, ctx);
}
