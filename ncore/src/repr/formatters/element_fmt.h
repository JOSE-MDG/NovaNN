/**
 * @file element_fmt.h
 * @brief Dispatch table interface for per-dtype element formatting.
 *
 * @details
 * This header defines the function pointer type and global dispatch table
 * used to convert raw tensor elements into human-readable strings. The
 * mechanism is designed for O(1) performance, selecting the correct
 * formatter based on the tensor's @ref DType_ identifier.
 *
 * Each formatter implementation is responsible for type-safe data extraction
 * and applying visual parameters from the @ref ReprContext.
 *
 * ## Architecture
 * - **element_formatter_t**: The standard signature for all dtype-specific
 *   formatting functions.
 * - **g_element_formatters**: The externalized dispatch table indexed by
 *   DType value.
 * - **format_element()**: Inline wrapper that provides a clean, type-agnostic
 *   interface for layout renderers.
 *
 * @see element_fmt.c Table initialization and handlers.
 * @see  float_formatter.h Floating-point internals.
 */

#pragma once

#include <ncore/core/dtype.h>
#include <ncore/repr/repr_context.h>
#include <ncore/tensor.h>
#include <stddef.h>

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
 * @brief Global dispatch table containing formatters for all DTypes.
 *
 * @details
 * This table is initialised in @ref element_fmt.c and is indexed directly
 * by @ref DType_ values (0..NUM_DTYPES-1).
 */
extern element_formatter_t g_element_formatters[NUM_DTYPES];

/**
 * @brief High-level dispatch wrapper for element formatting.
 *
 * @details
 * Looks up the correct formatter function for the tensor's data type
 * and executes it. This is the primary interface for layout renderers.
 *
 * @param[out] buf      Output string buffer.
 * @param[in]  buf_size Buffer capacity.
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
