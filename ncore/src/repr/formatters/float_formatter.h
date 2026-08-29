/**
 * @file float_formatter.h
 * @brief Floating-point element formatting interface.
 *
 * @details
 * Declares @ref float_format_value(), the low-level routine that converts
 * IEEE 754 floating-point values into human-readable strings. Supports
 * standard decimal (@c %f) and scientific (@c %e) notation and IEEE 754
 * special values (NaN, inf).
 *
 * @see float_formatter.c  Implementation details.
 * @see element_fmt.h      Higher-level dispatch table.
 */

#pragma once

#include <stddef.h>

#include <ncore/repr/repr_context.h>

/**
 * @brief Convert a floating-point value into a character buffer.
 *
 * @details
 * Performs the low-level string conversion using @c snprintf. IEEE 754
 * special values (@c nan, @c inf, @c -inf) are handled explicitly.
 *
 * @param[out] buf       Target string buffer. Must not be @c nullptr.
 * @param[in]  buf_size  Capacity of @p buf in bytes.
 * @param[in]  val       The numeric value to format.
 * @param[in]  ctx       Active representation context. Must not be
 *                       @c nullptr.
 *
 * @return Number of characters written (excluding the null-terminator).
 */
int float_format_value(char *buf, size_t buf_size, double val,
                       const ReprContext *ctx);
