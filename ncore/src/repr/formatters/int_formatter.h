/**
 * @file int_formatter.h
 * @brief Integer element formatting interface.
 *
 * @details
 * Declares @ref int_format_value() and @ref uint_format_value(), the
 * low-level routines that convert signed and unsigned integer values
 * into human-readable strings. Supports boolean interpretation for
 * unsigned 8-bit types.
 *
 * @see int_formatter.c  Implementation details.
 * @see element_fmt.h    Higher-level dispatch table.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Convert a signed 64-bit integer into a string.
 *
 * @param[out] buf      Target string buffer. Must not be `nullptr`.
 * @param[in]  buf_size Capacity of @p buf in bytes.
 * @param[in]  val      The integer value to format.
 *
 * @return Number of characters written (excluding null-terminator).
 */
int int_format_value(char *buf, size_t buf_size, int64_t val);

/**
 * @brief Convert an unsigned 64-bit integer into a string.
 *
 * @details
 * If @p is_bool is `true`, the value is rendered as `"True"` (non-zero)
 * or `"False"` (zero) instead of a numeric label.
 *
 * @param[out] buf      Target string buffer. Must not be `nullptr`.
 * @param[in]  buf_size Capacity of @p buf in bytes.
 * @param[in]  val      The unsigned integer value to format.
 * @param[in]  is_bool  If `true`, use boolean labels for 0 and 1.
 *
 * @return Number of characters written (excluding null-terminator).
 */
int uint_format_value(char *buf, size_t buf_size, uint64_t val, bool is_bool);
