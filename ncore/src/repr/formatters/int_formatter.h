/**
 * @file int_formatter.h
 * @brief Logic for formatting integer tensor elements.
 *
 * @details
 * This header defines the interface for converting signed and unsigned
 * integers into human-readable strings. It also supports specialized
 * boolean interpretation for unsigned 8-bit types.
 *
 * @see int_formatter.c Implementation details.
 * @see element_fmt.h Higher-level dispatch table.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Convert a signed 64-bit integer into a string.
 *
 * @param[out] buf      Target string buffer.
 * @param[in]  buf_size Capacity of the buffer in bytes.
 * @param[in]  val      The integer value to format.
 *
 * @return Number of characters written.
 */
int int_format_value(char *buf, size_t buf_size, int64_t val);

/**
 * @brief Convert an unsigned 64-bit integer into a string.
 *
 * @details
 * If the `is_bool` parameter is set, the function renders 1 as "True"
 * and 0 as "False" instead of using numeric labels.
 *
 * @param[out] buf      Target string buffer.
 * @param[in]  buf_size Capacity of the buffer in bytes.
 * @param[in]  val      The unsigned integer value to format.
 * @param[in]  is_bool  If true, use boolean labels for 0 and 1.
 *
 * @return Number of characters written.
 */
int uint_format_value(char *buf, size_t buf_size, uint64_t val, bool is_bool);
