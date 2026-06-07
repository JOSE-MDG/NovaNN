/**
 * @file int_formatter.h
 * @brief Integer element formatting.
 *
 * @details
 * Formats signed and unsigned integer values.  UnSigned8 can optionally
 * be rendered as True / False when the is_bool flag is set.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Format a signed integer.
 *
 * @param[out] buf      Output buffer.
 * @param[in]  buf_size Buffer size.
 * @param[in]  val      Signed integer value.
 * @return Number of chars written (excl. null).
 */
int int_format_value(char *buf, size_t buf_size, int64_t val);

/**
 * @brief Format an unsigned integer.
 *
 * When is_bool is true and val is 0 or 1, writes "False" or "True".
 *
 * @param[out] buf      Output buffer.
 * @param[in]  buf_size Buffer size.
 * @param[in]  val      Unsigned integer value.
 * @param[in]  is_bool  If true and val is 0 or 1, write "False"/"True".
 * @return Number of chars written (excl. null).
 */
int uint_format_value(char *buf, size_t buf_size, uint64_t val, bool is_bool);
