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
 * @return Number of chars written (excl. null).
 */
int int_format_value(char *buf, size_t buf_size, int64_t val);

/**
 * @brief Format an unsigned integer.
 *
 * When is_bool is true and val is 0 or 1, writes "False" or "True".
 *
 * @return Number of chars written (excl. null).
 */
int uint_format_value(char *buf, size_t buf_size, uint64_t val, bool is_bool);
