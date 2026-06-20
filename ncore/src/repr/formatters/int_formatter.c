/**
 * @file int_formatter.c
 * @brief Implementation of integer element formatting logic.
 *
 * @details
 * This module provides the conversion of integer types to strings, using
 * standard C integer format specifiers. It includes a specialized path
 * for boolean interpretation of unsigned values.
 *
 * ## Architecture
 * - **Format Specifiers**: Uses `PRId64` and `PRIu64` for maximum
 *   portability across 64-bit platforms.
 * - **Boolean Path**: Overrides numeric output with string literals
 *   when requested by the @ref ReprContext.
 *
 * @see int_formatter.h Interface definitions.
 */

#include <inttypes.h>
#include <stdio.h>

#include "int_formatter.h"

/**
 * @brief Convert a signed 64-bit integer into a string.
 */
int int_format_value(char *buf, size_t buf_size, int64_t val) {
  return snprintf(buf, buf_size, "%" PRId64, val);
}

/**
 * @brief Convert an unsigned 64-bit integer into a string.
 */
int uint_format_value(char *buf, size_t buf_size, uint64_t val, bool is_bool) {
  if (is_bool) {
    return snprintf(buf, buf_size, "%s", val ? "True" : "False");
  }
  return snprintf(buf, buf_size, "%" PRIu64, val);
}
