/**
 * @file int_formatter.c
 * @brief Integer element formatting implementation.
 *
 * @details
 * Implements the conversion of signed and unsigned integer types into
 * human-readable strings. Uses portable format specifiers (@c PRId64,
 * @c PRIu64) and supports boolean interpretation of unsigned values.
 *
 * @see int_formatter.h   Interface definitions.
 * @see element_fmt.c     Element dispatch table.
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
