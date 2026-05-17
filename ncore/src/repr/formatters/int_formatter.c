/**
 * @file int_formatter.c
 * @brief Integer element formatter implementation.
 *
 * @details
 * Signed values use %ld formatting; unsigned values use %lu.  The
 * UnSigned8 + is_bool path writes "True" or "False".
 */

#include "int_formatter.h"
#include <inttypes.h>
#include <stdio.h>

int int_format_value(char *buf, size_t buf_size, int64_t val) {
  return snprintf(buf, buf_size, "%" PRId64, val);
}

int uint_format_value(char *buf, size_t buf_size, uint64_t val, bool is_bool) {
  if (is_bool) {
    return snprintf(buf, buf_size, "%s", val ? "True" : "False");
  }
  return snprintf(buf, buf_size, "%" PRIu64, val);
}
