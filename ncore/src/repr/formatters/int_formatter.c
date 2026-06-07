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

/**
 * @brief Format a signed integer.
 *
 * @param[out] buf      Output buffer.
 * @param[in]  buf_size Buffer size.
 * @param[in]  val      Signed integer value.
 * @return Number of chars written (excl. null).
 */
int int_format_value(char *buf, size_t buf_size, int64_t val) {
  return snprintf(buf, buf_size, "%" PRId64, val);
}

/**
 * @brief Format an unsigned integer.
 *
 * @param[out] buf      Output buffer.
 * @param[in]  buf_size Buffer size.
 * @param[in]  val      Unsigned integer value.
 * @param[in]  is_bool  If true and val is 0 or 1, write "False"/"True".
 * @return Number of chars written (excl. null).
 */
int uint_format_value(char *buf, size_t buf_size, uint64_t val, bool is_bool) {
  if (is_bool) {
    return snprintf(buf, buf_size, "%s", val ? "True" : "False");
  }
  return snprintf(buf, buf_size, "%" PRIu64, val);
}
