/**
 * @file float_formatter.c
 * @brief Float-element formatter implementation.
 *
 * @details
 * Handles normal decimal (%f), scientific (%e), and special values
 * (inf, -inf, nan).  The caller is responsible for passing the
 * sci/precision parameters from the ReprContext.
 */

#include "float_formatter.h"
#include <math.h>
#include <stdio.h>

/**
 * @brief Format a float value into a buffer.
 *
 * @param[out] buf       Output buffer.
 * @param[in]  buf_size  Buffer size.
 * @param[in]  val       The float value (double precision).
 * @param[in]  sci       If true, use %e notation; otherwise %f.
 * @param[in]  precision Decimal places (passed to printf).
 * @return Number of chars written (excl. null), or negative on error.
 */
int float_format_value(char *buf, size_t buf_size, double val, bool sci,
                       int precision) {
  if (__builtin_isnan(val)) {
    return snprintf(buf, buf_size, "%s", "nan");
  }
  if (__builtin_isinf(val)) {
    if (val > 0) {
      return snprintf(buf, buf_size, "%s", "inf");
    }
    return snprintf(buf, buf_size, "%s", "-inf");
  }
  if (sci) {
    return snprintf(buf, buf_size, "%.*e", precision, val);
  }
  return snprintf(buf, buf_size, "%.*f", precision, val);
}
