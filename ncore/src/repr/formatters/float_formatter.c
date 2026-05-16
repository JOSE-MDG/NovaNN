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

int float_format_value(char *buf, size_t buf_size, double val, bool sci,
                       int precision) {
  if (isnan(val)) {
    return snprintf(buf, buf_size, "%s", "nan");
  }
  if (isinf(val)) {
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
