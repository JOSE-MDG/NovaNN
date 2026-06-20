/**
 * @file float_formatter.c
 * @brief Implementation of floating-point element formatting logic.
 *
 * @details
 * This module implements the conversion of IEEE 754 floating-point values
 * into strings. It is designed to be called by the element dispatch table
 * and relies on the calling context to provide formatting parameters
 * like precision and notation mode.
 *
 * ## Architecture
 * - **Special Case Handling**: Explicit checks for `isnan` and `isinf` ensure
 *   consistent output ("nan", "inf", "-inf") across platforms.
 * - **Notation Switching**: Uses a simple conditional to select between
 *   the `%.*e` and `%.*f` format specifiers.
 *
 * @see float_formatter.h Interface definitions.
 */

#include <math.h>
#include <stdio.h>

#include "float_formatter.h"

/**
 * @brief Convert a floating-point value into a character buffer.
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
