/**
 * @file float_formatter.c
 * @brief Floating-point element formatting implementation.
 *
 * @details
 * Implements the conversion of IEEE 754 floating-point values into
 * human-readable strings. Designed to be called by the element
 * dispatch table (@ref g_element_formatters) and relies on the
 * calling context for formatting parameters (precision, notation).
 *
 * @section special-cases Special Cases
 *
 * @li @c nan — rendered as the literal string @c "nan".
 * @li @c +inf / @c -inf — rendered as @c "inf" / @c "-inf".
 * @li Subnormal values are handled transparently by @c snprintf.
 *
 * @see float_formatter.h   Interface definitions.
 * @see element_fmt.c       Element dispatch table.
 * @see repr_context.h      Formatting context.
 */

#include <math.h>
#include <stdio.h>

#include "float_formatter.h"

/**
 * @brief Convert a floating-point value into a character buffer.
 */
int float_format_value(char *buf, size_t buf_size, double val,
                       const ReprContext *ctx) {
  if (isnan(val)) {
    return snprintf(buf, buf_size, "%s", "nan");
  }
  if (isinf(val)) {
    if (val > 0) {
      return snprintf(buf, buf_size, "%s", "inf");
    }
    return snprintf(buf, buf_size, "%s", "-inf");
  }
  if (ctx->use_sci) {
    return snprintf(buf, buf_size, "%.*e", ctx->effective_precision, val);
  }
  return snprintf(buf, buf_size, "%.*f", ctx->effective_precision, val);
}
