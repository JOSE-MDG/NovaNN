/**
 * @file qint_formatter.c
 * @brief Quantized integer element formatter.
 *
 * @details
 * Renders the raw quantized value and, when show_dequantized is true and
 * scale > 0, appends the dequantized float in parentheses:
 *
 *   42 (0.3294)
 *
 * Dequantization: (raw - zero_point) * scale.
 */

#include "qint_formatter.h"
#include <stdio.h>

/**
 * @brief Format a quantized element.
 *
 * @param[out] buf              Output buffer.
 * @param[in]  buf_size         Buffer size.
 * @param[in]  raw_val          Raw quantized integer value.
 * @param[in]  scale            Quantization scale factor.
 * @param[in]  zero_point       Quantization zero-point.
 * @param[in]  show_dequantized If true and scale > 0, append " (%.4f)".
 * @return Number of chars written (excl. null).
 */
int qint_format_value(char *buf, size_t buf_size, int raw_val, float scale,
                      int32_t zero_point, bool show_dequantized) {
  if (show_dequantized && scale > 0.0F) {
    double dq = ((double)raw_val - (double)zero_point) * (double)scale;
    return snprintf(buf, buf_size, "%d (%.4f)", raw_val, dq);
  }
  return snprintf(buf, buf_size, "%d", raw_val);
}
