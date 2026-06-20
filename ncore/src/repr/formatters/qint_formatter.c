/**
 * @file qint_formatter.c
 * @brief Implementation of quantized element formatting logic.
 *
 * @details
 * This module implements the string conversion for quantized data types.
 * It provides an optional dequantization path that calculates the real-value
 * representation using the tensor's scale and zero-point metadata.
 *
 * ## Architecture
 * - **Formula**: Uses the standard affine dequantization formula:
 *   `float_val = (raw - zero_point) * scale`.
 * - **Combined Output**: If enabled, the dequantized value is appended in
 *   parentheses after the raw integer for diagnostic clarity.
 *
 * @see qint_formatter.h Interface definitions.
 */

#include "qint_formatter.h"
#include <stdio.h>

/**
 * @brief Convert a quantized element into a string.
 */
int qint_format_value(char *buf, size_t buf_size, int raw_val, float scale,
                      int32_t zero_point, bool show_dequantized) {
  if (show_dequantized && scale > 0.0F) {
    double dq = ((double)raw_val - (double)zero_point) * (double)scale;
    return snprintf(buf, buf_size, "%d (%.4f)", raw_val, dq);
  }
  return snprintf(buf, buf_size, "%d", raw_val);
}
