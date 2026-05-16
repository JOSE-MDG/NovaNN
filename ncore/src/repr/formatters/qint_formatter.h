/**
 * @file qint_formatter.h
 * @brief Quantized integer element formatting.
 *
 * @details
 * Writes the raw quantized value and, when show_dequantized is true,
 * appends the dequantized float in parentheses:
 *   "42 (0.3294)"
 *
 * Dequantization formula: (raw - zero_point) * scale.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Format a quantized element.
 *
 * @param buf              Output buffer.
 * @param buf_size         Buffer size.
 * @param raw_val          Raw quantized integer value.
 * @param scale            Quantization scale factor.
 * @param zero_point       Quantization zero-point.
 * @param show_dequantized If true and scale > 0, append " (%.4f)".
 * @return Number of chars written (excl. null).
 */
int qint_format_value(char *buf, size_t buf_size, int raw_val, float scale,
                      int32_t zero_point, bool show_dequantized);
