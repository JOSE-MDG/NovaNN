/**
 * @file qint_formatter.c
 * @brief Quantized element formatting implementation.
 *
 * @details
 * Implements the conversion of quantized data types into strings.
 * Provides an optional dequantization path that calculates the
 * real-value representation using the tensor's scale and zero-point
 * metadata.
 *
 * @section dequantization-formula Dequantization Formula
 *
 * @c "float_val = (raw - zero_point) * scale"
 *
 * When @ref ReprOptions::show_dequantized is enabled, the output
 * includes both the raw integer and the dequantized value for
 * diagnostic clarity.
 *
 * @see qint_formatter.h   Interface definitions.
 * @see element_fmt.c      Element dispatch table.
 * @see repr_options.h     Dequantization display option.
 */

#include <stdio.h>

#include "qint_formatter.h"

/**
 * @brief Convert a quantized element into a string.
 *
 * @details
 * If @p show_dequantized is @c true, the output includes both the raw
 * integer and the dequantized float value in the format
 * @c "raw (float)".  The dequantization formula is:
 * @c "float_val = (raw - zero_point) * scale".
 *
 * @param[out] buf              Target string buffer. Must not be @c nullptr.
 * @param[in]  buf_size         Capacity of @p buf in bytes.
 * @param[in]  raw_val          The raw quantized integer value.
 * @param[in]  scale            Quantization scale factor.
 * @param[in]  zero_point       Quantization zero-point offset.
 * @param[in]  show_dequantized If @c true, append dequantized float value.
 *
 * @return Number of characters written (excluding the null-terminator).
 */
int qint_format_value(char *buf, size_t buf_size, int raw_val, float scale,
                      int32_t zero_point, bool show_dequantized) {
  if (show_dequantized && scale > 0.0F) {
    auto dq = ((double)raw_val - (double)zero_point) * (double)scale;
    return snprintf(buf, buf_size, "%d (%.4f)", raw_val, dq);
  }
  return snprintf(buf, buf_size, "%d", raw_val);
}
