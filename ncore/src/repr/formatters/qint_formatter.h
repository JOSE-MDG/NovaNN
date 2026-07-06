/**
 * @file qint_formatter.h
 * @brief Quantized element formatting interface.
 *
 * @details
 * Declares @ref qint_format_value(), the low-level routine that
 * converts quantized (QSigned8, QUnSigned8, QSigned16, QUnSigned16,
 * QSigned32, QUnSigned32) tensor elements into human-readable strings.
 * Supports optional dequantization display.
 *
 * @see qint_formatter.c  Implementation details.
 * @see element_fmt.h     Higher-level dispatch table.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Convert a quantized element into a string.
 *
 * @details
 * If @p show_dequantized is `true`, the output includes both the raw
 * integer and the dequantized float value in the format
 * `"raw (float)"`. The dequantization formula is:
 * `float_val = (raw - zero_point) * scale`.
 *
 * @param[out] buf              Target string buffer. Must not be `nullptr`.
 * @param[in]  buf_size         Capacity of @p buf in bytes.
 * @param[in]  raw_val          The raw quantized integer value.
 * @param[in]  scale            Quantization scale factor.
 * @param[in]  zero_point       Quantization zero-point offset.
 * @param[in]  show_dequantized If `true`, append dequantized float value.
 *
 * @return Number of characters written (excluding null-terminator).
 */
int qint_format_value(char *buf, size_t buf_size, int raw_val, float scale,
                      int32_t zero_point, bool show_dequantized);
