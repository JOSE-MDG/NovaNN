/**
 * @file qint_formatter.h
 * @brief Logic for formatting quantized tensor elements.
 *
 * @details
 * This header defines the interface for converting quantized (QSigned8,
 * QUnSigned8) tensor elements into strings. It supports displaying either
 * the raw integer value or the dequantized floating-point value.
 *
 * @see qint_formatter.c Implementation details.
 * @see element_fmt.h Higher-level dispatch table.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Convert a quantized element into a string.
 *
 * @details
 * If `show_dequantized` is true, the output includes both the raw
 * integer and the calculated float: "raw (float)".
 *
 * @param[out] buf              Target string buffer.
 * @param[in]  buf_size         Capacity of the buffer in bytes.
 * @param[in]  raw_val          The raw quantized integer.
 * @param[in]  scale            Quantization scale factor.
 * @param[in]  zero_point       Quantization zero-point offset.
 * @param[in]  show_dequantized If true, perform and append dequantization.
 *
 * @return Number of characters written.
 */
int qint_format_value(char *buf, size_t buf_size, int raw_val, float scale,
                      int32_t zero_point, bool show_dequantized);
