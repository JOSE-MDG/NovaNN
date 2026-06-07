/**
 * @file float_formatter.h
 * @brief Floating-point element formatting.
 *
 * @details
 * Formats a single float/double value into a caller-owned buffer.
 * Handles inf, nan, scientific notation, and normal decimal notation.
 * Float16 and BFloat16 values must be cast to double before calling.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>

/**
 * @brief Format a float value into a buffer.
 *
 * Handles nan, inf, -inf, scientific notation (%e), and normal
 * decimal notation (%f).
 *
 * @param[out] buf       Output buffer.
 * @param[in]  buf_size  Buffer size.
 * @param[in]  val       The float value (double precision).
 * @param[in]  sci       If true, use %e notation; otherwise %f.
 * @param[in]  precision Decimal places (passed to printf).
 * @return Number of chars written (excl. null), or negative on error.
 */
int float_format_value(char *buf, size_t buf_size, double val, bool sci,
                       int precision);
