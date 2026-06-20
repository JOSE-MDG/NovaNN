/**
 * @file float_formatter.h
 * @brief Logic for formatting floating-point tensor elements.
 *
 * @details
 * This header defines the interface for converting floating-point values
 * (double precision) into human-readable strings. It supports scientific
 * notation (%e), standard decimal (%f), and correctly handles IEEE 754
 * special values such as infinity and NaN.
 *
 * @see float_formatter.c Implementation details.
 * @see element_fmt.h Higher-level dispatch table.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>

/**
 * @brief Convert a floating-point value into a character buffer.
 *
 * @details
 * Performs the low-level string conversion using standard library utilities.
 * The formatter handles precision and notation overrides as specified by
 * the active representation context.
 *
 * @param[out] buf       Target string buffer.
 * @param[in]  buf_size  Capacity of the buffer in bytes.
 * @param[in]  val       The numeric value to format (as double).
 * @param[in]  sci       If true, force scientific notation; else decimal.
 * @param[in]  precision Number of decimal places to include.
 *
 * @return Number of characters written (excluding null-terminator).
 */
int float_format_value(char *buf, size_t buf_size, double val, bool sci,
                       int precision);
