/**
 * @file dtype_tables.h
 * @brief Lookup tables for dtype classification.
 *
 * This header provides a set of global lookup tables that categorize
 * data types (dtypes) into different categories such as floating-point,
 * integer, signed, unsigned, and quantized variants. These tables are
 * used throughout the codebase to efficiently check the properties of
 * a given dtype without requiring conditional logic.
 *
 * Each table is a 2D array of size [NUM_DTYPES][1], where NUM_DTYPES
 * is the total number of recognized data types in the system. The boolean
 * value at index i indicates whether the dtype at position i belongs to
 * the corresponding category.
 *
 * @note These tables are defined externally and must be initialized in
 *       a corresponding source file.
 */

#pragma once

#include <ncore/macros.h>

/**
 * @brief Indicates which dtypes are floating-point types.
 */
extern const bool floating[NUM_DTYPES][1];

/**
 * @brief Indicates which dtypes are integer types.
 */
extern const bool integer[NUM_DTYPES][1];

/**
 * @brief Indicates which dtypes are signed integer types.
 */
extern const bool signed_integer[NUM_DTYPES][1];

/**
 * @brief Indicates which dtypes are unsigned integer types.
 */
extern const bool unsigned_integer[NUM_DTYPES][1];

/**
 * @brief Indicates which dtypes are quantized signed integer types.
 */
extern const bool quantized_signed_integer[NUM_DTYPES][1];

/**
 * @brief Indicates which dtypes are quantized unsigned integer types.
 */
extern const bool quantized_unsigned_integer[NUM_DTYPES][1];

/**
 * @brief Specify the size for each data type
 */
extern const size_t lookup_dtype_sizes[NUM_DTYPES];
