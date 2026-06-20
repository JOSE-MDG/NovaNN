/**
 * @file dtype_tables.h
 * @brief Lookup tables for dtype classification and size queries.
 *
 * @details
 * This header declares a set of global `const` lookup tables that
 * categorise @ref DType_ values into boolean masks.  Each table is
 * a `NUM_DTYPES × 1` array of `bool`, indexed by `DType_` value.
 * A `true` entry at index `i` means the dtype at position `i`
 * belongs to the corresponding category.
 *
 * ## Tables
 */
// clang-format off
/**
 * | Table                             | True for                                       |
 * |-----------------------------------|------------------------------------------------|
 * | @ref floating                     | Float32, Float64, Float16, BFloat16            |
 * | @ref integer                      | All integer + quantized integer types          |
 * | @ref signed_integer               | Signed8, QSigned8, Signed32, Signed64          |
 * | @ref unsigned_integer             | UnSigned8, QUnSigned8, UnSigned32, UnSigned64  |
 * | @ref quantized_signed_integer     | QSigned8 only                                  |
 * | @ref quantized_unsigned_integer   | QUnSigned8 only                                |
 */
// clang-format on
/**
 * The @ref lookup_dtype_sizes table stores the byte-width of each
 * dtype (e.g., `Float32 → 4`, `Signed64 → 8`).
 *
 * ## Usage
 *
 * The classification functions in @ref dtype.c (e.g.,
 * `is_floating()`) index into these tables with `input->dtype` as
 * the row index and cast the `uint8_t` result to `bool`.
 *
 * @note These tables are `const` and read-only after process
 *       startup.  They are safe to access from any thread.
 *
 * @see dtype.h       DType_ enumeration and classification API.
 * @see dtype.c       Classification function implementations.
 * @see dtype_tables.c  Table definitions (storage).
 */

#pragma once

#include <ncore/headeronly/macros.h>
#include <stdbool.h>
#include <stddef.h>

/**
 * @var floating
 * @brief Boolean mask indicating which dtypes are floating-point.
 *
 * @details
 * `floating[dtype][0]` is `true` when `dtype` is `Float32`,
 * `Float64`, `Float16`, or `BFloat16`.
 *
 * @see is_floating()  Classification function using this table.
 * @see DType_         Data-type enumeration.
 */
extern const bool floating[NUM_DTYPES][1];

/**
 * @var integer
 * @brief Boolean mask indicating which dtypes are integer types
 *        (signed, unsigned, or quantized).
 *
 * @details
 * `integer[dtype][0]` is `true` for all integer and quantized
 * integer dtypes: `Signed8`, `UnSigned8`, `QSigned8`,
 * `QUnSigned8`, `Signed32`, `UnSigned32`, `Signed64`,
 * `UnSigned64`.
 *
 * @see is_integer()  Classification function using this table.
 */
extern const bool integer[NUM_DTYPES][1];

/**
 * @var signed_integer
 * @brief Boolean mask indicating which dtypes are signed integer
 *        types (including quantized).
 *
 * @details
 * `signed_integer[dtype][0]` is `true` for `Signed8`,
 * `QSigned8`, `Signed32`, and `Signed64`.  Note that the
 * quantized type `QSigned8` is included here.
 *
 * @see is_signed_integer()  Classification function using this table.
 */
extern const bool signed_integer[NUM_DTYPES][1];

/**
 * @var unsigned_integer
 * @brief Boolean mask indicating which dtypes are unsigned integer
 *        types (including quantized).
 *
 * @details
 * `unsigned_integer[dtype][0]` is `true` for `UnSigned8`,
 * `QUnSigned8`, `UnSigned32`, and `UnSigned64`.  Note that the
 * quantized type `QUnSigned8` is included here.
 *
 * @see is_unsigned_integer()  Classification function using this table.
 */
extern const bool unsigned_integer[NUM_DTYPES][1];

/**
 * @var quantized_signed_integer
 * @brief Boolean mask indicating which dtypes are quantized signed
 *        integer types.
 *
 * @details
 * `quantized_signed_integer[dtype][0]` is `true` only for
 * `QSigned8`.
 *
 * @see is_quantized_signed_integer()  Classification function using this table.
 */
extern const bool quantized_signed_integer[NUM_DTYPES][1];

/**
 * @var quantized_unsigned_integer
 * @brief Boolean mask indicating which dtypes are quantized
 *        unsigned integer types.
 *
 * @details
 * `quantized_unsigned_integer[dtype][0]` is `true` only for
 * `QUnSigned8`.
 *
 * @see is_quantized_unsigned_integer()  Classification function using this
 * table.
 */
extern const bool quantized_unsigned_integer[NUM_DTYPES][1];

/**
 * @var lookup_dtype_sizes
 * @brief Byte-width of each @ref DType_.
 *
 * @details
 * `lookup_dtype_sizes[dtype]` returns `sizeof` the corresponding
 * C type.  For example:
 * - `Float32 → sizeof(float32) = 4`
 * - `Signed64 → sizeof(int64) = 8`
 * - `QSigned8 → sizeof(qint8) = 1`
 *
 * @see dtype_size()  Public function querying this table.
 * @see DType_        Data-type enumeration.
 */
extern const size_t lookup_dtype_sizes[NUM_DTYPES];
