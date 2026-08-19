/**
 * @file dtype_tables.h
 * @brief Lookup tables for dtype classification and size queries.
 *
 * @details
 * This header declares a set of global @c const lookup tables that
 * categorize @ref DType_ values into boolean masks.  Each table is
 * a @c NUM_DTYPES × 1 array of @c bool, indexed by @c DType_ value.
 * A @c true entry at index @c i means the dtype at position @c i
 * belongs to the corresponding category.
 *
 * @section tables Tables
 *
 * @li @ref floating — True for Float32, Float64, Float16, BFloat16,
 *   Float8E4M3fn, Float8E5M2, Float4E2M1fn
 * @li @ref integer — True for all integer + quantized integer types
 * @li @ref signed_integer — True for Signed8, QSigned8, Signed16,
 *   QSigned16, Signed32, QSigned32, Signed64
 * @li @ref unsigned_integer — True for UnSigned8, QUnSigned8,
 *   UnSigned16, QUnSigned16, UnSigned32, QUnSigned32, UnSigned64
 * @li @ref quantized_signed_integer — True for QSigned8, QSigned16,
 *   QSigned32
 * @li @ref quantized_unsigned_integer — True for QUnSigned8,
 *   QUnSigned16, QUnSigned32
 *
 * The @ref lookup_dtype_sizes table stores the byte-width of each
 * dtype (e.g., @c Float32 → 4, @c Signed64 → 8).
 *
 * @section usage Usage
 *
 * The classification functions in @ref dtype.c (e.g.,
 * @c is_floating()) index into these tables with @c input->dtype as
 * the row index and read the @c bool entry at column 0.
 *
 * @note These tables are @c const and read-only after process
 *       startup.  They are safe to access from any thread.
 *
 * @see dtype.h        DType_ enumeration and classification API.
 * @see dtype.c        Classification function implementations.
 * @see dtype_tables.c Table definitions (storage).
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>

#include <ncore/headeronly/macros.h>

/**
 * @var floating
 * @brief Boolean mask indicating which dtypes are floating-point.
 *
 * @details
 * @c floating[dtype][0] is @c true when @c dtype is @c Float32,
 * @c Float64, @c Float16, @c BFloat16, @c Float8E4M3fn, @c Float8E5M2,
 * or @c Float4E2M1fn.
 *
 * @see is_floating()  Classification function using this table.
 * @see DType_         Data-type enumeration.
 */
extern const bool floating[NUM_DTYPES][1];

/**
 * @var integer
 * @brief Boolean mask indicating which dtypes are integer types.
 *
 * @details
 * @c integer[dtype][0] is @c true for all non-quantized integer
 * dtypes: @c Signed8, @c UnSigned8, @c Signed16, @c UnSigned16,
 * @c Signed32, @c UnSigned32, @c Signed64, @c UnSigned64.
 *
 * @see is_integer()  Classification function using this table.
 */
extern const bool integer[NUM_DTYPES][1];

/**
 * @var signed_integer
 * @brief Boolean mask indicating which dtypes are signed integer
 *        types.
 *
 * @details
 * @c signed_integer[dtype][0] is @c true for @c Signed8, @c Signed16,
 * @c Signed32, and @c Signed64.
 *
 * @see is_signed_integer()  Classification function using this table.
 */
extern const bool signed_integer[NUM_DTYPES][1];

/**
 * @var unsigned_integer
 * @brief Boolean mask indicating which dtypes are unsigned integer
 *        types.
 *
 * @details
 * @c unsigned_integer[dtype][0] is @c true for @c UnSigned8,
 * @c UnSigned16, @c UnSigned32, and @c UnSigned64.
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
 * @c quantized_signed_integer[dtype][0] is @c true for @c QSigned8,
 * @c QSigned16, and @c QSigned32.
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
 * @c quantized_unsigned_integer[dtype][0] is @c true for
 * @c QUnSigned8, @c QUnSigned16, and @c QUnSigned32.
 *
 * @see is_quantized_unsigned_integer()  Classification function using this
 * table.
 */
extern const bool quantized_unsigned_integer[NUM_DTYPES][1];

/**
 * @var quantizable_dtype
 * @brief Boolean mask indicating which dtypes can be quantized.
 *
 * @details
 * @c quantizable_dtype[dtype][0] is @c true for @c Float4E2M1fn,
 * @c QSigned8, @c QUnSigned8, @c QSigned16, @c QUnSigned16,
 * @c QSigned32, and @c QUnSigned32.  These types are eligible to
 * participate in quantization operations.
 *
 * @see is_quantizable_dtype()  Classification function using this table.
 */
extern const bool quantizable_dtype[NUM_DTYPES][1];

/**
 * @var lookup_dtype_sizes
 * @brief Byte-width of each @ref DType_.
 *
 * @details
 * @c lookup_dtype_sizes[dtype] returns @c sizeof the corresponding
 * C type.  For example:
 * @li @c Float32 → sizeof(float32) = 4
 * @li @c Float8E4M3fn → sizeof(float8_e4m3fn) = 1
 * @li @c Signed64 → sizeof(int64) = 8
 * @li @c QSigned8 → sizeof(qint8) = 1
 *
 * @see dtype_size()  Public function querying this table.
 * @see DType_        Data-type enumeration.
 */
extern const size_t lookup_dtype_sizes[NUM_DTYPES];
