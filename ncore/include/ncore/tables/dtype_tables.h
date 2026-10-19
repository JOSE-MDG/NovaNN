/**
 * @file dtype_tables.h
 * @brief Lookup tables for dtype classification and size queries.
 *
 * @details
 * Each boolean table is a @c NUM_DTYPES × 1 array indexed by @c DType_
 * value; a @c true entry means the dtype belongs to that category. All
 * tables are @c const and read-only after process startup, safe to
 * access from any thread.
 *
 * The @ref lookup_dtype_sizes table stores the byte-width of each
 * dtype (e.g., @c Float32 → 4, @c Signed64 → 8).
 *
 * @see dtype.h   DType_ enumeration and classification API.
 * @see dtype_tables.c Table definitions (storage).
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>

#include <ncore/headeronly/macros.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief True for Float32, Float64, Float16, BFloat16, Float8E4M3fn,
 * Float8E5M2, Float4E2M1fn.
 */
extern const bool floating[NUM_DTYPES][1];

/**
 * @brief True for Signed8, UnSigned8, Signed16, UnSigned16, Signed32,
 * UnSigned32, Signed64, UnSigned64.
 */
extern const bool integer[NUM_DTYPES][1];

/**
 * @brief True for Signed8, Signed16, Signed32, Signed64.
 */
extern const bool signed_integer[NUM_DTYPES][1];

/**
 * @brief True for UnSigned8, UnSigned16, UnSigned32, UnSigned64.
 */
extern const bool unsigned_integer[NUM_DTYPES][1];

/**
 * @brief True for QSigned8, QSigned16, QSigned32.
 */
extern const bool quantized_signed_integer[NUM_DTYPES][1];

/**
 * @brief True for QUnSigned8, QUnSigned16, QUnSigned32.
 */
extern const bool quantized_unsigned_integer[NUM_DTYPES][1];

/**
 * @brief True for Float4E2M1fn and all quantized types.
 */
extern const bool quantizable_dtype[NUM_DTYPES][1];

/**
 * @brief Byte-width of each @ref DType_, i.e. the @c sizeof of the
 * corresponding C type: @c Float32 → 4, @c Signed64 → 8,
 * @c QSigned8 → 1.
 */
extern const size_t lookup_dtype_sizes[NUM_DTYPES];

#ifdef __cplusplus
}
#endif
