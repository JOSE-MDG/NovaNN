/**
 * @file dtype.h
 * @brief Data-type definitions and classification for the NovaNN library.
 *
 * @details
 * This header provides:
 *
 * - **Type aliases** — Portable names for the numeric types used by
 *   tensor storage (`float32`, `int8`, `qint8`, etc.).  Each alias
 *   maps directly to a standard C or compiler-extension type.
 * - **DType_ enumeration** — A packed enum that identifies a data
 *   type at run time, used for dispatch tables and tensor metadata.
 * - **Classification functions** — `is_floating()`, `is_integer()`,
 *   etc. that test a tensor's dtype against category lookup tables.
 * - **Cast and size utilities** — `cast()` for type conversion and
 *   `dtype_size()` for byte-width queries.
 *
 * ## Type Alias Convention
 *
 * Public aliases use lowercase names (`float32`, `int8`, …) without
 * a prefix.  They are defined as direct typedefs to the underlying
 * C / compiler types and are stable across platforms.
 *
 * @see macros.h    ATTR(packed) and NOVA_INTERNAL_ASSERT macros.
 * @see tensor.h    Tensor struct embedding a @ref DType_ field.
 * @see storage.h   data_ptr union using these types.
 */

#pragma once

#include <ncore/headeronly/macros.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct Tensor;
typedef struct Tensor Tensor;

/** @brief 32-bit IEEE 754 single-precision float. */
typedef float float32;

/** @brief 64-bit IEEE 754 double-precision float. */
typedef double float64;

/** @brief 16-bit IEEE 754 half-precision float (compiler extension). */
typedef _Float16 float16;

/** @brief Brain Floating Point 16-bit (BF16, compiler extension). */
typedef __bf16 bfloat16;

/** @brief Signed 8-bit two's-complement integer. */
typedef int8_t int8;

/** @brief Unsigned 8-bit integer. */
typedef uint8_t uint8;

/** @brief Quantised signed 8-bit integer (typically int8 storage). */
typedef int8_t qint8;

/** @brief Quantised unsigned 8-bit integer (typically uint8 storage). */
typedef uint8_t quint8;

/** @brief Signed 32-bit two's-complement integer. */
typedef int32_t int32;

/** @brief Unsigned 32-bit integer. */
typedef uint32_t uint32;

/** @brief Signed 64-bit two's-complement integer. */
typedef int64_t int64;

/** @brief Unsigned 64-bit integer. */
typedef uint64_t uint64;

/**
 * @enum DType_
 * @brief Runtime data-type identifier for tensor elements.
 *
 * @details
 * Each tensor carries a @ref DType_ value that selects the correct
 * kernel, copy routine, and print formatter at run time.  The
 * values are sequential integers starting from 0, which allows
 * them to be used as array indices in dispatch tables (e.g.,
 * `cast_dispatch`, `lookup_dtype_sizes`).
 */
// clang-format off
/**
 * | Value | Name          | C type        | Bytes |
 * |-------|---------------|---------------|-------|
 * | 0     | `Float32`     | `float`       | 4     |
 * | 1     | `Float64`     | `double`      | 8     |
 * | 2     | `Float16`     | `_Float16`    | 2     |
 * | 3     | `BFloat16`    | `__bf16`      | 2     |
 * | 4     | `Signed8`     | `int8_t`      | 1     |
 * | 5     | `UnSigned8`   | `uint8_t`     | 1     |
 * | 6     | `QSigned8`    | `int8_t`      | 1     |
 * | 7     | `QUnSigned8`  | `uint8_t`     | 1     |
 * | 8     | `Signed32`    | `int32_t`     | 4     |
 * | 9     | `UnSigned32`  | `uint32_t`    | 4     |
 * | 10    | `Signed64`    | `int64_t`     | 8     |
 * | 11    | `UnSigned64`  | `uint64_t`    | 8     |
 */
// clang-format on
/**
 * @note This enum is packed (`ATTR(packed)`) to minimise its
 *       footprint in structs that are serialised or copied
 *       frequently.
 *
 * @see dtype_size()     Byte-width lookup.
 * @see is_floating()    Classification helpers.
 * @see cast()           Type conversion.
 */
typedef enum ATTR(packed) {
  Float32 = 0,     ///< 32-bit floating point.
  Float64 = 1,     ///< 64-bit floating point (double precision).
  Float16 = 2,     ///< 16-bit floating point (half precision).
  BFloat16 = 3,    ///< Brain floating point (16-bit).
  Signed8 = 4,     ///< Signed 8-bit integer.
  UnSigned8 = 5,   ///< Unsigned 8-bit integer.
  QSigned8 = 6,    ///< Quantized signed 8-bit integer.
  QUnSigned8 = 7,  ///< Quantized unsigned 8-bit integer.
  Signed32 = 8,    ///< Signed 32-bit integer.
  UnSigned32 = 9,  ///< Unsigned 32-bit integer.
  Signed64 = 10,   ///< Signed 64-bit integer.
  UnSigned64 = 11, ///< Unsigned 64-bit integer.
} DType_;

/**
 * @brief Check whether a tensor's dtype is a floating-point type.
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   `NULL`.
 *
 * @return `true` if `input->dtype` is `Float32`, `Float64`,
 *         `Float16`, or `BFloat16`.  `false` otherwise.
 *
 * @see is_integer()
 * @see is_signed_integer()
 */
bool is_floating(const Tensor *restrict input);

/**
 * @brief Check whether a tensor's dtype is an integer type
 *        (signed or unsigned, including quantized).
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   `NULL`.
 *
 * @return `true` if `input->dtype` is any integer or quantized
 *         integer type.  `false` otherwise.
 *
 * @see is_floating()
 * @see is_signed_integer()
 * @see is_unsigned_integer()
 */
bool is_integer(const Tensor *restrict input);

/**
 * @brief Check whether a tensor's dtype is a signed integer type
 *        (including quantized).
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   `NULL`.
 *
 * @return `true` if `input->dtype` is `Signed8`, `QSigned8`,
 *         `Signed32`, or `Signed64`.  `false` otherwise.
 *
 * @see is_unsigned_integer()
 * @see is_quantized_signed_integer()
 */
bool is_signed_integer(const Tensor *restrict input);

/**
 * @brief Check whether a tensor's dtype is an unsigned integer type
 *        (including quantized).
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   `NULL`.
 *
 * @return `true` if `input->dtype` is `UnSigned8`, `QUnSigned8`,
 *         `UnSigned32`, or `UnSigned64`.  `false` otherwise.
 *
 * @see is_signed_integer()
 * @see is_quantized_unsigned_integer()
 */
bool is_unsigned_integer(const Tensor *restrict input);

/**
 * @brief Check whether a tensor's dtype is a quantized signed
 *        integer type.
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   `NULL`.
 *
 * @return `true` if `input->dtype` is `QSigned8`.  `false`
 *         otherwise.
 *
 * @see is_quantized_unsigned_integer()
 * @see is_signed_integer()
 */
bool is_quantized_signed_integer(const Tensor *restrict input);

/**
 * @brief Check whether a tensor's dtype is a quantized unsigned
 *        integer type.
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   `NULL`.
 *
 * @return `true` if `input->dtype` is `QUnSigned8`.  `false`
 *         otherwise.
 *
 * @see is_quantized_signed_integer()
 * @see is_unsigned_integer()
 */
bool is_quantized_unsigned_integer(const Tensor *restrict input);

/* ────────────────────────────────────────────────────────────────
 *  Cast and size utilities
 * ──────────────────────────────────────────────────────────────── */

/**
 * @brief Cast a tensor's data to a different dtype.
 *
 * @details
 * Dispatches through the `cast_dispatch` table (defined in
 * @ref dtype.c) to select the correct element-wise conversion.
 * The destination tensor must be pre-allocated with the target
 * dtype and matching shape.
 *
 * @param[in]  src           Source tensor.  Must not be `NULL`.
 * @param[in]  target_dtype  Desired output @ref DType_.
 * @param[out] dst           Destination tensor (must be
 *                           pre-allocated).  Must not be `NULL`.
 *
 * @pre  @p dst must have been created via
 *       `create_unallocated_tensor()` with the correct shape.
 * @post On success, @p dst contains the type-converted copy of
 *       @p src.
 *
 * @see cast_dispatch
 * @see DType_
 */
void cast(const Tensor *restrict src, DType_ target_dtype,
          Tensor *restrict dst);

/**
 * @brief Return the size in bytes of a given @ref DType_.
 *
 * @details
 * Looks up the byte-width from the precomputed
 * `lookup_dtype_sizes` table indexed by @p dtype.
 *
 * @param[in] dtype  The data type to query.
 *
 * @return Size of one element of @p dtype in bytes.
 *
 * @see DType_
 */
size_t dtype_size(DType_ dtype);

#ifdef __cplusplus
}
#endif
