/**
 * @file dtype.h
 * @brief Data-type definitions and classification for the NovaNN library.
 *
 * @details
 * This header provides:
 *
 * @li Type aliases — Portable names for the numeric types used by
 *   tensor storage (@c float32, @c int8, @c qint8, etc.).  Each alias
 *   maps to a standard C type.
 * @li DType_ enumeration — A packed enum that identifies a data
 *   type at run time, used for dispatch tables and tensor metadata.
 * @li Classification functions — @c is_floating(), @c is_integer(),
 *   etc. that test a tensor's dtype against category lookup tables.
 * @li Cast and size utilities — @c cast() for type conversion and
 *   @c dtype_size() for byte-width queries.
 *
 * @section type-alias-convention Type Alias Convention
 *
 * Public aliases use lowercase names (@c float32, @c int8, …) without
 * a prefix.  They are defined as direct typedefs to the underlying
 * C types and are stable across platforms.
 *
 * @see macros.h    NOVA_INTERNAL_ASSERT macro.
 * @see tensor.h    Tensor struct embedding a @ref DType_ field.
 * @see storage.h   data_ptr union using these types.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <ncore/headeronly/macros.h>

#ifdef __cplusplus
extern "C" {
#endif

struct Tensor;
typedef struct Tensor Tensor;

/** @brief 32-bit IEEE 754 single-precision float. */
typedef float float32;

/** @brief 64-bit IEEE 754 double-precision float. */
typedef double float64;

#if defined(__GNUC__) || defined(__clang__)
/** @brief 16-bit IEEE 754 half-precision float (compiler extension). */
typedef _Float16 float16;

/** @brief Brain Floating Point 16-bit (compiler extension). */
typedef __bf16 bfloat16;
#else
/** @brief 16-bit IEEE 754 half-precision float (native implementation). */
typedef unsigned short float16;

/** @brief Brain Floating Point 16-bit (native implementation). */
typedef unsigned short bfloat16;
#endif

/** @brief Floating Point 8-bit (FP8 E4M3FN native implementation). */
typedef uint8_t float8_e4m3fn;

/** @brief Floating Point 8-bit (FP8 E5M2 native implementation). */
typedef uint8_t float8_e5m2;

/** @brief Floating Point 4-bit (FP4 E2M1FN packed-pair native
   implementation). */
typedef uint8_t float4_e2m1fn_x2;

/** @brief Signed 8-bit two's-complement integer. */
typedef int8_t int8;

/** @brief Unsigned 8-bit integer. */
typedef uint8_t uint8;

/** @brief Quantized signed 8-bit integer (int8 storage). */
typedef int8_t qint8;

/** @brief Quantized unsigned 8-bit integer (uint8 storage). */
typedef uint8_t quint8;

/** @brief Signed 16-bit two's-complement integer. */
typedef int16_t int16;

/** @brief Unsigned 16-bit integer. */
typedef uint16_t uint16;

/** @brief Quantized signed 16-bit integer (int16 storage). */
typedef int16_t qint16;

/** @brief Quantized unsigned 16-bit integer (uint16 storage). */
typedef uint16_t quint16;

/** @brief Signed 32-bit two's-complement integer. */
typedef int32_t int32;

/** @brief Unsigned 32-bit integer. */
typedef uint32_t uint32;

/** @brief Quantized Signed 32-bit two's-complement integer (int32 storage). */
typedef int32_t qint32;

/** @brief Quantized unsigned 32-bit integer (uint32 storage). */
typedef uint32_t quint32;

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
 * @c cast_dispatch, @c lookup_dtype_sizes).
 *
 * @li @c 0 — @c Float32 — @c float — 4 bytes
 * @li @c 1 — @c Float64 — @c double — 8 bytes
 * @li @c 2 — @c Float16 — @c uint16_t — 2 bytes
 * @li @c 3 — @c BFloat16 — @c uint16_t — 2 bytes
 * @li @c 4 — @c Float8E4M3fn — @c uint8_t — 1 byte
 * @li @c 5 — @c Float8E5M2 — @c uint8_t — 1 byte
 * @li @c 6 — @c Float4E2M1fn — @c uint8_t — 1 byte
 * @li @c 7 — @c Signed8 — @c int8_t — 1 byte
 * @li @c 8 — @c UnSigned8 — @c uint8_t — 1 byte
 * @li @c 9 — @c QSigned8 — @c int8_t — 1 byte
 * @li @c 10 — @c QUnSigned8 — @c uint8_t — 1 byte
 * @li @c 11 — @c Signed16 — @c int16_t — 2 bytes
 * @li @c 12 — @c UnSigned16 — @c uint16_t — 2 bytes
 * @li @c 13 — @c QSigned16 — @c int16_t — 2 bytes
 * @li @c 14 — @c QUnSigned16 — @c uint16_t — 2 bytes
 * @li @c 15 — @c Signed32 — @c int32_t — 4 bytes
 * @li @c 16 — @c UnSigned32 — @c uint32_t — 4 bytes
 * @li @c 17 — @c QSigned32 — @c int32_t — 4 bytes
 * @li @c 18 — @c QUnSigned32 — @c uint32_t — 4 bytes
 * @li @c 19 — @c Signed64 — @c int64_t — 8 bytes
 * @li @c 20 — @c UnSigned64 — @c uint64_t — 8 bytes
 *
 * @note This enum uses a @c uint8_t underlying type (C23) to minimise its
 *       footprint in structs that are serialised or copied
 *       frequently.
 *
 * @see dtype_size()     Byte-width lookup.
 * @see is_floating()    Classification helpers.
 * @see cast()           Type conversion.
 */
typedef enum DType_ : uint8_t {
  Float32,      ///< 32-bit floating point.
  Float64,      ///< 64-bit floating point (double precision).
  Float16,      ///< 16-bit floating point (half precision).
  BFloat16,     ///< Brain floating point (16-bit).
  Float8E4M3fn, ///< 8-bit floating point E4M3fn
  Float8E5M2,   ///< 8-bit floating point E5M2
  Float4E2M1fn, ///< 4-bit packed-pair floating point E2M1
  Signed8,      ///< Signed 8-bit integer.
  UnSigned8,    ///< Unsigned 8-bit integer.
  QSigned8,     ///< Quantized signed 8-bit integer.
  QUnSigned8,   ///< Quantized unsigned 8-bit integer.
  Signed16,     ///< Signed 16-bit integer.
  UnSigned16,   ///< Unsigned 16-bit integer.
  QSigned16,    ///< Quantized signed 16-bit integer.
  QUnSigned16,  ///< Quantized unsigned 16-bit integer.
  Signed32,     ///< Signed 32-bit integer.
  UnSigned32,   ///< Unsigned 32-bit integer.
  QSigned32,    ///< Quantized signed 32-bit integer.
  QUnSigned32,  ///< Quantized unsigned 32-bit integer.
  Signed64,     ///< Signed 64-bit integer.
  UnSigned64,   ///< Unsigned 64-bit integer.
} DType_;

/**
 * @brief Check whether a tensor's dtype is a floating-point type.
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   @c nullptr.
 *
 * @return @c true if @c input->dtype is @c Float32, @c Float64,
 *         @c Float16, @c BFloat16, @c Float8E4M3fn, @c Float8E5M2,
 *         or @c Float4E2M1fn.  @c false otherwise.
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
 *                   @c nullptr.
 *
 * @return @c true if @c input->dtype is any integer or quantized
 *         integer type.  @c false otherwise.
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
 *                   @c nullptr.
 *
 * @return @c true if @c input->dtype is @c Signed8, @c QSigned8,
 *         @c Signed16, @c QSigned16, @c Signed32, @c QSigned32,
 *         or @c Signed64.  @c false otherwise.
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
 *                   @c nullptr.
 *
 * @return @c true if @c input->dtype is @c UnSigned8, @c QUnSigned8,
 *         @c UnSigned16, @c QUnSigned16, @c UnSigned32,
 *         @c QUnSigned32, or @c UnSigned64.  @c false otherwise.
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
 *                   @c nullptr.
 *
 * @return @c true if @c input->dtype is @c QSigned8, @c QSigned16,
 *         or @c QSigned32.  @c false otherwise.
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
 *                   @c nullptr.
 *
 * @return @c true if @c input->dtype is @c QUnSigned8,
 *         @c QUnSigned16, or @c QUnSigned32.  @c false otherwise.
 *
 * @see is_quantized_signed_integer()
 * @see is_unsigned_integer()
 */
bool is_quantized_unsigned_integer(const Tensor *restrict input);

/**
 * @brief Check whether a given @ref DType_ can be quantized.
 *
 * @details
 * Some data types are inherently quantized (e.g., @c QSigned8, @c QSigned16,
 * @c QUnSigned8), while others represent full-precision values.  This function
 * reports whether a type is eligible to participate in quantization operations.
 *
 * @param[in] dtype  The data type to query.
 *
 * @return @c true if @p dtype is a quantizable type (@c Float4E2M1fn,
 *         @c QSigned8, @c QUnSigned8, @c QSigned16, @c QUnSigned16,
 *         @c QSigned32, @c QUnSigned32).  @c false otherwise.
 *
 * @see is_floating()
 * @see is_integer()
 */
bool is_quantizable_dtype(DType_ dtype);

/* ────────────────────────────────────────────────────────────────
 *  Cast and size utilities
 * ──────────────────────────────────────────────────────────────── */

/**
 * @brief Cast a tensor's data to a different dtype.
 *
 * @details
 * Dispatches through the @c cast_dispatch table (defined in
 * @ref dtype.c) to select the correct element-wise conversion.
 * The destination tensor must be pre-allocated with the target
 * dtype and matching shape.
 *
 * @param[in]  src           Source tensor.  Must not be @c nullptr.
 * @param[in]  target_dtype  Desired output @ref DType_.
 * @param[out] dst           Destination tensor (must be
 *                           pre-allocated).  Must not be @c nullptr.
 *
 * @pre  @p dst must have been created via
 *       @c create_unallocated_tensor() with the correct shape.
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
 * @c lookup_dtype_sizes table indexed by @p dtype.
 *
 * @param[in] dtype  The data type to query.
 *
 * @return Size of one element of @p dtype in bytes.
 *
 * @see DType_
 */
size_t dtype_size(DType_ dtype);

/**
 * @brief Return the packing factor of a given @ref DType_.
 *
 * @details
 * For most types the packing factor is 1 (one logical element per
 * storage unit).  For packed types like @ref Float4E2M1fn the factor
 * is 2, because each storage byte holds two logical elements.
 *
 * @param[in] dtype  The data type to query.
 *
 * @return Number of logical elements packed into one storage unit.
 */
/**
 * @brief Return the packing factor of a given @ref DType_.
 *
 * @details
 * For most types the packing factor is 1 (one logical element per
 * storage unit).  For packed types like @ref Float4E2M1fn the factor
 * is 2, because each storage byte holds two logical elements.
 *
 * @param[in] dtype  The data type to query.
 *
 * @return Number of logical elements packed into one storage unit.
 *
 * @see dtype_size()
 * @see DType_
 */
size_t dtype_packing_factor(DType_ dtype);

#ifdef __cplusplus
}
#endif
