/**
 * @file dtype.h
 * @brief Data-type aliases and classification for the NovaNN library.
 *
 * Public aliases use lowercase names (@c float32, @c int8, ...) and
 * map straight to standard C types.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>

#ifdef __cplusplus
extern "C" {
#endif

struct Tensor;                ///< Forward declaration of the Tensor struct.
typedef struct Tensor Tensor; ///< Type alias for the Tensor struct.

/** @brief 32-bit IEEE 754 single-precision float. */
typedef float float32;

/** @brief 64-bit IEEE 754 double-precision float. */
typedef double float64;

#if defined(_GNUC_CLANG_)
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
 * implementation). */
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
 * @brief Cast function pointer type.
 * @param[in]  src Source tensor (read-only)
 * @param[out] dst Destination tensor (write-only, must be same size as src)
 */
typedef void (*CastFn)(const Tensor *restrict, Tensor *restrict,
                       novaStatus_t *status);

/**
 * @enum DType_
 * @brief Runtime data-type identifier, sequential from 0 so it indexes
 *        directly into dispatch tables (@c cast_dispatch,
 *        @c lookup_dtype_sizes). Backed by @c uint8_t (C23).
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
 * @return @c true when @c input->dtype is one of the floating types,
 *         @c false otherwise.
 */
bool is_floating(const Tensor *restrict input);

/**
 * @brief Check whether a tensor's dtype is a (non-quantized) integer type
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   @c nullptr.
 *
 * @return @c true when @c input->dtype is a non-quantized signed or
 *         unsigned integer, @c false otherwise.
 */
bool is_integer(const Tensor *restrict input);

/**
 * @brief Check whether a tensor's dtype is a signed integer type
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   @c nullptr.
 * @return @c true for @c Signed8/16/32/64, @c false otherwise.
 */
bool is_signed_integer(const Tensor *restrict input);

/**
 * @brief Check whether a tensor's dtype is an unsigned integer type
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   @c nullptr.
 * @return @c true for @c UnSigned8/16/32/64, @c false otherwise.
 */
bool is_unsigned_integer(const Tensor *restrict input);

/**
 * @brief Check whether a tensor's dtype is a quantized signed
 *        integer type.
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   @c nullptr.
 * @return @c true for @c QSigned8/16/32, @c false otherwise.
 */
bool is_quantized_signed_integer(const Tensor *restrict input);

/**
 * @brief Check whether a tensor's dtype is a quantized unsigned
 *        integer type.
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   @c nullptr.
 * @return @c true for @c QUnSigned8/16/32, @c false otherwise.
 */
bool is_quantized_unsigned_integer(const Tensor *restrict input);

/**
 * @brief Check whether a given @ref DType_ can be quantized.
 *
 * @param[in] dtype  The data type to query.
 *
 * @return @c true for quantized types and @ref Float4E2M1fn,
 *         @c false otherwise.
 */
bool is_quantizable_dtype(DType_ dtype);

/* Cast and size utilities */

/**
 * @brief Cast a tensor's data to a different dtype.
 *
 * Tensors on the compute device dispatch to the GPU casting kernels,
 * which report their own launch status. All other tensors dispatch
 * through the @c cast_dispatch table (defined in @ref dtype.c).  The
 * destination tensor must be pre-allocated with the target dtype and
 * matching shape.
 *
 * @param[in]  src           Source tensor.  Must not be @c nullptr.
 * @param[out] dst           Destination tensor (must be
 *                           pre-allocated).  Must not be @c nullptr.
 * @param[in]  target_dtype  Desired output @ref DType_.
 *
 * @return The selected kernel's status with @p dst holding the
 *         type-converted copy on success, an error status otherwise
 *         (unsupported pair or a GPU launch error).
 *
 * @pre  @p dst must have been created via
 *       @c create_unallocated_tensor() with the correct shape.
 */
novaStatus_t cast(const Tensor *restrict src, Tensor *restrict dst,
                  DType_ target_dtype);

/**
 * @brief Return the @ref CastFn registered for a (source, target)
 *        dtype pair.
 *
 * Looks up @c cast_dispatch[src_dtype][target_dtype] without
 * executing anything. Identity pairs and unsupported pairs store
 * @c nullptr, so check the pointer before invoking.
 *
 * @param[in] src_dtype    Source data type.
 * @param[in] target_dtype Destination data type.
 * @return The stored kernel, or @c nullptr.
 */
CastFn get_dispatched_cast_func(DType_ src_dtype, DType_ target_dtype);

/**
 * @brief Return the size in bytes of a given @ref DType_.
 *
 * Looked up from the @c lookup_dtype_sizes table.
 *
 * @param[in] dtype  The data type to query.
 * @return Size of one element of @p dtype in bytes.
 */
size_t dtype_size(DType_ dtype);

/**
 * @brief Return the packing factor of a given @ref DType_.
 *
 * 1 for types where one storage unit holds one element, 2 for packed
 * types like @ref Float4E2M1fn (two elements per byte).
 *
 * @param[in] dtype  The data type to query.
 * @return Number of logical elements packed into one storage unit.
 */
size_t dtype_packing_factor(DType_ dtype);

#ifdef __cplusplus
}
#endif
