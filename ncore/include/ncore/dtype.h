#pragma once

/**
 * @file dtype.h
 * @brief Data type definitions for the NovaNN library.
 *
 * This header provides type aliases and an enumeration for representing
 * the various data types supported by the library, including floating-point
 * and integer types (both standard and quantized).
 */

#include <ncore/macros.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct Tensor;
typedef struct Tensor Tensor;

/**
 * @brief Internal type alias for 16-bit half-precision floating-point.
 */
typedef _Float16 nova_f16;

/**
 * @brief Internal type alias for brain floating point (16-bit).
 */
typedef __bf16 nova_bf16;

/**
 * @brief Internal type alias for 32-bit floating-point.
 */
typedef float nova_f32;

/**
 * @brief Internal type alias for 64-bit floating-point (double precision).
 */
typedef double nova_f64;

/**
 * @brief Internal type alias for signed 8-bit integer.
 */
typedef int8_t nova_s8;

/**
 * @brief Internal type alias for unsigned 8-bit integer.
 */
typedef uint8_t nova_u8;

/**
 * @brief Internal type alias for quantized signed 8-bit integer.
 */
typedef int8_t nova_qs8;

/**
 * @brief Internal type alias for quantized unsigned 8-bit integer.
 */
typedef uint8_t nova_qu8;

/**
 * @brief Internal type alias for signed 32-bit integer.
 */
typedef int32_t nova_s32;

/**
 * @brief Internal type alias for unsigned 32-bit integer.
 */
typedef uint32_t nova_u32;

/**
 * @brief Internal type alias for signed 64-bit integer.
 */
typedef int64_t nova_s64;

/**
 * @brief Internal type alias for unsigned 64-bit integer.
 */
typedef uint64_t nova_u64;

/**
 * @brief Public type alias for 32-bit floating-point.
 */
typedef nova_f32 float32;

/**
 * @brief Public type alias for 64-bit floating-point (double precision).
 */
typedef nova_f64 float64;

/**
 * @brief Public type alias for 16-bit half-precision floating-point.
 */
typedef nova_f16 float16;

/**
 * @brief Public type alias for brain floating point (16-bit).
 */
typedef nova_bf16 bfloat16;

/**
 * @brief Public type alias for signed 8-bit integer.
 */
typedef nova_s8 int8;

/**
 * @brief Public type alias for unsigned 8-bit integer.
 */
typedef nova_u8 uint8;

/**
 * @brief Public type alias for quantized signed 8-bit integer.
 */
typedef nova_qs8 qint8;

/**
 * @brief Public type alias for quantized unsigned 8-bit integer.
 */
typedef nova_qu8 quint8;

/**
 * @brief Public type alias for signed 32-bit integer.
 */
typedef nova_s32 int32;

/**
 * @brief Public type alias for signed 64-bit integer.
 */
typedef nova_s64 int64;

/**
 * @brief Public type alias for unsigned 32-bit integer.
 */
typedef nova_u32 uint32;

/**
 * @brief Public type alias for unsigned 64-bit integer.
 */
typedef nova_u64 uint64;

/**
 * @brief Enumeration of supported data types.
 *
 * Used throughout the library to identify the data type of tensors
 * and perform appropriate operations.
 */
typedef enum ATTR(packed) {
  Float32 = 0,     ///< 32-bit floating point
  Float64 = 1,     ///< 64-bit floating point
  Float16 = 2,     ///< 16-bit floating point (half precision)
  BFloat16 = 3,    ///< Brain floating point (16-bit)
  Signed8 = 4,     ///< Signed 8-bit integer (note: likely typo for Signed8)
  UnSigned8 = 5,   ///< Unsigned 8-bit integer
  QSigned8 = 6,    ///< Quantized signed 8-bit integer
  QUnSigned8 = 7,  ///< Quantized unsigned 8-bit integer
  Signed32 = 8,    ///< Signed 32-bit integer
  UnSigned32 = 9,  ///< Unsigned 32-bit integer
  Signed64 = 10,   ///< Signed 64-bit integer
  UnSigned64 = 11, ///< Unsigned 64-bit integer
} DType_;

/**
 * @brief Checks if the tensor's dtype is a floating-point type.
 * @param input Pointer to the input tensor.
 * @return true if the dtype is floating-point, false otherwise.
 */
bool is_floating(const Tensor *restrict input);

/**
 * @brief Checks if the tensor's dtype is an integer type.
 * @param input Pointer to the input tensor.
 * @return true if the dtype is an integer type, false otherwise.
 */
bool is_integer(const Tensor *restrict input);

/**
 * @brief Checks if the tensor's dtype is a signed integer type.
 * @param input Pointer to the input tensor.
 * @return true if the dtype is a signed integer, false otherwise.
 */
bool is_signed_integer(const Tensor *restrict input);

/**
 * @brief Checks if the tensor's dtype is an unsigned integer type.
 * @param input Pointer to the input tensor.
 * @return true if the dtype is an unsigned integer, false otherwise.
 */
bool is_unsigned_integer(const Tensor *restrict input);

/**
 * @brief Checks if the tensor's dtype is a quantized signed integer type.
 * @param input Pointer to the input tensor.
 * @return true if the dtype is quantized signed integer, false otherwise.
 */
bool is_quantized_signed_integer(const Tensor *restrict input);

/**
 * @brief Checks if the tensor's dtype is a quantized unsigned integer type.
 * @param input Pointer to the input tensor.
 * @return true if the dtype is quantized unsigned integer, false otherwise.
 */
bool is_quantized_unsigned_integer(const Tensor *restrict input);

/**
 * @brief Casts a tensor to the target dtype.
 *
 * Uses the dispatch table to call the appropriate cast function based
 * on the source and target data types.
 *
 * @param src Pointer to the source tensor.
 * @param target_dtype The desired output data type.
 * @param dst Pointer to the destination tensor (must be pre-allocated).
 */
void cast(const Tensor *restrict src, DType_ target_dtype,
          Tensor *restrict dst);

/**
 * @brief Returns the size in bytes of a given data type.
 *
 * Looks up the size from a precomputed lookup table indexed by DType_.
 *
 * @param dtype The data type to query.
 * @return size_t Size of the dtype in bytes.
 */
size_t dtype_size(DType_ dtype);

#ifdef __cplusplus
}
#endif
