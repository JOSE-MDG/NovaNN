/**
 * @file dtype.c
 * @brief Implementation of dtype classification and casting functions.
 *
 * This file provides functions to check the properties of a Tensor's
 * data type using lookup tables defined in dtype_tables.h, as well as
 * a generic cast function that dispatches to the appropriate cast
 * implementation based on source and target dtypes.
 */

#include <ncore/dtype.h>
#include <ncore/headeronly/cast.h>
#include <ncore/tables/dtype_tables.h>
#include <ncore/tensor.h>

/**
 * @brief Dispatch table for cast functions.
 *
 * 2D array indexed by [source_dtype][target_dtype], containing
 * function pointers to the appropriate cast implementation.
 */
extern castFn cast_dispatch[NUM_DTYPES][NUM_DTYPES];

/**
 * @brief Checks if the tensor's dtype is a floating-point type.
 * @param input Pointer to the input tensor.
 * @return true if the dtype is floating-point, false otherwise.
 */
bool is_floating(const Tensor *restrict input) {
  return (bool)floating[input->dtype][0];
}

/**
 * @brief Checks if the tensor's dtype is an integer type.
 * @param input Pointer to the input tensor.
 * @return true if the dtype is an integer type, false otherwise.
 */
bool is_integer(const Tensor *restrict input) {
  return (bool)integer[input->dtype][0];
}

/**
 * @brief Checks if the tensor's dtype is a signed integer type.
 * @param input Pointer to the input tensor.
 * @return true if the dtype is a signed integer, false otherwise.
 */
bool is_signed_integer(const Tensor *restrict input) {
  return (bool)signed_integer[input->dtype][0];
}

/**
 * @brief Checks if the tensor's dtype is an unsigned integer type.
 * @param input Pointer to the input tensor.
 * @return true if the dtype is an unsigned integer, false otherwise.
 */
bool is_unsigned_integer(const Tensor *restrict input) {
  return (bool)unsigned_integer[input->dtype][0];
}

/**
 * @brief Checks if the tensor's dtype is a quantized signed integer type.
 * @param input Pointer to the input tensor.
 * @return true if the dtype is quantized signed integer, false otherwise.
 */
bool is_quantized_signed_integer(const Tensor *restrict input) {
  return (bool)quantized_signed_integer[input->dtype][0];
}

/**
 * @brief Checks if the tensor's dtype is a quantized unsigned integer type.
 * @param input Pointer to the input tensor.
 * @return true if the dtype is quantized unsigned integer, false otherwise.
 */
bool is_quantized_unsigned_integer(const Tensor *restrict input) {
  return (bool)quantized_unsigned_integer[input->dtype][0];
}

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
          Tensor *restrict dst) {
  castFn func = cast_dispatch[src->dtype][target_dtype];
  func(src, dst);
}

/**
 * @brief Returns the size in bytes of a given data type.
 *
 * @param dtype The data type to query.
 * @return size_t Size of the dtype in bytes.
 */
size_t dtype_size(DType_ dtype) { return lookup_dtype_sizes[dtype]; }
