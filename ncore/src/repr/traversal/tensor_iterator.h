/**
 * @file tensor_iterator.h
 * @brief N-dimensional tensor iterator wrapping the odometer pattern.
 *
 * @details
 * Walks every element of a tensor in row-major order regardless of
 * contiguity by updating a multi-dimensional coordinate vector and
 * computing byte offsets from the tensor's strides array.
 *
 * Byte-offset formula:
 *   offset = ten->offset + sum(coords[d] * ten->strides[d])
 *
 * @see iter_byte_offset()  Compute the offset from current coordinates.
 */

#pragma once

#include <ncore/headeronly/tensor_utils.h>
#include <ncore/tensor.h>
#include <stdbool.h>
#include <stddef.h>

/**
 * @struct TensorIterator
 * @brief Iterator state.
 */
typedef struct {
  const Tensor *tensor; ///< The tensor being iterated.
  coords_t coords;      ///< Current multi-dimensional coordinate.
  size_t linear_index;  ///< Element count from start.
  bool done;            ///< True when all elements have been visited.
} TensorIterator;

/**
 * @brief Initialise an iterator at the first element.
 *
 * @param[out] it  Uninitialised struct.
 * @param[in]  ten Tensor to iterate.
 */
void iter_init(TensorIterator *it, const Tensor *ten);

/**
 * @brief Advance to the next element in row-major order.
 *
 * @param[in] it Iterator to advance.
 */
void iter_advance(TensorIterator *it);

/**
 * @brief Compute the byte offset of the current element.
 *
 * @param[in] it Iterator.
 * @return Byte offset into the tensor's data buffer.
 */
size_t iter_byte_offset(const TensorIterator *it);

/**
 * @brief Check whether iteration is complete.
 *
 * @param[in] it Iterator.
 * @return true if all elements have been visited.
 */
bool iter_done(const TensorIterator *it);
