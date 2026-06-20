/**
 * @file tensor_iterator.c
 * @brief Implementation of the multidimensional strided tensor iterator.
 *
 * @details
 * This module provides the logic for traversing tensors element-by-element
 * in row-major order, regardless of their physical memory layout. It wraps
 * the low-level odometer pattern to provide a clean state-based interface
 * for layout renderers and other modules needing to visit every element
 * of a view.
 *
 * ## Architecture
 * - **Odometer Logic**: The iterator maintains a coordinate vector that is
 *   incremented using the @ref odometer() algorithm, ensuring correct
 *   carry propagation across dimensions.
 * - **Stride Mapping**: At each step, the iterator translates the current
 *   coordinate vector into a linear byte offset using the tensor's
 *   stride array.
 * - **State Management**: The @ref TensorIterator struct tracks the linear
 *   element count and a termination flag (`done`) to simplify iteration loops.
 *
 * @see repr/traversal/tensor_iterator.h Public descriptor and API.
 * @see ncore/headeronly/tensor_utils.h Underlying coordinate arithmetic.
 */

#include <ncore/headeronly/tensor_utils.h>
#include <string.h>

#include "tensor_iterator.h"

/**
 * @brief Initialise a new iterator to the first element of a tensor.
 *
 * @param[out] it  Pointer to the uninitialised iterator state.
 * @param[in]  ten Pointer to the tensor to traverse.
 */
void iter_init(TensorIterator *it, const Tensor *ten) {
  it->tensor = ten;
  memset(it->coords, 0, sizeof(it->coords));
  it->linear_index = 0;
  it->done = (ten->size == 0);
}

/**
 * @brief Advance the iterator to the next logical element.
 *
 * @details
 * Increments the internal coordinate vector and checks for termination.
 * If the end of the tensor is reached, the `done` flag is set.
 */
void iter_advance(TensorIterator *it) {
  if (it->done) {
    return;
  }
  it->linear_index++;
  if (it->linear_index >= it->tensor->size) {
    it->done = true;
    return;
  }
  odometer(it->coords, it->tensor->ndims, it->tensor->shape);
}

/**
 * @brief Compute the current element's memory offset.
 *
 * @param[in] it Pointer to the active iterator.
 * @return Byte offset from the start of the tensor's data pointer.
 */
size_t iter_byte_offset(const TensorIterator *it) {
  size_t off = compute_linear_byte_offset(it->coords, it->tensor->ndims,
                                          it->tensor->strides);
  return off;
}

/**
 * @brief Check if the iteration has been completed.
 *
 * @return true if all elements have been visited.
 */
bool iter_done(const TensorIterator *it) { return it->done; }
