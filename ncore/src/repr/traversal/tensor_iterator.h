/**
 * @file tensor_iterator.h
 * @brief Public interface for the multidimensional strided tensor iterator.
 *
 * @details
 * This header defines the @ref TensorIterator structure and its associated
 * control functions. The iterator provides a high-level abstraction for
 * traversing tensors of any physical layout (contiguous or strided) in a
 * consistent row-major order.
 *
 * It is primarily used by layout renderers to decouple the visual formatting
 * logic from the complexities of strided memory access.
 *
 * ## Architecture
 * - **Stateful Iteration**: The iterator encapsulates the current position
 *   using a coordinate vector and a linear element counter.
 * - **Stride Transparency**: Callers retrieve memory offsets via
 *   @ref iter_byte_offset(), hiding the internal arithmetic involving
 *   strides and offsets.
 *
 * @see tensor_iterator.c Implementation details.
 * @see tensor_utils.h Underlying coordinate arithmetic.
 */

#pragma once

#include <ncore/headeronly/tensor_utils.h>
#include <ncore/tensor.h>
#include <stdbool.h>
#include <stddef.h>

/**
 * @struct TensorIterator
 * @brief State descriptor for multidimensional tensor traversal.
 *
 * @details
 * Tracks the current position within an n-dimensional tensor. The iterator
 * is considered "done" when all elements in the logical shape have been
 * visited.
 */
typedef struct {
  const Tensor *tensor; ///< Pointer to the tensor being traversed.
  coords_t coords;      ///< Current multidimensional coordinate vector.
  size_t linear_index;  ///< Current logical element count (0 to size-1).
  bool done;            ///< Termination flag (true when iteration complete).
} TensorIterator;

/**
 * @brief Initialise an iterator instance for a specific tensor.
 *
 * @param[out] it  Pointer to the iterator to initialise.
 * @param[in]  ten Pointer to the tensor to traverse.
 */
void iter_init(TensorIterator *it, const Tensor *ten);

/**
 * @brief Advance the iterator to the next logical element.
 *
 * @param[in,out] it Pointer to the active iterator.
 */
void iter_advance(TensorIterator *it);

/**
 * @brief Compute the byte offset of the current element in memory.
 *
 * @param[in] it Pointer to the active iterator.
 * @return Byte offset from the tensor's base data pointer.
 */
size_t iter_byte_offset(const TensorIterator *it);

/**
 * @brief Query whether the iterator has reached the end of the tensor.
 *
 * @param[in] it Pointer to the iterator.
 * @return true if all logical elements have been visited.
 */
bool iter_done(const TensorIterator *it);
