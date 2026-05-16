/**
 * @file tensor_iterator.c
 * @brief Implementation of the odometer-based tensor iterator.
 *
 * @details
 * Walks through every element of a tensor in row-major order using the
 * odometer() function from tensor_utils.h.  Byte offsets are computed
 * from the current multi-dimensional coordinate and the tensor's stride
 * array, so the iterator works correctly on both contiguous and strided
 * (view) tensors.
 */

#include "tensor_iterator.h"
#include <ncore/headeronly/tensor_utils.h>
#include <string.h>

void iter_init(TensorIterator *it, const Tensor *ten) {
  it->tensor = ten;
  memset(it->coords, 0, sizeof(it->coords));
  it->linear_index = 0;
  it->done = (ten->size == 0);
}

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

size_t iter_byte_offset(const TensorIterator *it) {
  size_t off = it->tensor->offset;
  for (size_t dim = 0; dim < it->tensor->ndims; dim++) {
    off += it->coords[dim] * it->tensor->strides[dim];
  }
  return off;
}

bool iter_done(const TensorIterator *it) { return it->done; }
