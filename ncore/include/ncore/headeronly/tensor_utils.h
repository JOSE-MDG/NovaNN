/**
 * @file tensor_utils.h
 * @brief Inline tensor initialisation and coordinate utilities.
 *
 * Provides contiguous-stride computation, element-count calculation,
 * unallocated tensor factory helpers, and a row-major odometer for
 * multi-dimensional iteration.
 */

#pragma once

#include <ncore/dtype.h>
#include <ncore/macros.h>
#include <ncore/tensor.h>
#include <stdlib.h>
#include <string.h>

typedef size_t coords_t[NOVA_MAX_DIMS];

/**
 * @brief Compute per-dimension strides for a contiguous tensor.
 *
 * Stride[i] = item_size * product(shape[i+1..ndims-1]).
 * The last dimension always has stride == item_size.
 *
 * @param ten       Tensor whose strides[] will be written.
 * @param ndims     Number of dimensions.
 * @param shape     Dimension sizes.
 * @param item_size Element size in bytes.
 */
static inline void compute_tensor_strides_(Tensor *ten, size_t ndims,
                                           const shape_t shape,
                                           size_t item_size) {
  ten->strides[ndims - 1] = item_size;
  for (size_t dim = ndims - 1; dim-- > 0;) {
    ten->strides[dim] = ten->strides[dim + 1] * shape[dim + 1];
  }
}

/**
 * @brief Compute the total number of elements in a tensor.
 *
 * Iterates over all dimensions and multiplies their sizes together.
 * The result is stored in ten->size.  Assumes ten->ndims has already
 * been populated.
 *
 * @param ten   Tensor whose size will be set.
 * @param shape Dimension sizes (must have at least ten->ndims entries).
 */
static inline void compute_tensor_size_(Tensor *ten, const shape_t shape) {
  size_t size = 1;
  for (size_t dim = 0; dim < ten->ndims; dim++) {
    size *= shape[dim];
  }
  ten->size = size;
}

/**
 * @brief Compute per-dimension strides for a contiguous grad tensor.
 *
 * Stride[i] = item_size * product(shape[i+1..ndims-1]).
 * The last dimension always has stride == item_size.
 *
 * @param grad      TensorGrad whose strides[] will be written.
 * @param ndims     Number of dimensions.
 * @param shape     Dimension sizes.
 * @param item_size Element size in bytes.
 */
static inline void compute_grad_tensor_strides_(TensorGrad grad, size_t ndims,
                                                const shape_t shape,
                                                size_t item_size) {
  grad->strides[ndims - 1] = item_size;
  for (size_t dim = ndims - 1; dim-- > 0;) {
    grad->strides[dim] = grad->strides[dim + 1] * shape[dim + 1];
  }
}

/**
 * @brief Compute the total number of elements in a grad tensor.
 *
 * Iterates over all dimensions and multiplies their sizes together.
 * The result is stored in grad->size.  Assumes grad->ndims has already
 * been populated.
 *
 * @param grad  TensorGrad whose size will be set.
 * @param shape Dimension sizes (must have at least grad->ndims entries).
 */
static inline void compute_grad_tensor_size_(TensorGrad grad,
                                             const shape_t shape) {
  size_t size = 1;
  for (size_t dim = 0; dim < grad->ndims; dim++) {
    size *= shape[dim];
  }
  grad->size = size;
}

/**
 * @brief Create a tensor without allocating a data buffer.
 *
 * @param shape         Dimension sizes.
 * @param dtype         Element data type.
 * @param device        Target device.
 * @param requires_grad Whether to track gradients.
 * @param ndims         Number of dimensions.
 * @return Initialised Tensor with no backing storage.
 */
static inline Tensor create_unallocated_tensor(const shape_t shape,
                                               DType_ dtype, Device device,
                                               bool requires_grad,
                                               size_t ndims) {
  Tensor tensor = {0};
  memcpy(tensor.shape, shape, ndims * sizeof(size_t));
  tensor.dtype = dtype;
  tensor.device = device;
  tensor.ndims = ndims;
  tensor.item_size = dtype_size(dtype);
  tensor.offset = 0;
  tensor.grad = NULL;
  tensor.is_leaf_ = true;
  tensor.is_view_ = false;
  tensor.requires_grad_ = requires_grad;
  tensor.retain_grad_ = false;
  tensor.grad_fn_ = NULL;
  tensor.scale_ = 1.0F;
  tensor.zero_point_ = 0;
  tensor.storage = NULL;
  tensor.data.data = NULL;
  tensor.is_allocated_ = false;
  tensor.version_ = 0;
  compute_tensor_size_(&tensor, shape);
  compute_tensor_strides_(&tensor, ndims, shape, tensor.item_size);
  return tensor;
}

/**
 * @brief Create a heap-allocated unallocated grad tensor.
 *
 * @param shape  Dimension sizes.
 * @param dtype  Element data type.
 * @param device Target device.
 * @param ndims  Number of dimensions.
 * @return Pointer to newly allocated Tensor with no backing storage.
 */
static inline TensorGrad create_unallocated_grad_tensor(const shape_t shape,
                                                        DType_ dtype,
                                                        Device device,
                                                        size_t ndims) {
  TensorGrad grad = (TensorGrad)malloc(sizeof(Tensor));
  NOVA_INTERNAL_ASSERT(
      grad != NULL,
      "[GRAD] create_unallocated_grad_tensor: malloc returned NULL\n");
  *grad = create_unallocated_tensor(shape, dtype, device, false, ndims);
  return grad;
}

/**
 * @brief Create a scalar tensor without allocating a data buffer.
 *
 * @param dtype         Element data type.
 * @param device        Target device.
 * @param requires_grad Whether to track gradients.
 * @return Initialised scalar Tensor with no backing storage.
 */
static inline Tensor create_unallocated_scalar_tensor(DType_ dtype,
                                                      Device device,
                                                      bool requires_grad) {
  Tensor tensor = {0};
  tensor.shape[0] = 0;
  tensor.strides[0] = 0;
  tensor.size = 1;
  tensor.ndims = 0;
  tensor.dtype = dtype;
  tensor.device = device;
  tensor.item_size = dtype_size(dtype);
  tensor.offset = 0;
  tensor.grad = NULL;
  tensor.is_leaf_ = true;
  tensor.is_view_ = false;
  tensor.requires_grad_ = requires_grad;
  tensor.retain_grad_ = false;
  tensor.grad_fn_ = NULL;
  tensor.scale_ = 1.0F;
  tensor.zero_point_ = 0;
  tensor.storage = NULL;
  tensor.data.data = NULL;
  tensor.is_allocated_ = false;
  tensor.version_ = 0;
  return tensor;
}

/**
 * @brief Create a heap-allocated unallocated scalar grad tensor.
 *
 * @param dtype  Element data type.
 * @param device Target device.
 * @return Pointer to newly allocated scalar Tensor with no backing storage.
 */
static inline TensorGrad create_unallocated_scalar_grad_tensor(DType_ dtype,
                                                               Device device) {
  TensorGrad grad = (TensorGrad)malloc(sizeof(Tensor));
  NOVA_INTERNAL_ASSERT(
      grad != NULL,
      "[GRAD] create_unallocated_scalar_grad_tensor: malloc returned NULL\n");
  *grad = create_unallocated_scalar_tensor(dtype, device, false);
  return grad;
}

/**
 * @brief Advance a multi-dimensional coordinate by one step (row-major).
 *
 * Increments the last dimension (ndims-1) by one and propagates the
 * carry to earlier dimensions when a dimension reaches shape[dim],
 * resetting it to zero.  When all dimensions overflow the coordinate
 * wraps back to all zeros, matching the behaviour of a traditional
 * mechanical odometer.
 *
 * @param coords Current coordinates, updated in-place.
 * @param ndims  Number of dimensions.
 * @param shape  Dimension sizes (upper bound for each coordinate).
 */
static inline void odometer(coords_t coords, size_t ndims,
                            const shape_t shape) {
  for (size_t dim = ndims; dim-- > 0;) {
    coords[dim]++;
    if (coords[dim] < shape[dim]) {
      break;
    }
    coords[dim] = 0;
  }
}
