/**
 * @file tensor.c
 * @brief Backend implementation of Tensor creation, views, and lifecycle.
 *
 * @details
 * Implements the full Tensor API: allocation through the Rust FFI allocator
 * (reserve/retain/release), strided layout computation, view sharing with
 * reference-counted storage, META-tensor special cases, and recursive
 * cleanup of gradient sub-graphs.
 */

#include <assert.h>
#include <ncore/alloc.h>
#include <ncore/dtype.h>
#include <ncore/headeronly/tensor_utils.h>
#include <ncore/macros.h>
#include <ncore/storage.h>
#include <ncore/tensor.h>
#include <string.h>

/**
 * @brief Create a fully allocated tensor.
 *
 * Zero-initialises the Tensor struct, copies shape metadata, computes
 * size and strides, allocates a data buffer via allocate_tensor_buffer(),
 * and optionally creates an unallocated gradient tensor.
 *
 * @param shape         Dimension sizes.
 * @param dtype         Element data type.
 * @param device        Target device (CPU, GPU, or META).
 * @param requires_grad Whether to track gradients.
 * @param ndims         Number of dimensions.
 * @return Initialised Tensor (value type, heap-allocated storage).
 */
Tensor create_tensor(const shape_t shape, DType_ dtype, Device device,
                     bool requires_grad, size_t ndims) {

  Tensor tensor = {0};

  memcpy(tensor.shape, shape, ndims * sizeof(size_t));
  tensor.dtype = dtype;
  tensor.device = device;
  tensor.ndims = ndims;
  tensor.item_size = dtype_size(dtype);
  tensor.offset = 0;
  tensor.is_leaf_ = true;
  tensor.is_view_ = false;
  tensor.requires_grad_ = requires_grad;
  tensor.retain_grad_ = false;
  tensor.grad_fn_ = NULL;
  tensor.scale_ = 1.0F;
  tensor.zero_point_ = 0;
  tensor.version_ = 0;
  compute_tensor_size_(&tensor, shape);
  compute_tensor_strides_(&tensor, ndims, shape, tensor.item_size);

  TensorStorage *storage =
      allocate_tensor_buffer(tensor.item_size * tensor.size, tensor.device);
  if (device == DEVICE_META) {
    NOVA_INTERNAL_ASSERT(storage == NULL,
                         "[STORAGE] create_tensor: META tensor must have NULL "
                         "storage, but non-NULL storage was returned\n");
    tensor.storage = NULL;
    tensor.data.data = NULL;
    tensor.is_allocated_ = false;
    if (requires_grad) {
      tensor.grad = create_unallocated_grad_tensor(shape, dtype, device, ndims);
    }
    return tensor;
  }
  NOVA_INTERNAL_ASSERT(storage != NULL,
                       "[STORAGE] create_tensor: CPU/GPU tensor must have "
                       "non-NULL storage, but allocation returned NULL\n");
  tensor.storage = storage;
  tensor.data = tensor.storage->ptr;
  tensor.is_allocated_ = true;

  if (requires_grad) {
    tensor.grad = create_unallocated_grad_tensor(shape, dtype, device, ndims);
  }

  return tensor;
}

/**
 * @brief Create a fully allocated 0-dimensional (scalar) tensor.
 *
 * Allocates a single-element data buffer on the specified device.
 * Shape and strides are zeroed and ndims is set to 0.
 *
 * @param dtype         Element data type.
 * @param device        Target device (CPU, GPU, or META).
 * @param requires_grad Whether to track gradients.
 * @return Initialised scalar Tensor with backing storage.
 */
Tensor create_scalar_tensor(DType_ dtype, Device device, bool requires_grad) {
  Tensor tensor = {0};

  tensor.shape[0] = 0;
  tensor.strides[0] = 0;
  tensor.size = 1;
  tensor.dtype = dtype;
  tensor.device = device;
  tensor.ndims = 0;
  tensor.item_size = dtype_size(dtype);
  tensor.offset = 0;
  tensor.is_leaf_ = true;
  tensor.is_view_ = false;
  tensor.requires_grad_ = requires_grad;
  tensor.retain_grad_ = false;
  tensor.grad_fn_ = NULL;
  tensor.scale_ = 1.0F;
  tensor.zero_point_ = 0;
  tensor.version_ = 0;

  TensorStorage *storage =
      allocate_tensor_buffer(tensor.item_size * tensor.size, tensor.device);
  if (device == DEVICE_META) {
    NOVA_INTERNAL_ASSERT(storage == NULL,
                         "[STORAGE] create_tensor: META tensor must have NULL "
                         "storage, but non-NULL storage was returned\n");
    tensor.storage = NULL;
    tensor.data.data = NULL;
    tensor.is_allocated_ = false;
    if (requires_grad) {
      tensor.grad = create_unallocated_scalar_grad_tensor(dtype, device);
    }
    return tensor;
  }
  NOVA_INTERNAL_ASSERT(storage != NULL,
                       "[STORAGE] create_tensor: CPU/GPU tensor must have "
                       "non-NULL storage, but allocation returned NULL\n");
  tensor.storage = storage;
  tensor.data = tensor.storage->ptr;
  tensor.is_allocated_ = true;

  if (requires_grad) {
    tensor.grad = create_unallocated_scalar_grad_tensor(dtype, device);
  }

  return tensor;
}

/**
 * @brief Create a view of an existing tensor with a new shape.
 *
 * Shares the same underlying storage (incrementing the Rust-side
 * reference count), recomputes strides for the new shape, and marks
 * the resulting tensor as a non-leaf view.  If the source has a
 * gradient, an unallocated gradient tensor is created for the view.
 *
 * @param src       Source tensor to view (must outlive the view).
 * @param new_shape New dimension sizes.  Product must equal src->size.
 * @param new_ndims Number of dimensions in the new shape.
 * @return View tensor sharing src's storage.
 */
Tensor create_view(const Tensor *restrict src, const shape_t new_shape,
                   size_t new_ndims) {

  Tensor dst = *src;

  // Increase rust reference counter
  retain(&dst.storage->handle);

  // Copy tensor metadata
  dst.ndims = new_ndims;
  memcpy(dst.shape, new_shape, new_ndims * sizeof(size_t));
  compute_tensor_strides_(&dst, dst.ndims, dst.shape, dst.item_size);

  dst.is_view_ = true;
  dst.is_leaf_ = false;

  if (src->grad != NULL) {
    dst.grad = create_unallocated_grad_tensor(new_shape, src->grad->dtype,
                                              src->device, new_ndims);
  }

  return dst;
}

/**
 * @brief Check whether a tensor's data buffer is contiguous in memory.
 *
 * A tensor is contiguous when the strides are strictly decreasing by
 * a factor of shape[dim] for each dimension, meaning elements are
 * stored in row-major order with no gaps.
 *
 * @param ten Tensor to check.
 * @return true if the tensor is contiguous, false otherwise.
 */
bool is_contiguous(const Tensor *restrict ten) {
  size_t expected = ten->item_size;
  for (int dim = (int)ten->ndims - 1; dim >= 0; dim--) {
    if (ten->strides[dim] != expected) {
      return false;
    }
    expected *= ten->strides[dim];
  }
  return true;
}

/**
 * @brief Move ownership of tensor resources from src to dst.
 *
 * Collects any existing resources in dst, then transfers the contents
 * of src into dst.  src is zeroed (storage, data, grad, grad_fn_ are
 * set to NULL / false) so that a subsequent collect() on src is a
 * no-op.
 *
 * @param dst Destination tensor (previous resources are freed).
 * @param src Source tensor (ownership is transferred; src becomes a
 *            hollow shell).
 */
void move_tensor(Tensor *restrict dst, Tensor *restrict src) {
  collect(dst);

  *dst = *src;
  src->storage = NULL;
  src->data.data = NULL;
  src->grad = NULL;
  src->grad_fn_ = NULL;
  src->is_allocated_ = false;
}

/**
 * @brief Recursively release tensor memory.
 *
 * Decrements the reference count of the tensor's storage via release().
 * When the reference count reaches zero the storage is freed.  The
 * gradient sub-graph is then traversed and freed recursively.
 *
 * @param ten Tensor to collect (may be NULL, in which case this is a
 *            no-op).
 */
void collect(Tensor *ten) {
  if (ten == NULL) {
    return;
  }

  if (ten->storage != NULL) {
    bool should_free = release(&ten->storage->handle);
    if (should_free) {
      free(ten->storage);
      ten->storage = NULL;
      ten->data.data = NULL;
      ten->is_allocated_ = false;
    }
  }

  if (ten->grad != NULL) {
    collect(ten->grad);
    free(ten->grad);
    ten->grad = NULL;
  }
}

/**
 * @brief Check whether a tensor is 0-dimensional (scalar).
 *
 * A tensor is a scalar when ndims is 0, shape[0] and strides[0] are 0,
 * and total size is 1 (single element).
 *
 * @param ten Tensor to check.
 * @return true if the tensor is a scalar, false otherwise.
 */
bool is_scalar(const Tensor *ten) {
  return (bool)(ten->shape[0] == 0 && ten->strides[0] == 0 && ten->size == 1 &&
                ten->ndims == 0);
}

/**
 * @brief Check whether a gradient tensor is 0-dimensional (scalar).
 *
 * @param grad Gradient tensor to check.
 * @return true if the gradient is a scalar, false otherwise.
 */
bool is_scalar_grad(TensorGrad grad) {
  return (bool)(grad->shape[0] == 0 && grad->strides[0] == 0 &&
                grad->size == 1 && grad->ndims == 0);
}

/**
 * @brief Check whether a tensor's data buffer is 64-byte aligned.
 *
 * 64-byte alignment is required for optimal SIMD vectorization.
 *
 * @param ten Tensor to check.
 * @return true if the data pointer is 64-byte aligned, false otherwise.
 * @pre ten->storage must not be NULL.
 */
bool is_aligned(const Tensor *ten) {
  return (bool)(((uintptr_t)ten->storage->ptr.v % 64) == 0);
}

/**
 * @brief Check whether a gradient tensor's data buffer is 64-byte aligned.
 *
 * @param grad Gradient tensor to check.
 * @return true if the gradient data pointer is 64-byte aligned, false
 *         otherwise.
 * @pre grad->storage must not be NULL.
 */
bool is_grad_aligned(TensorGrad grad) {
  return (bool)(((uintptr_t)grad->storage->ptr.v % 64) == 0);
}

/**
 * @brief Check whether a tensor has been collected (freed).
 *
 * A tensor is considered collected when both its storage and data pointer
 * are NULL, typically after a call to collect().
 *
 * @param ten Tensor to check.
 * @return true if the tensor has been collected, false otherwise.
 */
bool is_collected(const Tensor *ten) {
  return (bool)(ten->storage == NULL && ten->data.data == NULL);
}

/**
 * @brief Check whether a gradient tensor has been collected (freed).
 *
 * @param grad Gradient tensor to check.
 * @return true if the gradient has been collected, false otherwise.
 */
bool is_grad_collected(TensorGrad grad) {
  return (bool)(grad->storage == NULL && grad->data.data == NULL);
}
