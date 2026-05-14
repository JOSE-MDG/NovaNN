/**
 * @file tensor.c
 * @brief Implementation of tensor creation, unallocation, and collection.
 *
 * Provides factory functions for allocating and initialising Tensor
 * instances (with or without a data buffer), gradient-tensor creation,
 * and recursive memory reclamation via collect().
 */

#include <ncore/alloc.h>
#include <ncore/dtype.h>
#include <ncore/macros.h>
#include <ncore/storage.h>
#include <ncore/tables/dtype_tables.h>
#include <ncore/tensor.h>
#include <string.h>

/* =========================================================================
 * Internal helpers
 * ========================================================================= */

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
  strides_t strides;

  ten->strides[ndims - 1] = item_size;
  for (size_t dim = ndims - 1; dim-- > 0;) {
    strides[dim] = ten->strides[dim + 1] * ten->shape[dim + 1];
  }

  memcpy(ten->strides, strides, ndims * sizeof(size_t));
}

/**
 * @brief Compute the total number of elements in a tensor.
 *
 * The result is the product of all dimension sizes and is stored in
 * ten->size.
 *
 * @param ten   Tensor whose size will be set.
 * @param shape Dimension sizes.
 */
static inline void compute_tensor_size_(Tensor *ten, const shape_t shape) {

  size_t size = 1;
  for (size_t dim = 0; dim < ten->ndims; dim++) {
    size *= shape[dim];
  }

  ten->size = size;
}

/* =========================================================================
 * Public API
 * ========================================================================= */

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
  tensor.item_size = lookup_dtype_sizes[dtype];
  tensor.offset = 0;
  tensor.is_leaf_ = true;
  tensor.is_view_ = false;
  tensor.requires_grad_ = requires_grad;
  tensor.retain_grad_ = false;
  tensor.grad_fn_ = NULL;
  tensor.scale_ = 1.0F;
  tensor.zero_point_ = 0;
  compute_tensor_size_(&tensor, shape);
  compute_tensor_strides_(&tensor, ndims, shape, tensor.item_size);

  TensorStorage *storage =
      allocate_tensor_buffer(tensor.item_size * tensor.size, tensor.device);
  if (device == DEVICE_META) {
    NOVA_INTERNAL_ASSERT(storage == NULL,
                         "[STORAGE] create_tensor: META tensor must have NULL "
                         "storage, but non-NULL storage was returned\n")
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
                       "non-NULL storage, but allocation returned NULL\n")
  tensor.storage = storage;
  tensor.data = tensor.storage->ptr;
  tensor.is_allocated_ = true;

  if (requires_grad) {
    tensor.grad = create_unallocated_grad_tensor(shape, dtype, device, ndims);
  }

  return tensor;
}

/**
 * @brief Create a tensor without allocating a data buffer.
 *
 * Initialises all metadata fields and sets storage / data to NULL.
 * Useful as a destination for deepcopy() or external buffer injection.
 *
 * @param shape         Dimension sizes.
 * @param dtype         Element data type.
 * @param device        Target device.
 * @param requires_grad Whether to track gradients.
 * @param ndims         Number of dimensions.
 * @return Initialised Tensor with no backing storage.
 */
Tensor create_unallocated_tensor(const shape_t shape, DType_ dtype,
                                 Device device, bool requires_grad,
                                 size_t ndims) {

  Tensor tensor = {0};

  memcpy(tensor.shape, shape, ndims * sizeof(size_t));
  tensor.dtype = dtype;
  tensor.device = device;
  tensor.ndims = ndims;
  tensor.item_size = lookup_dtype_sizes[dtype];
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
  compute_tensor_size_(&tensor, shape);
  compute_tensor_strides_(&tensor, ndims, shape, tensor.item_size);
  return tensor;
}

/**
 * @brief Create a heap-allocated gradient tensor with no data buffer.
 *
 * Allocates a Tensor on the heap via malloc() and initialises it as an
 * unallocated tensor with requires_grad_ = false.  Must be freed by the
 * caller (typically via collect()).
 *
 * @param shape  Dimension sizes.
 * @param dtype  Element data type.
 * @param device Target device.
 * @param ndims  Number of dimensions.
 * @return Pointer to a newly allocated, zero-initialised Tensor, or
 *         aborts on allocation failure.
 */
TensorGrad create_unallocated_grad_tensor(const shape_t shape, DType_ dtype,
                                          Device device, size_t ndims) {

  TensorGrad grad = (TensorGrad)malloc(sizeof(Tensor));

  NOVA_INTERNAL_ASSERT(
      grad != NULL,
      "[GRAD] create_unallocated_grad_tensor: malloc returned NULL\n")

  memcpy(grad->shape, shape, ndims * sizeof(size_t));
  grad->dtype = dtype;
  grad->device = device;
  grad->ndims = ndims;
  grad->item_size = lookup_dtype_sizes[dtype];
  grad->offset = 0;
  grad->grad = NULL;
  grad->is_leaf_ = true;
  grad->is_view_ = false;
  grad->requires_grad_ = false;
  grad->retain_grad_ = false;
  grad->grad_fn_ = NULL;
  grad->scale_ = 1.0F;
  grad->zero_point_ = 0;
  grad->storage = NULL;
  grad->data.data = NULL;
  grad->is_allocated_ = false;
  compute_tensor_size_(grad, shape);
  compute_tensor_strides_(grad, ndims, shape, grad->item_size);

  return grad;
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
    }
  }

  if (ten->grad != NULL) {
    collect(ten->grad);
    free(ten->grad);
    ten->grad = NULL;
  }
}
