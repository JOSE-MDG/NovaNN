/**
 * @file tensor.c
 * @brief Backend implementation of Tensor creation, views, and lifecycle.
 *
 * @details
 * Implements the full Tensor API: allocation through the Rust FFI allocator
 * (reserve/retain/release), strided layout computation, view sharing with
 * reference-counted storage, META-tensor special cases, and recursive
 * cleanup of gradient sub-graphs.
 *
 * ## Implementation Notes
 *
 * - All creation functions zero-initialise the `Tensor` struct before
 *   populating fields — no stale data is possible.
 * - The Rust allocator (`allocate_tensor_buffer`) is called exactly once
 *   per tensor; for `DEVICE_META` tensors, the allocator is never invoked
 *   and storage is left `NULL`.
 * - Views increment the Rust reference count via `retain()` and decrement
 *   it when `collect()` is called on the view — no double-free is possible.
 * - `collect()` is safe to call with `NULL` (no-op).
 *
 * @see tensor.h   Public API declarations and struct documentation.
 * @see alloc.h    allocate_tensor_buffer().
 * @see storage.h  retain() / release() reference-count operations.
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
 * @brief Create a fully allocated n-dimensional tensor.
 *
 * @details
 * 1. Zero-initialises a `Tensor` struct.
 * 2. Copies `shape` (only the first `ndims` entries).
 * 3. Computes `size` via `compute_tensor_size_()`.
 * 4. Computes `strides` via `compute_tensor_strides_()` (row-major).
 * 5. Calls `allocate_tensor_buffer()` with the total byte count.
 * 6. For `DEVICE_META`: asserts storage is NULL and marks the tensor
 *    as unallocated.  Returns immediately.
 * 7. For `DEVICE_CPU` / `DEVICE_GPU`: asserts storage is non-NULL,
 *    sets `is_allocated_ = true`, and populates `data` from
 *    `storage->ptr`.
 * 8. If `requires_grad`, creates an unallocated gradient tensor.
 *
 * @param[in]  shape         Dimension sizes (only the first `ndims`
 *                           entries are read).
 * @param[in]  dtype         Element data type (@ref DType_).
 * @param[in]  device        Target device (`DEVICE_CPU`,
 *                           `DEVICE_GPU`, or `DEVICE_META`).
 * @param[in]  requires_grad If `true`, creates an unallocated
 *                           gradient tensor in `grad`.
 * @param[in]  pin_memory    If `true`, request page-locked host
 *                           memory (CPU only).
 * @param[in]  ndims         Number of dimensions.  Must be <=
 *                           @ref NOVA_MAX_DIMS.
 *
 * @return Initialised `Tensor` with valid backing storage, or a
 *         META tensor with NULL storage.
 *
 * @pre  `ndims` must not exceed `NOVA_MAX_DIMS`.
 * @pre  `product(shape[0..ndims])` must be > 0 for non-META.
 * @post On success, `is_allocated_ == true` (unless META).
 * @post If `requires_grad`, `grad` points to an unallocated tensor.
 *
 * @see create_scalar_tensor()   Scalar (0-D) variant.
 * @see create_tensor_like()     Clone metadata from an existing tensor.
 * @see allocate_tensor_buffer() Underlying allocation.
 */
Tensor create_tensor(const shape_t shape, DType_ dtype, Device device,
                     bool requires_grad, bool pin_memory, size_t ndims) {

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
  tensor.is_pinned = pin_memory;
  compute_tensor_size_(&tensor, shape);
  compute_tensor_strides_(&tensor, ndims, shape, tensor.item_size);

  TensorStorage *storage = allocate_tensor_buffer(
      tensor.item_size * tensor.size, tensor.device, pin_memory);
  if (device == DEVICE_META) {
    NOVA_INTERNAL_ASSERT(storage == NULL,
                         "[STORAGE] create_tensor: META tensor must have NULL "
                         "storage, but non-NULL storage was returned\n");
    tensor.storage = NULL;
    tensor.data.data = NULL;
    tensor.is_allocated_ = false;
    if (requires_grad) {
      tensor.grad = create_unallocated_grad_tensor(shape, dtype, device,
                                                   pin_memory, ndims);
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
    tensor.grad = create_unallocated_grad_tensor(shape, dtype, device,
                                                  pin_memory, ndims);
  }

  return tensor;
}

/**
 * @brief Create a fully allocated 0-dimensional (scalar) tensor.
 *
 * @details
 * Allocates a single-element data buffer.  Shape and strides are
 * zeroed, `ndims` is set to 0, and `size` is set to 1.
 * The implementation mirrors `create_tensor()` but bypasses the
 * `compute_tensor_size_()` / `compute_tensor_strides_()` calls
 * since scalars have a trivial layout.
 *
 * @param[in]  dtype         Element data type (@ref DType_).
 * @param[in]  device        Target device (`DEVICE_CPU`,
 *                           `DEVICE_GPU`, or `DEVICE_META`).
 * @param[in]  requires_grad If `true`, creates an unallocated
 *                           scalar gradient tensor.
 * @param[in]  pin_memory    If `true`, request page-locked host
 *                           memory (CPU only).
 *
 * @return Initialised scalar `Tensor` with backing storage.
 *
 * @pre  Device must be one of `DEVICE_CPU`, `DEVICE_GPU`, or
 *       `DEVICE_META`.
 * @post `is_allocated_ == true` (unless META).
 *
 * @see create_tensor()  N-dimensional variant.
 * @see is_scalar()      Query predicate.
 */
Tensor create_scalar_tensor(DType_ dtype, Device device, bool requires_grad,
                            bool pin_memory) {
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
  tensor.is_pinned = pin_memory;

  TensorStorage *storage = allocate_tensor_buffer(
      tensor.item_size * tensor.size, tensor.device, pin_memory);
  if (device == DEVICE_META) {
    NOVA_INTERNAL_ASSERT(storage == NULL,
                         "[STORAGE] create_tensor: META tensor must have NULL "
                         "storage, but non-NULL storage was returned\n");
    tensor.storage = NULL;
    tensor.data.data = NULL;
    tensor.is_allocated_ = false;
    if (requires_grad) {
      tensor.grad =
          create_unallocated_scalar_grad_tensor(dtype, device, pin_memory);
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
    tensor.grad =
        create_unallocated_scalar_grad_tensor(dtype, device, pin_memory);
  }

  return tensor;
}

/**
 * @brief Create a tensor with the same shape, dtype, and device as
 *        another.
 *
 * @details
 * Inspects the source tensor and produces a new tensor with
 * identical metadata:
 * - If `is_scalar(ten)` is true, delegates to
 *   `create_scalar_tensor()` or `create_unallocated_scalar_tensor()`.
 * - Otherwise delegates to `create_tensor()` or
 *   `create_unallocated_tensor()`.
 *
 * The new tensor is independent of the source — it owns its own
 * storage and does not share the source's buffer.
 *
 * @param[in] ten  Source tensor to copy metadata from.  Must not
 *                 be `NULL`.
 *
 * @return New `Tensor` with matching metadata and allocation state.
 *
 * @pre  @p ten must not be `NULL`.
 * @post The returned tensor has the same shape, dtype, device,
 *       requires_grad, and is_pinned as @p ten.
 * @post The returned tensor owns its own storage (not shared).
 *
 * @see create_tensor()           Allocated variant.
 * @see create_unallocated_tensor() Unallocated variant.
 */
Tensor create_tensor_like(const Tensor *ten) {
  Tensor tensor = {0};
  if (is_scalar(ten)) {
    tensor = ((int)is_allocated(ten))
                 ? create_scalar_tensor(ten->dtype, ten->device,
                                        ten->requires_grad_, ten->is_pinned)
                 : create_unallocated_scalar_tensor(ten->dtype, ten->device,
                                                    ten->requires_grad_,
                                                    ten->is_pinned);
  } else {
    tensor =
        ((int)is_allocated(ten))
            ? create_tensor(ten->shape, ten->dtype, ten->device,
                            ten->requires_grad_, ten->is_pinned, ten->ndims)
            : create_unallocated_tensor(ten->shape, ten->dtype, ten->device,
                                        ten->requires_grad_, ten->is_pinned,
                                        ten->ndims);
  }
  return tensor;
}

/**
 * @brief Create a view of an existing tensor with a new shape.
 *
 * @details
 * 1. Shallow-copy the source `Tensor` into `dst`.
 * 2. Increment the Rust reference count via `retain()`.
 * 3. Overwrite `ndims` and `shape` with the new values.
 * 4. Recompute `strides` for the new shape via
 *    `compute_tensor_strides_()`.
 * 5. Mark `is_view_ = true` and `is_leaf_ = false`.
 * 6. If the source has a gradient, create an unallocated gradient
 *    tensor for the view (preserving the gradient dtype).
 *
 * The returned view shares the source's storage — no data copy is
 * performed.  The source must outlive the view.
 *
 * @param[in]  src        Source tensor to view.  Must have non-NULL
 *                        storage.  Must outlive the view.
 * @param[in]  new_shape  New dimension sizes.  Product must equal
 *                        `src->size`.
 * @param[in]  new_ndims  Number of dimensions in @p new_shape.
 *
 * @return View `Tensor` sharing @p src's storage.
 *
 * @pre  `product(new_shape[0..new_ndims])` must equal `src->size`.
 * @pre  `src->storage` must not be `NULL`.
 * @post The returned tensor has `is_view_ = true` and
 *       `is_leaf_ = false`.
 * @post The Rust reference count is incremented by one.
 * @post If `src->grad != NULL`, the view has an unallocated gradient.
 *
 * @see retain()   Increments the storage reference count.
 * @see collect()  Decrements it (and may free storage).
 * @see is_view()  Query predicate.
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
                                               src->device, src->is_pinned,
                                               new_ndims);
  }

  return dst;
}

/**
 * @brief Check whether a tensor's data buffer is contiguous in
 *        memory.
 *
 * @details
 * A tensor is contiguous when elements are stored in row-major
 * order without gaps.  This is verified by checking that each
 * stride matches the expected value:
 *
 * ```
 * expected = item_size
 * for dim = ndims-1 .. 0:
 *     if strides[dim] != expected: not contiguous
 *     expected *= shape[dim]
 * ```
 *
 * For a scalar (`ndims == 0`), the function returns `true`
 * immediately.
 *
 * @param[in] ten  Tensor to check.  Must not be `NULL`.
 *
 * @return `true` if the tensor is contiguous, `false` otherwise.
 *
 * @see strides_t    Stride array.
 * @see create_view()  Views may break contiguity.
 */
bool is_contiguous(const Tensor *restrict ten) {
  if (is_scalar(ten)) {
    return true;
  }
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
 * @details
 * 1. Frees any existing resources in `dst` via `collect()`.
 * 2. Bitwise-copies `src` into `dst`.
 * 3. Zeroes out `src` (storage, data, grad, grad_fn_ set to NULL;
 *    `is_allocated_` and `is_pinned` set to `false`) so that a
 *    subsequent `collect()` on `src` is a no-op.
 *
 * This is a move semantics wrapper — after the call, only `dst`
 * owns the resources.
 *
 * @param[in,out] dst  Destination tensor (previous resources are
 *                     freed via `collect()`).
 * @param[in,out] src  Source tensor (ownership transferred; `src`
 *                     becomes a hollow shell).
 *
 * @pre  @p dst and @p src must not be `NULL`.
 * @post @p dst owns all resources previously held by @p src.
 * @post @p src is in a valid but unallocated state.
 *
 * @see collect()  Called on `dst` before the move.
 */
void move_tensor(Tensor *restrict dst, Tensor *restrict src) {
  collect(dst);

  *dst = *src;
  src->storage = NULL;
  src->data.data = NULL;
  src->grad = NULL;
  src->grad_fn_ = NULL;
  src->is_allocated_ = false;
  src->is_pinned = false;
}

/**
 * @brief Recursively release tensor memory and gradients.
 *
 * @details
 * 1. If `ten` is `NULL`, returns immediately (no-op).
 * 2. If `ten->storage` is non-NULL, calls `release()` to
 *    decrement the Rust reference count.
 * 3. If `release()` returns `true` (count reached zero), frees
 *    the `TensorStorage` with `free()` and nullifies `storage`,
 *    `data`, and `is_allocated_`.
 * 4. If `ten->grad` is non-NULL, recursively calls `collect()`
 *    on the gradient, then frees the gradient `Tensor` with
 *    `free()` and nullifies `grad`.
 *
 * This ensures the full gradient sub-graph is freed, not just
 * the top-level tensor.
 *
 * @param[in,out] ten  Tensor to collect.  May be `NULL`.
 *
 * @post `ten->storage` reference count is decremented.
 * @post If the count reaches zero, `storage` and `data` are set
 *       to NULL and `is_allocated_` to `false`.
 * @post The gradient sub-graph is recursively freed.
 *
 * @see release()       Decrements the Rust reference count.
 * @see is_collected()  Query predicate after collection.
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
 * @details
 * A tensor is a scalar when all four conditions hold:
 * - `ndims == 0`
 * - `shape[0] == 0`
 * - `strides[0] == 0`
 * - `size == 1`
 *
 * This is the invariant established by `create_scalar_tensor()`.
 *
 * @param[in] ten  Tensor to check.  Must not be `NULL`.
 *
 * @return `true` if the tensor is a scalar, `false` otherwise.
 *
 * @see is_scalar_grad()        Gradient variant.
 * @see create_scalar_tensor()  Constructor for scalars.
 */
bool is_scalar(const Tensor *ten) {
  return (bool)(ten->shape[0] == 0 && ten->strides[0] == 0 && ten->size == 1 &&
                ten->ndims == 0);
}

/**
 * @brief Check whether a gradient tensor is 0-dimensional (scalar).
 *
 * @details
 * Same logic as `is_scalar()` but operates on a `TensorGrad`
 * (pointer-to-`Tensor`).
 *
 * @param[in] grad  Gradient tensor to check.  May be `NULL`.
 *
 * @return `true` if the gradient is a scalar, `false` otherwise
 *         (including when @p grad is `NULL`).
 *
 * @see is_scalar()  Tensor variant.
 */
bool is_scalar_grad(TensorGrad grad) {
  return (bool)(grad->shape[0] == 0 && grad->strides[0] == 0 &&
                grad->size == 1 && grad->ndims == 0);
}

/**
 * @brief Check whether a tensor's data buffer is properly aligned.
 *
 * @details
 * Alignment requirements differ by device:
 * - **GPU** (non-pinned): 512-byte alignment.
 * - **CPU** (pinned): 64-byte alignment.
 *
 * The check reads `ten->is_pinned` to select the threshold and
 * tests `ten->storage->ptr.v` modulo the threshold.
 *
 * @param[in] ten  Tensor to check.  Must not be `NULL`.
 *
 * @return `true` if the data pointer meets the alignment
 *         requirement, `false` otherwise.
 *
 * @pre  `ten->storage` must not be `NULL`.
 *
 * @see is_grad_aligned()  Gradient variant.
 */
bool is_aligned(const Tensor *ten) {
  return (bool)(!ten->is_pinned ? (((uintptr_t)ten->storage->ptr.v % 512) ==
                                   0) // Aligned by default to 512 bytes (GPU)
                                : (((uintptr_t)ten->storage->ptr.v % 64) ==
                                   0)); // Aligned by default to 64 bytes (CPU)
}

/**
 * @brief Check whether a gradient tensor's data buffer is properly
 *        aligned.
 *
 * @details
 * Same alignment logic as `is_aligned()` — 512-byte for GPU, 64-byte
 * for CPU.
 *
 * @param[in] grad  Gradient tensor to check.  Must not be `NULL`.
 *
 * @return `true` if the gradient data pointer meets the alignment
 *         requirement, `false` otherwise.
 *
 * @pre  `grad->storage` must not be `NULL`.
 *
 * @see is_aligned()  Tensor variant.
 */
bool is_grad_aligned(TensorGrad grad) {
  return (!grad->is_pinned ? (((uintptr_t)grad->storage->ptr.v % 512) ==
                              0) // Aligned by default to 512 bytes (GPU)
                           : (((uintptr_t)grad->storage->ptr.v % 64) == 0)) !=
         0; // Aligned by default to 64 bytes (CPU)
}

/**
 * @brief Check whether a tensor has been collected (freed).
 *
 * @details
 * A tensor is considered collected when all three conditions hold:
 * - `data.data == NULL`
 * - `storage == NULL`
 * - `is_allocated_ == false`
 *
 * This is the state after `collect()` has fully released the
 * tensor's storage.
 *
 * @param[in] ten  Tensor to check.  Must not be `NULL`.
 *
 * @return `true` if the tensor has been collected, `false`
 *         otherwise.
 *
 * @see collect()
 * @see is_grad_collected()  Gradient variant.
 */
bool is_collected(const Tensor *ten) {
  return (bool)(ten->data.data == NULL && ten->storage == NULL &&
                !ten->is_allocated_);
}

/**
 * @brief Check whether a gradient tensor has been collected.
 *
 * @details
 * If @p grad is `NULL`, returns `true` (a NULL gradient is
 * logically "collected").  Otherwise, checks the same three
 * conditions as `is_collected()`.
 *
 * @param[in] grad  Gradient tensor to check.  May be `NULL`.
 *
 * @return `true` if the gradient has been collected (or is
 *         `NULL`), `false` otherwise.
 *
 * @see is_collected()  Tensor variant.
 */
bool is_grad_collected(TensorGrad grad) {
  if (grad != NULL) {
    return (bool)(grad->data.data == NULL && grad->storage == NULL &&
                  !grad->is_allocated_);
  }
  return true;
}

/**
 * @brief Check whether a tensor's data buffer has been allocated.
 *
 * @details
 * Returns `true` when both conditions hold:
 * - `is_allocated_ == true`
 * - `data.data != NULL`
 * - `storage != NULL`
 *
 * This double-check ensures the tensor was both marked as
 * allocated and actually has a valid pointer.
 *
 * @param[in] ten  Tensor to check.  Must not be `NULL`.
 *
 * @return `true` if the tensor is allocated, `false` otherwise.
 *
 * @see is_grad_allocated()  Gradient variant.
 * @see is_collected()       Inverse check.
 */
bool is_allocated(const Tensor *ten) {
  return (bool)(ten->is_allocated_ && ten->storage != NULL &&
                ten->data.data != NULL);
}

/**
 * @brief Check whether a gradient tensor has been allocated.
 *
 * @details
 * If @p grad is `NULL`, returns `false`.  Otherwise, checks the
 * same two conditions as `is_allocated()`.
 *
 * @param[in] grad  Gradient tensor to check.  May be `NULL`.
 *
 * @return `true` if the gradient has a valid backing buffer,
 *         `false` otherwise (including when @p grad is `NULL`).
 *
 * @see is_allocated()  Tensor variant.
 */
bool is_grad_allocated(TensorGrad grad) {
  if (grad != NULL) {
    return (bool)(grad->is_allocated_ && grad->storage != NULL &&
                  grad->data.data != NULL);
  }
  return false;
}
