/**
 * @file tensor.c
 * @brief Backend implementation of Tensor creation, views, and lifecycle.
 *
 * @details
 * Implements the full Tensor API: allocation through @ref safe_allocator(),
 * strided layout computation, view sharing with reference-counted storage,
 * META-tensor special cases, and recursive cleanup of gradient sub-graphs.
 *
 * All creation and mutation functions receive an output
 * @ref novaStatus_t pointer.  On failure the returned tensor is zeroed
 * and the caller must not use it.
 *
 * @section implementation-notes Implementation Notes
 *
 * @li All creation functions zero-initialise the @c Tensor struct before
 *   populating fields — no stale data is possible.
 * @li @c safe_allocator() is the single allocation entry point for CPU and
 *   GPU tensors; for @c DEVICE_META tensors it is never invoked and
 *   storage is left @c nullptr.
 * @li Views increment the Rust reference count via @c retain() and decrement
 *   it when @c collect() is called on the view — no double-free is possible.
 * @li @c collect() is safe to call with @c nullptr (no-op).
 *
 * @see tensor.h   Public API declarations and struct documentation.
 * @see alloc.h    safe_allocator().
 * @see storage.h  retain() / release() reference-count operations.
 */

#include <ncore/core/alloc.h>
#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/core/storage.h>
#include <ncore/headeronly/macros.h>
#include <ncore/headeronly/tensor_utils.h>
#include <ncore/tensor.h>
#include <string.h>

/**
 * @brief Create a fully allocated n-dimensional tensor.
 *
 * @details
 * @li 1. Zero-initialises a @c Tensor struct.
 * @li 2. Copies @c shape (only the first @c ndims entries).
 * @li 3. Computes @c size via @c compute_tensor_size_().
 * @li 4. Computes @c strides via @c compute_tensor_strides_() (row-major).
 * @li 5. For @c DEVICE_META: marks the tensor as unallocated and returns
 *    immediately (the allocator is never invoked).
 * @li 6. For @c DEVICE_CPU / @c DEVICE_GPU: calls @c safe_allocator() to
 *    allocate the data buffer.  On failure the tensor is zeroed.
 * @li 7. If @c requires_grad, creates an unallocated gradient tensor.
 *    On failure, releases any already-allocated storage via
 *    @c collect() and zeroes the tensor.
 *
 * @param[in]  shape         Dimension sizes (only the first @c ndims
 *                           entries are read).
 * @param[in]  dtype         Element data type (@ref DType_).
 * @param[in]  device        Target device (@c DEVICE_CPU,
 *                           @c DEVICE_GPU, or @c DEVICE_META).
 * @param[in]  requires_grad If @c true, creates an unallocated
 *                           gradient tensor in @c grad.
 * @param[in]  pin_memory    If @c true, request page-locked host
 *                           memory (CPU only).
 * @param[in]  ndims         Number of dimensions.  Must be <=
 *                           @ref NOVA_MAX_DIMS.
 * @param[out] status        Receives the operation result.
 *
 * @return Initialised @c Tensor with valid backing storage, or a
 *         META tensor with nullptr storage.
 *
 * @pre  @c ndims must not exceed @c NOVA_MAX_DIMS.
 * @pre  @c product(shape[0..ndims]) must be > 0 for non-META.
 * @pre  @p status must not be @c nullptr.
 * @post On success, @c is_allocated_ == true (unless META).
 * @post If @c requires_grad, @c grad points to an unallocated tensor.
 *
 * @see create_scalar_tensor()   Scalar (0-D) variant.
 * @see create_tensor_like()     Clone metadata from an existing tensor.
 * @see safe_allocator()         Underlying allocation.
 */
Tensor create_tensor(const shape_t shape, DType_ dtype, Device_ device,
                     bool requires_grad, bool pin_memory, size_t ndims,
                     novaStatus_t *status) {

  Tensor tensor = {};
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
  tensor.grad_fn_ = nullptr;
  tensor.scale_ = 1.0F;
  tensor.zero_point_ = 0;
  tensor.version_ = 0;
  tensor.is_pinned = pin_memory;
  compute_tensor_size_(&tensor, shape);
  compute_tensor_strides_(&tensor, ndims, shape, tensor.item_size);

  if (device == DEVICE_META) {
    tensor.storage = nullptr;
    tensor.data.data = nullptr;
    tensor.is_allocated_ = false;
    if (requires_grad) {
      tensor.grad = create_unallocated_grad_tensor(shape, dtype, device,
                                                   pin_memory, ndims, status);
      if (status->err != novaSuccess) {
        memset(&tensor, 0, sizeof(Tensor));
        return tensor;
      }
    }
    return tensor;
  }

  *status = safe_allocator(tensor.size * tensor.item_size, device, pin_memory,
                           nullptr, &tensor, true);

  if (status->err != novaSuccess) {
    memset(&tensor, 0, sizeof(Tensor));
    return tensor;
  }

  if (requires_grad) {
    tensor.grad = create_unallocated_grad_tensor(shape, dtype, device,
                                                 pin_memory, ndims, status);
    if (status->err != novaSuccess) {
      collect(&tensor);
      memset(&tensor, 0, sizeof(Tensor));
      return tensor;
    }
  }

  return tensor;
}

/**
 * @brief Create a fully allocated 0-dimensional (scalar) tensor.
 *
 * @details
 * Allocates a single-element data buffer.  Shape and strides are
 * zeroed, @c ndims is set to 0, and @c size is set to 1.
 * The implementation mirrors @c create_tensor() but bypasses the
 * @c compute_tensor_size_() / @c compute_tensor_strides_() calls
 * since scalars have a trivial layout.
 *
 * For @c DEVICE_META, storage is left @c nullptr and @c is_allocated_ is
 * set to @c false.  On failure the tensor is zeroed and the error
 * is reported through @p status.
 *
 * @param[in]  dtype         Element data type (@ref DType_).
 * @param[in]  device        Target device (@c DEVICE_CPU,
 *                           @c DEVICE_GPU, or @c DEVICE_META).
 * @param[in]  requires_grad If @c true, creates an unallocated
 *                           scalar gradient tensor.
 * @param[in]  pin_memory    If @c true, request page-locked host
 *                           memory (CPU only).
 * @param[out] status        Receives the operation result.
 *
 * @return Initialised scalar @c Tensor with backing storage.
 *
 * @pre  Device_ must be one of @c DEVICE_CPU, @c DEVICE_GPU, or
 *       @c DEVICE_META.
 * @pre  @p status must not be @c nullptr.
 * @post On success, @c is_allocated_ == true (unless META).
 *
 * @see create_tensor()  N-dimensional variant.
 * @see is_scalar()      Query predicate.
 */
Tensor create_scalar_tensor(DType_ dtype, Device_ device, bool requires_grad,
                            bool pin_memory, novaStatus_t *status) {
  Tensor tensor = {};

  tensor.shape[0] = 0;
  tensor.strides[0] = 0;
  tensor.size = 1;
  tensor.logical_size = 1;
  tensor.dtype = dtype;
  tensor.device = device;
  tensor.ndims = 0;
  tensor.item_size = dtype_size(dtype);
  tensor.offset = 0;
  tensor.is_leaf_ = true;
  tensor.is_view_ = false;
  tensor.requires_grad_ = requires_grad;
  tensor.retain_grad_ = false;
  tensor.grad_fn_ = nullptr;
  tensor.scale_ = 1.0F;
  tensor.zero_point_ = 0;
  tensor.version_ = 0;
  tensor.is_pinned = pin_memory;

  if (device == DEVICE_META) {
    tensor.storage = nullptr;
    tensor.data.data = nullptr;
    tensor.is_allocated_ = false;
    if (requires_grad) {
      tensor.grad = create_unallocated_scalar_grad_tensor(dtype, device,
                                                          pin_memory, status);
      if (status->err != novaSuccess) {
        memset(&tensor, 0, sizeof(Tensor));
        return tensor;
      }
    }
    return tensor;
  }

  *status = safe_allocator(tensor.size * tensor.item_size, device, pin_memory,
                           nullptr, &tensor, true);

  if (status->err != novaSuccess) {
    memset(&tensor, 0, sizeof(Tensor));
    return tensor;
  }

  if (requires_grad) {
    tensor.grad = create_unallocated_scalar_grad_tensor(dtype, device,
                                                        pin_memory, status);
    if (status->err != novaSuccess) {
      collect(&tensor);
      memset(&tensor, 0, sizeof(Tensor));
      return tensor;
    }
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
 * @li If @c is_scalar(ten) is true, delegates to
 *   @c create_scalar_tensor() or @c create_unallocated_scalar_tensor().
 * @li Otherwise delegates to @c create_tensor() or
 *   @c create_unallocated_tensor().
 *
 * The new tensor is independent of the source — it owns its own
 * storage and does not share the source's buffer.
 *
 * @param[in]  ten    Source tensor to copy metadata from.  Must not
 *                    be @c nullptr.
 * @param[out] status Receives the operation result.
 *
 * @return New @c Tensor with matching metadata and allocation state.
 *
 * @pre  @p ten must not be @c nullptr.
 * @pre  @p status must not be @c nullptr.
 * @post The returned tensor has the same shape, dtype, device,
 *       requires_grad, and is_pinned as @p ten.
 * @post The returned tensor owns its own storage (not shared).
 *
 * @see create_tensor()           Allocated variant.
 * @see create_unallocated_tensor() Unallocated variant.
 */
Tensor create_tensor_like(const Tensor *ten, novaStatus_t *status) {
  Tensor tensor = {};
  if (is_scalar(ten)) {
    tensor =
        ((int)is_allocated(ten))
            ? create_scalar_tensor(ten->dtype, ten->device, ten->requires_grad_,
                                   ten->is_pinned, status)
            : create_unallocated_scalar_tensor(ten->dtype, ten->device,
                                               ten->requires_grad_,
                                               ten->is_pinned, status);
  } else {
    tensor = ((int)is_allocated(ten))
                 ? create_tensor(ten->shape, ten->dtype, ten->device,
                                 ten->requires_grad_, ten->is_pinned,
                                 ten->ndims, status)
                 : create_unallocated_tensor(
                       ten->shape, ten->dtype, ten->device, ten->requires_grad_,
                       ten->is_pinned, ten->ndims, status);
  }
  return tensor;
}

/**
 * @brief Create a view of an existing tensor with a new shape.
 *
 * @details
 * @li 1. Shallow-copy the source @c Tensor into @c dst.
 * @li 2. Increment the Rust reference count via @c retain().
 * @li 3. Overwrite @c ndims and @c shape with the new values (skipped for
 *    scalar sources).
 * @li 4. Recompute @c strides for the new shape via
 *    @c compute_tensor_strides_().
 * @li 5. Mark @c is_view_ = true and @c is_leaf_ = false.
 * @li 6. If the source has a gradient, create an unallocated gradient
 *    tensor for the view (preserving the gradient dtype).  On
 *    failure, release the view's storage via @c collect() and zero
 *    the tensor.
 *
 * The returned view shares the source's storage — no data copy is
 * performed.  The source must outlive the view.
 *
 * @param[in]  src        Source tensor to view.  Must have non-nullptr
 *                        storage.  Must outlive the view.
 * @param[in]  new_shape  New dimension sizes.  Product must equal
 *                        @c src->size.
 * @param[in]  new_ndims  Number of dimensions in @p new_shape.
 * @param[out] status     Receives the operation result.
 *
 * @return View @c Tensor sharing @p src's storage.
 *
 * @pre  @c product(new_shape[0..new_ndims]) must equal @c src->size.
 * @pre  @c src->storage must not be @c nullptr.
 * @pre  @p status must not be @c nullptr.
 * @post The returned tensor has @c is_view_ = true and
 *       @c is_leaf_ = false.
 * @post The Rust reference count is incremented by one.
 * @post If @c src->grad != nullptr, the view has an unallocated gradient.
 *
 * @see retain()   Increments the storage reference count.
 * @see collect()  Decrements it (and may free storage).
 * @see is_view()  Query predicate.
 */
Tensor create_view(const Tensor *restrict src, const shape_t new_shape,
                   size_t new_ndims, novaStatus_t *status) {

  Tensor dst = *src;

  // Increase rust reference counter
  retain(&dst.storage->handle);

  // Copy tensor metadata
  dst.ndims = new_ndims;
  if (!is_scalar(src)) {
    memcpy(dst.shape, new_shape, new_ndims * sizeof(size_t));
    compute_tensor_strides_(&dst, dst.ndims, dst.shape, dst.item_size);
  }

  dst.is_view_ = true;
  dst.is_leaf_ = false;

  if (src->grad != nullptr) {
    dst.grad =
        create_unallocated_grad_tensor(new_shape, src->grad->dtype, src->device,
                                       src->is_pinned, new_ndims, status);

    if (status->err != novaSuccess) {
      collect(&dst); // Decrease rust reference counter
      memset(&dst, 0, sizeof(Tensor));
      return dst;
    }
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
 * @code
 * expected = item_size
 * for dim = ndims-1 .. 0:
 *     if strides[dim] != expected: not contiguous
 *     expected *= shape[dim]
 * @endcode
 *
 * For a scalar (@c ndims == 0), the function returns @c true
 * immediately.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the tensor is contiguous, @c false otherwise.
 *
 * @see strides_t    Stride array.
 * @see create_view()  Views may break contiguity.
 */
bool is_contiguous(const Tensor *restrict ten) {
  if (is_scalar(ten)) {
    return true;
  }
  size_t expected, ndims;
  expected = ten->item_size, ndims = ten->ndims;
  for (size_t dim = 0; dim < ndims; ++dim) {
    if (ten->strides[ndims - 1 - dim] != expected) {
      return false;
    }
    expected *= ten->shape[ndims - 1 - dim];
  }
  return true;
}

Tensor contiguous(const Tensor *restrict ten, novaStatus_t *status) {
  if (is_contiguous(ten)) {
    return create_view(ten, ten->shape, ten->ndims, status);
  }

  Tensor dst =
      (int)is_scalar(ten)
          ? create_scalar_tensor(ten->dtype, ten->device, ten->requires_grad_,
                                 ten->is_pinned, status)

          : create_tensor(ten->shape, ten->dtype, ten->device,
                          ten->requires_grad_, ten->is_pinned, ten->ndims,
                          status);

  if (status->err != novaSuccess) {
    return dst; // zeroaed tensor
  }

  // TODO: Implement dispatching contiguous operation

  return dst;
}

/**
 * @brief Move ownership of tensor resources from src to dst.
 *
 * @details
 * @li 1. Frees any existing resources in @c dst via @c collect().
 * @li 2. Bitwise-copies @c src into @c dst.
 * @li 3. Zeroes out @c src (storage, data, grad, grad_fn_ set to nullptr;
 *    and @c is_allocated_ set to @c false) so that a
 *    subsequent @c collect() on @c src is a no-op.
 *
 * This is a move semantics wrapper — after the call, only @c dst
 * owns the resources.
 *
 * @param[in,out] dst  Destination tensor (previous resources are
 *                     freed via @c collect()).
 * @param[in,out] src  Source tensor (ownership transferred; @c src
 *                     becomes a hollow shell).
 *
 * @pre  @p dst and @p src must not be @c nullptr.
 * @post @p dst owns all resources previously held by @p src.
 * @post @p src is in a valid but unallocated state.
 *
 * @see collect()  Called on @c dst before the move.
 */
void move_tensor(Tensor *restrict dst, Tensor *restrict src) {
  collect(dst);

  *dst = *src;
  src->storage = nullptr;
  src->data.data = nullptr;
  src->grad = nullptr;
  src->grad_fn_ = nullptr;
  src->is_allocated_ = false;
}

/**
 * @brief Recursively release tensor memory and gradients.
 *
 * @details
 * @li 1. If @c ten is @c nullptr, returns immediately (no-op).
 * @li 2. If @c ten->storage is non-nullptr, calls @c release() to
 *    decrement the Rust reference count.
 * @li 3. If @c release() returns @c true (count reached zero), frees
 *    the @c TensorStorage with @c free() and nullifies @c storage,
 *    @c data, and @c is_allocated_.
 * @li 4. If @c ten->grad is non-nullptr, recursively calls @c collect()
 *    on the gradient, then frees the gradient @c Tensor with
 *    @c free() and nullifies @c grad.
 *
 * This ensures the full gradient sub-graph is freed, not just
 * the top-level tensor.
 *
 * @param[in,out] ten  Tensor to collect.  May be @c nullptr.
 *
 * @post @c ten->storage reference count is decremented.
 * @post If the count reaches zero, @c storage and @c data are set
 *       to nullptr and @c is_allocated_ to @c false.
 * @post The gradient sub-graph is recursively freed.
 *
 * @see release()       Decrements the Rust reference count.
 * @see is_collected()  Query predicate after collection.
 */
void collect(Tensor *ten) {
  if (ten == nullptr) {
    return;
  }

  if (ten->storage != nullptr) {
    bool should_free = release(&ten->storage->handle);
    if (should_free) {
      free(ten->storage);
      ten->storage = nullptr;
      ten->data.data = nullptr;
      ten->is_allocated_ = false;
    }
  }

  if (ten->grad != nullptr) {
    collect(ten->grad);
    free(ten->grad);
    ten->grad = nullptr;
  }
}

/**
 * @brief Check whether a tensor is 0-dimensional (scalar).
 *
 * @details
 * A tensor is a scalar when all four conditions hold:
 * @li @c ndims == 0
 * @li @c shape[0] == 0
 * @li @c strides[0] == 0
 * @li @c size == 1
 *
 * This is the invariant established by @c create_scalar_tensor().
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the tensor is a scalar, @c false otherwise.
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
 * Same logic as @c is_scalar() but operates on a @c TensorGrad
 * (pointer-to-@c Tensor).
 *
 * @param[in] grad  Gradient tensor to check.  May be @c nullptr.
 *
 * @return @c true if the gradient is a scalar, @c false otherwise
 *         (including when @p grad is @c nullptr).
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
 * @li GPU (@c DEVICE_GPU): 512-byte alignment.
 * @li CPU (@c DEVICE_CPU): 64-byte alignment.
 * @li META (@c DEVICE_META): always returns @c true.
 *
 * The check reads @c ten->device to select the threshold and
 * tests @c ten->storage->ptr.v modulo the threshold.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the data pointer meets the alignment
 *         requirement, @c false otherwise.
 *
 * @pre  @c ten->storage must not be @c nullptr (except META).
 *
 * @see is_grad_aligned()  Gradient variant.
 */
bool is_aligned(const Tensor *ten) {
  if (ten->device == DEVICE_META) {
    return true;
  }
  return (bool)(ten->device == DEVICE_GPU
                    ? (((uintptr_t)ten->storage->ptr.v % 512) ==
                       0) // Aligned by default to 512 bytes (GPU)
                    : (((uintptr_t)ten->storage->ptr.v % 64) ==
                       0)); // Aligned by default to 64 bytes (CPU)
}

/**
 * @brief Check whether a gradient tensor's data buffer is properly
 *        aligned.
 *
 * @details
 * Same alignment logic as @c is_aligned() — 512-byte for GPU,
 * 64-byte for CPU, and always @c true for META tensors.
 *
 * @param[in] grad  Gradient tensor to check.  May be @c nullptr.
 *
 * @return @c true if the gradient data pointer meets the alignment
 *         requirement, @c false otherwise (including @c nullptr grad).
 *
 * @see is_aligned()  Tensor variant.
 */
bool is_grad_aligned(TensorGrad grad) {
  if (grad->device == DEVICE_META) {
    return true;
  }
  return (grad->device == DEVICE_GPU
              ? (((uintptr_t)grad->storage->ptr.v % 512) ==
                 0) // Aligned by default to 512 bytes (GPU)
              : (((uintptr_t)grad->storage->ptr.v % 64) == 0)) !=
         0; // Aligned by default to 64 bytes (CPU)
}

/**
 * @brief Check whether a tensor has been collected (freed).
 *
 * @details
 * A tensor is considered collected when all three conditions hold:
 * @li @c data.data == nullptr
 * @li @c storage == nullptr
 * @li @c is_allocated_ == false
 *
 * This is the state after @c collect() has fully released the
 * tensor's storage.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the tensor has been collected, @c false
 *         otherwise.
 *
 * @see collect()
 * @see is_grad_collected()  Gradient variant.
 */
bool is_collected(const Tensor *ten) {
  return (bool)(ten->data.data == nullptr && ten->storage == nullptr &&
                !ten->is_allocated_);
}

/**
 * @brief Check whether a gradient tensor has been collected.
 *
 * @details
 * If @p grad is @c nullptr, returns @c true (a nullptr gradient is
 * logically "collected").  Otherwise, checks the same three
 * conditions as @c is_collected().
 *
 * @param[in] grad  Gradient tensor to check.  May be @c nullptr.
 *
 * @return @c true if the gradient has been collected (or is
 *         @c nullptr), @c false otherwise.
 *
 * @see is_collected()  Tensor variant.
 */
bool is_grad_collected(TensorGrad grad) {
  if (grad != nullptr) {
    return (bool)(grad->data.data == nullptr && grad->storage == nullptr &&
                  !grad->is_allocated_);
  }
  return true;
}

/**
 * @brief Check whether a tensor's data buffer has been allocated.
 *
 * @details
 * Returns @c true when all three conditions hold:
 * @li @c is_allocated_ == true
 * @li @c storage != nullptr
 * @li @c data.data != nullptr
 *
 * This triple-check ensures the tensor was both marked as
 * allocated and actually has a valid pointer.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the tensor is allocated, @c false otherwise.
 *
 * @see is_grad_allocated()  Gradient variant.
 * @see is_collected()       Inverse check.
 */
bool is_allocated(const Tensor *ten) {
  return (bool)(ten->is_allocated_ && ten->storage != nullptr &&
                ten->data.data != nullptr);
}

/**
 * @brief Check whether a gradient tensor has been allocated.
 *
 * @details
 * If @p grad is @c nullptr, returns @c false.  Otherwise, checks the
 * same three conditions as @ref is_allocated(): @c is_allocated_,
 * @c storage != nullptr, and @c data.data != nullptr.
 *
 * @param[in] grad  Gradient tensor to check.  May be @c nullptr.
 *
 * @return @c true if the gradient has a valid backing buffer,
 *         @c false otherwise (including when @p grad is @c nullptr).
 *
 * @see is_allocated()  Tensor variant.
 */
bool is_grad_allocated(TensorGrad grad) {
  if (grad != nullptr) {
    return (bool)(grad->is_allocated_ && grad->storage != nullptr &&
                  grad->data.data != nullptr);
  }
  return false;
}

/**
 * @brief Check whether a tensor is a view (shares storage).
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 * @return @c true if the tensor is a view, @c false otherwise.
 */
bool is_view(const Tensor *ten) { return ten->is_view_; }

/**
 * @brief Check whether a gradient tensor is a view.
 *
 * @param[in] grad  Gradient tensor to check.  May be @c nullptr.
 * @return @c true if the gradient is a view, @c false otherwise.
 */
bool is_grad_view(TensorGrad grad) { return grad->is_view_; }

/**
 * @brief Common transfer logic for device-to-host and host-to-device
 *        moves.
 *
 * @details
 * Validates the precondition via @p condition, then delegates to
 * @ref transfer_to() which returns the result directly.
 *
 * @param[in]  src       Source tensor.
 * @param[in]  dst       Destination tensor.
 * @param[in]  condition If @c true, the transfer direction is invalid
 *                       and @ref novaInvalidTransfDirection is returned.
 *
 * @return @ref novaStatus_t describing the outcome.
 */
static inline novaStatus_t transf_tensor_commom(const Tensor *restrict src,
                                                Tensor *restrict dst,
                                                bool condition) {
  novaStatus_t status;
  if (condition) {
    status.err = novaInvalidTransfDirection;
    status.message = nova_get_error_msg(status.err, nullptr);
    return status;
  }
  return transfer_to(src->device, dst->device, (const void *)src->data.v,
                     dst->data.v, src->storage->size_bytes);
}

/**
 * @brief Transfer tensor data from GPU device memory to CPU host
 *        memory.
 *
 * @details
 * Validates that @p src is GPU-resident with device-backed storage
 * and that @p dst is an allocated CPU tensor.  Delegates to
 * @ref transf_tensor_commom() for the actual transfer.
 *
 * @param[in]  src  Source tensor on GPU.
 * @param[in,out] dst  Destination tensor on CPU.
 * @return @ref novaStatus_t with the result of the transfer.
 */
novaStatus_t transf_tensor_from_device(const Tensor *restrict src,
                                       Tensor *restrict dst) {

  bool condition = (bool)(((src->device != DEVICE_GPU ||
                            !is_device_memory_handle(&src->storage->handle)) &&
                           (dst->device != DEVICE_CPU || !is_allocated(dst))));

  return transf_tensor_commom(src, dst, condition);
}

/**
 * @brief Transfer tensor data from CPU host memory to GPU device
 *        memory.
 *
 * @details
 * Validates that @p src is an allocated CPU tensor and that @p dst
 * is GPU-resident with device-backed storage.  Delegates to
 * @ref transf_tensor_commom() for the actual transfer.
 *
 * @param[in]  src  Source tensor on CPU.
 * @param[in,out] dst  Destination tensor on GPU.
 * @return @ref novaStatus_t with the result of the transfer.
 */
novaStatus_t transf_tensor_from_host(const Tensor *restrict src,
                                     Tensor *restrict dst) {
  bool condition = (bool)(((src->device != DEVICE_CPU || !is_allocated(src)) &&
                           (dst->device != DEVICE_GPU ||
                            !is_device_memory_handle(&dst->storage->handle))));

  return transf_tensor_commom(src, dst, condition);
}
