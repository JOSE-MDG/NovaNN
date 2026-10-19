/**
 * @file tensor.c
 * @brief Backend implementation of Tensor creation, views, and lifecycle.
 *
 * @details
 * Implements the full Tensor API: allocation through @ref safe_allocator(),
 * strided layout computation, view sharing with reference-counted storage,
 * META-tensor special cases, and cleanup of gradient chains.
 *
 * All creation and mutation functions receive an output
 * @ref novaStatus_t pointer.  On failure the returned tensor is zeroed
 * and the caller must not use it.
 *
 * Creation functions zero-initialize the @c Tensor struct before
 * populating fields.  @c safe_allocator() is the single allocation
 * entry point for CPU and GPU tensors; META tensors never invoke it.
 * Views bump the Rust reference count via @c retain() and drop it when
 * @c collect() runs on the view; a failed release leaves the storage
 * attached so cleanup can be retried.
 *
 * @see tensor.h   Public API declarations and struct documentation.
 * @see alloc.h    safe_allocator().
 * @see storage.h  retain() / release() reference-count operations.
 */

#include <stdio.h>
#include <string.h>

#include <ncore/core/alloc.h>
#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/core/storage.h>
#include <ncore/headeronly/macros.h>
#include <ncore/headeronly/tensor_utils.h>
#include <ncore/native/cpu/layout/contiguous.h>
#include <ncore/native/kernels/contiguous.h>
#include <ncore/tensor.h>

/**
 * @brief Create a fully allocated n-dimensional tensor.
 *
 * @details
 * Copies the shape, computes row-major strides, and allocates backing
 * storage through @ref safe_allocator().  META tensors skip allocation
 * and keep @c nullptr storage.  With @c requires_grad an unallocated
 * gradient tensor is attached; if that fails the just-allocated
 * storage is collected and the tensor zeroed.
 *
 * @param[in]  shape         Dimension sizes (only the first @c ndims
 *                           entries are read).
 * @param[in]  dtype         Element data type (@ref DType_).
 * @param[in]  device        Target device.
 * @param[in]  requires_grad If @c true, creates an unallocated
 *                           gradient tensor in @c grad.
 * @param[in]  pin_memory    If @c true, request page-locked host
 *                           memory (CPU only).
 * @param[in]  ndims         Number of dimensions, in
 *                           [1, @ref NOVA_MAX_DIMS].  Scalars
 *                           (@c ndims == 0) must be created via
 *                           @c create_scalar_tensor().
 * @param[out] status        Receives the operation result.
 *
 * @return Initialized @c Tensor with valid backing storage, or a
 *         META tensor with nullptr storage.
 *
 * @pre  @c product(shape[0..ndims]) must be > 0 for non-META.
 *
 * @see create_scalar_tensor()  Scalar (0-D) variant.
 * @see create_tensor_like()    Clone metadata from an existing tensor.
 */
Tensor create_tensor(const shape_t shape, DType_ dtype, Device_ device,
                     bool requires_grad, bool pin_memory, size_t ndims,
                     novaStatus_t *status) {

  Tensor tensor = {};
  shape_t storage_shape = {};
  storage_shape_(storage_shape, shape, dtype, ndims, status);
  if (status->err != novaSuccess) {
    return tensor;
  }
  memcpy(tensor.shape, storage_shape, ndims * sizeof(size_t));
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
  tensor.is_pinned_ = pin_memory;
  compute_tensor_size_(&tensor, storage_shape);
  compute_tensor_strides_(&tensor, ndims, storage_shape, tensor.item_size);
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
      (void)collect(&tensor);
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
 * Allocates a single-element buffer.  Shape and strides are zeroed,
 * @c ndims is 0 and @c size is 1.  For META, storage stays @c nullptr.
 *
 * @param[in]  dtype         Element data type (@ref DType_).
 * @param[in]  device        Target device.
 * @param[in]  requires_grad If @c true, creates an unallocated
 *                           scalar gradient tensor.
 * @param[in]  pin_memory    If @c true, request page-locked host
 *                           memory (CPU only).
 * @param[out] status        Receives the operation result.
 *
 * @return Initialized scalar @c Tensor with backing storage.
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
  tensor.logical_size = dtype_packing_factor(dtype);
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
  tensor.is_pinned_ = pin_memory;

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
      (void)collect(&tensor);
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
 * Delegates to the scalar or n-dimensional constructor depending on
 * @c is_scalar(ten).  The new tensor is independent of the source:
 * it owns its own storage and does not share the source's buffer.
 *
 * @param[in]  ten    Source tensor to copy metadata from.  Must not
 *                    be @c nullptr.
 * @param[out] status Receives the operation result.
 *
 * @return New @c Tensor with matching metadata and allocation state.
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
                                   ten->is_pinned_, status)
            : create_unallocated_scalar_tensor(ten->dtype, ten->device,
                                               ten->requires_grad_,
                                               ten->is_pinned_, status);
  } else {
    /*
     * The source's shape is stored in storage units.  Creation
     * functions take logical shapes and normalize internally, so
     * the innermost dimension is scaled back by the packing factor
     * before delegating.
     */
    shape_t logical_shape = {};
    const size_t packing = dtype_packing_factor(ten->dtype);
    memcpy(logical_shape, ten->shape, ten->ndims * sizeof(size_t));
    if (packing > 1) {
      logical_shape[ten->ndims - 1] *= packing;
    }
    tensor =
        ((int)is_allocated(ten))
            ? create_tensor(logical_shape, ten->dtype, ten->device,
                            ten->requires_grad_, ten->is_pinned_, ten->ndims,
                            status)
            : create_unallocated_tensor(logical_shape, ten->dtype, ten->device,
                                        ten->requires_grad_, ten->is_pinned_,
                                        ten->ndims, status);
  }
  return tensor;
}

/**
 * @brief Create a view of an existing tensor with a new shape.
 *
 * @details
 * Shallow-copies @p src, bumps the storage reference count via
 * @c retain(), then rewrites shape and strides for @p new_shape.
 * The view shares the source's storage, no data is copied and the
 * source must outlive the view.  If the source carries a gradient,
 * an unallocated gradient tensor is created for the view; on failure
 * the view's storage is collected and the tensor zeroed.
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
 *
 * @see retain()   Increments the storage reference count.
 * @see collect()  Decrements it (and may free storage).
 * @see is_view()  Query predicate.
 */
Tensor create_view(const Tensor *restrict src, const shape_t new_shape,
                   size_t new_ndims, novaStatus_t *status) {

  Tensor dst = *src;

  novaStatus_t retain_status = retain(&dst.storage->handle);
  if (retain_status.err != novaSuccess) {
    *status = retain_status;
    memset(&dst, 0, sizeof(Tensor));
    return dst;
  }

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
                                       src->is_pinned_, new_ndims, status);

    if (status->err != novaSuccess) {
      (void)collect(&dst);
      memset(&dst, 0, sizeof(Tensor));
      return dst;
    }
  }

  status->err = novaSuccess;
  status->message = nova_get_error_msg(status->err, nullptr);

  return dst;
}

/**
 * @brief Check whether a tensor's data buffer is contiguous in
 *        memory.
 *
 * @details
 * Contiguous means row-major without gaps: every stride must match
 * the running product of the inner shapes.  Scalars are always
 * contiguous.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the tensor is contiguous, @c false otherwise.
 *
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

/**
 * @brief Return a tensor whose data is stored contiguously.
 *
 * @details
 * If @p ten is already contiguous, a view sharing its storage is
 * returned via @ref create_view() and no data is copied.  Otherwise
 * a new tensor with the same metadata is allocated and the elements
 * are copied in.  Scalars go through @ref create_scalar_tensor() so
 * the result keeps @c ndims == 0.
 *
 * @param[in]  ten    Tensor to make contiguous.  Must not be
 *                    @c nullptr.
 * @param[out] status Receives the operation result.
 *
 * @return Contiguous @c Tensor with the same shape and metadata as
 *         @p ten, or a zeroed tensor on failure.
 *
 * @see is_contiguous()  Query predicate.
 * @see create_view()    Fast path for already-contiguous tensors.
 */
Tensor contiguous(const Tensor *restrict ten, novaStatus_t *status) {
  if (is_contiguous(ten)) {
    return create_view(ten, ten->shape, ten->ndims, status);
  }

  shape_t logical_shape = {};
  memcpy(logical_shape, ten->shape, ten->ndims * sizeof(size_t));
  const size_t packing = dtype_packing_factor(ten->dtype);
  if (packing > 1 && ten->ndims > 0) {
    logical_shape[ten->ndims - 1] *= packing;
  }

  Tensor dst =
      (int)is_scalar(ten)
          ? create_scalar_tensor(ten->dtype, ten->device, ten->requires_grad_,
                                 ten->is_pinned_, status)

          : create_tensor(logical_shape, ten->dtype, ten->device,
                          ten->requires_grad_, ten->is_pinned_, ten->ndims,
                          status);

  if (status->err != novaSuccess) {
    return dst;
  }

  if (in_device(ten)) {
    *status = launchContiguousKernel(ten, &dst);
    if (status->err != novaSuccess) {
      (void)collect(&dst);
    }
  } else if (in_host(ten)) {
    *status = contiguous_cpu_impl(ten, &dst);
    if (status->err != novaSuccess) {
      (void)collect(&dst);
    }
  }

  return dst;
}

/**
 * @var err_msg_buf
 * @brief Scratch space for formatted error messages.
 *
 * @details Thread-local so concurrent failures never clobber each
 * other, mirroring the Rust side (@c STATUS_MESSAGE in
 * @c status.rs). Valid until the next formatted error on the same
 * thread; callers consume it immediately through @c status.message.
 */
static thread_local char err_msg_buf[512];

/**
 * @brief Swap two dimensions of a tensor, returning a view.
 *
 * @details
 * Builds a view sharing @p ten's storage (no data is copied) with
 * the two dimensions exchanged. Both shape and strides come from the
 * source, so the result is a genuine transposed view: element
 * @c [i, j] of the result aliases @c [j, i] of the source, and the
 * result is generally not contiguous. Swapping a dimension with
 * itself is a no-op. Negative indices count from the last dimension
 * (@c -1 is the innermost), matching the usual convention. Packed
 * dtypes (packing factor above 1) may not move the last dimension:
 * swapping it with another fails with @ref novaInvalidMemoryLayout.
 *
 * @param[in]  ten   Source tensor. Must not be @c nullptr.
 * @param[out] st    Receives the operation result.
 * @param[in]  dim0  First dimension to swap. May be negative.
 * @param[in]  dim1  Second dimension to swap. May be negative.
 *
 * @return View @c Tensor sharing @p ten's storage.  On failure the
 *         result must not be used.
 *
 * @pre  @p ten must not be @c nullptr.
 * @pre  @p st must not be @c nullptr.
 * @post On success, the result has @c is_view_ == true and swapped
 *       shape and strides at @p dim0 and @p dim1.
 *
 * @see permute()      General dimension reordering.
 * @see create_view()  Storage sharing behind the result.
 */
Tensor transpose(const Tensor *ten, novaStatus_t *st, int dim0, int dim1) {

  auto dst = create_view(ten, ten->shape, ten->ndims, st);

  if (st->err != novaSuccess) {
    return dst;
  }

  const int ndims = (int)ten->ndims;
  const int d0 = dim0 < 0 ? ndims + dim0 : dim0;
  const int d1 = dim1 < 0 ? ndims + dim1 : dim1;

  if ((d0 < 0 || d0 >= ndims) || (d1 < 0 || d1 >= ndims)) {
    st->err = novaInvalidValue;
    snprintf(err_msg_buf, sizeof(err_msg_buf),
             "transpose(): dims (%d, %d) out of range for tensor with %d dims",
             dim0, dim1, ndims);
    st->message = err_msg_buf;
    (void)collect(&dst);
    return dst;
  }

  if (ten->ndims > 0 && dtype_packing_factor(ten->dtype) > 1) {
    // Packed dtypes store several logical values per storage byte along the
    // last dimension, so moving it would split values across rows, which byte
    // strides cannot express. Swapping the dimension with itself is a no-op
    // and stays allowed.
    const size_t packing = dtype_packing_factor(ten->dtype);
    const int last = ndims - 1;
    if ((d0 == last) != (d1 == last)) {
      st->err = novaInvalidMemoryLayout;
      snprintf(err_msg_buf, sizeof(err_msg_buf),
               "transpose(): cannot move the packed last dim\n(packing "
               "factor %zu, several logical values share one storage "
               "byte):\nswapping dims (%d, %d) would split values across "
               "rows,\nwhich byte strides cannot express.\nTranspose dims "
               "that exclude the last one.\n",
               packing, dim0, dim1);
      st->message = err_msg_buf;
      (void)collect(&dst);
      return dst;
    }
  }

  memcpy(dst.strides, ten->strides, (size_t)ndims * sizeof(size_t));

  dst.shape[d0] = ten->shape[d1];
  dst.shape[d1] = ten->shape[d0];
  dst.strides[d0] = ten->strides[d1];
  dst.strides[d1] = ten->strides[d0];

  st->err = novaSuccess;
  st->message = nova_get_error_msg(st->err, nullptr);
  return dst;
}

/**
 * @brief Reorder tensor dimensions following an explicit permutation.
 *
 * @details
 * Builds a view sharing @p ten's storage (no data is copied) whose
 * @c i-th dimension takes the shape and stride of source dimension
 * @c dims[i]. Negative entries count from the last dimension. Every
 * source dimension must appear exactly once: out-of-range indices
 * and duplicates are rejected before anything is written, so a
 * failed call leaves no half-permuted tensor behind. Packed dtypes
 * (packing factor above 1) must keep the last dimension innermost:
 * anything else fails with @ref novaInvalidMemoryLayout.
 *
 * Only the first @p ndims entries of @p dims are read; the rest of
 * the @c NOVA_MAX_DIMS array is ignored.
 *
 * @param[in]  ten   Source tensor. Must not be @c nullptr.
 * @param[out] st    Receives the operation result.
 * @param[in]  dims  Permutation of @c [0, ndims). Negative entries
 *                   count from the last dimension.
 *
 * @return View @c Tensor sharing @p ten's storage.  On failure the
 *         result must not be used.
 *
 * @pre  @p ten must not be @c nullptr.
 * @pre  @p st must not be @c nullptr.
 * @pre  @p dims holds each index in @c [0, ndims) exactly once
 *       (negatives allowed).
 * @post On success, the result has @c is_view_ == true with shape
 *       and strides permuted after @p dims.
 *
 * @see transpose()    Two-dimension swap.
 * @see create_view()  Storage sharing behind the result.
 */
Tensor permute(const Tensor *ten, novaStatus_t *st,
               const int dims[NOVA_MAX_DIMS]) {
  Tensor dst = create_view(ten, ten->shape, ten->ndims, st);

  if (st->err != novaSuccess) {
    return dst;
  }
  const int ndims = (int)ten->ndims;

  shape_t tmp_shape;
  strides_t tmp_strides;
  memcpy(tmp_shape, ten->shape, (size_t)ndims * sizeof(size_t));
  memcpy(tmp_strides, ten->strides, (size_t)ndims * sizeof(size_t));

  bool seen[NOVA_MAX_DIMS] = {false};
  for (int i = 0; i < ndims; ++i) {
    const int dim = dims[i] < 0 ? ndims + dims[i] : dims[i];
    if (dim < 0 || dim >= ndims) {
      st->err = novaInvalidValue;
      snprintf(err_msg_buf, sizeof(err_msg_buf),
               "permute(): dims[%d] = %d out of range for tensor with %d dims",
               i, dims[i], ndims);
      st->message = err_msg_buf;
      (void)collect(&dst);
      return dst;
    }
    if (seen[dim]) {
      st->err = novaInvalidValue;
      snprintf(err_msg_buf, sizeof(err_msg_buf),
               "permute(): duplicate dimension %d in permutation", dim);
      st->message = err_msg_buf;
      (void)collect(&dst);
      return dst;
    }
    seen[dim] = true;
    dst.shape[i] = tmp_shape[dim];
    dst.strides[i] = tmp_strides[dim];
  }

  if (ten->ndims > 0 && dtype_packing_factor(ten->dtype) > 1) {
    // Same constraint as transpose(): the packed last dimension must stay
    // innermost, otherwise values would split across rows. dims[] was
    // validated in range above, so normalizing the innermost entry is safe.
    const size_t packing = dtype_packing_factor(ten->dtype);
    const int last = ndims - 1;
    const int inner = dims[last] < 0 ? ndims + dims[last] : dims[last];
    if (inner != last) {
      st->err = novaInvalidMemoryLayout;
      snprintf(err_msg_buf, sizeof(err_msg_buf),
               "permute(): cannot move the packed last dim\n(packing factor "
               "%zu, several logical values share one storage byte):\nthe "
               "permutation takes it out of the innermost position,\nwhich "
               "byte strides cannot express.\nKeep the last dim "
               "innermost.\n",
               packing);
      st->message = err_msg_buf;
      (void)collect(&dst);
      return dst;
    }
  }

  st->err = novaSuccess;
  st->message = nova_get_error_msg(st->err, nullptr);
  return dst;
}

/**
 * @brief Move ownership of tensor resources from src to dst.
 *
 * @details
 * Frees any resources currently held by @p dst via @c collect(),
 * bitwise-copies @c src over it, then hollows out @c src (pointers
 * nulled, allocation flag cleared) so a later @c collect() on
 * @c src is a no-op.  After the call only @c dst owns the resources.
 *
 * @param[in,out] dst  Destination tensor (previous resources are
 *                     freed via @c collect()).
 * @param[in,out] src  Source tensor (ownership transferred; @c src
 *                     becomes a hollow shell).
 *
 * @return Operation status.  On @c collect() failure @p src is left
 *         untouched.
 *
 * @pre  @p dst and @p src must not be the same object.
 *
 * @see collect()  Called on @c dst before the move.
 */
novaStatus_t move_tensor(Tensor *restrict dst, Tensor *restrict src) {
  if (dst == nullptr || src == nullptr) {
    return (novaStatus_t){.err = novaInvalidPointer,
                          .message =
                              nova_get_error_msg(novaInvalidPointer, nullptr)};
  }
  if (dst == src) {
    return (novaStatus_t){.err = novaSuccess,
                          .message = nova_get_error_msg(novaSuccess, nullptr)};
  }
  novaStatus_t st = collect(dst);
  if (st.err != novaSuccess) {
    return st;
  }

  *dst = *src;
  src->storage = nullptr;
  src->data.data = nullptr;
  src->grad = nullptr;
  src->grad_fn_ = nullptr;
  src->is_allocated_ = false;
  return (novaStatus_t){.err = novaSuccess,
                        .message = nova_get_error_msg(novaSuccess, nullptr)};
}

/**
 * @brief Release tensor memory and gradients.
 *
 * @details
 * Releases @c ten->storage with @c release() and, when the reference
 * count reaches zero, frees the @c TensorStorage.  On release
 * failure the storage stays attached for a later cleanup attempt.
 * The @c grad chain is walked iteratively the same way; a node whose
 * release fails stays linked under @c ten->grad with the remainder
 * of the chain for retry.
 *
 * @param[in,out] ten  Tensor to collect.  May be @c nullptr.
 *
 * @return First release error encountered, or @c novaSuccess.
 *
 * @see release()       Decrements the Rust reference count.
 * @see is_collected()  Query predicate after collection.
 */
novaStatus_t collect(Tensor *ten) {
  novaStatus_t st = {.err = novaSuccess,
                     .message = nova_get_error_msg(novaSuccess, nullptr)};
  if (ten == nullptr) {
    return st;
  }

  if (ten->storage != nullptr) {
    novaStatus_t release_status = {};
    bool should_free = release(&ten->storage->handle, &release_status);
    if (release_status.err != novaSuccess) {
      st = release_status;
    } else if (should_free) {
      free(ten->storage);
      ten->storage = nullptr;
      ten->data.data = nullptr;
      ten->is_allocated_ = false;
    }
  }

  TensorGrad failed = nullptr;
  auto cur = ten->grad;
  ten->grad = nullptr;
  while (cur != nullptr) {
    auto nxt = cur->grad;
    cur->grad = nullptr;
    if (cur->storage != nullptr) {
      novaStatus_t release_status = {};
      bool should_free = release(&cur->storage->handle, &release_status);
      if (release_status.err != novaSuccess) {
        if (st.err == novaSuccess) {
          st = release_status;
        }
        cur->grad = nxt;
        failed = cur;
        break;
      }
      if (should_free) {
        free(cur->storage);
        cur->storage = nullptr;
        cur->data.data = nullptr;
        cur->is_allocated_ = false;
      }
    }
    free(cur);
    cur = nxt;
  }
  if (failed != nullptr) {
    ten->grad = failed;
  }
  return st;
}

/**
 * @brief Check whether a tensor is 0-dimensional (scalar).
 *
 * @details
 * Matches the invariant established by @c create_scalar_tensor().
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
 * Same logic as @c is_scalar() on a @c TensorGrad.
 *
 * @param[in] grad  Gradient tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the gradient is a scalar, @c false otherwise.
 *
 * @see is_scalar()  Tensor variant.
 */
bool is_scalar_grad(TensorGrad grad) {
  return (bool)(grad->shape[0] == 0 && grad->strides[0] == 0 &&
                grad->size == 1 && grad->ndims == 0);
}

/**
 * @brief Check whether a tensor has been collected (freed).
 *
 * @details
 * True when the data pointer and storage are gone and the allocation
 * flag is cleared, the state after @c collect() has fully released
 * the tensor.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the tensor has been collected, @c false
 *         otherwise.
 *
 * @see collect()           Tensor collection.
 * @see is_grad_collected() Gradient variant.
 */
bool is_collected(const Tensor *ten) {
  return (bool)(ten->data.data == nullptr && ten->storage == nullptr &&
                !ten->is_allocated_);
}

/**
 * @brief Check whether a gradient tensor has been collected.
 *
 * @details
 * A @c nullptr gradient counts as collected.  Otherwise the same
 * three conditions as @c is_collected() apply.
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
 * Checks the allocation flag plus both pointers, so a tensor only
 * counts as allocated when it was marked as such and actually holds
 * a valid buffer.
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
 * Same conditions as @ref is_allocated(); a @c nullptr gradient is
 * never allocated.
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
 * @details
 * True for tensors created via @ref create_view().  Views share the
 * underlying storage with their source and have @c is_leaf_ == false.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the tensor is a view, @c false otherwise.
 *
 * @see create_view()   Constructor for views.
 * @see is_grad_view()  Gradient variant.
 */
bool is_view(const Tensor *ten) { return ten->is_view_; }

/**
 * @brief Check whether a gradient tensor is a view.
 *
 * @details
 * Same logic as @ref is_view() on a @c TensorGrad.
 *
 * @param[in] grad  Gradient tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the gradient is a view, @c false otherwise.
 *
 * @see is_view()  Tensor variant.
 */
bool is_grad_view(TensorGrad grad) { return grad->is_view_; }

/**
 * @brief Check whether a tensor's data resides in GPU device memory.
 *
 * @details
 * Requires an allocated tensor flagged GPU-resident whose storage
 * handle is device-managed.  This separates genuine VRAM-resident
 * tensors from CPU tensors, including pinned host tensors whose
 * handles are also device-managed but which are not in VRAM.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the tensor is allocated, GPU-resident, and
 *         device-backed, @c false otherwise.
 *
 * @see in_host()  Complementary check.
 */
bool in_device(const Tensor *ten) {
  return (bool)(is_allocated(ten) && ten->device == DEVICE_GPU &&
                is_device_memory_handle(&ten->storage->handle));
}

/**
 * @brief Check whether a tensor's data resides in CPU host memory.
 *
 * @details
 * A tensor is considered to be "in host" when it is allocated and
 * host-resident. Pinned host memory counts: it is still CPU
 * memory, directly addressable by host kernels, additionally
 * reachable by DMA. Only the placement field decides, never the
 * handle kind.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the tensor is allocated and host-resident,
 *         @c false otherwise.
 *
 * @see in_device()                   Complementary check.
 * @see is_allocated()                Allocation precondition.
 * @see is_device_memory_handle()     DMA-reachability check, used
 *                                    by transfers and repr, not here.
 */
bool in_host(const Tensor *ten) {
  return (bool)(is_allocated(ten) && ten->device == DEVICE_CPU);
}

/**
 * @brief Common transfer logic for device-to-host and host-to-device
 *        moves.
 *
 * @details
 * @p condition carries the caller's direction/precondition check; on
 * failure the transfer is refused with
 * @ref novaInvalidTransfDirection.  Otherwise the work goes to
 * @ref transfer_to().
 *
 * @param[in]  src       Source tensor.
 * @param[in]  dst       Destination tensor.
 * @param[in]  condition If @c true, the transfer direction is invalid
 *                       and @ref novaInvalidTransfDirection is returned.
 *
 * @return @ref novaStatus_t describing the outcome.
 */
static inline novaStatus_t transf_tensor_common(const Tensor *restrict src,
                                                Tensor *restrict dst,
                                                bool condition) {
  if (condition) {
    return (novaStatus_t){.err = novaTransferError,
                          .message =
                              nova_get_error_msg(novaTransferError, nullptr)};
  }

  return transfer_to(src->device, dst->device, src->data.data + src->offset,
                     dst->data.data, src->storage->handle.requested_bytes);
}

/**
 * @brief Transfer tensor data from GPU device memory to CPU host
 *        memory.
 *
 * @details
 * @p src must be GPU-resident with device-backed storage and @p dst
 * an allocated CPU tensor.
 *
 * @param[in]     src  Source tensor on GPU.
 * @param[in,out] dst  Destination tensor on CPU.
 * @return @ref novaStatus_t with the result of the transfer.
 */
novaStatus_t transf_tensor_from_device(const Tensor *restrict src,
                                       Tensor *restrict dst) {

  bool condition = (src->device != DEVICE_GPU || !is_allocated(src) ||
                    !is_device_memory_handle(&src->storage->handle) ||
                    (dst->device != DEVICE_CPU || !is_allocated(dst))) != 0;

  return transf_tensor_common(src, dst, condition);
}

/**
 * @brief Transfer tensor data from CPU host memory to GPU device
 *        memory.
 *
 * @details
 * @p src must be an allocated CPU tensor and @p dst GPU-resident
 * with device-backed storage.
 *
 * @param[in]     src  Source tensor on CPU.
 * @param[in,out] dst  Destination tensor on GPU.
 * @return @ref novaStatus_t with the result of the transfer.
 */
novaStatus_t transf_tensor_from_host(const Tensor *restrict src,
                                     Tensor *restrict dst) {
  bool condition = (src->device != DEVICE_CPU || !is_allocated(src) ||
                    (dst->device != DEVICE_GPU || !is_allocated(dst) ||
                     !is_device_memory_handle(&dst->storage->handle))) != 0;

  return transf_tensor_common(src, dst, condition);
}
