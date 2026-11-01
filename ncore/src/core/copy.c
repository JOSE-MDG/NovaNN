/**
 * @file copy.c
 * @brief Generic tensor copy routines and deep-copy dispatch.
 *
 * @details
 * Tensor deep-copy machinery declared in @ref copy.h: a host and a
 * device copy routine, the (device, dtype) dispatch tables, and the
 * public @c deepcopy() entry point.
 *
 * Rather than maintaining one copy function per dtype (as in earlier
 * versions), a single @c static inline @c copy_host_buffer uses
 * @c src->data.v / @c dst->data.v (@c void*) so the compiler can still
 * inline the @c memcpy call while keeping the codebase minimal.
 * Likewise @c copy_device_buffer handles all dtypes in one routine.
 * The dispatch tables are @c const static and zero-initialized; only
 * the entries for supported dtypes are filled.
 *
 * @see copy.h       Public API for deep-copy.
 * @see device.h     Device placement and transfer functions.
 * @see tensor.h     Tensor structure and data-layout details.
 * @see alloc.h      Storage allocation.
 */

#include <string.h>

#include <ncore/core/alloc.h>
#include <ncore/core/copy.h>
#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/core/storage.h>
#include <ncore/headeronly/macros.h>
#include <ncore/headeronly/tensor_utils.h>
#include <ncore/tensor.h>

/**
 * @brief Bitwise-copy all element data from @p src to @p dst via
 *        @c memcpy.
 *
 * @details
 * Copies exactly @c src->storage->handle.requested_bytes bytes from the
 * source data buffer to the destination buffer using @c memcpy.  The copy
 * is dtype-agnostic. It operates on the raw @c void* pointers
 * (@c src->data.v, @c dst->data.v) so a single routine serves every
 * dtype.  Only the theoretical payload moves. The rounded reservation
 * padding is never copied.  The @c status parameter is accepted for
 * interface uniformity with @ref CopyFn but is not modified (the
 * operation is infallible).
 *
 * @param[in]  src     Source tensor with allocated storage.  The
 *                     data buffer is read-only.
 * @param[out] dst     Destination tensor with pre-allocated storage.
 * @param[out] status  Unused for this function (always left
 *                     unchanged).  Provided for signature
 *                     compatibility with @ref CopyFn.
 *
 * @pre  Both @p src and @p dst have non-null, allocated storage.
 * @pre  @c dst->storage->handle.requested_bytes >=
 *       @c src->storage->handle.requested_bytes.
 */
static inline void copy_host_buffer(const Tensor *restrict src,
                                    Tensor *restrict dst,
                                    novaStatus_t *status) {

  (void)status;
  memcpy(dst->data.data, src->data.data + src->offset,
         src->storage->handle.requested_bytes);
}

/**
 * @brief Copy element data between device buffers via
 *        @ref transfer_to().
 *
 * @details
 * Guards the transfer with a device-availability check through
 * @ref get_detected_device_kind().  If no compute device has been
 * detected, sets @p status to @ref novaDeviceNotAvailable and
 * returns early.  Otherwise delegates to @ref transfer_to() and
 * assigns the returned @ref novaStatus_t directly.
 *
 * @param[in]  src     Source tensor residing on a GPU device.  Must
 *                     have allocated storage.
 * @param[out] dst     Destination tensor on a GPU device.
 * @param[out] status  Receives the transfer result, or
 *                     @ref novaDeviceNotAvailable when no device has
 *                     been detected.
 *
 * @pre  A compute device (CUDA or HIP) must have been detected via
 *       @ref is_device_available().
 * @pre  Both @p src and @p dst must have non-null, allocated storage
 *       on the active device.
 * @pre  @c dst->storage->handle.requested_bytes >=
 *       @c src->storage->handle.requested_bytes.
 *
 * @see transfer_to()   Low-level device memory transfer.
 */
static inline void copy_device_buffer(const Tensor *restrict src,
                                      Tensor *restrict dst,
                                      novaStatus_t *status) {
  if (get_detected_device_kind() == NULL_DEVICE) {

    status->err = novaDeviceNotAvailable;
    status->message = "Can not copy bytes in an invalid device.\n Please check "
                      "if any device was detected previously";
    return;
  }

  *status = transfer_to(src->device, dst->device, src->data.data + src->offset,
                        dst->data.data, src->storage->handle.requested_bytes);
}

/**
 * @var lookup_host_copy
 * @brief Host-to-host copy dispatch table, indexed by @c DType_.
 *
 * @details
 * A @c NUM_DTYPES × 1 array of @ref CopyFn pointers.  Every entry
 * points to the generic @ref copy_host_buffer function, which handles
 * all dtypes via a @c void* @c memcpy.  Used by @ref deepcopy() when
 * @c src->device == DEVICE_CPU.
 */
static const CopyFn lookup_host_copy[NUM_DTYPES][1] = {
    /* Low-precision floats */
    [Float4E2M1fn] = {copy_host_buffer},
    [Float8E4M3fn] = {copy_host_buffer},
    [Float8E5M2] = {copy_host_buffer},

    /* Standard floats */
    [Float16] = {copy_host_buffer},
    [BFloat16] = {copy_host_buffer},
    [Float32] = {copy_host_buffer},
    [Float64] = {copy_host_buffer},

    /* Signed integers */
    [Signed8] = {copy_host_buffer},
    [Signed16] = {copy_host_buffer},
    [Signed32] = {copy_host_buffer},
    [Signed64] = {copy_host_buffer},

    /* Unsigned integers */
    [UnSigned8] = {copy_host_buffer},
    [UnSigned16] = {copy_host_buffer},
    [UnSigned32] = {copy_host_buffer},
    [UnSigned64] = {copy_host_buffer},

    /* Quantized signed */
    [QSigned8] = {copy_host_buffer},
    [QSigned16] = {copy_host_buffer},
    [QSigned32] = {copy_host_buffer},

    /* Quantized unsigned */
    [QUnSigned8] = {copy_host_buffer},
    [QUnSigned16] = {copy_host_buffer},
    [QUnSigned32] = {copy_host_buffer},
};

/**
 * @var lookup_device_copy
 * @brief Device-to-device copy dispatch table, indexed by @c DType_.
 *
 * @details
 * A @c NUM_DTYPES × 1 array of @ref CopyFn pointers.  Every entry
 * points to the generic @ref copy_device_buffer function, which
 * handles all dtypes in a single routine.  Used by @ref deepcopy()
 * when @c src->device == DEVICE_GPU.
 */
static const CopyFn lookup_device_copy[NUM_DTYPES][1] = {
    /* Low-precision floats */
    [Float4E2M1fn] = {copy_device_buffer},
    [Float8E4M3fn] = {copy_device_buffer},
    [Float8E5M2] = {copy_device_buffer},

    /* Standard floats */
    [Float16] = {copy_device_buffer},
    [BFloat16] = {copy_device_buffer},
    [Float32] = {copy_device_buffer},
    [Float64] = {copy_device_buffer},

    /* Signed integers */
    [Signed8] = {copy_device_buffer},
    [Signed16] = {copy_device_buffer},
    [Signed32] = {copy_device_buffer},
    [Signed64] = {copy_device_buffer},

    /* Unsigned integers */
    [UnSigned8] = {copy_device_buffer},
    [UnSigned16] = {copy_device_buffer},
    [UnSigned32] = {copy_device_buffer},
    [UnSigned64] = {copy_device_buffer},

    /* Quantized signed */
    [QSigned8] = {copy_device_buffer},
    [QSigned16] = {copy_device_buffer},
    [QSigned32] = {copy_device_buffer},

    /* Quantized unsigned */
    [QUnSigned8] = {copy_device_buffer},
    [QUnSigned16] = {copy_device_buffer},
    [QUnSigned32] = {copy_device_buffer},
};

/**
 * @var lookup_copy
 * @brief Top-level dispatch table mapping @ref Device_ to the
 *        appropriate per-dtype copy table.
 *
 * @details
 * 2-element array indexed by @ref Device_ pointing at
 * @ref lookup_host_copy and @ref lookup_device_copy.  Used by
 * @ref deepcopy() as @c lookup_copy[src->device][src->dtype] to
 * resolve the correct @ref CopyFn in a single array lookup.
 */
static const CopyFn *lookup_copy[2] = {
    [DEVICE_CPU] = (CopyFn *)lookup_host_copy,
    [DEVICE_GPU] = (CopyFn *)lookup_device_copy,
};

/**
 * @brief Deep-copy a tensor, including metadata, data, and
 *        gradients.
 *
 * @details
 * Allocates new storage for @p dst via @ref safe_allocator(), copies
 * all metadata and element data from @p src, and recursively
 * deep-copies the gradient subtree.  The copy is dispatched through
 * the @ref lookup_copy table based on @c src->device and
 * @c src->dtype.
 *
 * The destination starts from fixed view/autograd metadata:
 * @c is_view_ is @c false, @c offset is 0, @c grad_fn_ is
 * @c nullptr, and @c is_leaf_ is @c true.
 *
 * @param[in]  src     Source tensor.  May be @c nullptr (no-op).
 * @param[out] dst     Destination tensor.  Must not be @c nullptr.
 *                     Must have @c is_allocated_ == false (i.e.,
 *                     created by @c create_unallocated_tensor()).
 * @param[out] status  Receives the result.  On failure @p dst is
 *                     collected (freed) and must not be used.
 *
 * @pre  If @p src has a non-nullptr @c storage, its @c requested_bytes
 *       must be > 0.
 *
 * @see CopyFn            Per-dtype copy function pointer type.
 * @see lookup_copy       Dispatch table selecting host vs device.
 * @see safe_allocator()  Storage allocator.
 */
void deepcopy(const Tensor *restrict src, Tensor *restrict dst,
              novaStatus_t *status) {

  if (src == nullptr) {
    return;
  }

  if (dst == nullptr) {
    status->err = novaInvalidTensor;
    status->message = "dst Tensor ptr is nullptr\n";
    return;
  }

  if (is_allocated(dst)) {
    status->err = novaInvalidTensor;
    status->message = "dst must be an unallocated, use "
                      "`create_unallocated_tensor()`\n";
    return;
  }

  if (src->device != dst->device) {
    status->err =
        src->dtype != dst->dtype ? novaInvalidDtype : novaInvalidDevice;
    status->message = nova_get_error_msg(status->err, nullptr);
    return;
  }

  memcpy(dst->shape, src->shape, src->ndims * sizeof(size_t));
  memcpy(dst->strides, src->strides, src->ndims * sizeof(size_t));
  dst->item_size = src->item_size;
  dst->size = src->size;
  dst->logical_size = src->logical_size;
  dst->ndims = src->ndims;
  dst->dtype = src->dtype;
  dst->device = src->device;
  dst->scale_ = src->scale_;
  dst->zero_point_ = src->zero_point_;
  dst->requires_grad_ = src->requires_grad_;
  dst->retain_grad_ = src->retain_grad_;
  dst->is_leaf_ = true;
  dst->is_view_ = false;
  dst->is_pinned_ = src->is_pinned_;
  dst->grad_fn_ = nullptr;
  dst->offset = 0;
  dst->version_ = 0;

  if (src->storage != nullptr) {

    *status = safe_allocator(src->storage->handle.requested_bytes, src->device,
                             src->is_pinned_, nullptr, dst, true);

    if (status->err != novaSuccess) {
      return;
    }

    const CopyFn func = lookup_copy[src->device][src->dtype];
    func(src, dst, status);
    if (status->err != novaSuccess) {
      return;
    }
  }

  if (src->grad != nullptr) {
    auto new_grad =
        (int)is_scalar(src->grad)
            ? create_unallocated_scalar_grad_tensor(
                  src->grad->dtype, src->grad->device, src->grad->is_pinned_,
                  status)
            : create_unallocated_grad_tensor(
                  src->grad->shape, src->grad->dtype, src->grad->device,
                  src->grad->is_pinned_, src->grad->ndims, status);

    if (status->err != novaSuccess) {
      (void)collect(dst);
      return;
    }

    dst->grad = new_grad;
    deepcopy(src->grad, dst->grad, status);
    if (status->err != novaSuccess) {
      (void)collect(dst);
      return;
    }
  } else {
    dst->grad = nullptr;
  }
}
