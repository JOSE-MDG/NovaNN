/**
 * @file copy.c
 * @brief Generic tensor copy routines and deep-copy dispatch.
 *
 * @details
 * This translation unit implements the tensor deep-copy machinery
 * declared in @ref copy.h.  It provides:
 *
 * @li CPU copy function — a single @c static inline routine
 *    (@ref copy_host_buffer) that performs host-to-host @c memcpy
 *    via a @c void* pointer, working for every dtype.
 * @li GPU copy function — a single @c static inline routine
 *    (@ref copy_device_buffer) that delegates to @ref transfer_to()
 *    for device-to-device copies.
 * @li Dispatch tables — @ref lookup_host_copy, @ref lookup_device_copy,
 *    and @ref lookup_copy map @c (device, dtype) pairs to the correct
 *    @ref CopyFn.
 * @li @c deepcopy() — the public entry point that allocates storage,
 *    copies metadata and data, and recursively copies the gradient
 *    subtree.
 *
 * @section design Design
 *
 * Rather than maintaining one copy function per dtype (as in earlier
 * versions), a single @c static inline @c copy_host_buffer uses
 * @c src->data.v / @c dst->data.v (@c void*) so the compiler can still
 * inline the @c memcpy call while keeping the codebase minimal.
 * Similarly, @c copy_device_buffer handles all dtypes in one routine,
 * directly returning the @ref transfer_to() result.  The dispatch
 * tables are @c const static and zero-initialised; only the entries
 * that correspond to supported dtypes are filled in.
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
 * Copies exactly @c src->storage->size_bytes bytes from the source
 * data buffer to the destination buffer using @c memcpy.  The copy is
 * dtype-agnostic — it operates on the raw @c void* pointers
 * (@c src->data.v, @c dst->data.v) so a single routine serves every
 * dtype.  The @c status parameter is accepted for interface uniformity
 * with @ref CopyFn but is not modified (the operation is infallible).
 *
 * @param[in]  src     Source tensor with allocated storage.  The
 *                     data buffer is read-only.
 * @param[out] dst     Destination tensor with pre-allocated storage
 *                     of at least @c src->storage->size_bytes.
 * @param[out] status  Unused for this function (always left
 *                     unchanged).  Provided for signature
 *                     compatibility with @ref CopyFn.
 *
 * @pre  Both @p src and @p dst have non-null, allocated storage.
 * @pre  @c dst->storage->size_bytes >= @c src->storage->size_bytes.
 * @post The first @c src->storage->size_bytes bytes of @c dst->data
 *       are a bitwise copy of @c src->data.
 */
static inline void copy_host_buffer(const Tensor *restrict src,
                                    Tensor *restrict dst,
                                    novaStatus_t *status) {

  (void)status;
  memcpy(dst->data.v, src->data.v, src->storage->size_bytes);
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
 * assigns the returned @ref novaStatus_t directly — the intermediate
 * @c map_code2err() translation is no longer required because
 * @ref transfer_to() now returns a @ref novaStatus_t natively.
 *
 * @param[in]  src     Source tensor residing on a GPU device.  Must
 *                     have allocated storage.
 * @param[out] dst     Destination tensor on a GPU device.  Must have
 *                     pre-allocated storage of at least
 *                     @c src->storage->size_bytes.
 * @param[out] status  Receives the result.  On success, set to
 *                     @ref novaSuccess.  If no device is available,
 *                     set to @ref novaDeviceNotAvailable.  On
 *                     transfer failure, set to the error code
 *                     returned by @ref transfer_to().
 *
 * @pre  A compute device (CUDA or HIP) must have been detected and
 *       initialised via @ref nova_initialize_device().
 * @pre  Both @p src and @p dst must have non-null, allocated storage
 *       on the active device.
 * @pre  @c dst->storage->size_bytes >= @c src->storage->size_bytes.
 * @post On success, @c dst->data contains a device-side bitwise copy
 *       of @c src->data and @p status is @ref novaSuccess.
 * @post On failure, @p status contains the appropriate error code
 *       and the contents of @c dst->data are undefined.
 *
 * @see transfer_to()   Low-level device memory transfer.
 */
static inline void copy_device_buffer(const Tensor *restrict src,
                                      Tensor *restrict dst,
                                      novaStatus_t *status) {
  if (get_detected_device_kind() != NULL_DEVICE) {

    status->err = novaDeviceNotAvailable;
    status->message = "Can not copy bytes in an invalid device.\n Please check "
                      "if any device was detected previously";
    return;
  }

  *status = transfer_to(src->device, dst->device, src->data.v, dst->data.v,
                        src->storage->size_bytes);
}

/**
 * @var lookup_host_copy
 * @brief Host-to-host copy dispatch table, indexed by @c DType_.
 *
 * @details
 * A @c NUM_DTYPES × 1 array of @ref CopyFn pointers.  Every entry
 * points to the generic @ref copy_host_buffer function, which handles
 * all dtypes via a @c void* @c memcpy.  Twenty-one dtypes are supported
 * (all standard float, low-precision, signed/unsigned integer, and
 * quantized variants).  Used by @ref deepcopy() when
 * @c src->device == DEVICE_CPU.
 */
const CopyFn lookup_host_copy[NUM_DTYPES][1] = {
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
 * handles all dtypes in a single routine.  Twenty-one dtypes are
 * supported (all standard float, low-precision, signed/unsigned
 * integer, and quantized variants).  Used by @ref deepcopy() when
 * @c src->device == DEVICE_GPU.
 */
const CopyFn lookup_device_copy[NUM_DTYPES][1] = {
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
 * A 2-element array indexed by @ref Device_:
 * @li @c lookup_copy[DEVICE_CPU] → @ref lookup_host_copy
 * @li @c lookup_copy[DEVICE_GPU] → @ref lookup_device_copy
 *
 * Used by @ref deepcopy() as @c lookup_copy[src->device][src->dtype]
 * to resolve the correct @ref CopyFn in a single array lookup.
 */
const CopyFn *lookup_copy[2] = {
    [DEVICE_CPU] = (CopyFn *)lookup_host_copy,
    [DEVICE_GPU] = (CopyFn *)lookup_device_copy,
};

/**
 * @brief Deep-copy a tensor, including metadata, data, and
 *        gradients.
 *
 * @details
 * Allocates new storage for @p dst via @ref safe_allocator(),
 * copies all metadata and element data from @p src, and recursively
 * deep-copies the gradient subtree.  The copy is dispatched through
 * the @ref lookup_copy table based on @c src->device and
 * @c src->dtype.
 *
 * @section behaviour Behaviour
 *
 * @li 1. All metadata fields (@c shape, @c strides, @c item_size, @c size,
 *    @c ndims, @c dtype, @c device, @c scale_, @c zero_point_,
 *    @c is_pinned, gradient flags, etc) are copied element-by-element.
 *    Fields @c is_view_, @c grad_fn_, and @c offset are set to fixed
 *    values (@c false, @c nullptr, @c 0 respectively).
 * @li 2. If @c src->storage is non-nullptr, a new @ref TensorStorage is
 *    allocated via @ref safe_allocator() and the data is copied
 *    using the appropriate @ref CopyFn.
 * @li 3. If @c src->grad is non-nullptr, the gradient tensor is recursively
 *    deep-copied via a self-recursive call.  Gradient copy errors
 *    are propagated through @p status.
 * @li 4. The destination tensor is marked as @c is_allocated_ = true,
 *    @c is_leaf_ = true, and @c is_view_ = false.
 *
 * @param[in]  src     Source tensor.  May be @c nullptr (no-op).
 * @param[out] dst     Destination tensor.  Must not be @c nullptr.  Must
 *                     have @c is_allocated_ == false (i.e., created
 *                     by @c create_unallocated_tensor()).
 * @param[out] status  Receives the result of the deep-copy
 *                     operation.  On success, set to
 *                     @ref novaSuccess.  On failure, set to the
 *                     appropriate error code.
 *
 * @pre  @p dst must be an unallocated tensor.
 * @pre  If @p src has a non-nullptr @c storage, its @c size_bytes must
 *       be > 0.
 * @post On success, @p dst is a complete independent copy of
 *       @p src, including gradient history.
 * @post On failure, @p status contains the error code and @p dst
 *       is left in a valid but unmodified state.
 *
 * @see CopyFn            Per-dtype copy function pointer type.
 * @see lookup_copy       Dispatch table selecting host vs device.
 * @see safe_allocator()  Storage allocator.
 * @see Device_            Device placement enum.
 * @see DType_            Data-type enum used for dispatch.
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
  dst->is_pinned = src->is_pinned;
  dst->grad_fn_ = nullptr;
  dst->offset = 0;
  dst->version_ = 0;

  if (src->storage != nullptr) {

    *status = safe_allocator(src->storage->size_bytes, src->device,
                             src->is_pinned, nullptr, dst, true);

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
                  src->grad->dtype, src->grad->device, src->grad->is_pinned,
                  status)
            : create_unallocated_grad_tensor(
                  src->grad->shape, src->grad->dtype, src->grad->device,
                  src->grad->is_pinned, src->grad->ndims, status);

    if (status->err != novaSuccess) {
      collect(dst);
      return;
    }

    dst->grad = new_grad;
    deepcopy(src->grad, dst->grad, status);
    if (status->err != novaSuccess) {
      collect(dst);
      return;
    }
  } else {
    dst->grad = nullptr;
  }
}
