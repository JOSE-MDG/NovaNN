/**
 * @file copy.c
 * @brief Per-dtype tensor copy routines and deep-copy dispatch.
 *
 * @details
 * This translation unit implements the tensor deep-copy machinery
 * declared in @ref copy.h.  It provides:
 *
 * 1. **CPU copy functions** — 12 `static inline` routines (one per
 *    dtype) that perform host-to-host `memcpy`.
 * 2. **GPU copy functions** — 12 `static inline` routines (one per
 *    dtype) that delegate to @ref transfer_to() for device-to-device
 *    copies.
 * 3. **Dispatch tables** — @ref lookup_host_copy, @ref lookup_device_copy,
 *    and @ref lookup_copy map `(device, dtype)` pairs to the correct
 *    @ref CopyFn.
 * 4. **deepcopy()** — the public entry point that allocates storage,
 *    copies metadata and data, and recursively copies the gradient
 *    subtree.
 *
 * ## Design
 *
 * The per-dtype functions are kept as `static inline` so that the
 * compiler can inline the `memcpy` / `transfer_to` call and eliminate
 * the function-call overhead for small tensors.  The dispatch tables
 * are `const static` and zero-initialised; only the entries that
 * correspond to supported dtypes are filled in.
 *
 * ## Thread Safety
 *
 * The dispatch tables are read-only after process startup and are
 * safe to access from any thread.  The GPU copy functions delegate
 * to @ref transfer_to(), which is expected to be thread-safe.
 *
 * @see copy.h       Public API for deep-copy.
 * @see device.h     Device placement and transfer functions.
 * @see tensor.h     Tensor structure and data-layout details.
 * @see alloc.h      Storage allocation.
 */

#include "ncore/core/storage.h"
#include <ncore/core/alloc.h>
#include <ncore/core/copy.h>
#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <ncore/headeronly/tensor_utils.h>
#include <ncore/tensor.h>
#include <string.h>

/* ────────────────────────────────────────────────────────────────
 *  CPU copy functions — host-to-host memcpy, one per dtype
 * ──────────────────────────────────────────────────────────────── */

/** @brief Copy Float32 elements from @p src to @p dst via `memcpy`. */
static inline void copy_f32_host_buffer(const Tensor *restrict src,
                                        Tensor *restrict dst,
                                        novaStatus_t *status) {

  (void)status;
  memcpy(dst->data.f32, src->data.f32, src->storage->size_bytes);
}

/** @brief Copy Float64 elements from @p src to @p dst via `memcpy`. */
static inline void copy_f64_host_buffer(const Tensor *restrict src,
                                        Tensor *restrict dst,
                                        novaStatus_t *status) {
  (void)status;
  memcpy(dst->data.f64, src->data.f64, src->storage->size_bytes);
}

/** @brief Copy Float16 elements from @p src to @p dst via `memcpy`. */
static inline void copy_f16_host_buffer(const Tensor *restrict src,
                                        Tensor *restrict dst,
                                        novaStatus_t *status) {
  (void)status;
  memcpy(dst->data.half, src->data.half, src->storage->size_bytes);
}

/** @brief Copy BFloat16 elements from @p src to @p dst via `memcpy`. */
static inline void copy_bf16_host_buffer(const Tensor *restrict src,
                                         Tensor *restrict dst,
                                         novaStatus_t *status) {
  (void)status;
  memcpy(dst->data.bf16, src->data.bf16, src->storage->size_bytes);
}

/** @brief Copy Signed8 elements from @p src to @p dst via `memcpy`. */
static inline void copy_s8_host_buffer(const Tensor *restrict src,
                                       Tensor *restrict dst,
                                       novaStatus_t *status) {
  (void)status;
  memcpy(dst->data.s8, src->data.s8, src->storage->size_bytes);
}

/** @brief Copy UnSigned8 elements from @p src to @p dst via `memcpy`. */
static inline void copy_u8_host_buffer(const Tensor *restrict src,
                                       Tensor *restrict dst,
                                       novaStatus_t *status) {
  (void)status;
  memcpy(dst->data.u8, src->data.u8, src->storage->size_bytes);
}

/** @brief Copy QSigned8 elements from @p src to @p dst via `memcpy`. */
static inline void copy_qs8_host_buffer(const Tensor *restrict src,
                                        Tensor *restrict dst,
                                        novaStatus_t *status) {
  (void)status;
  memcpy(dst->data.qs8, src->data.qs8, src->storage->size_bytes);
}

/** @brief Copy QUnSigned8 elements from @p src to @p dst via `memcpy`. */
static inline void copy_qu8_host_buffer(const Tensor *restrict src,
                                        Tensor *restrict dst,
                                        novaStatus_t *status) {
  (void)status;
  memcpy(dst->data.qu8, src->data.qu8, src->storage->size_bytes);
}

/** @brief Copy Signed32 elements from @p src to @p dst via `memcpy`. */
static inline void copy_s32_host_buffer(const Tensor *restrict src,
                                        Tensor *restrict dst,
                                        novaStatus_t *status) {
  (void)status;
  memcpy(dst->data.s32, src->data.s32, src->storage->size_bytes);
}

/** @brief Copy UnSigned32 elements from @p src to @p dst via `memcpy`. */
static inline void copy_u32_host_buffer(const Tensor *restrict src,
                                        Tensor *restrict dst,
                                        novaStatus_t *status) {
  (void)status;
  memcpy(dst->data.u32, src->data.u32, src->storage->size_bytes);
}

/** @brief Copy Signed64 elements from @p src to @p dst via `memcpy`. */
static inline void copy_s64_host_buffer(const Tensor *restrict src,
                                        Tensor *restrict dst,
                                        novaStatus_t *status) {
  (void)status;
  memcpy(dst->data.s64, src->data.s64, src->storage->size_bytes);
}

/** @brief Copy UnSigned64 elements from @p src to @p dst via `memcpy`. */
static inline void copy_u64_host_buffer(const Tensor *restrict src,
                                        Tensor *restrict dst,
                                        novaStatus_t *status) {
  (void)status;
  memcpy(dst->data.u64, src->data.u64, src->storage->size_bytes);
}

/* ────────────────────────────────────────────────────────────────
 *  GPU copy functions — device-to-device via transfer_to()
 * ──────────────────────────────────────────────────────────────── */

/**
 * @brief Map a DeviceStatus integer code to a novaError_t.
 *
 * @details
 * Translates the low-level device transfer return code into the
 * centralized error enumeration.  The integer codes are
 * identity-mapped from the CUDA/HIP backends through the FFI
 * layer:
 *
 * ```
 * cudaTransfer() / hipTransfer()   (CudaIO.cpp / HipIO.cpp)
 *   → mapError()                (backend-specific)
 *   → device_transfer_c()          (ffi.cpp, direct passthrough)
 *   → transfer_to()              (device.c, returned verbatim)
 *   → map_code2err()             (this function)
 * ```
 *
 * The backend `mapError()` functions produce these integer codes:
 * | Code | Meaning                          |
 * |------|----------------------------------|
 * |  0   | Success                          |
 * |  1   | Invalid value / null pointer     |
 * |  2   | Invalid memcpy direction         |
 * |  3   | Invalid resource handle          |
 * | -1   | Catch-all / unrecognised error   |
 *
 * Negative codes (e.g., `-1`) are mapped to @ref novaTransferError.
 * Positive codes are looked up in the local table; unmapped positive
 * codes also fall back to @ref novaTransferError.
 *
 * @param[in] err_code  Code returned by @ref transfer_to() via
 *                      @ref DeviceStatus::code.
 * @return Corresponding @ref novaError_t.
 */
static inline novaError_t map_code2err(int err_code) {
  const novaError_t table[NUM_ERRORS] = {novaSuccess, novaInvalidValue,
                                         novaInvalidTransfDirection,
                                         novaInvalidResourceHandle};
  return (err_code >= 0) ? table[err_code] : novaTransferError;
}

/**
 * @brief Copy Float32 elements between GPU buffers via
 *        @ref transfer_to().
 *
 * @details
 * Checks that a valid device has been detected via
 * @ref get_detected_device_kind().  If no device is available, sets
 * @p status to @ref novaDeviceNotAvailable.  Otherwise delegates to
 * @ref transfer_to() and maps the result through @ref map_code2err().
 */
static inline void copy_f32_device_buffer(const Tensor *restrict src,
                                          Tensor *restrict dst,
                                          novaStatus_t *status) {
  if (get_detected_device_kind() != NULL_DEVICE) {

    status->err = novaDeviceNotAvailable;
    status->message = "Can not copy bytes in an invalid device.\n Please check "
                      "if any device was detected previously";
    return;
  }

  DeviceStatus dstatus = transfer_to(src->device, dst->device, src->data.f32,
                                     dst->data.f32, src->storage->size_bytes);

  status->err = map_code2err(dstatus.code);
  status->message = nova_get_error_msg(status->err, NULL);
}

/**
 * @brief Copy Float64 elements between GPU buffers via
 *        @ref transfer_to().
 *
 * @details
 * Checks that a valid device has been detected via
 * @ref get_detected_device_kind().  If no device is available, sets
 * @p status to @ref novaDeviceNotAvailable.  Otherwise delegates to
 * @ref transfer_to() and maps the result through @ref map_code2err().
 */
static inline void copy_f64_device_buffer(const Tensor *restrict src,
                                          Tensor *restrict dst,
                                          novaStatus_t *status) {

  if (get_detected_device_kind() != NULL_DEVICE) {
    status->err = novaDeviceNotAvailable;
    status->message = "Can not copy bytes in an invalid device.\n Please check "
                      "if any device was detected previously\n";
    return;
  }
  DeviceStatus dstatus = transfer_to(src->device, dst->device, src->data.f64,
                                     dst->data.f64, src->storage->size_bytes);

  status->err = map_code2err(dstatus.code);
  status->message = nova_get_error_msg(status->err, NULL);
}

/**
 * @brief Copy Float16 elements between GPU buffers via
 *        @ref transfer_to().
 *
 * @details
 * Checks that a valid device has been detected via
 * @ref get_detected_device_kind().  If no device is available, sets
 * @p status to @ref novaDeviceNotAvailable.  Otherwise delegates to
 * @ref transfer_to() and maps the result through @ref map_code2err().
 */
static inline void copy_f16_device_buffer(const Tensor *restrict src,
                                          Tensor *restrict dst,
                                          novaStatus_t *status) {
  if (get_detected_device_kind() != NULL_DEVICE) {
    status->err = novaDeviceNotAvailable;
    status->message = "Can not copy bytes in an invalid device.\n Please check "
                      "if any device was detected previously\n";
    return;
  }
  DeviceStatus dstatus = transfer_to(src->device, dst->device, src->data.half,
                                     dst->data.half, src->storage->size_bytes);

  status->err = map_code2err(dstatus.code);
  status->message = nova_get_error_msg(status->err, NULL);
}

/**
 * @brief Copy BFloat16 elements between GPU buffers via
 *        @ref transfer_to().
 *
 * @details
 * Checks that a valid device has been detected via
 * @ref get_detected_device_kind().  If no device is available, sets
 * @p status to @ref novaDeviceNotAvailable.  Otherwise delegates to
 * @ref transfer_to() and maps the result through @ref map_code2err().
 */
static inline void copy_bf16_device_buffer(const Tensor *restrict src,
                                           Tensor *restrict dst,
                                           novaStatus_t *status) {
  if (get_detected_device_kind() != NULL_DEVICE) {
    status->err = novaDeviceNotAvailable;
    status->message = "Can not copy bytes in an invalid device.\n Please check "
                      "if any device was detected previously\n";
    return;
  }

  DeviceStatus dstatus = transfer_to(src->device, dst->device, src->data.bf16,
                                     dst->data.bf16, src->storage->size_bytes);

  status->err = map_code2err(dstatus.code);
  status->message = nova_get_error_msg(status->err, NULL);
}

/**
 * @brief Copy Signed8 elements between GPU buffers via
 *        @ref transfer_to().
 *
 * @details
 * Checks that a valid device has been detected via
 * @ref get_detected_device_kind().  If no device is available, sets
 * @p status to @ref novaDeviceNotAvailable.  Otherwise delegates to
 * @ref transfer_to() and maps the result through @ref map_code2err().
 */
static inline void copy_s8_device_buffer(const Tensor *restrict src,
                                         Tensor *restrict dst,
                                         novaStatus_t *status) {
  if (get_detected_device_kind() != NULL_DEVICE) {
    status->err = novaDeviceNotAvailable;
    status->message = "Can not copy bytes in an invalid device.\n Please check "
                      "if any device was detected previously\n";
    return;
  }

  DeviceStatus dstatus = transfer_to(src->device, dst->device, src->data.s8,
                                     dst->data.s8, src->storage->size_bytes);

  status->err = map_code2err(dstatus.code);
  status->message = nova_get_error_msg(status->err, NULL);
}

/**
 * @brief Copy UnSigned8 elements between GPU buffers via
 *        @ref transfer_to().
 *
 * @details
 * Checks that a valid device has been detected via
 * @ref get_detected_device_kind().  If no device is available, sets
 * @p status to @ref novaDeviceNotAvailable.  Otherwise delegates to
 * @ref transfer_to() and maps the result through @ref map_code2err().
 */
static inline void copy_u8_device_buffer(const Tensor *restrict src,
                                         Tensor *restrict dst,
                                         novaStatus_t *status) {
  if (get_detected_device_kind() != NULL_DEVICE) {
    status->err = novaDeviceNotAvailable;
    status->message = "Can not copy bytes in an invalid device.\n Please check "
                      "if any device was detected previously\n";
    return;
  }
  DeviceStatus dstatus = transfer_to(src->device, dst->device, src->data.u8,
                                     dst->data.u8, src->storage->size_bytes);

  status->err = map_code2err(dstatus.code);
  status->message = nova_get_error_msg(status->err, NULL);
}

/**
 * @brief Copy QSigned8 elements between GPU buffers via
 *        @ref transfer_to().
 *
 * @details
 * Checks that a valid device has been detected via
 * @ref get_detected_device_kind().  If no device is available, sets
 * @p status to @ref novaDeviceNotAvailable.  Otherwise delegates to
 * @ref transfer_to() and maps the result through @ref map_code2err().
 */
static inline void copy_qs8_device_buffer(const Tensor *restrict src,
                                          Tensor *restrict dst,
                                          novaStatus_t *status) {
  if (get_detected_device_kind() != NULL_DEVICE) {
    status->err = novaDeviceNotAvailable;
    status->message = "Can not copy bytes in an invalid device.\n Please check "
                      "if any device was detected previously\n";
    return;
  }
  DeviceStatus dstatus = transfer_to(src->device, dst->device, src->data.qs8,
                                     dst->data.qs8, src->storage->size_bytes);

  status->err = map_code2err(dstatus.code);
  status->message = nova_get_error_msg(status->err, NULL);
}

/**
 * @brief Copy QUnSigned8 elements between GPU buffers via
 *        @ref transfer_to().
 *
 * @details
 * Checks that a valid device has been detected via
 * @ref get_detected_device_kind().  If no device is available, sets
 * @p status to @ref novaDeviceNotAvailable.  Otherwise delegates to
 * @ref transfer_to() and maps the result through @ref map_code2err().
 */
static inline void copy_qu8_device_buffer(const Tensor *restrict src,
                                          Tensor *restrict dst,
                                          novaStatus_t *status) {
  if (get_detected_device_kind() != NULL_DEVICE) {
    status->err = novaDeviceNotAvailable;
    status->message = "Can not copy bytes in an invalid device.\n Please check "
                      "if any device was detected previously\n";
    return;
  }
  DeviceStatus dstatus = transfer_to(src->device, dst->device, src->data.qu8,
                                     dst->data.qu8, src->storage->size_bytes);

  status->err = map_code2err(dstatus.code);
  status->message = nova_get_error_msg(status->err, NULL);
}

/**
 * @brief Copy Signed32 elements between GPU buffers via
 *        @ref transfer_to().
 *
 * @details
 * Checks that a valid device has been detected via
 * @ref get_detected_device_kind().  If no device is available, sets
 * @p status to @ref novaDeviceNotAvailable.  Otherwise delegates to
 * @ref transfer_to() and maps the result through @ref map_code2err().
 */
static inline void copy_s32_device_buffer(const Tensor *restrict src,
                                          Tensor *restrict dst,
                                          novaStatus_t *status) {
  if (get_detected_device_kind() != NULL_DEVICE) {
    status->err = novaDeviceNotAvailable;
    status->message = "Can not copy bytes in an invalid device.\n Please check "
                      "if any device was detected previously\n";
    return;
  }
  DeviceStatus dstatus = transfer_to(src->device, dst->device, src->data.s32,
                                     dst->data.s32, src->storage->size_bytes);

  status->err = map_code2err(dstatus.code);
  status->message = nova_get_error_msg(status->err, NULL);
}

/**
 * @brief Copy UnSigned32 elements between GPU buffers via
 *        @ref transfer_to().
 *
 * @details
 * Checks that a valid device has been detected via
 * @ref get_detected_device_kind().  If no device is available, sets
 * @p status to @ref novaDeviceNotAvailable.  Otherwise delegates to
 * @ref transfer_to() and maps the result through @ref map_code2err().
 */
static inline void copy_u32_device_buffer(const Tensor *restrict src,
                                          Tensor *restrict dst,
                                          novaStatus_t *status) {
  if (get_detected_device_kind() != NULL_DEVICE) {
    status->err = novaDeviceNotAvailable;
    status->message = "Can not copy bytes in an invalid device.\n Please check "
                      "if any device was detected previously\n";
    return;
  }

  DeviceStatus dstatus = transfer_to(src->device, dst->device, src->data.u32,
                                     dst->data.u32, src->storage->size_bytes);

  status->err = map_code2err(dstatus.code);
  status->message = nova_get_error_msg(status->err, NULL);
}

/**
 * @brief Copy Signed64 elements between GPU buffers via
 *        @ref transfer_to().
 *
 * @details
 * Checks that a valid device has been detected via
 * @ref get_detected_device_kind().  If no device is available, sets
 * @p status to @ref novaDeviceNotAvailable.  Otherwise delegates to
 * @ref transfer_to() and maps the result through @ref map_code2err().
 */
static inline void copy_s64_device_buffer(const Tensor *restrict src,
                                          Tensor *restrict dst,
                                          novaStatus_t *status) {
  if (get_detected_device_kind() != NULL_DEVICE) {
    status->err = novaDeviceNotAvailable;
    status->message = "Can not copy bytes in an invalid device.\n Please check "
                      "if any device was detected previously\n";
    return;
  }

  DeviceStatus dstatus = transfer_to(src->device, dst->device, src->data.s64,
                                     dst->data.s64, src->storage->size_bytes);

  status->err = map_code2err(dstatus.code);
  status->message = nova_get_error_msg(status->err, NULL);
}

/**
 * @brief Copy UnSigned64 elements between GPU buffers via
 *        @ref transfer_to().
 *
 * @details
 * Checks that a valid device has been detected via
 * @ref get_detected_device_kind().  If no device is available, sets
 * @p status to @ref novaDeviceNotAvailable.  Otherwise delegates to
 * @ref transfer_to() and maps the result through @ref map_code2err().
 */
static inline void copy_u64_device_buffer(const Tensor *restrict src,
                                          Tensor *restrict dst,
                                          novaStatus_t *status) {
  if (get_detected_device_kind() != NULL_DEVICE) {
    status->err = novaDeviceNotAvailable;
    status->message = "Can not copy bytes in an invalid device.\n Please check "
                      "if any device was detected previously\n";
    return;
  }

  DeviceStatus dstatus = transfer_to(src->device, dst->device, src->data.u64,
                                     dst->data.u64, src->storage->size_bytes);

  status->err = map_code2err(dstatus.code);
  status->message = nova_get_error_msg(status->err, NULL);
}

/**
 * @var lookup_host_copy
 * @brief Host-to-host copy dispatch table, indexed by `DType_`.
 *
 * @details
 * A `NUM_DTYPES × 1` array of @ref CopyFn pointers.  Each entry
 * maps a dtype to its corresponding `copy_*_host_buffer()` function.
 * Used by @ref deepcopy() when `src->device == DEVICE_CPU`.
 */
const CopyFn lookup_host_copy[NUM_DTYPES][1] = {
    [Float32] = {copy_f32_host_buffer},  [Float64] = {copy_f64_host_buffer},
    [Float16] = {copy_f16_host_buffer},  [BFloat16] = {copy_bf16_host_buffer},
    [Signed8] = {copy_s8_host_buffer},   [UnSigned8] = {copy_u8_host_buffer},
    [QSigned8] = {copy_qs8_host_buffer}, [QUnSigned8] = {copy_qu8_host_buffer},
    [Signed32] = {copy_s32_host_buffer}, [UnSigned32] = {copy_u32_host_buffer},
    [Signed64] = {copy_s64_host_buffer}, [UnSigned64] = {copy_u64_host_buffer},
};

/**
 * @var lookup_device_copy
 * @brief Device-to-device copy dispatch table, indexed by `DType_`.
 *
 * @details
 * A `NUM_DTYPES × 1` array of @ref CopyFn pointers.  Each entry
 * maps a dtype to its corresponding `copy_*_device_buffer()` function.
 * Used by @ref deepcopy() when `src->device == DEVICE_GPU`.
 */
const CopyFn lookup_device_copy[NUM_DTYPES][1] = {
    [Float32] = {copy_f32_device_buffer},
    [Float64] = {copy_f64_device_buffer},
    [Float16] = {copy_f16_device_buffer},
    [BFloat16] = {copy_bf16_device_buffer},
    [Signed8] = {copy_s8_device_buffer},
    [UnSigned8] = {copy_u8_device_buffer},
    [QSigned8] = {copy_qs8_device_buffer},
    [QUnSigned8] = {copy_qu8_device_buffer},
    [Signed32] = {copy_s32_device_buffer},
    [UnSigned32] = {copy_u32_device_buffer},
    [Signed64] = {copy_s64_device_buffer},
    [UnSigned64] = {copy_u64_device_buffer},
};

/**
 * @var lookup_copy
 * @brief Top-level dispatch table mapping @ref Device to the
 *        appropriate per-dtype copy table.
 *
 * @details
 * A 2-element array indexed by @ref Device:
 * - `lookup_copy[DEVICE_CPU]` → @ref lookup_host_copy
 * - `lookup_copy[DEVICE_GPU]` → @ref lookup_device_copy
 *
 * Used by @ref deepcopy() as `lookup_copy[src->device][src->dtype]`
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
 * the @ref lookup_copy table based on `src->device` and
 * `src->dtype`.
 *
 * ## Behaviour
 *
 * 1. All metadata fields (`shape`, `strides`, `item_size`, `size`,
 *    `ndims`, `dtype`, `device`, `scale_`, `zero_point_`,
 *    `is_pinned`, gradient flags) are copied element-by-element.
 *    Fields `is_view_`, `grad_fn_`, and `offset` are set to fixed
 *    values (`false`, `NULL`, `0` respectively).
 * 2. If `src->storage` is non-NULL, a new @ref TensorStorage is
 *    allocated via @ref safe_allocator() and the data is copied
 *    using the appropriate @ref CopyFn.
 * 3. If `src->grad` is non-NULL, the gradient tensor is recursively
 *    deep-copied via a self-recursive call.  Gradient copy errors
 *    are propagated through @p status.
 * 4. The destination tensor is marked as `is_allocated_ = true`,
 *    `is_leaf_ = true`, and `is_view_ = false`.
 *
 * @param[in]  src     Source tensor.  May be `NULL` (no-op).
 * @param[out] dst     Destination tensor.  Must not be `NULL`.  Must
 *                     have `is_allocated_ == false` (i.e., created
 *                     by `create_unallocated_tensor()`).
 * @param[out] status  Receives the result of the deep-copy
 *                     operation.  On success, set to
 *                     @ref novaSuccess.  On failure, set to the
 *                     appropriate error code.
 *
 * @pre  @p dst must be an unallocated tensor.
 * @pre  If @p src has a non-NULL `storage`, its `size_bytes` must
 *       be > 0.
 * @post On success, @p dst is a complete independent copy of
 *       @p src, including gradient history.
 * @post On failure, @p status contains the error code and @p dst
 *       is left in a valid but unmodified state.
 *
 * @see CopyFn            Per-dtype copy function pointer type.
 * @see lookup_copy       Dispatch table selecting host vs device.
 * @see safe_allocator()  Storage allocator.
 * @see Device            Device placement enum.
 * @see DType_            Data-type enum used for dispatch.
 */
void deepcopy(const Tensor *restrict src, Tensor *restrict dst,
              novaStatus_t *status) {

  if (src == NULL) {
    return;
  }

  if (dst == NULL) {
    status->err = novaInvalidTensor;
    status->message = "dst Tensor ptr is NULL\n";
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
    status->message = nova_get_error_msg(status->err, NULL);
    return;
  }

  memcpy(dst->shape, src->shape, src->ndims * sizeof(size_t));
  memcpy(dst->strides, src->strides, src->ndims * sizeof(size_t));
  dst->item_size = src->item_size;
  dst->size = src->size;
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
  dst->grad_fn_ = NULL;
  dst->offset = 0;
  dst->version_ = 0;

  if (src->storage != NULL) {

    *status = safe_allocator(src->storage->size_bytes, src->device,
                             src->is_pinned, NULL, dst, true);

    if (status->err != novaSuccess) {
      return;
    }

    const CopyFn func = lookup_copy[src->device][src->dtype];
    func(src, dst, status);
    if (status->err != novaSuccess) {
      return;
    }
  }

  if (src->grad != NULL) {
    TensorGrad new_grad =
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
    dst->grad = NULL;
  }
}
