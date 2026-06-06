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
 * 3. **Dispatch tables** — @ref lookup_cpu_copy, @ref lookup_gpu_copy,
 *    and @ref lookup_copy map `(device, dtype)` pairs to the correct
 *    @ref copyFn.
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

#include <ncore/alloc.h>
#include <ncore/copy.h>
#include <ncore/device.h>
#include <ncore/headeronly/tensor_utils.h>
#include <ncore/macros.h>
#include <ncore/tensor.h>
#include <string.h>

/* ────────────────────────────────────────────────────────────────
 *  CPU copy functions — host-to-host memcpy, one per dtype
 * ──────────────────────────────────────────────────────────────── */

/** @brief Copy Float32 elements from @p src to @p dst via `memcpy`. */
static inline void copy_f32_cpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  memcpy(dst->data.f32, src->data.f32, src->storage->size_bytes);
}

/** @brief Copy Float64 elements from @p src to @p dst via `memcpy`. */
static inline void copy_f64_cpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  memcpy(dst->data.f64, src->data.f64, src->storage->size_bytes);
}

/** @brief Copy Float16 elements from @p src to @p dst via `memcpy`. */
static inline void copy_f16_cpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  memcpy(dst->data.half, src->data.half, src->storage->size_bytes);
}

/** @brief Copy BFloat16 elements from @p src to @p dst via `memcpy`. */
static inline void copy_bf16_cpu_buffer_(const Tensor *restrict src,
                                         Tensor *restrict dst) {
  memcpy(dst->data.bf16, src->data.bf16, src->storage->size_bytes);
}

/** @brief Copy Signed8 elements from @p src to @p dst via `memcpy`. */
static inline void copy_s8_cpu_buffer_(const Tensor *restrict src,
                                       Tensor *restrict dst) {
  memcpy(dst->data.s8, src->data.s8, src->storage->size_bytes);
}

/** @brief Copy UnSigned8 elements from @p src to @p dst via `memcpy`. */
static inline void copy_u8_cpu_buffer_(const Tensor *restrict src,
                                       Tensor *restrict dst) {
  memcpy(dst->data.u8, src->data.u8, src->storage->size_bytes);
}

/** @brief Copy QSigned8 elements from @p src to @p dst via `memcpy`. */
static inline void copy_qs8_cpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  memcpy(dst->data.qs8, src->data.qs8, src->storage->size_bytes);
}

/** @brief Copy QUnSigned8 elements from @p src to @p dst via `memcpy`. */
static inline void copy_qu8_cpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  memcpy(dst->data.qu8, src->data.qu8, src->storage->size_bytes);
}

/** @brief Copy Signed32 elements from @p src to @p dst via `memcpy`. */
static inline void copy_s32_cpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  memcpy(dst->data.s32, src->data.s32, src->storage->size_bytes);
}

/** @brief Copy UnSigned32 elements from @p src to @p dst via `memcpy`. */
static inline void copy_u32_cpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  memcpy(dst->data.u32, src->data.u32, src->storage->size_bytes);
}

/** @brief Copy Signed64 elements from @p src to @p dst via `memcpy`. */
static inline void copy_s64_cpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  memcpy(dst->data.s64, src->data.s64, src->storage->size_bytes);
}

/** @brief Copy UnSigned64 elements from @p src to @p dst via `memcpy`. */
static inline void copy_u64_cpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  memcpy(dst->data.u64, src->data.u64, src->storage->size_bytes);
}

/* ────────────────────────────────────────────────────────────────
 *  GPU copy functions — device-to-device via transfer_to()
 * ──────────────────────────────────────────────────────────────── */

/**
 * @brief Copy Float32 elements between GPU buffers via
 *        @ref transfer_to().
 *
 * @details
 * Asserts that a device has been detected (via
 * @ref get_detected_device_kind()) before attempting the transfer.
 * The `is_pinned` flag is set to `false` because both source and
 * destination reside in device memory.
 */
static inline void copy_f32_gpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  NOVA_INTERNAL_ASSERT(
      get_detected_device_kind() != NULL_DEVICE,
      "[COPY] Error: Can not copy bytes in an invalid device.\n Please check "
      "if any device was detected previously\n");

  transfer_to(src->device, dst->device, src->data.f32, dst->data.f32, false,
              src->storage->size_bytes);
}

/**
 * @brief Copy Float64 elements between GPU buffers via
 *        @ref transfer_to().
 */
static inline void copy_f64_gpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {

  NOVA_INTERNAL_ASSERT(
      get_detected_device_kind() != NULL_DEVICE,
      "[COPY] Error: Can not copy bytes in an invalid device.\n Please check "
      "if any device was detected previously\n");

  transfer_to(src->device, dst->device, src->data.f64, dst->data.f64, false,
              src->storage->size_bytes);
}

/**
 * @brief Copy Float16 elements between GPU buffers via
 *        @ref transfer_to().
 */
static inline void copy_f16_gpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  NOVA_INTERNAL_ASSERT(
      get_detected_device_kind() != NULL_DEVICE,
      "[COPY] Error: Can not copy bytes in an invalid device.\n Please check "
      "if any device was detected previously\n");

  transfer_to(src->device, dst->device, src->data.half, dst->data.half, false,
              src->storage->size_bytes);
}

/**
 * @brief Copy BFloat16 elements between GPU buffers via
 *        @ref transfer_to().
 */
static inline void copy_bf16_gpu_buffer_(const Tensor *restrict src,
                                         Tensor *restrict dst) {
  NOVA_INTERNAL_ASSERT(
      get_detected_device_kind() != NULL_DEVICE,
      "[COPY] Error: Can not copy bytes in an invalid device.\n Please check "
      "if any device was detected previously\n");

  transfer_to(src->device, dst->device, src->data.bf16, dst->data.bf16, false,
              src->storage->size_bytes);
}

/**
 * @brief Copy Signed8 elements between GPU buffers via
 *        @ref transfer_to().
 */
static inline void copy_s8_gpu_buffer_(const Tensor *restrict src,
                                       Tensor *restrict dst) {
  NOVA_INTERNAL_ASSERT(
      get_detected_device_kind() != NULL_DEVICE,
      "[COPY] Error: Can not copy bytes in an invalid device.\n Please check "
      "if any device was detected previously\n");

  transfer_to(src->device, dst->device, src->data.s8, dst->data.s8, false,
              src->storage->size_bytes);
}

/**
 * @brief Copy UnSigned8 elements between GPU buffers via
 *        @ref transfer_to().
 */
static inline void copy_u8_gpu_buffer_(const Tensor *restrict src,
                                       Tensor *restrict dst) {
  NOVA_INTERNAL_ASSERT(
      get_detected_device_kind() != NULL_DEVICE,
      "[COPY] Error: Can not copy bytes in an invalid device.\n Please check "
      "if any device was detected previously\n");

  transfer_to(src->device, dst->device, src->data.u8, dst->data.u8, false,
              src->storage->size_bytes);
}

/**
 * @brief Copy QSigned8 elements between GPU buffers via
 *        @ref transfer_to().
 */
static inline void copy_qs8_gpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  NOVA_INTERNAL_ASSERT(
      get_detected_device_kind() != NULL_DEVICE,
      "[COPY] Error: Can not copy bytes in an invalid device.\n Please check "
      "if any device was detected previously\n");

  transfer_to(src->device, dst->device, src->data.qs8, dst->data.qs8, false,
              src->storage->size_bytes);
}

/**
 * @brief Copy QUnSigned8 elements between GPU buffers via
 *        @ref transfer_to().
 */
static inline void copy_qu8_gpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  NOVA_INTERNAL_ASSERT(
      get_detected_device_kind() != NULL_DEVICE,
      "[COPY] Error: Can not copy bytes in an invalid device.\n Please check "
      "if any device was detected previously\n");

  transfer_to(src->device, dst->device, src->data.qu8, dst->data.qu8, false,
              src->storage->size_bytes);
}

/**
 * @brief Copy Signed32 elements between GPU buffers via
 *        @ref transfer_to().
 */
static inline void copy_s32_gpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  NOVA_INTERNAL_ASSERT(
      get_detected_device_kind() != NULL_DEVICE,
      "[COPY] Error: Can not copy bytes in an invalid device.\n Please check "
      "if any device was detected previously\n");

  transfer_to(src->device, dst->device, src->data.s32, dst->data.s32, false,
              src->storage->size_bytes);
}

/**
 * @brief Copy UnSigned32 elements between GPU buffers via
 *        @ref transfer_to().
 */
static inline void copy_u32_gpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  NOVA_INTERNAL_ASSERT(
      get_detected_device_kind() != NULL_DEVICE,
      "[COPY] Error: Can not copy bytes in an invalid device.\n Please check "
      "if any device was detected previously\n");

  transfer_to(src->device, dst->device, src->data.u32, dst->data.u32, false,
              src->storage->size_bytes);
}

/**
 * @brief Copy Signed64 elements between GPU buffers via
 *        @ref transfer_to().
 */
static inline void copy_s64_gpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  NOVA_INTERNAL_ASSERT(
      get_detected_device_kind() != NULL_DEVICE,
      "[COPY] Error: Can not copy bytes in an invalid device.\n Please check "
      "if any device was detected previously\n");

  transfer_to(src->device, dst->device, src->data.s64, dst->data.s64, false,
              src->storage->size_bytes);
}

/**
 * @brief Copy UnSigned64 elements between GPU buffers via
 *        @ref transfer_to().
 */
static inline void copy_u64_gpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  NOVA_INTERNAL_ASSERT(
      get_detected_device_kind() != NULL_DEVICE,
      "[COPY] Error: Can not copy bytes in an invalid device.\n Please check "
      "if any device was detected previously\n");

  transfer_to(src->device, dst->device, src->data.u64, dst->data.u64, false,
              src->storage->size_bytes);
}

/**
 * @var lookup_cpu_copy
 * @brief Host-to-host copy dispatch table, indexed by `DType_`.
 *
 * @details
 * A `NUM_DTYPES × 1` array of @ref copyFn pointers.  Each entry
 * maps a dtype to its corresponding `copy_*_cpu_buffer_()` function.
 * Used by @ref deepcopy() when `src->device == DEVICE_CPU`.
 */
const copyFn lookup_cpu_copy[NUM_DTYPES][1] = {
    [Float32] = {copy_f32_cpu_buffer_},  [Float64] = {copy_f64_cpu_buffer_},
    [Float16] = {copy_f16_cpu_buffer_},  [BFloat16] = {copy_bf16_cpu_buffer_},
    [Signed8] = {copy_s8_cpu_buffer_},   [UnSigned8] = {copy_u8_cpu_buffer_},
    [QSigned8] = {copy_qs8_cpu_buffer_}, [QUnSigned8] = {copy_qu8_cpu_buffer_},
    [Signed32] = {copy_s32_cpu_buffer_}, [UnSigned32] = {copy_u32_cpu_buffer_},
    [Signed64] = {copy_s64_cpu_buffer_}, [UnSigned64] = {copy_u64_cpu_buffer_},
};

/**
 * @var lookup_gpu_copy
 * @brief Device-to-device copy dispatch table, indexed by `DType_`.
 *
 * @details
 * A `NUM_DTYPES × 1` array of @ref copyFn pointers.  Each entry
 * maps a dtype to its corresponding `copy_*_gpu_buffer_()` function.
 * Used by @ref deepcopy() when `src->device == DEVICE_GPU`.
 */
const copyFn lookup_gpu_copy[NUM_DTYPES][1] = {
    [Float32] = {copy_f32_gpu_buffer_},  [Float64] = {copy_f64_gpu_buffer_},
    [Float16] = {copy_f16_gpu_buffer_},  [BFloat16] = {copy_bf16_gpu_buffer_},
    [Signed8] = {copy_s8_gpu_buffer_},   [UnSigned8] = {copy_u8_gpu_buffer_},
    [QSigned8] = {copy_qs8_gpu_buffer_}, [QUnSigned8] = {copy_qu8_gpu_buffer_},
    [Signed32] = {copy_s32_gpu_buffer_}, [UnSigned32] = {copy_u32_gpu_buffer_},
    [Signed64] = {copy_s64_gpu_buffer_}, [UnSigned64] = {copy_u64_gpu_buffer_},
};

/**
 * @var lookup_copy
 * @brief Top-level dispatch table mapping @ref Device to the
 *        appropriate per-dtype copy table.
 *
 * @details
 * A 2-element array indexed by @ref Device:
 * - `lookup_copy[DEVICE_CPU]` → @ref lookup_cpu_copy
 * - `lookup_copy[DEVICE_GPU]` → @ref lookup_gpu_copy
 *
 * Used by @ref deepcopy() as `lookup_copy[src->device][src->dtype]`
 * to resolve the correct @ref copyFn in a single array lookup.
 */
const copyFn *lookup_copy[2] = {
    [DEVICE_CPU] = (copyFn *)lookup_cpu_copy,
    [DEVICE_GPU] = (copyFn *)lookup_gpu_copy,
};

/**
 * @brief Function pointer type for a per-dtype tensor copy routine.
 *
 * @details
 * Every dtype that the framework supports has a dedicated copy
 * function matching this signature.  The function copies
 * `src->storage->size_bytes` from the source tensor's data buffer
 * into the destination tensor's data buffer.
 *
 * @param[in]  src  Source tensor (read-only).  Must have
 *                  `is_allocated_ == true` and a valid `storage`
 *                  pointer.
 * @param[out] dst  Destination tensor (write-only).  Must have
 *                  `is_allocated_ == true` and a pre-allocated
 *                  `storage` of at least the same size as @p src.
 *
 * @pre  Both @p src and @p dst must have non-NULL, allocated storage.
 * @pre  `dst->storage->size_bytes >= src->storage->size_bytes`.
 * @post On success, `dst->data` contains a bitwise copy of
 *       `src->data`.
 */
typedef void (*copyFn)(const Tensor *restrict src, Tensor *restrict dst);

/**
 * @brief Deep-copy a tensor, including metadata, data, and
 *        gradients.
 *
 * @details
 * Allocates new storage for @p dst, copies all metadata and element
 * data from @p src, and recursively deep-copies the gradient
 * subtree.  The copy is dispatched through the @ref lookup_copy
 * table based on `src->device` and `src->dtype`.
 *
 * ## Behaviour
 *
 * 1. All metadata fields (`shape`, `strides`, `item_size`, `size`,
 *    `ndims`, `dtype`, `device`, `scale_`, `zero_point_`,
 *    `is_pinned`, gradient flags) are copied element-by-element.
 *    Fields `is_view_`, `grad_fn_`, and `offset` are set to fixed
 *    values (`false`, `NULL`, `0` respectively).
 * 2. If `src->storage` is non-NULL, a new @ref TensorStorage is
 *    allocated via @ref allocate_tensor_buffer() and the data is
 *    copied using the appropriate @ref copyFn.
 * 3. If `src->grad` is non-NULL, the gradient tensor is recursively
 *    deep-copied via a self-recursive call.
 * 4. The destination tensor is marked as `is_allocated_ = true`,
 *    `is_leaf_ = true`, and `is_view_ = false`.
 *
 * @param[in]  src  Source tensor.  May be `NULL` (no-op).
 * @param[out] dst  Destination tensor.  Must not be `NULL`.  Must
 *                  have `is_allocated_ == false` (i.e., created by
 *                  `create_unallocated_tensor()`).
 *
 * @pre  @p dst must be an unallocated tensor.
 * @pre  If @p src has a non-NULL `storage`, its `size_bytes` must
 *       be > 0.
 * @post On success, @p dst is a complete independent copy of
 *       @p src, including gradient history.
 * @post On failure (assertion), the process aborts.
 *
 * @see copyFn            Per-dtype copy function pointer type.
 * @see lookup_copy       Dispatch table selecting CPU vs GPU copy.
 * @see allocate_tensor_buffer()  Storage allocator.
 * @see Device            Device placement enum.
 * @see DType_            Data-type enum used for dispatch.
 */
void deepcopy(const Tensor *restrict src, Tensor *restrict dst) {

  if (src == NULL) {
    return;
  }

  NOVA_INTERNAL_ASSERT(dst != NULL,
                       "[COPY] deepcopy: dst Tensor ptr is NULL\n");

  NOVA_INTERNAL_ASSERT(!dst->is_allocated_,
                       "[COPY] deepcopy: dst must be an unallocated tensor "
                       "created by create_unallocated_tensor()\n");

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
    TensorStorage *new_storage = allocate_tensor_buffer(
        src->storage->size_bytes, src->device, src->is_pinned);

    NOVA_INTERNAL_ASSERT(new_storage != NULL,
                         "[STORAGE] deepcopy: CPU/GPU tensor must have "
                         "non-NULL storage, but allocation returned NULL\n");

    dst->storage = new_storage;
    dst->data = new_storage->ptr;
    dst->is_allocated_ = true;

    const copyFn func = lookup_copy[src->device][src->dtype];
    func(src, dst);
  }

  if (src->grad != NULL) {
    TensorGrad new_grad =
        create_unallocated_grad_tensor(src->grad->shape, src->grad->dtype,
                                       src->grad->device, src->grad->is_pinned,
                                       src->grad->ndims);
    dst->grad = new_grad;
    deepcopy(src->grad, dst->grad);
  } else {
    dst->grad = NULL;
  }
}
