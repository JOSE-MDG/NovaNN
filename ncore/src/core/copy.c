/**
 * @file copy.c
 * @brief Implementation of typed tensor buffer copy routines.
 *
 * Provides per-dtype CPU and GPU (stub) copy implementations, along
 * with device-indexed lookup tables and the top-level deepcopy()
 * entry point.
 *
 * CPU routines use memcpy() through the correct data_ptr union member.
 * GPU routines are stubs awaiting a CUDA/ROCm backend.
 * Lookup tables are indexed first by device, then by DType_.
 */

#include <ncore/alloc.h>
#include <ncore/copy.h>
#include <ncore/macros.h>
#include <ncore/tensor.h>
#include <string.h>

/* =========================================================================
 * CPU implementations (memcpy through typed data_ptr members)
 * ========================================================================= */

/**
 * @brief Copy a 32-bit float tensor buffer on CPU.
 * @param src Source tensor.
 * @param dst Destination tensor.
 */
static inline void copy_f32_cpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  memcpy(dst->data.f32, src->data.f32, src->item_size * src->size);
}

/**
 * @brief Copy a 64-bit float (double) tensor buffer on CPU.
 * @param src Source tensor.
 * @param dst Destination tensor.
 */
static inline void copy_f64_cpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  memcpy(dst->data.f64, src->data.f64, src->item_size * src->size);
}

/**
 * @brief Copy a half-precision (16-bit float) tensor buffer on CPU.
 * @param src Source tensor.
 * @param dst Destination tensor.
 */
static inline void copy_f16_cpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  memcpy(dst->data.half, src->data.half, src->item_size * src->size);
}

/**
 * @brief Copy a bfloat16 tensor buffer on CPU.
 * @param src Source tensor.
 * @param dst Destination tensor.
 */
static inline void copy_bf16_cpu_buffer_(const Tensor *restrict src,
                                         Tensor *restrict dst) {
  memcpy(dst->data.bf16, src->data.bf16, src->item_size * src->size);
}

/**
 * @brief Copy a signed 8-bit integer tensor buffer on CPU.
 * @param src Source tensor.
 * @param dst Destination tensor.
 */
static inline void copy_s8_cpu_buffer_(const Tensor *restrict src,
                                       Tensor *restrict dst) {
  memcpy(dst->data.s8, src->data.s8, src->item_size * src->size);
}

/**
 * @brief Copy an unsigned 8-bit integer tensor buffer on CPU.
 * @param src Source tensor.
 * @param dst Destination tensor.
 */
static inline void copy_u8_cpu_buffer_(const Tensor *restrict src,
                                       Tensor *restrict dst) {
  memcpy(dst->data.u8, src->data.u8, src->item_size * src->size);
}

/**
 * @brief Copy a quantized signed 8-bit integer tensor buffer on CPU.
 * @param src Source tensor.
 * @param dst Destination tensor.
 */
static inline void copy_qs8_cpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  memcpy(dst->data.qs8, src->data.qs8, src->item_size * src->size);
}

/**
 * @brief Copy a quantized unsigned 8-bit integer tensor buffer on CPU.
 * @param src Source tensor.
 * @param dst Destination tensor.
 */
static inline void copy_qu8_cpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  memcpy(dst->data.qu8, src->data.qu8, src->item_size * src->size);
}

/**
 * @brief Copy a signed 32-bit integer tensor buffer on CPU.
 * @param src Source tensor.
 * @param dst Destination tensor.
 */
static inline void copy_s32_cpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  memcpy(dst->data.s32, src->data.s32, src->item_size * src->size);
}

/**
 * @brief Copy an unsigned 32-bit integer tensor buffer on CPU.
 * @param src Source tensor.
 * @param dst Destination tensor.
 */
static inline void copy_u32_cpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  memcpy(dst->data.u32, src->data.u32, src->item_size * src->size);
}

/**
 * @brief Copy a signed 64-bit integer tensor buffer on CPU.
 * @param src Source tensor.
 * @param dst Destination tensor.
 */
static inline void copy_s64_cpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  memcpy(dst->data.s64, src->data.s64, src->item_size * src->size);
}

/**
 * @brief Copy an unsigned 64-bit integer tensor buffer on CPU.
 * @param src Source tensor.
 * @param dst Destination tensor.
 */
static inline void copy_u64_cpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  memcpy(dst->data.u64, src->data.u64, src->item_size * src->size);
}

/* =========================================================================
 * GPU implementations (stubs — CUDA/ROCm backend not yet integrated)
 * ========================================================================= */

/** @brief GPU copy stub for 32-bit float (TODO). */
static inline void copy_f32_gpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  // TODO: Implement buffer copy with CUDA/ROCM
}

/** @brief GPU copy stub for 64-bit float (TODO). */
static inline void copy_f64_gpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  // TODO: Implement buffer copy with CUDA/ROCM
}

/** @brief GPU copy stub for half-precision 16-bit float (TODO). */
static inline void copy_f16_gpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  // TODO: Implement buffer copy with CUDA/ROCM
}

/** @brief GPU copy stub for bfloat16 (TODO). */
static inline void copy_bf16_gpu_buffer_(const Tensor *restrict src,
                                         Tensor *restrict dst) {
  // TODO: Implement buffer copy with CUDA/ROCM
}

/** @brief GPU copy stub for signed 8-bit integer (TODO). */
static inline void copy_s8_gpu_buffer_(const Tensor *restrict src,
                                       Tensor *restrict dst) {
  // TODO: Implement buffer copy with CUDA/ROCM
}

/** @brief GPU copy stub for unsigned 8-bit integer (TODO). */
static inline void copy_u8_gpu_buffer_(const Tensor *restrict src,
                                       Tensor *restrict dst) {
  // TODO: Implement buffer copy with CUDA/ROCM
}

/** @brief GPU copy stub for quantized signed 8-bit integer (TODO). */
static inline void copy_qs8_gpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  // TODO: Implement buffer copy with CUDA/ROCM
}

/** @brief GPU copy stub for quantized unsigned 8-bit integer (TODO). */
static inline void copy_qu8_gpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  // TODO: Implement buffer copy with CUDA/ROCM
}

/** @brief GPU copy stub for signed 32-bit integer (TODO). */
static inline void copy_s32_gpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  // TODO: Implement buffer copy with CUDA/ROCM
}

/** @brief GPU copy stub for unsigned 32-bit integer (TODO). */
static inline void copy_u32_gpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  // TODO: Implement buffer copy with CUDA/ROCM
}

/** @brief GPU copy stub for signed 64-bit integer (TODO). */
static inline void copy_s64_gpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  // TODO: Implement buffer copy with CUDA/ROCM
}

/** @brief GPU copy stub for unsigned 64-bit integer (TODO). */
static inline void copy_u64_gpu_buffer_(const Tensor *restrict src,
                                        Tensor *restrict dst) {
  // TODO: Implement buffer copy with CUDA/ROCM
}

/* =========================================================================
 * Lookup tables — dispatch by device then by dtype
 * ========================================================================= */

/**
 * @brief CPU copy dispatch table indexed by DType_.
 * Each entry points to the corresponding copy_*_cpu_buffer_() function.
 */
static copyFn lookup_cpu_copy[NUM_DTYPES][1] = {
    [Float32] = {copy_f32_cpu_buffer_},  [Float64] = {copy_f64_cpu_buffer_},
    [Float16] = {copy_f16_cpu_buffer_},  [BFloat16] = {copy_bf16_cpu_buffer_},
    [Signed8] = {copy_s8_cpu_buffer_},   [UnSigned8] = {copy_u8_cpu_buffer_},
    [QSigned8] = {copy_qs8_cpu_buffer_}, [QUnSigned8] = {copy_qu8_cpu_buffer_},
    [Signed32] = {copy_s32_cpu_buffer_}, [UnSigned32] = {copy_u32_cpu_buffer_},
    [Signed64] = {copy_s64_cpu_buffer_}, [UnSigned64] = {copy_u64_cpu_buffer_},
};

/**
 * @brief GPU copy dispatch table indexed by DType_ (all stubs).
 * Each entry points to the corresponding copy_*_gpu_buffer_() function.
 */
static copyFn lookup_gpu_copy[NUM_DTYPES][1] = {
    [Float32] = {copy_f32_gpu_buffer_},  [Float64] = {copy_f64_gpu_buffer_},
    [Float16] = {copy_f16_gpu_buffer_},  [BFloat16] = {copy_bf16_gpu_buffer_},
    [Signed8] = {copy_s8_gpu_buffer_},   [UnSigned8] = {copy_u8_gpu_buffer_},
    [QSigned8] = {copy_qs8_gpu_buffer_}, [QUnSigned8] = {copy_qu8_gpu_buffer_},
    [Signed32] = {copy_s32_gpu_buffer_}, [UnSigned32] = {copy_u32_gpu_buffer_},
    [Signed64] = {copy_s64_gpu_buffer_}, [UnSigned64] = {copy_u64_gpu_buffer_},
};

/**
 * @brief Top-level copy dispatch table indexed by device.
 *
 *   lookup_copy[DEVICE_CPU] -> lookup_cpu_copy[]
 *   lookup_copy[DEVICE_GPU] -> lookup_gpu_copy[]
 *
 * The caller retrieves lookup_copy[src->device], then indexes
 * by src->dtype to obtain the matching copy function pointer.
 */
static copyFn *lookup_copy[2] = {
    [DEVICE_CPU] = (copyFn *)lookup_cpu_copy,
    [DEVICE_GPU] = (copyFn *)lookup_gpu_copy,
};

/**
 * @brief Deep-copy a tensor: allocate new storage and copy element data.
 *
 * Copies all static metadata from src to dst, allocates a fresh buffer
 * via allocate_tensor_buffer(), then dispatches through the
 * device/dtype lookup table to perform the actual element copy.
 * Gradients are recursively deep-copied when present.
 *
 * @param src Source tensor (must not be NULL).
 * @param dst Pre-allocated destination tensor (must NOT be is_allocated_).
 */
void deepcopy(const Tensor *restrict src, Tensor *restrict dst) {

  if (src == NULL) {
    return;
  }

  NOVA_INTERNAL_ASSERT(dst != NULL, "[COPY] deepcopy: dst is NULL\n")

  NOVA_INTERNAL_ASSERT(
      !dst->is_allocated_,
      "[COPY] deepcopy: dst must be an unallocated tensor created by "
      "create_unallocated_tensor()")

  // Copy static members
  memcpy(dst->shape, src->shape, src->ndims * sizeof(size_t));
  memcpy(dst->strides, src->strides, src->ndims * sizeof(size_t));
  dst->item_size = src->item_size;
  dst->size = src->size;
  dst->ndims = src->ndims;
  dst->dtype = src->dtype;
  dst->device = src->device;
  dst->scale_ = src->scale_;
  dst->requires_grad_ = src->requires_grad_;
  dst->retain_grad_ = src->retain_grad_;
  dst->is_leaf_ = true;
  dst->is_view_ = false;
  dst->grad_fn_ = NULL;
  dst->offset = 0;

  if (src->storage != NULL) {
    TensorStorage *new_storage =
        allocate_tensor_buffer(src->storage->size_bytes, src->device);

    NOVA_INTERNAL_ASSERT(new_storage != NULL,
                         "[STORAGE] deepcopy: CPU/GPU tensor must have "
                         "non-NULL storage, but allocation returned NULL\n")

    dst->storage = new_storage;
    dst->data = new_storage->ptr;
    dst->is_allocated_ = true;

    // Copy buffer and storage
    copyFn func = lookup_copy[src->device][src->dtype];
    func(src, dst);
  }

  // Copy grad recursively
  if (src->grad != NULL) {
    TensorGrad new_grad =
        create_unallocated_grad_tensor(src->grad->shape, src->grad->dtype,
                                       src->grad->device, src->grad->ndims);
    dst->grad = new_grad;
    deepcopy(src->grad, dst->grad);
  } else {
    dst->grad = NULL;
  }
}
