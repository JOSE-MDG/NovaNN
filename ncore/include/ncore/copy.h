/**
 * @file copy.h
 * @brief Tensor deep-copy interface.
 *
 * @details
 * Declares the @ref copyFn function-pointer type and the top-level
 * @ref deepcopy() entry point.  The per-dtype dispatch tables
 * (@ref lookup_cpu_copy, @ref lookup_gpu_copy) are defined in
 * @ref copy.c and remain private to that translation unit.
 *
 * @see copy.c      Implementation of per-dtype copy routines.
 * @see tensor.h    Tensor structure and data-layout details.
 * @see device.h    Device placement and inter-device transfers.
 */

#pragma once

#include <ncore/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

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
void deepcopy(const Tensor *restrict src, Tensor *restrict dst);

#ifdef __cplusplus
}
#endif
