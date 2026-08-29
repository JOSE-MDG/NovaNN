/**
 * @file copy.h
 * @brief Tensor deep-copy interface.
 *
 * @details
 * Declares the @ref CopyFn function-pointer type and the top-level
 * @ref deepcopy() entry point.  The per-dtype dispatch tables
 * (@ref lookup_host_copy, @ref lookup_device_copy) are defined in
 * @ref copy.c and remain private to that translation unit.
 *
 * @see copy.c      Implementation of per-dtype copy routines.
 * @see tensor.h    Tensor structure and data-layout details.
 * @see device.h    Device placement and inter-device transfers.
 */

#pragma once

#include <ncore/core/status.h>
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
 * @c src->storage->size_bytes from the source tensor's data buffer
 * into the destination tensor's data buffer.
 *
 * @param[in]  src     Source tensor (read-only).  Must have
 *                     @c is_allocated_ == true and a valid @c storage
 *                     pointer.
 * @param[out] dst     Destination tensor (write-only).  Must have
 *                     @c is_allocated_ == true and a pre-allocated
 *                     @c storage of at least the same size as @p src.
 * @param[out] status  Receives the result of the copy operation.
 *                     On success, set to @ref novaSuccess.  On
 *                     failure, set to the appropriate error code.
 *
 * @pre  Both @p src and @p dst must have non-nullptr, allocated storage.
 * @pre  @c dst->storage->size_bytes >= src->storage->size_bytes.
 * @post On success, @c dst->data contains a bitwise copy of
 *       @c src->data and @p status is @ref novaSuccess.
 * @post On failure, @p status contains the error code.
 */
typedef void (*CopyFn)(const Tensor *restrict src, Tensor *restrict dst,
                       novaStatus_t *status);

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
 *    @c is_pinned_, gradient flags) are copied element-by-element.
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
              novaStatus_t *status);

#ifdef __cplusplus
}
#endif
