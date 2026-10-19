/**
 * @file copy.h
 * @brief Tensor deep-copy interface.
 *
 * The per-dtype dispatch table @ref lookup_copy is defined in
 * @ref copy.c and stays private to that translation unit.
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
 * Copies @c src->storage->handle.requested_bytes from the source data
 * buffer into the destination buffer.
 *
 * @param[in]  src     Source tensor (read-only).  Must have
 *                     @c is_allocated_ == true and a valid @c storage
 *                     pointer.
 * @param[out] dst     Destination tensor (write-only).  Must have
 *                     @c is_allocated_ == true and a pre-allocated
 *                     @c storage of at least the size of @p src's.
 * @param[out] status  Receives the result of the copy operation.
 */
typedef void (*CopyFn)(const Tensor *restrict src, Tensor *restrict dst,
                       novaStatus_t *status);

/**
 * @brief Deep-copy a tensor, including metadata, data, and
 *        gradients.
 *
 * Allocates new storage for @p dst via @ref safe_allocator(), copies
 * all metadata and element data from @p src, and recursively
 * deep-copies the gradient subtree (gradient copy errors propagate
 * through @p status). The copy resets view linkage: @p dst becomes a
 * leaf with @c is_view_ false and @c offset 0.
 *
 * @param[in]  src     Source tensor.  May be @c nullptr (no-op).
 * @param[out] dst     Destination tensor.  Must not be @c nullptr and
 *                     must have @c is_allocated_ == false (created
 *                     by @c create_unallocated_tensor()).
 * @param[out] status  Receives the result of the deep-copy
 *                     operation.
 *
 * @pre  If @p src has a non-nullptr @c storage, its @c requested_bytes
 *       must be > 0.
 * @post On success, @p dst is a complete independent copy of @p src,
 *       including gradient history.
 * @post On failure, @p status contains the error code and @p dst is
 *       left in a valid but unmodified state.
 */
void deepcopy(const Tensor *restrict src, Tensor *restrict dst,
              novaStatus_t *status);

#ifdef __cplusplus
}
#endif
