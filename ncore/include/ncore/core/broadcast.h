/**
 * @file broadcast.h
 * @brief Shape broadcasting rules for tensor groups.
 *
 * @details
 * Broadcasting stretches a group of tensors to a single shape so that
 * they can be read element by element.  The dimensions are compared
 * from the last one towards the first:  two dimensions are compatible
 * when they are equal or when one of them is 1, the result dimension
 * takes the size of the other, and a dimension missing from one of the
 * shapes counts as 1.
 *
 * The result is a view, not a copy.  A stretched dimension and a
 * missing dimension both carry a stride of 0, so the same element is
 * read more than once without touching memory.
 *
 * The group functions take arrays of pointers:  the caller owns the
 * tensor and layout elements, and the functions read and write through
 * the pointers without copying them.  Only the array pointers are
 * validated:  the upper layers (the Cython bindings and the Python
 * API) are the only callers of this family, and they never pass a
 * null element.
 *
 * @see tensor.h  broadcast(), the tensor level entry point.
 */

#pragma once

#include <ncore/core/status.h>
#include <ncore/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @struct BroadcastLayout
 * @brief Layout of a tensor, before and after broadcasting.
 *
 * @details
 * A snapshot of the shape, the strides, the rank and the storage
 * offset of one tensor.  The offsets of a group are independent:  each
 * layout keeps the offset its own tensor had.
 */
typedef struct BroadcastLayout {
  shape_t shape;     ///< Dimension sizes of the tensor.
  strides_t strides; ///< Byte strides per dimension.
  size_t ndims;      ///< Number of dimensions in @c shape.
  size_t offset;     ///< Offset of the first element inside the storage.
} BroadcastLayout;

/**
 * @brief Layout of a tensor after broadcasting.
 *
 * @details
 * A view is a layout whose shape is the broadcast shape of the group
 * and whose strides are 0 on the stretched and on the missing
 * dimensions.  Both names carry the same layout:  what separates them
 * is the contract, not the representation.
 */
typedef struct BroadcastLayout BroadcastView;

/**
 * @brief Check whether a tensor reads the same element more than once.
 *
 * @details
 * True when a dimension larger than one carries a stride of 0, which is
 * how a broadcast view exposes a stretched or a missing dimension
 * without copying data.  The property is structural, so it survives a
 * transpose or a permutation of the dimensions.
 *
 * A broadcast view counts more elements than its storage holds:  its
 * @c size is the product of the broadcast shape, while the buffer only
 * holds the elements of the source.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 * @return @c true if the tensor repeats an element.
 */
bool is_broadcast_view(const Tensor *ten);

/**
 * @brief Report whether a group of shapes broadcasts against each other.
 *
 * @param[in] shapes Per-tensor shapes.  Each entry holds @c ndims[i]
 *                   dimensions.
 * @param[in] ndims  Per-tensor dimension counts.
 * @param[in] count  Number of tensors in the group.  Zero is broadcastable.
 *
 * @return @c true when every paired dimension agrees or one side is one,
 *         @c false when two paired dimensions differ and neither is one,
 *         when a rank exceeds @ref NOVA_MAX_DIMS, or when @p shapes or
 *         @p ndims is @c nullptr with a non empty group.
 */
bool shapes_broadcastable(const shape_t shapes[], const size_t ndims[],
                          size_t count);

/**
 * @brief Compute the broadcast shape of a group of shapes.
 *
 * @param[in]  shapes    Per-tensor shapes.  Each entry holds @c ndims[i]
 *                       dimensions.
 * @param[in]  ndims     Per-tensor dimension counts.
 * @param[in]  count     Number of tensors in the group.
 * @param[out] out_shape Receives the broadcast shape, left aligned.
 * @param[out] out_ndims Receives the rank of the broadcast shape.
 *
 * @return @ref novaStatus_t with @c err set to @ref novaSuccess on
 *         success, @ref novaInvalidPointer when @p shapes, @p ndims,
 *         @p out_shape or @p out_ndims is @c nullptr, @ref novaInvalidNdims
 *         when a rank exceeds @ref NOVA_MAX_DIMS, or
 *         @ref novaInvalidShape when two dimensions are incompatible.
 */
novaStatus_t broadcast_shapes(const shape_t shapes[], const size_t ndims[],
                              size_t count, shape_t out_shape,
                              size_t *out_ndims);

/**
 * @brief Compute the broadcast shape of a group of tensors.
 *
 * @details
 * The group must be homogeneous:  tensors with different dtypes or
 * different devices cannot be broadcast against each other, because
 * the shape of a tensor is expressed in the storage units of its
 * dtype.
 *
 * @param[in]  tensors   Group of tensors.  Must not be @c nullptr.
 * @param[in]  count     Number of tensors in the group.
 * @param[out] out_shape Receives the broadcast shape, left aligned.
 * @param[out] out_ndims Receives the rank of the broadcast shape.
 *
 * @return @ref novaStatus_t with @c err set to @ref novaSuccess on
 *         success, @ref novaInvalidPointer when @p tensors, @p out_shape
 *         or @p out_ndims is @c nullptr, @ref novaInvalidNdims when a rank
 *         exceeds @ref NOVA_MAX_DIMS, @ref novaInvalidDtype or
 *         @ref novaInvalidDevice when the group is not homogeneous, or
 *         @ref novaInvalidShape when two dimensions are incompatible.
 */
novaStatus_t broadcast_shapes_from_tensors(const Tensor *const tensors[],
                                           size_t count, shape_t out_shape,
                                           size_t *out_ndims);

/**
 * @brief Snapshot the source layout of every tensor of a group.
 *
 * @param[in]  tensors Group of tensors.  Must not be @c nullptr.
 * @param[out] outputs Receives one layout per tensor.  Must not be
 *                    @c nullptr.
 * @param[in]  count   Number of tensors in the group.
 *
 * @return @ref novaStatus_t with @c err set to @ref novaSuccess on
 *         success, @ref novaInvalidPointer when an array is @c nullptr,
 *         or the error propagated from
 *         @ref broadcast_shapes_from_tensors().
 *
 * @note The group is validated but not required to hold any storage:
 *       a META tensor carries the metadata this function snapshots.
 */
novaStatus_t create_layouts_from_tensors(const Tensor *const tensors[],
                                         BroadcastLayout *const outputs[],
                                         size_t count);

/**
 * @brief Apply the broadcasting rules to a group of layouts.
 *
 * @param[in]  inputs  Source layouts of the group.
 * @param[out] outputs Receives one view per layout, all of them carrying
 *                    the broadcast shape of the group.  Must not be
 *                    @c nullptr.
 * @param[in]  counts  Number of layouts in the group.
 *
 * @return @ref novaStatus_t with @c err set to @ref novaSuccess on
 *         success, @ref novaInvalidPointer when @p inputs or @p outputs
 *         is @c nullptr, @ref novaInvalidNdims when a rank exceeds
 *         @ref NOVA_MAX_DIMS, or @ref novaInvalidShape when two
 *         dimensions are incompatible.
 *
 * @note @p outputs may be the same array as @p inputs:  the expansion
 *       copies what it reads before it writes.
 */
novaStatus_t broadcast_layouts(const BroadcastLayout *const inputs[],
                               BroadcastView *const outputs[], size_t counts);

/**
 * @brief Apply broadcast views to a group of tensors.
 *
 * @details
 * The tensors keep their storage, their data pointer and their offset:
 * only the shape, the strides and the rank change.  The size is
 * recomputed from the new shape, so it counts the elements reachable
 * through the strides and not the distinct storage units.
 *
 * The gradient and the backward node are dropped, because they describe
 * the shape the tensor had before.
 *
 * @param[in,out] tensors Tensors to reshape.  Must not be @c nullptr.
 * @param[in]     views   Views to apply.  Must not be @c nullptr.
 * @param[in]     count   Number of tensors in the group.
 *
 * @return @ref novaStatus_t with @c err set to @ref novaSuccess on
 *         success, @ref novaInvalidPointer when @p tensors or @p views
 *         is @c nullptr, or @ref novaInvalidNdims when a rank exceeds
 *         @ref NOVA_MAX_DIMS.
 */
novaStatus_t broadcast_tensors_from_layouts(Tensor *const tensors[],
                                            const BroadcastView *const views[],
                                            size_t count);

#ifdef __cplusplus
}
#endif
