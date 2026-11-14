/**
 * @file broadcast_utils.h
 * @brief Header-only broadcast rules.
 */

#pragma once

#include <stddef.h>
#include <string.h>

#include <ncore/headeronly/macros.h>
#include <ncore/tensor.h>

/**
 * @struct BroadcastAccum
 * @brief Accumulator for the broadcast shape of a group.
 *
 * @details
 * Dimensions are indexed from the innermost: position 0 is the last
 * dimension.  Indexing that way makes the right alignment automatic,
 * so an element with fewer dimensions simply does not contribute to the
 * outer positions.
 *
 * @c sources[pos] holds one more than the element that fixed the size,
 * so zero means the position was never contributed to and counts as a
 * dimension of size one.  The offset of one also tells an absent
 * dimension apart from a dimension that really is zero.
 */
typedef struct {
  size_t
      sizes[NOVA_MAX_DIMS]; ///< Size per dimension, counted from the innermost.
  size_t sources[NOVA_MAX_DIMS]; ///< Element plus one that fixed each size.
  size_t ndims;                  ///< Largest rank among the group elements.
} BroadcastAccum;

/**
 * @brief Compute the size of a result dimension.
 *
 * @details
 * Two dimensions are compatible when they are equal or when one of them
 * is 1.  When one is 1, the size of the result dimension is the size of
 * the other.
 *
 * @param[in]  left   Size of the left dimension.
 * @param[in]  right  Size of the right dimension.
 * @param[out] out    Receives the result size.
 *
 * @return @c true when the two dimensions are compatible.
 */
static inline bool compatible_dim_size_(size_t left, size_t right,
                                        size_t *out) {
  if (left == right) {
    *out = left;
    return true;
  }
  if (left == 1) {
    *out = right;
    return true;
  }
  if (right == 1) {
    *out = left;
    return true;
  }
  return false;
}

/**
 * @brief Merge the dimensions of one element into the accumulator.
 *
 * @details
 * An element with fewer dimensions does not contribute to the outer
 * positions: those dimensions are assumed to have size one, which makes
 * them compatible with any size.
 *
 * @param[in,out] acc              Accumulator of the group.
 * @param[in]     shape            Shape of the element, left aligned.
 * @param[in]     ndims            Rank of the element shape.
 * @param[in]     element          Index of the element in the group.
 * @param[out]    conflict_pos     Receives the position (counted from the
 *                                 innermost) of the incompatible dimension.
 * @param[out]    conflict_element Receives the element that carries the
 *                                 size that disagrees.
 *
 * @return @c true when every merged dimension is compatible.
 */
static inline bool broadcast_merge_(BroadcastAccum *acc, const shape_t shape,
                                    size_t ndims, size_t element,
                                    size_t *conflict_pos,
                                    size_t *conflict_element) {
  if (ndims > acc->ndims) {
    acc->ndims = ndims;
  }
  for (size_t pos = 0; pos < ndims; ++pos) {
    const size_t current = shape[ndims - 1 - pos];
    const size_t left = (acc->sources[pos] == 0) ? 1 : acc->sizes[pos];
    size_t merged = 0;
    if (!compatible_dim_size_(left, current, &merged)) {
      *conflict_pos = pos;
      *conflict_element = element;
      return false;
    }
    acc->sizes[pos] = merged;
    acc->sources[pos] = element + 1;
  }
  return true;
}

/**
 * @brief Expand a shape and its strides to the broadcast shape.
 *
 * @details
 * A stretched dimension (size 1 in the source) and a missing dimension
 * (the source has fewer dimensions) receive stride 0, so the view reads
 * the same element more than once without copying data.  Every other
 * dimension keeps its source stride.
 *
 * The source is copied before it is written, so the destination may
 * overlap the source:  a layout with fewer dimensions reads dimensions
 * that the expansion has already overwritten.
 *
 * The operation is idempotent: expanding an already expanded layout
 * leaves it unchanged.
 *
 * @param[in]  src_shape        Source shape, left aligned.
 * @param[in]  src_strides      Source strides, in bytes.
 * @param[in]  src_ndims        Rank of the source shape.
 * @param[in]  broadcast_shape  Broadcast shape of the group.
 * @param[in]  broadcast_ndims  Rank of the broadcast shape.
 * @param[out] out_shape        Receives the expanded shape.
 * @param[out] out_strides      Receives the expanded strides.
 *
 * @pre  @p broadcast_ndims is at least @p src_ndims, because it is the
 *       largest rank of the group.
 */
static inline void expand_layout_(const shape_t src_shape,
                                  const strides_t src_strides, size_t src_ndims,
                                  const shape_t broadcast_shape,
                                  size_t broadcast_ndims, shape_t out_shape,
                                  strides_t out_strides) {
  shape_t src = {};
  strides_t src_str = {};
  const size_t lead = broadcast_ndims - src_ndims;
  memcpy(src, src_shape, src_ndims * sizeof(size_t));
  memcpy(src_str, src_strides, src_ndims * sizeof(size_t));

  for (size_t d = 0; d < broadcast_ndims; ++d) {
    const size_t src_dim = (d >= lead) ? src[d - lead] : 1;
    out_shape[d] = broadcast_shape[d];
    out_strides[d] = (d < lead || (src_dim == 1 && broadcast_shape[d] != 1))
                         ? 0
                         : src_str[d - lead];
  }
}
