/**
 * @file broadcast.c
 * @brief Implementation of the shape broadcasting rules.
 *
 * @details
 * The pairing rule compares dimensions from the last one towards the
 * first.  Two dimensions are compatible when they are equal or when
 * one of them is 1, the result dimension takes the size of the other,
 * and a dimension missing from one of the shapes counts as 1.
 *
 * The accumulator of @ref BroadcastAccum indexes the dimensions from
 * the innermost, so the right alignment needs no knowledge of the
 * largest rank in the group.  Every collection that carries shapes
 * feeds the same accumulator.
 *
 * Every entry point validates the rank of each element before it is
 * accumulated:  @ref broadcast_merge_ indexes the accumulator with the
 * position of each dimension, so a rank above @ref NOVA_MAX_DIMS would
 * read and write out of bounds.
 */

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <ncore/core/broadcast.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/broadcast_utils.h>
#include <ncore/headeronly/macros.h>
#include <ncore/headeronly/tensor_utils.h>
#include <ncore/tensor.h>

/**
 * @var err_msg_buf
 * @brief Scratch space for formatted error messages.
 *
 * @details Thread-local so concurrent failures never clobber each
 * other, mirroring the Rust side (@c STATUS_MESSAGE in
 * @c status.rs).  Valid until the next formatted error on the same
 * thread;  callers consume it immediately through @c status.message.
 */
static thread_local char err_msg_buf[512];

/**
 * @brief Build a status carrying a formatted message.
 *
 * @param[in] err  Error code to report.
 * @param[in] fmt  @c printf-style format string.
 * @param ...  Format arguments.
 *
 * @return Status with @c err set to @p err and @c message pointing at
 *         the formatted text.
 */
ATTR(format(printf, 2, 3)) static inline novaStatus_t
broadcast_error(novaError_t err, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vsnprintf(err_msg_buf, sizeof(err_msg_buf), fmt, args);
  va_end(args);
  return (novaStatus_t){.err = err, .message = err_msg_buf};
}

/// @brief Report the success status
static inline novaStatus_t broadcast_ok() {
  return (novaStatus_t){.err = novaSuccess,
                        .message = nova_get_error_msg(novaSuccess, nullptr)};
}

/**
 * @brief Report a pair of dimensions that cannot be broadcast together.
 *
 * @param[in] err        Error code to report.
 * @param[in] noun       Name of the group element, as the message reads it.
 * @param[in] position   Position of the dimension, counted from the
 *                       innermost.
 * @param[in] left       Size the dimension had before the conflict.
 * @param[in] left_index Element that fixed that size.
 * @param[in] right      Size the dimension carries in the failing element.
 * @param[in] right_index Element that carries the failing size.
 *
 * @return Status with @c err set to @p err and a message naming both
 *         dimensions and both elements.
 */
static inline novaStatus_t
broadcast_incompatible(novaError_t err, const char *noun, size_t position,
                       size_t left, size_t left_index, size_t right,
                       size_t right_index) {
  return broadcast_error(
      err,
      "The %s cannot be broadcast together.\nDimension -%zu pairs size %zu of "
      "%s %zu with size %zu of %s %zu,\nand neither of them is one, so no "
      "size fits both.\n",
      noun, position + 1, left, noun, left_index, right, noun, right_index);
}

/**
 * @brief Merge a group of shapes into the accumulator.
 *
 * @param[in,out] acc              Accumulator of the group.
 * @param[in]     shapes           Per-tensor shapes.
 * @param[in]     ndims            Per-tensor dimension counts.
 * @param[in]     count            Number of tensors in the group.
 * @param[out]    conflict_pos     Receives the position (counted from the
 *                                 innermost) of the incompatible dimension.
 * @param[out]    conflict_element Receives the element that carries the
 *                                 size that disagrees.
 *
 * @return @c true when every merged dimension is compatible.
 */
static inline bool
broadcast_accumulate_shapes(BroadcastAccum *acc, const shape_t shapes[],
                            const size_t ndims[], size_t count,
                            size_t *restrict conflict_pos,
                            size_t *restrict conflict_element) {
  for (size_t i = 0; i < count; ++i) {
    if (!broadcast_merge_(acc, shapes[i], ndims[i], i, conflict_pos,
                          conflict_element)) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Merge a group of tensors into the accumulator.
 *
 * @param[in,out] acc              Accumulator of the group.
 * @param[in]     tensors          Group of tensors.
 * @param[in]     count            Number of tensors in the group.
 * @param[out]    conflict_pos     Receives the position (counted from the
 *                                 innermost) of the incompatible dimension.
 * @param[out]    conflict_element Receives the element that carries the
 *                                 size that disagrees.
 *
 * @return @c true when every merged dimension is compatible.
 */
static inline bool
broadcast_accumulate_tensors(BroadcastAccum *acc, const Tensor *const tensors[],
                             size_t count, size_t *restrict conflict_pos,
                             size_t *restrict conflict_element) {
  for (size_t i = 0; i < count; ++i) {
    if (!broadcast_merge_(acc, tensors[i]->shape, tensors[i]->ndims, i,
                          conflict_pos, conflict_element)) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Merge a group of layouts into the accumulator.
 *
 * @param[in,out] acc              Accumulator of the group.
 * @param[in]     inputs           Source layouts of the group.
 * @param[in]     count            Number of layouts in the group.
 * @param[out]    conflict_pos     Receives the position (counted from the
 *                                 innermost) of the incompatible dimension.
 * @param[out]    conflict_element Receives the element that carries the
 *                                 size that disagrees.
 *
 * @return @c true when every merged dimension is compatible.
 */
static inline bool broadcast_accumulate_layouts(
    BroadcastAccum *acc, const BroadcastLayout *const inputs[], size_t count,
    size_t *restrict conflict_pos, size_t *restrict conflict_element) {
  for (size_t i = 0; i < count; ++i) {
    if (!broadcast_merge_(acc, inputs[i]->shape, inputs[i]->ndims, i,
                          conflict_pos, conflict_element)) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Write the accumulated broadcast shape, left aligned.
 *
 * @param[in]  acc       Accumulator of the group.
 * @param[out] out_shape Receives the broadcast shape.
 * @param[out] out_ndims Receives the rank of the broadcast shape.
 */
static inline void broadcast_shape_of(const BroadcastAccum *acc,
                                      shape_t out_shape, size_t *out_ndims) {
  for (size_t pos = 0; pos < acc->ndims; ++pos) {
    out_shape[acc->ndims - 1 - pos] = acc->sizes[pos];
  }
  *out_ndims = acc->ndims;
}

/**
 * @brief Check whether a tensor reads the same element more than once.
 *
 * @details
 * A dimension larger than one carrying a stride of 0 is the signature
 * of a broadcast view:  the dimension was stretched, so every position
 * along it reads the same element.  The check is structural, so a
 * transposed or permuted broadcast view still answers true.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the tensor repeats an element.
 */
bool is_broadcast_view(const Tensor *ten) {
  for (size_t dim = 0; dim < ten->ndims; ++dim) {
    if (ten->shape[dim] > 1 && ten->strides[dim] == 0) {
      return true;
    }
  }
  return false;
}

/**
 * @brief Report whether a group of shapes broadcasts against each other.
 *
 * @details
 * Dimensions are compared from the last one towards the first.  Two
 * dimensions are compatible when they are equal or when one of them is
 * 1, and a dimension missing from one of the shapes counts as 1.
 *
 * @param[in] shapes Per-tensor shapes.  Each entry holds @c ndims[i]
 *                   dimensions.
 * @param[in] ndims  Per-tensor dimension counts.
 * @param[in] count  Number of tensors in the group.  Zero is broadcastable.
 *
 * @return @c true when every paired dimension agrees or one side is one,
 *         @c false when two paired dimensions differ and neither is one
 *         or when a rank exceeds @ref NOVA_MAX_DIMS.
 */
bool shapes_broadcastable(const shape_t shapes[], const size_t ndims[],
                          size_t count) {
  if (count == 0) {
    return true;
  }
  if (shapes == nullptr || ndims == nullptr) {
    return false;
  }

  BroadcastAccum acc = {};
  size_t conflict = 0;
  size_t conflict_element = 0;
  return broadcast_accumulate_shapes(&acc, shapes, ndims, count, &conflict,
                                     &conflict_element);
}

/**
 * @brief Compute the broadcast shape of a group of shapes.
 *
 * @param[in]  shapes    Per-tensor shapes.
 * @param[in]  ndims     Per-tensor dimension counts.
 * @param[in]  count     Number of tensors in the group.
 * @param[out] out_shape Receives the broadcast shape, left aligned.
 * @param[out] out_ndims Receives the rank of the broadcast shape.
 *
 * @return @ref novaStatus_t describing the outcome.
 */
novaStatus_t broadcast_shapes(const shape_t shapes[], const size_t ndims[],
                              size_t count, shape_t out_shape,
                              size_t *out_ndims) {
  if (out_shape == nullptr || out_ndims == nullptr) {
    return (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Broadcasting needs a destination shape and its rank. Got "
                   "null.\n"};
  }
  if (count == 0) {
    memset(out_shape, 0, sizeof(shape_t));
    *out_ndims = 0;
    return broadcast_ok();
  }
  if (shapes == nullptr || ndims == nullptr) {
    return (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Broadcasting needs a live shape group and its dimension "
                   "counts. Got null.\n"};
  }

  BroadcastAccum acc = {};
  size_t conflict = 0;
  size_t conflict_element = 0;
  if (!broadcast_accumulate_shapes(&acc, shapes, ndims, count, &conflict,
                                   &conflict_element)) {
    const size_t right =
        shapes[conflict_element][ndims[conflict_element] - conflict - 1];
    return broadcast_incompatible(
        novaInvalidShape, "shape", conflict, acc.sizes[conflict],
        acc.sources[conflict] - 1, right, conflict_element);
  }

  broadcast_shape_of(&acc, out_shape, out_ndims);
  return broadcast_ok();
}

/**
 * @brief Compute the broadcast shape of a group of tensors.
 *
 * @details
 * The group must be homogeneous:  the shape of a tensor is expressed in
 * the storage units of its dtype, so tensors with different dtypes do
 * not describe comparable shapes.  Tensors on different devices do
 * not share a buffer either.
 *
 * A packed dtype (packing factor above 1) stores several logical values
 * in one storage unit along the last dim, so the innermost logical dim
 * is never of size one and cannot be stretched:  the last storage dim
 * of every non-scalar member must agree.
 *
 * @param[in]  tensors   Group of tensors.
 * @param[in]  count     Number of tensors in the group.
 * @param[out] out_shape Receives the broadcast shape, left aligned.
 * @param[out] out_ndims Receives the rank of the broadcast shape.
 *
 * @return @ref novaStatus_t describing the outcome.
 */
novaStatus_t broadcast_shapes_from_tensors(const Tensor *const tensors[],
                                           size_t count, shape_t out_shape,
                                           size_t *out_ndims) {
  if (out_shape == nullptr || out_ndims == nullptr) {
    return (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Broadcasting needs a destination shape and its rank. Got "
                   "null.\n"};
  }
  if (count == 0) {
    memset(out_shape, 0, sizeof(shape_t));
    *out_ndims = 0;
    return broadcast_ok();
  }
  if (tensors == nullptr) {
    return (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Broadcasting needs a live tensor group. Got null.\n"};
  }

  for (size_t i = 0; i < count; ++i) {
    if (i > 0) {
      if (tensors[i]->dtype != tensors[0]->dtype) {
        return broadcast_error(
            novaInvalidDtype,
            "A broadcast group must share one dtype.\nTensor %zu carries a "
            "different dtype than tensor 0.\n",
            i);
      }
      if (tensors[i]->device != tensors[0]->device) {
        return broadcast_error(
            novaInvalidDevice,
            "A broadcast group must share one device.\nTensor %zu lives on a "
            "different device than tensor 0.\n",
            i);
      }
    }
  }

  // A packed dtype stores several logical values in one storage unit
  // along the last dim, so the innermost logical dim is never of size
  // one and cannot be stretched:  the last storage dim of every
  // non-scalar member must agree.
  const size_t packing = dtype_packing_factor(tensors[0]->dtype);
  if (packing > 1) {
    size_t last = 0;
    bool has_last = false;
    for (size_t i = 0; i < count; ++i) {
      if (is_scalar(tensors[i])) {
        // A scalar contributes no dimension.
        continue;
      }
      const size_t dim = tensors[i]->shape[tensors[i]->ndims - 1];
      if (!has_last) {
        last = dim;
        has_last = true;
      } else if (dim != last) {
        return broadcast_error(
            novaInvalidShape,
            "A packed dtype cannot stretch the last dim.\nSeveral logical "
            "values share one storage unit (packing factor %zu).\nTensor %zu "
            "pairs storage size %zu with size %zu.\n",
            packing, i, last, dim);
      }
    }
  }

  BroadcastAccum acc = {};
  size_t conflict = 0;
  size_t conflict_element = 0;
  if (!broadcast_accumulate_tensors(&acc, tensors, count, &conflict,
                                    &conflict_element)) {
    const Tensor *failing = tensors[conflict_element];
    const size_t right = failing->shape[failing->ndims - conflict - 1];
    return broadcast_incompatible(
        novaInvalidShape, "tensor", conflict, acc.sizes[conflict],
        acc.sources[conflict] - 1, right, conflict_element);
  }

  broadcast_shape_of(&acc, out_shape, out_ndims);
  return broadcast_ok();
}

/**
 * @brief Snapshot the source layout of every tensor of a group.
 *
 * @param[in]  tensors Group of tensors.
 * @param[out] outputs Receives one layout per tensor.
 * @param[in]  count   Number of tensors in the group.
 *
 * @return @ref novaStatus_t describing the outcome.
 */
novaStatus_t create_layouts_from_tensors(const Tensor *const tensors[],
                                         BroadcastLayout *const outputs[],
                                         size_t count) {
  if (count == 0) {
    return broadcast_ok();
  }
  if (tensors == nullptr || outputs == nullptr) {
    return (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Snapshotting the group needs a live tensor group and a "
                   "destination layout. Got null.\n"};
  }

  shape_t broadcast_shape = {};
  size_t broadcast_ndims = 0;
  const novaStatus_t st = broadcast_shapes_from_tensors(
      tensors, count, broadcast_shape, &broadcast_ndims);
  if (st.err != novaSuccess) {
    return st;
  }

  for (size_t i = 0; i < count; ++i) {
    const size_t ndims = tensors[i]->ndims;
    memcpy(outputs[i]->shape, tensors[i]->shape, ndims * sizeof(size_t));
    memcpy(outputs[i]->strides, tensors[i]->strides, ndims * sizeof(size_t));
    outputs[i]->ndims = ndims;
    outputs[i]->offset = tensors[i]->offset;
  }

  return broadcast_ok();
}

/**
 * @brief Apply the broadcasting rules to a group of layouts.
 *
 * @details
 * The destination may be the same array as the source, because
 * @ref expand_layout_ copies what it reads before it writes.
 *
 * @param[in]  inputs  Source layouts of the group.
 * @param[out] outputs Receives one view per layout.
 * @param[in]  counts  Number of layouts in the group.
 *
 * @return @ref novaStatus_t describing the outcome.
 */
novaStatus_t broadcast_layouts(const BroadcastLayout *const inputs[],
                               BroadcastView *const outputs[], size_t counts) {
  if (counts == 0) {
    return broadcast_ok();
  }
  if (inputs == nullptr || outputs == nullptr) {
    return (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Expanding the group needs a live layout group and a "
                   "destination view. Got null.\n"};
  }

  BroadcastAccum acc = {};
  size_t conflict = 0;
  size_t conflict_element = 0;
  if (!broadcast_accumulate_layouts(&acc, inputs, counts, &conflict,
                                    &conflict_element)) {
    const BroadcastLayout *failing = inputs[conflict_element];
    const size_t right = failing->shape[failing->ndims - conflict - 1];
    return broadcast_incompatible(
        novaInvalidShape, "layout", conflict, acc.sizes[conflict],
        acc.sources[conflict] - 1, right, conflict_element);
  }

  shape_t broadcast_shape = {};
  size_t broadcast_ndims = 0;
  broadcast_shape_of(&acc, broadcast_shape, &broadcast_ndims);

  for (size_t i = 0; i < counts; ++i) {
    expand_layout_(inputs[i]->shape, inputs[i]->strides, inputs[i]->ndims,
                   broadcast_shape, broadcast_ndims, outputs[i]->shape,
                   outputs[i]->strides);
    outputs[i]->ndims = broadcast_ndims;
    outputs[i]->offset = inputs[i]->offset;
  }

  return broadcast_ok();
}

/**
 * @brief Apply broadcast views to a group of tensors.
 *
 * @details
 * The tensors keep their storage, their data pointer and their offset:
 * only the shape, the strides and the rank change.  The size is
 * recomputed from the new shape, so it counts the elements reachable
 * through the strides and not the distinct storage units.
 *
 * The gradient and the backward node are dropped:  they describe the
 * shape the tensor had before, which no longer matches.
 *
 * @param[in,out] tensors Tensors to reshape.
 * @param[in]     views   Views to apply.
 * @param[in]     count   Number of tensors in the group.
 *
 * @return @ref novaStatus_t describing the outcome.
 */
novaStatus_t broadcast_tensors_from_layouts(Tensor *const tensors[],
                                            const BroadcastView *const views[],
                                            size_t count) {
  if (count == 0) {
    return broadcast_ok();
  }
  if (tensors == nullptr || views == nullptr) {
    return (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Applying the views needs a live tensor group and a live "
                   "view group. Got null.\n"};
  }

  for (size_t i = 0; i < count; ++i) {
    const size_t ndims = views[i]->ndims;
    memcpy(tensors[i]->shape, views[i]->shape, ndims * sizeof(size_t));
    memcpy(tensors[i]->strides, views[i]->strides, ndims * sizeof(size_t));
    tensors[i]->ndims = ndims;
    compute_tensor_size_(tensors[i], tensors[i]->shape);

    /**
    * @brief Autograd configuration (empty yet)
    * ...
    */
  }

  return broadcast_ok();
}
