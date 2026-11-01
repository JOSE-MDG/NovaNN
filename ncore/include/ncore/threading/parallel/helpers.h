/**
 * @file helpers.h
 * @brief Call-site builders from @ref Tensor metadata and raw demand.
 *
 * @details
 * Translates tensor fields into the plain facts the decision
 * functions read, folding the structural checks (allocated on
 * CPU with a valid dtype, non-scalar, non-empty) into the
 * @c valid flag. Raw-buffer stages with no Tensor in hand
 * (pinned staging, scratch draws) build their facts through
 * @ref make_rng_work(). Builders are header-inline so the
 * translation stays at the call site with no extra call overhead.
 *
 * @see pattern.h  Work structs produced here.
 * @see decide.h   Decisions consuming the produced facts.
 */

#pragma once

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/headeronly/macros.h>
#include <ncore/tensor.h>
#include <ncore/threading/parallel/pattern.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Build streaming-map facts from a tensor.
 *
 * @details
 * Counts unpacked elements (matters for packed pairs where
 * storage units hold several values) and reads backing bytes
 * for the one-byte-element rule.
 *
 * @param[in] ten  Source tensor. May be @c nullptr.
 *
 * @return Facts with @c valid false on any structural failure.
 */
static inline ElementwiseWork make_elementwise_work(const Tensor *ten) {
  ElementwiseWork work = {
      .logical_size = 0,
      .total_bytes = 0,
      .item_size = 0,
      .packing = 1,
      .valid = false,
  };
  if (ten == nullptr) {
    return work;
  }
  if (!is_allocated(ten)) {
    return work;
  }
  if (ten->device != DEVICE_CPU) {
    return work;
  }
  if (ten->dtype >= NUM_DTYPES) {
    return work;
  }
  if (ten->ndims == 0 || ten->size == 0 || ten->logical_size == 0) {
    return work;
  }
  work.logical_size = ten->logical_size;
  work.total_bytes = ten->storage->handle.requested_bytes;
  work.item_size = ten->item_size;
  work.packing = dtype_packing_factor(ten->dtype);
  work.heavy = ((is_quantizable_dtype(ten->dtype) || work.packing > 1) != 0);
  work.valid = true;
  return work;
}

/**
 * @brief Build streaming-map facts for a raw random-value buffer.
 *
 * @details
 * Random stages run over caller-owned storage with no Tensor in
 * hand (pinned staging, scratch draws), so the tensor builders
 * cannot describe them: their structural checks demand a live CPU
 * tensor and would refuse the call site. Facts adopt the
 * streaming-map shape, one value per element, and class the cost
 * heavy because block draws or transcendental transforms inflate
 * the per-value work beyond a plain copy. The caller owns the
 * demand being a real buffer length.
 *
 * @param[in] n          Values to process.
 * @param[in] item_size  Bytes per value.
 *
 * @return Facts with @c valid true. An empty demand stays valid;
 *         the rule reads it as a serial stage through the grain
 *         quotient.
 */
static inline ElementwiseWork make_rng_work(size_t n, size_t item_size) {
  ElementwiseWork work = {
      .logical_size = n,
      .total_bytes = n * item_size,
      .item_size = item_size,
      .packing = 1,
      .heavy = true,
      .valid = true,
  };
  return work;
}

/**
 * @brief Build layout facts from a tensor.
 *
 * @details
 * Same structural checks as the streaming builder plus the
 * dense flag supplied by the caller (plain copy eligibility is
 * decided by the caller branches, not by the grain rule).
 *
 * @param[in] ten      Source tensor. May be @c nullptr.
 * @param[in] is_dense True when the move is a plain copy.
 *
 * @return Facts with @c valid false on any structural failure.
 */
static inline LayoutWork make_layout_work(const Tensor *ten, bool is_dense) {
  LayoutWork work = {
      .num_elements = 0,
      .total_bytes = 0,
      .item_size = 0,
      .packing = 1,
      .is_dense = is_dense,
      .valid = false,
  };
  if (ten == nullptr) {
    return work;
  }
  if (!is_allocated(ten)) {
    return work;
  }
  if (ten->device != DEVICE_CPU) {
    return work;
  }
  if (ten->dtype >= NUM_DTYPES) {
    return work;
  }
  if (ten->ndims == 0 || ten->size == 0 || ten->logical_size == 0) {
    return work;
  }
  work.num_elements = ten->size;
  work.total_bytes = ten->storage->handle.requested_bytes;
  work.item_size = ten->item_size;
  work.packing = dtype_packing_factor(ten->dtype);
  work.heavy = ((is_quantizable_dtype(ten->dtype) || work.packing > 1) != 0);
  work.valid = true;
  return work;
}

#ifdef __cplusplus
}
#endif
