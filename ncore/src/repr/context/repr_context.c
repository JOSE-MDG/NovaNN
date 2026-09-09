/**
 * @file repr_context.c
 * @brief Derived display context builder implementation.
 *
 * @details
 * Implements @ref build_repr_context(), which performs a one-time
 * analysis of a tensor and its formatting options to produce a
 * persistent @ref ReprContext. This context drives all subsequent
 * layout and formatting operations, ensuring consistent alignment
 * and uniform numeric representation.
 *
 * @section context-building-process Context Building Process
 *
 * The builder performs three passes:
 *
 * @li 1. Classification: Maps the tensor's metadata (dtype, device,
 *    ndims) to categorical flags in the context.
 * @li 2. Sampling: Walks head and tail logical elements through
 *    shape-derived coordinates (correct for strided views), bounded
 *    by the options, recording the maximum sampled magnitude.
 * @li 3. Analysis: Applies the notation rule to the maximum and
 *    measures formatted widths over the same sample.
 *
 * The notation rule keys on the maximum alone because maxima
 * concentrate across samples while minima do not: identical data
 * keeps identical notation run to run.
 *
 * @see repr_context.h  Structure definition.
 * @see element_fmt.h   Element-wise formatting dispatch.
 * @see repr_options.h  User-facing formatting options.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/fp_utils.h>
#include <ncore/core/storage.h>
#include <ncore/headeronly/tensor_utils.h>
#include <ncore/repr/repr_context.h>
#include <ncore/tensor.h>

#include "repr/formatters/element_fmt.h"

#ifdef _GNUC_CLANG_
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
#endif

/**
 * @brief Absolute value of one element as double, from pointer and lane.
 *
 * @details
 * Used exclusively during the notation-detection phase. The pointer
 * already accounts for strides and packing; this helper only
 * converts the lane to @c double.
 */
static inline double lane_abs_double(const Tensor *ten, const void *ptr,
                                     size_t sub) {
  double v = 0.0;
  switch (ten->dtype) {
  case Float32:
    v = (double)*(const float *)ptr;
    break;
  case Float64:
    v = *(const double *)ptr;
    break;
  case Float16:
#ifdef _GNUC_CLANG_
    v = (double)*(const float16 *)ptr;
#else
    v = (double)fp16_to_float(*(const float16 *)ptr);
#endif
    break;
  case BFloat16:
#ifdef _GNUC_CLANG_
    v = (double)*(const bfloat16 *)ptr;
#else
    v = (double)fp16_to_float(*(const bfloat16 *)ptr);
#endif
    break;
  case Float8E4M3fn:
    v = (double)fp8e4m3fn_to_float(*(const float8_e4m3fn *)ptr);
    break;
  case Float8E5M2:
    v = (double)fp8e5m2_to_float(*(const float8_e5m2 *)ptr);
    break;
  case Float4E2M1fn: {
    float lo;
    float hi;
    fp4e2m1x2_to_floats(*(const float4_e2m1fn_x2 *)ptr, &lo, &hi);
    v = (double)(sub == 0 ? lo : hi);
    break;
  }
  default:
    break;
  }
  return fabs(v);
}

/**
 * @brief Build a ReprContext from a tensor and options.
 *
 * @param[in]  ten  Pointer to the tensor to analyze. Must not be
 *                  @c nullptr.
 * @param[in]  opts Pointer to formatting options. If @c nullptr,
 *                  defaults are used.
 *
 * @return A fully initialised @ref ReprContext structure.
 */
ReprContext build_repr_context(const Tensor *ten, const ReprOptions *opts) {
  ReprContext ctx = {};
  ctx.tensor = ten;
  ctx.options = (opts != nullptr) ? *opts : repr_default_options();
  ctx.is_float = is_floating(ten);
  ctx.is_integer = is_integer(ten);
  ctx.is_quantized = ((is_quantized_signed_integer(ten) ||
                       is_quantized_unsigned_integer(ten)) != 0);
  ctx.is_bool = (((opts != nullptr) ? (int)opts->is_bool : 0) != 0);
  ctx.is_scalar = is_scalar(ten);
  ctx.is_meta = (ten->device == DEVICE_META);
  ctx.is_gpu = (ten->device == DEVICE_GPU);
  ctx.effective_precision = opts ? opts->precision : 4;

  size_t edge = ctx.options.edge_items != 0 ? ctx.options.edge_items : 1U;
  if (ctx.options.threshold != 0 && edge > ctx.options.threshold) {
    /* No single dimension needs more edge items than the total
     * budget; this also keeps the tightening loop below short. */
    edge = ctx.options.threshold;
  }
  const size_t total = ten->logical_size;
  ctx.is_summarized = (total > ctx.options.threshold);
  for (size_t d = 0; d < ten->ndims && !ctx.is_summarized; ++d) {
    ctx.is_summarized = (ten->shape[d] > 2U * edge);
  }

  /* Tighten the edge count so high-rank output stays bounded: the
   * shown-element product must fit four thresholds worth. Never
   * loosened, never below one per edge. */
  if (ctx.is_summarized && ctx.options.threshold != 0) {
    const uint64_t cap = ctx.options.threshold > ~0ULL / 4U
                             ? ~0ULL
                             : (uint64_t)ctx.options.threshold * 4U;
    while (edge > 1U) {
      uint64_t shown = 1U;
      bool fits = true;
      for (size_t d = 0; d < ten->ndims; ++d) {
        const uint64_t m = ten->shape[d] < (2U * edge) + 1U
                               ? (uint64_t)ten->shape[d]
                               : (2U * edge) + 1U;
        if (m != 0 && shown > cap / m) {
          fits = false;
          break;
        }
        shown *= m;
      }
      if (fits) {
        break;
      }
      edge--;
    }
    ctx.options.edge_items = edge;
  }

  ctx.use_sci = (((opts != nullptr) ? (int)opts->sci_mode : 0) != 0);

  if (ctx.is_meta || total == 0) {
    ctx.element_width = ((int)ctx.is_meta ? 3U : 1U);
    if (ctx.is_meta) {
      ctx.use_sci = false;
    }
    return ctx;
  }

  /* Sample head and tail logical elements, bounded by the options. */
  size_t limit = ctx.options.threshold;
  if (limit < 2U * edge) {
    limit = 2U * edge;
  }
  if (limit > total) {
    limit = total;
  }
  const size_t head = (limit + 1U) / 2U;

  /* Notation detection over the sample maximum */
  const bool sci_mode_auto =
      ((opts != nullptr) ? (int)opts->sci_mode_auto : 1) != 0;
  const bool sci_mode = ((opts != nullptr) ? (int)opts->sci_mode : 0) != 0;

  double max_abs = 0.0;
  if (ctx.is_float && sci_mode_auto && !sci_mode) {
    for (size_t i = 0; i < limit; i++) {
      const size_t logical = i < head ? i : total - limit + i;
      coords_t coords = {};
      compute_coords_from_linear_index_(logical, ten->ndims, ten->shape,
                                        coords);
      const void *ptr = ten->data.data + compute_linear_byte_offset(
                                             coords, ten->ndims, ten->strides);
      const double av = lane_abs_double(ten, ptr, 0);
      if (!(av > max_abs)) {
        continue;
      }
      max_abs = av;
    }
    if (isinf(max_abs) || isnan(max_abs)) {
      max_abs = 0.0;
    }
    if (max_abs >= 1e4 || (max_abs > 0.0 && max_abs < 1e-4)) {
      ctx.use_sci = true;
    }
  }

  /* Maximum formatted element width over the same sample */
  size_t max_w = 0;
  const size_t packing = dtype_packing_factor(ten->dtype);
  for (size_t i = 0; i < limit; i++) {
    const size_t logical = i < head ? i : total - limit + i;
    coords_t coords = {};
    compute_coords_from_linear_index_(logical, ten->ndims, ten->shape, coords);
    const void *ptr = ten->data.data + compute_linear_byte_offset(
                                           coords, ten->ndims, ten->strides);

    char fmt_buf[128];
    ReprContext lane = ctx;
    lane.sub_element_index = (size_t)(logical % (packing != 0 ? packing : 1U));
    const int w = format_element(fmt_buf, sizeof(fmt_buf), ptr, ten, &lane);
    if (w > 0 && (size_t)w > max_w) {
      max_w = (size_t)w;
    }
  }

  ctx.element_width = max_w;
  return ctx;
}

#ifdef _GNUC_CLANG_
#pragma GCC diagnostic pop
#endif
