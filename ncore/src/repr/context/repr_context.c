/**
 * @file repr_context.c
 * @brief Derived display context builder implementation.
 *
 * @details
 * Implements @ref build_repr_context(), which performs a one-time
 * analysis of a tensor and its formatting options to produce a
 * persistent @ref ReprContext. This context drives all subsequent
 * layout and formatting operations, ensuring consistent alignment
 * and numeric representation.
 *
 * @section context-building-process Context Building Process
 *
 * The builder performs two primary passes:
 *
 * @li 1. Classification: Maps the tensor's metadata (dtype, device,
 *    ndims) to categorical flags in the context.
 * @li 2. Analysis Pass: Samples the tensor's data (up to 1000
 *    elements from each edge) to:
 *    @li Apply an auto-detection heuristic for scientific notation.
 *    @li Calculate the maximum formatted width of elements, accounting
 *      for strided (view) layouts.
 *
 * @see repr_context.h  Structure definition.
 * @see element_fmt.h   Element-wise formatting dispatch.
 * @see repr_options.h  User-facing formatting options.
 */

#include <math.h>
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

/**
 * @brief Extract a numeric value as double for range analysis.
 *
 * @details
 * Used exclusively during the scientific notation auto-detection
 * phase to calculate the absolute range of the tensor's elements.
 *
 * @param[in] ten Pointer to the tensor.
 * @param[in] idx Linear index of the element.
 *
 * @return The element value converted to @c double.
 */
static inline double get_double_value(const Tensor *ten, size_t idx) {
  if (ten->dtype == Float4E2M1fn) {
    size_t packing = dtype_packing_factor(ten->dtype);
    size_t byte_off = (ten->offset / ten->item_size) + (idx / packing);
    size_t sub = idx % packing;
    float lo;
    float hi;
    fp4e2m1x2_to_floats(ten->data.fp4e2m1fn_x2[byte_off], &lo, &hi);
    return (sub == 0) ? (double)lo : (double)hi;
  }
  const size_t elem_off = (ten->offset / ten->item_size) + idx;
  switch (ten->dtype) {
  case Float32:
    return (double)ten->data.f32[elem_off];
  case Float64:
    return ten->data.f64[elem_off];
  case Float16:
#ifdef _GNUC_CLANG_
    return (double)ten->data.half[elem_off];
#else
    return (double)fp16_to_float(ten->data.half[elem_off]);
#endif
  case BFloat16:
#ifdef _GNUC_CLANG_
    return (double)ten->data.bf16[elem_off];
#else
    return (double)fp16_to_float(ten->data.bf16[elem_off]);
#endif
  case Float8E4M3fn:
    return (double)fp8e4m3fn_to_float(ten->data.fp8e4m3fn[elem_off]);
  case Float8E5M2:
    return (double)fp8e5m2_to_float(ten->data.fp8e5m2[elem_off]);
  default:
    return 0.0;
  }
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
  ctx.is_summarized = (ten->logical_size > ctx.options.threshold);
  ctx.use_sci = (((opts != nullptr) ? (int)opts->sci_mode : 0) != 0);

  size_t n = ten->logical_size > 1000 ? 1000 : ten->logical_size;

  if (ctx.is_meta) {
    ctx.element_width = 3;
    ctx.use_sci = false;
    return ctx;
  }

  /* --- Scientific-notation auto-detection --- */
  const bool sci_mode_auto =
      ((opts != nullptr) ? (int)opts->sci_mode_auto : 1) != 0;
  const bool sci_mode = ((opts != nullptr) ? (int)opts->sci_mode : 0) != 0;

  if (ctx.is_float && sci_mode_auto && !sci_mode && n > 0) {
    double max_abs = 0.0;
    double min_nonzero_abs = 1e100;
    bool found = false;

    /* Scan start and end elements for the auto-detection heuristic */
    size_t scan_count = (ten->logical_size > 2000) ? 1000 : ten->logical_size;

    for (size_t i = 0; i < scan_count; i++) {
      size_t idx = (i < 1000) ? i : (ten->logical_size - (scan_count - i));
      auto v = get_double_value(ten, idx);
      if (isinf(v) || isnan(v)) {
        continue;
      }
      auto av = fabs(v);
      if (av <= 0.0) {
        continue;
      }
      found = true;
      if (av > max_abs) {
        max_abs = av;
      }
      if (av < min_nonzero_abs) {
        min_nonzero_abs = av;
      }
    }

    if (found) {
      if (min_nonzero_abs < 1e-4 || max_abs >= 1e4 ||
          (min_nonzero_abs > 0 && max_abs / min_nonzero_abs > 1e3)) {
        ctx.use_sci = true;
      }
    }
  }

  if (ten->logical_size == 0) {
    ctx.element_width = 1;
    return ctx;
  }

  /* --- Compute maximum formatted element width --- */
  size_t max_w = 0;
  size_t packing = dtype_packing_factor(ten->dtype);
  size_t scan_limit = (ten->logical_size > 2000) ? 1000 : ten->logical_size;

  for (size_t i = 0; i < scan_limit; i++) {
    size_t idx;
    if (ten->logical_size > 2000 && i >= 500) {
      idx = ten->logical_size - 1000 + i;
    } else {
      idx = i;
    }

    char fmt_buf[128];
    coords_t coords;
    size_t byte_idx = idx / packing;
    compute_coords_given_linear_byte_offset_(byte_idx * ten->item_size,
                                             ten->ndims, coords, ten->strides);
    const void *ptr =
        (const uint8 *)ten->data.u8 +
        compute_linear_byte_offset(coords, ten->ndims, ten->strides);

    int w = format_element(fmt_buf, sizeof(fmt_buf), ptr, ten, &ctx);
    if ((size_t)w > max_w) {
      max_w = (size_t)w;
    }
  }

  ctx.element_width = max_w;
  return ctx;
}
