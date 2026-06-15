/**
 * @file repr_context.c
 * @brief Implementation of the derived display context builder.
 *
 * @details
 * This module implements @ref build_repr_context(), which performs a one-time
 * analysis of a tensor and its formatting options to produce a persistent
 * @ref ReprContext. This context drives all subsequent layout and formatting
 * operations, ensuring consistent alignment and numeric representation.
 *
 * The builder performs a data scan to auto-detect optimal formatting
 * (scientific vs. decimal) and to compute the maximum string width for
 * column alignment in multi-dimensional output.
 *
 * ## Architecture
 * The context building process consists of two primary passes:
 * 1. **Classification**: Maps the tensor's metadata (dtype, device, ndims)
 *    to categorical flags in the context.
 * 2. **Analysis Pass**: Samples the tensor's data (sampling up to 1000
 *    elements from the start and end) to:
 *    - Apply the PyTorch heuristic for scientific notation.
 *    - Calculate the maximum formatted width of elements, accounting for
 *      strided (view) layouts.
 *
 * @see repr_context.h Structure definition.
 * @see element_fmt.h Element-wise formatting dispatch.
 */

#include <math.h>
#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/storage.h>
#include <ncore/headeronly/tensor_utils.h>
#include <ncore/repr/repr_context.h>
#include <ncore/tensor.h>
#include <stdio.h>
#include <string.h>

#include "repr/formatters/element_fmt.h"

/**
 * @brief Internal helper to check floating-point data types.
 *
 * @param[in] d DType to check.
 * @return true if d is Float32, Float64, Float16, or BFloat16.
 */
static inline bool is_float_dtype(DType_ d) {
  return d == Float32 || d == Float64 || d == Float16 || d == BFloat16;
}

/**
 * @brief Internal helper to check integer data types.
 *
 * @param[in] d DType to check.
 * @return true if d is any non-quantized signed or unsigned integer type.
 */
static inline bool is_integer_dtype(DType_ d) {
  return d == Signed8 || d == UnSigned8 || d == Signed32 || d == UnSigned32 ||
         d == Signed64 || d == UnSigned64;
}

/**
 * @brief Internal helper to check quantized data types.
 *
 * @param[in] d DType to check.
 * @return true if d is QSigned8 or QUnSigned8.
 */
static inline bool is_quantized_dtype(DType_ d) {
  return d == QSigned8 || d == QUnSigned8;
}

/**
 * @brief Extract a numeric value as double for range analysis.
 *
 * @details
 * This is used exclusively during the scientific notation auto-detection
 * phase to calculate the absolute range of the tensor's elements.
 *
 * @param[in] ten Pointer to the tensor.
 * @param[in] idx Linear index of the element.
 *
 * @return The element value converted to double.
 */
static inline double get_float_value(const Tensor *ten, size_t idx) {
  const size_t elem_off = (ten->offset / ten->item_size) + idx;
  switch (ten->dtype) {
  case Float32:
    return (double)ten->data.f32[elem_off];
  case Float64:
    return ten->data.f64[elem_off];
  case Float16:
    return (double)ten->data.half[elem_off];
  case BFloat16:
    return (double)ten->data.bf16[elem_off];
  default:
    return 0.0;
  }
}

/**
 * @brief Build a ReprContext from a tensor and options.
 *
 * @details
 * Performs metadata classification and a strided-aware data scan to
 * derive column widths and numeric notation settings. The scan is
 * performance-gated, sampling a maximum of 2000 elements even for
 * very large tensors.
 *
 * @param[in] ten Pointer to the tensor to analyze. Must not be NULL.
 * @param[in] opts Pointer to the formatting options. May be NULL for defaults.
 *
 * @return A fully populated ReprContext.
 */
ReprContext build_repr_context(const Tensor *ten, const ReprOptions *opts) {
  ReprContext ctx = {0};
  ctx.tensor = ten;
  ctx.options = (opts != NULL) ? *opts : repr_default_options();
  ctx.is_float = is_float_dtype(ten->dtype);
  ctx.is_integer = is_integer_dtype(ten->dtype);
  ctx.is_quantized = is_quantized_dtype(ten->dtype);
  ctx.is_bool = (opts != NULL) ? opts->is_bool : false;
  ctx.is_scalar = is_scalar(ten);
  ctx.is_meta = (bool)(ten->device == DEVICE_META);
  ctx.is_gpu = (bool)(ten->device == DEVICE_GPU);
  ctx.effective_precision = opts ? opts->precision : 4;
  ctx.is_summarized = (ten->size > ctx.options.threshold);
  ctx.use_sci = (opts != NULL) ? opts->sci_mode : false;

  size_t n = ten->size > 1000 ? 1000 : ten->size;

  if (ctx.is_meta) {
    ctx.element_width = 3;
    ctx.use_sci = false;
    return ctx;
  }

  /* --- Scientific-notation auto-detection --- */
  const bool sci_mode_auto = (opts != NULL) ? opts->sci_mode_auto : true;
  const bool sci_mode = (opts != NULL) ? opts->sci_mode : false;

  if (ctx.is_float && sci_mode_auto && !sci_mode && n > 0) {
    double max_abs = 0.0;
    double min_nonzero_abs = 1e100;
    bool found = false;

    /* Scan start and end elements for the PyTorch heuristic */
    size_t scan_count = (ten->size > 2000) ? 1000 : ten->size;

    for (size_t i = 0; i < scan_count; i++) {
      size_t idx = (i < 1000) ? i : (ten->size - (scan_count - i));
      double v = get_float_value(ten, idx);
      if (isinf(v) || isnan(v)) {
        continue;
      }
      double av = fabs(v);
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

  if (ten->size == 0) {
    ctx.element_width = 1;
    return ctx;
  }

  /* --- Compute maximum formatted element width --- */
  size_t max_w = 0;
  size_t scan_limit = (ten->size > 2000) ? 1000 : ten->size;

  for (size_t i = 0; i < scan_limit; i++) {
    size_t idx;
    if (ten->size > 2000 && i >= 500) {
      idx = ten->size - 1000 + i;
    } else {
      idx = i;
    }

    char fmt_buf[128];
    coords_t coords;
    compute_coords_given_linear_byte_offset_(idx * ten->item_size, ten->ndims,
                                             coords, ten->strides);
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
