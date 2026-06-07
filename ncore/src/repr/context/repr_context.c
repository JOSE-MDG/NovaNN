/**
 * @file repr_context.c
 * @brief ReprContext builder — scans tensor data and derives display
 * parameters.
 *
 * @details
 * Implements build_repr_context() which analyses a tensor once and
 * produces a ReprContext that layout and formatter layers consume
 * without re-scanning.  The scan computes:
 *   - dtype category flags (float / integer / quantized / bool / scalar)
 *   - device / allocation state (meta / gpu)
 *   - scientific-notation auto-detection (PyTorch heuristic)
 *   - maximum formatted element width (for column alignment)
 *
 * The element-width scan uses the dispatch table from element_fmt.h
 * instead of a hand-rolled switch, keeping the formatting logic in
 * one place.
 */

#include <math.h>
#include <ncore/device.h>
#include <ncore/dtype.h>
#include <ncore/repr/repr_context.h>
#include <ncore/storage.h>
#include <ncore/tensor.h>
#include <stdio.h>
#include <string.h>

#include "repr/formatters/element_fmt.h"

/**
 * @brief Check if a DType is a floating-point type.
 *
 * @param[in] d DType value.
 * @return true if d is Float32, Float64, Float16, or BFloat16.
 */
static inline bool is_float_dtype(DType_ d) {
  return d == Float32 || d == Float64 || d == Float16 || d == BFloat16;
}

/**
 * @brief Check if a DType is an integer type.
 *
 * @param[in] d DType value.
 * @return true if d is any signed or unsigned integer type.
 */
static inline bool is_integer_dtype(DType_ d) {
  return d == Signed8 || d == UnSigned8 || d == Signed32 || d == UnSigned32 ||
         d == Signed64 || d == UnSigned64;
}

/**
 * @brief Check if a DType is a quantized type.
 *
 * @param[in] d DType value.
 * @return true if d is QSigned8 or QUnSigned8.
 */
static inline bool is_quantized_dtype(DType_ d) {
  return d == QSigned8 || d == QUnSigned8;
}

/**
 * @brief Extract a float value from tensor storage as a double.
 *
 * Used only during scientific-notation auto-detection.  Keeps a small
 * switch rather than adding a second dispatch path.
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
 * Scans up to 1000 elements to determine the maximum element width and,
 * for float types, whether scientific notation should be enabled via the
 * PyTorch heuristic.  Meta and GPU tensors skip the data scan.
 *
 * @param[in]  ten  Tensor to render (must not be NULL).
 * @param[in]  opts Options (NULL = use repr_default_options()).
 * @return Fully populated ReprContext.
 */
ReprContext build_repr_context(const Tensor *ten, const ReprOptions *opts) {
  ReprContext ctx = {0};
  ctx.tensor = ten;
  ctx.options = opts ? *opts : repr_default_options();
  ctx.is_float = is_float_dtype(ten->dtype);
  ctx.is_integer = is_integer_dtype(ten->dtype);
  ctx.is_quantized = is_quantized_dtype(ten->dtype);
  ctx.is_bool = opts ? opts->is_bool : false;
  ctx.is_scalar = is_scalar(ten);
  ctx.is_meta = (ten->device == DEVICE_META);
  ctx.is_gpu = (ten->device == DEVICE_GPU);
  ctx.effective_precision = opts ? opts->precision : 4;
  ctx.is_summarized = (ten->size > ctx.options.threshold);
  ctx.use_sci = opts ? opts->sci_mode : false;

  size_t n = ten->size > 1000 ? 1000 : ten->size;

  // TODO: If the Tensor is on device move it to the Host

  if (ctx.is_meta || ctx.is_gpu) {
    ctx.element_width = 3;
    ctx.use_sci = false;
    return ctx;
  }

  /* --- Scientific-notation auto-detection --- */
  const bool sci_mode_auto = opts ? opts->sci_mode_auto : true;
  const bool sci_mode = opts ? opts->sci_mode : false;
  if (ctx.is_float && sci_mode_auto && !sci_mode && n > 0) {
    double max_abs = 0.0;
    double min_nonzero_abs = 1e100;
    bool found = false;
    for (size_t i = 0; i < n; i++) {
      double v = get_float_value(ten, i);
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

  if (n == 0) {
    ctx.element_width = 1;
    return ctx;
  }

  /* --- Compute maximum formatted element width --- */
  size_t max_w = 0;
  for (size_t i = 0; i < n; i++) {
    char fmt_buf[128];
    const void *ptr =
        (const uint8 *)ten->data.u8 + (ten->offset + (i * ten->item_size));
    format_element(fmt_buf, sizeof(fmt_buf), ptr, ten, &ctx);
    size_t w = strlen(fmt_buf);
    if (w > max_w) {
      max_w = w;
    }
  }

  if (ten->size > n && n > 0) {
    for (size_t i = ten->size - n; i < ten->size; i++) {
      char fmt_buf[128];
      const void *ptr =
          (const uint8 *)ten->data.u8 + (ten->offset + (i * ten->item_size));
      format_element(fmt_buf, sizeof(fmt_buf), ptr, ten, &ctx);
      size_t w = strlen(fmt_buf);
      if (w > max_w) {
        max_w = w;
      }
    }
  }

  ctx.element_width = max_w;
  return ctx;
}
