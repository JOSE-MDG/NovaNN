/**
 * @file metadata_fmt.c
 * @brief Suffix formatter for tensor metadata.
 *
 * @details
 * Appends the closing suffix after the data-block's closing parenthesis.
 *
 * Normal mode:
 *   - Suppressed entirely when dtype == Float32 and no grad info.
 *   - Meta tensors always show dtype + device.
 *   - grad_fn takes priority over requires_grad.
 *
 * Debug mode:
 *   Always appends:
 *     dtype=..., shape=(...), device=..., requires_grad=...
 *   on a continuation line indented to align with the data block.
 */

#include "metadata_fmt.h"
#include <ncore/device.h>
#include <ncore/dtype.h>
#include <stdio.h>

static const char *g_dtype_string[NUM_DTYPES] = {
    [Float32] = "float32",   [Float64] = "float64",   [Float16] = "float16",
    [BFloat16] = "bfloat16", [Signed8] = "int8",      [UnSigned8] = "uint8",
    [QSigned8] = "qint8",    [QUnSigned8] = "quint8", [Signed32] = "int32",
    [UnSigned32] = "uint32", [Signed64] = "int64",    [UnSigned64] = "uint64",
};

static const char *g_device_string[3] = {
    [DEVICE_CPU] = "cpu", [DEVICE_GPU] = "cuda", [DEVICE_META] = "meta"};

static const char *dtype_string(DType_ d) {
  if (d >= NUM_DTYPES) return "unknown";
  return g_dtype_string[d];
}

static const char *device_string(Device d) {
  if ((int)d >= 3) return "unknown";
  return g_device_string[d];
}

void metadata_fmt_append(const ReprContext *ctx, StringBuilder *sb) {
  const Tensor *ten = ctx->tensor;
  ReprMode mode = ctx->options.mode;

  /* ---- Normal mode ---- */
  if (mode == REPR_MODE_NORMAL) {
    /* Meta tensors always show dtype + device. */
    if (ctx->is_meta) {
      sb_append(sb, ", ");
      sb_append(sb, "dtype=");
      sb_append(sb, dtype_string(ten->dtype));
      sb_append(sb, ", device=meta");
      sb_append(sb, ")");
      return;
    }

    bool show_dtype = (ten->dtype != Float32);
    bool show_grad = (ten->requires_grad_ && ten->grad_fn_ == NULL) != 0;
    bool show_grad_fn = ten->grad_fn_ != NULL;

    if (!show_dtype && !show_grad && !show_grad_fn) {
      sb_append(sb, ")");
      return;
    }

    sb_append(sb, ", ");
    sb_append(sb, "dtype=");
    sb_append(sb, dtype_string(ten->dtype));

    if (show_grad_fn) {
      sb_append(sb, ", ");
      // TODO: pass grad_fn member as string of <BackwardNode> (via C++)
      sb_append(sb, "grad_fn=<BackwardNode>");
    } else if (show_grad) {
      sb_append(sb, ", ");
      sb_append(sb, "requires_grad=True");
    }

    sb_append(sb, ")");
    return;
  }

  /* ---- Debug mode ---- */
  sb_append(sb, ",\n       ");
  sb_append(sb, "dtype=");
  sb_append(sb, dtype_string(ten->dtype));

  sb_append(sb, ", shape=(");
  if (ten->ndims == 0) {
    sb_append(sb, ")");
  } else {
    for (size_t dim = 0; dim < ten->ndims; dim++) {
      if (dim > 0) {
        sb_append(sb, ", ");
      }
      char buf[32];
      snprintf(buf, sizeof(buf), "%zu", ten->shape[dim]);
      sb_append(sb, buf);
    }
    sb_append(sb, ")");
  }

  sb_append(sb, ", device=");
  sb_append(sb, device_string(ten->device));

  if (ten->grad_fn_ != NULL) {
    // TODO: pass grad_fn member as string of <BackwardNode> (via C++)
    sb_append(sb, ", grad_fn=<BackwardNode>");
  } else {
    sb_append(sb, ", requires_grad=");
    sb_append(sb, ten->requires_grad_ ? "True" : "False");
  }

  sb_append(sb, ")");
}
