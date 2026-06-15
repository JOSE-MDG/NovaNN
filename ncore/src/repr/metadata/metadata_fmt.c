/**
 * @file metadata_fmt.c
 * @brief Implementation of the tensor metadata suffix formatter.
 *
 * @details
 * This module handles the generation of the metadata footer that follows
 * the tensor's data block. It provides contextual information such as
 * data type, shape, device placement, and autograd status, depending
 * on the display mode (NORMAL vs DEBUG).
 *
 * ## Architecture
 * - **Context Sensitivity**: Normal mode suppresses information that is
 *   considered "default" (e.g., Float32, CPU) to reduce visual noise.
 * - **Debug Path**: Forces emission of all metadata fields on a new line
 *   to facilitate diagnostics.
 * - **String Mapping**: Uses precomputed string tables for @ref DType_
 *   and @ref Device enumeration values.
 *
 * @see metadata_fmt.h Footer interface.
 * @see repr_options.h Mode definitions.
 */

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <stdio.h>

#include "metadata_fmt.h"

/**
 * @var static const char *g_dtype_string
 * @brief String table for mapping DType_ values to human-readable labels.
 */
static const char *g_dtype_string[NUM_DTYPES] = {
    [Float32] = "float32",   [Float64] = "float64",   [Float16] = "float16",
    [BFloat16] = "bfloat16", [Signed8] = "int8",      [UnSigned8] = "uint8",
    [QSigned8] = "qint8",    [QUnSigned8] = "quint8", [Signed32] = "int32",
    [UnSigned32] = "uint32", [Signed64] = "int64",    [UnSigned64] = "uint64",
};

/**
 * @var static const char *g_device_string
 * @brief String table for mapping Device values to human-readable labels.
 */
static const char *g_device_string[3] = {
    [DEVICE_CPU] = "cpu", [DEVICE_GPU] = "cuda", [DEVICE_META] = "meta"};

/**
 * @brief Map a DType_ value to its human-readable string representation.
 *
 * @param[in] d The DType to look up.
 * @return A static string literal, or "unknown" if the type is invalid.
 */
static const char *dtype_string(DType_ d) {
  if (d >= NUM_DTYPES) {
    return "unknown";
  }
  return g_dtype_string[d];
}

/**
 * @brief Map a Device enum value to its human-readable string representation.
 *
 * @param[in] d The Device to look up.
 * @return A static string literal, or "unknown" if the device is invalid.
 */
static const char *device_string(Device d) {
  if ((int)d >= 3) {
    return "unknown";
  }
  return g_device_string[d];
}

/**
 * @brief Append the metadata suffix and close the outer tensor representation.
 *
 * @details
 * This function appends the final ")" and optionally a comma-separated
 * metadata block based on the current mode and tensor state.
 *
 * @param[in]     ctx Pointer to the representation context.
 * @param[in,out] sb  Pointer to the StringBuilder.
 */
void metadata_fmt_append(const ReprContext *ctx, StringBuilder *sb) {
  const Tensor *ten = ctx->tensor;
  ReprMode mode = ctx->options.mode;

  /* ---- Normal mode ---- */
  if (mode == ReprModeNormal) {
    /* Meta tensors always show dtype + device. */
    if (ctx->is_meta) {
      sb_append(sb, ", ");
      sb_append(sb, "dtype=");
      sb_append(sb, dtype_string(ten->dtype));
      sb_append(sb, ", device=meta");
      sb_append(sb, ")");
      return;
    }

    if (ctx->is_gpu) {
      sb_append(sb, ", ");
      sb_append(sb, "dtype=");
      sb_append(sb, dtype_string(ten->dtype));
      sb_append(sb, ", device=cuda");
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
    sb_append(sb, (int)ten->requires_grad_ ? "True" : "False");
  }

  sb_append(sb, ")");
}
