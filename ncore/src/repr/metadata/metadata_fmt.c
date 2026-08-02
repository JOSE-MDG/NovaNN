/**
 * @file metadata_fmt.c
 * @brief Metadata suffix formatter implementation.
 *
 * @details
 * Generates the metadata footer that follows the tensor's data block.
 * Provides contextual information such as data type, shape, device
 * placement, and autograd status, depending on the display mode
 * (@c ReprModeNormal vs @c ReprModeDebug).
 *
 * @section mode-behavior Mode Behavior
 *
 * @li Normal mode: Suppresses "default" information (e.g.,
 *   @c Float32, @c CPU) to reduce visual noise. Meta and GPU tensors
 *   always show dtype and device.
 * @li Debug mode: Forces emission of all metadata fields on a new
 *   line for diagnostic clarity.
 *
 * @section string-mapping String Mapping
 *
 * Uses precomputed string tables (@c g_dtype_string,
 * @c g_device_string) for @ref DType_ and @ref Device enumeration
 * values.
 *
 * @see metadata_fmt.h    Footer interface.
 * @see repr_options.h    Mode definitions.
 * @see dtype.h           DType_ enumeration.
 * @see device.h          Device enumeration.
 */

#include <stdio.h>

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>

#include "metadata_fmt.h"

/**
 * @var g_dtype_string
 * @brief String table mapping @ref DType_ values to human-readable labels.
 *
 * @details
 * Indexed by @ref DType_ values (@c 0 .. @c NUM_DTYPES-1). Every
 * entry is a static string literal.
 *
 * @see dtype_string()
 */
static const char *g_dtype_string[NUM_DTYPES] = {
    [Float32] = "float32",
    [Float64] = "float64",
    [Float16] = "float16",
    [BFloat16] = "bfloat16",
    [Float8E4M3fn] = "float8_e4m3fn",
    [Float8E5M2] = "float8_e5m2",
    [Float4E2M1fn] = "float4_e2m1fn_x2",
    [Signed8] = "int8",
    [UnSigned8] = "uint8",
    [QSigned8] = "qint8",
    [QUnSigned8] = "quint8",
    [Signed16] = "int16",
    [UnSigned16] = "uint16",
    [QSigned16] = "qint16",
    [QUnSigned16] = "quint16",
    [Signed32] = "int32",
    [UnSigned32] = "uint32",
    [QSigned32] = "qint32",
    [QUnSigned32] = "quint32",
    [Signed64] = "int64",
    [UnSigned64] = "uint64",
};

/**
 * @var g_device_string
 * @brief String table mapping @ref Device values to human-readable labels.
 *
 * @see device_string()
 */
static const char *g_device_string[3] = {
    [DEVICE_CPU] = "cpu", [DEVICE_GPU] = "cuda", [DEVICE_META] = "meta"};

/**
 * @brief Map a @ref DType_ value to its human-readable string.
 *
 * @param[in] d The data type to look up.
 *
 * @return A static string literal, or @c "unknown" if the type is
 *         out of range.
 */
static const char *dtype_string(DType_ d) {
  if (d >= NUM_DTYPES) {
    return "unknown";
  }
  return g_dtype_string[d];
}

/**
 * @brief Map a @ref Device value to its human-readable string.
 *
 * @param[in] d The device to look up.
 *
 * @return A static string literal, or @c "unknown" if the device is
 *         out of range.
 */
static const char *device_string(Device d) {
  if ((int)d >= 3) {
    return "unknown";
  }
  return g_device_string[d];
}

/**
 * @brief Append the metadata suffix and close the outer tensor
 *        representation.
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
    bool show_grad = ten->requires_grad_;
    bool show_grad_fn = ten->grad_fn_ != nullptr;

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

  if (ten->grad_fn_ != nullptr) {
    // TODO: pass grad_fn member as string of <BackwardNode> (via C++)
    sb_append(sb, ", grad_fn=<BackwardNode>");
  } else {
    sb_append(sb, ", requires_grad=");
    sb_append(sb, (int)ten->requires_grad_ ? "True" : "False");
  }

  sb_append(sb, ")");
}
