/**
 * @file repr_context.h
 * @brief Derived display context for a single tensor repr call.
 *
 * @details
 * ReprContext is built once by build_repr_context() and carries every
 * piece of derived state that the layout and formatter layers need:
 * dtype category flags, summarisation decision, scientific-notation
 * flag, effective precision, maximum element width, and device /
 * allocation state.
 *
 * Layout and formatter functions receive a const pointer to this
 * struct and never re-scan the tensor or the options.
 *
 * @see build_repr_context()  Context constructor.
 * @see tensor_repr.h        Top-level API.
 */

#pragma once

#include <ncore/repr/repr_options.h>
#include <ncore/tensor.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Carries all derived state needed by layout and formatter layers.
 */
typedef struct {
  const Tensor *tensor; ///< The tensor being rendered.
  ReprOptions options;  ///< Original user options.

  /* ---- Derived from a data scan (floats only) ---- */
  bool use_sci;            ///< Final sci-notation decision.
  int effective_precision; ///< Precision after any auto-adjust.
  size_t element_width;    ///< Max formatted element width (padding).
  bool is_summarized;      ///< ten->size > options.threshold.

  /* ---- Dtype category shortcuts ---- */
  bool is_float;     ///< Float32/64/16/BFloat16. */
  bool is_integer;   ///< Any non-quantized integer type. */
  bool is_quantized; ///< QSigned8 or QUnSigned8. */
  bool is_bool;      ///< Copied from options.is_bool. */
  bool is_scalar;    ///< ten->ndims == 0. */

  /* ---- Device / allocation state ---- */
  bool is_meta; ///< ten->device == DEVICE_META. */
  bool is_gpu;  ///< ten->device == DEVICE_GPU. */
} ReprContext;

/**
 * @brief Build a ReprContext from a tensor and options.
 *
 * Scans up to 1000 elements to determine the maximum element width and,
 * for float types, whether scientific notation should be enabled via the
 * PyTorch heuristic.  Meta and GPU tensors skip the data scan.
 *
 * @param ten  Tensor to render (must not be NULL).
 * @param opts Options (NULL = use repr_default_options()).
 * @return Fully populated ReprContext.
 */
ReprContext build_repr_context(const Tensor *ten, const ReprOptions *opts);

#ifdef __cplusplus
}
#endif
