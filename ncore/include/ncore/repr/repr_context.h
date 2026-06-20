/**
 * @file repr_context.h
 * @brief Derived display context for tensor string representation.
 *
 * @details
 * This header defines the @ref ReprContext structure, which acts as the
 * central derived-state engine for the representation module. A context is
 * built once per representation call by @ref build_repr_context() and
 * carries all parameters needed for layout rendering and element formatting.
 *
 * By caching classification flags (is_float, is_integer) and formatting
 * decisions (use_sci, element_width), the context eliminates the need for
 * redundant metadata queries and data scans during the recursive rendering
 * process.
 *
 * @see build_repr_context()  Constructor for the context.
 * @see tensor_repr.h  Top-level public API.
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
 * @struct ReprContext
 * @brief Carries all derived state needed by layout and formatter layers.
 *
 * @details
 * The context is the "single source of truth" during the representation
 * pipeline. It is populated by scanning a sample of the tensor's data
 * (up to 1000 elements at each edge) to ensure consistent column alignment
 * and optimal numeric notation.
 *
 * ## Context Categories
 * - **Derived State**: results of the data scan (width, scientific notation).
 * - **Category Flags**: shortcuts for dtype classification to avoid switches.
 * - **Placement Info**: device and allocation status for dispatching.
 */
typedef struct {
  const Tensor *tensor; ///< Pointer to the tensor being rendered.
  ReprOptions options;  ///< Snapshot of the original formatting options.

  /* ---- Derived from data scan ---- */
  bool use_sci;            ///< Final decision on scientific notation (%e).
  int effective_precision; ///< Target number of decimal places for floats.
  size_t element_width;    ///< Maximum formatted width for column alignment.
  bool is_summarized;      ///< True if tensor size exceeds threshold.

  /* ---- DType Classification Shortcuts ---- */
  bool is_float;     ///< True for floating-point types (Float32/64/16/BF16).
  bool is_integer;   ///< True for signed/unsigned integer types.
  bool is_quantized; ///< True for QSigned8 or QUnSigned8 types.
  bool is_bool;      ///< Flag to treat UnSigned8 as boolean True/False.
  bool is_scalar;    ///< True if the tensor rank (ndims) is zero.

  /* ---- Device & Allocation State ---- */
  bool is_meta; ///< True if the tensor resides on @ref DEVICE_META.
  bool is_gpu;  ///< True if the tensor resides on @ref DEVICE_GPU.
} ReprContext;

/**
 * @brief Build a fully populated ReprContext from a tensor and options.
 *
 * @details
 * Analyzes the input tensor's metadata and performs a partial data scan
 * (sampling the first and last 1000 elements) to derive optimal display
 * parameters.
 *
 * For floating-point tensors, it applies the PyTorch heuristic to determine
 * if scientific notation should be enabled based on the range of absolute
 * values. For all types, it calculates the maximum string width of elements
 * to ensure that multi-dimensional output is correctly aligned in columns.
 *
 * @param[in]  ten  Pointer to the tensor to analyze. Must not be NULL.
 * @param[in]  opts Pointer to formatting options. If NULL, defaults are used.
 *
 * @return A fully initialised @ref ReprContext structure.
 *
 * @pre  The tensor's data must be host-accessible (shadowed to CPU if GPU).
 * @post The returned context can be passed safely to layout renderers.
 *
 * @see repr_options.h
 */
ReprContext build_repr_context(const Tensor *ten, const ReprOptions *opts);

#ifdef __cplusplus
}
#endif
