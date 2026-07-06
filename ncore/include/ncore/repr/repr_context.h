/**
 * @file repr_context.h
 * @brief Derived display context for tensor string representation.
 *
 * @details
 * Declares @ref ReprContext, the central derived-state structure for
 * the representation module. A context is built once per
 * representation call by @ref build_repr_context() and carries all
 * parameters needed for layout rendering and element formatting.
 *
 * By caching classification flags (`is_float`, `is_integer`) and
 * formatting decisions (`use_sci`, `element_width`), the context
 * eliminates redundant metadata queries and data scans during the
 * recursive rendering process.
 *
 * @see build_repr_context()  Constructor for the context.
 * @see tensor_repr.h         Top-level public API.
 * @see repr_options.h        User-facing formatting options.
 */

#pragma once

#include <stddef.h>

#include <ncore/repr/repr_options.h>
#include <ncore/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @struct ReprContext
 * @brief Carries all derived state needed by layout and formatter layers.
 *
 * @details
 * The context is the "single source of truth" during the
 * representation pipeline. It is populated by scanning a sample of
 * the tensor's data (up to 1000 elements at each edge) to ensure
 * consistent column alignment and optimal numeric notation.
 *
 * ## Field Categories
 *
 * - **Derived State**: Results of the data scan (`element_width`,
 *   `use_sci`).
 * - **Classification Shortcuts**: Precomputed boolean flags for
 *   dtype category dispatch.
 * - **Placement Info**: Device and allocation status for routing.
 */
typedef struct {
  const Tensor *tensor; ///< Pointer to the tensor being rendered.
  ReprOptions options;  ///< Snapshot of the original formatting options.

  /* ---- Derived from data scan ---- */
  bool use_sci;            ///< Final decision on scientific notation (`%e`).
  int effective_precision; ///< Target number of decimal places for floats.
  size_t element_width;    ///< Maximum formatted width for column alignment.
  bool is_summarized;      ///< `true` if tensor size exceeds threshold.
  size_t sub_element_index; ///< Sub-element index within a packed storage unit.

  /* ---- DType Classification Shortcuts ---- */
  bool is_float;     ///< `true` for floating-point types.
  bool is_integer;   ///< `true` for signed/unsigned integer types.
  bool is_quantized; ///< `true` for quantized integer types.
  bool is_bool;      ///< `true` to treat UnSigned8 as boolean True/False.
  bool is_scalar;    ///< `true` if the tensor rank (`ndims`) is zero.

  /* ---- Device & Allocation State ---- */
  bool is_meta; ///< `true` if the tensor resides on `DEVICE_META`.
  bool is_gpu;  ///< `true` if the tensor resides on `DEVICE_GPU`.
} ReprContext;

/**
 * @brief Build a fully populated ReprContext from a tensor and options.
 *
 * @details
 * Analyzes the input tensor's metadata and performs a partial data
 * scan (sampling the first and last 1000 elements) to derive optimal
 * display parameters.
 *
 * For floating-point tensors, it applies an auto-detection heuristic
 * to determine if scientific notation should be enabled based on the
 * range of absolute values. For all types, it calculates the maximum
 * string width of elements to ensure multi-dimensional output is
 * correctly column-aligned.
 *
 * @param[in]  ten  Pointer to the tensor to analyze. Must not be
 *                  `nullptr`.
 * @param[in]  opts Pointer to formatting options. If `nullptr`,
 *                  defaults are used.
 *
 * @return A fully initialised @ref ReprContext structure.
 *
 * @pre  The tensor's data must be host-accessible (shadowed to CPU
 *       if GPU).
 * @post The returned context can be passed safely to layout
 *       renderers.
 *
 * @see repr_options.h       Formatting options.
 * @see repr_default_options()  Returns default options.
 */
ReprContext build_repr_context(const Tensor *ten, const ReprOptions *opts);

#ifdef __cplusplus
}
#endif
