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
 * By caching classification flags (@c is_float, @c is_integer) and
 * formatting decisions (@c use_sci, @c element_width), the context
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
 * @section field-categories Field Categories
 *
 * @li Derived State: Results of the data scan (@c element_width,
 *   @c use_sci).
 * @li Classification Shortcuts: Precomputed boolean flags for
 *   dtype category dispatch.
 * @li Placement Info: Device and allocation status for routing.
 */
typedef struct {
  const Tensor *tensor; ///< Pointer to the tensor being rendered.
  ReprOptions options;  ///< Snapshot of the original formatting options.

  /* ---- Derived from data scan ---- */
  bool use_sci;            ///< Final decision on scientific notation (@c %e).
  int effective_precision; ///< Target number of decimal places for floats.
  size_t element_width;    ///< Maximum formatted width for column alignment.
  bool is_summarized;      ///< @c true if tensor size exceeds threshold.
  size_t sub_element_index; ///< Sub-element index within a packed storage unit.

  /* ---- DType Classification Shortcuts ---- */
  bool is_float;     ///< @c true for floating-point types.
  bool is_integer;   ///< @c true for signed/unsigned integer types.
  bool is_quantized; ///< @c true for quantized integer types.
  bool is_bool;      ///< @c true to treat UnSigned8 as boolean True/False.
  bool is_scalar;    ///< @c true if the tensor rank (@c ndims) is zero.

  /* ---- Device & Allocation State ---- */
  bool is_meta; ///< @c true if the tensor resides on @c DEVICE_META.
  bool is_gpu;  ///< @c true if the tensor resides on @c DEVICE_GPU.
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
 *                  @c nullptr.
 * @param[in]  opts Pointer to formatting options. If @c nullptr,
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
