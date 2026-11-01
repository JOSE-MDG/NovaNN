/**
 * @file repr_context.h
 * @brief Derived display context for tensor string representation.
 *
 * @details
 * A context is built once per representation call by
 * @ref build_repr_context() and caches the classification flags and
 * formatting decisions shared by the layout and element formatters.
 *
 * @see build_repr_context()
 * @see repr_options.h  User-facing formatting options.
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
 * Populated by scanning bounded head and tail samples of the tensor's
 * data so columns align and numeric notation stays uniform.
 */
typedef struct {
  const Tensor *tensor; ///< Pointer to the tensor being rendered.
  ReprOptions options;  ///< Snapshot of the original formatting options.

  /* Derived from data scan */
  bool use_sci;             ///< Final decision on scientific notation (@c %e).
  int effective_precision;  ///< Target number of decimal places for floats.
  size_t element_width;     ///< Maximum formatted width for column alignment.
  bool is_summarized;       ///< true when truncated rendering applies:
                            ///< size over threshold or a dimension over
                            ///< twice the edge count.
  size_t sub_element_index; ///< Sub-element index within a packed storage unit.

  /* DType classification shortcuts */
  bool is_float;     ///< true for floating-point types.
  bool is_integer;   ///< true for signed/unsigned integer types.
  bool is_quantized; ///< true for quantized integer types.
  bool is_bool;      ///< true to treat UnSigned8 as boolean True/False.
  bool is_scalar;    ///< true if the tensor rank (@c ndims) is zero.

  /* Device & allocation state */
  bool is_meta; ///< @c true if the tensor resides on @c DEVICE_META.
  bool is_gpu;  ///< @c true if the tensor resides on @c DEVICE_GPU.
} ReprContext;

/**
 * @brief Build a fully populated ReprContext from a tensor and options.
 *
 * @details
 * Performs a partial data scan (head and tail logical elements,
 * bounded by the options), then computes the maximum string width of
 * elements so multi-dimensional output is column-aligned.
 *
 * For floating-point tensors, scientific notation is auto-detected
 * from the maximum sampled magnitude: extremes in either direction
 * select scientific, ordinary ranges stay fixed. Minima and ratios
 * are deliberately ignored (sample noise flips them run to run, while
 * maxima concentrate).
 *
 * @param[in]  ten  Pointer to the tensor to analyze. Must not be
 *                  @c nullptr.
 * @param[in]  opts Pointer to formatting options. If @c nullptr,
 *                  defaults are used.
 *
 * @return A fully initialized @ref ReprContext structure.
 *
 * @pre The tensor's data must be host-accessible (shadowed to CPU
 *      if GPU).
 *
 * @see repr_options.h       Formatting options.
 * @see repr_default_options()  Returns default options.
 */
ReprContext build_repr_context(const Tensor *ten, const ReprOptions *opts);

#ifdef __cplusplus
}
#endif
