/**
 * @file tensor_repr.h
 * @brief Public API for tensor string representation in NovaNN.
 *
 * @details
 * Renders tensors to human-readable strings in normal or debug mode.
 * Every function returns @ref novaStatus_t: failures carry a reason
 * instead of an ambiguous null. Functions producing a string take an
 * output slot; on error the slot holds @c nullptr, on success a
 * heap-allocated buffer the caller must @c free().
 *
 * @see ReprOptions  Configuration for formatting behavior.
 * @see ReprContext  Internal derived state engine.
 */

#pragma once

#include <ncore/core/status.h>
#include <ncore/repr/repr_options.h>
#include <ncore/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Produce a normal-mode string representation of a tensor.
 *
 * @details
 * Renders the tensor's data values in a bracketed, multidimensional
 * format. In this mode, a @c dtype suffix is only appended if the
 * tensor's data type is not the library default (@c Float32).
 *
 * @param[in]  ten Pointer to the tensor to render.
 * @param[out] out Slot for the heap-allocated null-terminated string.
 *
 * @return @c novaSuccess with @c *out owned by the caller, or the
 *         failure reason with @c *out set to @c nullptr
 *         (@c novaInvalidPointer for null arguments,
 *         @c novaInvalidTensor for unallocated input,
 *         @c novaOutOfMemory when the string cannot be built,
 *         propagated transfer status for GPU tensors).
 */
novaStatus_t tensor_repr(const Tensor *ten, char **out);

/**
 * @brief Produce a debug-mode string representation of a tensor.
 *
 * @details
 * Same rendering as @ref tensor_repr() plus a metadata footer:
 * dtype, shape, device placement, and autograd info
 * (@c requires_grad or @c grad_fn).
 *
 * @param[in]  ten Pointer to the tensor to render.
 * @param[out] out Slot for the heap-allocated null-terminated string.
 *
 * @return Same contract as @ref tensor_repr().
 */
novaStatus_t tensor_repr_debug(const Tensor *ten, char **out);

/**
 * @brief Produce a string representation with full control via
 *        options.
 *
 * @details
 * Renders the tensor under a @ref ReprOptions controlling thresholds,
 * precision, scientific notation, and other formatting parameters.
 *
 * @param[in]  ten  Pointer to the tensor to render.
 * @param[in]  opts Pointer to a @ref ReprOptions struct. If
 *                  @c nullptr, library defaults are used.
 * @param[out] out  Slot for the heap-allocated null-terminated string.
 *
 * @return Same contract as @ref tensor_repr().
 *
 * @see repr_default_options()
 */
novaStatus_t tensor_repr_with_options(const Tensor *ten,
                                      const ReprOptions *opts, char **out);

/**
 * @brief Print a tensor's normal-mode representation to standard
 *        output.
 *
 * @details
 * Convenience wrapper that renders, writes the result to @c stdout
 * followed by a newline, and frees the allocated memory.
 *
 * @param[in] ten Pointer to the tensor to print.
 *
 * @return @c novaSuccess, or the rendering failure reason. A
 *         @c printf failure reports @c novaRuntimeError.
 */
novaStatus_t tensor_print(const Tensor *ten);

/**
 * @brief Print a tensor's debug-mode representation to standard
 *        output.
 *
 * @details
 * Convenience wrapper that renders in debug mode, writes the result
 * to @c stdout followed by a newline, and frees the allocated
 * memory.
 *
 * @param[in] ten Pointer to the tensor to print.
 *
 * @return Same contract as @ref tensor_print().
 */
novaStatus_t tensor_print_debug(const Tensor *ten);

#ifdef __cplusplus
}
#endif
