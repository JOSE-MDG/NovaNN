/**
 * @file tensor_repr.h
 * @brief Public API for tensor string representation in NovaNN.
 *
 * @details
 * This header declares the high-level entry points for generating
 * human-readable, PyTorch-style string representations of tensors.
 * It supports various verbosity modes (normal vs. debug) and provides
 * both heap-allocating functions and convenience printing wrappers.
 *
 * All functions returning `char*` transfer ownership of the heap-allocated
 * memory to the caller. The caller is responsible for calling `free()` on
 * the result to prevent memory leaks.
 *
 * ## Typical Usage
 * @code{.c}
 *   Tensor t = create_tensor(...);
 *   
 *   // Print to stdout immediately
 *   tensor_print(&t);
 *
 *   // Or capture the string for logging
 *   char *s = tensor_repr(&t);
 *   if (s) {
 *       LOG_INFO("Result: %s", s);
 *       free(s);
 *   }
 *
 *   collect(&t);
 * @endcode
 *
 * @see ReprOptions  Configuration for formatting behavior.
 * @see ReprContext  Internal derived state engine.
 */

#pragma once

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
 * format. In this mode, a `dtype` suffix is only appended if the tensor's
 * data type is not the library default (@ref Float32).
 *
 * @param[in] ten Pointer to the tensor to render. Must not be NULL.
 *
 * @return Heap-allocated null-terminated string on success, or NULL on
 *         allocation failure. The caller must `free()` the result.
 */
char *tensor_repr(const Tensor *ten);

/**
 * @brief Produce a debug-mode string representation of a tensor.
 *
 * @details
 * Similar to @ref tensor_repr(), but appends a comprehensive metadata
 * footer containing the tensor's dtype, shape, device placement, and
 * autograd information (requires_grad or grad_fn).
 *
 * @param[in] ten Pointer to the tensor to render. Must not be NULL.
 *
 * @return Heap-allocated null-terminated string on success, or NULL on
 *         allocation failure. The caller must `free()` the result.
 */
char *tensor_repr_debug(const Tensor *ten);

/**
 * @brief Produce a string representation with full control via options.
 *
 * @details
 * Advanced entry point that accepts a @ref ReprOptions struct to
 * customize thresholds, precision, scientific notation, and other
 * formatting parameters.
 *
 * @param[in]  ten  Pointer to the tensor to render. Must not be NULL.
 * @param[in]  opts Pointer to a @ref ReprOptions struct. If NULL, library
 *                  defaults are used (equivalent to @ref tensor_repr()).
 *
 * @return Heap-allocated null-terminated string on success, or NULL on
 *         allocation failure. The caller must `free()` the result.
 *
 * @see repr_default_options()
 */
char *tensor_repr_with_options(const Tensor *ten, const ReprOptions *opts);

/**
 * @brief Print a tensor's normal-mode representation to standard output.
 *
 * @details
 * Convenience wrapper that internally calls @ref tensor_repr(), writes
 * the result to `stdout` followed by a newline, and automatically
 * frees the allocated memory.
 *
 * @param[in] ten Pointer to the tensor to print. Must not be NULL.
 */
void tensor_print(const Tensor *ten);

/**
 * @brief Print a tensor's debug-mode representation to standard output.
 *
 * @details
 * Convenience wrapper that internally calls @ref tensor_repr_debug(),
 * writes the result to `stdout` followed by a newline, and automatically
 * frees the allocated memory.
 *
 * @param[in] ten Pointer to the tensor to print. Must not be NULL.
 */
void tensor_print_debug(const Tensor *ten);

#ifdef __cplusplus
}
#endif
