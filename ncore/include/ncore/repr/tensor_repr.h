/**
 * @file tensor_repr.h
 * @brief Public API for tensor string representation.
 *
 * @details
 * Entry point for producing human-readable PyTorch-style tensor strings.
 * All functions return a heap-allocated char* that the caller must free
 * with free().  Returns NULL on allocation failure.
 *
 * Convenience wrappers tensor_print() and tensor_print_debug() handle
 * memory management internally and write directly to stdout.
 *
 * ## Typical usage
 * @code{.cpp}
 * Tensor t = create_tensor(...);
 * tensor_print(&t);
 *
 * collect(&t);
 * @endcode
 *
 * @see ReprOptions  Fine-grained control.
 * @see ReprContext  Derived state built internally.
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
 * Shows data values.  A dtype suffix is appended only when the dtype
 * is not the default (Float32).
 *
 * @param[in] ten Tensor to render.
 * @return Heap-allocated string (caller must free()), or NULL on
 *         allocation failure.
 */
char *tensor_repr(const Tensor *ten);

/**
 * @brief Produce a debug-mode string representation of a tensor.
 *
 * Shows data values followed by dtype, shape, device, and gradient
 * information (requires_grad or grad_fn).
 *
 * @param[in] ten Tensor to render.
 * @return Heap-allocated string (caller must free()), or NULL on
 *         allocation failure.
 */
char *tensor_repr_debug(const Tensor *ten);

/**
 * @brief Produce a string representation with full control.
 *
 * @param[in]  ten  Tensor to render.
 * @param[in]  opts Pointer to a ReprOptions struct.  If NULL, defaults are
 *                  used (equivalent to tensor_repr()).
 * @return Heap-allocated string (caller must free()), or NULL on
 *         allocation failure.
 */
char *tensor_repr_with_options(const Tensor *ten, const ReprOptions *opts);

/**
 * @brief Print a tensor's normal-mode representation to stdout.
 *
 * Internally calls tensor_repr(), prints the result, and frees the
 * memory.  Safe to call on any valid tensor.
 *
 * @param[in] ten Tensor to print.
 */
void tensor_print(const Tensor *ten);

/**
 * @brief Print a tensor's debug-mode representation to stdout.
 *
 * Internally calls tensor_repr_debug(), prints the result, and frees
 * the memory.
 *
 * @param[in] ten Tensor to print.
 */
void tensor_print_debug(const Tensor *ten);

#ifdef __cplusplus
}
#endif
