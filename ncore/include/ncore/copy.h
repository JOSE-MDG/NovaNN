/**
 * @file copy.h
 * @brief Tensor deep-copy interface.
 *
 * Declares the copyFn function-pointer type and the top-level
 * deepcopy() entry point.  The dispatch tables are defined in
 * copy.c and remain private to the translation unit.
 */

#pragma once

#include <ncore/tensor.h>

/**
 * @brief Copy function pointer type.
 * @param src Source tensor (read-only).
 * @param dst Destination tensor (write-only, same size as src).
 */
typedef void (*copyFn)(const Tensor *restrict, Tensor *restrict dst);

/**
 * @brief Type of a device-specific dispatch table covering all dtypes.
 *
 * An array of copyFn pointers indexed by DType_.  Used internally
 * by deepcopy() to select the correct per-dtype copy routine.
 */
typedef copyFn table[NUM_DTYPES];

/**
 * @brief Deep-copy a tensor.
 *
 * Allocates new storage for dst, copies all metadata and element data
 * from src, and recursively deep-copies the gradient graph.
 *
 * @param src Source tensor.
 * @param dst Destination tensor (must not be is_allocated_).
 */
void deepcopy(const Tensor *restrict src, Tensor *restrict dst);
