/**
 * @file layouts.h
 * @brief Public interface for multidimensional layout renderers.
 *
 * @details
 * Declares the entry points for the three primary layout engines in
 * the NovaNN representation module. Each engine is specialized for a
 * specific tensor topology or size:
 *
 * - **Dense**: High-performance rendering for contiguous tensors.
 * - **Strided**: General-purpose rendering for non-contiguous views.
 * - **Summarized**: Edge-item truncation for tensors exceeding display
 *   limits.
 *
 * All renderers operate on a @ref ReprContext and write their output
 * incrementally to a @ref StringBuilder.
 *
 * @see repr_context.h    Input parameters.
 * @see string_builder.h  Output mechanism.
 */

#pragma once

#include <ncore/repr/repr_context.h>

#include "repr/string_builder/string_builder.h"

/**
 * @brief Render a contiguous, non-summarised tensor to a string.
 *
 * @details
 * Performs optimized rendering of tensors within the truncation
 * threshold. Automatically detects contiguity to use fast pointer
 * arithmetic where possible.
 *
 * @param[in]     ctx Pointer to the representation context. Must not
 *                    be `nullptr`.
 * @param[in,out] sb  Pointer to the StringBuilder. Must not be
 *                    `nullptr`.
 */
void dense_layout_render(const ReprContext *ctx, StringBuilder *sb);

/**
 * @brief Render a non-contiguous view tensor using strided iteration.
 *
 * @details
 * Provides a robust fallback for views (slices, transposed tensors)
 * where data is not contiguous in memory. Uses @ref TensorIterator
 * to safely navigate the strided layout.
 *
 * @param[in]     ctx Pointer to the representation context. Must not
 *                    be `nullptr`.
 * @param[in,out] sb  Pointer to the StringBuilder. Must not be
 *                    `nullptr`.
 */
void strided_layout_render(const ReprContext *ctx, StringBuilder *sb);

/**
 * @brief Render a large tensor with edge-item truncation.
 *
 * @details
 * Prevents overwhelming output for large tensors by showing only a
 * fixed number of elements at each dimension's start and end,
 * replacing the middle with `"..."` ellipses.
 *
 * @param[in]     ctx Pointer to the representation context. Must not
 *                    be `nullptr`.
 * @param[in,out] sb  Pointer to the StringBuilder. Must not be
 *                    `nullptr`.
 */
void summarized_layout_render(const ReprContext *ctx, StringBuilder *sb);
