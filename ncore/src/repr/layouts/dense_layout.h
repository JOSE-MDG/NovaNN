/**
 * @file dense_layout.h
 * @brief Public layout-renderer entry points.
 *
 * @details
 * Three render functions are exposed from this single header:
 *   - dense_layout_render()       -- contiguous, non-summarised tensors.
 *   - strided_layout_render()     -- view tensors (is_view_ == true).
 *   - summarized_layout_render()  -- tensors larger than the threshold.
 *
 * All take a const ReprContext and write into a StringBuilder.
 */

#pragma once

#include "repr/string_builder/string_builder.h"
#include <ncore/repr/repr_context.h>

/**
 * @brief Render a contiguous, non-summarised tensor.
 *
 * Produces the PyTorch-style bracketed, indented output for tensors
 * whose element count is within the summarisation threshold.
 *
 * @param[in] ctx ReprContext (must not be NULL).
 * @param[in] sb  Output StringBuilder (must not be NULL).
 */
void dense_layout_render(const ReprContext *ctx, StringBuilder *sb);

/**
 * @brief Render a view tensor using strided iteration.
 *
 * Identical output format to dense_layout_render(), but element access
 * goes through TensorIterator so that non-contiguous stride patterns
 * are handled correctly.
 *
 * @param[in] ctx ReprContext (must not be NULL).
 * @param[in] sb  Output StringBuilder (must not be NULL).
 */
void strided_layout_render(const ReprContext *ctx, StringBuilder *sb);

/**
 * @brief Render a tensor with edge-item truncation.
 *
 * Shows edge_items elements at each end of every dimension and replaces
 * omitted middle slices with `...`.
 *
 * @param[in] ctx ReprContext (must not be NULL).
 * @param[in] sb  Output StringBuilder (must not be NULL).
 */
void summarized_layout_render(const ReprContext *ctx, StringBuilder *sb);
