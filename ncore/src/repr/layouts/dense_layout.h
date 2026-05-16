/**
 * @file dense_layout.h
 * @brief Public layout-renderer entry points.
 *
 * @details
 * Three render functions are exposed from this single header:
 *   - dense_layout_render()       – contiguous, non-summarised tensors.
 *   - strided_layout_render()     – view tensors (is_view_ == true).
 *   - summarized_layout_render()  – tensors larger than the threshold.
 *
 * All take a const ReprContext and write into a StringBuilder.
 */

#pragma once

#include "repr/string_builder/string_builder.h"
#include <ncore/repr/repr_context.h>

/**
 * @brief Render a contiguous, non-summarised tensor.
 */
void dense_layout_render(const ReprContext *ctx, StringBuilder *sb);

/**
 * @brief Render a view tensor using strided iteration.
 */
void strided_layout_render(const ReprContext *ctx, StringBuilder *sb);

/**
 * @brief Render a tensor with edge-item truncation.
 */
void summarized_layout_render(const ReprContext *ctx, StringBuilder *sb);
