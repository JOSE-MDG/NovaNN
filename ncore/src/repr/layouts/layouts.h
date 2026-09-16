/**
 * @file layouts.h
 * @brief Public interface for the layout renderer.
 *
 * @details
 * Declares the single recursive layout engine. It renders
 * contiguous tensors through a base-pointer fast path and strided
 * views through coordinates, truncating any dimension wider than
 * twice the edge count. Contiguity selects pointer arithmetic,
 * never the code path.
 *
 * @see repr_context.h    Input parameters.
 * @see string_builder.h  Output mechanism.
 */

#pragma once

#include <ncore/repr/repr_context.h>

#include "repr/string_builder/string_builder.h"

/**
 * @brief Render a tensor of any layout with edge-item truncation.
 *
 * @param[in]     ctx Pointer to the representation context. Must not
 *                    be @c nullptr.
 * @param[in,out] sb  Pointer to the StringBuilder. Must not be
 *                    @c nullptr.
 */
void layout_render(const ReprContext *ctx, StringBuilder *sb);
