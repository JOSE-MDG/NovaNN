/**
 * @file metadata_fmt.h
 * @brief Metadata footer (dtype, shape, device, grad info).
 *
 * @details
 * Appends the closing suffix after the data block.  Behaviour depends
 * on the display mode:
 *
 * Normal mode:
 *   - dtype suffix omitted when dtype == Float32 and no grad info.
 *   - Meta tensors always show dtype and device.
 *   - grad_fn takes priority over requires_grad.
 *
 * Debug mode:
 *   - Always appends all fields on a continuation line.
 */

#pragma once

#include "repr/string_builder/string_builder.h"
#include <ncore/repr/repr_context.h>

/**
 * @brief Append the metadata suffix and close the outer `)`.
 *
 * The suffix is appended after the data-block closing `)` has already
 * been placed by the layout code.
 *
 * @param ctx ReprContext (mode, tensor pointer, etc.).
 * @param sb  Output builder.
 */
void metadata_fmt_append(const ReprContext *ctx, StringBuilder *sb);
