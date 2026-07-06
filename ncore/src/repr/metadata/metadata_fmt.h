/**
 * @file metadata_fmt.h
 * @brief Metadata suffix formatter for tensor representation.
 *
 * @details
 * Declares @ref metadata_fmt_append(), which appends the metadata
 * footer (dtype, shape, device, grad info) to a tensor's string
 * representation. The footer logic is mode-sensitive, providing
 * minimal output in normal mode and comprehensive diagnostics in
 * debug mode.
 *
 * @see metadata_fmt.c  Footer implementation.
 * @see tensor_repr.h   High-level API.
 */

#pragma once

#include <ncore/repr/repr_context.h>

#include "repr/string_builder/string_builder.h"

/**
 * @brief Append the metadata suffix and finalize the tensor string.
 *
 * @details
 * Writes the closing `")"` and any required metadata fields (e.g.,
 * `dtype=float32`, `device=cuda`) to the provided builder. In
 * normal mode, only non-default values are emitted. In debug mode,
 * all fields are always shown on a new line.
 *
 * @param[in]     ctx Pointer to the representation context. Must not
 *                    be `nullptr`.
 * @param[in,out] sb  Pointer to the StringBuilder. Must not be
 *                    `nullptr`.
 */
void metadata_fmt_append(const ReprContext *ctx, StringBuilder *sb);
