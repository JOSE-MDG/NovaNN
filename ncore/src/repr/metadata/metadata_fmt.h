/**
 * @file metadata_fmt.h
 * @brief Metadata suffix formatter for tensor representation.
 *
 * @details
 * This header defines the interface for appending the metadata footer
 * (dtype, shape, device, grad info) to a tensor's string representation.
 * The footer logic is mode-sensitive, providing minimal output in
 * normal mode and comprehensive diagnostics in debug mode.
 *
 * @see metadata_fmt.c Footer implementation.
 * @see tensor_repr.h High-level API.
 */

#pragma once

#include "repr/string_builder/string_builder.h"
#include <ncore/repr/repr_context.h>

/**
 * @brief Append the metadata suffix and finalize the tensor string.
 *
 * @details
 * Writes the closing ")" and any required metadata fields (e.g.,
 * `dtype=float32`, `device=cuda`) to the provided builder.
 *
 * @param[in]     ctx Pointer to the representation context.
 * @param[in,out] sb  Pointer to the StringBuilder.
 */
void metadata_fmt_append(const ReprContext *ctx, StringBuilder *sb);
