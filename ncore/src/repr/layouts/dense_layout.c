/**
 * @file dense_layout.c
 * @brief implementation of the optimized contiguous-tensor layout renderer.
 *
 * @details
 * This module produces PyTorch-style bracketed output for tensors whose
 * size is within the summarization threshold. It features a high-performance
 * fast-path for contiguous tensors that uses direct pointer increments
 * in the innermost loops, bypassing expensive multidimensional offset
 * calculations.
 *
 * Separators follow library conventions:
 * - Last dimension: ", " between elements.
 * - Second-to-last: ",\n" + indentation between rows.
 * - Outer dimensions: ",\n\n" + indentation between higher-dimensional slices.
 *
 * ## Architecture
 * - **Recursive Descent**: `render_dim` walks the tensor dimensions from
 *   outer (dim 0) to inner (ndims-1).
 * - **Fast-Path**: If @ref is_contiguous() returns true, a base pointer
 *   is propagated down the stack for direct data access.
 * - **Element Alignment**: Column widths from @ref ReprContext are used
 *   to ensure elements align vertically across rows.
 *
 * @see dense_layout.h Renderer interface.
 * @see repr_context.h Formatting parameters.
 */

#include <ncore/core/dtype.h>
#include <ncore/headeronly/macros.h>
#include <ncore/headeronly/tensor_utils.h>
#include <string.h>

#include "layouts.h"
#include "repr/formatters/element_fmt.h"

/**
 * @brief Calculate the byte pointer for a set of coordinates.
 *
 * @param[in] ten    Pointer to the tensor.
 * @param[in] coords Coordinate vector.
 *
 * @return Byte pointer to the element in storage.
 */
static void *elem_ptr(const Tensor *ten, coords_t coords) {
  size_t off = compute_linear_byte_offset(coords, ten->ndims, ten->strides);
  return (uint8 *)ten->data.u8 + off;
}

/**
 * @brief Append a string to the builder with right-justified padding.
 *
 * @param[in,out] sb    Output StringBuilder.
 * @param[in]     val   The formatted string to append.
 * @param[in]     len   Length of the string (excluding null).
 * @param[in]     width Desired column width.
 */
static void pad_and_append(StringBuilder *sb, const char *val, int len,
                           size_t width) {
  for (size_t i = (size_t)len; i < width; i++) {
    sb_append_char(sb, ' ');
  }
  sb_append(sb, val);
}

/**
 * @brief Format and append a single element to the builder.
 *
 * @details
 * Dispatches to @ref format_element() and applies column alignment
 * for multi-dimensional tensors.
 */
static void append_elem(StringBuilder *sb, const ReprContext *ctx,
                        const Tensor *ten, const void *ptr) {
  char buf[128];
  int len = format_element(buf, sizeof(buf), ptr, ten, ctx);
  if (ten->ndims > 1) {
    pad_and_append(sb, buf, len, ctx->element_width);
  } else {
    sb_append(sb, buf);
  }
}

/**
 * @brief Recursively render tensor dimensions with layout optimization.
 *
 * @details
 * Walks through each dimension. If the tensor is contiguous, it uses
 * the `base` pointer to perform linear access in the innermost loop.
 *
 * @param[in,out] sb     Output StringBuilder.
 * @param[in]     ctx    Pointer to the representation context.
 * @param[in]     dim    Index of the dimension being rendered.
 * @param[in]     indent Column position of the opening bracket.
 * @param[in,out] coords Coordinate vector (used for non-contiguous paths).
 * @param[in]     base   Current base pointer for the slice (contiguous only).
 */
static void render_dim(StringBuilder *sb, const ReprContext *ctx, size_t dim,
                       int indent, coords_t coords, const uint8 *base) {
  const Tensor *ten = ctx->tensor;
  bool contiguous = is_contiguous(ten);
  sb_append_char(sb, '[');

  if (dim == ten->ndims - 1) {
    for (size_t i = 0; i < ten->shape[dim]; i++) {
      if (i > 0) {
        sb_append(sb, ", ");
      }
      const void *ptr;
      if (contiguous && base) {
        ptr = base + (i * ten->item_size);
      } else {
        coords[dim] = i;
        ptr = elem_ptr(ten, coords);
      }
      append_elem(sb, ctx, ten, ptr);
    }
  } else {
    for (size_t i = 0; i < ten->shape[dim]; i++) {
      if (i > 0) {
        if (dim == ten->ndims - 2) {
          sb_append(sb, ",\n");
        } else {
          sb_append(sb, ",\n\n");
        }
        sb_append_repeated(sb, ' ', (size_t)indent + 1);
      }

      const uint8 *next_base = NULL;
      if (contiguous && base) {
        next_base = base + (i * ten->strides[dim]);
      } else if (!contiguous) {
        coords[dim] = i;
      }

      render_dim(sb, ctx, dim + 1, indent + 1, coords, next_base);
    }
  }

  sb_append_char(sb, ']');
}

/**
 * @brief Render a contiguous, non-summarised tensor.
 *
 * @details
 * Entry point for tensors within the truncation threshold. Automatically
 * detects contiguity to enable optimized rendering paths.
 *
 * @param[in]  ctx Pointer to a fully initialised ReprContext.
 * @param[out] sb  Pointer to the StringBuilder.
 */
void dense_layout_render(const ReprContext *ctx, StringBuilder *sb) {
  coords_t coords = {0};
  const uint8 *base = (const uint8 *)ctx->tensor->data.u8;
  render_dim(sb, ctx, 0, 7, coords, base);
}
