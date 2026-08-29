/**
 * @file dense_layout.c
 * @brief Optimized contiguous-tensor layout renderer implementation.
 *
 * @details
 * Produces NovaNN bracketed output for tensors whose size is
 * within the summarization threshold. Features a high-performance
 * fast-path for contiguous tensors that uses direct pointer
 * increments in the innermost loops, bypassing expensive
 * multidimensional offset calculations.
 *
 * @section separators Separators
 *
 * @li Last dimension: @c ", " between elements.
 * @li Second-to-last: @c ",\n" + indentation between rows.
 * @li Outer dimensions: @c ",\n\n" + indentation between higher-
 *   dimensional slices.
 *
 * @section architecture Architecture
 *
 * @li Recursive Descent: @c render_dim walks the tensor dimensions
 *   from outer (dim 0) to inner (@c ndims-1).
 * @li Fast-Path: If @ref is_contiguous() returns @c true, a base
 *   pointer is propagated down the stack for direct data access.
 * @li Element Alignment: Column widths from @ref ReprContext ensure
 *   elements align vertically across rows.
 *
 * @see layouts.h        Renderer interface declarations.
 * @see repr_context.h   Formatting parameters.
 * @see element_fmt.h    Element formatting dispatch.
 */

#include <string.h>

#include <ncore/core/dtype.h>
#include <ncore/headeronly/macros.h>
#include <ncore/headeronly/tensor_utils.h>

#include "layouts.h"
#include "repr/formatters/element_fmt.h"

/**
 * @brief Compute the byte pointer for a coordinate vector.
 *
 * @param[in] ten    Pointer to the tensor.
 * @param[in] coords Coordinate vector.
 *
 * @return Byte pointer to the element in storage.
 */
static void *elem_ptr(const Tensor *ten, coords_t coords) {
  size_t off = compute_linear_byte_offset(coords, ten->ndims, ten->strides);
  return ten->data.u8 + off;
}

/**
 * @brief Append a string with right-justified padding.
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
 *
 * @param[in,out] sb   Output StringBuilder.
 * @param[in]     ctx  Representation context.
 * @param[in]     ten  Tensor being rendered.
 * @param[in]     ptr  Pointer to the element data.
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
 * Walks through each dimension. If the tensor is contiguous, the
 * @c base pointer is used for linear access in the innermost loop.
 *
 * @param[in,out] sb     Output StringBuilder.
 * @param[in]     ctx    Representation context.
 * @param[in]     dim    Current dimension index (0 .. @c ndims-1).
 * @param[in]     indent Indentation level for the current row.
 * @param[in]     coords Current coordinate vector.
 * @param[in]     base   Base data pointer for the contiguous fast path
 *                       (may be @c nullptr).
 */
static void render_dim(StringBuilder *sb, const ReprContext *ctx, size_t dim,
                       int indent, coords_t coords, const uint8 *base) {
  const Tensor *ten = ctx->tensor;
  bool contiguous = is_contiguous(ten);
  sb_append_char(sb, '[');

  if (dim == ten->ndims - 1) {
    size_t packing = dtype_packing_factor(ten->dtype);
    size_t count = ten->shape[dim] * packing;
    for (size_t i = 0; i < count; i++) {
      if (i > 0) {
        sb_append(sb, ", ");
      }
      size_t byte_idx = i / packing;
      size_t sub_idx = i % packing;
      const void *ptr = nullptr;
      if (contiguous && base) {
        ptr = base + (byte_idx * ten->item_size);
      } else {
        coords[dim] = byte_idx;
        ptr = elem_ptr(ten, coords);
      }
      ((ReprContext *)ctx)->sub_element_index = sub_idx;
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

      const uint8 *next_base = nullptr;
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
 * @brief Render a contiguous, non-summarized tensor.
 *
 * @param[in]     ctx Pointer to the representation context. Must not
 *                    be @c nullptr.
 * @param[in,out] sb  Pointer to the StringBuilder. Must not be
 *                    @c nullptr.
 */
void dense_layout_render(const ReprContext *ctx, StringBuilder *sb) {
  coords_t coords = {};
  const uint8 *base = (const uint8 *)ctx->tensor->data.u8;
  render_dim(sb, ctx, 0, 7, coords, base);
}
