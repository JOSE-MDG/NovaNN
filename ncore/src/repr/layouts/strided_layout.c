/**
 * @file strided_layout.c
 * @brief General-purpose strided layout renderer implementation.
 *
 * @details
 * Provides a robust rendering path for tensors that do not meet
 * contiguity requirements (e.g., slices, transposed views). Uses
 * @ref TensorIterator to navigate the multidimensional strided
 * space element-by-element.
 *
 * While slightly less performant than @ref dense_layout.c, it
 * ensures correct data access for any valid stride pattern
 * supported by the NovaNN core.
 *
 * @section architecture Architecture
 *
 * @li Iterator-Driven: Uses @ref iter_init() and @ref iter_advance()
 *   to handle row-major navigation regardless of physical data
 *   layout.
 * @li Recursive Formatting: Mirrors the bracketed and indented
 *   structure of the dense renderer for visual consistency.
 *
 * @see layouts.h          Renderer interface declarations.
 * @see tensor_iterator.h  Iteration engine.
 * @see repr_context.h     Formatting parameters.
 */

#include <string.h>

#include <ncore/core/dtype.h>
#include <ncore/headeronly/macros.h>

#include "layouts.h"
#include "repr/formatters/element_fmt.h"
#include "repr/traversal/tensor_iterator.h"

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
 * @brief Format and append an element at a specific byte offset.
 */
static void append_elem_at(StringBuilder *sb, const ReprContext *ctx,
                           const Tensor *ten, size_t byte_off) {
  char buf[128];
  void *ptr = ten->data.u8 + byte_off;
  int len = format_element(buf, sizeof(buf), ptr, ten, ctx);
  if (ten->ndims > 1) {
    pad_and_append(sb, buf, len, ctx->element_width);
  } else {
    sb_append(sb, buf);
  }
}

/**
 * @brief Recursively render dimensions using strided offsets.
 */
static void render_dim(StringBuilder *sb, const ReprContext *ctx, size_t dim,
                       int indent, TensorIterator *it) {
  const Tensor *ten = ctx->tensor;
  sb_append_char(sb, '[');

  if (dim == ten->ndims - 1) {
    size_t packing = dtype_packing_factor(ten->dtype);
    for (size_t i = 0; i < ten->shape[dim]; i++) {
      if (i > 0) {
        sb_append(sb, ", ");
      }
      size_t off = iter_byte_offset(it);
      for (size_t s = 0; s < packing; s++) {
        if (s > 0) {
          sb_append(sb, ", ");
        }
        ((ReprContext *)ctx)->sub_element_index = s;
        append_elem_at(sb, ctx, ten, off);
      }
      iter_advance(it);
    }
  } else if (dim == ten->ndims - 2) {
    for (size_t i = 0; i < ten->shape[dim]; i++) {
      if (i > 0) {
        sb_append(sb, ",\n");
        sb_append_repeated(sb, ' ', (size_t)indent + 1);
      }
      render_dim(sb, ctx, dim + 1, indent + 1, it);
    }
  } else {
    for (size_t i = 0; i < ten->shape[dim]; i++) {
      if (i > 0) {
        sb_append(sb, ",\n\n");
        sb_append_repeated(sb, ' ', (size_t)indent + 1);
      }
      render_dim(sb, ctx, dim + 1, indent + 1, it);
    }
  }

  sb_append_char(sb, ']');
}

/**
 * @brief Render a view tensor using strided iteration.
 */
void strided_layout_render(const ReprContext *ctx, StringBuilder *sb) {
  TensorIterator it;
  iter_init(&it, ctx->tensor);
  render_dim(sb, ctx, 0, 7, &it);
}
