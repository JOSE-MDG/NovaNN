/**
 * @file strided_layout.c
 * @brief Implementation of the general-purpose strided layout renderer.
 *
 * @details
 * This module provides a robust rendering path for tensors that do not
 * meet contiguity requirements (e.g., slices, transposed views). It
 * leverages the @ref TensorIterator to navigate the multidimensional
 * strided space element-by-element.
 *
 * While slightly less performant than @ref dense_layout.c, it ensures
 * correct data access for any valid stride pattern supported by the
 * NovaNN core.
 *
 * ## Architecture
 * - **Iterator-Driven**: Uses @ref iter_init() and @ref iter_advance()
 *   to handle row-major navigation regardless of physical data layout.
 * - **Recursive Formatting**: Mirrors the bracketed and indented structure
 *   of the dense renderer for visual consistency.
 *
 * @see tensor_iterator.h Iteration engine.
 * @see dense_layout.h Renderer interface.
 */

#include <ncore/core/dtype.h>
#include <ncore/headeronly/macros.h>
#include <string.h>

#include "layouts.h"
#include "repr/formatters/element_fmt.h"
#include "repr/traversal/tensor_iterator.h"

/**
 * @brief Helper to append padded strings for alignment.
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
 *
 * @param[in,out] sb       Output StringBuilder.
 * @param[in]     ctx      Pointer to the representation context.
 * @param[in]     ten      Pointer to the tensor.
 * @param[in]     byte_off Byte distance from the start of the data buffer.
 */
static void append_elem_at(StringBuilder *sb, const ReprContext *ctx,
                           const Tensor *ten, size_t byte_off) {
  char buf[128];
  void *ptr = (uint8 *)ten->data.u8 + byte_off;
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
    for (size_t i = 0; i < ten->shape[dim]; i++) {
      if (i > 0) {
        sb_append(sb, ", ");
      }
      size_t off = iter_byte_offset(it);
      append_elem_at(sb, ctx, ten, off);
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
 *
 * @param[in]  ctx Pointer to a fully initialised ReprContext.
 * @param[out] sb  Pointer to the StringBuilder.
 */
void strided_layout_render(const ReprContext *ctx, StringBuilder *sb) {
  TensorIterator it;
  iter_init(&it, ctx->tensor);
  render_dim(sb, ctx, 0, 7, &it);
}
