/**
 * @file strided_layout.c
 * @brief Layout for view tensors (is_view_ == true).
 *
 * @details
 * Identical output format to dense_layout, but element access goes
 * through TensorIterator::iter_byte_offset() instead of sequential
 * pointer arithmetic so that non-contiguous stride patterns are
 * handled correctly.  Per-element formatting goes through the
 * dispatch table in element_fmt.h.
 */

#include "dense_layout.h"
#include "repr/formatters/element_fmt.h"
#include "repr/traversal/tensor_iterator.h"
#include <ncore/dtype.h>
#include <ncore/macros.h>
#include <string.h>

/**
 * @brief Write a string to the builder, right-padded to a fixed width.
 *
 * @param[in] sb     Output StringBuilder.
 * @param[in] val    String value to append.
 * @param[in] len    Length of the string (excl. null).
 * @param[in] width  Desired minimum width (padding added before val).
 */
static void pad_and_append(StringBuilder *sb, const char *val, int len,
                           size_t width) {
  for (size_t i = (size_t)len; i < width; i++) {
    sb_append_char(sb, ' ');
  }
  sb_append(sb, val);
}

/**
 * @brief Format and append one element accessed via byte offset.
 *
 * @param[in] sb       Output StringBuilder.
 * @param[in] ctx      ReprContext.
 * @param[in] ten      The tensor.
 * @param[in] byte_off Byte offset into tensor storage.
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
 * @brief Recursively render dimensions via TensorIterator.
 *
 * @param[in] sb     Output StringBuilder.
 * @param[in] ctx    ReprContext.
 * @param[in] dim    Current dimension index (0 = outermost).
 * @param[in] indent Column position of the opening `[`.
 * @param[in] it     TensorIterator for byte-offset computation.
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
        sb_append_repeated(sb, ' ', (size_t)(indent + 1));
      }
      render_dim(sb, ctx, dim + 1, indent + 1, it);
    }
  } else {
    for (size_t i = 0; i < ten->shape[dim]; i++) {
      if (i > 0) {
        sb_append(sb, ",\n\n");
        sb_append_repeated(sb, ' ', (size_t)(indent + 1));
      }
      render_dim(sb, ctx, dim + 1, indent + 1, it);
    }
  }

  sb_append_char(sb, ']');
}

/**
 * @brief Render a view tensor using strided iteration.
 *
 * @param[in] ctx ReprContext (must not be NULL).
 * @param[in] sb  Output StringBuilder (must not be NULL).
 */
void strided_layout_render(const ReprContext *ctx, StringBuilder *sb) {
  TensorIterator it;
  iter_init(&it, ctx->tensor);
  render_dim(sb, ctx, 0, 7, &it);
}
