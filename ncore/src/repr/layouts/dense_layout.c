/**
 * @file dense_layout.c
 * @brief PyTorch-style dense-tensor string layout.
 *
 * @details
 * Produces the bracketed, indented output for contiguous tensors whose
 * element count is within the summarisation threshold.  Delegates per-
 * element formatting to the dispatch table in element_fmt.h so that
 * dtype-specific formatting lives in one place.
 */

#include "dense_layout.h"
#include "repr/formatters/element_fmt.h"
#include <ncore/dtype.h>
#include <ncore/macros.h>
#include <string.h>

/**
 * @brief Compute a byte pointer to an element from its coordinates.
 */
static void *elem_ptr(const Tensor *ten, const size_t *coords) {
  size_t off = ten->offset;
  for (size_t d = 0; d < ten->ndims; d++)
    off += coords[d] * ten->strides[d];
  return (uint8_t *)ten->data.data + off;
}

/**
 * @brief Write a string to the builder, right-padded to a fixed width.
 */
static void pad_and_append(StringBuilder *sb, const char *val, int len,
                           size_t width) {
  for (size_t i = (size_t)len; i < width; i++)
    sb_append_char(sb, ' ');
  sb_append(sb, val);
}

/**
 * @brief Format one element and append it (padded for 2D+, raw for 1D).
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
 * @brief Recursively render dimensions dim .. ndims-1.
 *
 * @param sb      Output builder.
 * @param ctx     ReprContext.
 * @param dim     Current dimension index (0 = outermost).
 * @param indent  Column position of the opening `[`.
 * @param coords  Coordinate array (updated in place).
 */
static void render_dim(StringBuilder *sb, const ReprContext *ctx, size_t dim,
                       int indent, size_t *coords) {
  const Tensor *ten = ctx->tensor;
  sb_append_char(sb, '[');

  if (dim == ten->ndims - 1) {
    for (size_t i = 0; i < ten->shape[dim]; i++) {
      if (i > 0) {
        sb_append(sb, ", ");
      }
      coords[dim] = i;
      void *ptr = elem_ptr(ten, coords);
      append_elem(sb, ctx, ten, ptr);
    }
  } else if (dim == ten->ndims - 2) {
    for (size_t i = 0; i < ten->shape[dim]; i++) {
      if (i > 0) {
        sb_append(sb, ",\n");
        sb_append_repeated(sb, ' ', (size_t)(indent + 1));
      }
      coords[dim] = i;
      render_dim(sb, ctx, dim + 1, indent + 1, coords);
    }
  } else {
    for (size_t i = 0; i < ten->shape[dim]; i++) {
      if (i > 0) {
        sb_append(sb, ",\n\n");
        sb_append_repeated(sb, ' ', (size_t)(indent + 1));
      }
      coords[dim] = i;
      render_dim(sb, ctx, dim + 1, indent + 1, coords);
    }
  }

  sb_append_char(sb, ']');
}

void dense_layout_render(const ReprContext *ctx, StringBuilder *sb) {
  size_t coords[NOVA_MAX_DIMS] = {0};
  render_dim(sb, ctx, 0, 7, coords);
}
