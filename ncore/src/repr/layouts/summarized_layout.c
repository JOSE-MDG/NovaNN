/**
 * @file summarized_layout.c
 * @brief Summarised (truncated) layout for tensors larger than the
 * threshold.
 *
 * @details
 * Shows edge_items elements at each end of every dimension and replaces
 * omitted middle slices with `...`.  Follows PyTorch's convention of
 * blank-line separators between 2D slices.  Per-element formatting
 * delegates to the dispatch table in element_fmt.h.
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
  for (size_t i = (size_t)len; i < width; i++) {
    sb_append_char(sb, ' ');
  }
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
 * @brief Append `...` without extra padding.
 */
static void append_ellipsis(StringBuilder *sb, const ReprContext *ctx) {
  (void)ctx;
  sb_append(sb, "...");
}

/**
 * @brief Render one dimension with edge-item truncation.
 *
 * For each dimension, if shape[dim] > 2 * edge_items, the middle is
 * replaced by a single `...` entry.  Inner dimensions are rendered
 * recursively so that 2D slices get blank-line separators.
 */
static void render_range(StringBuilder *sb, const ReprContext *ctx, size_t dim,
                         int indent, size_t *coords) {
  const Tensor *ten = ctx->tensor;
  sb_append_char(sb, '[');

  size_t shape_dim = ten->shape[dim];
  size_t edge = ctx->options.edge_items;
  bool truncate = (shape_dim > 2 * edge);
  size_t n_show = truncate ? (edge * 2) + 1 : shape_dim;

  if (dim == ten->ndims - 1) {
    for (size_t d = 0; d < n_show; d++) {
      if (d > 0) {
        sb_append(sb, ", ");
      }
      size_t actual_idx;
      bool is_ellipsis = false;
      if (truncate && d == edge) {
        is_ellipsis = true;
        actual_idx = 0;
      } else if (truncate && d > edge) {
        actual_idx = shape_dim - edge + (d - edge - 1);
      } else {
        actual_idx = d;
      }
      if (is_ellipsis) {
        append_ellipsis(sb, ctx);
      } else {
        coords[dim] = actual_idx;
        void *ptr = elem_ptr(ten, coords);
        append_elem(sb, ctx, ten, ptr);
      }
    }
  } else if (dim == ten->ndims - 2) {
    for (size_t d = 0; d < n_show; d++) {
      if (d > 0) {
        sb_append(sb, ",\n");
        sb_append_repeated(sb, ' ', (size_t)(indent + 1));
      }
      size_t actual_idx;
      bool is_ellipsis = false;
      if (truncate && d == edge) {
        is_ellipsis = true;
        actual_idx = 0;
      } else if (truncate && d > edge) {
        actual_idx = shape_dim - edge + (d - edge - 1);
      } else {
        actual_idx = d;
      }
      if (is_ellipsis) {
        append_ellipsis(sb, ctx);
      } else {
        coords[dim] = actual_idx;
        render_range(sb, ctx, dim + 1, indent + 1, coords);
      }
    }
  } else {
    for (size_t d = 0; d < n_show; d++) {
      if (d > 0) {
        sb_append(sb, ",\n\n");
        sb_append_repeated(sb, ' ', (size_t)(indent + 1));
      }
      size_t actual_idx;
      bool is_ellipsis = false;
      if (truncate && d == edge) {
        is_ellipsis = true;
        actual_idx = 0;
      } else if (truncate && d > edge) {
        actual_idx = shape_dim - edge + (d - edge - 1);
      } else {
        actual_idx = d;
      }
      if (is_ellipsis) {
        append_ellipsis(sb, ctx);
      } else {
        coords[dim] = actual_idx;
        render_range(sb, ctx, dim + 1, indent + 1, coords);
      }
    }
  }

  sb_append_char(sb, ']');
}

void summarized_layout_render(const ReprContext *ctx, StringBuilder *sb) {
  size_t coords[64] = {0};
  render_range(sb, ctx, 0, 7, coords);
}
