/**
 * @file summarized_layout.c
 * @brief Implementation of the truncated (summarized) tensor layout renderer.
 *
 * @details
 * This module provides the logic for displaying high-dimensional tensors
 * that exceed the user-defined element threshold. It prevents terminal
 * saturation by only showing a small neighborhood of elements (edge items)
 * around the boundaries of each dimension.
 *
 * The renderer is fully strided-aware, allowing it to correctly summarize
 * both contiguous and view tensors.
 *
 * ## Architecture
 * - **Recursive Range Rendering**: `render_range` decides for each dimension
 *   whether to show all elements or apply truncation.
 * - **Ellipsis Injection**: If a dimension is truncated, a central "..."
 *   marker is injected to represent the omitted elements.
 * - **Strided Navigation**: Uses @ref compute_linear_byte_offset() to ensure
 *   that the sampled edge items are correctly resolved in memory.
 *
 * @see dense_layout.h Renderer interface.
 * @see repr_options.h Edge-item and threshold configuration.
 */

#include <ncore/core/dtype.h>
#include <ncore/headeronly/macros.h>
#include <ncore/headeronly/tensor_utils.h>
#include <string.h>

#include "layouts.h"
#include "repr/formatters/element_fmt.h"

/**
 * @brief Helper to resolve a pointer for a coordinate vector.
 */
static void *elem_ptr(const Tensor *ten, const coords_t coords) {
  size_t off = compute_linear_byte_offset(coords, ten->ndims, ten->strides);
  return (uint8 *)ten->data.u8 + off;
}

/**
 * @brief Helper to append right-padded element strings.
 */
static void pad_and_append(StringBuilder *sb, const char *val, int len,
                           size_t width) {
  for (size_t i = (size_t)len; i < width; i++) {
    sb_append_char(sb, ' ');
  }
  sb_append(sb, val);
}

/**
 * @brief Format and append a single element.
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
 * @brief Append the truncation marker to the builder.
 */
static void append_ellipsis(StringBuilder *sb) { sb_append(sb, "..."); }

/**
 * @brief Recursively render a range of elements for a specific dimension.
 */
static void render_range(StringBuilder *sb, const ReprContext *ctx, size_t dim,
                         int indent, coords_t coords) {
  const Tensor *ten = ctx->tensor;
  sb_append_char(sb, '[');

  size_t shape_dim = ten->shape[dim];
  size_t edge = ctx->options.edge_items;
  bool truncate = (shape_dim > 2 * edge);
  size_t n_show = (int)truncate ? (edge * 2) + 1 : shape_dim;

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
        append_ellipsis(sb);
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
        sb_append_repeated(sb, ' ', (size_t)indent + 1);
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
        append_ellipsis(sb);
      } else {
        coords[dim] = actual_idx;
        render_range(sb, ctx, dim + 1, indent + 1, coords);
      }
    }
  } else {
    for (size_t d = 0; d < n_show; d++) {
      if (d > 0) {
        sb_append(sb, ",\n\n");
        sb_append_repeated(sb, ' ', (size_t)indent + 1);
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
        append_ellipsis(sb);
      } else {
        coords[dim] = actual_idx;
        render_range(sb, ctx, dim + 1, indent + 1, coords);
      }
    }
  }

  sb_append_char(sb, ']');
}

/**
 * @brief Render a tensor with edge-item truncation.
 */
void summarized_layout_render(const ReprContext *ctx, StringBuilder *sb) {
  coords_t coords = {0};
  render_range(sb, ctx, 0, 7, coords);
}
