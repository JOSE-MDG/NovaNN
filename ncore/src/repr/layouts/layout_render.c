/**
 * @file layout_render.c
 * @brief Single recursive layout renderer for all tensor topologies.
 *
 * @details
 * Renders contiguous tensors through a base-pointer fast path and
 * strided views through coordinates, truncating any dimension wider
 * than twice the edge count. One engine replaces the former
 * dense/strided/summarized split: contiguity only selects the
 * pointer arithmetic, never the code path.
 */

#include <string.h>

#include <ncore/core/dtype.h>
#include <ncore/headeronly/macros.h>
#include <ncore/headeronly/tensor_utils.h>

#include "layouts.h"
#include "repr/formatters/element_fmt.h"

/**
 * @brief Compute the byte pointer for a coordinate vector.
 */
static void *elem_ptr(const Tensor *ten, const coords_t coords) {
  size_t off = compute_linear_byte_offset(coords, ten->ndims, ten->strides);
  return ten->data.data + off;
}

/**
 * @brief Append a string right-justified to the column width.
 */
static void pad_and_append(StringBuilder *sb, const char *val, int len,
                           size_t width) {
  for (size_t i = (size_t)len; i < width; ++i) {
    sb_append_char(sb, ' ');
  }
  sb_append(sb, val);
}

/**
 * @brief Format and append a single element.
 *
 * @details
 * A corrupt dtype renders as a visible marker instead of failing
 * silently; column alignment applies to multi-dimensional output.
 */
static void append_elem(StringBuilder *sb, const ReprContext *ctx,
                        const Tensor *ten, const void *ptr) {
  char buf[128];
  int len = format_element(buf, sizeof(buf), ptr, ten, ctx);
  if (len < 0) {
    sb_append(sb, "?");
    return;
  }
  if (ten->ndims > 1) {
    pad_and_append(sb, buf, len, ctx->element_width);
  } else {
    sb_append(sb, buf);
  }
}

/**
 * @brief Map a display slot to a real index.
 *
 * @return true when the slot holds the truncation marker.
 */
static bool trunc_index(size_t d, size_t edge, bool truncate, size_t shape_dim,
                        size_t *actual) {
  if (truncate && d == edge) {
    return true;
  }
  *actual = (truncate && d > edge) ? shape_dim - edge + (d - edge - 1) : d;
  return false;
}

/**
 * @brief Recursively render one dimension.
 *
 * @details
 * The innermost dimension formats elements (expanding packed lanes);
 * outer dimensions recurse with row separators. Truncation and the
 * contiguous fast path compose freely: wide dims show edges with an
 * ellipsis regardless of layout.
 */
static void render_dim(StringBuilder *sb, const ReprContext *ctx, size_t dim,
                       int indent, coords_t coords, const unsigned char *base) {
  const Tensor *ten = ctx->tensor;
  const bool contiguous = is_contiguous(ten);
  const size_t edge = ctx->options.edge_items;
  const size_t shape_dim = ten->shape[dim];
  const bool truncate = shape_dim > 2U * edge;
  const size_t n_show = (int)truncate ? (2U * edge) + 1U : shape_dim;
  const bool last = (dim == ten->ndims - 1);
  sb_append_char(sb, '[');

  for (size_t d = 0; d < n_show; d++) {
    if (d > 0) {
      if (last) {
        sb_append(sb, ", ");
      } else {
        sb_append(sb, dim == ten->ndims - 2 ? ",\n" : ",\n\n");
        sb_append_repeated(sb, ' ', (size_t)indent + 1);
      }
    }
    size_t actual = 0;
    if (trunc_index(d, edge, truncate, shape_dim, &actual)) {
      sb_append(sb, "...");
      continue;
    }
    if (last) {
      const size_t packing = dtype_packing_factor(ten->dtype);
      const void *ptr = nullptr;
      if (contiguous && base) {
        ptr = (const void *)(base + (actual * ten->item_size));
      } else {
        coords[dim] = actual;
        ptr = elem_ptr(ten, coords);
      }
      for (size_t s = 0; s < packing; s++) {
        if (s > 0) {
          sb_append(sb, ", ");
        }
        ReprContext lane = *ctx;
        lane.sub_element_index = s;
        append_elem(sb, &lane, ten, ptr);
      }
    } else {
      const unsigned char *next_base = nullptr;
      if (contiguous && base) {
        next_base = base + (actual * ten->strides[dim]);
      } else {
        coords[dim] = actual;
      }
      render_dim(sb, ctx, dim + 1, indent + 1, coords, next_base);
    }
  }

  sb_append_char(sb, ']');
}

/**
 * @brief Render a tensor of any layout with edge-item truncation.
 *
 * @param[in]     ctx Pointer to the representation context. Must not
 *                    be @c nullptr.
 * @param[in,out] sb  Pointer to the StringBuilder. Must not be
 *                    @c nullptr.
 */
void layout_render(const ReprContext *ctx, StringBuilder *sb) {
  coords_t coords = {};
  /* Indent under the "tensor(" prefix. */
  const int indent = (int)(sizeof("tensor(") - 1);
  const unsigned char *base = ctx->tensor->data.data;
  render_dim(sb, ctx, 0, indent, coords, base);
}
