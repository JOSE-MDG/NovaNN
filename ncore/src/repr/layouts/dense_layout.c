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
#include <ncore/headeronly/tensor_utils.h>
#include <ncore/macros.h>
#include <string.h>

/**
 * @brief Compute a byte pointer to an element from its coordinates.
 *
 * @param[in] ten    The tensor.
 * @param[in] coords Multi-dimensional coordinate array.
 * @return Byte pointer to the element in tensor storage.
 */
static void *elem_ptr(const Tensor *ten, coords_t coords) {
  size_t off = compute_linear_byte_offset(coords, ten->ndims, ten->strides);
  return ten->data.u8 + off;
}

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
  for (size_t i = (size_t)len; i < width; i++)
    sb_append_char(sb, ' ');
  sb_append(sb, val);
}

/**
 * @brief Format one element and append it (padded for 2D+, raw for 1D).
 *
 * @param[in] sb  Output StringBuilder.
 * @param[in] ctx ReprContext.
 * @param[in] ten The tensor.
 * @param[in] ptr Pointer to the element in storage.
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
 * Separator conventions:
 *   - Last dimension:  ", " between elements.
 *   - Second-to-last:  ",\\n" + indent between 2D slices.
 *   - Other dims:      ",\\n\\n" + indent between higher-d slices.
 *
 * @param[in] sb     Output StringBuilder.
 * @param[in] ctx    ReprContext.
 * @param[in] dim    Current dimension index (0 = outermost).
 * @param[in] indent Column position of the opening `[`.
 * @param[in] coords Coordinate array (updated in place).
 */
static void render_dim(StringBuilder *sb, const ReprContext *ctx, size_t dim,
                       int indent, coords_t coords) {
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

/**
 * @brief Render a contiguous, non-summarised tensor.
 *
 * @param[in] ctx ReprContext (must not be NULL).
 * @param[in] sb  Output StringBuilder (must not be NULL).
 */
void dense_layout_render(const ReprContext *ctx, StringBuilder *sb) {
  size_t coords[NOVA_MAX_DIMS] = {0};
  render_dim(sb, ctx, 0, 7, coords);
}
