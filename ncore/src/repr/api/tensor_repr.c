/**
 * @file tensor_repr.c
 * @brief Top-level tensor repr implementation.
 *
 * @details
 * Implements the public API declared in tensor_repr.h.  Each function
 * follows the same internal pipeline:
 *   1. Obtain / merge ReprOptions.
 *   2. build_repr_context() -- scan tensor, derive display parameters.
 *   3. Initialise a StringBuilder.
 *   4. Emit "tensor(" prefix.
 *   5. Dispatch to the correct layout renderer:
 *        - Scalar:       format element directly.
 *        - Meta:         print "...".
 *        - Summarised:   summarized_layout_render().
 *        - View:         strided_layout_render() (via TensorIterator).
 *        - Dense:        dense_layout_render().
 *   6. metadata_fmt_append() -- close the ")" with optional suffix.
 *   7. sb_build() -- transfer heap buffer to caller.
 *
 * All returned strings are heap-allocated via the StringBuilder's
 * internal malloc/realloc.  The caller must free() every result.
 */

#include <ncore/dtype.h>
#include <ncore/macros.h>
#include <stdio.h>
#include <stdlib.h>

#include "repr/formatters/element_fmt.h"
#include "repr/layouts/dense_layout.h"
#include "repr/metadata/metadata_fmt.h"
#include "repr/string_builder/string_builder.h"
#include <ncore/repr/repr_context.h>
#include <ncore/repr/tensor_repr.h>

/**
 * @brief Internal implementation shared by all public entry points.
 *
 * @param[in] ten  Tensor to render.
 * @param[in] opts ReprOptions (must not be NULL).
 * @return Heap-allocated string (caller must free()), or NULL on failure.
 */
static char *repr_internal(const Tensor *ten, const ReprOptions *opts) {
  ReprContext ctx = build_repr_context(ten, opts);

  StringBuilder sb;
  sb_init(&sb, 256);

  sb_append(&sb, "tensor(");

  if (ctx.is_scalar) {
    const Tensor *t = ctx.tensor;
    char buf[128];
    const void *ptr = (const uint8 *)t->data.u8;
    /* Use the dispatch table for scalar elements too. */
    format_element(buf, sizeof(buf), ptr, t, &ctx);
    sb_append(&sb, buf);
  } else if (ctx.is_meta) {
    sb_append(&sb, "...");
  } else if (ctx.is_summarized) {
    summarized_layout_render(&ctx, &sb);
  } else if (ten->is_view_) {
    strided_layout_render(&ctx, &sb);
  } else {
    dense_layout_render(&ctx, &sb);
  }

  metadata_fmt_append(&ctx, &sb);
  return sb_build(&sb);
}

/**
 * @brief Produce a normal-mode string representation of a tensor.
 *
 * @param[in] ten Tensor to render.
 * @return Heap-allocated string (caller must free()), or NULL on
 *         allocation failure.
 */
char *tensor_repr(const Tensor *ten) {
  if (!ten) {
    return NULL;
  }
  ReprOptions opts = repr_default_options();
  opts.mode = REPR_MODE_NORMAL;
  return repr_internal(ten, &opts);
}

/**
 * @brief Produce a debug-mode string representation of a tensor.
 *
 * @param[in] ten Tensor to render.
 * @return Heap-allocated string (caller must free()), or NULL on
 *         allocation failure.
 */
char *tensor_repr_debug(const Tensor *ten) {
  if (!ten) {
    return NULL;
  }
  ReprOptions opts = repr_default_options();
  opts.mode = REPR_MODE_DEBUG;
  return repr_internal(ten, &opts);
}

/**
 * @brief Produce a string representation with full control.
 *
 * @param[in]  ten  Tensor to render.
 * @param[in]  opts Pointer to a ReprOptions struct.  If NULL, defaults are
 *                  used (equivalent to tensor_repr()).
 * @return Heap-allocated string (caller must free()), or NULL on
 *         allocation failure.
 */
char *tensor_repr_with_options(const Tensor *ten, const ReprOptions *opts) {
  if (!ten) {
    return NULL;
  }
  if (!opts) {
    return tensor_repr(ten);
  }
  return repr_internal(ten, opts);
}

/**
 * @brief Print a tensor's normal-mode representation to stdout.
 *
 * @param[in] ten Tensor to print.
 */
void tensor_print(const Tensor *ten) {
  if (!ten) {
    return;
  }
  char *s = tensor_repr(ten);
  if (s) {
    printf("%s\n", s);
    free(s);
  }
}

/**
 * @brief Print a tensor's debug-mode representation to stdout.
 *
 * @param[in] ten Tensor to print.
 */
void tensor_print_debug(const Tensor *ten) {
  if (!ten) {
    return;
  }
  char *s = tensor_repr_debug(ten);
  if (s) {
    printf("%s\n", s);
    free(s);
  }
}
