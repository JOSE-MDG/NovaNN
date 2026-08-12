/**
 * @file tensor_repr.c
 * @brief Top-level tensor string representation API implementation.
 *
 * @details
 * Implements the public interface for converting @ref Tensor objects
 * into human-readable strings. Orchestrates the entire representation
 * pipeline: device-to-host synchronization, context building, layout
 * selection, and metadata formatting.
 *
 * All returned strings are heap-allocated via the internal
 * @ref StringBuilder and must be explicitly freed by the caller using
 * @c free().
 *
 * @section representation-pipeline Representation Pipeline
 *
 * @li 1. Sanitization: Validates input tensor and handles
 *    device-to-host transfers.
 * @li 2. Contextualization: Calls @ref build_repr_context() to scan
 *    the tensor and derive formatting parameters.
 * @li 3. Initialization: Sets up a @ref StringBuilder with an
 *    initial capacity of 256 bytes.
 * @li 4. Header Emission: Appends the @c "tensor(" prefix.
 * @li 5. Layout Dispatch: Routes rendering to the appropriate
 *    engine:
 *    @li @c Scalar: Single element formatted directly.
 *    @li @c Summarized: Large tensors with edge-item truncation.
 *    @li @c Strided: Non-contiguous views via @ref TensorIterator.
 *    @li @c Dense: Optimized path for contiguous tensors.
 * @li 6. Metadata & Closure: Appends the closing suffix and
 *    metadata footer.
 *
 * If any step encounters a memory allocation failure, the
 * @ref StringBuilder propagates the error via its @ref SBStatus field
 * and the pipeline returns @c nullptr gracefully.
 *
 * @see tensor_repr.h     Public API definitions.
 * @see repr_context.h    Context and scanning logic.
 * @see string_builder.h  Memory-safe string construction.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ncore/core/alloc.h>
#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/core/storage.h>
#include <ncore/headeronly/macros.h>
#include <ncore/repr/repr_context.h>
#include <ncore/repr/tensor_repr.h>
#include <ncore/tensor.h>

#include "repr/formatters/element_fmt.h"
#include "repr/layouts/layouts.h"
#include "repr/metadata/metadata_fmt.h"
#include "repr/string_builder/string_builder.h"

/**
 * @brief Internal representation engine shared by all public entry
 *        points.
 *
 * @details
 * Performs the actual orchestration of the repr pipeline: builds the
 * context, initializes the builder, dispatches to layout renderers,
 * and appends metadata. On allocation failure, the error is
 * propagated and @c nullptr is returned.
 *
 * @param[in] ten  Pointer to the tensor to render. Must be
 *                 host-accessible.
 * @param[in] opts Pointer to the formatting options. Must not be
 *                 @c nullptr.
 *
 * @return Heap-allocated string on success, or @c nullptr on
 *         allocation failure. The caller takes ownership.
 *
 * @see build_repr_context()
 * @see metadata_fmt_append()
 */
static char *repr_internal(const Tensor *ten, const ReprOptions *opts) {
  auto ctx = build_repr_context(ten, opts);

  StringBuilder sb;
  sb_init(&sb, 256);

  if (sb_get_status(&sb) != SbOk) {
    return nullptr;
  }

  sb_append(&sb, "tensor(");

  if (ctx.is_scalar) {
    if (!ctx.is_meta) {
      const Tensor *t = ctx.tensor;
      char buf[128];
      const void *ptr = t->data.v;
      /*
       * Packed dtypes (e.g. Float4E2M1fn) store multiple sub-elements
       * in one storage unit. Emit every lane, not just the first.
       */
      const size_t packing = dtype_packing_factor(t->dtype);
      for (size_t i = 0; i < packing; i++) {
        if (i > 0) {
          sb_append(&sb, ", ");
        }
        ctx.sub_element_index = i;
        format_element(buf, sizeof(buf), ptr, t, &ctx);
        sb_append(&sb, buf);
      }
    }
  } else if (ctx.is_meta) {
    sb_append(&sb, "...");
  } else if (ctx.is_summarized) {
    summarized_layout_render(&ctx, &sb);
  } else if (is_view(ten)) {
    strided_layout_render(&ctx, &sb);
  } else {
    dense_layout_render(&ctx, &sb);
  }

  metadata_fmt_append(&ctx, &sb);

  if (sb_get_status(&sb) != SbOk) {
    sb_free(&sb);
    return nullptr;
  }
  return sb_build(&sb);
}

/**
 * @brief Produce a normal-mode string representation of a tensor.
 *
 * @details
 * Renders the tensor's data values in a bracketed, multidimensional
 * format. If the tensor is resident on a GPU, a temporary host-side
 * shadow is created via @ref safe_allocator() and the data is
 * transferred with @ref transf_tensor_from_device(). The shadow is
 * freed after rendering.
 *
 * @param[in] ten Pointer to the tensor to render. May be @c nullptr
 *                (returns @c nullptr).
 *
 * @return Heap-allocated string (caller must @c free()), or
 *         @c nullptr on failure.
 *
 * @see tensor_repr_debug()
 * @see tensor_repr_with_options()
 */
char *tensor_repr(const Tensor *ten) {
  if (ten == nullptr) {
    return nullptr;
  }

  if (ten->device != DEVICE_META && !is_allocated(ten)) {
    return nullptr;
  }

  Tensor rten = {};
  bool swapped = false;
  memcpy(&rten, ten, sizeof(Tensor));

  if (ten->device == DEVICE_GPU &&
      is_device_memory_handle(&ten->storage->handle)) {
    swapped = true;
    rten.storage = nullptr;
    rten.data.data = nullptr;
    rten.is_allocated_ = false;
    novaStatus_t status;

    // Create a storage for rten (in-place)
    status = safe_allocator(ten->storage->size_bytes, DEVICE_CPU, false,
                            nullptr, &rten, true);

    if (status.err != novaSuccess) {
      return nullptr;
    }

    /*
     * Temporarily change the `device` member from `rten` to
     * `DEVICE_CPU` to satisfy the `transf_tensor_from_device()` check
     * and allow printing the tensor with the metadata from
     * `DEVICE_GPU`.
     */
    rten.device = DEVICE_CPU;
    status = transf_tensor_from_device(ten, &rten);
    if (status.err != novaSuccess) {
      return nullptr;
    }
    rten.device = DEVICE_GPU;
  }

  auto opts = repr_default_options();
  char *repr = repr_internal(&rten, &opts);
  if (swapped) {
    if (ten->requires_grad_) {
      /* Note: Avoid releasing the original TensorGrad of source */
      rten.grad = nullptr;
    }
    collect(&rten);
  }
  return repr;
}

/**
 * @brief Produce a debug-mode string representation of a tensor.
 *
 * @details
 * Similar to @ref tensor_repr(), but sets the @ref ReprModeDebug
 * flag before dispatching. If the tensor is resident on a GPU, a
 * temporary host-side shadow is created and the data is transferred
 * to CPU for rendering.
 *
 * @param[in] ten Pointer to the tensor to render. May be @c nullptr
 *                (returns @c nullptr).
 *
 * @return Heap-allocated string (caller must @c free()), or
 *         @c nullptr on failure.
 *
 * @see tensor_repr()
 * @see tensor_repr_with_options()
 */
char *tensor_repr_debug(const Tensor *ten) {
  if (ten == nullptr) {
    return nullptr;
  }

  if (ten->device != DEVICE_META && !is_allocated(ten)) {
    return nullptr;
  }

  Tensor rten = {};
  bool swapped = false;
  memcpy(&rten, ten, sizeof(Tensor));

  if (ten->device == DEVICE_GPU &&
      is_device_memory_handle(&ten->storage->handle)) {
    swapped = true;
    rten.storage = nullptr;
    rten.data.data = nullptr;
    rten.is_allocated_ = false;
    novaStatus_t status;
    status = safe_allocator(ten->storage->size_bytes, DEVICE_CPU, false,
                            nullptr, &rten, true);

    if (status.err != novaSuccess) {
      return nullptr;
    }

    /*
     * Temporarily change the `device` member from `rten` to
     * `DEVICE_CPU` to satisfy the `transf_tensor_from_device()` check
     * and allow printing the tensor with the metadata from
     * `DEVICE_GPU`.
     */
    rten.device = DEVICE_CPU;
    status = transf_tensor_from_device(ten, &rten);
    if (status.err != novaSuccess) {
      collect(&rten);
      return nullptr;
    }
    rten.device = DEVICE_GPU;
  }

  auto opts = repr_default_options();
  opts.mode = ReprModeDebug;
  char *repr = repr_internal(&rten, &opts);
  if (swapped) {
    if (ten->requires_grad_) {
      /* Note: Avoid releasing the original TensorGrad of source */
      rten.grad = nullptr;
    }
    collect(&rten);
  }
  return repr;
}

/**
 * @brief Produce a string representation with full control via
 *        options.
 *
 * @details
 * Advanced entry point that accepts a @ref ReprOptions struct to
 * customize thresholds, precision, scientific notation, and other
 * formatting parameters. If @p opts is @c nullptr, defaults are used
 * (equivalent to @ref tensor_repr()).
 *
 * @param[in]  ten  Pointer to the tensor to render. May be @c nullptr
 *                  (returns @c nullptr).
 * @param[in]  opts Pointer to a @ref ReprOptions struct. If
 *                  @c nullptr, defaults are used.
 *
 * @return Heap-allocated string (caller must @c free()), or
 *         @c nullptr on failure.
 *
 * @see repr_default_options()
 * @see tensor_repr()
 */
char *tensor_repr_with_options(const Tensor *ten, const ReprOptions *opts) {
  if (ten == nullptr) {
    return nullptr;
  }
  if (opts == nullptr) {
    return tensor_repr(ten);
  }
  if (ten->device != DEVICE_META && !is_allocated(ten)) {
    return nullptr;
  }

  Tensor rten = {};
  bool swapped = false;
  memcpy(&rten, ten, sizeof(Tensor));

  if (ten->device == DEVICE_GPU &&
      is_device_memory_handle(&ten->storage->handle)) {
    swapped = true;
    rten.storage = nullptr;
    rten.data.data = nullptr;
    rten.is_allocated_ = false;
    novaStatus_t status;
    status = safe_allocator(ten->storage->size_bytes, DEVICE_CPU, false,
                            nullptr, &rten, true);
    if (status.err != novaSuccess) {
      return nullptr;
    }

    /*
     * Temporarily change the `device` member from `rten` to
     * `DEVICE_CPU` to satisfy the `transf_tensor_from_device()` check
     * and allow printing the tensor with the metadata from
     * `DEVICE_GPU`.
     */
    rten.device = DEVICE_CPU;
    status = transf_tensor_from_device(ten, &rten);
    if (status.err != novaSuccess) {
      return nullptr;
    }
    rten.device = DEVICE_GPU;
  }

  char *repr = repr_internal(&rten, opts);
  if (swapped) {
    if (ten->requires_grad_) {
      /* Note: Avoid releasing the original TensorGrad of source */
      rten.grad = nullptr;
    }
    collect(&rten);
  }
  return repr;
}

/**
 * @brief Print a tensor's normal-mode representation to standard
 *        output.
 *
 * @details
 * Convenience wrapper that calls @ref tensor_repr(), writes the
 * result to @c stdout followed by a newline, and automatically frees
 * the allocated memory. GPU tensors are handled transparently via
 * host-side shadowing.
 *
 * @param[in] ten Pointer to the tensor to print. May be @c nullptr
 *                (no-op).
 */
void tensor_print(const Tensor *ten) {
  if (ten == nullptr) {
    return;
  }
  char *s = tensor_repr(ten);
  if (s != nullptr) {
    printf("%s\n", s);
    free(s);
  }
}

/**
 * @brief Print a tensor's debug-mode representation to standard
 *        output.
 *
 * @details
 * Convenience wrapper that calls @ref tensor_repr_debug(), writes
 * the result to @c stdout followed by a newline, and automatically
 * frees the allocated memory. GPU tensors are handled transparently
 * via host-side shadowing.
 *
 * @param[in] ten Pointer to the tensor to print. May be @c nullptr
 *                (no-op).
 */
void tensor_print_debug(const Tensor *ten) {
  if (ten == nullptr) {
    return;
  }
  char *s = tensor_repr_debug(ten);
  if (s != nullptr) {
    printf("%s\n", s);
    free(s);
  }
}
