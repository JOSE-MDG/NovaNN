/**
 * @file tensor_repr.c
 * @brief Top-level implementation of the tensor string representation API.
 *
 * @details
 * This module implements the public interface for converting @ref Tensor
 * objects into human-readable strings. It orchestrates the entire
 * representation pipeline, including device-to-host synchronization,
 * context building, layout selection, and metadata formatting.
 *
 * All returned strings are heap-allocated via the internal @ref StringBuilder
 * and must be explicitly freed by the caller using `free()`.
 *
 * ## Architecture
 * The representation process follows a strict 6-step pipeline:
 * 1. **Sanitization**: Validates input tensor and handles Device-to-Host
 * transfers.
 * 2. **Contextualization**: Calls @ref build_repr_context() to scan the tensor
 *    and derive formatting parameters (precision, sci-notation, alignment).
 * 3. **Initialization**: Sets up a @ref StringBuilder with an initial capacity.
 * 4. **Header Emission**: Appends the "tensor(" prefix to the builder.
 * 5. **Layout Dispatch**: Routes the rendering to specialized layout engines:
 *    - **Scalar**: Formats the single element directly.
 *    - **Summarized**: Handles large tensors with edge-item truncation.
 *    - **Strided**: Handles non-contiguous views via @ref TensorIterator.
 *    - **Dense**: Optimized path for contiguous multi-dimensional tensors.
 * 6. **Metadata & Closure**: Appends the closing suffix and metadata footer.
 *
 * @see tensor_repr.h    Public API definitions.
 * @see repr_context.h   Context and scanning logic.
 * @see string_builder.h Memory-safe string construction.
 */

#include <ncore/core/alloc.h>
#include <ncore/core/dtype.h>
#include <ncore/core/storage.h>
#include <ncore/headeronly/macros.h>
#include <ncore/repr/repr_context.h>
#include <ncore/repr/tensor_repr.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "repr/formatters/element_fmt.h"
#include "repr/layouts/layouts.h"
#include "repr/metadata/metadata_fmt.h"
#include "repr/string_builder/string_builder.h"

/**
 * @brief Internal representation engine shared by all public entry points.
 *
 * @details
 * This static function performs the actual orchestration of the repr
 * pipeline. It builds the context, initializes the builder, dispatches
 * to layout renderers, and appends metadata.
 *
 * @param[in] ten  Pointer to the tensor to render. Must be host-accessible.
 * @param[in] opts Pointer to the formatting options. Must not be NULL.
 *
 * @return Heap-allocated string on success, or NULL on allocation failure.
 *         The caller takes ownership of the returned pointer.
 *
 * @see build_repr_context()
 * @see metadata_fmt_append()
 */
static char *repr_internal(const Tensor *ten, const ReprOptions *opts) {
  ReprContext ctx = build_repr_context(ten, opts);

  StringBuilder sb;
  sb_init(&sb, 256);

  sb_append(&sb, "tensor(");

  if (ctx.is_scalar) {
    if (!ctx.is_meta) {
      const Tensor *t = ctx.tensor;
      char buf[128];
      const void *ptr = (const uint8 *)t->data.u8;
      format_element(buf, sizeof(buf), ptr, t, &ctx);
      sb_append(&sb, buf);
    }
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
 * @brief Move a device-resident tensor's data to host memory for inspection.
 *
 * @details
 * Allocates a temporary buffer on the CPU and performs a asynchronous
 * device-to-host transfer. The metadata of the destination tensor
 * (including the device field) is preserved to ensure the representation
 * correctly identifies the tensor's native location.
 *
 * @param[in]     from Pointer to the source device tensor.
 * @param[in,out] dst  Pointer to the destination host-shadow tensor.
 *
 * @return @ref DeviceStatus indicating the result of the transfer.
 *         Success is indicated by a code of 0.
 *
 * @see transfer_to() Low-level memory transfer utility.
 */
static DeviceStatus move_from_device(const Tensor *restrict from,
                                     Tensor *restrict dst) {

  dst->storage =
      allocate_tensor_buffer(from->storage->size_bytes, DEVICE_CPU, false);

  if (dst->storage != NULL) {
    dst->data = dst->storage->ptr;
    dst->is_allocated_ = true;
    /* Note: We explicitly keep dst->device as from->device (GPU)
       so that metadata formatting correctly reports the original device. */
  }

  DeviceStatus status =
      transfer_to(DEVICE_GPU, DEVICE_CPU, (const void *)from->data.v,
                  dst->data.v, from->storage->size_bytes);

  return status;
}

/**
 * @brief Produce a normal-mode string representation of a tensor.
 *
 * @details
 * Performs a standard representation. If the tensor is resident on a
 * GPU, a temporary host-side shadow is created for the data scan.
 *
 * @param[in] ten Pointer to the tensor to render.
 *
 * @return Heap-allocated string (caller must free()), or NULL on failure.
 *
 * @see tensor_repr_debug()
 */
char *tensor_repr(const Tensor *ten) {
  if (ten == NULL) {
    return NULL;
  }

  if (ten->device != DEVICE_META && !is_allocated(ten)) {
    return NULL;
  }

  Tensor rten = {0};
  bool swapped = false;
  memcpy(&rten, ten, sizeof(Tensor));

  if (ten->device == DEVICE_GPU &&
      is_device_memory_handle(&ten->storage->handle)) {
    swapped = true;
    DeviceStatus status = move_from_device(ten, &rten);
    if (status.code != 0) {
      return NULL;
    }
  }

  ReprOptions opts = repr_default_options();
  char *repr = repr_internal(&rten, &opts);
  if (swapped) {
    if (ten->requires_grad_) {
      /* Note: Avoid releasing the original TensorGrad of source */
      rten.grad = NULL;
    }
    collect(&rten);
  }
  return repr;
}

/**
 * @brief Produce a debug-mode string representation of a tensor.
 *
 * @details
 * Performs a standard representation. If the tensor is resident on a
 * GPU, a temporary host-side shadow is created for the data scan.
 *
 * Sets the @ref RerpModeDebug flag before dispatching to the
 * internal representation engine.
 *
 * @param[in] ten Pointer to the tensor to render.
 *
 * @return Heap-allocated string (caller must free()), or NULL on failure.
 */
char *tensor_repr_debug(const Tensor *ten) {
  if (ten == NULL) {
    return NULL;
  }

  if (ten->device != DEVICE_META && !is_allocated(ten)) {
    return NULL;
  }

  Tensor rten = {0};
  bool swapped = false;
  memcpy(&rten, ten, sizeof(Tensor));

  if (ten->device == DEVICE_GPU &&
      is_device_memory_handle(&ten->storage->handle)) {
    swapped = true;
    DeviceStatus status = move_from_device(ten, &rten);
    if (status.code != 0) {
      return NULL;
    }
  }

  ReprOptions opts = repr_default_options();
  opts.mode = ReprModeDebug;
  char *repr = repr_internal(&rten, &opts);
  if (swapped) {
    if (ten->requires_grad_) {
      /* Note: Avoid releasing the original TensorGrad of source */
      rten.grad = NULL;
    }
    collect(&rten);
  }
  return repr;
}

/**
 * @brief Produce a string representation with full control via options.
 *
 * @details
 * Performs a standard representation. If the tensor is resident on a
 * GPU, a temporary host-side shadow is created for the data scan.
 *
 * @param[in]  ten  Pointer to the tensor to render.
 * @param[in]  opts Pointer to a @ref ReprOptions struct. If NULL, defaults
 *                  are used.
 *
 * @return Heap-allocated string (caller must free()), or NULL on failure.
 */
char *tensor_repr_with_options(const Tensor *ten, const ReprOptions *opts) {
  if (ten == NULL) {
    return NULL;
  }
  if (opts == NULL) {
    return tensor_repr(ten);
  }
  if (ten->device != DEVICE_META && !is_allocated(ten)) {
    return NULL;
  }

  Tensor rten = {0};
  bool swapped = false;
  memcpy(&rten, ten, sizeof(Tensor));

  if (ten->device == DEVICE_GPU &&
      is_device_memory_handle(&ten->storage->handle)) {
    swapped = true;
    DeviceStatus status = move_from_device(ten, &rten);
    if (status.code != 0) {
      return NULL;
    }
  }

  char *repr = repr_internal(&rten, opts);
  if (swapped) {
    if (ten->requires_grad_) {
      /* Note: Avoid releasing the original TensorGrad of source */
      rten.grad = NULL;
    }
    collect(&rten);
  }
  return repr;
}

/**
 * @brief Print a tensor's normal-mode representation to standard output.
 *
 * @details
 * Performs a standard representation. If the tensor is resident on a
 * GPU, a temporary host-side shadow is created for the data scan.
 *
 * @param[in] ten Pointer to the tensor to print.
 */
void tensor_print(const Tensor *ten) {
  if (ten == NULL) {
    return;
  }
  char *s = tensor_repr(ten);
  if (s != NULL) {
    printf("%s\n", s);
    free(s);
  }
}

/**
 * @brief Print a tensor's debug-mode representation to standard output.
 *
 * @details
 * Performs a standard representation. If the tensor is resident on a
 * GPU, a temporary host-side shadow is created for the data scan.
 *
 * @param[in] ten Pointer to the tensor to print.
 */
void tensor_print_debug(const Tensor *ten) {
  if (ten == NULL) {
    return;
  }
  char *s = tensor_repr_debug(ten);
  if (s == NULL) {
    printf("%s\n", s);
    free(s);
  }
}
