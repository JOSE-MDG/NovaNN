/**
 * @file tensor_repr.c
 * @brief Top-level tensor string representation API implementation.
 *
 * @details
 * Implements the public interface for converting @ref Tensor objects
 * into human-readable strings. Every public entry validates its
 * arguments into @ref novaStatus_t, then funnels through
 * @ref render_owned(), the single engine orchestrating the pipeline:
 * device-to-host shadowing, context building, layout rendering, and
 * metadata formatting.
 *
 * @section representation-pipeline Representation Pipeline
 *
 * @li 1. Validation: null arguments and unallocated tensors fail
 *    fast with a reason; nothing renders.
 * @li 2. Shadowing: GPU tensors render through a temporary host copy
 *    that is always released, including on transfer failure.
 * @li 3. Contextualization: @ref build_repr_context() scans the
 *    tensor and derives formatting parameters.
 * @li 4. Layout: one recursive renderer handles contiguous, strided,
 *    and truncated output behind a 256-byte @ref StringBuilder.
 * @li 5. Metadata & closure: footer appended, builder errors map to
 *    @c novaOutOfMemory, ownership transfers to the caller.
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
 * @brief Build a status with the library message for a code.
 */
static inline novaStatus_t repr_status(novaError_t err) {
  novaStatus_t st;
  st.err = err;
  st.message = nova_get_error_msg(err, nullptr);
  return st;
}

/**
 * @brief Internal representation engine shared by all public entry
 *        points.
 *
 * @details
 * Performs the actual orchestration of the repr pipeline: shadows
 * GPU tensors to host, builds the context, renders the layout, and
 * appends metadata. The shadow copy is released on every exit after
 * a successful allocation, including transfer failure (which the
 * triplicated predecessors handled inconsistently).
 *
 * @param[in]  ten  Pointer to the tensor to render. Allocated, or
 *                  META.
 * @param[in]  opts Pointer to the formatting options. Must not be
 *                  @c nullptr.
 * @param[out] out  Slot for the owned string, @c nullptr on error.
 *
 * @return @c novaSuccess with @c *out owned by the caller, or the
 *         failure reason (@c novaInvalidDevice for GPU tensors
 *         without device backing, propagated transfer status,
 *         @c novaOutOfMemory when the string cannot be built).
 *
 * @see build_repr_context()
 * @see metadata_fmt_append()
 */
static inline novaStatus_t render_owned(const Tensor *ten,
                                        const ReprOptions *opts, char **out) {
  *out = nullptr;
  Tensor rten = {};
  bool swapped = false;
  memcpy(&rten, ten, sizeof(Tensor));

  if (ten->device == DEVICE_GPU) {
    if (!on_device(ten)) {
      return repr_status(novaInvalidDevice);
    }
    swapped = true;
    rten.storage = nullptr;
    rten.data.data = nullptr;
    rten.is_allocated_ = false;

    // Create a storage for rten (in-place)
    novaStatus_t st = safe_allocator(ten->storage->size_bytes, DEVICE_CPU,
                                     false, nullptr, &rten, true);
    if (st.err != novaSuccess) {
      return st;
    }

    /*
     * Temporarily change the `device` member from `rten` to
     * `DEVICE_CPU` to satisfy the `transf_tensor_from_device()` check
     * and allow printing the tensor with the metadata from
     * `DEVICE_GPU`.
     */
    rten.device = DEVICE_CPU;
    st = transf_tensor_from_device(ten, &rten);
    if (st.err != novaSuccess) {
      collect(&rten);
      return st;
    }
    rten.device = DEVICE_GPU;
  }

  auto ctx = build_repr_context(&rten, opts);

  StringBuilder sb;
  sb_init(&sb, 256);
  if (sb_get_status(&sb) != SbOk) {
    if (swapped) {
      collect(&rten);
    }
    return repr_status(novaOutOfMemory);
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
        auto lane = ctx;
        lane.sub_element_index = i;
        format_element(buf, sizeof(buf), ptr, t, &lane);
        sb_append(&sb, buf);
      }
    }
  } else if (ctx.is_meta) {
    sb_append(&sb, "...");
  } else {
    layout_render(&ctx, &sb);
  }

  metadata_fmt_append(&ctx, &sb);

  if (swapped) {
    if (ten->requires_grad_) {
      /* Note: Avoid releasing the original TensorGrad of source */
      rten.grad = nullptr;
    }
    collect(&rten);
  }
  if (sb_get_status(&sb) != SbOk) {
    sb_free(&sb);
    return repr_status(novaOutOfMemory);
  }
  *out = sb_build(&sb);
  if (*out == nullptr) {
    return repr_status(novaOutOfMemory);
  }
  return repr_status(novaSuccess);
}

/**
 * @brief Produce a normal-mode string representation of a tensor.
 *
 * @param[in]  ten Pointer to the tensor to render.
 * @param[out] out Slot for the heap-allocated string (caller frees).
 *
 * @return Status per @ref tensor_repr.h.
 */
novaStatus_t tensor_repr(const Tensor *ten, char **out) {
  if (out == nullptr) {
    return repr_status(novaInvalidPointer);
  }
  *out = nullptr;
  if (ten == nullptr) {
    return repr_status(novaInvalidPointer);
  }
  if (ten->device != DEVICE_META && !is_allocated(ten)) {
    return repr_status(novaInvalidTensor);
  }
  auto opts = repr_default_options();
  return render_owned(ten, &opts, out);
}

/**
 * @brief Produce a debug-mode string representation of a tensor.
 *
 * @param[in]  ten Pointer to the tensor to render.
 * @param[out] out Slot for the heap-allocated string (caller frees).
 *
 * @return Status per @ref tensor_repr.h.
 */
novaStatus_t tensor_repr_debug(const Tensor *ten, char **out) {
  if (out == nullptr) {
    return repr_status(novaInvalidPointer);
  }
  *out = nullptr;
  if (ten == nullptr) {
    return repr_status(novaInvalidPointer);
  }
  if (ten->device != DEVICE_META && !is_allocated(ten)) {
    return repr_status(novaInvalidTensor);
  }
  auto opts = repr_default_options();
  opts.mode = ReprModeDebug;
  return render_owned(ten, &opts, out);
}

/**
 * @brief Produce a string representation with full control via
 *        options.
 *
 * @param[in]  ten  Pointer to the tensor to render.
 * @param[in]  opts Formatting options, or @c nullptr for defaults.
 * @param[out] out  Slot for the heap-allocated string (caller frees).
 *
 * @return Status per @ref tensor_repr.h.
 */
novaStatus_t tensor_repr_with_options(const Tensor *ten,
                                      const ReprOptions *opts, char **out) {
  if (out == nullptr) {
    return repr_status(novaInvalidPointer);
  }
  *out = nullptr;
  if (ten == nullptr) {
    return repr_status(novaInvalidPointer);
  }
  if (ten->device != DEVICE_META && !is_allocated(ten)) {
    return repr_status(novaInvalidTensor);
  }
  const ReprOptions eff = (opts != nullptr) ? *opts : repr_default_options();
  return render_owned(ten, &eff, out);
}

/**
 * @brief Print a tensor's normal-mode representation to standard
 *        output.
 *
 * @param[in] ten Pointer to the tensor to print.
 *
 * @return Status per @ref tensor_repr.h (@c novaRuntimeError when
 *         @c printf itself fails).
 */
novaStatus_t tensor_print(const Tensor *ten) {
  char *s = nullptr;
  auto st = tensor_repr(ten, &s);
  if (st.err != novaSuccess) {
    return st;
  }
  if (printf("%s\n", s) < 0) {
    free(s);
    return repr_status(novaRuntimeError);
  }
  free(s);
  return repr_status(novaSuccess);
}

/**
 * @brief Print a tensor's debug-mode representation to standard
 *        output.
 *
 * @param[in] ten Pointer to the tensor to print.
 *
 * @return Status per @ref tensor_repr.h.
 */
novaStatus_t tensor_print_debug(const Tensor *ten) {
  char *s = nullptr;
  auto st = tensor_repr_debug(ten, &s);
  if (st.err != novaSuccess) {
    return st;
  }
  if (printf("%s\n", s) < 0) {
    free(s);
    return repr_status(novaRuntimeError);
  }
  free(s);
  return repr_status(novaSuccess);
}
