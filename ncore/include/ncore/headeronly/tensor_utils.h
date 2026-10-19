/**
 * @file tensor_utils.h
 * @brief Header-only tensor utilities: metadata shells, shape/stride
 *        math, coordinate arithmetic, and view collapsing.
 *
 * All functions are @c static inline and operate on the public
 * @c Tensor layout from @ref tensor.h. Creation functions accept a
 * @ref novaStatus_t pointer; on failure the returned tensor is zeroed
 * and must not be used.
 */

#pragma once

#include <stdlib.h>
#include <string.h>

#include <ncore/core/alloc.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/core/storage.h>
#include <ncore/headeronly/macros.h>
#include <ncore/tensor.h>

/**
 * @typedef CollapsedView
 * @brief Reduced-dimension view of a tensor after collapsing
 *        contiguous dimensions.
 *
 * Adjacent dimensions with @c strides[d] == @c strides[d+1] *
 * @c shape[d+1] merge into one larger dimension; the total element
 * count is preserved. @ref collapse() produces it.
 */
typedef struct {
  shape_t shape;     ///< Collapsed dimension sizes (row-major order).
  strides_t strides; ///< Corresponding strides in bytes.
  size_t ndims;      ///< Number of collapsed dimensions (<= original ndims).
} CollapsedView;

/**
 * @brief Forward declaration of create_unallocated_grad_tensor().
 *
 * Needed for circular calls with @c create_unallocated_tensor().
 */
static inline TensorGrad
create_unallocated_grad_tensor(const shape_t shape, DType_ dtype,
                               Device_ device, bool pin_memory, size_t ndims,
                               novaStatus_t *status);

/**
 * @brief Forward declaration of create_unallocated_scalar_tensor().
 *
 * Needed for circular calls with
 * @c create_unallocated_scalar_grad_tensor().
 */
static inline Tensor create_unallocated_scalar_tensor(DType_ dtype,
                                                      Device_ device,
                                                      bool requires_grad,
                                                      bool pin_memory,
                                                      novaStatus_t *status);

/**
 * @brief Forward declaration of create_unallocated_scalar_grad_tensor().
 *
 * Needed for circular calls with @c create_unallocated_scalar_tensor().
 */
static inline TensorGrad
create_unallocated_scalar_grad_tensor(DType_ dtype, Device_ device,
                                      bool pin_memory, novaStatus_t *status);

/**
 * @typedef coords_t
 * @brief Multi-dimensional coordinate array type.
 *
 * Fixed-size array of @c NOVA_MAX_DIMS elements; only the first
 * @c ndims entries are meaningful.
 */
typedef size_t coords_t[NOVA_MAX_DIMS];

/**
 * @brief Compute per-dimension contiguous strides for a Tensor.
 *
 * Fills @c ten->strides[0..ndims-1] with row-major strides:
 *
 * @code
 * strides[ndims-1] = item_size
 * strides[dim]     = strides[dim+1] * shape[dim+1]   for dim = ndims-2 .. 0
 * @endcode
 *
 * @param[in]      ten       Tensor whose @c strides[] will be
 *                           written.
 * @param[in]      ndims     Number of dimensions.
 * @param[in]      shape     Dimension sizes.
 * @param[in]      item_size Element size in bytes (from
 *                           @c dtype_size()).
 */
static inline void compute_tensor_strides_(Tensor *ten, size_t ndims,
                                           const shape_t shape,
                                           size_t item_size) {
  ten->strides[ndims - 1] = item_size;
  for (size_t dim = ndims - 1; dim-- > 0;) {
    ten->strides[dim] = ten->strides[dim + 1] * shape[dim + 1];
  }
}

/**
 * @brief Compute the total element count for a Tensor.
 *
 * Stores the product of the sizes in @c ten->size (1 when
 * @c ndims == 0) and multiplies by the packing factor for
 * @c logical_size.
 *
 * @param[in,out] ten    Tensor whose @c size will be set.
 * @param[in]     shape  Dimension sizes (first @c ten->ndims
 *                       entries).
 */
static inline void compute_tensor_size_(Tensor *ten, const shape_t shape) {
  size_t size = 1;
  for (size_t dim = 0; dim < ten->ndims; ++dim) {
    size *= shape[dim];
  }
  ten->size = size;
  ten->logical_size = size * dtype_packing_factor(ten->dtype);
}

/**
 * @brief Convert a logical shape to storage units for packed dtypes.
 *
 * Packed dtypes (e.g. @c Float4E2M1fn) store @c packing logical
 * elements in one storage unit along the innermost dimension. The
 * physical layout (shape, strides, size, theoretical byte count) is
 * expressed in storage units, so the logical last dimension is
 * divided by the packing factor. Non-packed dtypes copy the shape
 * verbatim.
 *
 * @param[out] storage  Receives the storage-unit shape.
 * @param[in]  shape    Logical dimension sizes (first @c ndims
 *                      entries).
 * @param[in]  dtype    Element data type (@ref DType_).
 * @param[in]  ndims    Number of dimensions.
 * @param[out] status   Receives @ref novaInvalidShape when the last
 *                      dimension is not divisible by the packing
 *                      factor.
 *
 * On error @p storage still holds a verbatim copy of @p shape
 * (the copy runs before the divisibility check), so check @p status
 * before using it.
 */
static inline void storage_shape_(shape_t storage, const shape_t shape,
                                  DType_ dtype, size_t ndims,
                                  novaStatus_t *status) {
  if (ndims == 0) {
    return;
  }
  const size_t packing = dtype_packing_factor(dtype);
  memcpy(storage, shape, ndims * sizeof(size_t));
  if (packing > 1) {
    if (storage[ndims - 1] % packing != 0) {
      status->err = novaInvalidShape;
      status->message =
          "The last dimension must be a multiple of the dtype packing "
          "factor\n";
      return;
    }
    storage[ndims - 1] /= packing;
  }
}

/**
 * @brief Compute per-dimension contiguous strides for a TensorGrad.
 *
 * Same logic as @c compute_tensor_strides_() for a pointer tensor.
 */
static inline void compute_grad_tensor_strides_(TensorGrad grad, size_t ndims,
                                                const shape_t shape,
                                                size_t item_size) {
  grad->strides[ndims - 1] = item_size;
  for (size_t dim = ndims - 1; dim-- > 0;) {
    grad->strides[dim] = grad->strides[dim + 1] * shape[dim + 1];
  }
}

/**
 * @brief Compute the total element count for a TensorGrad.
 *
 * Same logic as @c compute_tensor_size_() for a pointer tensor.
 */
static inline void compute_grad_tensor_size_(TensorGrad grad,
                                             const shape_t shape) {
  size_t size = 1;
  for (size_t dim = 0; dim < grad->ndims; ++dim) {
    size *= shape[dim];
  }
  grad->size = size;
  grad->logical_size = size * dtype_packing_factor(grad->dtype);
}

/**
 * @brief Compute the linear byte offset from multi-dimensional
 *        coordinates.
 *
 * Applies the standard strided formula:
 *
 * @code
 * offset = Σ (coords[dim] × strides[dim])   for dim = 0 .. ndims-1
 * @endcode
 *
 * @param[in] coords   Multi-dimensional coordinates.  Only the
 *                     first @c ndims entries are read.
 * @param[in] ndims    Number of dimensions.
 * @param[in] strides  Per-dimension stride values (in bytes).
 *
 * @return Byte offset from the start of the data buffer. Bounds are
 *         not validated.
 */
static inline size_t compute_linear_byte_offset(const coords_t coords,
                                                size_t ndims,
                                                const strides_t strides) {
  size_t offset = 0;
  for (size_t dim = 0; dim < ndims; ++dim) {
    offset += coords[dim] * strides[dim];
  }
  return offset;
}

/**
 * @brief Reconstruct multi-dimensional coordinates from a linear
 *        byte offset.
 *
 * Inverse of @c compute_linear_byte_offset(): divides the offset by
 * each dimension's stride in turn. For a valid position the
 * remainder ends at 0.
 *
 * @param[in]  offset  Linear byte offset into the data buffer.
 * @param[in]  ndims   Number of dimensions.
 * @param[out] coords  Output array (written in-place).  Only the
 *                     first @c ndims entries are written.
 * @param[in]  strides Per-dimension stride values (in bytes).
 */
static inline void compute_coords_given_linear_byte_offset_(
    size_t offset, size_t ndims, coords_t coords, const strides_t strides) {
  for (size_t dim = 0; dim < ndims; ++dim) {
    coords[dim] = (offset / strides[dim]);
    offset %= strides[dim];
  }
}

/**
 * @brief Reconstruct multi-dimensional coordinates from a linear
 *        element index.
 *
 * Walks element counts, taking the remainder against each dimension
 * size from the innermost dimension outward. This variant works
 * where @c compute_coords_given_linear_byte_offset_() cannot: strided
 * or collapsed views whose byte stride is not a multiple of the
 * element size. Each thread computes its own @c item, so parallel
 * loops need no shared state (see @c contiguous_cpu_impl()).
 *
 * @param[in]  item   Linear element index.
 * @param[in]  ndims  Number of dimensions.
 * @param[in]  shape  Dimension sizes.  Only the first @c ndims
 *                     entries are read.
 * @param[out] coords Output array (written in-place).
 */
NCORE_HOST_DEVICE static inline void
compute_coords_from_linear_index_(size_t item, size_t ndims,
                                  const size_t *shape, coords_t coords) {
  for (size_t dim = ndims; dim-- > 0;) {
    coords[dim] = item % shape[dim];
    item /= shape[dim];
  }
}

/**
 * @brief Create a Tensor without allocating a data buffer.
 *
 * Metadata-only shell: @c storage and @c data.data stay @c nullptr.
 * When @p requires_grad is @c true, an unallocated gradient tensor is
 * created via @ref create_unallocated_grad_tensor(); a failure there
 * zeroes the tensor and reports through @p status.
 *
 * @param[in]  shape         Dimension sizes.
 * @param[in]  dtype         Element data type (@c DType_).
 * @param[in]  device        Target device (@c DEVICE_CPU,
 *                           @c DEVICE_GPU, or @c DEVICE_META).
 * @param[in]  requires_grad Whether to track gradients.
 * @param[in]  pin_memory    If @c true, request page-locked host
 *                           memory (CPU only).
 * @param[in]  ndims         Number of dimensions.
 * @param[out] status        Receives the operation result.
 *
 * @pre  @c ndims must not exceed @c NOVA_MAX_DIMS.
 * @post @c shape, @c strides, @c size, and @c item_size are valid;
 *       the tensor is not allocated.
 */
static inline Tensor create_unallocated_tensor(const shape_t shape,
                                               DType_ dtype, Device_ device,
                                               bool requires_grad,
                                               bool pin_memory, size_t ndims,
                                               novaStatus_t *status) {
  Tensor tensor = {};
  shape_t storage_shape = {};
  storage_shape_(storage_shape, shape, dtype, ndims, status);
  if (status->err != novaSuccess) {
    return tensor;
  }
  memcpy(tensor.shape, storage_shape, ndims * sizeof(size_t));
  tensor.dtype = dtype;
  tensor.device = device;
  tensor.ndims = ndims;
  tensor.item_size = dtype_size(dtype);
  tensor.offset = 0;
  tensor.is_leaf_ = true;
  tensor.is_view_ = false;
  tensor.requires_grad_ = requires_grad;
  tensor.retain_grad_ = false;
  tensor.grad_fn_ = nullptr;
  tensor.scale_ = 1.0f;
  tensor.zero_point_ = 0;
  tensor.storage = nullptr;
  tensor.data.data = nullptr;
  tensor.is_allocated_ = false;
  tensor.version_ = 0;
  tensor.is_pinned_ = pin_memory;
  compute_tensor_size_(&tensor, storage_shape);
  compute_tensor_strides_(&tensor, ndims, storage_shape, tensor.item_size);

  if (requires_grad) {
    tensor.grad = create_unallocated_grad_tensor(shape, dtype, device,
                                                 pin_memory, ndims, status);

    if (status->err != novaSuccess) {
      memset(&tensor, 0, sizeof(Tensor));
      return tensor;
    }
  }

  return tensor;
}

/**
 * @brief Create a heap-allocated unallocated TensorGrad.
 *
 * The gradient is a metadata-only shell. @p status gets
 * @ref novaInvalidPointer and the return is @c nullptr when the
 * heap allocation fails. The caller owns the returned pointer and
 * must free it (or let @ref collect() on the parent handle it).
 *
 * @param[in]  shape      Dimension sizes.
 * @param[in]  dtype      Element data type (@c DType_).
 * @param[in]  device     Target device.
 * @param[in]  pin_memory If @c true, request page-locked host
 *                        memory (CPU only).
 * @param[in]  ndims      Number of dimensions.
 * @param[out] status     Receives the operation result.
 *
 * @return Newly allocated @c Tensor (as @c TensorGrad), or
 *         @c nullptr on allocation failure.
 */
static inline TensorGrad
create_unallocated_grad_tensor(const shape_t shape, DType_ dtype,
                               Device_ device, bool pin_memory, size_t ndims,
                               novaStatus_t *status) {
#if defined(_WIN64)
  TensorGrad grad = STATIC_CAST(TensorGrad, malloc(sizeof(Tensor)));
#else
  void *raw = nullptr;
  if (posix_memalign(&raw, alignof(Tensor), sizeof(Tensor)) != 0) {
    raw = nullptr;
  }
  TensorGrad grad = STATIC_CAST(TensorGrad, raw);
#endif
  if (grad == nullptr) {
    status->err = novaInvalidPointer;
    status->message =
        "Failed to allocate gradient tensor: malloc returned nullptr\n";
    return nullptr;
  }
  *grad = create_unallocated_tensor(shape, dtype, device, false, pin_memory,
                                    ndims, status);
  return grad;
}

/**
 * @brief Create a scalar (0-D) Tensor without allocating a data
 *        buffer.
 *
 * Sets the scalar invariant: @c shape[0] = 0, @c strides[0] = 0,
 * @c size = 1, @c ndims = 0. When @p requires_grad is @c true, an
 * unallocated scalar gradient tensor is created; a failure there
 * zeroes the tensor and reports through @p status.
 *
 * @param[in]  dtype         Element data type (@c DType_).
 * @param[in]  device        Target device.
 * @param[in]  requires_grad Whether to track gradients.
 * @param[in]  pin_memory    If @c true, request page-locked host
 *                           memory (CPU only).
 * @param[out] status        Receives the operation result.
 *
 * @return Initialized scalar @c Tensor with no backing storage.
 */
static inline Tensor create_unallocated_scalar_tensor(DType_ dtype,
                                                      Device_ device,
                                                      bool requires_grad,
                                                      bool pin_memory,
                                                      novaStatus_t *status) {
  Tensor tensor = {};
  tensor.shape[0] = 0;
  tensor.strides[0] = 0;
  tensor.size = 1;
  tensor.logical_size = dtype_packing_factor(dtype);
  tensor.ndims = 0;
  tensor.dtype = dtype;
  tensor.device = device;
  tensor.item_size = dtype_size(dtype);
  tensor.offset = 0;
  tensor.is_leaf_ = true;
  tensor.is_view_ = false;
  tensor.requires_grad_ = requires_grad;
  tensor.retain_grad_ = false;
  tensor.grad_fn_ = nullptr;
  tensor.scale_ = 1.0f;
  tensor.zero_point_ = 0;
  tensor.storage = nullptr;
  tensor.data.data = nullptr;
  tensor.is_allocated_ = false;
  tensor.version_ = 0;
  tensor.is_pinned_ = pin_memory;

  if (requires_grad) {
    tensor.grad = create_unallocated_scalar_grad_tensor(dtype, device,
                                                        pin_memory, status);

    if (status->err != novaSuccess) {
      memset(&tensor, 0, sizeof(Tensor));
      return tensor;
    }
  }

  return tensor;
}

/**
 * @brief Create a heap-allocated unallocated scalar TensorGrad.
 *
 * Metadata-only shell. @p status gets @ref novaInvalidPointer and the
 * return is @c nullptr when the heap allocation fails.
 *
 * @param[in]  dtype      Element data type (@c DType_).
 * @param[in]  device     Target device.
 * @param[in]  pin_memory If @c true, request page-locked host
 *                        memory (CPU only).
 * @param[out] status     Receives the operation result.
 *
 * @return Newly allocated scalar @c Tensor (as @c TensorGrad), or
 *         @c nullptr on allocation failure. Heap memory: the caller
 *         owns it.
 */
static inline TensorGrad
create_unallocated_scalar_grad_tensor(DType_ dtype, Device_ device,
                                      bool pin_memory, novaStatus_t *status) {
#if defined(_WIN64)
  TensorGrad grad = STATIC_CAST(TensorGrad, malloc(sizeof(Tensor)));
#else
  void *raw = nullptr;
  if (posix_memalign(&raw, alignof(Tensor), sizeof(Tensor)) != 0) {
    raw = nullptr;
  }
  TensorGrad grad = STATIC_CAST(TensorGrad, raw);
#endif
  if (grad == nullptr) {
    status->err = novaInvalidPointer;
    status->message =
        "Failed to allocate gradient tensor: malloc returned nullptr\n";
    return nullptr;
  }
  *grad = create_unallocated_scalar_tensor(dtype, device, false, pin_memory,
                                           status);
  return grad;
}

/**
 * @brief Advance a multi-dimensional coordinate by one step in
 *        row-major order, odometer style.
 *
 * Increments the last dimension and propagates the carry to earlier
 * dimensions; overflows wrap back to all zeros.
 *
 * @code{.c}
 * coords_t coord = {};
 * for (size_t i = 0; i < tensor.size; i++) {
 *     // process element at coord
 *     odometer(coord, tensor.ndims, tensor.shape);
 * }
 * @endcode
 *
 * @param[in,out] coords  Current coordinates, updated in-place.
 * @param[in]     ndims   Number of dimensions.
 * @param[in]     shape   Dimension sizes (upper bounds).
 */
static inline void odometer(coords_t coords, size_t ndims,
                            const shape_t shape) {
  for (size_t dim = ndims; dim-- > 0;) {
    coords[dim]++;
    if (coords[dim] < shape[dim]) {
      break;
    }
    coords[dim] = 0;
  }
}

/**
 * @brief Collapse contiguous dimensions of a tensor into a
 *        reduced-dimension view.
 *
 * Walks the dimensions from the innermost outward and merges
 * adjacent ones that are contiguous in memory, i.e.
 *
 * @code
 * strides[d] == strides[d+1] * shape[d+1]
 * @endcode
 *
 * (the row-major adjacency condition). Merging needs no copy, so
 * consumers such as @c contiguous_cpu_impl() iterate the same
 * element sequence with fewer odometer steps.
 *
 * For a scalar tensor (@c ndims == 0) the view comes back empty.
 *
 * @param[in] ten  Input tensor. Must not be @c nullptr.  Read-only:
 *                 shape, strides, and ndims are not modified.
 *
 * @return A @c CollapsedView describing the collapsed layout, with
 *         @c ndims <= @c ten->ndims and
 *         product(shape) == @c ten->size.
 */
static inline CollapsedView collapse(const Tensor *restrict ten) {
  CollapsedView cv = {};
  if (is_scalar(ten)) {
    return cv;
  }

  int ndims_in = STATIC_CAST(int, ten->ndims);
  shape_t tmp_shape;
  strides_t tmp_strides;

  int out_idx = 0;
  tmp_shape[0] = ten->shape[ndims_in - 1];
  tmp_strides[0] = ten->strides[ndims_in - 1];

  for (int dim = ndims_in - 2; dim >= 0; dim--) {
    if (ten->strides[dim] == tmp_strides[out_idx] * tmp_shape[out_idx]) {
      // If these dimensions are contiguous merge them
      tmp_shape[out_idx] *= ten->shape[dim];
    } else {
      out_idx++;
      tmp_shape[out_idx] = ten->shape[dim];
      tmp_strides[out_idx] = ten->strides[dim];
    }
  }

  cv.ndims = STATIC_CAST(size_t, out_idx) + 1;
  for (size_t i = 0; i < cv.ndims; ++i) {
    cv.shape[i] = tmp_shape[cv.ndims - 1 - i];
    cv.strides[i] = tmp_strides[cv.ndims - 1 - i];
  }

  return cv;
}
