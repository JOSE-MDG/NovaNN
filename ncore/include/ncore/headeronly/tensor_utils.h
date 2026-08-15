/**
 * @file tensor_utils.h
 * @brief Header-only tensor utilities: creation, metadata
 *        computation, coordinate manipulation, and view collapsing.
 *
 * @details
 * This module provides a collection of @c static inline utilities
 * for low-level tensor manipulation.  It is designed for
 * zero-overhead inclusion in translation units that need tensor
 * metadata operations without pulling in heavy dependencies.
 *
 * The utilities fall into several categories:
 *
 * @li Tensor creation (unallocated): Functions to initialise
 *   @c Tensor and @c TensorGrad metadata shells — including scalar
 *   (0-D) variants — without allocating backing storage.  These
 *   are used for deferred allocation and gradient tracking.
 *   All creation functions accept a @ref novaStatus_t pointer
 *   for error propagation; on failure the returned tensor is
 *   zeroed and must not be used.
 *
 * @li Metadata computation: Row-major stride and size
 *   computation for both @c Tensor and @c TensorGrad types.
 *
 * @li Coordinate arithmetic: Conversion between multi-
 *   dimensional coordinates and linear byte offsets, plus an
 *   odometer-style iterator for visiting every element in
 *   row-major order.
 *
 * @li Dimension collapsing: The @ref CollapsedView type and
 *   @ref collapse() function merge contiguous dimensions,
 *   reducing odometer overhead in memory-bound kernels.
 *
 * All functions operate on the public @c Tensor layout defined in
 * @ref tensor.h and use the @c DType_ enumeration from @ref dtype.h.
 * The maximum supported rank is @ref NOVA_MAX_DIMS from
 * @ref macros.h.
 *
 * @see tensor.h    Tensor struct, public API, and layout
 *                  invariants.
 * @see dtype.h     DType_ enum and dtype_size().
 * @see macros.h    NOVA_MAX_DIMS, ALIGN(), and other compile-time
 *                  limits.
 * @see alloc.h     Storage allocation (used by allocated variants
 *                  in tensor.c).
 * @see status.h    novaStatus_t error reporting.
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
 * @details
 * When a tensor has adjacent dimensions that are contiguous in
 * memory (i.e., @c strides[d] == @c strides[d+1] * @c shape[d+1]), they
 * can be merged into a single larger dimension without copying
 * data.  @c CollapsedView captures the result of this merging
 * operation performed by @ref collapse().
 *
 * The collapsed view preserves the total element count:
 * @c product(shape[0..ndims-1]) == @c original_tensor->size.
 *
 * This view is used by operations like @c contiguous_cpu_impl() to
 * iterate over the tensor with fewer odometer steps, reducing
 * loop overhead.
 *
 * @see collapse()               Produces this view.
 * @see contiguous_cpu_impl()    Consumer of this view.
 * @see odometer()               Iterates using collapsed shape.
 */
typedef struct {
  shape_t shape;     ///< Collapsed dimension sizes (row-major order).
  strides_t strides; ///< Corresponding strides in bytes.
  size_t ndims;      ///< Number of collapsed dimensions (<= original ndims).
} CollapsedView;

/**
 * @brief Forward declaration of create_unallocated_grad_tensor().
 *
 * @details
 * Declared here to allow circular calls between
 * @c create_unallocated_tensor() and @c create_unallocated_grad_tensor().
 * The full implementation appears later in this file.
 *
 * @see create_unallocated_grad_tensor()  Full implementation.
 * @see create_unallocated_tensor()       Calls this function.
 */
static inline TensorGrad
create_unallocated_grad_tensor(const shape_t shape, DType_ dtype,
                               Device_ device, bool pin_memory, size_t ndims,
                               novaStatus_t *status);

/**
 * @brief Forward declaration of create_unallocated_scalar_tensor().
 *
 * @details
 * Declared here to allow circular calls between
 * @c create_unallocated_scalar_grad_tensor() and
 * @c create_unallocated_scalar_tensor().  The full implementation
 * appears later in this file.
 *
 * @see create_unallocated_scalar_tensor()       Full implementation.
 * @see create_unallocated_scalar_grad_tensor()  Calls this function.
 */
static inline Tensor create_unallocated_scalar_tensor(DType_ dtype,
                                                      Device_ device,
                                                      bool requires_grad,
                                                      bool pin_memory,
                                                      novaStatus_t *status);

/**
 * @brief Forward declaration of create_unallocated_scalar_grad_tensor().
 *
 * @details
 * Declared here to allow circular calls between
 * @c create_unallocated_scalar_tensor() and
 * @c create_unallocated_scalar_grad_tensor().  The full
 * implementation appears later in this file.
 *
 * @see create_unallocated_scalar_grad_tensor()  Full implementation.
 * @see create_unallocated_scalar_tensor()       Calls this function.
 */
static inline TensorGrad
create_unallocated_scalar_grad_tensor(DType_ dtype, Device_ device,
                                      bool pin_memory, novaStatus_t *status);

/**
 * @typedef coords_t
 * @brief Multi-dimensional coordinate array type.
 *
 * @details
 * Fixed-size array of @c NOVA_MAX_DIMS elements representing a
 * position within an n-dimensional tensor.  Only the first @c ndims
 * entries are meaningful.
 *
 * Used by @c compute_linear_byte_offset(),
 * @c compute_coords_given_linear_byte_offset_(), and @c odometer().
 */
typedef size_t coords_t[NOVA_MAX_DIMS];

/**
 * @brief Compute per-dimension contiguous strides for a Tensor.
 *
 * @details
 * Fills @c ten->strides[0..ndims-1] with row-major strides:
 *
 * @code
 * strides[ndims-1] = item_size
 * strides[dim]     = strides[dim+1] * shape[dim+1]   for dim = ndims-2 .. 0
 * @endcode
 *
 * This produces the standard C row-major layout where the last
 * dimension varies fastest.
 *
 * @param[in]      ten       Tensor whose @c strides[] will be
 *                           written.
 * @param[in]      ndims     Number of dimensions.
 * @param[in]      shape     Dimension sizes.
 * @param[in]      item_size Element size in bytes (from
 *                           @c dtype_size()).
 *
 * @pre  @c ten must not be @c nullptr.
 * @pre  @c ndims must be > 0.
 * @post @c ten->strides[0..ndims-1] contain valid row-major strides.
 *
 * @see compute_tensor_size_()          Computes @c ten->size.
 * @see compute_grad_tensor_strides_()  Gradient variant.
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
 * @details
 * Multiplies all @c ndims dimension sizes together and stores the
 * result in @c ten->size.  For a scalar (@c ndims == 0), @c ten->size
 * is set to 1.
 *
 * @param[in,out] ten    Tensor whose @c size will be set.
 * @param[in]     shape  Dimension sizes (first @c ten->ndims
 *                       entries).
 *
 * @pre  @c ten must not be @c nullptr.
 * @post @c ten->size == @c product(shape[0..ten->ndims-1]), or 1 if
 *       @c ten->ndims == 0.
 *
 * @see compute_tensor_strides_()          Computes @c ten->strides.
 * @see compute_grad_tensor_size_()        Gradient variant.
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
 * @details
 * Packed dtypes (e.g. @c Float4E2M1fn) store @c packing logical
 * elements in a single storage unit along the innermost dimension.
 * The physical layout — @c shape, @c strides, @c size, and the
 * allocated byte count — is expressed in storage units, so the
 * logical last dimension is divided by the dtype's packing factor.
 *
 * The output shape is the caller-provided shape verbatim when the
 * dtype is not packed (@c packing == 1).
 *
 * @param[out] storage  Receives the storage-unit shape.
 * @param[in]  shape    Logical dimension sizes (first @c ndims
 *                      entries).
 * @param[in]  dtype    Element data type (@ref DType_).
 * @param[in]  ndims    Number of dimensions.
 * @param[out] status   Receives the operation result.  Set to
 *                      @ref novaInvalidShape when the last dimension
 *                      is not divisible by the packing factor.
 *
 * @pre  @p storage and @p status must not be @c nullptr.
 * @post On error @p storage is modified: it holds a verbatim copy of
 *       the logical @p shape, because the packing-divisibility check
 *       runs after the copy.  The caller must check @p status before
 *       using it.
 *
 * @see dtype_packing_factor()  Packing factor lookup.
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
 * @details
 * Identical logic to @c compute_tensor_strides_() but operates on
 * a @c TensorGrad (pointer-to-@c Tensor).
 *
 * @param[in,out] grad      TensorGrad whose @c strides[] will be
 *                          written.
 * @param[in]     ndims     Number of dimensions.
 * @param[in]     shape     Dimension sizes.
 * @param[in]     item_size Element size in bytes.
 *
 * @pre  @c grad must not be @c nullptr.
 * @pre  @c ndims must be > 0.
 * @post @c grad->strides[0..ndims-1] contain valid row-major strides.
 *
 * @see compute_tensor_strides_()  Tensor variant.
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
 * @details
 * Identical logic to @c compute_tensor_size_() but operates on a
 * @c TensorGrad.
 *
 * @param[in,out] grad   TensorGrad whose @c size will be set.
 * @param[in]     shape  Dimension sizes (first @c grad->ndims
 *                       entries).
 *
 * @pre  @c grad must not be @c nullptr.
 * @post @c grad->size == @c product(shape[0..grad->ndims-1]), or 1 if
 *       @c grad->ndims == 0.
 *
 * @see compute_tensor_size_()  Tensor variant.
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
 * @details
 * Applies the standard strided offset formula:
 *
 * @code
 * offset = Σ (coords[dim] × strides[dim])   for dim = 0 .. ndims-1
 * @endcode
 *
 * This is the byte distance from the start of the data buffer to
 * the element at the given coordinates.
 *
 * @param[in] coords   Multi-dimensional coordinates.  Only the
 *                     first @c ndims entries are read.
 * @param[in] ndims    Number of dimensions.
 * @param[in] strides  Per-dimension stride values (in bytes).
 *
 * @return Byte offset from the start of the data buffer.
 *
 * @pre  @c ndims must be > 0.
 * @pre  Each @c coords[dim] must be < @c shape[dim] (not validated).
 *
 * @see compute_coords_given_linear_byte_offset_()  Inverse
 *      operation.
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
 * @details
 * Performs the inverse of @c compute_linear_byte_offset() by
 * repeatedly dividing the offset by each dimension's stride:
 *
 * @code
 * for dim = 0 .. ndims-1:
 *     coords[dim] = offset / strides[dim]
 *     offset      = offset % strides[dim]
 * @endcode
 *
 * After the loop, @c offset should be 0 for a valid position.
 *
 * @param[in]  offset  Linear byte offset into the data buffer.
 * @param[in]  ndims   Number of dimensions.
 * @param[out] coords  Output array (written in-place).  Only the
 *                     first @c ndims entries are written.
 * @param[in]  strides Per-dimension stride values (in bytes).
 *
 * @pre  @c ndims must be > 0.
 * @pre  @c offset must be < total buffer size (not validated).
 * @post @c coords[0..ndims-1] contain the reconstructed position.
 *
 * @see compute_linear_byte_offset()  Forward operation.
 */
static inline void compute_coords_given_linear_byte_offset_(
    size_t offset, size_t ndims, coords_t coords, const strides_t strides) {
  for (size_t dim = 0; dim < ndims; ++dim) {
    coords[dim] = (offset / strides[dim]);
    offset %= strides[dim];
  }
}

/**
 * @brief Create a Tensor without allocating a data buffer.
 *
 * @details
 * Zero-initialises a @c Tensor, copies shape metadata, computes
 * size and strides, but leaves @c storage = nullptr,
 * @c data.data = nullptr, and @c is_allocated_ = false.  The tensor
 * is a valid metadata-only shell ready for deferred allocation.
 *
 * When @p requires_grad is @c true, an unallocated gradient tensor
 * is created via @ref create_unallocated_grad_tensor().  On
 * failure the tensor is zeroed and the error is reported through
 * @p status.
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
 * @return Initialised @c Tensor with no backing storage.
 *
 * @pre  @c ndims must not exceed @c NOVA_MAX_DIMS.
 * @pre  @p status must not be @c nullptr.
 * @post @c is_allocated_ == @c false, @c storage == @c nullptr,
 *       @c data.data == @c nullptr.
 * @post @c shape, @c strides, @c size, and @c item_size are valid.
 *
 * @see create_tensor()                    Allocated variant.
 * @see create_unallocated_scalar_tensor() Scalar variant.
 * @see create_unallocated_grad_tensor()   Heap-allocated gradient.
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
 * @details
 * Allocates a @c Tensor on the heap via @c malloc(), then
 * initialises it as an unallocated tensor via
 * @c create_unallocated_tensor().  The gradient is a metadata-only
 * shell — no data buffer is allocated.
 *
 * If @c malloc() fails, @p status is set to @ref novaInvalidPointer
 * and @c nullptr is returned.  The caller takes ownership of the
 * returned pointer and must free it (or let @ref collect() on the
 * parent handle it).
 *
 * @param[in]  shape      Dimension sizes.
 * @param[in]  dtype      Element data type (@c DType_).
 * @param[in]  device     Target device.
 * @param[in]  pin_memory If @c true, request page-locked host
 *                        memory (CPU only).
 * @param[in]  ndims      Number of dimensions.
 * @param[out] status     Receives the operation result.
 *
 * @return Pointer to newly allocated @c Tensor (as @c TensorGrad),
 *         or @c nullptr on allocation failure.
 *
 * @pre  @p status must not be @c nullptr.
 * @post The returned pointer is heap-allocated and must be freed.
 * @post @c grad->is_allocated_ == @c false.
 *
 * @see create_unallocated_tensor()              Non-heap variant.
 * @see create_unallocated_scalar_grad_tensor()  Scalar variant.
 */
static inline TensorGrad
create_unallocated_grad_tensor(const shape_t shape, DType_ dtype,
                               Device_ device, bool pin_memory, size_t ndims,
                               novaStatus_t *status) {
  TensorGrad grad = (TensorGrad)malloc(sizeof(Tensor));
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
 * @details
 * Zero-initialises a @c Tensor and sets the scalar invariant:
 * @c shape[0] = 0, @c strides[0] = 0, @c size = 1, @c ndims = 0.
 * No data buffer is allocated.
 *
 * When @p requires_grad is @c true, an unallocated scalar gradient
 * tensor is created via @ref create_unallocated_scalar_grad_tensor().
 * On failure the tensor is zeroed and the error is reported through
 * @p status.
 *
 * @param[in]  dtype         Element data type (@c DType_).
 * @param[in]  device        Target device.
 * @param[in]  requires_grad Whether to track gradients.
 * @param[in]  pin_memory    If @c true, request page-locked host
 *                           memory (CPU only).
 * @param[out] status        Receives the operation result.
 *
 * @return Initialised scalar @c Tensor with no backing storage.
 *
 * @pre  @p status must not be @c nullptr.
 * @post @c is_allocated_ == @c false, @c storage == @c nullptr,
 *       @c data.data == @c nullptr.
 * @post @c ndims == 0, @c size == 1, @c shape[0] == 0,
 *       @c strides[0] == 0.
 *
 * @see create_unallocated_tensor()           N-dimensional variant.
 * @see create_unallocated_scalar_grad_tensor() Scalar gradient.
 * @see create_scalar_tensor()                Allocated variant.
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
 * @details
 * Allocates a @c Tensor on the heap via @c malloc(), then
 * initialises it as an unallocated scalar tensor via
 * @c create_unallocated_scalar_tensor().
 *
 * If @c malloc() fails, @p status is set to @ref novaInvalidPointer
 * and @c nullptr is returned.
 *
 * @param[in]  dtype      Element data type (@c DType_).
 * @param[in]  device     Target device.
 * @param[in]  pin_memory If @c true, request page-locked host
 *                        memory (CPU only).
 * @param[out] status     Receives the operation result.
 *
 * @return Pointer to newly allocated scalar @c Tensor (as
 *         @c TensorGrad), or @c nullptr on allocation failure.
 *
 * @pre  @p status must not be @c nullptr.
 * @post The returned pointer is heap-allocated and must be freed.
 * @post @c grad->is_allocated_ == @c false, @c grad->ndims == @c 0.
 *
 * @see create_unallocated_scalar_tensor()  Non-heap variant.
 * @see create_unallocated_grad_tensor()    N-dimensional variant.
 */
static inline TensorGrad
create_unallocated_scalar_grad_tensor(DType_ dtype, Device_ device,
                                      bool pin_memory, novaStatus_t *status) {
  TensorGrad grad = (TensorGrad)malloc(sizeof(Tensor));
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
 * @brief Advance a multi-dimensional coordinate by one step
 *        (row-major).
 *
 * @details
 * Increments the last dimension (@c ndims-1) by one and propagates
 * the carry to earlier dimensions when a dimension reaches its
 * upper bound (@c shape[dim]), resetting it to zero.
 *
 * When all dimensions overflow, the coordinate wraps back to all
 * zeros — matching the behaviour of a traditional mechanical
 * odometer.
 *
 * Typical usage for iterating over all elements:
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
 * @param[in]     shape   Dimension sizes (upper bound for each
 *                        coordinate).
 *
 * @pre  @c ndims must be > 0.
 * @post @c coords is advanced to the next position, or wrapped to
 *       @c {0, 0, ..., 0} if at the end.
 *
 * @see compute_linear_byte_offset()  Convert coords to byte
 *      offset.
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
 * @details
 * Iterates over the tensor's dimensions from the innermost
 * (last) to the outermost (first) and merges adjacent dimensions
 * that are contiguous in memory. Two dimensions @c d and
 * @c d+1 are contiguous when:
 *
 * @code
 * strides[d] == strides[d+1] * shape[d+1]
 * @endcode
 *
 * This is exactly the condition for row-major (C-style) memory
 * layout where dimension @c d+1 varies fastest. When the
 * condition holds, the two dimensions can be treated as a
 * single larger dimension without copying data.
 *
 * The algorithm:
 * @li 1. Start with the innermost dimension as the first output
 *    dimension.
 * @li 2. Walk outward: if the current dimension is contiguous with
 *    the accumulated output dimension, multiply the output
 *    shape by the current shape.
 * @li 3. Otherwise, start a new output dimension.
 * @li 4. Reverse the collected dimensions so the result is in
 *    standard order (outermost first).
 *
 * For a scalar tensor (@c ndims == 0), the function returns an
 * empty @c CollapsedView with @c ndims == 0.
 *
 * The returned @c CollapsedView contains the collapsed @c shape_t
 * and @c strides_t arrays, and the new dimension count @c ndims.
 * This view can be used to iterate over the tensor with fewer
 * odometer steps, reducing loop overhead in operations such as
 * @c contiguous_cpu_impl().
 *
 * @param[in] ten  Input tensor. Must not be @c nullptr.
 *                 The tensor's @c shape_t, @c strides_t, and
 *                 @c ndims are read but not modified.
 *
 * @return A @c CollapsedView describing the collapsed layout.
 *         If the input is a scalar, @c cv.ndims == 0 and the
 *         @c shape_t/@c strides_t arrays are zeroed.
 *
 * @pre  @c ten must not be @c nullptr.
 * @pre  @c ten->ndims must not exceed @c NOVA_MAX_DIMS.
 * @post @c cv.ndims <= @c ten->ndims.
 * @post @c cv.shape and @c cv.strides describe a valid
 *       row-major layout for the same data buffer.
 * @post The product of @c cv.shape[0..cv.ndims-1] equals
 *       @c ten->size (total element count is preserved).
 *
 * @see CollapsedView           Returned view structure.
 * @see is_scalar()             Scalar check used internally.
 * @see contiguous_cpu_impl()   Consumer of this function.
 * @see odometer()              Iteration helper that benefits
 *                              from collapsed views.
 */
static inline CollapsedView collapse(const Tensor *restrict ten) {
  CollapsedView cv = {};
  if (is_scalar(ten)) {
    return cv;
  }

  int ndims_in = (int)ten->ndims;
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

  cv.ndims = (size_t)out_idx + 1;
  for (size_t i = 0; i < cv.ndims; ++i) {
    cv.shape[i] = tmp_shape[cv.ndims - 1 - i];
    cv.strides[i] = tmp_strides[cv.ndims - 1 - i];
  }

  return cv;
}
