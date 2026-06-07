/**
 * @file tensor_utils.h
 * @brief Inline tensor initialisation and coordinate utilities.
 *
 * @details
 * Header-only file providing low-level helpers used by `tensor.c`
 * and other internal modules.  All functions are `static inline`
 * for zero-overhead inclusion.
 *
 * ## Provided Utilities
 */
// clang-format off
/**
 * | Function                                        | Purpose                                      |
 * |-------------------------------------------------|----------------------------------------------|
 * | `compute_tensor_strides_()`                     | Row-major strides for a Tensor               |
 * | `compute_tensor_size_()`                        | Total element count for a Tensor             |
 * | `compute_grad_tensor_strides_()`                | Row-major strides for a TensorGrad           |
 * | `compute_grad_tensor_size_()`                   | Total element count for a TensorGrad         |
 * | `compute_linear_byte_offset()`                  | Multi-dim coords → linear byte offset        |
 * | `compute_coords_given_linear_byte_offset_()`    | Linear byte offset → multi-dim coords        |
 * | `create_unallocated_tensor()`                   | Unallocated Tensor (no data buffer)          |
 * | `create_unallocated_grad_tensor()`              | Heap-allocated unallocated TensorGrad        |
 * | `create_unallocated_scalar_tensor()`            | Unallocated scalar Tensor                    |
 * | `create_unallocated_scalar_grad_tensor()`       | Heap-allocated unallocated scalar TensorGrad |
 * | `odometer()`                                    | Row-major coordinate increment               |
 */
// clang-format on
/**
 * ## Naming Convention
 *
 * Functions ending with `_` (e.g., `compute_tensor_size_`) are
 * internal helpers — they must not be called from outside the
 * `ncore` library.
 *
 * @see tensor.h    Tensor struct and public API.
 * @see dtype.h     DType_ enum and dtype_size().
 * @see macros.h    NOVA_MAX_DIMS, ALIGN().
 */

#pragma once

#include <ncore/dtype.h>
#include <ncore/macros.h>
#include <ncore/tensor.h>
#include <stdlib.h>
#include <string.h>

/**
 * @typedef coords_t
 * @brief Multi-dimensional coordinate array type.
 *
 * @details
 * Fixed-size array of `NOVA_MAX_DIMS` elements representing a
 * position within an n-dimensional tensor.  Only the first `ndims`
 * entries are meaningful.
 *
 * Used by `compute_linear_byte_offset()`,
 * `compute_coords_given_linear_byte_offset_()`, and `odometer()`.
 */
typedef size_t coords_t[NOVA_MAX_DIMS];

/**
 * @brief Compute per-dimension contiguous strides for a Tensor.
 *
 * @details
 * Fills `ten->strides[0..ndims-1]` with row-major strides:
 *
 * ```
 * strides[ndims-1] = item_size
 * strides[dim]     = strides[dim+1] * shape[dim+1]   for dim = ndims-2 .. 0
 * ```
 *
 * This produces the standard C row-major layout where the last
 * dimension varies fastest.
 *
 * @param[in]      ten       Tensor whose `strides[]` will be
 *                           written.
 * @param[in]      ndims     Number of dimensions.
 * @param[in]      shape     Dimension sizes.
 * @param[in]      item_size Element size in bytes (from
 *                           `dtype_size()`).
 *
 * @pre  `ten` must not be `NULL`.
 * @pre  `ndims` must be > 0.
 * @post `ten->strides[0..ndims-1]` contain valid row-major strides.
 *
 * @see compute_tensor_size_()          Computes `ten->size`.
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
 * Multiplies all `ndims` dimension sizes together and stores the
 * result in `ten->size`.  For a scalar (`ndims == 0`), `ten->size`
 * is set to 1.
 *
 * @param[in,out] ten    Tensor whose `size` will be set.
 * @param[in]     shape  Dimension sizes (first `ten->ndims`
 *                       entries).
 *
 * @pre  `ten` must not be `NULL`.
 * @post `ten->size == product(shape[0..ten->ndims-1])`, or 1 if
 *       `ten->ndims == 0`.
 *
 * @see compute_tensor_strides_()          Computes `ten->strides`.
 * @see compute_grad_tensor_size_()        Gradient variant.
 */
static inline void compute_tensor_size_(Tensor *ten, const shape_t shape) {
  size_t size = 1;
  for (size_t dim = 0; dim < ten->ndims; dim++) {
    size *= shape[dim];
  }
  ten->size = size;
}

/**
 * @brief Compute per-dimension contiguous strides for a TensorGrad.
 *
 * @details
 * Identical logic to `compute_tensor_strides_()` but operates on
 * a `TensorGrad` (pointer-to-`Tensor`).
 *
 * @param[in,out] grad      TensorGrad whose `strides[]` will be
 *                          written.
 * @param[in]     ndims     Number of dimensions.
 * @param[in]     shape     Dimension sizes.
 * @param[in]     item_size Element size in bytes.
 *
 * @pre  `grad` must not be `NULL`.
 * @pre  `ndims` must be > 0.
 * @post `grad->strides[0..ndims-1]` contain valid row-major strides.
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
 * Identical logic to `compute_tensor_size_()` but operates on a
 * `TensorGrad`.
 *
 * @param[in,out] grad   TensorGrad whose `size` will be set.
 * @param[in]     shape  Dimension sizes (first `grad->ndims`
 *                       entries).
 *
 * @pre  `grad` must not be `NULL`.
 * @post `grad->size == product(shape[0..grad->ndims-1])`, or 1 if
 *       `grad->ndims == 0`.
 *
 * @see compute_tensor_size_()  Tensor variant.
 */
static inline void compute_grad_tensor_size_(TensorGrad grad,
                                             const shape_t shape) {
  size_t size = 1;
  for (size_t dim = 0; dim < grad->ndims; dim++) {
    size *= shape[dim];
  }
  grad->size = size;
}

/**
 * @brief Compute the linear byte offset from multi-dimensional
 *        coordinates.
 *
 * @details
 * Applies the standard strided offset formula:
 *
 * ```
 * offset = Σ (coords[dim] × strides[dim])   for dim = 0 .. ndims-1
 * ```
 *
 * This is the byte distance from the start of the data buffer to
 * the element at the given coordinates.
 *
 * @param[in] coords   Multi-dimensional coordinates.  Only the
 *                     first `ndims` entries are read.
 * @param[in] ndims    Number of dimensions.
 * @param[in] strides  Per-dimension stride values (in bytes).
 *
 * @return Byte offset from the start of the data buffer.
 *
 * @pre  `ndims` must be > 0.
 * @pre  Each `coords[dim]` must be < `shape[dim]` (not validated).
 *
 * @see compute_coords_given_linear_byte_offset_()  Inverse
 *      operation.
 */
static inline size_t compute_linear_byte_offset(const coords_t coords,
                                                size_t ndims,
                                                const strides_t strides) {
  size_t offset = 0;
  for (size_t dim = 0; dim < ndims; dim++) {
    offset += coords[dim] * strides[dim];
  }
  return offset;
}

/**
 * @brief Reconstruct multi-dimensional coordinates from a linear
 *        byte offset.
 *
 * @details
 * Performs the inverse of `compute_linear_byte_offset()` by
 * repeatedly dividing the offset by each dimension's stride:
 *
 * ```
 * for dim = 0 .. ndims-1:
 *     coords[dim] = offset / strides[dim]
 *     offset      = offset % strides[dim]
 * ```
 *
 * After the loop, `offset` should be 0 for a valid position.
 *
 * @param[in]  offset  Linear byte offset into the data buffer.
 * @param[in]  ndims   Number of dimensions.
 * @param[out] coords  Output array (written in-place).  Only the
 *                     first `ndims` entries are written.
 * @param[in]  strides Per-dimension stride values (in bytes).
 *
 * @pre  `ndims` must be > 0.
 * @pre  `offset` must be < total buffer size (not validated).
 * @post `coords[0..ndims-1]` contain the reconstructed position.
 *
 * @see compute_linear_byte_offset()  Forward operation.
 */
static inline void compute_coords_given_linear_byte_offset_(
    size_t offset, size_t ndims, coords_t coords, const strides_t strides) {
  for (size_t dim = 0; dim < ndims; dim++) {
    coords[dim] = (offset / strides[dim]);
    offset %= strides[dim];
  }
}

/**
 * @brief Create a Tensor without allocating a data buffer.
 *
 * @details
 * Zero-initialises a `Tensor`, copies shape metadata, computes
 * size and strides, but leaves `storage = NULL`,
 * `data.data = NULL`, and `is_allocated_ = false`.  The tensor
 * is a valid metadata-only shell ready for deferred allocation.
 *
 * Used internally by `create_tensor_like()` when the source
 * tensor is not yet allocated.
 *
 * @param[in]  shape         Dimension sizes.
 * @param[in]  dtype         Element data type (`DType_`).
 * @param[in]  device        Target device (`DEVICE_CPU`,
 *                           `DEVICE_GPU`, or `DEVICE_META`).
 * @param[in]  requires_grad Whether to track gradients.
 * @param[in]  pin_memory    If `true`, request page-locked host
 *                           memory (CPU only).
 * @param[in]  ndims         Number of dimensions.
 *
 * @return Initialised `Tensor` with no backing storage.
 *
 * @pre  `ndims` must not exceed `NOVA_MAX_DIMS`.
 * @post `is_allocated_ == false`, `storage == NULL`,
 *       `data.data == NULL`.
 * @post `shape`, `strides`, `size`, and `item_size` are valid.
 *
 * @see create_tensor()                   Allocated variant.
 * @see create_unallocated_scalar_tensor() Scalar variant.
 * @see create_unallocated_grad_tensor()   Heap-allocated gradient.
 */
static inline Tensor create_unallocated_tensor(const shape_t shape,
                                               DType_ dtype, Device device,
                                               bool requires_grad,
                                               bool pin_memory,
                                               size_t ndims) {
  Tensor tensor = {0};
  memcpy(tensor.shape, shape, ndims * sizeof(size_t));
  tensor.dtype = dtype;
  tensor.device = device;
  tensor.ndims = ndims;
  tensor.item_size = dtype_size(dtype);
  tensor.offset = 0;
  tensor.grad = NULL;
  tensor.is_leaf_ = true;
  tensor.is_view_ = false;
  tensor.requires_grad_ = requires_grad;
  tensor.retain_grad_ = false;
  tensor.grad_fn_ = NULL;
  tensor.scale_ = 1.0F;
  tensor.zero_point_ = 0;
  tensor.storage = NULL;
  tensor.data.data = NULL;
  tensor.is_allocated_ = false;
  tensor.version_ = 0;
  tensor.is_pinned = pin_memory;
  compute_tensor_size_(&tensor, shape);
  compute_tensor_strides_(&tensor, ndims, shape, tensor.item_size);
  return tensor;
}

/**
 * @brief Create a heap-allocated unallocated TensorGrad.
 *
 * @details
 * Allocates a `Tensor` on the heap via `malloc()`, then
 * initialises it as an unallocated tensor via
 * `create_unallocated_tensor()`.  The gradient is a metadata-only
 * shell — no data buffer is allocated.
 *
 * The returned pointer must be freed by the caller (or via
 * `collect()` on the parent tensor, which recursively frees
 * gradients).
 *
 * @param[in]  shape      Dimension sizes.
 * @param[in]  dtype      Element data type (`DType_`).
 * @param[in]  device     Target device.
 * @param[in]  pin_memory If `true`, request page-locked host
 *                        memory (CPU only).
 * @param[in]  ndims      Number of dimensions.
 *
 * @return Pointer to newly allocated `Tensor` (as `TensorGrad`).
 *
 * @pre  `malloc()` must succeed (asserted internally).
 * @post The returned pointer is heap-allocated and must be freed.
 * @post `grad->is_allocated_ == false`.
 *
 * @see create_unallocated_tensor()              Non-heap variant.
 * @see create_unallocated_scalar_grad_tensor()  Scalar variant.
 */
static inline TensorGrad create_unallocated_grad_tensor(const shape_t shape,
                                                        DType_ dtype,
                                                        Device device,
                                                        bool pin_memory,
                                                        size_t ndims) {
  TensorGrad grad = (TensorGrad)malloc(sizeof(Tensor));
  NOVA_INTERNAL_ASSERT(
      grad != NULL,
      "[GRAD] create_unallocated_grad_tensor: malloc returned NULL\n");
  *grad = create_unallocated_tensor(shape, dtype, device, false, pin_memory,
                                    ndims);
  return grad;
}

/**
 * @brief Create a scalar (0-D) Tensor without allocating a data
 *        buffer.
 *
 * @details
 * Zero-initialises a `Tensor` and sets the scalar invariant:
 * `shape[0] = 0`, `strides[0] = 0`, `size = 1`, `ndims = 0`.
 * No data buffer is allocated.
 *
 * @param[in]  dtype         Element data type (`DType_`).
 * @param[in]  device        Target device.
 * @param[in]  requires_grad Whether to track gradients.
 * @param[in]  pin_memory    If `true`, request page-locked host
 *                           memory (CPU only).
 *
 * @return Initialised scalar `Tensor` with no backing storage.
 *
 * @post `is_allocated_ == false`, `storage == NULL`,
 *       `data.data == NULL`.
 * @post `ndims == 0`, `size == 1`, `shape[0] == 0`,
 *       `strides[0] == 0`.
 *
 * @see create_unallocated_tensor()           N-dimensional variant.
 * @see create_unallocated_scalar_grad_tensor() Scalar gradient.
 * @see create_scalar_tensor()                Allocated variant.
 */
static inline Tensor create_unallocated_scalar_tensor(DType_ dtype,
                                                      Device device,
                                                      bool requires_grad,
                                                      bool pin_memory) {
  Tensor tensor = {0};
  tensor.shape[0] = 0;
  tensor.strides[0] = 0;
  tensor.size = 1;
  tensor.ndims = 0;
  tensor.dtype = dtype;
  tensor.device = device;
  tensor.item_size = dtype_size(dtype);
  tensor.offset = 0;
  tensor.grad = NULL;
  tensor.is_leaf_ = true;
  tensor.is_view_ = false;
  tensor.requires_grad_ = requires_grad;
  tensor.retain_grad_ = false;
  tensor.grad_fn_ = NULL;
  tensor.scale_ = 1.0F;
  tensor.zero_point_ = 0;
  tensor.storage = NULL;
  tensor.data.data = NULL;
  tensor.is_allocated_ = false;
  tensor.version_ = 0;
  tensor.is_pinned = pin_memory;
  return tensor;
}

/**
 * @brief Create a heap-allocated unallocated scalar TensorGrad.
 *
 * @details
 * Allocates a `Tensor` on the heap via `malloc()`, then
 * initialises it as an unallocated scalar tensor via
 * `create_unallocated_scalar_tensor()`.
 *
 * @param[in]  dtype      Element data type (`DType_`).
 * @param[in]  device     Target device.
 * @param[in]  pin_memory If `true`, request page-locked host
 *                        memory (CPU only).
 *
 * @return Pointer to newly allocated scalar `Tensor` (as
 *         `TensorGrad`).
 *
 * @pre  `malloc()` must succeed (asserted internally).
 * @post The returned pointer is heap-allocated and must be freed.
 * @post `grad->is_allocated_ == false`, `grad->ndims == 0`.
 *
 * @see create_unallocated_scalar_tensor()  Non-heap variant.
 * @see create_unallocated_grad_tensor()    N-dimensional variant.
 */
static inline TensorGrad create_unallocated_scalar_grad_tensor(DType_ dtype,
                                                               Device device,
                                                               bool pin_memory) {
  TensorGrad grad = (TensorGrad)malloc(sizeof(Tensor));
  NOVA_INTERNAL_ASSERT(
      grad != NULL,
      "[GRAD] create_unallocated_scalar_grad_tensor: malloc returned NULL\n");
  *grad = create_unallocated_scalar_tensor(dtype, device, false, pin_memory);
  return grad;
}

/**
 * @brief Advance a multi-dimensional coordinate by one step
 *        (row-major).
 *
 * @details
 * Increments the last dimension (`ndims-1`) by one and propagates
 * the carry to earlier dimensions when a dimension reaches its
 * upper bound (`shape[dim]`), resetting it to zero.
 *
 * When all dimensions overflow, the coordinate wraps back to all
 * zeros — matching the behaviour of a traditional mechanical
 * odometer.
 *
 * Typical usage for iterating over all elements:
 *
 * ```c
 * coords_t coord = {0};
 * for (size_t i = 0; i < tensor.size; i++) {
 *     // process element at coord
 *     odometer(coord, tensor.ndims, tensor.shape);
 * }
 * ```
 *
 * @param[in,out] coords  Current coordinates, updated in-place.
 * @param[in]     ndims   Number of dimensions.
 * @param[in]     shape   Dimension sizes (upper bound for each
 *                        coordinate).
 *
 * @pre  `ndims` must be > 0.
 * @post `coords` is advanced to the next position, or wrapped to
 *       `{0, 0, ..., 0}` if at the end.
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
