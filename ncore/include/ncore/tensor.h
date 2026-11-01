/**
 * @file tensor.h
 * @brief Core Tensor type: the n-dimensional array with autograd
 *        support.
 *
 * A Tensor bundles layout metadata, a reference-counted
 * @ref TensorStorage owned by the Rust allocator, optional gradient
 * fields for autograd, and quantization parameters (scale and
 * zero-point). Views can share one storage through
 * @c retain() / @c release().
 *
 * Every creation and mutation function accepts an output
 * @ref novaStatus_t pointer that receives the operation result.
 * Check @c status.err after each call; on failure the returned tensor
 * is zeroed and must not be used.
 */

#pragma once

#include <stdint.h>

#include <ncore/core/backend.h>
#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/core/storage.h>
#include <ncore/headeronly/macros.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @typedef shape_t
 * @brief Fixed-size array for tensor dimension sizes.
 *
 * Holds up to @ref NOVA_MAX_DIMS elements; only the first @c ndims
 * entries are meaningful. @c shape = {2, 3, 4} with @c ndims = 3 is a
 * 2×3×4 tensor.
 */
typedef size_t shape_t[NOVA_MAX_DIMS];

/**
 * @typedef strides_t
 * @brief Fixed-size array for tensor byte strides.
 *
 * @c strides[i] is the number of bytes to advance in the data buffer
 * to reach the next element along dimension @c i. For a contiguous
 * float32 @c {2, 3, 4} tensor the strides are @c {48, 16, 4}.
 */
typedef size_t strides_t[NOVA_MAX_DIMS];

/**
 * @struct Node
 * @brief Backward pass node for automatic differentiation.
 *
 * Opaque node of the computational graph. An operation that requires
 * gradient tracking stores a @ref BackwardNode in the output tensor's
 * @c grad_fn_ field.
 */
struct Node;
typedef struct Node Node;

/** @brief Pointer-to-Node type alias used in @ref Tensor::grad_fn_. */
typedef Node *BackwardNode;

/**
 * @struct Tensor
 * @brief Multi-dimensional array with automatic differentiation
 *        support.
 *
 * Cache-line aligned (@c ALIGN(64)) for SIMD access to the metadata
 * fields.
 */
struct Tensor {
  size_t item_size;            ///< Bytes per element (from @ref dtype_size).
  size_t offset;               ///< Byte offset into the storage buffer.
  size_t ndims;                ///< Number of dimensions (0 for scalars).
  size_t size;                 ///< Total element count (product of shape).
  size_t logical_size;         ///< Unpacked element count (individual values).
  DType_ dtype;                ///< Element data type (@ref DType_ enum).
  Device_ device;              ///< Placement device (@ref Device_ enum).
  ALIGN(64) shape_t shape;     ///< Dimension sizes (up to @ref NOVA_MAX_DIMS).
  ALIGN(64) strides_t strides; ///< Byte strides per dimension.
  TensorStorage
      *storage;    ///< Reference-counted backing buffer (nullptr for META).
  data_ptr data;   ///< Typed pointer into @c storage.
  TensorGrad grad; ///< Gradient tensor (nullptr if no gradient tracked).
  BackwardNode grad_fn_; ///< Backward graph node (nullptr for leaves).
  bool requires_grad_;   ///< If @c true, track gradients during backward.
  bool retain_grad_;     ///< If @c true, retain gradient for non-leaf nodes.
  bool is_view_;         ///< If @c true, shares storage with another tensor.
  bool is_leaf_;         ///< If @c true, leaf node in the computation graph.
  bool is_allocated_;    ///< If @c true, @c storage and @c data are valid.
  bool is_pinned_;       ///< If @c true, page-locked host memory.
  float scale_;          ///< Quantization scale (1.0 if not quantized).
  int32_t zero_point_;   ///< Quantization zero point (0 if not quantized).
  int64_t version_;      ///< Mutation counter for in-place alias detection.
};

/** @brief Gradient tensor type alias: a pointer to a @ref Tensor. */
typedef Tensor *TensorGrad;

/**
 * @brief Create a fully allocated n-dimensional tensor.
 *
 * Zero-initializes the struct, copies the shape, computes @c size and
 * @c strides, and allocates a data buffer via @ref safe_allocator().
 * When @p requires_grad is @c true, an unallocated gradient tensor is
 * created in @c grad with the same shape, dtype, and device.
 *
 * For @c DEVICE_META, storage stays @c nullptr and the Rust allocator
 * is never invoked.
 *
 * @param[in]  shape         Dimension sizes (only the first @c ndims
 *                           entries are read).
 * @param[in]  dtype         Element data type (@ref DType_).
 * @param[in]  device        Target device (@c DEVICE_CPU,
 *                           @c DEVICE_GPU, or @c DEVICE_META).
 * @param[in]  requires_grad If @c true, creates an unallocated
 *                           gradient tensor in @c grad.
 * @param[in]  pin_memory    If @c true, request page-locked host
 *                           memory (CPU only).
 * @param[in]  ndims         Number of dimensions.  Must be > 0 and <=
 *                           @ref NOVA_MAX_DIMS; a scalar (@c ndims == 0)
 *                           must be created via @c create_scalar_tensor().
 * @param[out] status        Receives the operation result.
 *
 * @return Initialized @ref Tensor with valid backing storage, or a
 *         META tensor with nullptr storage.
 */
Tensor create_tensor(const shape_t shape, DType_ dtype, Device_ device,
                     bool requires_grad, bool pin_memory, size_t ndims,
                     novaStatus_t *status);

/**
 * @brief Create a fully allocated 0-dimensional (scalar) tensor.
 *
 * Shape and strides are zeroed, @c ndims is 0 and @c size is 1.
 *
 * @param[in]  dtype         Element data type (@ref DType_).
 * @param[in]  device        Target device (@c DEVICE_CPU,
 *                           @c DEVICE_GPU, or @c DEVICE_META).
 * @param[in]  requires_grad If @c true, creates an unallocated
 *                           scalar gradient tensor.
 * @param[in]  pin_memory    If @c true, request page-locked host
 *                           memory (CPU only).
 * @param[out] status        Receives the operation result.
 *
 * @return Initialized scalar @ref Tensor with backing storage.
 */
Tensor create_scalar_tensor(DType_ dtype, Device_ device, bool requires_grad,
                            bool pin_memory, novaStatus_t *status);

/**
 * @brief Create a tensor with the same metadata as an existing one.
 *
 * Copies shape, dtype, device, requires_grad, and pin_memory. If the
 * source is allocated the result is also allocated; otherwise an
 * unallocated tensor comes back. Scalars route through
 * @ref create_scalar_tensor() or
 * @ref create_unallocated_scalar_tensor().
 *
 * The result is independent: it owns its own storage.
 *
 * @param[in]  ten    Source tensor to copy metadata from.  Must not
 *                    be @c nullptr.
 * @param[out] status Receives the operation result.
 * @return New @ref Tensor with matching metadata and allocation state.
 */
Tensor create_tensor_like(const Tensor *ten, novaStatus_t *status);

/**
 * @brief Create a view of an existing tensor with a new shape.
 *
 * Increments the Rust-side reference count via @ref retain(),
 * recomputes strides for the new shape, and marks the result as a
 * non-leaf view. No data is copied.
 *
 * Scalar sources keep their shape and strides. If the source has a
 * gradient, an unallocated gradient tensor is created for the view;
 * on grad-creation failure the view's storage is released via
 * @ref collect() and the tensor is zeroed.
 *
 * @param[in]  src        Source tensor to view.  Must outlive the
 *                        view and have non-nullptr storage.
 * @param[in]  new_shape  New dimension sizes.  Product must equal
 *                        @c src->size.
 * @param[in]  new_ndims  Number of dimensions in @p new_shape.
 * @param[out] status     Receives the operation result.
 *
 * @return View @ref Tensor sharing @p src's storage.
 */
Tensor create_view(const Tensor *restrict src, const shape_t new_shape,
                   size_t new_ndims, novaStatus_t *status);

/**
 * @brief Move ownership of tensor resources from src to dst.
 *
 * Collects any existing resources in @p dst, then bitwise-copies
 * @p src into @p dst. @p src is zeroed (storage, data, grad, grad_fn_
 * null) so a later @ref collect() on it is a no-op. If releasing
 * @p dst fails, @p src is left untouched and the error is returned.
 *
 * @param[in,out] dst  Destination tensor (previous resources are
 *                     freed via @ref collect()).
 * @param[in,out] src  Source tensor (ownership transferred).
 *
 * @return @ref novaStatus_t with @c novaSuccess on success, or the
 *         release error from @ref collect().  @c novaInvalidPointer
 *         when @p dst or @p src is @c nullptr.
 *
 * @pre  @p dst and @p src must not be the same object.
 */
novaStatus_t move_tensor(Tensor *restrict dst, Tensor *restrict src);

/**
 * @brief Release tensor memory and gradients.
 *
 * Decrements the storage reference count via @ref release(). When the
 * count reaches zero and release succeeds, the @c TensorStorage
 * descriptor is freed and the gradient chain is traversed and freed
 * iteratively. On a release error the storage descriptor remains
 * attached for retry.
 *
 * @param[in,out] ten  Tensor to collect.  May be @c nullptr
 *                     (returns @c novaSuccess).
 * @return @ref novaStatus_t with @c novaSuccess on success, or the
 *         first release error encountered.
 */
novaStatus_t collect(Tensor *ten);

/**
 * @brief Check whether a tensor is contiguous in memory.
 *
 * Elements live in row-major order without gaps: strides decrease
 * strictly by @c shape[dim] × item_size per dimension.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 * @return @c true if the tensor is contiguous.
 */
bool is_contiguous(const Tensor *restrict ten);

/**
 * @brief Ensure a tensor's data is stored contiguously.
 *
 * Returns a row-major tensor. If @p ten is already contiguous, a view
 * sharing its storage comes back via @ref create_view() with no copy;
 * otherwise a new tensor is allocated and the elements are copied
 * into it. Scalars route through @ref create_scalar_tensor() so the
 * result keeps @c ndims == 0.
 *
 * @param[in]  ten    Tensor to make contiguous.  Must not be
 *                    @c nullptr.
 * @param[out] status Receives the operation result.
 * @return Contiguous @ref Tensor, or a zeroed tensor on failure.
 */
Tensor contiguous(const Tensor *restrict ten, novaStatus_t *status);

/**
 * @brief Swap two dimensions of a tensor, returning a view.
 *
 * Shares @p ten's storage without copying: element @c [i, j] of the
 * result aliases @c [j, i] of the source. Negative indices count from
 * the last dimension.
 *
 * @param[in]  ten   Source tensor. Must not be @c nullptr.
 * @param[out] st    Receives the operation result.
 * @param[in]  dim0  First dimension to swap. May be negative.
 * @param[in]  dim1  Second dimension to swap. May be negative.
 *
 * @return View @c Tensor sharing @p ten's storage.  On failure the
 *         result must not be used.
 */
Tensor transpose(const Tensor *ten, novaStatus_t *st, int dim0, int dim1);

/**
 * @brief Reorder tensor dimensions following an explicit permutation.
 *
 * Shares @p ten's storage without copying: dimension @c i of the
 * result takes the shape and stride of source dimension @c dims[i].
 * Every source dimension must appear exactly once; out-of-range
 * indices and duplicates fail with @ref novaInvalidValue. Negative
 * entries count from the last dimension.
 *
 * @param[in]  ten   Source tensor. Must not be @c nullptr.
 * @param[out] st    Receives the operation result.
 * @param[in]  dims  Permutation of @c [0, ndims). Only the first
 *                   @c ndims entries are read.
 *
 * @return View @c Tensor sharing @p ten's storage.  On failure the
 *         result must not be used.
 */
Tensor permute(const Tensor *ten, novaStatus_t *st,
               const int dims[NOVA_MAX_DIMS]);

/**
 * @brief Check whether a tensor is 0-dimensional (scalar).
 *
 * Scalar means @c ndims == 0, @c shape[0] == 0, @c strides[0] == 0,
 * and @c size == 1.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 * @return @c true if the tensor is a scalar.
 */
bool is_scalar(const Tensor *ten);

/**
 * @brief Check whether a gradient tensor is 0-dimensional (scalar).
 *
 * @param[in] grad  Gradient tensor to check.  Must not be @c nullptr.
 * @return @c true if the gradient is a scalar.
 */
bool is_scalar_grad(TensorGrad grad);

/**
 * @brief Check whether a tensor has been collected (freed).
 *
 * True when @c data.data == @c nullptr, @c storage == @c nullptr and
 * @c is_allocated_ == false, the state after @ref collect() fully
 * released the tensor.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 * @return @c true if the tensor has been collected.
 */
bool is_collected(const Tensor *ten);

/**
 * @brief Check whether a gradient tensor has been collected.
 *
 * @param[in] grad  Gradient tensor to check.  May be @c nullptr.
 * @return @c true if the gradient has been collected (or is
 *         @c nullptr).
 */
bool is_grad_collected(TensorGrad grad);

/**
 * @brief Check whether a tensor's data buffer has been allocated.
 *
 * @c is_allocated_, @c storage and @c data.data all must hold valid
 * values for this to be true.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 * @return @c true if the tensor is allocated.
 */
bool is_allocated(const Tensor *ten);

/**
 * @brief Check whether a gradient tensor has been allocated.
 *
 * Same triple check as @ref is_allocated(); a @c nullptr gradient
 * returns @c false.
 *
 * @param[in] grad  Gradient tensor to check.  May be @c nullptr.
 * @return @c true if the gradient has a valid backing buffer.
 */
bool is_grad_allocated(TensorGrad grad);

/**
 * @brief Check whether a tensor is a view (shares storage with
 *        another tensor).
 *
 * Views are created by @ref create_view() and carry @c is_leaf_ ==
 * false.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 * @return @c true if the tensor is a view.
 */
bool is_view(const Tensor *ten);

/**
 * @brief Check whether a gradient tensor is a view.
 *
 * @param[in] grad  Gradient tensor to check.  Must not be @c nullptr.
 * @return @c true if the gradient is a view.
 */
bool is_grad_view(TensorGrad grad);

/**
 * @brief Check whether a tensor's data resides in GPU device memory.
 *
 * Requires allocation, @c ten->device == @c DEVICE_GPU, and a
 * device-managed storage handle. Genuine VRAM residency: pinned host
 * tensors have device-managed handles too, but they do not count
 * here.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 * @return @c true if the tensor is allocated, GPU-resident, and
 *         device-backed.
 */
bool in_device(const Tensor *ten);

/**
 * @brief Check whether a tensor's data resides in CPU host memory.
 *
 * Requires allocation, @c ten->device == @c DEVICE_CPU, and a storage
 * handle that is _not_ device-managed. Pinned memory fails this
 * check on purpose; it belongs to the GPU backend.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 * @return @c true if the tensor is allocated and backed by plain
 *         host memory.
 */
bool in_host(const Tensor *ten);

/**
 * @brief Transfer tensor data from GPU device memory to CPU host
 *        memory.
 *
 * Validates that @p src is a GPU-resident tensor with device memory
 * and that @p dst is an allocated CPU tensor, then performs the copy
 * via @ref transfer_to().
 *
 * @param[in]  src  Source tensor on GPU.  Must have non-nullptr storage
 *                   and be backed by device memory.
 * @param[in,out] dst  Destination tensor on CPU.  Must be allocated.
 * @return @ref novaStatus_t with @c novaSuccess on success, or an
 *         error code describing the failure.
 */
novaStatus_t transf_tensor_from_device(const Tensor *restrict src,
                                       Tensor *restrict dst);

/**
 * @brief Transfer tensor data from CPU host memory to GPU device
 *        memory.
 *
 * Validates that @p src is an allocated CPU tensor and that @p dst
 * is a GPU-resident tensor with device memory, then performs the
 * copy via @ref transfer_to().
 *
 * @param[in]  src  Source tensor on CPU.  Must be allocated.
 * @param[in,out] dst  Destination tensor on GPU.  Must have non-nullptr
 *                     storage and be backed by device memory.
 * @return @ref novaStatus_t with @c novaSuccess on success, or an
 *         error code describing the failure.
 */
novaStatus_t transf_tensor_from_host(const Tensor *restrict src,
                                     Tensor *restrict dst);

#ifdef __cplusplus
}
#endif
