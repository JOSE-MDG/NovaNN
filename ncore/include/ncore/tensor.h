/**
 * @file tensor.h
 * @brief Core Tensor type — the fundamental n-dimensional array with
 *        autograd support.
 *
 * @details
 * Defines the @ref Tensor struct, its associated typedefs
 * (@ref shape_t, @ref strides_t, @ref TensorGrad), and the public
 * API for creation, views, memory management, and query predicates.
 * All cache-line aligned fields (`ALIGN(64)`) are padded for optimal
 * SIMD vectorisation.
 *
 * All creation and mutation functions accept an output
 * @ref novaStatus_t pointer that receives the operation result.
 * Callers must check `status.err` after every call; on failure the
 * returned tensor is zeroed and must not be used.
 *
 * ## Architecture
 *
 * A Tensor bundles:
 * - **Metadata** — shape, strides, dtype, device, ndims, size, etc.
 * - **Storage** — a reference-counted @ref TensorStorage owned by the
 *   Rust allocator.  Multiple tensors (views) can share the same
 *   storage via `retain()` / `release()`.
 * - **Autograd** — optional gradient tensor (@ref TensorGrad) and
 *   backward function node (@ref BackwardNode) for automatic
 *   differentiation.
 * - **Quantisation** — scale and zero-point fields for quantised
 *   inference.
 *
 * ## Lifecycle
 *
 * 1. Create via @ref create_tensor() or @ref create_scalar_tensor().
 * 2. Use in computations; views share storage via @ref create_view().
 * 3. Release via @ref collect() — decrements reference count and
 *    recursively frees gradients.
 *
 * @see dtype.h      Data-type definitions and DType_ enum.
 * @see storage.h    TensorStorage, RustHandle, and FFI allocation.
 * @see device.h     Device placement and memory transfers.
 * @see alloc.h      safe_allocator() and typed buffer allocation.
 */

#pragma once

#include <ncore/core/backend.h>
#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/core/storage.h>
#include <ncore/headeronly/macros.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @typedef shape_t
 * @brief Fixed-size array for tensor dimension sizes.
 *
 * @details
 * Holds up to @ref NOVA_MAX_DIMS elements.  Only the first `ndims`
 * entries are meaningful.  For a scalar, all entries are 0.
 *
 * Example: `shape = {2, 3, 4}` with `ndims = 3` represents a
 * 2×3×4 tensor.
 */
typedef size_t shape_t[NOVA_MAX_DIMS] ALIGN(64);

/**
 * @typedef strides_t
 * @brief Fixed-size array for tensor byte strides.
 *
 * @details
 * `strides[i]` is the number of bytes to advance in the data
 * buffer to reach the next element along dimension `i`.  For a
 * contiguous tensor, `strides[i] = item_size × product(shape[i+1..])`.
 *
 * Example: for a contiguous float32 tensor with shape `{2, 3, 4}`,
 * strides are `{48, 16, 4}`.
 */
typedef size_t strides_t[NOVA_MAX_DIMS] ALIGN(64);

/**
 * @struct Node
 * @brief Backward pass node for automatic differentiation.
 *
 * @details
 * Opaque type representing a node in the computational graph.
 * Each operation that requires gradient tracking stores a
 * @ref BackwardNode in the output tensor's `grad_fn_` field.
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
 * @details
 * Core data structure used throughout NovaNN.  Every tensor
 * carries:
 * - Shape, strides, and total element count for layout.
 * - A @ref DType_ and @ref Device for type and placement.
 * - A reference-counted @ref TensorStorage for the backing buffer.
 * - Optional gradient fields for autograd.
 * - Quantisation parameters (scale, zero-point).
 *
 * The struct is cache-line aligned (`ALIGN(64)`) to enable
 * efficient SIMD operations on the metadata fields.
 *
 * ## Field Groups
 */
// clang-format off
/**
 * | Group          | Fields                                                  |
 * |----------------|---------------------------------------------------------|
 * | Layout         | `shape`, `strides`, `ndims`, `size`, `item_size`, `offset` |
 * | Type/Device    | `dtype`, `device`                                       |
 * | Storage        | `storage`, `data`, `is_allocated_`, `is_pinned`         |
 * | Autograd       | `grad`, `grad_fn_`, `requires_grad_`, `retain_grad_`, `is_leaf_`, `is_view_` |
 * | Quantisation   | `scale_`, `zero_point_`                                 |
 * | Mutation track | `version_`                                              |
 */
// clang-format on
/**
 * @see shape_t      Fixed-size shape array.
 * @see strides_t    Fixed-size strides array.
 * @see TensorGrad   Gradient tensor alias.
 */
struct ALIGN(64) Tensor {
  size_t item_size;       ///< Bytes per element (from @ref dtype_size).
  size_t offset;          ///< Byte offset into the storage buffer.
  size_t ndims;           ///< Number of dimensions (0 for scalars).
  size_t size;            ///< Total element count (product of shape).
  DType_ dtype;           ///< Element data type (@ref DType_ enum).
  Device device;          ///< Placement device (@ref Device enum).
  shape_t shape;          ///< Dimension sizes (up to @ref NOVA_MAX_DIMS).
  strides_t strides;      ///< Byte strides per dimension.
  TensorStorage *storage; ///< Reference-counted backing buffer (NULL for META).
  data_ptr data;          ///< Typed pointer into `storage` (offset-adjusted).
  TensorGrad grad;        ///< Gradient tensor (NULL if no gradient tracked).
  BackwardNode grad_fn_;  ///< Backward graph node (NULL for leaves).
  bool requires_grad_;    ///< If `true`, track gradients during backward.
  bool retain_grad_;      ///< If `true`, retain gradient for non-leaf nodes.
  bool is_view_;          ///< If `true`, shares storage with another tensor.
  bool is_leaf_;          ///< If `true`, leaf node in the computation graph.
  bool is_allocated_;     ///< If `true`, `storage` and `data` are valid.
  bool is_pinned;         ///< If `true`, page-locked host memory.
  float scale_;           ///< Quantisation scale (1.0 if not quantised).
  int32_t zero_point_;    ///< Quantisation zero point (0 if not quantised).
  int64_t version_;       ///< Mutation counter for in-place alias detection.
};

/** @brief Gradient tensor type alias — a pointer to a @ref Tensor. */
typedef Tensor *TensorGrad;

/**
 * @brief Create a fully allocated n-dimensional tensor.
 *
 * @details
 * Zero-initialises the @ref Tensor struct, copies the shape, computes
 * `size` and `strides`, and allocates a data buffer via
 * @ref safe_allocator().  When @p requires_grad is `true`, an
 * unallocated gradient tensor is created in `grad` with the same
 * shape, dtype, and device as the parent.
 *
 * For `DEVICE_META`, storage is left `NULL` and `is_allocated_` is
 * set to `false`.  The Rust allocator is never invoked.
 *
 * On any allocation or gradient-creation failure, the tensor is
 * zeroed and the caller receives a non-success status through
 * @p status.
 *
 * @param[in]  shape         Dimension sizes (only the first `ndims`
 *                           entries are read).
 * @param[in]  dtype         Element data type (@ref DType_).
 * @param[in]  device        Target device (`DEVICE_CPU`,
 *                           `DEVICE_GPU`, or `DEVICE_META`).
 * @param[in]  requires_grad If `true`, creates an unallocated
 *                           gradient tensor in `grad`.
 * @param[in]  pin_memory    If `true`, request page-locked host
 *                           memory (CPU only).
 * @param[in]  ndims         Number of dimensions.  Must be <=
 *                           @ref NOVA_MAX_DIMS.
 * @param[out] status        Receives the operation result.  On
 *                           success `err` is @ref novaSuccess.
 *
 * @return Initialised @ref Tensor with valid backing storage, or a
 *         META tensor with NULL storage.
 *
 * @pre  @p ndims must not exceed @ref NOVA_MAX_DIMS.
 * @pre  `product(shape[0..ndims])` must be > 0 for non-META.
 * @pre  @p status must not be `NULL`.
 * @post On success, `is_allocated_ == true` (unless META).
 * @post If `requires_grad`, `grad` points to an unallocated tensor.
 *
 * @see create_scalar_tensor()   Scalar (0-D) variant.
 * @see create_tensor_like()     Clone metadata from an existing tensor.
 * @see safe_allocator()         Underlying allocation.
 */
Tensor create_tensor(const shape_t shape, DType_ dtype, Device device,
                     bool requires_grad, bool pin_memory, size_t ndims,
                     novaStatus_t *status);

/**
 * @brief Create a fully allocated 0-dimensional (scalar) tensor.
 *
 * @details
 * Allocates a single-element data buffer.  Shape and strides are
 * zeroed, `ndims` is set to 0, and `size` is set to 1.
 * Allocation is performed via @ref safe_allocator().  When
 * @p requires_grad is `true`, an unallocated scalar gradient tensor
 * is created in `grad`.
 *
 * For `DEVICE_META`, storage is left `NULL` and `is_allocated_` is
 * set to `false`.  On failure, the tensor is zeroed and the error
 * is reported through @p status.
 *
 * @param[in]  dtype         Element data type (@ref DType_).
 * @param[in]  device        Target device (`DEVICE_CPU`,
 *                           `DEVICE_GPU`, or `DEVICE_META`).
 * @param[in]  requires_grad If `true`, creates an unallocated
 *                           scalar gradient tensor.
 * @param[in]  pin_memory    If `true`, request page-locked host
 *                           memory (CPU only).
 * @param[out] status        Receives the operation result.
 *
 * @return Initialised scalar @ref Tensor with backing storage.
 *
 * @pre  @p status must not be `NULL`.
 * @post On success, `is_allocated_ == true` (unless META).
 *
 * @see create_tensor()        N-dimensional variant.
 * @see is_scalar()            Query predicate.
 */
Tensor create_scalar_tensor(DType_ dtype, Device device, bool requires_grad,
                            bool pin_memory, novaStatus_t *status);

/**
 * @brief Create a tensor with the same metadata as an existing one.
 *
 * @details
 * Inspects the source tensor and produces a new tensor with
 * identical shape, dtype, device, requires_grad, and pin_memory.
 * If the source is allocated, the result is also allocated;
 * otherwise an unallocated tensor is returned.  Scalar tensors are
 * handled via @ref create_scalar_tensor() or
 * @ref create_unallocated_scalar_tensor().
 *
 * The new tensor is independent of the source — it owns its own
 * storage and does not share the source's buffer.
 *
 * @param[in]  ten    Source tensor to copy metadata from.  Must not
 *                    be `NULL`.
 * @param[out] status Receives the operation result.
 *
 * @return New @ref Tensor with matching metadata and allocation
 *         state.
 *
 * @pre  @p ten must not be `NULL`.
 * @pre  @p status must not be `NULL`.
 *
 * @see create_tensor()
 * @see create_unallocated_tensor()
 */
Tensor create_tensor_like(const Tensor *ten, novaStatus_t *status);

/**
 * @brief Create a view of an existing tensor with a new shape.
 *
 * @details
 * Shares the same underlying storage (incrementing the Rust-side
 * reference count via @ref retain()), recomputes strides for the
 * new shape, and marks the result as a non-leaf view.  The
 * original data buffer is not copied.
 *
 * Scalar sources are handled specially — shape and strides are
 * left unchanged.  If the source has a gradient, an unallocated
 * gradient tensor is created for the view.  On grad-creation
 * failure, the view's storage is released via @ref collect() and
 * the tensor is zeroed.
 *
 * @param[in]  src        Source tensor to view.  Must outlive the
 *                        view.  Must have non-NULL storage.
 * @param[in]  new_shape  New dimension sizes.  Product must equal
 *                        `src->size`.
 * @param[in]  new_ndims  Number of dimensions in @p new_shape.
 * @param[out] status     Receives the operation result.
 *
 * @return View @ref Tensor sharing @p src's storage.
 *
 * @pre  `product(new_shape[0..new_ndims])` must equal `src->size`.
 * @pre  `src->storage` must not be NULL.
 * @pre  @p status must not be `NULL`.
 * @post The returned tensor has `is_view_ = true` and
 *       `is_leaf_ = false`.
 * @post The Rust reference count is incremented by one.
 *
 * @see retain()        Increments the storage reference count.
 * @see is_view()       Query predicate.
 */
Tensor create_view(const Tensor *restrict src, const shape_t new_shape,
                   size_t new_ndims, novaStatus_t *status);

/**
 * @brief Move ownership of tensor resources from src to dst.
 *
 * @details
 * Collects any existing resources in @p dst, then performs a
 * bitwise copy of @p src into @p dst.  @p src is then zeroed
 * (storage, data, grad, grad_fn_ set to NULL) so that a
 * subsequent @ref collect() on @p src is a no-op.
 *
 * @param[in,out] dst  Destination tensor (previous resources are
 *                     freed via @ref collect()).
 * @param[in,out] src  Source tensor (ownership transferred; @p src
 *                     becomes a hollow shell).
 *
 * @pre  @p dst and @p src must not be `NULL`.
 * @post @p dst owns all resources previously held by @p src.
 * @post @p src is in a valid but unallocated state.
 *
 * @see collect()  Frees the destination before the move.
 */
void move_tensor(Tensor *restrict dst, Tensor *restrict src);

/**
 * @brief Recursively release tensor memory and gradients.
 *
 * @details
 * Decrements the reference count of the tensor's storage via
 * @ref release().  When the count reaches zero, the storage is
 * freed with `free()`.  The gradient sub-graph is then traversed
 * and freed recursively via self-recursive calls.
 *
 * Safe to call with `NULL` (no-op).
 *
 * @param[in,out] ten  Tensor to collect.  May be `NULL`.
 *
 * @post @p ten's storage reference count is decremented.
 * @post If the count reaches zero, `storage` and `data` are set
 *       to NULL and `is_allocated_` to `false`.
 * @post The gradient sub-graph is recursively freed.
 *
 * @see release()    Decrements the Rust reference count.
 * @see is_collected()  Query predicate after collection.
 */
void collect(Tensor *ten);

/**
 * @brief Check whether a tensor is contiguous in memory.
 *
 * @details
 * A tensor is contiguous when elements are stored in row-major
 * order without gaps — strides are strictly decreasing by
 * `shape[dim] × item_size` for each dimension.
 *
 * @param[in] ten  Tensor to check.  Must not be `NULL`.
 *
 * @return `true` if the tensor is contiguous, `false` otherwise.
 *
 * @see strides_t    Stride array.
 */
bool is_contiguous(const Tensor *restrict ten);

/**
 * @brief Check whether a tensor is 0-dimensional (scalar).
 *
 * @details
 * A tensor is a scalar when `ndims == 0`, `shape[0] == 0`,
 * `strides[0] == 0`, and `size == 1`.
 *
 * @param[in] ten  Tensor to check.  Must not be `NULL`.
 *
 * @return `true` if the tensor is a scalar, `false` otherwise.
 *
 * @see is_scalar_grad()  Gradient variant.
 * @see create_scalar_tensor()
 */
bool is_scalar(const Tensor *ten);

/**
 * @brief Check whether a gradient tensor is 0-dimensional (scalar).
 *
 * @param[in] grad  Gradient tensor to check.  May be `NULL`.
 *
 * @return `true` if the gradient is a scalar, `false` otherwise
 *         (including when @p grad is `NULL`).
 *
 * @see is_scalar()  Tensor variant.
 */
bool is_scalar_grad(TensorGrad grad);

/**
 * @brief Check whether a tensor's data buffer is properly aligned.
 *
 * @details
 * Alignment requirements differ by device:
 * - **GPU** (`DEVICE_GPU`): 512-byte alignment.
 * - **CPU** (`DEVICE_CPU`): 64-byte alignment.
 * - **META** (`DEVICE_META`): always returns `true`.
 *
 * The check selects the threshold based on @ref Tensor::device and
 * tests `ten->storage->ptr.v` modulo the threshold.
 *
 * @param[in] ten  Tensor to check.  Must not be `NULL`.
 *
 * @return `true` if the data pointer meets the alignment
 *         requirement, `false` otherwise.
 *
 * @pre  `ten->storage` must not be `NULL` (except META).
 *
 * @see is_grad_aligned()  Gradient variant.
 */
bool is_aligned(const Tensor *ten);

/**
 * @brief Check whether a gradient tensor's data buffer is aligned.
 *
 * @details
 * Same alignment logic as @ref is_aligned(): 512-byte for GPU,
 * 64-byte for CPU, and always `true` for META tensors.
 *
 * @param[in] grad  Gradient tensor to check.  May be `NULL`.
 *
 * @return `true` if the gradient data pointer meets the alignment
 *         requirement, `false` otherwise (including `NULL` grad).
 *
 * @see is_aligned()  Tensor variant.
 */
bool is_grad_aligned(TensorGrad grad);

/**
 * @brief Check whether a tensor has been collected (freed).
 *
 * @details
 * A tensor is considered collected when `data.data == NULL`,
 * `storage == NULL`, and `is_allocated_ == false`.  This is the
 * state after @ref collect() has fully released the tensor.
 *
 * @param[in] ten  Tensor to check.  Must not be `NULL`.
 *
 * @return `true` if the tensor has been collected, `false`
 *         otherwise.
 *
 * @see collect()
 * @see is_grad_collected()  Gradient variant.
 */
bool is_collected(const Tensor *ten);

/**
 * @brief Check whether a gradient tensor has been collected.
 *
 * @param[in] grad  Gradient tensor to check.  May be `NULL`.
 *
 * @return `true` if the gradient has been collected (or is
 *         `NULL`), `false` otherwise.
 *
 * @see is_collected()  Tensor variant.
 */
bool is_grad_collected(TensorGrad grad);

/**
 * @brief Check whether a tensor's data buffer has been allocated.
 *
 * @details
 * Returns `true` when all three conditions hold:
 * - `is_allocated_ == true`
 * - `storage != NULL`
 * - `data.data != NULL`
 *
 * This triple-check ensures the tensor was both marked as
 * allocated and actually has a valid pointer.
 *
 * @param[in] ten  Tensor to check.  Must not be `NULL`.
 *
 * @return `true` if the tensor is allocated, `false` otherwise.
 *
 * @see is_grad_allocated()  Gradient variant.
 * @see is_collected()       Inverse check.
 */
bool is_allocated(const Tensor *ten);

/**
 * @brief Check whether a gradient tensor has been allocated.
 *
 * @details
 * If @p grad is `NULL`, returns `false`.  Otherwise, checks the
 * same three conditions as @ref is_allocated(): `is_allocated_`,
 * `storage != NULL`, and `data.data != NULL`.
 *
 * @param[in] grad  Gradient tensor to check.  May be `NULL`.
 *
 * @return `true` if the gradient has a valid backing buffer,
 *         `false` otherwise (including when @p grad is `NULL`).
 *
 * @see is_allocated()  Tensor variant.
 */
bool is_grad_allocated(TensorGrad grad);

/**
 * @brief Check whether a tensor is a view (shares storage with
 *        another tensor).
 *
 * @details
 * A tensor is marked as a view when it was created via
 * @ref create_view().  Views share the underlying storage with
 * their source tensor and have `is_leaf_ == false`.
 *
 * @param[in] ten  Tensor to check.  Must not be `NULL`.
 *
 * @return `true` if the tensor is a view, `false` otherwise.
 *
 * @see create_view()          Constructor for views.
 * @see is_grad_view()         Gradient variant.
 */
bool is_view(const Tensor *ten);

/**
 * @brief Check whether a gradient tensor is a view.
 *
 * @param[in] grad  Gradient tensor to check.  May be `NULL`.
 *
 * @return `true` if the gradient is a view, `false` otherwise
 *         (including when @p grad is `NULL`).
 *
 * @see is_view()  Tensor variant.
 */
bool is_grad_view(TensorGrad grad);

/**
 * @brief Transfer tensor data from GPU device memory to CPU host
 *        memory.
 *
 * @details
 * Validates that @p src is a GPU-resident tensor with device memory
 * and that @p dst is an allocated CPU tensor.  Performs a
 * device-to-host memory transfer via @ref transfer_to().
 *
 * @param[in]  src  Source tensor on GPU.  Must have non-NULL storage
 *                   and be backed by device memory.
 * @param[in,out] dst  Destination tensor on CPU.  Must be allocated.
 *
 * @return @ref novaStatus_t with `novaSuccess` on success, or an
 *         error code describing the failure.
 *
 * @pre  @p src must reside on GPU with device-backed storage.
 * @pre  @p dst must reside on CPU and be allocated.
 *
 * @see transf_tensor_from_host()  Reverse direction.
 * @see transfer_to()              Low-level memory transfer.
 */
novaStatus_t transf_tensor_from_device(const Tensor *restrict src,
                                       Tensor *restrict dst);

/**
 * @brief Transfer tensor data from CPU host memory to GPU device
 *        memory.
 *
 * @details
 * Validates that @p src is an allocated CPU tensor and that @p dst
 * is a GPU-resident tensor with device memory.  Performs a
 * host-to-device memory transfer via @ref transfer_to().
 *
 * @param[in]  src  Source tensor on CPU.  Must be allocated.
 * @param[in,out] dst  Destination tensor on GPU.  Must have non-NULL
 *                     storage and be backed by device memory.
 *
 * @return @ref novaStatus_t with `novaSuccess` on success, or an
 *         error code describing the failure.
 *
 * @pre  @p src must reside on CPU and be allocated.
 * @pre  @p dst must reside on GPU with device-backed storage.
 *
 * @see transf_tensor_from_device()  Reverse direction.
 * @see transfer_to()                Low-level memory transfer.
 */
novaStatus_t transf_tensor_from_host(const Tensor *restrict src,
                                     Tensor *restrict dst);

#ifdef __cplusplus
}
#endif
