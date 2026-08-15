/**
 * @file tensor.h
 * @brief Core Tensor type — the fundamental n-dimensional array with
 *        autograd support.
 *
 * @details
 * Defines the @ref Tensor struct, its associated typedefs
 * (@ref shape_t, @ref strides_t, @ref TensorGrad), and the public
 * API for creation, views, memory management, and query predicates.
 * All cache-line aligned fields (@c ALIGN(64)) are padded for optimal
 * SIMD vectorisation.
 *
 * All creation and mutation functions accept an output
 * @ref novaStatus_t pointer that receives the operation result.
 * Callers must check @c status.err after every call; on failure the
 * returned tensor is zeroed and must not be used.
 *
 * @section architecture Architecture
 *
 * A Tensor bundles:
 * @li Metadata — shape, strides, dtype, device, ndims, size, etc.
 * @li Storage — a reference-counted @ref TensorStorage owned by the
 *   Rust allocator.  Multiple tensors (views) can share the same
 *   storage via @c retain() / @c release().
 * @li Autograd — optional gradient tensor (@ref TensorGrad) and
 *   backward function node (@ref BackwardNode) for automatic
 *   differentiation.
 * @li Quantisation — scale and zero-point fields for quantised
 *   inference.
 *
 * @section lifecycle Lifecycle
 *
 * @li 1. Create via @ref create_tensor() or @ref create_scalar_tensor().
 * @li 2. Use in computations; views share storage via @ref create_view().
 * @li 3. Release via @ref collect() — decrements reference count and
 *    recursively frees gradients.
 *
 * @see dtype.h      Data-type definitions and DType_ enum.
 * @see storage.h    TensorStorage, RustHandle, and FFI allocation.
 * @see device.h     Device placement and memory transfers.
 * @see alloc.h      safe_allocator() and typed buffer allocation.
 */

#pragma once

#include <stdbool.h>
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
 * @details
 * Holds up to @ref NOVA_MAX_DIMS elements.  Only the first @c ndims
 * entries are meaningful.  For a scalar, all entries are 0.
 *
 * Example: @c shape = {2, 3, 4} with @c ndims = 3 represents a
 * 2×3×4 tensor.
 */
#ifndef _WIN64
typedef size_t shape_t[NOVA_MAX_DIMS] ALIGN(64);
#else
typedef size_t shape_t[NOVA_MAX_DIMS];
#endif

/**
 * @typedef strides_t
 * @brief Fixed-size array for tensor byte strides.
 *
 * @details
 * @c strides[i] is the number of bytes to advance in the data
 * buffer to reach the next element along dimension @c i.  For a
 * contiguous tensor, @c strides[i] = item_size × product(shape[i+1..]).
 *
 * Example: for a contiguous float32 tensor with shape @c {2, 3, 4},
 * strides are @c {48, 16, 4}.
 */
#ifndef _WIN64
typedef size_t strides_t[NOVA_MAX_DIMS] ALIGN(64);
#else
typedef size_t strides_t[NOVA_MAX_DIMS];
#endif

/**
 * @struct Node
 * @brief Backward pass node for automatic differentiation.
 *
 * @details
 * Opaque type representing a node in the computational graph.
 * Each operation that requires gradient tracking stores a
 * @ref BackwardNode in the output tensor's @c grad_fn_ field.
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
 * @li Shape, strides, and total element count for layout.
 * @li A @ref DType_ and @ref Device_ for type and placement.
 * @li A reference-counted @ref TensorStorage for the backing buffer.
 * @li Optional gradient fields for autograd.
 * @li Quantisation parameters (scale, zero-point).
 *
 * The struct is cache-line aligned (@c ALIGN(64)) to enable
 * efficient SIMD operations on the metadata fields.
 *
 * @section field-groups Field Groups
 *
 * @li Layout — @c shape, @c strides, @c ndims, @c size,
 *     @c item_size, @c offset
 * @li Type/Device_ — @c dtype, @c device
 * @li Storage — @c storage, @c data, @c is_allocated_, @c is_pinned_
 * @li Autograd — @c grad, @c grad_fn_, @c requires_grad_,
 *     @c retain_grad_, @c is_leaf_, @c is_view_
 * @li Quantisation — @c scale_, @c zero_point_
 * @li Mutation track — @c version_
 *
 * @see shape_t      Fixed-size shape array.
 * @see strides_t    Fixed-size strides array.
 * @see TensorGrad   Gradient tensor alias.
 */
#ifndef _WIN64
struct ALIGN(64) Tensor {
#else
struct Tensor {
#endif
  size_t item_size;    ///< Bytes per element (from @ref dtype_size).
  size_t offset;       ///< Byte offset into the storage buffer.
  size_t ndims;        ///< Number of dimensions (0 for scalars).
  size_t size;         ///< Total element count (product of shape).
  size_t logical_size; ///< Unpacked element count (individual values).
  DType_ dtype;        ///< Element data type (@ref DType_ enum).
  Device_ device;      ///< Placement device (@ref Device_ enum).
  shape_t shape;       ///< Dimension sizes (up to @ref NOVA_MAX_DIMS).
  strides_t strides;   ///< Byte strides per dimension.
  TensorStorage
      *storage;    ///< Reference-counted backing buffer (nullptr for META).
  data_ptr data;   ///< Typed pointer into @c storage (offset-adjusted).
  TensorGrad grad; ///< Gradient tensor (nullptr if no gradient tracked).
  BackwardNode grad_fn_; ///< Backward graph node (nullptr for leaves).
  bool requires_grad_;   ///< If @c true, track gradients during backward.
  bool retain_grad_;     ///< If @c true, retain gradient for non-leaf nodes.
  bool is_view_;         ///< If @c true, shares storage with another tensor.
  bool is_leaf_;         ///< If @c true, leaf node in the computation graph.
  bool is_allocated_;    ///< If @c true, @c storage and @c data are valid.
  bool is_pinned_;       ///< If @c true, page-locked host memory.
  float scale_;          ///< Quantisation scale (1.0 if not quantised).
  int32_t zero_point_;   ///< Quantisation zero point (0 if not quantised).
  int64_t version_;      ///< Mutation counter for in-place alias detection.
};

/** @brief Gradient tensor type alias — a pointer to a @ref Tensor. */
typedef Tensor *TensorGrad;

/**
 * @brief Create a fully allocated n-dimensional tensor.
 *
 * @details
 * Zero-initialises the @ref Tensor struct, copies the shape, computes
 * @c size and @c strides, and allocates a data buffer via
 * @ref safe_allocator().  When @p requires_grad is @c true, an
 * unallocated gradient tensor is created in @c grad with the same
 * shape, dtype, and device as the parent.
 *
 * For @c DEVICE_META, storage is left @c nullptr and @c is_allocated_ is
 * set to @c false.  The Rust allocator is never invoked.
 *
 * On any allocation or gradient-creation failure, the tensor is
 * zeroed and the caller receives a non-success status through
 * @p status.
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
 * @param[out] status        Receives the operation result.  On
 *                           success @c err is @ref novaSuccess.
 *
 * @return Initialised @ref Tensor with valid backing storage, or a
 *         META tensor with nullptr storage.
 *
 * @pre  @p ndims must be in [1, @ref NOVA_MAX_DIMS].  Scalars
 *       (@p ndims == 0) are out of contract here — use
 *       @ref create_scalar_tensor().
 * @pre  @c product(shape[0..ndims]) must be > 0 for non-META.
 * @pre  @p status must not be @c nullptr.
 * @post On success, @c is_allocated_ == true (unless META).
 * @post If @c requires_grad, @c grad points to an unallocated tensor.
 *
 * @see create_scalar_tensor()   Scalar (0-D) variant.
 * @see create_tensor_like()     Clone metadata from an existing tensor.
 * @see safe_allocator()         Underlying allocation.
 */
Tensor create_tensor(const shape_t shape, DType_ dtype, Device_ device,
                     bool requires_grad, bool pin_memory, size_t ndims,
                     novaStatus_t *status);

/**
 * @brief Create a fully allocated 0-dimensional (scalar) tensor.
 *
 * @details
 * Allocates a single-element data buffer.  Shape and strides are
 * zeroed, @c ndims is set to 0, and @c size is set to 1.
 * Allocation is performed via @ref safe_allocator().  When
 * @p requires_grad is @c true, an unallocated scalar gradient tensor
 * is created in @c grad.
 *
 * For @c DEVICE_META, storage is left @c nullptr and @c is_allocated_ is
 * set to @c false.  On failure, the tensor is zeroed and the error
 * is reported through @p status.
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
 * @return Initialised scalar @ref Tensor with backing storage.
 *
 * @pre  @p status must not be @c nullptr.
 * @post On success, @c is_allocated_ == true (unless META).
 *
 * @see create_tensor()        N-dimensional variant.
 * @see is_scalar()            Query predicate.
 */
Tensor create_scalar_tensor(DType_ dtype, Device_ device, bool requires_grad,
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
 *                    be @c nullptr.
 * @param[out] status Receives the operation result.
 *
 * @return New @ref Tensor with matching metadata and allocation
 *         state.
 *
 * @pre  @p ten must not be @c nullptr.
 * @pre  @p status must not be @c nullptr.
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
 *                        view.  Must have non-nullptr storage.
 * @param[in]  new_shape  New dimension sizes.  Product must equal
 *                        @c src->size.
 * @param[in]  new_ndims  Number of dimensions in @p new_shape.
 * @param[out] status     Receives the operation result.
 *
 * @return View @ref Tensor sharing @p src's storage.
 *
 * @pre  @c product(new_shape[0..new_ndims]) must equal @c src->size.
 * @pre  @c src->storage must not be nullptr.
 * @pre  @p status must not be @c nullptr.
 * @post The returned tensor has @c is_view_ = true and
 *       @c is_leaf_ = false.
 * @post The Rust reference count is incremented by one.
 *
 * @see retain()    Increments the storage reference count.
 * @see is_view()   Query predicate.
 */
Tensor create_view(const Tensor *restrict src, const shape_t new_shape,
                   size_t new_ndims, novaStatus_t *status);

/**
 * @brief Move ownership of tensor resources from src to dst.
 *
 * @details
 * Collects any existing resources in @p dst, then performs a
 * bitwise copy of @p src into @p dst.  @p src is then zeroed
 * (storage, data, grad, grad_fn_ set to nullptr) so that a
 * subsequent @ref collect() on @p src is a no-op.
 *
 * @param[in,out] dst  Destination tensor (previous resources are
 *                     freed via @ref collect()).
 * @param[in,out] src  Source tensor (ownership transferred; @p src
 *                     becomes a hollow shell).
 *
 * @pre  @p dst and @p src must not be @c nullptr.
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
 * freed with @c free().  The gradient sub-graph is then traversed
 * and freed recursively via self-recursive calls.
 *
 * Safe to call with @c nullptr (no-op).
 *
 * @param[in,out] ten  Tensor to collect.  May be @c nullptr.
 *
 * @post @p ten's storage reference count is decremented.
 * @post If the count reaches zero, @c storage and @c data are set
 *       to nullptr and @c is_allocated_ to @c false.
 * @post The gradient sub-graph is recursively freed.
 *
 * @see release()       Decrements the Rust reference count.
 * @see is_collected()  Query predicate after collection.
 */
void collect(Tensor *ten);

/**
 * @brief Check whether a tensor is contiguous in memory.
 *
 * @details
 * A tensor is contiguous when elements are stored in row-major
 * order without gaps — strides are strictly decreasing by
 * @c shape[dim] × item_size for each dimension.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the tensor is contiguous, @c false otherwise.
 *
 * @see strides_t    Stride array.
 */
bool is_contiguous(const Tensor *restrict ten);

/**
 * @brief Ensure a tensor's data is stored contiguously.
 *
 * @details
 * Returns a tensor whose elements are laid out contiguously in
 * row-major order (no gaps between elements).  If @p ten is already
 * contiguous, a view sharing its storage is returned via
 * @ref create_view() — no data is copied.  Otherwise, a new tensor
 * is allocated with the same metadata (dtype, device,
 * requires_grad, pin_memory) and the elements are copied into it.
 *
 * Scalars are handled via @ref create_scalar_tensor() so the result
 * keeps @c ndims == 0.
 *
 * @param[in]  ten    Tensor to make contiguous.  Must not be
 *                    @c nullptr.
 * @param[out] status Receives the operation result.
 *
 * @return Contiguous @ref Tensor with the same shape and metadata
 *         as @p ten, or a zeroed tensor on failure.
 *
 * @pre  @p ten must not be @c nullptr.
 * @pre  @p status must not be @c nullptr.
 * @post On success, @ref is_contiguous() returns @c true for the
 *       result.
 *
 * @see is_contiguous()  Query predicate.
 * @see create_view()    Fast path for already-contiguous tensors.
 */
Tensor contiguous(const Tensor *restrict ten, novaStatus_t *status);

/**
 * @brief Check whether a tensor is 0-dimensional (scalar).
 *
 * @details
 * A tensor is a scalar when @c ndims == 0, @c shape[0] == 0,
 * @c strides[0] == 0, and @c size == 1.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the tensor is a scalar, @c false otherwise.
 *
 * @see is_scalar_grad()    Gradient variant.
 * @see create_scalar_tensor()
 */
bool is_scalar(const Tensor *ten);

/**
 * @brief Check whether a gradient tensor is 0-dimensional (scalar).
 *
 * @param[in] grad  Gradient tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the gradient is a scalar, @c false otherwise.
 *
 * @pre  @p grad must not be @c nullptr.
 *
 * @see is_scalar()  Tensor variant.
 */
bool is_scalar_grad(TensorGrad grad);

/**
 * @brief Check whether a tensor's data buffer is properly aligned.
 *
 * @details
 * Alignment requirements differ by device:
 * @li GPU (@c DEVICE_GPU): 512-byte alignment.
 * @li CPU (@c DEVICE_CPU): 64-byte alignment.
 * @li META (@c DEVICE_META): always returns @c true.
 *
 * The check selects the threshold based on @ref Tensor::device and
 * tests @c ten->storage->ptr.v modulo the threshold.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the data pointer meets the alignment
 *         requirement, @c false otherwise.
 *
 * @pre  @c ten->storage must not be @c nullptr (except META).
 *
 * @see is_grad_aligned()  Gradient variant.
 */
bool is_aligned(const Tensor *ten);

/**
 * @brief Check whether a gradient tensor's data buffer is aligned.
 *
 * @details
 * Same alignment logic as @ref is_aligned(): 512-byte for GPU,
 * 64-byte for CPU, and always @c true for META tensors.
 *
 * @param[in] grad  Gradient tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the gradient data pointer meets the alignment
 *         requirement, @c false otherwise.
 *
 * @pre  @p grad must not be @c nullptr.
 *
 * @see is_aligned()  Tensor variant.
 */
bool is_grad_aligned(TensorGrad grad);

/**
 * @brief Check whether a tensor has been collected (freed).
 *
 * @details
 * A tensor is considered collected when @c data.data == nullptr,
 * @c storage == nullptr, and @c is_allocated_ == false.  This is the
 * state after @ref collect() has fully released the tensor.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the tensor has been collected, @c false
 *         otherwise.
 *
 * @see collect()           Tensor collection.
 * @see is_grad_collected() Gradient variant.
 */
bool is_collected(const Tensor *ten);

/**
 * @brief Check whether a gradient tensor has been collected.
 *
 * @param[in] grad  Gradient tensor to check.  May be @c nullptr.
 *
 * @return @c true if the gradient has been collected (or is
 *         @c nullptr), @c false otherwise.
 *
 * @see is_collected()  Tensor variant.
 */
bool is_grad_collected(TensorGrad grad);

/**
 * @brief Check whether a tensor's data buffer has been allocated.
 *
 * @details
 * Returns @c true when all three conditions hold:
 * @li @c is_allocated_ == true
 * @li @c storage != nullptr
 * @li @c data.data != nullptr
 *
 * This triple-check ensures the tensor was both marked as
 * allocated and actually has a valid pointer.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the tensor is allocated, @c false otherwise.
 *
 * @see is_grad_allocated()  Gradient variant.
 * @see is_collected()       Inverse check.
 */
bool is_allocated(const Tensor *ten);

/**
 * @brief Check whether a gradient tensor has been allocated.
 *
 * @details
 * If @p grad is @c nullptr, returns @c false.  Otherwise, checks the
 * same three conditions as @ref is_allocated(): @c is_allocated_,
 * @c storage != nullptr, and @c data.data != nullptr.
 *
 * @param[in] grad  Gradient tensor to check.  May be @c nullptr.
 *
 * @return @c true if the gradient has a valid backing buffer,
 *         @c false otherwise (including when @p grad is @c nullptr).
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
 * their source tensor and have @c is_leaf_ == false.
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the tensor is a view, @c false otherwise.
 *
 * @see create_view()    Constructor for views.
 * @see is_grad_view()   Gradient variant.
 */
bool is_view(const Tensor *ten);

/**
 * @brief Check whether a gradient tensor is a view.
 *
 * @param[in] grad  Gradient tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the gradient is a view, @c false otherwise.
 *
 * @pre  @p grad must not be @c nullptr.
 *
 * @see is_view()  Tensor variant.
 */
bool is_grad_view(TensorGrad grad);

/**
 * @brief Check whether a tensor's data resides in GPU device memory.
 *
 * @details
 * A tensor is considered to be "on device" when all three conditions
 * hold:
 * @li @c is_allocated(ten) — the tensor has valid backing storage.
 * @li @c ten->device == @c DEVICE_GPU — the tensor is GPU-resident.
 * @li @c is_device_memory_handle(&ten->storage->handle) — the storage
 *   is backed by device-managed memory.
 *
 * This distinguishes genuine VRAM-resident tensors from CPU tensors
 * (including pinned host tensors, whose handles are also device-managed
 * but are not located in VRAM).
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the tensor is allocated, GPU-resident, and
 *         device-backed, @c false otherwise.
 *
 * @see on_host()                    Complementary check.
 * @see is_allocated()               Allocation precondition.
 * @see is_device_memory_handle()    Storage backing check.
 */
bool on_device(const Tensor *ten);

/**
 * @brief Check whether a tensor's data resides in CPU host memory.
 *
 * @details
 * A tensor is considered to be "on host" when all three conditions
 * hold:
 * @li @c is_allocated(ten) — the tensor has valid backing storage.
 * @li @c ten->device == @c DEVICE_CPU — the tensor is host-resident.
 * @li @c !is_device_memory_handle(&ten->storage->handle) — the storage
 *   is NOT backed by device-managed memory (i.e., plain host
 *   allocation, not pinned memory).
 *
 * @param[in] ten  Tensor to check.  Must not be @c nullptr.
 *
 * @return @c true if the tensor is allocated, host-resident, and
 *         backed by plain host memory, @c false otherwise.
 *
 * @see on_device()                   Complementary check.
 * @see is_allocated()                Allocation precondition.
 * @see is_device_memory_handle()     Storage backing check.
 */
bool on_host(const Tensor *ten);

/**
 * @brief Transfer tensor data from GPU device memory to CPU host
 *        memory.
 *
 * @details
 * Validates that @p src is a GPU-resident tensor with device memory
 * and that @p dst is an allocated CPU tensor.  Performs a
 * device-to-host memory transfer via @ref transfer_to().
 *
 * @param[in]  src  Source tensor on GPU.  Must have non-nullptr storage
 *                   and be backed by device memory.
 * @param[in,out] dst  Destination tensor on CPU.  Must be allocated.
 *
 * @return @ref novaStatus_t with @c novaSuccess on success, or an
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
 * @param[in,out] dst  Destination tensor on GPU.  Must have non-nullptr
 *                     storage and be backed by device memory.
 *
 * @return @ref novaStatus_t with @c novaSuccess on success, or an
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
