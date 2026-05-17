/**
 * @file tensor.h
 * @brief Core Tensor type — the fundamental n-dimensional array with autograd.
 *
 * @details
 * Defines the Tensor struct, its associated typedefs (shape_t, strides_t,
 * TensorGrad), and the public API for creation, views, memory management,
 * and scalar/alignment/collected query predicates.  All cache-line aligned
 * fields (ALIGN(64)) are padded for optimal SIMD vectorization.
 *
 * @see dtype.h, storage.h, simd.h
 */

#pragma once

#include <ncore/backend.h>
#include <ncore/device.h>
#include <ncore/dtype.h>
#include <ncore/macros.h>
#include <ncore/storage.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Backward pass node for automatic differentiation.
 */
struct Node;
typedef struct Node Node;
typedef Node *BackwardNode;

/**
 * @brief Multi-dimensional array with automatic differentiation support.
 *
 * Core tensor structure used throughout NovaNN for storing n-dimensional
 * data with support for:
 * - Multiple data types (float, int, quantized)
 * - Multiple devices (CPU, GPU)
 * - Automatic gradient computation
 * - Quantization (scale/zero_point)
 */
struct Tensor;
typedef struct Tensor Tensor;

///< Gradient tensor alias.
typedef Tensor *TensorGrad;

/**
 * @brief Fixed-size array for tensor shape dimensions.
 *
 * Contains the size of each dimension up to NOVA_MAX_DIMS.
 * Example: shape[0]=2, shape[1]=3, shape[2]=4 represents a 2x3x4 tensor.
 */
typedef size_t shape_t[NOVA_MAX_DIMS] ALIGN(64);

/**
 * @brief Fixed-size array for tensor stride values.
 *
 * Stride represents bytes to skip in the data buffer to reach the next
 * element along each dimension. For a contiguous tensor, stride[i] equals
 * the product of shape[i+1] * item_size.
 */
typedef size_t strides_t[NOVA_MAX_DIMS] ALIGN(64);

/**
 * @brief Tensor structure.
 *
 * @note Fields with ALIGN(64) are cache-line aligned for SIMD operations.
 */
struct ALIGN(64) Tensor {
  size_t item_size;       ///< Size of each element in bytes
  size_t offset;          ///< Offset into the storage buffer
  size_t ndims;           ///< Number of dimensions
  size_t size;            ///< Total number of elements (product of shape)
  DType_ dtype;           ///< Data type (e.g., float, int32)
  Device device;          ///< Device where tensor resides (CPU, GPU)
  shape_t shape;          ///< Size of each dimension
  strides_t strides;      ///< Stride for each dimension (bytes)
  TensorStorage *storage; ///< Pointer to underlying storage buffer
  data_ptr data;   ///< Pointer to data buffer (relative to storage + offset)
  TensorGrad grad; ///< Gradient tensor (null if no gradient)
  BackwardNode grad_fn_; ///< Backward function node for gradient computation
  bool requires_grad_;   ///< If true, track gradients during backward pass
  bool retain_grad_;     ///< If true, retain gradient for non-leaf tensors
  bool is_view_;         ///< If true, tensor is a view of another tensor
  bool is_leaf_;         ///< If true, tensor is a leaf in computation graph
  bool is_allocated_;    ///< If true, tensor is allocated in memory
  float scale_;          ///< Quantization scale (0 if not quantized)
  int32_t zero_point_;   ///< Quantization zero point (0 if not quantized)
  int64_t version_;      ///< Mutation counter; incremented on every in-place
                    ///< operation for version-tracking and alias detection.
};

/**
 * @brief Create a fully allocated tensor.
 *
 * Initialises shape, strides, size, and dtype metadata, allocates a
 * data buffer on the specified device, and optionally creates an
 * unallocated gradient tensor for autograd tracking.
 *
 * @param shape         Dimension sizes.
 * @param dtype         Element data type.
 * @param device        Target device (CPU, GPU, or META).
 * @param requires_grad If true, an unallocated gradient tensor is created.
 * @param ndims         Number of dimensions.
 * @return Initialised Tensor with backing storage (or NULL storage for
 *         DEVICE_META).
 */
Tensor create_tensor(const shape_t shape, DType_ dtype, Device device,
                     bool requires_grad, size_t ndims);

/**
 * @brief Create a fully allocated 0-dimensional (scalar) tensor.
 *
 * Allocates a single-element data buffer on the specified device.
 * Shape and strides are zeroed and ndims is set to 0.
 *
 * @param dtype         Element data type.
 * @param device        Target device (CPU, GPU, or META).
 * @param requires_grad If true, an unallocated gradient tensor is created.
 * @return Initialised scalar Tensor with backing storage.
 */
Tensor create_scalar_tensor(DType_ dtype, Device device, bool requires_grad);

/**
 * @brief Create a view of an existing tensor with a new shape.
 *
 * Shares the underlying storage (incrementing the reference count),
 * recomputes strides, and marks the result as a non-leaf view.
 *
 * @param src       Source tensor to view (must outlive the view).
 * @param new_shape New dimension sizes.  Product must equal src->size.
 * @param new_ndims Number of dimensions in the new shape.
 * @return View tensor sharing src's storage.
 */
Tensor create_view(const Tensor *restrict src, const shape_t new_shape,
                   size_t new_ndims);

/**
 * @brief Check whether a tensor's data buffer is contiguous.
 *
 * A tensor is contiguous when elements are stored in row-major order
 * without gaps, i.e. strides are strictly decreasing by shape[dim].
 *
 * @param ten Tensor to check.
 * @return true if the tensor is contiguous, false otherwise.
 */
bool is_contiguous(const Tensor *restrict ten);

/**
 * @brief Move ownership of tensor resources from src to dst.
 *
 * Collects dst, transfers src's contents, then resets src to a hollow
 * shell (storage/data/grad set to NULL).
 *
 * @param dst Destination tensor (previous resources are freed).
 * @param src Source tensor (ownership transferred; src invalidated).
 */
void move_tensor(Tensor *restrict dst, Tensor *restrict src);

/**
 * @brief Recursively release tensor memory and gradients.
 *
 * Decrements the storage reference count, freeing it when it reaches
 * zero, then recursively frees the gradient sub-graph.
 *
 * @param ten Tensor to collect (may be NULL).
 */
void collect(Tensor *ten);

/**
 * @brief Check whether a tensor is 0-dimensional (scalar).
 *
 * @param ten Tensor to check.
 * @return true if the tensor is a scalar, false otherwise.
 */
bool is_scalar(const Tensor *ten);

/**
 * @brief Check whether a gradient tensor is 0-dimensional (scalar).
 *
 * @param grad Gradient tensor to check.
 * @return true if the gradient is a scalar, false otherwise.
 */
bool is_scalar_grad(TensorGrad grad);

/**
 * @brief Check whether a tensor's data buffer is 64-byte aligned.
 *
 * 64-byte alignment is required for optimal SIMD vectorization.
 *
 * @param ten Tensor to check.
 * @return true if the data pointer is 64-byte aligned, false otherwise.
 */
bool is_aligned(const Tensor *ten);

/**
 * @brief Check whether a gradient tensor's data buffer is 64-byte aligned.
 *
 * @param grad Gradient tensor to check.
 * @return true if the gradient data pointer is 64-byte aligned, false
 *         otherwise.
 */
bool is_grad_aligned(TensorGrad grad);

/**
 * @brief Check whether a tensor has been collected (freed).
 *
 * @param ten Tensor to check.
 * @return true if the tensor has been collected, false otherwise.
 */
bool is_collected(const Tensor *ten);

/**
 * @brief Check whether a gradient tensor has been collected (freed).
 *
 * @param grad Gradient tensor to check.
 * @return true if the gradient has been collected, false otherwise.
 */
bool is_grad_collected(TensorGrad grad);

/**
 * @brief Check whether a tensor's data buffer has been allocated.
 *
 * @param ten Tensor to check.
 * @return true if the tensor has a valid backing storage, false otherwise.
 */
bool is_allocated(const Tensor *ten);

/**
 * @brief Check whether a gradient tensor's data buffer has been allocated.
 *
 * Returns false if grad is NULL.
 *
 * @param grad Gradient tensor to check.
 * @return true if the gradient has a valid backing storage, false otherwise.
 */
bool is_grad_allocated(TensorGrad grad);

/**
 * @brief Create a tensor with the same shape, dtype, and device as another.
 *
 * Inspects the source tensor and produces a new tensor with identical
 * metadata.  If the source is allocated, the result is also allocated;
 * otherwise an unallocated tensor is returned.  Scalar tensors are
 * handled specially via create_scalar_tensor /
 * create_unallocated_scalar_tensor.
 *
 * @param ten Source tensor to copy metadata from.
 * @return New tensor with matching shape, dtype, device, and requires_grad.
 */
Tensor create_tensor_like(const Tensor *ten);

#ifdef __cplusplus
}
#endif
