/**
 * @file alloc.h
 * @brief Unified memory allocation API for NovaNN tensors.
 *
 * @details
 * Provides @ref safe_allocator(), the single entry point for all
 * buffer allocation in NovaNN.  Allocates memory through the Rust
 * FFI allocator, returning structured error status.
 *
 * Two operating modes are supported via the @p create_storage flag:
 *
 * @li @c false — @p create_storage — valid @p handle — @c nullptr
 *     @p ten — Raw RustHandle allocation only
 * @li @c true — @p create_storage — @c nullptr @p handle — valid
 *     @p ten — Full TensorStorage + Tensor init
 *
 * @see alloc.c      Implementation
 * @see storage.h    RustHandle and TensorStorage definitions.
 * @see status.h     novaStatus_t error reporting.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/core/storage.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Allocate device memory and optionally initialize a tensor's
 *        storage.
 *
 * @details
 * Delegates to @ref reserve() through the Rust FFI to obtain a
 * @ref RustHandle.  Alignment is selected automatically: 512 bytes
 * for GPU, 64 bytes otherwise.
 *
 * When @p create_storage is @c false, only a raw @ref RustHandle is
 * returned via @p handle.  When @c true, a @ref TensorStorage is
 * heap-allocated, populated with the handle and data pointer, and
 * attached to the tensor pointed to by @p ten.
 *
 * @param[in]  bytes          Requested allocation size in bytes.
 * @param[in]  device         Target device (@c DEVICE_CPU, @c DEVICE_GPU,
 *                            or @c DEVICE_META).
 * @param[in]  pin_memory     If @c true, request page-locked host memory
 *                            (CPU only).
 * @param[out] handle         Pointer to receive the raw RustHandle.
 *                            Must not be @c nullptr when @p create_storage
 *                            is @c false.
 * @param[in,out] ten         Tensor to initialize.  Must not be @c nullptr
 *                            when @p create_storage is @c true, and must
 *                            not already be allocated.
 * @param[in]  create_storage If @c true, create a full TensorStorage and
 *                            attach it to @p ten.  If @c false, only
 *                            populate @p handle.
 *
 * @return @ref novaStatus_t with @c novaSuccess on success.
 *
 * @retval novaInvalidPointer  @p ten is @c nullptr when @p create_storage is
 *                             @c true, or @p ten is already allocated.
 * @retval novaSuccess         META device or successful allocation.
 * @retval ...                 Forwarded from @ref reserve().
 *
 * @pre  @p bytes must be greater than zero.
 * @pre  Exactly one of @p handle or @p ten must be non-nullptr,
 *       determined by @p create_storage.
 *
 * @post On success with @p create_storage @c true, @p ten is marked
 *       as allocated and its @c storage, @c data, and @c is_allocated_
 *       fields are populated.
 *
 * @see reserve()             Rust-backed allocation.
 * @see get_data_from()       Resolve data pointer from handle.
 * @see TensorStorage         Storage descriptor struct.
 */
novaStatus_t safe_allocator(size_t bytes, Device_ device, bool pin_memory,
                            RustHandle *handle, Tensor *ten,
                            bool create_storage);

#ifdef __cplusplus
}
#endif
