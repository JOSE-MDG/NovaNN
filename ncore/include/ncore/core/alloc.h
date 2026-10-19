/**
 * @file alloc.h
 * @brief Unified memory allocation API for NovaNN tensors.
 *
 * Every buffer allocation goes through @ref safe_allocator(), which
 * reserves memory through the Rust FFI allocator.
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
 * Delegates to @ref reserve() through the Rust FFI to obtain a
 * @ref RustHandle carrying both the theoretical and the real reservation
 * sizes.  The theoretical size is first resolved to its size class
 * (device and pinned reservations round up while plain host is identity).
 *
 * @param[in]  bytes          Theoretical allocation size in bytes.
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
 * @return @ref novaStatus_t with @c novaSuccess on success, or the error
 *         forwarded from @ref reserve().  With @p create_storage true,
 *         @c novaInvalidPointer when @p ten is nullptr or already allocated.
 *
 * @pre  @p bytes must be greater than zero.
 * @pre  Exactly one of @p handle or @p ten must be non-nullptr,
 *       determined by @p create_storage.
 *
 * @post On success with @p create_storage @c true, @p ten is marked
 *       as allocated and its @c storage, @c data, and @c is_allocated_
 *       fields are populated.
 */
novaStatus_t safe_allocator(size_t bytes, Device_ device, bool pin_memory,
                            RustHandle *handle, Tensor *ten,
                            bool create_storage);

#ifdef __cplusplus
}
#endif
