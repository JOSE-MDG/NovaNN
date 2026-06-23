/**
 * @file alloc.c
 * @brief Unified memory allocation implementation.
 *
 * @details
 * This module provides @ref safe_allocator(), the single entry point
 * for all buffer allocation in NovaNN.  Allocates through the Rust
 * FFI allocator and returns structured @ref novaStatus_t errors.
 *
 * ## Operating Modes
 *
 * The function operates in one of two modes determined by
 * @p create_storage:
 *
 * 1. **Raw handle mode** (@p create_storage == `false`):
 *    Allocates memory via @ref safe_reserve() and returns the
 *    @ref RustHandle through @p handle.  The caller owns the handle
 *    and must manage its lifetime.
 *
 * 2. **Full tensor mode** (@p create_storage == `true`):
 *    Allocates memory, heap-allocates a @ref TensorStorage, and
 *    attaches it to the tensor pointed to by @p ten.  The tensor's
 *    `storage`, `data`, and `is_allocated_` fields are populated.
 *
 * ## Alignment
 *
 * Buffer alignment is selected automatically:
 * - **GPU**: 512 bytes (coalesced memory access).
 * - **CPU / other**: 64 bytes (cache-line aligned).
 *
 * ## Thread Safety
 *
 * All functions are thread-safe.  The underlying Rust allocator
 * manages its own synchronisation.
 *
 * @see alloc.h      Public API declarations.
 * @see storage.h    RustHandle and TensorStorage definitions.
 * @see device.h     Device enumeration.
 * @see status.h     novaStatus_t error reporting.
 */

#include <ncore/core/alloc.h>
#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/core/storage.h>
#include <ncore/headeronly/macros.h>
#include <ncore/tensor.h>

/**
 * @brief Translate a @ref Device enum to the C-string expected by
 *        the Rust FFI.
 *
 * @param[in] device  The device selection (CPU, GPU, or META).
 *
 * @return `"cpu"` for `DEVICE_CPU`, `"device"` for `DEVICE_GPU`,
 *         or `"none"` as a fallback for META / unknown values.
 */
static const char *map_device2string(Device device) {
  switch (device) {
  case DEVICE_CPU:
    return "cpu";
  case DEVICE_GPU:
    return "device";
  default:
    return "none";
  }
}

/**
 * @brief Allocate device memory and optionally initialize a tensor's
 *        storage.
 *
 * @details
 * Delegates to @ref safe_reserve() through the Rust FFI to obtain a
 * @ref RustHandle.  Alignment is selected automatically: 512 bytes
 * for GPU, 64 bytes otherwise.
 *
 * When @p create_storage is `false`, only a raw @ref RustHandle is
 * returned via @p handle.  When `true`, a @ref TensorStorage is
 * heap-allocated, populated with the handle and data pointer, and
 * attached to the tensor pointed to by @p ten.
 *
 * @param[in]  bytes          Requested allocation size in bytes.
 * @param[in]  device         Target device (`DEVICE_CPU`, `DEVICE_GPU`,
 *                            or `DEVICE_META`).
 * @param[in]  pin_memory     If `true`, request page-locked host memory
 *                            (CPU only).
 * @param[out] handle         Pointer to receive the raw RustHandle.
 *                            Must not be `NULL` when @p create_storage
 *                            is `false`.
 * @param[in,out] ten         Tensor to initialize.  Must not be `NULL`
 *                            when @p create_storage is `true`, and must
 *                            not already be allocated.
 * @param[in]  create_storage If `true`, create a full TensorStorage and
 *                            attach it to @p ten.  If `false`, only
 *                            populate @p handle.
 *
 * @return @ref novaStatus_t with `novaSuccess` on success.
 *
 * @retval novaInvalidPointer  @p ten is `NULL` when @p create_storage is
 *                             `true`, or @p ten is already allocated.
 * @retval novaSuccess         META device or successful allocation.
 * @retval ...                 Forwarded from @ref safe_reserve().
 *
 * @pre  @p bytes must be greater than zero.
 * @pre  Exactly one of @p handle or @p ten must be non-NULL,
 *       determined by @p create_storage.
 *
 * @post On success with @p create_storage `true`, @p ten is marked
 *       as allocated and its @c storage, @c data, and @c is_allocated_
 *       fields are populated.
 *
 * @see safe_reserve()        Low-level Rust FFI allocation.
 * @see get_data_from()       Resolve data pointer from handle.
 * @see TensorStorage         Storage descriptor struct.
 */
novaStatus_t safe_allocator(size_t bytes, Device device, bool pin_memory,
                            RustHandle *handle, Tensor *ten,
                            bool create_storage) {

  novaStatus_t status;
  if (device == DEVICE_META) {
    status.err = novaSuccess;
    status.message = nova_get_error_msg(status.err, NULL);
    return status;
  }
  const size_t align = (device == DEVICE_GPU) ? 512 : 64;

  if (!create_storage) {
    status = safe_reserve(bytes, map_device2string(device), pin_memory, align,
                          handle);
    return status;
  } else {
    if (ten == NULL || (is_allocated(ten))) {
      status.err = novaInvalidPointer;
      status.message = nova_get_error_msg(status.err, NULL);
      return status;
    }

    if (ten->device == DEVICE_META) {
      status.err = novaSuccess;
      status.message = nova_get_error_msg(status.err, NULL);
      return status;
    }

    TensorStorage *storage = (TensorStorage *)malloc(sizeof(TensorStorage));

    if (storage == NULL) {
      status.err = novaInvalidPointer;
      status.message = "Failed to allocate tensor storage descriptor: malloc returned NULL\n";
      return status;
    }

    RustHandle storage_handle = {0};
    status = safe_reserve(bytes, map_device2string(device), pin_memory, align,
                          &storage_handle);

    if (status.err != novaSuccess) {
      return status;
    }

    storage->ptr.data = (unsigned char *)get_data_from(&storage_handle);
    storage->align = storage_handle.align;
    storage->handle = storage_handle;
    storage->size_bytes = storage_handle.size_bytes;

    ten->storage = storage;
    ten->data = storage->ptr;
    ten->is_allocated_ = true;
  }

  status.err = novaSuccess;
  status.message = nova_get_error_msg(status.err, NULL);
  return status;
}
