/**
 * @file alloc.c
 * @brief Unified memory allocation implementation.
 *
 * @details
 * Provides @ref safe_allocator(), the single entry point for all
 * buffer allocation in NovaNN.  Allocates through the Rust FFI
 * allocator and returns structured @ref novaStatus_t errors.  Thread
 * safety comes from the Rust allocator, which manages its own
 * synchronization.
 *
 * One function, two modes (@p create_storage): raw handle mode
 * allocates via @ref reserve() and returns the @ref RustHandle
 * through @p handle, with the caller owning its lifetime; full
 * tensor mode creates the @ref TensorStorage, attaches it to @p ten,
 * and populates the @c storage, @c data, and @c is_allocated_
 * fields.
 *
 * @see alloc.h      Public API declarations.
 * @see storage.h    RustHandle and TensorStorage definitions.
 * @see device.h     Device_ enumeration.
 * @see status.h     novaStatus_t error reporting.
 */

#include <string.h>

#include <ncore/core/alloc.h>
#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/core/storage.h>
#include <ncore/headeronly/macros.h>
#include <ncore/tensor.h>

/**
 * @brief Translate a @ref Device_ enum to the C-string expected by
 *        the Rust FFI.
 *
 * @param[in] device  The device selection (CPU, GPU, or META).
 *
 * @return @c "cpu" for @c DEVICE_CPU, @c "device" for @c DEVICE_GPU,
 *         or @c "none" as a fallback for META / unknown values.
 */
static inline const char *map_device2string(Device_ device) {
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
 * Resolves the theoretical @p bytes to its size class through
 * @ref resolve_rounded_bytes() and delegates to @ref reserve() through
 * the Rust FFI to obtain a @ref RustHandle carrying both the theoretical
 * and the real reservation sizes.  Alignment is selected automatically:
 * 512 bytes for GPU, 64 bytes otherwise.
 *
 * When @p create_storage is @c false, only a raw @ref RustHandle is
 * returned via @p handle.  When @c true, a @ref TensorStorage is
 * heap-allocated, populated with the handle and data pointer, and
 * attached to the tensor pointed to by @p ten.
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
 * @return @ref novaStatus_t with @c novaSuccess on success.
 *
 * @retval novaInvalidPointer  @p ten is @c nullptr when @p create_storage is
 *                             @c true, or @p ten is already allocated.
 * @retval novaSuccess         META device or successful allocation.
 * @retval ...                 Forwarded from @ref reserve() or the rounding.
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
                            bool create_storage) {

  novaStatus_t status;
  if (!create_storage && handle == nullptr) {
    status.err = novaInvalidHandle;
    status.message = nova_get_error_msg(status.err, nullptr);
    return status;
  }
  if (device == DEVICE_META) {
    if (!create_storage) {
      *handle = (RustHandle){0};
    } else if (ten == nullptr || is_allocated(ten)) {
      status.err = novaInvalidPointer;
      status.message = nova_get_error_msg(status.err, nullptr);
      return status;
    }
    status.err = novaSuccess;
    status.message = nova_get_error_msg(status.err, nullptr);
    return status;
  }

  if (bytes == 0) {
    status.err = novaInvalidValue;
    status.message = nova_get_error_msg(status.err, nullptr);
    return status;
  }

  const bool is_device = (device == DEVICE_GPU);
  const size_t rounded =
      resolve_rounded_bytes(bytes, is_device, pin_memory, &status);
  if (status.err != novaSuccess) {
    return status;
  }

  /**
   * @note
   * Specifying whether a tensor is aligned on the device
   * or pinned in host memory is irrelevant, since the driver
   * does not specify an alignment parameter.
   */
  const size_t align = 64;
  if (!create_storage) {
    *handle = reserve(rounded, bytes, map_device2string(device), pin_memory,
                      align, &status);
    return status;
  } else {
    if (ten == nullptr || is_allocated(ten)) {
      status.err = novaInvalidPointer;
      status.message = nova_get_error_msg(status.err, nullptr);
      return status;
    }

    if (ten->device == DEVICE_META) {
      status.err = novaSuccess;
      status.message = nova_get_error_msg(status.err, nullptr);
      return status;
    }

    auto storage = (TensorStorage *)malloc(sizeof(TensorStorage));

    if (storage == nullptr) {
      status.err = novaInvalidPointer;
      status.message = "Failed to allocate tensor storage descriptor: malloc "
                       "returned nullptr\n";
      return status;
    }

    RustHandle storage_handle = {};
    storage_handle = reserve(rounded, bytes, map_device2string(device),
                             pin_memory, align, &status);

    if (status.err != novaSuccess) {
      free(storage);
      return status;
    }

    storage->ptr.data = (unsigned char *)get_data_from(&storage_handle);
    storage->handle = storage_handle;

    ten->storage = storage;
    ten->data = storage->ptr;
    ten->is_allocated_ = true;
  }

  status.err = novaSuccess;
  status.message = nova_get_error_msg(status.err, nullptr);
  return status;
}
