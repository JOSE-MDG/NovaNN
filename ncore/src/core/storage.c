/**
 * @file storage.c
 * @brief Safe wrappers for Rust FFI memory management.
 *
 * @details
 * Provides @ref safe_reserve() and @ref safe_resize(), which wrap
 * the raw FFI functions @ref reserve() and @ref resize() with
 * structured @ref novaStatus_t error handling.  These functions
 * eliminate the need for callers to validate handles manually.
 *
 * @see storage.h    Public API declarations.
 * @see status.h     novaStatus_t error reporting.
 */

#include <string.h>

#include <ncore/core/status.h>
#include <ncore/core/storage.h>
#include <ncore/tensor.h>

/**
 * @brief Allocate a buffer with structured error handling.
 *
 * @details
 * Wraps @ref reserve() and validates the result with
 * @ref is_valid_handle().  On failure, retrieves the error message
 * from @ref get_last_reserve_error().
 *
 * @param[in]  bytes      Requested size in bytes.
 * @param[in]  device     Target device: @c "cpu" or @c "device".
 * @param[in]  pin_memory If @c true, allocate page-locked host memory.
 * @param[in]  align      Required alignment in bytes (power of two).
 * @param[out] handle     Pointer to receive the allocated handle.
 *
 * @return @ref novaStatus_t with @c novaSuccess on success, or
 *         @ref novaReserveError on failure.
 */
novaStatus_t safe_reserve(size_t bytes, const char *device, bool pin_memory,
                          size_t align, RustHandle *handle) {
  novaStatus_t status;

  *handle = reserve(bytes, device, pin_memory, align);
  if (!is_valid_handle(handle)) {
    status.err = novaReserveError;
    status.message = get_last_reserve_error();
    return status;
  }

  status.err = novaSuccess;
  status.message = nova_get_error_msg(status.err, nullptr);

  return status;
}

/**
 * @brief Resize an allocation with structured error handling.
 *
 * @details
 * Wraps @ref resize() and validates the result with
 * @ref is_valid_handle().  On failure, retrieves the error message
 * from @ref get_last_reserve_error().
 *
 * @param[in,out] handle   Pointer to the @ref RustHandle to resize.
 * @param[in]     new_size New size in bytes.
 *
 * @return @ref novaStatus_t with @c novaSuccess on success, or
 *         @ref novaOutOfMemory / @ref novaResizeError on failure.
 */
novaStatus_t safe_resize(RustHandle *handle, size_t new_size) {
  novaStatus_t status;

  if (!resize(handle, new_size)) {
    status.err = novaOutOfMemory;
    status.message = get_last_reserve_error();
  }

  if (!is_valid_handle(handle)) {
    status.err = novaResizeError;
    status.message = get_last_reserve_error();
    return status;
  }

  status.err = novaSuccess;
  status.message = nova_get_error_msg(status.err, nullptr);
  return status;
}
