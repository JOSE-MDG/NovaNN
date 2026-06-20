/**
 * @file ffi.hpp
 * @brief Device-agnostic FFI layer for GPU memory management.
 *
 * @details
 * Defines the top-level C API that the Rust FFI layer
 * (`ffi/cpp/bindings.rs`) calls into for GPU memory operations.
 * All functions are `extern "C"` to be callable from both Rust
 * and pure-C translation units.
 *
 * The implementation dispatches to the correct backend (CUDA or
 * HIP) based on the @ref deviceKind_t tag passed by the caller
 * or detected at runtime.
 *
 * @see ffi.cpp              Implementation of the dispatch layer.
 * @see admin.hpp            deviceKind_t enum and runtime probe.
 * @see ncore/cpp_ffi.h      C-callable memcpy wrapper (reverse direction).
 */

#pragma once

#include "admin.hpp"
#include <cstddef>

/**
 * @struct deviceBuffer_t
 * @brief Opaque descriptor for a GPU or pinned-host memory buffer.
 *
 * @details
 * This struct is the currency type passed between the Rust FFI
 * layer and the C++ backend.  The @ref deviceBufPtr member
 * holds a pointer to a backend-specific descriptor
 * (`cudaBuffer_t*` or `hipBuffer_t*`) that is allocated by
 * @ref deviceReserve and freed by @ref deviceRelease.
 *
 * The @ref device_kind member defaults to @ref deviceKind_t::DeviceNull.
 * It is set to the correct backend by @ref deviceReserve during
 * allocation; callers must not assume a backend before that point.
 *
 * Callers must not interpret @ref deviceBufPtr directly; it is
 * an opaque handle managed exclusively by the backend.
 */
struct deviceBuffer_t {
  void *ptr = nullptr;   ///< Pointer to device or pinned-host memory.
  std::size_t bytes = 0; ///< Usable size in bytes.
  bool isPinned = false; ///< Page-locked host memory flag.
  deviceKind_t deviceKind = deviceKind_t::DeviceNull; ///< Active backend.
  void *deviceBufPtr = nullptr; ///< Opaque backend-specific descriptor.
};

/**
 * @struct deviceStatus_t
 * @brief Result type for device memory operations.
 *
 * @details
 * Carries a numeric error code and a human-readable message.
 * A @ref code of `0` indicates success; negative values indicate
 * backend-level errors.
 */
struct deviceStatus_t {
  int code = 0;               ///< Zero on success, positive on failure.
  const char *message = "ok"; ///< Human-readable error description.
};

/**
 * @enum DeviceMemcpyKind
 * @brief Memory copy direction for inter-device transfers.
 */
enum class DeviceMemcpyKind : std::int8_t {
  deviceMemcpyHostToDevice = 1,  ///< Host → Device copy.
  deviceMemcpyDeviceToHost = 2,  ///< Device → Host copy.
  deviceMemcpyDeviceToDevice = 3 ///< Device → Device copy.
};

extern "C" {

/**
 * @brief Allocate a GPU or pinned-host memory buffer.
 *
 * @details
 * Dispatches to the appropriate backend allocator based on
 * @p kind.  The allocated buffer is described by @p out_buf,
 * which takes ownership of the backend-specific descriptor.
 *
 * @param[in]  bytes  Requested allocation size in bytes.
 * @param[out] out    Receives the buffer descriptor on success.
 * @param[in]  pinned If `true`, allocate page-locked host memory.
 * @param[in]  align  Alignment in bytes (default 512).
 * @param[in]  kind   Target backend (CUDA or HIP).
 *
 * @return @ref deviceStatus_t with `code == 0` on success.
 *
 * @pre  @p bytes must be greater than zero.
 * @pre  @p out must point to a valid @ref deviceBuffer_t.
 * @post On success, @p out->deviceBufPtr owns a backend
 *       descriptor that must be freed via @ref deviceRelease.
 *
 * @see deviceRelease()  Frees the buffer allocated here.
 * @see deviceResize()   Resizes an existing buffer.
 */
deviceStatus_t deviceReserve(std::size_t bytes, deviceBuffer_t *out_buf,
                             bool pinned = false, std::size_t align = 512,
                             deviceKind_t kind = deviceKind_t::DeviceCUDA);

/**
 * @brief Free a GPU or pinned-host memory buffer.
 *
 * @details
 * Casts @p buf->deviceBufPtr to the backend-specific type,
 * calls the backend release function, and zeroes the buffer
 * descriptor on success.
 *
 * @param[in,out] buf  Pointer to the buffer descriptor to free.
 *
 * @return @ref deviceStatus_t with `code == 0` on success.
 *
 * @pre  @p buf must point to a valid @ref deviceBuffer_t whose
 *       @ref deviceBufPtr was returned by @ref deviceReserve.
 * @post On success, @p buf is zeroed.
 *
 * @see deviceReserve()  Allocates the buffer freed here.
 */
deviceStatus_t deviceRelease(deviceBuffer_t *buf);

/**
 * @brief Resize an existing GPU or pinned-host memory buffer.
 *
 * @details
 * Casts @p buf->deviceBufPtr to the backend-specific type and
 * calls the backend resize function.  On success, the @ref ptr
 * and @ref bytes members of @p buf are updated.
 *
 * @param[in,out] buf       Pointer to the buffer descriptor to
 *                          resize.
 * @param[in]     new_bytes New size in bytes.
 * @param[in]     align     Required alignment in bytes.
 *
 * @return @ref deviceStatus_t with `code == 0` on success.
 *
 * @pre  @p buf must point to a valid @ref deviceBuffer_t whose
 *       @ref deviceBufPtr was returned by @ref deviceReserve.
 * @post On success, @p buf->ptr and @p buf->bytes reflect the
 *       new allocation.
 *
 * @warning On failure the original buffer may be in an
 *          inconsistent state.  Do not use it after a failed
 *          resize without checking.
 *
 * @see deviceReserve()  Initial allocation.
 * @see deviceRelease()  Explicit deallocation.
 */
deviceStatus_t deviceResize(deviceBuffer_t *buf, std::size_t new_bytes,
                            std::size_t align);

/**
 * @brief Copy memory between host and device (or device to device).
 *
 * @details
 * Auto-detects the active GPU backend via @ref getDeviceBackend()
 * and dispatches to the appropriate backend memcpy function.  The
 * copy direction is specified by @p kind.
 *
 * @param[in]  src       Source pointer.
 * @param[out] dst       Destination pointer.
 * @param[in]  kind      Copy direction (@ref DeviceMemcpyKind).
 * @param[in]  bytes     Number of bytes to copy.
 *
 * @return @ref deviceStatus_t with `code == 0` on success.
 *
 * @pre  @p src and @p dst must point to valid memory regions of
 *       at least @p bytes.
 * @pre  @p kind must match the actual memory types of @p src
 *       and @p dst.
 *
 * @see device_transfer_c()  C-callable variant using TransferKind.
 */
deviceStatus_t deviceTransfer(const void *src, void *dst, DeviceMemcpyKind kind,
                              std::size_t bytes);
}
