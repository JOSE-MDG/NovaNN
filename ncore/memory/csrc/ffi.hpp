/**
 * @file ffi.hpp
 * @brief Device-agnostic FFI layer for GPU memory management.
 *
 * @details
 * Defines the top-level C API that the Rust FFI layer
 * (@c ffi/cpp/bindings.rs) calls into for GPU memory operations.
 * All functions are @c extern "C" to be callable from both Rust
 * and pure-C translation units.
 *
 * The implementation dispatches to the correct backend (CUDA or
 * HIP) based on the @ref deviceKind_t tag passed by the caller
 * or detected at runtime.
 *
 * @see ffi.cpp              Implementation of the dispatch layer.
 * @see admin.hpp            deviceKind_t enum and runtime probe.
 * @see deviceTransfer()      C-callable memcpy wrapper (defined in ffi.cpp).
 */

#pragma once

#include <cstddef>

#include <ncore/core/device.h>
#include <ncore/core/status.h>

#include "admin.hpp"

/**
 * @struct deviceBuffer_t
 * @brief Opaque descriptor for a GPU or pinned-host memory buffer.
 *
 * @details
 * This struct is the currency type passed between the Rust FFI
 * layer and the C++ backend.  The @ref deviceBufPtr member
 * holds a pointer to a backend-specific descriptor
 * (@c cudaBuffer_t* or @c hipBuffer_t*) that is allocated by
 * @ref deviceReserve and freed by @ref deviceRelease.
 *
 * The @ref deviceKind member defaults to @ref deviceKind_t::DeviceNull.
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
 * @enum DeviceMemcpyKind
 * @brief Memory copy direction for inter-device transfers.
 */
enum class DeviceMemcpyKind : std::int8_t {
  deviceMemcpyHostToDevice = 1,  ///< Host → Device copy.
  deviceMemcpyDeviceToHost = 2,  ///< Device → Host copy.
  deviceMemcpyDeviceToDevice = 3 ///< Device → Device copy.
};

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

extern "C" {

/**
 * @brief Allocate a GPU or pinned-host memory buffer.
 *
 * @details
 * Dispatches to the appropriate backend allocator based on
 * @p kind.  The allocated buffer is described by @p out_buf,
 * which takes ownership of the backend-specific descriptor.
 *
 * @param[in]  bytes   Requested allocation size in bytes.
 * @param[out] out_buf Receives the buffer descriptor on success.
 * @param[in]  pinned  If @c true, allocate page-locked host memory.
 * @param[in]  kind    Target backend (CUDA or HIP).
 *
 * @return @ref novaStatus_t with @c err = novaSuccess on success.
 *
 * @pre  @p bytes must be greater than zero.
 * @pre  @p out_buf must point to a valid @ref deviceBuffer_t.
 * @post On success, @p out_buf->deviceBufPtr owns a backend
 *       descriptor that must be freed via @ref deviceRelease.
 * @post On failure, @p out_buf is reset and no ownership is transferred
 *       to the caller.
 *
 * @see deviceRelease()  Frees the buffer allocated here.
 * @see deviceResize()   Resizes an existing buffer.
 */
novaStatus_t deviceReserve(std::size_t bytes, deviceBuffer_t *out_buf,
                           bool pinned = false,
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
 * @return @ref novaStatus_t with @c err = novaSuccess on success.
 *
 * @pre  @p buf must point to a valid @ref deviceBuffer_t whose
 *       @ref deviceBufPtr was returned by @ref deviceReserve.
 * @post On success, @p buf is zeroed.
 * @post On failure, @p buf remains available for the caller's error-handling
 *       policy and is not treated as successfully released.
 *
 * @see deviceReserve()  Allocates the buffer freed here.
 */
novaStatus_t deviceRelease(deviceBuffer_t *buf);

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
 *
 * @return @ref novaStatus_t with @c err = novaSuccess on success.
 *
 * @pre  @p buf must point to a valid @ref deviceBuffer_t whose
 *       @ref deviceBufPtr was returned by @ref deviceReserve.
 * @post On success, @p buf->ptr and @p buf->bytes reflect the
 *       new allocation.
 *
 * @warning A backend-specific failure may leave the original buffer in an
 *          unusable or backend-defined state. Do not issue another operation
 *          on it until the backend's status contract has been evaluated.
 *
 * @see deviceReserve()  Initial allocation.
 * @see deviceRelease()  Explicit deallocation.
 */
novaStatus_t deviceResize(deviceBuffer_t *buf, std::size_t new_bytes);
/**
 * @brief Perform a memory transfer between the host and a GPU
 *        device.
 *
 * @details
 * Dispatches to the active GPU backend (CUDA or HIP) determined at
 * run time by @ref getDeviceBackend().  The @p kind selects the
 * copy direction and must match the actual memory types of
 * @p src and @p dst.
 *
 * @param[in]  src   Source pointer.
 * @param[out] dst   Destination pointer.
 * @param[in]  kind  Copy direction (@ref TransferKind).
 * @param[in]  bytes Number of bytes to copy.
 *
 * @return @ref novaStatus_t with @c err = novaSuccess on success.
 *
 * @pre  @p src and @p dst must point to valid memory regions of
 *       at least @p bytes.
 * @pre  @p kind must match the actual memory types of @p src and
 *       @p dst.
 *
 * @see deviceReserve()  Allocates a GPU buffer.
 * @see deviceRelease()  Frees a GPU buffer.
 */
novaStatus_t deviceTransfer(const void *src, void *dst, TransferKind kind,
                            size_t bytes);
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif
