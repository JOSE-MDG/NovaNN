/**
 * @file HipAllocator.cpp
 * @brief HIP memory allocation, release, and resize implementation.
 *
 * @details
 * Implements the three core allocation primitives used by the
 * device-agnostic FFI layer (`ffi.cpp`).  All device-memory
 * operations use a temporary HIP stream for async allocation
 * and synchronise before returning.
 *
 * The file is conditionally compiled behind `NOVA_HAS_HIP` and
 * `__has_include(<hip/hip_runtime_api.h>)`.  When HIP headers are
 * unavailable (e.g., during linting), stub functions that return
 * an error status are provided.
 *
 * A `__HIP_PLATFORM_AMD__` macro is defined when neither AMD nor
 * NVIDIA platform macros are set, ensuring clangd and clang-tidy
 * can parse the HIP headers correctly.
 *
 * ## Architecture
 *
 * Internal helpers (within anonymous namespace):
 * - @ref mapError — maps any `hipError_t` to an integer code.
 * - @ref alignUp — rounds a byte count up to a multiple.
 * - @ref streamCreate — creates a temporary HIP stream.
 * - @ref streamSync — blocks until stream work completes.
 * - @ref streamDestroy — destroys a HIP stream.
 *
 * ## Error Mapping
 *
 * All HIP errors are mapped via @ref mapError:
 * - `hipSuccess` → 0
 * - `hipErrorInvalidValue` → 1
 * - `hipErrorNotSupported` → 2
 * - `hipErrorOutOfMemory` → 3
 * - `hipErrorInvalidResourceHandle` → 4
 * - All others → -1
 *
 * @see HipAllocator.hpp  Type declarations and function signatures.
 * @see HipIO.cpp         HIP data transfer implementation.
 * @see ffi.cpp           Dispatch layer that calls into this file.
 */

#ifdef NOVA_HAS_HIP
#if __has_include(<hip/hip_runtime_api.h>)
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__ 1
#endif
#include "HipAllocator.hpp"
#include <cstring>
#include <hip/hip_runtime_api.h>

namespace {

/**
 * @brief Map a HIP error code to an integer.
 *
 * @details
 * Converts `hipError_t` values into project-standard integer
 * codes.  Covers errors from allocation (`hipMallocAsync`,
 * `hipHostMalloc`), deallocation (`hipFreeAsync`,
 * `hipFreeHost`), stream operations, and memcpy.
 *
 * @param[in] err  The HIP error to map.
 *
 * @return Integer code: 0 for success, 1-4 for specific errors,
 *         -1 for unrecognised errors.
 */
int mapError(hipError_t err) {
  switch (err) {
  case hipSuccess:
    return 0;
  case hipErrorInvalidValue:
    return 1;
  case hipErrorNotSupported:
    return 2;
  case hipErrorOutOfMemory:
    return 3;
  case hipErrorInvalidResourceHandle:
    return 4;
  default:
    return -1;
  }
}

/**
 * @brief Round @p bytes up to the nearest multiple of @p align.
 *
 * @details
 * Uses bitwise AND to round up efficiently.  Requires @p align
 * to be a power of two.
 *
 * @param[in] bytes  The value to align.
 * @param[in] align  The alignment (must be a power of two).
 *
 * @return The smallest value >= @p bytes that is a multiple of
 *         @p align.
 */
constexpr std::size_t alignUp(std::size_t bytes, std::size_t align) noexcept {
  return (bytes + (align - 1)) & ~(align - 1);
}

/**
 * @brief Create a HIP stream.
 *
 * @param[out] stream  Receives the new stream handle.
 * @param[out] status  Receives the error status on failure.
 *
 * @return `true` on success, `false` on failure.
 */
bool streamCreate(hipStream_t *stream, hipStatus_t *status) {
  const hipError_t err = hipStreamCreate(stream);
  if (err != hipSuccess) {
    status->code = mapError(err);
    status->msg = hipGetErrorString(err);
    return false;
  }
  return true;
}

/**
 * @brief Block until all work on @p stream has completed.
 *
 * @param[in]  stream  The stream to synchronise.
 * @param[out] status  Receives the error status on failure.
 *
 * @return `true` on success, `false` on failure.
 */
bool streamSync(hipStream_t stream, hipStatus_t *status) {
  const hipError_t err = hipStreamSynchronize(stream);
  if (err != hipSuccess) {
    status->code = mapError(err);
    status->msg = hipGetErrorString(err);
    return false;
  }
  return true;
}

/**
 * @brief Destroy a HIP stream.
 *
 * @param[in]  stream  The stream to destroy.
 * @param[out] status  Receives the error status on failure.
 *
 * @return `true` on success, `false` on failure.
 */
bool streamDestroy(hipStream_t stream, hipStatus_t *status) {
  const hipError_t err = hipStreamDestroy(stream);
  if (err != hipSuccess) {
    status->code = mapError(err);
    status->msg = hipGetErrorString(err);
    return false;
  }
  return true;
}

} // namespace

/**
 * @brief Allocate a HIP memory buffer.
 *
 * @details
 * For pinned memory, calls `hipHostMalloc`.  For device memory,
 * creates a temporary stream, calls `hipMallocAsync`,
 * synchronises, and destroys the stream.
 *
 * @param[in]  bytes  Requested size in bytes.
 * @param[in]  align  Alignment in bytes.
 * @param[in]  pinned If `true`, allocate page-locked host memory.
 * @param[out] out    Receives the buffer descriptor on success.
 *
 * @return @ref HIP_OK on success, or an error status.
 *
 * @pre  @p bytes must be greater than zero.
 * @pre  @p out must not be null.
 * @post On success, @p out->ptr points to valid HIP memory.
 */
hipStatus_t hipReserve(std::size_t bytes, std::size_t align, bool pinned,
                       hipBuffer_t *out) {
  hipStatus_t status = {};
  const std::size_t allocBytes = (align > 1) ? alignUp(bytes, align) : bytes;
  void *ptr = nullptr;

  if (pinned) {
    const hipError_t err =
        hipHostMalloc(&ptr, allocBytes, hipHostMallocDefault);
    if (err != hipSuccess) {
      status.code = mapError(err);
      status.msg = hipGetErrorString(err);
      return status;
    }
  } else {
    hipStream_t stream = nullptr;
    if (!streamCreate(&stream, &status)) {
      return status;
    }

    const hipError_t err = hipMallocAsync(&ptr, allocBytes, stream);
    if (err != hipSuccess) {
      if (!streamDestroy(stream, &status)) {
        return status;
      }
      status.code = mapError(err);
      status.msg = hipGetErrorString(err);
      return status;
    }

    if (!streamSync(stream, &status)) {
      if (!streamDestroy(stream, &status)) {
        return status;
      }
      return status;
    }
    if (!streamDestroy(stream, &status)) {
      return status;
    }
  }

  out->ptr = ptr;
  out->bytes = allocBytes;
  out->isPinned = pinned;
  return HIP_OK;
}

/**
 * @brief Free a HIP memory buffer.
 *
 * @details
 * For pinned memory, calls `hipFreeHost`.  For device memory,
 * creates a temporary stream, calls `hipFreeAsync`, synchronises,
 * and destroys the stream.  On success, the buffer is zeroed.
 *
 * @param[in,out] buf  Buffer descriptor to free.  Must not be null.
 *
 * @return @ref HIP_OK on success, or an error status.
 *
 * @post On success, @p buf is zeroed.
 */
hipStatus_t hipRelease(hipBuffer_t *buf) {
  if (buf == nullptr || buf->ptr == nullptr) {
    return hipStatus_t{.code = 1,
                       .msg = "hipRelease: buf or buf->ptr is null"
                              " — nothing to free\n"};
  }

  hipStatus_t status = {};

  if (buf->isPinned) {
    const hipError_t err = hipFreeHost(buf->ptr);
    if (err != hipSuccess) {
      status.code = mapError(err);
      status.msg = hipGetErrorString(err);
      return status;
    }
  } else {
    hipStream_t stream = nullptr;
    if (!streamCreate(&stream, &status)) {
      return status;
    }

    const hipError_t err = hipFreeAsync(buf->ptr, stream);
    if (err != hipSuccess) {
      if (!streamDestroy(stream, &status)) {
        return status;
      }
      status.code = mapError(err);
      status.msg = hipGetErrorString(err);
      return status;
    }

    if (!streamSync(stream, &status)) {
      if (!streamDestroy(stream, &status)) {
        return status;
      }
      return status;
    }
    if (!streamDestroy(stream, &status)) {
      return status;
    }
  }

  buf->ptr = nullptr;
  buf->bytes = 0;
  buf->isPinned = false;
  return HIP_OK;
}

/**
 * @brief Resize a HIP memory buffer.
 *
 * @details
 * Allocates a new buffer, copies `min(old, new)` bytes, then frees
 * the old buffer.  For pinned memory the copy uses
 * `std::memcpy`; for device memory it uses `hipMemcpyAsync` on a
 * temporary stream.
 *
 * @param[in,out] buf       Buffer descriptor to resize.
 * @param[in]     new_bytes New size in bytes.
 * @param[in]     align     Alignment in bytes.
 *
 * @return @ref HIP_OK on success, or an error status.
 *
 * @post On success, @p buf->ptr and @p buf->bytes are updated.
 *
 * @warning On failure the original buffer may be freed.
 */
hipStatus_t hipResize(hipBuffer_t *buf, std::size_t new_bytes,
                      std::size_t align) {
  if (buf == nullptr || buf->ptr == nullptr) {
    return hipStatus_t{.code = 1,
                       .msg = "hipResize: buf or buf->ptr is null"
                              " — nothing to reallocate\n"};
  }

  hipStatus_t status = {};
  const std::size_t allocBytes =
      (align > 1) ? alignUp(new_bytes, align) : new_bytes;
  const std::size_t copyBytes =
      buf->bytes < allocBytes ? buf->bytes : allocBytes;
  void *newPtr = nullptr;

  if (buf->isPinned) {
    const hipError_t err =
        hipHostMalloc(&newPtr, allocBytes, hipHostMallocDefault);
    if (err != hipSuccess) {
      status.code = mapError(err);
      status.msg = hipGetErrorString(err);
      return status;
    }

    std::memcpy(newPtr, buf->ptr, copyBytes);

    const hipError_t freeErr = hipFreeHost(buf->ptr);
    if (freeErr != hipSuccess) {
      status.code = mapError(freeErr);
      status.msg = hipGetErrorString(freeErr);
      return status;
    }
  } else {
    hipStream_t stream = nullptr;
    if (!streamCreate(&stream, &status)) {
      return status;
    }

    hipError_t err = hipMallocAsync(&newPtr, allocBytes, stream);
    if (err != hipSuccess) {
      if (!streamDestroy(stream, &status)) {
        return status;
      }
      status.code = mapError(err);
      status.msg = hipGetErrorString(err);
      return status;
    }

    err = hipMemcpyAsync(newPtr, buf->ptr, copyBytes, hipMemcpyDeviceToDevice,
                         stream);
    if (err != hipSuccess) {
      if (!streamDestroy(stream, &status)) {
        return status;
      }
      status.code = mapError(err);
      status.msg = hipGetErrorString(err);
      return status;
    }

    err = hipFreeAsync(buf->ptr, stream);
    if (err != hipSuccess) {
      if (!streamDestroy(stream, &status)) {
        return status;
      }
      status.code = mapError(err);
      status.msg = hipGetErrorString(err);
      return status;
    }

    if (!streamSync(stream, &status)) {
      if (!streamDestroy(stream, &status)) {
        return status;
      }
      return status;
    }
    if (!streamDestroy(stream, &status)) {
      return status;
    }
  }

  buf->ptr = newPtr;
  buf->bytes = allocBytes;
  return HIP_OK;
}

#else // !__has_include(<hip/hip_runtime_api.h>)

/** @brief Stub: HIP runtime headers not available. */
hipStatus_t hipReserve(std::size_t, std::size_t, bool, hipBuffer_t *) {
  return hipStatus_t{.code = -1, .msg = "HIP runtime headers not available\n"};
}

/** @brief Stub: HIP runtime headers not available. */
hipStatus_t hipRelease(hipBuffer_t *) {
  return hipStatus_t{.code = -1, .msg = "HIP runtime headers not available\n"};
}

/** @brief Stub: HIP runtime headers not available. */
hipStatus_t hipResize(hipBuffer_t *, std::size_t, std::size_t) {
  return hipStatus_t{.code = -1, .msg = "HIP runtime headers not available\n"};
}

#endif
#endif /* NOVA_HAS_HIP */
