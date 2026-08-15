/**
 * @file HipAllocator.cpp
 * @brief HIP memory allocation, release, and resize implementation.
 *
 * @details
 * Implements the three core allocation primitives used by the
 * device-agnostic FFI layer (@c ffi.cpp).  All device-memory
 * operations use a temporary HIP stream for async allocation
 * and synchronise before returning.
 *
 * The file is conditionally compiled behind @c NOVA_HAS_HIP and
 * @c __has_include(<hip/hip_runtime_api.h>).  When HIP headers are
 * unavailable (e.g., during linting), stub functions that return
 * an error status are provided.
 *
 * A @c __HIP_PLATFORM_AMD__ macro is defined when neither AMD nor
 * NVIDIA platform macros are set, ensuring clangd and clang-tidy
 * can parse the HIP headers correctly.
 *
 * @section architecture Architecture
 *
 * Internal helpers (within anonymous namespace):
 * @li @ref mapError — maps any @c hipError_t to a @ref novaError_t.
 * @li @ref streamCreate — creates a temporary HIP stream.
 * @li @ref streamSync — blocks until stream work completes.
 * @li @ref streamDestroy — destroys a HIP stream.
 *
 * @section error-mapping Error Mapping
 *
 * All HIP errors are mapped via @ref mapError to the common Nova error
 * enumeration. HIP has no @c hipErrorExternalDevice equivalent, so only the
 * error categories exposed by the HIP runtime are mapped here.
 *
 * @see HipAllocator.hpp  Type declarations and function signatures.
 * @see HipIO.cpp         HIP data transfer implementation.
 * @see ffi.cpp           Dispatch layer that calls into this file.
 */

#include <ncore/core/status.h>

#ifdef NOVA_HAS_HIP
#if __has_include(<hip/hip_runtime_api.h>)
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <cstring>
#include <hip/hip_runtime_api.h>

#include "../DetectHipDevice.hpp"
#include "HipAllocator.hpp"

namespace {

/**
 * @brief Map a HIP error code to a @ref novaError_t.
 *
 * @details
 * Converts @c hipError_t values into the project-standard
 * @ref novaError_t enumeration.  Covers errors from allocation
 * (@c hipMallocAsync, @c hipHostMalloc), deallocation
 * (@c hipFreeAsync, @c hipFreeHost), stream operations, and memcpy.
 *
 * @param[in] err  The HIP error to map.
 *
 * @return The corresponding Nova error category.
 */
novaError_t mapError(hipError_t err) {
  switch (err) {
  case hipSuccess:
    return novaSuccess;
  case hipErrorInvalidValue:
    return novaInvalidValue;
  case hipErrorOutOfMemory:
    return novaOutOfMemory;
  case hipErrorNotSupported:
    return novaNotImplemented;
  case hipErrorInvalidDevicePointer:
    return novaInvalidPointer;
  case hipErrorInvalidResourceHandle:
    return novaInvalidResourceHandle;
  default:
    return novaNotImplemented;
  }
}

/**
 * @brief Query whether the active HIP device supports memory pools.
 *
 * @details
 * Uses @c getHipDeviceId() to query the
 * @c hipDeviceAttributeMemoryPoolsSupported attribute.  This is
 * safe because HIP device detection is performed before any memory
 * is allocated on the device.
 *
 * @return @c true if memory pools are supported, @c false otherwise.
 */
bool supportMemoryPool() {
  static int supported = 0;

  const hipError_t err = hipDeviceGetAttribute(
      &supported, hipDeviceAttributeMemoryPoolsSupported, getHipDeviceId());

  return err == hipSuccess && static_cast<bool>(supported);
}

/**
 * @brief Create a HIP stream.
 *
 * @param[out] stream  Receives the new stream handle.
 * @param[out] status  Receives the error status on failure.
 *
 * @return @c true on success, @c false on failure.
 */
bool streamCreate(hipStream_t *stream, novaStatus_t *status) {
  const hipError_t err = hipStreamCreate(stream);
  if (err != hipSuccess) {
    status->err = mapError(err);
    status->message = nova_get_error_msg(status->err, nullptr);
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
 * @return @c true on success, @c false on failure.
 */
bool streamSync(hipStream_t stream, novaStatus_t *status) {
  const hipError_t err = hipStreamSynchronize(stream);
  if (err != hipSuccess) {
    status->err = mapError(err);
    status->message = nova_get_error_msg(status->err, nullptr);
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
 * @return @c true on success, @c false on failure.
 */
bool streamDestroy(hipStream_t stream, novaStatus_t *status) {
  const hipError_t err = hipStreamDestroy(stream);
  if (err != hipSuccess) {
    status->err = mapError(err);
    status->message = nova_get_error_msg(status->err, nullptr);
    return false;
  }
  return true;
}

} // namespace

/**
 * @brief Allocate a HIP memory buffer.
 *
 * @details
 * For pinned memory, calls @c hipHostMalloc.  For device memory,
 * creates a temporary stream, calls @c hipMallocAsync,
 * synchronises, and destroys the stream.
 *
 * @param[in]  bytes  Requested size in bytes.
 * @param[in]  pinned If @c true, allocate page-locked host memory.
 * @param[out] out    Receives the buffer descriptor on success.
 *
 * @return @ref HIP_OK on success, or an error status.
 *
 * @pre  @p bytes must be greater than zero.
 * @pre  @p out must not be null.
 * @post On success, @p out->ptr points to valid HIP memory.
 */
novaStatus_t hipReserve(std::size_t bytes, bool pinned, hipBuffer_t *out) {
  novaStatus_t status = {};
  void *ptr = nullptr;

  if (pinned) {
    const hipError_t err =
        hipHostMalloc(&ptr, bytes, hipHostMallocDefault);
    if (err != hipSuccess) {
      status.err = mapError(err);
      status.message = nova_get_error_msg(status.err, nullptr);
      return status;
    }
  } else {
    if (supportMemoryPool()) {
      hipStream_t stream = nullptr;
      if (!streamCreate(&stream, &status)) {
        return status;
      }

      const hipError_t err = hipMallocAsync(&ptr, bytes, stream);
      if (err != hipSuccess) {
        if (!streamDestroy(stream, &status)) {
          return status;
        }
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
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
    } else {
      const hipError_t err = hipMalloc(&ptr, bytes);
      if (err != hipSuccess) {
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
        return status;
      }
    }
  }

  out->ptr = ptr;
  out->bytes = bytes;
  out->isPinned = pinned;
  return HIP_OK;
}

/**
 * @brief Free a HIP memory buffer.
 *
 * @details
 * For pinned memory, calls @c hipFreeHost.  For device memory,
 * creates a temporary stream, calls @c hipFreeAsync, synchronises,
 * and destroys the stream.  On success, the buffer is zeroed.
 *
 * @param[in,out] buf  Buffer descriptor to free.  Must not be null.
 *
 * @return @ref HIP_OK on success, or an error status.
 *
 * @post On success, @p buf is zeroed.
 */
novaStatus_t hipRelease(hipBuffer_t *buf) {
  if (buf == nullptr || buf->ptr == nullptr) {
    return novaStatus_t{.err = novaInvalidPointer,
                        .message =
                            nova_get_error_msg(novaInvalidPointer, nullptr)};
  }

  novaStatus_t status = {};

  if (buf->isPinned) {
    const hipError_t err = hipFreeHost(buf->ptr);
    if (err != hipSuccess) {
      status.err = mapError(err);
      status.message = nova_get_error_msg(status.err, nullptr);
      return status;
    }
  } else {
    if (supportMemoryPool()) {
      hipStream_t stream = nullptr;
      if (!streamCreate(&stream, &status)) {
        return status;
      }

      const hipError_t err = hipFreeAsync(buf->ptr, stream);
      if (err != hipSuccess) {
        if (!streamDestroy(stream, &status)) {
          return status;
        }
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
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
    } else {
      const hipError_t err = hipFree(buf->ptr);
      if (err != hipSuccess) {
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
        return status;
      }
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
 * Allocates a new buffer, copies @c min(old, new) bytes, then frees
 * the old buffer.  For pinned memory the copy uses
 * @c std::memcpy; for device memory it uses @c hipMemcpyAsync on a
 * temporary stream.
 *
 * @param[in,out] buf       Buffer descriptor to resize.
 * @param[in]     new_bytes New size in bytes.
 *
 * @return @ref HIP_OK on success, or an error status.
 *
 * @post On success, @p buf->ptr and @p buf->bytes are updated.
 *
 * @warning On failure the original buffer may be freed.
 */
novaStatus_t hipResize(hipBuffer_t *buf, std::size_t new_bytes) {
  if (buf == nullptr || buf->ptr == nullptr) {
    return novaStatus_t{.err = novaInvalidPointer,
                        .message =
                            nova_get_error_msg(novaInvalidPointer, nullptr)};
  }

  novaStatus_t status = {};
  const std::size_t copyBytes =
      buf->bytes < new_bytes ? buf->bytes : new_bytes;
  void *newPtr = nullptr;

  if (buf->isPinned) {
    const hipError_t err =
        hipHostMalloc(&newPtr, new_bytes, hipHostMallocDefault);
    if (err != hipSuccess) {
      status.err = mapError(err);
      status.message = nova_get_error_msg(status.err, nullptr);
      return status;
    }

    std::memcpy(newPtr, buf->ptr, copyBytes);

    const hipError_t freeErr = hipFreeHost(buf->ptr);
    if (freeErr != hipSuccess) {
      status.err = mapError(freeErr);
      status.message = nova_get_error_msg(status.err, nullptr);
      return status;
    }
  } else {
    if (supportMemoryPool()) {
      hipStream_t stream = nullptr;
      if (!streamCreate(&stream, &status)) {
        return status;
      }

      hipError_t err = hipMallocAsync(&newPtr, new_bytes, stream);
      if (err != hipSuccess) {
        if (!streamDestroy(stream, &status)) {
          return status;
        }
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
        return status;
      }

      err = hipMemcpyAsync(newPtr, buf->ptr, copyBytes, hipMemcpyDeviceToDevice,
                           stream);
      if (err != hipSuccess) {
        const hipError_t freeAsyncErr = hipFreeAsync(newPtr, stream);
        if (freeAsyncErr != hipSuccess) {
          if (!streamSync(stream, &status)) {
            if (!streamDestroy(stream, &status)) {
              return status;
            }
            return status;
          }
          status.err = mapError(freeAsyncErr);
          status.message = nova_get_error_msg(status.err, nullptr);
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
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
        return status;
      }

      err = hipFreeAsync(buf->ptr, stream);
      if (err != hipSuccess) {
        const hipError_t freeAsyncErr = hipFreeAsync(newPtr, stream);
        if (freeAsyncErr != hipSuccess) {
          status.err = mapError(freeAsyncErr);
          status.message = nova_get_error_msg(status.err, nullptr);
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
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
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
    } else {
      hipError_t err = hipMalloc(&newPtr, new_bytes);
      if (err != hipSuccess) {
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
        return status;
      }

      err = hipMemcpy(newPtr, buf->ptr, copyBytes, hipMemcpyDeviceToDevice);
      if (err != hipSuccess) {
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
        return status;
      }

      err = hipFree(buf->ptr);
      if (err != hipSuccess) {
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
        return status;
      }
    }
  }

  buf->ptr = newPtr;
  buf->bytes = new_bytes;
  return HIP_OK;
}

#else // !__has_include(<hip/hip_runtime_api.h>)

/** @brief Stub: HIP runtime headers not available. */
novaStatus_t hipReserve(std::size_t, bool, hipBuffer_t *) {
  return novaStatus_t{.err = novaBackendNotCompiled,
                      .message =
                          nova_get_error_msg(novaBackendNotCompiled, nullptr)};
}

/** @brief Stub: HIP runtime headers not available. */
novaStatus_t hipRelease(hipBuffer_t *) {
  return novaStatus_t{.err = novaBackendNotCompiled,
                      .message =
                          nova_get_error_msg(novaBackendNotCompiled, nullptr)};
}

/** @brief Stub: HIP runtime headers not available. */
novaStatus_t hipResize(hipBuffer_t *, std::size_t) {
  return novaStatus_t{.err = novaBackendNotCompiled,
                      .message =
                          nova_get_error_msg(novaBackendNotCompiled, nullptr)};
}

#endif
#endif /* NOVA_HAS_HIP */
