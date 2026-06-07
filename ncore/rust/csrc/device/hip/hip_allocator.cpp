/**
 * @file hip_allocator.cpp
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
 * ## Error Mapping
 *
 * Each HIP error code is mapped to an integer:
 * - `0` — success
 * - `1` — invalid value
 * - `2` — not supported
 * - `3` — out of memory / invalid handle
 * - `-1` — unrecognised error
 *
 * @see hip_allocator.hpp  Type declarations and function signatures.
 * @see hip_io.cpp         HIP data transfer implementation.
 * @see ffi.cpp            Dispatch layer that calls into this file.
 */

#ifdef NOVA_HAS_HIP
#if __has_include(<hip/hip_runtime_api.h>)
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__ 1
#endif
#include "hip_allocator.hpp"
#include <cstring>
#include <hip/hip_runtime_api.h>

#define FREE_ASYNC_CHECK(ptr, stream, status)                                  \
  do {                                                                         \
    const hipError_t free_async_err = hipFreeAsync((ptr), stream);             \
                                                                               \
    if (free_async_err != hipSuccess) {                                        \
      (status).code = map_hip_error(free_async_err);                           \
      (status).msg = hipGetErrorString(free_async_err);                        \
      return status;                                                           \
    }                                                                          \
  } while (0);

/**
 * @brief Map a HIP error code to an integer error code.
 *
 * @param[in] err  The HIP error to map.
 *
 * @return `0` for success, `1` for invalid value, `2` for not
 *         supported, `3` for out of memory, `-1` otherwise.
 */
static int map_hip_error(hipError_t err) {
  switch (err) {
  case hipSuccess:
    return 0;
  case hipErrorInvalidValue:
    return 1;
  case hipErrorNotSupported:
    return 2;
  case hipErrorOutOfMemory:
    return 3;
  default:
    return -1;
  }
}

/**
 * @brief Map a HIP stream operation error to an integer code.
 *
 * @param[in] err  The HIP error from a stream operation.
 *
 * @return `0` for success, `1` for invalid value, `-1`
 *         otherwise.
 */
static int map_hip_stream_error(hipError_t err) {
  switch (err) {
  case hipSuccess:
    return 0;
  case hipErrorInvalidValue:
    return 1;
  default:
    return -1;
  }
}

/**
 * @brief Map a HIP stream destruction error to an integer code.
 *
 * @param[in] err  The HIP error from stream destruction.
 *
 * @return `0` for success, `1` for invalid handle, `-1`
 *         otherwise.
 */
static int map_hip_destroy_error(hipError_t err) {
  switch (err) {
  case hipSuccess:
    return 0;
  case hipErrorInvalidHandle:
    return 1;
  default:
    return -1;
  }
}

/**
 * @brief Map a HIP stream synchronisation error to an integer code.
 *
 * @param[in] err  The HIP error from stream synchronisation.
 *
 * @return `0` for success, `1` for invalid handle, `-1`
 *         otherwise.
 */
static int map_hip_sync_error(hipError_t err) {
  switch (err) {
  case hipSuccess:
    return 0;
  case hipErrorInvalidHandle:
    return 1;
  default:
    return -1;
  }
}

/**
 * @brief Round @p bytes up to the nearest multiple of @p align.
 *
 * @param[in] bytes  The value to align.
 * @param[in] align  The alignment (must be a power of two).
 *
 * @return The smallest value >= @p bytes that is a multiple of
 *         @p align.
 */
static constexpr std::size_t align_up(std::size_t bytes,
                                      std::size_t align) noexcept {
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
static bool stream_create(hipStream_t *stream, HipStatus_t *status) {
  const hipError_t err = hipStreamCreate(stream);
  if (err != hipSuccess) {
    status->code = map_hip_stream_error(err);
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
static bool stream_sync(hipStream_t stream, HipStatus_t *status) {
  const hipError_t err = hipStreamSynchronize(stream);
  if (err != hipSuccess) {
    status->code = map_hip_sync_error(err);
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
static bool stream_destroy(hipStream_t stream, HipStatus_t *status) {
  const hipError_t err = hipStreamDestroy(stream);
  if (err != hipSuccess) {
    status->code = map_hip_destroy_error(err);
    status->msg = hipGetErrorString(err);
    return false;
  }
  return true;
}

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
 * @post On success, @p out->ptr points to valid HIP memory.
 */
HipStatus_t hip_reserve(std::size_t bytes, std::size_t align, bool pinned,
                        HipBuffer_t *out) {

  HipStatus_t status = {};
  const std::size_t alloc_bytes = (align > 1) ? align_up(bytes, align) : bytes;
  void *ptr = nullptr;

  if (pinned) {

    const hipError_t err =
        hipHostMalloc(&ptr, alloc_bytes, hipHostMallocDefault);
    if (err != hipSuccess) {
      status.code = map_hip_error(err);
      status.msg = hipGetErrorString(err);
      return status;
    }

  } else {
    hipStream_t stream = nullptr;
    if (!stream_create(&stream, &status)) {
      return status;
    }

    const hipError_t err = hipMallocAsync(&ptr, alloc_bytes, stream);
    if (err != hipSuccess) {
      if (!stream_destroy(stream, &status)) {
        return status;
      }

      status.code = map_hip_error(err);
      status.msg = hipGetErrorString(err);
      return status;
    }

    if (!stream_sync(stream, &status)) {
      const hipError_t stream_err = hipStreamDestroy(stream);
      if (stream_err != hipSuccess) {
        status.code = map_hip_destroy_error(stream_err);
        status.msg = hipGetErrorString(stream_err);
        return status;
      }
      return status;
    }
    if (!stream_destroy(stream, &status)) {
      return status;
    }
  }

  out->ptr = ptr;
  out->bytes = alloc_bytes;
  out->is_pinned = pinned;
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
 * @param[in,out] buf  Buffer descriptor to free.
 *
 * @return @ref HIP_OK on success, or an error status.
 *
 * @post On success, @p buf is zeroed.
 */
HipStatus_t hip_release(HipBuffer_t *buf) {
  if (buf == nullptr || buf->ptr == nullptr) {
    return HipStatus_t{.code = 1,
                       .msg = "hip_release: buf or buf->ptr is null"
                              " — nothing to free"};
  }

  HipStatus_t status = {};

  if (buf->is_pinned) {

    const hipError_t err = hipFreeHost(buf->ptr);
    if (err != hipSuccess) {
      status.code = map_hip_error(err);
      status.msg = hipGetErrorString(err);
      return status;
    }
  } else {

    hipStream_t stream = nullptr;
    if (!stream_create(&stream, &status)) {
      return status;
    }

    const hipError_t err = hipFreeAsync(buf->ptr, stream);
    if (err != hipSuccess) {
      if (!stream_destroy(stream, &status)) {
        return status;
      }
      status.code = map_hip_error(err);
      status.msg = hipGetErrorString(err);
      return status;
    }

    if (!stream_sync(stream, &status)) {
      const hipError_t stream_err = hipStreamDestroy(stream);
      if (stream_err != hipSuccess) {
        status.code = map_hip_destroy_error(stream_err);
        status.msg = hipGetErrorString(stream_err);
        return status;
      }
      return status;
    }
    if (!stream_destroy(stream, &status)) {
      return status;
    }
  }

  buf->ptr = nullptr;
  buf->bytes = 0;
  buf->is_pinned = false;

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
HipStatus_t hip_resize(HipBuffer_t *buf, std::size_t new_bytes,
                       std::size_t align) {
  if (buf == nullptr || buf->ptr == nullptr) {
    return HipStatus_t{.code = 1,
                       .msg = "hip_resize: buf or buf->ptr is null"
                              " — nothing to reallocate"};
  }

  HipStatus_t status = {};
  const std::size_t alloc_bytes =
      (align > 1) ? align_up(new_bytes, align) : new_bytes;

  const std::size_t copy_bytes =
      (buf->bytes < alloc_bytes) ? buf->bytes : alloc_bytes;

  void *new_ptr = nullptr;

  if (buf->is_pinned) {

    const hipError_t err =
        hipHostMalloc(&new_ptr, alloc_bytes, hipHostMallocDefault);
    if (err != hipSuccess) {
      status.code = map_hip_error(err);
      status.msg = hipGetErrorString(err);
      return status;
    }

    std::memcpy(new_ptr, buf->ptr, copy_bytes);

    const hipError_t free_err = hipFreeHost(buf->ptr);
    if (free_err != hipSuccess) {
      const int code = map_hip_error(free_err);
      const int inner_code = map_hip_error(hipFreeHost(new_ptr));
      status.code = (inner_code != 0) ? inner_code : code;
      status.msg = hipGetErrorString(free_err);
      return status;
    }

  } else {

    hipStream_t stream = nullptr;
    if (!stream_create(&stream, &status)) {
      return status;
    }

    hipError_t err = hipMallocAsync(&new_ptr, alloc_bytes, stream);
    if (err != hipSuccess) {
      if (!stream_destroy(stream, &status)) {
        return status;
      }

      status.code = map_hip_error(err);
      status.msg = hipGetErrorString(err);
      return status;
    }

    err = hipMemcpyAsync(new_ptr, buf->ptr, copy_bytes, hipMemcpyDeviceToDevice,
                         stream);

    if (err != hipSuccess) {
      FREE_ASYNC_CHECK(new_ptr, stream, status);

      if (!stream_sync(stream, &status)) {
        return status;
      }

      if (!stream_destroy(stream, &status)) {
        return status;
      }

      status.code = map_hip_error(err);
      status.msg = hipGetErrorString(err);
      return status;
    }

    err = hipFreeAsync(buf->ptr, stream);
    if (err != hipSuccess) {
      FREE_ASYNC_CHECK(new_ptr, stream, status);
      if (!stream_sync(stream, &status)) {
        return status;
      }

      if (!stream_destroy(stream, &status)) {
        return status;
      }

      status.code = map_hip_error(err);
      status.msg = hipGetErrorString(err);
      return status;
    }

    if (!stream_sync(stream, &status)) {
      return status;
    }

    if (!stream_destroy(stream, &status)) {
      return status;
    }
  }

  buf->ptr = new_ptr;
  buf->bytes = alloc_bytes;

  return HIP_OK;
}

#else // !__has_include(<hip/hip_runtime_api.h>)

/** @brief Stub: HIP runtime headers not available. */
HipStatus_t hip_reserve(std::size_t, std::size_t, bool, HipBuffer_t *) {
  return HipStatus_t{.code = -1,
                     .msg = "HIP runtime headers not available at"
                            " build/lint time"};
}

/** @brief Stub: HIP runtime headers not available. */
HipStatus_t hip_release(HipBuffer_t *) {
  return HipStatus_t{.code = -1,
                     .msg = "HIP runtime headers not available at"
                            " build/lint time"};
}

/** @brief Stub: HIP runtime headers not available. */
HipStatus_t hip_resize(HipBuffer_t *, std::size_t, std::size_t) {
  return HipStatus_t{.code = -1,
                     .msg = "HIP runtime headers not available at"
                            " build/lint time"};
}

#endif
#endif /* NOVA_HAS_HIP */
