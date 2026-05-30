/**
 * @file hip_allocator.cpp
 * @brief Backend implementation of HIP device/pinned-host memory allocation.
 *
 * Implements hip_reserve, hip_resize and hip_release by wrapping the
 * HIP Runtime API (hipMalloc / hipHostMalloc / hipFree / hipHostFree /
 * hipMemcpy) with alignment rounding, descriptive error reporting, and a
 * mapping from HIP runtime error codes to application-level status codes.
 *
 * Pinned-host operations are fully synchronous (hipHostMalloc / hipHostFree /
 * memcpy).  Device operations use the synchronous hipMalloc / hipFree /
 * hipMemcpy calls, which are themselves blocking and do not require an
 * explicit stream.
 *
 * hip_resize allocates a new buffer of the requested size, copies the
 * minimum of the old and new sizes into it, frees the old buffer, and
 * updates the descriptor — cleaning up all intermediate resources on any
 * failure so that the original descriptor is always left untouched.
 */

#ifdef NOVA_HAS_HIP
#if __has_include(<hip/hip_runtime_api.h>)
// clangd/clang-tidy runs may not define a HIP platform macro, but the HIP
// headers require exactly one platform to be set.
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__ 1
#endif
#include "hip_allocator.hpp"
#include <cstring>
#include <hip/hip_runtime_api.h>

/**
 * @brief Macro to check the status of asynchronous memory release using
 * hipFreeAsync()
 *
 */
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
 * @brief Map a HIP runtime error to an application-level status code.
 *
 * @param err HIP error value returned by the runtime API.
 * @return 0  (hipSuccess),
 *         1  (hipErrorInvalidValue),
 *         2  (hipErrorNotSupported),
 *         3  (hipErrorOutOfMemory),
 *        -1  (any other error).
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
 * @brief Map a HIP stream-creation error to an application-level code.
 *
 * @param err CUDA error value from hipStreamCreate.
 * @return 0  (hipSuccess),
 *         1  (hipErrorInvalidValue)
 *        -1  (any other error).
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
 * @brief Map a HIP stream-destroy error to an application-level code.
 *
 * @param err hip error value from hipStreamDestroy.
 * @return 0  (hipSuccess),
 *         1  (hipErrorInvalidHandle),
 *        -1  (any other error).
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
 * @brief Map a HIP stream-synchronise error to an application-level code.
 *
 * @param err hip error value from hipStreamSynchronize.
 * @return 0  (hipSuccess),
 *         1  (hipErrorInvalidHandle),
 *        -1  (any other error).
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
 * @brief Round @p bytes up to the next multiple of @p align.
 *
 * @param bytes Size to round.
 * @param align Alignment boundary (must be a power of two).
 * @return The smallest multiple of @p align that is >= @p bytes.
 */
static constexpr std::size_t align_up(std::size_t bytes,
                                      std::size_t align) noexcept {
  return (bytes + (align - 1)) & ~(align - 1);
}

/**
 * @brief Create a HIP stream, populating @p status on failure.
 *
 * @param[out] stream  Receives the new stream handle on success.
 * @param[out] status  Populated with a non-zero code and message on failure.
 * @return true on success, false on failure.
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
 * @brief Synchronise @p stream, populating @p status on failure.
 *
 * @param stream  Stream to synchronise.
 * @param[out] status  Populated with a non-zero code and message on failure.
 * @return true on success, false on failure.
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
 * @brief Destroy @p stream, populating @p status on failure.
 *
 * @param stream       Stream to destroy.
 * @param[out] status  Populated with a non-zero code and message on failure.
 * @return true on success, false on failure.
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
 * @brief Allocate a HIP buffer with optional alignment.
 *
 * Uses hipMalloc for device memory or hipHostMalloc for pinned host memory.
 * If @p align > 1 the allocation size is rounded up so that the returned
 * buffer satisfies the alignment constraint.
 *
 * @param bytes  Minimum number of bytes to allocate.
 * @param align  Alignment requirement (power of two, or 1 for default).
 * @param pinned If true, allocate page-locked host memory via hipHostMalloc;
 *               otherwise allocate device memory via hipMalloc.
 * @param out    Output buffer descriptor (valid only when code == 0).
 * @return HIP_OK on success, or a HipStatus_t with a positive error
 *         code and a descriptive message on failure.
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
 * @brief Free a HIP buffer previously allocated with hip_reserve().
 *
 * Uses hipFree for device memory or hipHostFree for pinned host memory.
 * On success the descriptor is zeroed so that a subsequent release is a
 * safe no-op.
 *
 * @param buf Pointer to the buffer descriptor to free.  If @p buf or
 *            @p buf->ptr is NULL the function returns an error status
 *            without calling any HIP API.
 * @return HIP_OK on success, or a HipStatus_t with a positive error
 *         code and a descriptive message on failure.
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
 * @brief Reallocate a HIP buffer to a new size, preserving content.
 *
 * Allocates a new buffer of @p new_bytes (rounded up to @p align),
 * copies min(old_bytes, new_bytes) from the old buffer, frees the old
 * buffer, and updates the descriptor.
 *
 * For device memory the sequence uses hipMallocAasync,
 * hipMemcpyAsync(DeviceToDevice) and hipFreeAsync — all asynchronous.  For
 * pinned host memory it uses hipHostMalloc, host-side memcpy, and hipHostFree.
 *
 * On any failure the original buffer descriptor is left unchanged and
 * all newly-allocated resources are cleaned up before returning.
 *
 * @param buf       Pointer to the buffer descriptor to reallocate.
 *                  Must have been previously allocated with hip_reserve.
 * @param new_bytes Target size in bytes (before alignment rounding).
 * @param align     Alignment requirement (power of two, or 1 for default).
 * @return HIP_OK on success, or a HipStatus_t with a positive error
 *         code and a descriptive message on failure.
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

/**
 * @brief Fallback allocation entry point when HIP headers are unavailable.
 */
HipStatus_t hip_reserve(std::size_t, std::size_t, bool, HipBuffer_t *) {
  return HipStatus_t{.code = -1,
                     .msg = "HIP runtime headers not available at"
                            " build/lint time"};
}

/**
 * @brief Fallback release entry point when HIP headers are unavailable.
 */
HipStatus_t hip_release(HipBuffer_t *) {
  return HipStatus_t{.code = -1,
                     .msg = "HIP runtime headers not available at"
                            " build/lint time"};
}

/**
 * @brief Fallback realloc entry point when HIP headers are unavailable.
 */
HipStatus_t hip_resize(HipBuffer_t *, std::size_t, std::size_t) {
  return HipStatus_t{.code = -1,
                     .msg = "HIP runtime headers not available at"
                            " build/lint time"};
}

#endif
#endif /* NOVA_HAS_HIP */
