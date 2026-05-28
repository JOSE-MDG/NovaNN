/**
 * @file hip_allocator.cpp
 * @brief Backend implementation of HIP device/pinned-host memory allocation.
 *
 * Implements hip_reserve, hip_realloc and hip_release by wrapping the
 * HIP Runtime API (hipMalloc / hipHostMalloc / hipFree / hipHostFree /
 * hipMemcpy) with alignment rounding, descriptive error reporting, and a
 * mapping from HIP runtime error codes to application-level status codes.
 *
 * Pinned-host operations are fully synchronous (hipHostMalloc / hipHostFree /
 * memcpy).  Device operations use the synchronous hipMalloc / hipFree /
 * hipMemcpy calls, which are themselves blocking and do not require an
 * explicit stream.
 *
 * hip_realloc allocates a new buffer of the requested size, copies the
 * minimum of the old and new sizes into it, frees the old buffer, and
 * updates the descriptor — cleaning up all intermediate resources on any
 * failure so that the original descriptor is always left untouched.
 */

#include "hip_allocator.hpp"
#include <cstring>

#if __has_include(<hip/hip_runtime_api.h>)
// clangd/clang-tidy runs may not define a HIP platform macro, but the HIP
// headers require exactly one platform to be set.
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hip_runtime_api.h>

/**
 * @brief Map a HIP runtime error to an application-level status code.
 *
 * @param err HIP error value returned by the runtime API.
 * @return 0  (hipSuccess),
 *         1  (hipErrorInvalidValue),
 *         2  (hipErrorMemoryAllocation),
 *        -1  (any other error).
 */
static int map_hip_error(hipError_t err) {
  switch (err) {
  case hipSuccess:
    return 0;
  case hipErrorInvalidValue:
    return 1;
  case hipErrorMemoryAllocation:
    return 2;
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
  const std::size_t alloc_bytes = (align > 1) ? align_up(bytes, align) : bytes;
  void *ptr = nullptr;

  const hipError_t err =
      pinned ? hipHostMalloc(&ptr, alloc_bytes, hipHostMallocDefault)
             : hipMalloc(&ptr, alloc_bytes);

  const int code = map_hip_error(err);
  if (code != 0) {
    return HipStatus_t{.code = code, .msg = hipGetErrorString(err)};
  }

  *out = HipBuffer_t{.ptr = ptr, .bytes = alloc_bytes, .is_pinned = pinned};
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

  const hipError_t err =
      buf->is_pinned ? hipHostFree(buf->ptr) : hipFree(buf->ptr);

  const int code = map_hip_error(err);
  if (code != 0) {
    return HipStatus_t{.code = code, .msg = hipGetErrorString(err)};
  }

  *buf = HipBuffer_t{};
  return HIP_OK;
}

/**
 * @brief Reallocate a HIP buffer to a new size, preserving content.
 *
 * Allocates a new buffer of @p new_bytes (rounded up to @p align),
 * copies min(old_bytes, new_bytes) from the old buffer, frees the old
 * buffer, and updates the descriptor.
 *
 * For device memory the sequence uses hipMalloc, hipMemcpy(DeviceToDevice)
 * and hipFree — all synchronous.  For pinned host memory it uses
 * hipHostMalloc, host-side memcpy, and hipHostFree.
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
HipStatus_t hip_realloc(HipBuffer_t *buf, std::size_t new_bytes,
                        std::size_t align) {
  if (buf == nullptr || buf->ptr == nullptr) {
    return HipStatus_t{.code = 1,
                       .msg = "hip_realloc: buf or buf->ptr is null"
                              " — nothing to reallocate"};
  }

  const std::size_t alloc_bytes =
      (align > 1) ? align_up(new_bytes, align) : new_bytes;
  const std::size_t copy_bytes =
      buf->bytes < alloc_bytes ? buf->bytes : alloc_bytes;
  void *new_ptr = nullptr;

  hipError_t err = buf->is_pinned ? hipHostMalloc(&new_ptr, alloc_bytes,
                                                  hipHostMallocDefault)
                                  : hipMalloc(&new_ptr, alloc_bytes);
  {
    const int code = map_hip_error(err);
    if (code != 0) {
      return HipStatus_t{.code = code, .msg = hipGetErrorString(err)};
    }
  }

  if (buf->is_pinned) {
    std::memcpy(new_ptr, buf->ptr, copy_bytes);
  } else {
    err = hipMemcpy(new_ptr, buf->ptr, copy_bytes, hipMemcpyDeviceToDevice);
    const int code = map_hip_error(err);
    if (code != 0) {
      hipFree(new_ptr);
      return HipStatus_t{.code = code, .msg = hipGetErrorString(err)};
    }
  }

  err = buf->is_pinned ? hipHostFree(buf->ptr) : hipFree(buf->ptr);
  {
    const int code = map_hip_error(err);
    if (code != 0) {
      if (buf->is_pinned) {
        hipHostFree(new_ptr);
      } else {
        hipFree(new_ptr);
      }
      return HipStatus_t{.code = code, .msg = hipGetErrorString(err)};
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
HipStatus_t hip_realloc(HipBuffer_t *, std::size_t, std::size_t) {
  return HipStatus_t{.code = -1,
                     .msg = "HIP runtime headers not available at"
                            " build/lint time"};
}

#endif
