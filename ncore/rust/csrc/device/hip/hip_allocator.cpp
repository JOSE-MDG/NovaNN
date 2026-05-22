/**
 * @file hip_allocator.cpp
 * @brief Backend implementation of HIP device/pinned-host memory allocation.
 *
 * Implements hip_reserve and hip_release by wrapping the HIP Runtime
 * API (hipMalloc / hipMallocHost / hipFree / hipFreeHost) with
 * alignment rounding, descriptive error reporting, and a mapping from
 * HIP runtime error codes to application-level status codes.
 */

#include "hip_allocator.hpp"
#include <hip/hip_runtime_api.h>

/**
 * @brief Map a HIP runtime error to an application-level status code.
 *
 * @param err HIP error value returned by the runtime API.
 * @return 0 for hipSuccess, 1 for hipErrorInvalidValue,
 *         2 for hipErrorMemoryAllocation, -1 for any other error.
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
 * @brief Return a human-readable description for an application-level
 *        HIP status code.
 *
 * @param code Status code produced by map_hip_error().
 * @return Static string describing the error (never NULL).
 */
static const char *hip_error_msg(int code) {
  switch (code) {
  case 0:
    return "ok";
  case 1:
    return "hipErrorInvalidValue: one or more parameters is invalid (e.g., "
           "null pointer, misaligned size, or out-of-range argument)";
  case 2:
    return "hipErrorMemoryAllocation: the device or host memory allocation "
           "failed — possible causes: out of memory, fragmented heap, or "
           "driver limit reached";
  default:
    return "unknown hip error: an unrecognized HIP driver API error occurred";
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
 * Wraps hipHostMalloc (pinned host memory) or hipMalloc (device memory).
 * If @p align > 1, the allocation size is rounded up so that the returned
 * buffer satisfies the alignment constraint.
 *
 * @param bytes  Minimum number of bytes to allocate.
 * @param align  Alignment requirement (must be a power of two, or 1 for
 *               default alignment).
 * @param pinned If true, allocate page-locked host memory via
 *               hipHostMalloc; otherwise allocate device memory via
 *               hipMalloc.
 * @param out    Output buffer descriptor (only valid when the returned
 *               status has code == 0).
 * @return HIP_OK on success, or a HipStatus_t with a positive error code
 *         and a descriptive message on failure.
 */
HipStatus_t hip_reserve(std::size_t bytes, std::size_t align, bool pinned,
                        HipBuffer_t *out) {
  void *ptr = nullptr;
  const std::size_t alloc_bytes = (align > 1) ? align_up(bytes, align) : bytes;

  const hipError_t err =
      pinned ? hipHostMalloc(&ptr, alloc_bytes, hipHostMallocDefault)
             : hipMalloc(&ptr, alloc_bytes);

  const int code = map_hip_error(err);
  if (code != 0) {
    return HipStatus_t{.code = code, .msg = hip_error_msg(code)};
  }

  *out = HipBuffer_t{.ptr = ptr, .bytes = alloc_bytes, .is_pinned = pinned};
  return HIP_OK;
}

/**
 * @brief Free a HIP buffer previously allocated with hip_reserve().
 *
 * Wraps hipHostFree or hipFree depending on the buffer's is_pinned flag.
 * On success the output descriptor is zeroed out so that a subsequent
 * release is a safe no-op (aside from the null-pointer error check).
 *
 * @param buf Pointer to the buffer descriptor to free.  If buf or
 *            buf->ptr is NULL the function returns an error status
 *            without calling any HIP API.
 * @return HIP_OK on success, or a HipStatus_t with a positive error code
 *         and a descriptive message on failure.
 */
HipStatus_t hip_release(HipBuffer_t *buf) {
  if (buf == nullptr || buf->ptr == nullptr) {
    return HipStatus_t{
        .code = 1,
        .msg = "hip_release: buf or buf->ptr is null — nothing to free"};
  }

  const hipError_t err =
      buf->is_pinned ? hipHostFree(buf->ptr) : hipFree(buf->ptr);

  const int code = map_hip_error(err);
  if (code != 0) {
    return HipStatus_t{.code = code, .msg = hip_error_msg(code)};
  }

  *buf = HipBuffer_t{};
  return HIP_OK;
}
