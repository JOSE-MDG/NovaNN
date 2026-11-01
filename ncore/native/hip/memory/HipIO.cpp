/**
 * @file HipIO.cpp
 * @brief HIP data transfer implementation.
 *
 * @details
 * Implements the master memcpy dispatcher @ref hipTransfer using
 * @c hipMemcpyAsync.  A caller stream enqueues with no
 * synchronization; the null path resolves the device engine stream
 * and drains it before returning.
 *
 * The null path never enqueues on the null stream: rocclr drops
 * pending null-stream work when a stream is created, so the engine
 * owns every device operation it enqueues.
 *
 * Compiled only behind @c NOVA_HAS_HIP and
 * @c __has_include(<hip/hip_runtime_api.h>); otherwise a stub
 * returning an error status is provided.
 *
 * A @c __HIP_PLATFORM_AMD__ macro is defined when neither AMD nor
 * NVIDIA platform macros are set, so clangd and clang-tidy can
 * parse the HIP headers correctly.
 *
 * All HIP errors are mapped to a @ref novaStatus_t via the shared
 * @ref mapError2Status function.  HIP exposes no
 * @c hipErrorExternalDevice equivalent; unsupported or otherwise
 * unclassified failures map to @ref novaNotImplemented.
 *
 * @see HipIO.hpp        Function declaration.
 * @see HipAllocator.cpp HIP memory allocation implementation.
 * @see common.hh        Shared error-mapping and stream helpers.
 * @see ffi.cpp          Dispatch layer that calls into this file.
 */

#include <ncore/core/status.h>

#ifdef NOVA_HAS_HIP
#if __has_include(<hip/hip_runtime_api.h>)
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hip_runtime_api.h>

#include "HipAllocator.hpp"
#include "HipEngineStream.hpp"
#include "HipIO.hpp"
#include "common.hh"

/**
 * @brief Copy memory between host and device (or device to device).
 *
 * @details
 * Performs a memory transfer using @c hipMemcpyAsync.  With a caller
 * stream the copy is enqueued there and returns with no
 * synchronization.  With @p stream null the copy is enqueued on the
 * device engine stream and the call blocks on
 * @c hipStreamSynchronize, which also waits for every kernel
 * previously enqueued on that stream, so a DtoH transfer observes
 * them.  The transfer direction is determined by @p kind.
 *
 * Errors surface at two points: the @c hipMemcpyAsync launch and,
 * only on the engine path, the @c hipStreamSynchronize wait; both map
 * through @ref mapError2Status.
 *
 * @param[in]  bytes     Number of bytes to copy.
 * @param[in]  kind      Copy direction (@ref DeviceMemcpyKind).
 * @param[in]  src       Source pointer (host or device memory).
 * @param[out] dst       Destination pointer (host or device memory).
 * @param[in]  stream    Caller stream for chaining, or null to select
 *                       the device engine stream.
 *
 * @return @ref HIP_OK on success, or a @ref novaStatus_t with
 *         a non-success error and a descriptive message.
 */
novaStatus_t hipTransfer(std::size_t bytes, DeviceMemcpyKind kind,
                         const void *src, void *dst, hipStream_t stream) {
  hipStream_t work = stream;
  if (work == nullptr) {
    const novaStatus_t engineStatus = HipEngineStream::current(&work);
    if (engineStatus.err != novaSuccess) {
      return engineStatus;
    }
  }

  const hipError_t err =
      hipMemcpyAsync(dst, src, bytes, mapMemcpyKind(kind), work);
  if (err != hipSuccess) {
    return mapError2Status(err);
  }
  if (stream == nullptr) {
    const hipError_t syncErr = hipStreamSynchronize(work);
    if (syncErr != hipSuccess) {
      return mapError2Status(syncErr);
    }
  }

  return HIP_OK;
}

#else // !__has_include(<hip/hip_runtime_api.h>)

/** @brief Stub: HIP runtime headers not available. */
novaStatus_t hipTransfer(std::size_t, DeviceMemcpyKind, const void *, void *,
                         hipStream_t) {
  return novaStatus_t{.err = novaBackendNotCompiled,
                      .message =
                          nova_get_error_msg(novaBackendNotCompiled, nullptr)};
}

#endif
#endif /* NOVA_HAS_HIP */
