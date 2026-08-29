/**
 * @file HipIO.cpp
 * @brief HIP data transfer implementation.
 *
 * @details
 * Implements the master memcpy dispatcher @ref hipTransfer using
 * @c hipMemcpyAsync on a reusable HIP stream for all copy
 * directions.  The stream is created once and lives for the
 * lifetime of the process (singleton pattern).
 *
 * The file is conditionally compiled behind @c NOVA_HAS_HIP and
 * @c __has_include(<hip/hip_runtime_api.h>).  When HIP headers are
 * unavailable, a stub function returning an error status is
 * provided.
 *
 * A @c __HIP_PLATFORM_AMD__ macro is defined when neither AMD nor
 * NVIDIA platform macros are set, ensuring clangd and clang-tidy
 * can parse the HIP headers correctly.
 *
 * @section architecture Architecture
 *
 * The module exposes a single public function (@ref hipTransfer)
 * that handles H2D, D2H, and D2D transfers.  Internally it uses:
 * @li @ref mapError — maps @c hipError_t to @ref novaStatus_t.
 * @li @ref mapMemcpyKind — converts @ref DeviceMemcpyKind to
 *   @c hipMemcpyKind.
 * @li @ref getStream — returns a singleton HIP stream.
 *
 * @section error-mapping Error Mapping
 *
 * All HIP errors are mapped to a @ref novaStatus_t via a single
 * @ref mapError function. HIP exposes no @c hipErrorExternalDevice equivalent;
 * unsupported or otherwise unclassified failures map to
 * @ref novaNotImplemented.
 *
 * @see HipIO.hpp        Function declaration.
 * @see HipAllocator.cpp HIP memory allocation implementation.
 * @see ffi.cpp          Dispatch layer that calls into this file.
 */

#include <ncore/core/status.h>

#ifdef NOVA_HAS_HIP
#if __has_include(<hip/hip_runtime_api.h>)
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hip_runtime_api.h>

#include "../DetectHipDevice.hpp"
#include "HipAllocator.hpp"
#include "HipIO.hpp"

namespace {

/**
 * @brief Map a HIP runtime error to a @ref novaStatus_t.
 *
 * @details
 * Converts @c hipError_t codes returned by @c hipMemcpyAsync
 * and @c hipStreamSynchronize into the project-standard
 * @ref novaStatus_t format. Each error category is mapped to the shared Nova
 * error enumeration and the corresponding standard message is selected from
 * the Nova status table.
 *
 * The mapped categories include invalid values, invalid transfer direction,
 * invalid resource handles, and the generic unsupported-operation fallback.
 *
 * @param[in] err  The HIP error to map.
 *
 * @return @ref novaStatus_t with the mapped error and message.
 */
novaStatus_t mapError(hipError_t err) {
  novaStatus_t status = {};
  switch (err) {
  case hipSuccess:
    status.err = novaSuccess;
    break;
  case hipErrorInvalidValue:
    status.err = novaInvalidValue;
    break;
  case hipErrorInvalidMemcpyDirection:
    status.err = novaInvalidTransfDirection;
    break;
  case hipErrorInvalidResourceHandle:
    status.err = novaInvalidResourceHandle;
    break;
  default:
    status.err = novaNotImplemented;
    break;
  }
  status.message = nova_get_error_msg(status.err, nullptr);
  return status;
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
 * @brief Convert @ref DeviceMemcpyKind to @c hipMemcpyKind.
 *
 * @details
 * Maps the backend-agnostic @ref DeviceMemcpyKind enum to the
 * HIP-specific @c hipMemcpyKind enum used by @c hipMemcpyAsync.
 * The mapping is: @c deviceMemcpyHostToDevice →
 * @c hipMemcpyHostToDevice, @c deviceMemcpyDeviceToHost →
 * @c hipMemcpyDeviceToHost, @c deviceMemcpyDeviceToDevice →
 * @c hipMemcpyDeviceToDevice, default → @c hipMemcpyDefault.
 *
 * @param[in] kind  The device-agnostic copy direction.
 *
 * @return The corresponding HIP memcpy kind.
 */
hipMemcpyKind mapMemcpyKind(DeviceMemcpyKind kind) {
  switch (kind) {
  case DeviceMemcpyKind::deviceMemcpyHostToDevice:
    return hipMemcpyHostToDevice;
  case DeviceMemcpyKind::deviceMemcpyDeviceToHost:
    return hipMemcpyDeviceToHost;
  case DeviceMemcpyKind::deviceMemcpyDeviceToDevice:
    return hipMemcpyDeviceToDevice;
  default:
    return hipMemcpyDefault;
  }
}

/**
 * @brief Get or create the reusable HIP stream.
 *
 * @details
 * Returns a singleton HIP stream that is created on first call
 * and reused for all subsequent @ref hipTransfer operations.  The
 * stream is created with default flags (non-blocking, no
 * priority override).
 *
 * The stream is never explicitly destroyed.  The HIP runtime
 * reclaims all resources on process exit.  This avoids the
 * overhead of create/destroy per transfer and eliminates the
 * risk of use-after-free in concurrent scenarios.
 *
 * The @c static local variable holds the stream handle and is
 * zero-initialised before first use, but stream creation itself is
 * not synchronized: concurrent first calls from multiple threads can
 * race on the @c stream == nullptr check and each invoke
 * @c hipStreamCreate.  Callers that require a strictly once-created
 * stream must serialize the first call externally.
 *
 * @param[out] status  Receives an error status if stream creation
 *                     fails.  Unchanged on success.
 *
 * @return The singleton @c hipStream_t.
 */
hipStream_t getStream(novaStatus_t *status) {
  static hipStream_t stream = nullptr;
  if (stream == nullptr) {
    const hipError_t err = hipStreamCreate(&stream);
    if (err != hipSuccess) {
      *status = mapError(err);
    }
  }
  return stream;
}

} // namespace

/**
 * @brief Copy memory between host and device (or device to device).
 *
 * @details
 * Performs a memory transfer using @c hipMemcpyAsync on a
 * reusable internal HIP stream, then synchronizes the stream
 * before returning.  The transfer direction is determined by
 * @p kind.
 *
 * @subsection execution-flow Execution Flow
 *
 * @code{.cpp}
 *   novaStatus_t status;
 *   stream = getStream(&status);        // singleton stream
 *    if(status.err != novaSuccess) {
 *       // code
 *    }
 *   err = hipMemcpyAsync(dst, src,      // enqueue transfer
 *                        bytes, kind,
 *                        stream);
 *   if (err != hipSuccess) return mapError(err);
 *   sync_err = hipStreamSynchronize(stream);  // block
 *   if (sync_err != hipSuccess) return mapError(sync_err);
 *   return HIP_OK;
 * @endcode
 *
 * @subsection error-handling Error Handling
 *
 * Errors are detected at two points:
 * @li 1. @c hipMemcpyAsync launch failure — returns immediately.
 * @li 2. @c hipStreamSynchronize failure — detected after transfer
 *    completes (or fails asynchronously).
 *
 * Both use @ref mapError for consistent error mapping.
 *
 * @param[in]  bytes     Number of bytes to copy.
 * @param[in]  kind      Copy direction (@ref DeviceMemcpyKind).
 * @param[in]  src       Source pointer (host or device memory).
 * @param[out] dst       Destination pointer (host or device memory).
 *
 * @return @ref HIP_OK on success, or a @ref novaStatus_t with
 *         a non-success error and a descriptive message.
 */
novaStatus_t hipTransfer(std::size_t bytes, DeviceMemcpyKind kind,
                         const void *src, void *dst) {
  novaStatus_t status = {};
  hipStream_t stream = getStream(&status);

  if (status.err != novaSuccess) {
    return status;
  }

  if (supportMemoryPool()) {
    const hipError_t err =
        hipMemcpyAsync(dst, src, bytes, mapMemcpyKind(kind), stream);

    if (err != hipSuccess) {
      return mapError(err);
    }

    const hipError_t syncErr = hipStreamSynchronize(stream);
    if (syncErr != hipSuccess) {
      return mapError(syncErr);
    }
  } else {
    const hipError_t err = hipMemcpy(dst, src, bytes, mapMemcpyKind(kind));

    if (err != hipSuccess) {
      return mapError(err);
    }
  }

  return HIP_OK;
}

#else // !__has_include(<hip/hip_runtime_api.h>)

/** @brief Stub: HIP runtime headers not available. */
novaStatus_t hipTransfer(std::size_t, DeviceMemcpyKind, const void *, void *) {
  return novaStatus_t{.err = novaBackendNotCompiled,
                      .message =
                          nova_get_error_msg(novaBackendNotCompiled, nullptr)};
}

#endif
#endif /* NOVA_HAS_HIP */
