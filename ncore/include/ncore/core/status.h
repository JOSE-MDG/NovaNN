/**
 * @file status.h
 * @brief Error codes and status reporting for the NovaNN runtime.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include <ncore/headeronly/macros.h>

/**
 * @enum novaError_t
 * @brief Error codes for the whole runtime, grouped by source.
 */
typedef enum novaError_t : uint8_t {
  /* Success */
  novaSuccess,

  /* Invalid parameters */
  novaInvalidValue,
  novaInvalidTensor,
  novaInvalidPointer,
  novaInvalidDtype,
  novaInvalidDevice,
  novaInvalidNdims,
  novaInvalidAlignment,
  novaInvalidMemoryLayout,
  novaInvalidShape,
  novaInvalidIndex,
  novaInvalidNumThreads,
  novaInvalidParallelGroup,
  novaInvalidHandle,

  /* Memory */
  novaBufferOverflow,
  novaOutOfMemory,
  novaReserveError,
  novaReleaseError,
  novaResizeError,

  /* Data transfer */
  novaTransferError,
  novaTransferH2DError,
  novaTransferD2HError,
  novaInvalidTransfDirection,

  /* Device/Backend */
  novaDeviceNotAvailable,
  novaDeviceNotInitialized,
  novaExternalDeviceError,
  novaBackendNotCompiled,
  novaBackendNotSupported,

  /* OS/Platform */
  novaOsPlatformNotSupported,

  /* Dtype/Cast */
  novaDtypeNotSupported,
  novaCastNotSupported,
  novaShapeMismatch,

  /* GPU-specific */
  novaInvalidResourceHandle,
  novaKernelLaunchError,

  /* Internal */
  novaNotImplemented,
  novaInternalError,

  /* General */
  novaRuntimeError,

  /* Threads */
  novaThreadNotInitialized,

  /* Resource state */
  novaInvalidDeviceId,
  novaResourceBusy,

  /* Memory pools */
  novaMemoryPoolsNotNativelySupported,

} novaError_t;

/**
 * @struct novaStatus_t
 * @brief An error code with an optional message.
 */
typedef struct {
  novaError_t err;     ///< The specific error code.
  const char *message; ///< A detailed, human-readable error message.
} novaStatus_t;

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief Message for an error code.
 *
 * @param[in] err The error code to look up.
 * @param[in] fallback Custom message when @p err is out of bounds or the
 *                     table entry is nullptr.  May be nullptr.
 * @return A constant string containing the error message. If the code
 *         is not recognized and @p fallback is non-nullptr, returns
 *         @p fallback.  Otherwise returns Unknown error.
 */
NCORE_HOST_DEVICE const char *nova_get_error_msg(novaError_t err,
                                                 const char *fallback);
#ifdef __cplusplus
}
#endif
