/**
 * @file status.h
 * @brief Error types and status reporting for the NovaNN runtime.
 *
 * @details
 * This header defines the core error codes and status structures used
 * throughout the NovaNN codebase for consistent error propagation.
 * It provides a unified way to report failures across different
 * compute backends and internal modules.
 *
 * @section architecture Architecture
 * The error handling system is built around a centralized enumeration
 * of error codes (@ref novaError_t) that are mapped to human-readable
 * messages in constant time. This ensures that error reporting is both
 * lightweight and descriptive.
 *
 * @section usage Usage
 * Functions that can fail should return a @ref novaError_t or a
 * @ref novaStatus_t structure. Callers can then use @ref nova_get_error_msg()
 * to log or display a descriptive message about the failure.
 *
 * @see status.c Implementation of error message retrieval.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include <ncore/headeronly/macros.h>

/**
 * @enum novaError_t
 * @brief Enumeration of error codes used by the NovaNN runtime.
 *
 * @details
 * Standardized error codes categorized by their source. These codes are
 * returned by most core functions to indicate success or the specific
 * nature of a failure.
 *
 * @note The enumerators are organized into logical groups (Parameters,
 *       Memory, Transfers, etc.) to aid in categorization and debugging.
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

} novaError_t;

/**
 * @struct novaStatus_t
 * @brief Container for an error code and an optional detailed message.
 *
 * @details
 * Used in high-level APIs where a simple error code is insufficient and
 * additional context about the failure is required.
 */
typedef struct {
  novaError_t err;     ///< The specific error code.
  const char *message; ///< A detailed, human-readable error message.
} novaStatus_t;

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief Retrieves the error message associated with a given error code.
 *
 * @details
 * Performs an O(1) lookup in the internal error string table to return
 * a human-readable description of the provided error code.
 *
 * @param[in] err The error code to look up.
 * @param[in] fallback Custom fallback message when @p err is out of
 *                     bounds or the table entry is nullptr.  May be nullptr.
 * @return A constant string containing the error message. If the code
 *         is not recognized and @p fallback is non-nullptr, returns
 *         @p fallback.  Otherwise returns Unknown error.
 *
 * @see novaError_t
 */
const char *nova_get_error_msg(novaError_t err, const char *fallback);
#ifdef __cplusplus
}
#endif
