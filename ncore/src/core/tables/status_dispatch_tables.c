/**
 * @file status_dispatch_tables.c
 * @brief Initialisation of the global error message dispatch table.
 *
 * @details
 * Populates the `status_msg_dispatch` array that maps each @ref novaError_t
 * code to its corresponding human-readable message. The table is
 * populated once at program load time via a `__attribute__((constructor))`
 * function.
 *
 * This separation ensures that the core status logic remains decoupled
 * from the specific message strings, following the same pattern as other
 * dispatch tables in the runtime.
 *
 * ## Constructor ordering
 * `__attribute__((constructor))` runs at program load time. Because
 * this file only writes to the `status_msg_dispatch` array (no other
 * globals depend on it), there are no inter-file ordering constraints.
 *
 * @see status_msg_dispatch The message lookup array.
 * @see novaError_t          The error code enumeration.
 */

#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>

/**
 * @var status_msg_dispatch
 * @brief Global lookup table for human-readable error messages.
 *
 * @details
 * Maps each @ref novaError_t value to its corresponding human-readable
 * string. The strings include a newline character for consistent
 * console logging.
 *
 * The table is zero-initialized and fully populated by
 * @ref init_status_msg_dispatch() at program load time.
 *
 * @see init_status_msg_dispatch()
 */
const char *status_msg_dispatch[NUM_ERRORS] = {NULL};

/**
 * @brief Populate every entry in `status_msg_dispatch`.
 *
 * @details
 * Called automatically before `main()` via `__attribute__((constructor))`.
 * Assigns a descriptive string to each valid error code defined in
 * the @ref novaError_t enumeration.
 *
 * @post All entries in @ref status_msg_dispatch are set to valid
 *       string pointers.
 */
ATTR(constructor) static inline void init_status_msg_dispatch() {
  /* Success */
  status_msg_dispatch[novaSuccess] = "Success\n";

  /* Invalid parameters */
  status_msg_dispatch[novaInvalidValue] =
      "One or more input values are invalid or out of range\n";
  status_msg_dispatch[novaInvalidTensor] =
      "Tensor object is invalid or in an inconsistent state\n";
  status_msg_dispatch[novaInvalidPointer] =
      "Null or invalid pointer provided where a valid address is required\n";
  status_msg_dispatch[novaInvalidDtype] =
      "Data type is not recognized or not valid for this operation\n";
  status_msg_dispatch[novaInvalidDevice] =
      "Device identifier is not valid or not supported\n";
  status_msg_dispatch[novaInvalidNdims] =
      "Number of dimensions is not valid or not supported\n";
  status_msg_dispatch[novaInvalidAlignment] =
      "Memory alignment requirement not met for the target device\n";
  status_msg_dispatch[novaInvalidShape] =
      "Tensor shape is invalid or violates operation constraints\n";
  status_msg_dispatch[novaInvalidIndex] =
      "Index is out of bounds for the given tensor dimensions\n";

  /* Memory */
  status_msg_dispatch[novaBufferOverflow] =
      "Buffer overflow: access exceeds allocated memory bounds\n";
  status_msg_dispatch[novaOutOfMemory] =
      "Out of memory: allocation request could not be satisfied\n";
  status_msg_dispatch[novaReserveError] =
      "Failed to reserve memory through the underlying allocator\n";
  status_msg_dispatch[novaReleaseError] =
      "Failed to release memory allocation\n";
  status_msg_dispatch[novaResizeError] = "Failed to resize memory allocation\n";

  /* Data transfer */
  status_msg_dispatch[novaTransferError] =
      "Memory transfer between devices failed\n";
  status_msg_dispatch[novaTransferH2DError] =
      "Host-to-device memory transfer failed\n";
  status_msg_dispatch[novaTransferD2HError] =
      "Device-to-host memory transfer failed\n";
  status_msg_dispatch[novaInvalidTransfDirection] =
      "Invalid transfer direction specified for the operation\n";

  /* Device/Backend */
  status_msg_dispatch[novaDeviceNotAvailable] =
      "No compatible compute device is available\n";
  status_msg_dispatch[novaDeviceNotInitialized] =
      "Compute device has not been initialized\n";
  status_msg_dispatch[novaBackendNotCompiled] =
      "Requested backend was not compiled into this build\n";
  status_msg_dispatch[novaBackendNotSupported] =
      "Requested backend is not supported on this platform\n";

  /* Dtype/Cast */
  status_msg_dispatch[novaDtypeNotSupported] =
      "Data type is not supported by the requested operation\n";
  status_msg_dispatch[novaCastNotSupported] =
      "Cast between the specified data types is not supported\n";
  status_msg_dispatch[novaShapeMismatch] =
      "Tensor shape mismatch: dimensions are incompatible for the operation\n";

  /* GPU-specific */
  status_msg_dispatch[novaKernelLaunchError] = "GPU kernel launch failed\n";
  status_msg_dispatch[novaInvalidResourceHandle] =
      "Internal GPU resource handle is invalid or corrupted\n";

  /* Internal */
  status_msg_dispatch[novaNotImplemented] =
      "Requested functionality is not yet implemented\n";
  status_msg_dispatch[novaInternalError] =
      "An unexpected internal failure occurred\n";

  /* General */
  status_msg_dispatch[novaRuntimeError] =
      "A runtime error occurred during operation execution\n";
}
