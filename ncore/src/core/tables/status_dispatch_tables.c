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
  status_msg_dispatch[novaInvalidValue] = "Invalid value\n";
  status_msg_dispatch[novaInvalidTensor] = "Invalid tensor\n";
  status_msg_dispatch[novaInvalidPointer] = "Invalid pointer\n";
  status_msg_dispatch[novaInvalidDtype] = "Invalid data type\n";
  status_msg_dispatch[novaInvalidDevice] = "Invalid device\n";
  status_msg_dispatch[novaInvalidAlignment] = "Invalid alignment\n";
  status_msg_dispatch[novaInvalidShape] = "Invalid shape\n";
  status_msg_dispatch[novaInvalidIndex] = "Invalid index\n";

  /* Memory */
  status_msg_dispatch[novaBufferOverflow] = "Buffer overflow\n";
  status_msg_dispatch[novaOutOfMemory] = "Out of memory\n";
  status_msg_dispatch[novaReserveError] = "Reserve error\n";
  status_msg_dispatch[novaReleaseError] = "Release error\n";
  status_msg_dispatch[novaResizeError] = "Resize error\n";

  /* Data transfer */
  status_msg_dispatch[novaTransferError] = "Transfer error\n";
  status_msg_dispatch[novaTransferH2DError] = "Host-to-device transfer error\n";
  status_msg_dispatch[novaTransferD2HError] = "Device-to-host transfer error\n";
  status_msg_dispatch[novaInvalidTransfDirection] =
      "Invalid transfer direction error\n";

  /* Device/Backend */
  status_msg_dispatch[novaDeviceNotAvailable] = "Device not available\n";
  status_msg_dispatch[novaDeviceNotInitialized] = "Device not initialized\n";
  status_msg_dispatch[novaBackendNotCompiled] = "Backend not compiled\n";
  status_msg_dispatch[novaBackendNotSupported] = "Backend not supported\n";

  /* Dtype/Cast */
  status_msg_dispatch[novaDtypeNotSupported] = "Data type not supported\n";
  status_msg_dispatch[novaCastNotSupported] = "Cast not supported\n";
  status_msg_dispatch[novaShapeMismatch] = "Shape mismatch\n";

  /* GPU-specific */
  status_msg_dispatch[novaKernelLaunchError] = "Kernel launch error\n";
  status_msg_dispatch[novaInvalidResourceHandle] =
      "Internal resource handle error\n";

  /* Internal */
  status_msg_dispatch[novaNotImplemented] = "Not implemented\n";
  status_msg_dispatch[novaInternalError] = "Internal error\n";

  /* General */
  status_msg_dispatch[novaRuntimeError] = "Runtime error\n";
}
