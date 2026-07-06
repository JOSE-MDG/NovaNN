/**
 * @file status.c
 * @brief Implementation of error message retrieval.
 *
 * @details
 * This module provides the implementation for retrieving human-readable
 * error messages from @ref novaError_t codes. It utilizes a global
 * dispatch table defined in @ref status_dispatch_tables.c to provide
 * O(1) retrieval time.
 *
 * @see status.h Public interface and @ref novaError_t definition.
 * @see status_dispatch_tables.c Global message table definition.
 */

#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>

/**
 * @var status_msg_dispatch
 * @brief External reference to the global error message table.
 *
 * @details
 * Defined in @ref status_dispatch_tables.c. This table is populated
 * at program load time.
 */
extern const char *status_msg_dispatch[NUM_ERRORS];

/**
 * @brief Retrieves the human-readable message for a given error code.
 *
 * @details
 * Validates the provided error code against the table bounds (@ref NUM_ERRORS)
 * and returns the corresponding message string from @ref status_msg_dispatch.
 * If the code is out of bounds or the table entry is nullptr, returns the
 * caller-provided @p fallback message, or "Unknown error" if @p fallback
 * is nullptr.
 *
 * @param[in] err      The @ref novaError_t code to look up.
 * @param[in] fallback Custom fallback message.  May be nullptr.
 * @return A constant pointer to the error message string.
 *
 * @note This function is thread-safe as it only performs read operations
 *       on the global table after it has been initialized at load time.
 */
const char *nova_get_error_msg(novaError_t err, const char *fallback) {
  if (err < NUM_ERRORS) {
    const char *msg = status_msg_dispatch[err];
    if (msg) {
      return msg;
    }
  }
  return fallback ? fallback : "Unknown error";
}
