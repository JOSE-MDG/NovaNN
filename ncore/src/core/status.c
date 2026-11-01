/**
 * @file status.c
 * @brief Error message retrieval.
 *
 * @details
 * Human-readable messages for @ref novaError_t codes, served from the
 * global dispatch table defined in @ref status_dispatch_tables.c.
 *
 * @see status.h                 Public interface and @ref novaError_t definition.
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
 * @brief Retrieve the human-readable message for an error code.
 *
 * @details
 * Validates @p err against the table bounds (@ref NUM_ERRORS) and
 * returns the corresponding message from @ref status_msg_dispatch.
 * Out-of-bounds codes or null table entries return @p fallback, or
 * "Unknown error" if @p fallback is nullptr.
 *
 * @param[in] err      The @ref novaError_t code to look up.
 * @param[in] fallback Custom fallback message.  May be nullptr.
 * @return A constant pointer to the error message string.
 *
 * @note Thread-safe: read-only access to a table initialized at load
 *       time.
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
