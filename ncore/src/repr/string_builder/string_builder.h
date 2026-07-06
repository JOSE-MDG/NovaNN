/**
 * @file string_builder.h
 * @brief Dynamic string builder for incremental string construction.
 *
 * @details
 * Declares a robust, growable character buffer designed for
 * high-performance string assembly. Uses a standard `malloc`/`realloc`
 * strategy with geometric growth to minimize allocation overhead.
 *
 * Used throughout the representation module to avoid the quadratic
 * performance penalty of repeated `strcat` calls.
 *
 * ## Error Propagation
 *
 * The @ref SBStatus member tracks allocation failures. On error, all
 * subsequent append operations become no-ops and the error is
 * propagated to callers via @ref sb_get_status().
 *
 * @see string_builder.c  Implementation details.
 * @see repr_context.h    Consumer of this module.
 */

#pragma once

#include <stdarg.h>
#include <stddef.h>

#include <ncore/headeronly/macros.h>

/**
 * @enum SBStatus
 * @brief Error codes for StringBuilder operations.
 *
 * @details
 * Tracks the health of a @ref StringBuilder instance. Once an error
 * is set, all subsequent append operations become no-ops.
 */
typedef enum {
  SbOk         = 0, ///< No error.
  SbErrOom     = 1, ///< Memory allocation failure (`malloc`/`realloc` returned `nullptr`).
  SbErrOverflow = 2 ///< Size arithmetic overflow.
} SBStatus;

/**
 * @struct StringBuilder
 * @brief Primary descriptor for a growable character buffer.
 */
typedef struct {
  char *buf;       ///< Heap-allocated character buffer (null-terminated).
  size_t len;      ///< Current length of the string (excluding null).
  size_t cap;      ///< Total allocated capacity (including null).
  SBStatus status; ///< Current error status of the builder.
} StringBuilder;

/**
 * @brief Initialise a new StringBuilder instance.
 *
 * @details
 * Allocates an initial heap buffer. Recommended starting capacity
 * is 256 bytes. On allocation failure, the builder enters an error
 * state and all subsequent append operations become no-ops.
 *
 * @param[out] sb          Pointer to an uninitialised StringBuilder.
 * @param[in]  initial_cap Initial allocation size in bytes.
 */
void sb_init(StringBuilder *sb, size_t initial_cap);

/**
 * @brief Retrieve the current error status of the builder.
 *
 * @param[in] sb Pointer to the StringBuilder.
 *
 * @return The current @ref SBStatus code.
 */
SBStatus sb_get_status(const StringBuilder *sb);

/**
 * @brief Append a null-terminated string to the builder.
 *
 * @details
 * Automatically grows the internal buffer if necessary. If the
 * builder is in an error state or memory allocation fails, the
 * operation is silently ignored.
 *
 * @param[in,out] sb  Pointer to the StringBuilder.
 * @param[in]     str Null-terminated string to append. May be
 *                    `nullptr` (no-op).
 */
void sb_append(StringBuilder *sb, const char *str);

/**
 * @brief Append a formatted string (printf-style) to the builder.
 *
 * @details
 * Uses `vsnprintf` to determine the required length, then performs
 * a safe append. If the builder is in an error state, the operation
 * is silently ignored.
 *
 * @param[in,out] sb  Pointer to the StringBuilder.
 * @param[in]     fmt `printf`-compatible format string.
 */
ATTR(format(printf, 2, 3))
void sb_appendf(StringBuilder *sb, const char *fmt, ...);

/**
 * @brief Append a single character to the builder.
 *
 * @details
 * If the builder is in an error state, the operation is silently
 * ignored.
 *
 * @param[in,out] sb Pointer to the StringBuilder.
 * @param[in]     c  The character to append.
 */
void sb_append_char(StringBuilder *sb, char c);

/**
 * @brief Append a character repeated multiple times.
 *
 * @details
 * If the builder is in an error state, the operation is silently
 * ignored.
 *
 * @param[in,out] sb Pointer to the StringBuilder.
 * @param[in]     c  The character to repeat.
 * @param[in]     n  Number of repetitions.
 */
void sb_append_repeated(StringBuilder *sb, char c, size_t n);

/**
 * @brief Finalize the construction and transfer buffer ownership.
 *
 * @details
 * Returns the underlying heap buffer and resets the StringBuilder
 * to an empty state. The caller is responsible for calling
 * `free()` on the returned pointer.
 *
 * @pre  The builder must be in @ref SbOk state. If the builder has
 *       encountered an error, this function returns `nullptr` and
 *       frees the internal buffer.
 *
 * @param[in,out] sb Pointer to the StringBuilder.
 *
 * @return Heap-allocated null-terminated string, or `nullptr` on
 *         error.
 */
char *sb_build(StringBuilder *sb);

/**
 * @brief Free the internal buffer and reset the builder.
 *
 * @details
 * Call this if the representation process fails or if the resulting
 * string is no longer needed.
 *
 * @param[in,out] sb Pointer to the StringBuilder.
 */
void sb_free(StringBuilder *sb);
