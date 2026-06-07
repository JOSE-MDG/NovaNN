/**
 * @file string_builder.h
 * @brief Dynamic string builder using standard malloc/realloc.
 *
 * @details
 * Provides a growable character buffer for incremental string construction.
 * Uses malloc/realloc internally -- never the Rust FFI allocator.
 *
 * Growth strategy: double capacity on every expansion.  Initial capacity
 * of 256 bytes is recommended.
 *
 * After sb_build() the StringBuilder is invalidated (ownership transfers
 * to the caller).  Call sb_free() instead if discarding mid-build.
 */

#pragma once

#include <stdarg.h>
#include <stddef.h>

/**
 * @struct StringBuilder
 * @brief Growable character buffer.
 */
typedef struct {
  char *buf;  ///< Heap-allocated buffer (null-terminated).
  size_t len; ///< Length of the string (excl. null).
  size_t cap; ///< Allocated capacity (incl. null).
} StringBuilder;

/**
 * @brief Initialise a new StringBuilder.
 *
 * Allocates an internal buffer of initial_cap bytes.
 *
 * @param[out] sb          Uninitialised struct.
 * @param[in]  initial_cap Suggested: 256.
 */
void sb_init(StringBuilder *sb, size_t initial_cap);

/**
 * @brief Append a null-terminated string.
 *
 * @param[in] sb  Output StringBuilder.
 * @param[in] str Null-terminated string to append (may be NULL).
 */
void sb_append(StringBuilder *sb, const char *str);

/**
 * @brief Append a printf-formatted string.
 *
 * @param[in] sb  Output StringBuilder.
 * @param[in] fmt printf format string.
 */
void sb_appendf(StringBuilder *sb, const char *fmt, ...);

/**
 * @brief Append a single character.
 *
 * @param[in] sb Output StringBuilder.
 * @param[in] c  Character to append.
 */
void sb_append_char(StringBuilder *sb, char c);

/**
 * @brief Append a character repeated N times.
 *
 * @param[in] sb Output StringBuilder.
 * @param[in] c  Character to repeat.
 * @param[in] n  Number of repetitions.
 */
void sb_append_repeated(StringBuilder *sb, char c, size_t n);

/**
 * @brief Transfer buffer ownership to the caller.
 *
 * After this call the StringBuilder is invalid.  The caller must
 * free() the returned pointer.
 *
 * @param[in] sb Output StringBuilder.
 * @return Heap-allocated null-terminated string.
 */
char *sb_build(StringBuilder *sb);

/**
 * @brief Free the internal buffer without transferring ownership.
 *
 * Call this instead of sb_build() if discarding the result.
 *
 * @param[in] sb Output StringBuilder.
 */
void sb_free(StringBuilder *sb);
