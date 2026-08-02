/**
 * @file string_builder.c
 * @brief Dynamic growable string builder implementation.
 *
 * @details
 * Implements a standard memory-safe string builder that minimizes
 * heap reallocations through a geometric growth strategy. Provides
 * a subset of common string operations (append, append-format,
 * repeated characters) optimized for incremental construction of
 * large tensor representations.
 *
 * @section architecture Architecture
 *
 * @li Geometric Growth: The buffer capacity doubles whenever an
 *   append operation would exceed the current limit.
 * @li Two-Phase Formatting: @ref sb_appendf first measures the
 *   required length using @c "vsnprintf(nullptr, ...)" to ensure a
 *   single, perfectly-sized reallocation.
 * @li Ownership Model: Ownership of the internal buffer is
 *   transferred to the caller via @ref sb_build(), after which the
 *   builder instance is reset.
 * @li Error Propagation: Allocation failures set @ref SbErrOom on
 *   the builder's status field. Once set, all subsequent append
 *   operations become no-ops and @ref sb_build() returns @c nullptr.
 *
 * @see string_builder.h  Public descriptor and API.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "string_builder.h"

/**
 * @brief Initialise a StringBuilder with a specific starting capacity.
 */
void sb_init(StringBuilder *sb, size_t initial_cap) {
  sb->buf = (char *)malloc(initial_cap);
  if (sb->buf != nullptr) {
    sb->buf[0] = '\0';
    sb->len = 0;
    sb->cap = initial_cap;
    sb->status = SbOk;
  } else {
    sb->buf = nullptr;
    sb->len = 0;
    sb->cap = 0;
    sb->status = SbErrOom;
  }
}

/**
 * @brief Internal helper to grow the heap buffer.
 *
 * @details
 * Applies the geometric growth strategy (2x). If allocation fails,
 * sets @ref SbErrOom on the builder's status and leaves the buffer
 * unchanged to prevent memory leaks.
 */
static void sb_grow(StringBuilder *sb, size_t needed) {
  size_t new_cap = (sb->cap == 0) ? 256 : sb->cap;
  while (new_cap < needed) {
    size_t doubled = new_cap * 2;
    if (doubled < new_cap) {
      sb->status = SbErrOom;
      return;
    }
    new_cap = doubled;
  }

  char *new_buf = (char *)realloc(sb->buf, new_cap);
  if (new_buf == nullptr) {
    sb->status = SbErrOom;
    return;
  }
  sb->buf = new_buf;
  sb->cap = new_cap;
}

/**
 * @brief Retrieve the current error status of the builder.
 */
SBStatus sb_get_status(const StringBuilder *sb) { return sb->status; }

/**
 * @brief Append a null-terminated string to the builder.
 */
void sb_append(StringBuilder *sb, const char *str) {
  if (sb->status != SbOk || str == nullptr || sb->buf == nullptr) {
    return;
  }
  size_t slen = strlen(str);
  if ((sb->len + slen + 1) > sb->cap) {
    sb_grow(sb, sb->len + slen + 1);
  }
  if (sb->status != SbOk) {
    return;
  }
  memcpy(sb->buf + sb->len, str, slen + 1);
  sb->len += slen;
}

/**
 * @brief Append a printf-formatted string.
 *
 * @details
 * Uses a two-phase measurement and allocation strategy to ensure
 * formatting safety and memory efficiency.
 */
void sb_appendf(StringBuilder *sb, const char *fmt, ...) {
  if (sb->status != SbOk) {
    return;
  }
  va_list args;
  va_start(args, fmt);
  int n = vsnprintf(nullptr, 0, fmt, args);
  va_end(args);
  if (n < 0) {
    sb->status = SbErrOom;
    return;
  }
  size_t needed = sb->len + (size_t)n + 1;
  if (needed > sb->cap) {
    sb_grow(sb, needed);
  }
  if (sb->status != SbOk) {
    return;
  }
  va_start(args, fmt);
  vsnprintf(sb->buf + sb->len, (size_t)n + 1, fmt, args);
  va_end(args);
  sb->len += (size_t)n;
}

/**
 * @brief Append a single character.
 */
void sb_append_char(StringBuilder *sb, char c) {
  if (sb->status != SbOk) {
    return;
  }
  if (sb->len + 2 > sb->cap) {
    sb_grow(sb, sb->len + 2);
  }
  if (sb->status != SbOk) {
    return;
  }
  sb->buf[sb->len++] = c;
  sb->buf[sb->len] = '\0';
}

/**
 * @brief Append a character repeated N times.
 */
void sb_append_repeated(StringBuilder *sb, char c, size_t n) {
  if (sb->status != SbOk || n == 0) {
    return;
  }
  if (sb->len + n + 1 > sb->cap) {
    sb_grow(sb, sb->len + n + 1);
  }
  if (sb->status != SbOk) {
    return;
  }
  memset(sb->buf + sb->len, c, n);
  sb->len += n;
  sb->buf[sb->len] = '\0';
}

/**
 * @brief Finalize and return the built string.
 *
 * @details
 * If the builder is in an error state, the internal buffer is freed
 * and @c nullptr is returned.
 *
 * @return Ownership of the heap-allocated string, or @c nullptr on
 *         error.
 */
char *sb_build(StringBuilder *sb) {
  if (sb->status != SbOk) {
    free(sb->buf);
    sb->buf = nullptr;
    sb->len = 0;
    sb->cap = 0;
    return nullptr;
  }
  char *result = sb->buf;
  sb->buf = nullptr;
  sb->len = 0;
  sb->cap = 0;
  return result;
}

/**
 * @brief Deallocate the builder's internal memory.
 */
void sb_free(StringBuilder *sb) {
  free(sb->buf);
  sb->buf = nullptr;
  sb->len = 0;
  sb->cap = 0;
}
