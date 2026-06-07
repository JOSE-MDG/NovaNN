/**
 * @file string_builder.c
 * @brief Implementation of the standard-malloc string builder.
 *
 * @details
 * Uses malloc/realloc for all internal buffer management -- never the
 * Rust FFI allocator.  The buffer doubles in size whenever appending
 * would exceed capacity.
 *
 * sb_appendf uses a two-phase approach: first vsnprintf with a NULL
 * destination to determine the required length, then vsnprintf into
 * the buffer after ensuring sufficient capacity.
 */

#include "string_builder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Initialise a new StringBuilder.
 *
 * @param[out] sb          Uninitialised struct.
 * @param[in]  initial_cap Suggested: 256.
 */
void sb_init(StringBuilder *sb, size_t initial_cap) {
  sb->buf = (char *)malloc(initial_cap);
  if (sb->buf) {
    sb->buf[0] = '\0';
    sb->len = 0;
    sb->cap = initial_cap;
  } else {
    sb->len = 0;
    sb->cap = 0;
  }
}

/**
 * @brief Grow the buffer so that at least `needed` bytes are available.
 *
 * @param[in] sb     Output StringBuilder.
 * @param[in] needed Minimum required capacity.
 */
static void sb_grow(StringBuilder *sb, size_t needed) {
  if (sb->cap == 0) {
    needed = needed < 256 ? 256 : needed;
    sb->buf = (char *)malloc(needed);
    if (sb->buf) {
      sb->buf[0] = '\0';
      sb->cap = needed;
    }
    return;
  }
  size_t new_cap = sb->cap;
  while (new_cap < needed) {
    new_cap *= 2;
  }
  char *new_buf = (char *)realloc(sb->buf, new_cap);
  if (new_buf) {
    sb->buf = new_buf;
    sb->cap = new_cap;
  }
}

/**
 * @brief Append a null-terminated string.
 *
 * @param[in] sb  Output StringBuilder.
 * @param[in] str Null-terminated string to append (may be NULL).
 */
void sb_append(StringBuilder *sb, const char *str) {
  if (!str) {
    return;
  }
  size_t slen = strlen(str);
  if (sb->len + slen + 1 > sb->cap) {
    sb_grow(sb, sb->len + slen + 1);
  }
  if (sb->buf) {
    memcpy(sb->buf + sb->len, str, slen + 1);
    sb->len += slen;
  }
}

/**
 * @brief Append a printf-formatted string.
 *
 * Uses a two-phase approach: first vsnprintf with NULL to measure length,
 * then vsnprintf into the buffer after ensuring sufficient capacity.
 *
 * @param[in] sb  Output StringBuilder.
 * @param[in] fmt printf format string.
 */
void sb_appendf(StringBuilder *sb, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  int n = vsnprintf(NULL, 0, fmt, args);
  va_end(args);
  if (n < 0) {
    return;
  }
  size_t needed = sb->len + (size_t)n + 1;
  if (needed > sb->cap) {
    sb_grow(sb, needed);
  }
  if (sb->buf) {
    va_start(args, fmt);
    vsnprintf(sb->buf + sb->len, (size_t)n + 1, fmt, args);
    va_end(args);
    sb->len += (size_t)n;
  }
}

/**
 * @brief Append a single character.
 *
 * @param[in] sb Output StringBuilder.
 * @param[in] c  Character to append.
 */
void sb_append_char(StringBuilder *sb, char c) {
  if (sb->len + 2 > sb->cap) {
    sb_grow(sb, sb->len + 2);
  }
  if (sb->buf) {
    sb->buf[sb->len++] = c;
    sb->buf[sb->len] = '\0';
  }
}

/**
 * @brief Append a character repeated N times.
 *
 * @param[in] sb Output StringBuilder.
 * @param[in] c  Character to repeat.
 * @param[in] n  Number of repetitions.
 */
void sb_append_repeated(StringBuilder *sb, char c, size_t n) {
  if (n == 0) {
    return;
  }
  if (sb->len + n + 1 > sb->cap) {
    sb_grow(sb, sb->len + n + 1);
  }
  if (sb->buf) {
    memset(sb->buf + sb->len, c, n);
    sb->len += n;
    sb->buf[sb->len] = '\0';
  }
}

/**
 * @brief Transfer buffer ownership to the caller.
 *
 * @param[in] sb Output StringBuilder.
 * @return Heap-allocated null-terminated string.
 */
char *sb_build(StringBuilder *sb) {
  char *result = sb->buf;
  sb->buf = NULL;
  sb->len = 0;
  sb->cap = 0;
  return result;
}

/**
 * @brief Free the internal buffer without transferring ownership.
 *
 * @param[in] sb Output StringBuilder.
 */
void sb_free(StringBuilder *sb) {
  free(sb->buf);
  sb->buf = NULL;
  sb->len = 0;
  sb->cap = 0;
}
