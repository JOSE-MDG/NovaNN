/**
 * @file mtx.h
 * @brief Portable mutex macros for short critical sections.
 *
 * @details
 * The Windows implementation needs @c CRITICAL_SECTION, which comes from
 * @c <windows.h>.  That header defines macros that collide with
 * ordinary identifiers (@c small, @c near, @c far, @c interface,
 * @c IN, @c OUT), so it stays in a dedicated header instead of the
 * shared one, which every translation unit includes.  @c min and
 * @c max are excluded separately with @c NOMINMAX.
 *
 * @see macros.h  Shared C/C++ macros.
 */

#pragma once

/**
 * @def MTX_T
 * @brief Portable mutex type for short critical sections.
 *
 * @details
 * C11 @c mtx_t from @c <threads.h> on Linux, @c CRITICAL_SECTION from
 * @c <windows.h> on Windows.  Declare the storage directly and
 * initialize it once before use: @c mtx_init with @c mtx_plain on
 * Linux, @c InitializeCriticalSection on Windows.
 *
 * @see MTX_LOCK
 * @see MTX_UNLOCK
 */

/**
  * @def MTX_LOCK(mutex)
  * @brief Locks a portable mutex.
  *
  * @details
  * Blocks until the mutex is acquired.  The return value differs per
  * platform (@c int on Linux, @c void on Windows), so callers must not
  * use it.
  *
  * @param mutex  Pointer to @c MTX_T storage (@c MTX_T*).
  *
  * @see MTX_T
  * @see MTX_UNLOCK
  */

/**
  * @def MTX_UNLOCK(mutex)
  * @brief Unlocks a portable mutex.
  *
  * @details
  * Every lock needs a matching unlock on the same mutex, from the same
  * thread.
  *
  * @param mutex  Pointer to @c MTX_T storage (@c MTX_T*).
  *
  * @see MTX_T
  * @see MTX_LOCK
  */
#ifdef __linux__
#include <threads.h>
#define MTX_T mtx_t
#define MTX_LOCK(mutex) mtx_lock((mutex))
#define MTX_UNLOCK(mutex) mtx_unlock((mutex))
#elif defined(_WIN64)
#include <windows.h>
#define MTX_T CRITICAL_SECTION
#define MTX_LOCK(mutex) EnterCriticalSection((mutex))
#define MTX_UNLOCK(mutex) LeaveCriticalSection((mutex))
#endif
