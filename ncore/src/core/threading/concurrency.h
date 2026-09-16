/**
 * @file concurrency.h
 * @brief Internal platform-specific logical thread-count query.
 *
 * @details
 * Declares @ref get_num_logical_threads_impl(), the single entry
 * point through which @ref get_num_logical_threads() obtains the
 * number of logical threads available to the process.  The
 * implementation is platform-specific and lives in @ref concurrency.c.
 *
 * @see concurrency.c   Implementation for Linux, Windows, and
 *                      unsupported platforms.
 * @see threads.c       Public API consumer.
 * @see status.h        novaStatus_t result reporting.
 */

#pragma once

#include <ncore/core/dtype.h>
#include <ncore/core/status.h>

/**
  * @brief Query the number of logical threads available to the
  *        process.
  *
  * @details
  * Platform-specific implementation:
  * @li Linux: counts the CPUs in the process CPU affinity mask via
  *   @c sched_getaffinity().
  * @li Windows: counts the logical CPUs in the process affinity
  *   state. Single processor group: popcount of the process affinity
  *   mask. Multiple groups: exact per-group process masks collected
  *   by a helper thread visiting each process group.
  * @li Other platforms: returns @c 0 with
  *   @ref novaOsPlatformNotSupported.
  *
  * @param[out] status  Receives the result.  Set to @ref novaSuccess
  *                     on success, or to an error code (e.g.,
  *                     @ref novaInvalidNumThreads or
  *                     @ref novaOsPlatformNotSupported) on failure.
  *
  * @return The number of logical threads available to the process, or
  *         @c 0 on error.
  *
  * @see get_num_logical_threads()  Public wrapper.
  */
uint32 get_num_logical_threads_impl(novaStatus_t *status);
