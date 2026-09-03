/**
 * @file concurrency.c
 * @brief Platform-specific logical thread-count implementations.
 *
 * @details
 * Provides the implementation of @ref get_num_logical_threads_impl()
 * for each supported platform:
 * @li Linux: counts the CPUs in the process CPU affinity mask using
 *   @c sched_getaffinity(). This reflects logical CPUs (SMT/Hyperthreading
 *   siblings count separately) actually schedulable by this process —
 *   i.e. it respects cgroups / taskset / container CPU restrictions.
 * @li Windows: counts the set bits of the process affinity mask using
 *   @c GetProcessAffinityMask(). Same "logical, affinity-aware" semantics
 *   as the Linux path.
 * @li Unsupported platforms: fails with @ref novaOsPlatformNotSupported.
 *
 * @note This intentionally does NOT report physical core count. On SMT
 *   systems each physical core exposes multiple logical CPUs, and all of
 *   them are counted here — which is what you want when sizing a
 *   thread pool, since that's the number of hardware contexts the
 *   scheduler can actually run this process on concurrently.
 *
 * The function is only invoked through @ref get_num_logical_threads()
 * in @ref threads.c.
 *
 * @see concurrency.h  Declaration and public documentation.
 * @see threads.c      Public entry point.
 * @see status.h       novaStatus_t result reporting.
 */

#ifdef __linux__
#define _GNU_SOURCE

#include <sched.h>
#include <threads.h>

#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/threading/threads.h>

#include "concurrency.h"

/**
 * @brief Count logical CPUs from the Linux affinity mask.
 *
 * @details
 * Queries the calling thread's affinity with @c sched_getaffinity()
 * and counts set bits via @c CPU_COUNT(). Returns @c 0 on failure so
 * the caller can report @ref novaInvalidNumThreads.
 *
 * @return Number of affinity-allowed logical CPUs, or @c 0 on error.
 *
 * @see get_num_logical_threads_impl()  Public wrapper for this helper.
 */
static inline uint32 get_num_threads_gnu_impl() {
  cpu_set_t cpuset = {};
  if (sched_getaffinity(0, sizeof(cpuset), &cpuset) != 0) {
    return 0;
  }
  return (uint32)CPU_COUNT(&cpuset);
}

/**
  * @brief Linux implementation of the logical thread-count query.
  *
  * @details
  * Counts the CPUs in the process affinity mask via
  * @c sched_getaffinity(). A count of @c 0 is treated as a failure
  * and reported with @ref novaInvalidNumThreads.
  *
  * @param[out] status  Receives the result.  Set to @ref novaSuccess
  *                     on success, or to @ref novaInvalidNumThreads if
  *                     no CPUs are available to the process, or if
  *                     @c sched_getaffinity() itself failed.
  *
  * @return The number of logical threads available to the process, or
  *         @c 0 on error.
  */
uint32 get_num_logical_threads_impl(novaStatus_t *status) {

  auto num_threads = get_num_threads_gnu_impl();

  if (num_threads == 0) {
    status->err = novaInvalidNumThreads;
    status->message = "Error obtaining the number of logical threads; "
                      "sched_getaffinity() failed";
    return 0;
  }

  status->err = novaSuccess;
  status->message = nova_get_error_msg(status->err, nullptr);
  return num_threads;
}

#elif defined(_WIN64)
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/threading/threads.h>
#include <windows.h>

#include "concurrency.h"

/**
 * @brief Count logical CPUs from the Windows affinity mask.
 *
 * @details
 * Queries the process affinity with @c GetProcessAffinityMask() and
 * counts set bits via @c __builtin_popcountll(). Returns @c 0 on
 * failure so the caller can report @ref novaInvalidNumThreads.
 *
 * @return Number of affinity-allowed logical CPUs, or @c 0 on error.
 *
 * @see get_num_logical_threads_impl()  Public wrapper for this helper.
 */
static inline uint32 get_num_threads_windows_impl() {
  DWORD_PTR process_mask;
  DWORD_PTR system_mask;

  if (!GetProcessAffinityMask(GetCurrentProcess(), &process_mask,
                              &system_mask)) {
    return 0;
  }
  return (uint32)__builtin_popcountll(process_mask);
}

/**
  * @brief Windows implementation of the logical thread-count query.
  *
  * @details
  * Counts the set bits of the process affinity mask via
  * @c GetProcessAffinityMask(). A count of @c 0 is treated as a
  * failure and reported with @ref novaInvalidNumThreads.
  *
  * @param[out] status  Receives the result.  Set to @ref novaSuccess
  *                     on success, or to @ref novaInvalidNumThreads if
  *                     no CPUs are available to the process, or if
  *                     @c GetProcessAffinityMask() itself failed.
  *
  * @return The number of logical threads available to the process, or
  *         @c 0 on error.
  */
uint32 get_num_logical_threads_impl(novaStatus_t *status) {

  auto num_threads = get_num_threads_windows_impl();

  if (num_threads == 0) {
    status->err = novaInvalidNumThreads;
    status->message = "Error obtaining the number of logical threads; "
                      "GetProcessAffinityMask() failed";
    return 0;
  }

  status->err = novaSuccess;
  status->message = nova_get_error_msg(status->err, nullptr);
  return num_threads;
}

#else
/**
  * @brief Fallback implementation for unsupported platforms.
  *
  * @details
  * Reports @ref novaOsPlatformNotSupported and returns @c 0 without
  * querying any runtime API.
  *
  * @param[out] status  Receives @ref novaOsPlatformNotSupported.
  *
  * @return Always @c 0.
  */
uint32 get_num_logical_threads_impl(novaStatus_t *status) {
  status->err = novaOsPlatformNotSupported;
  status->message = nova_get_error_msg(status->err, nullptr);
  return 0;
}
#endif
