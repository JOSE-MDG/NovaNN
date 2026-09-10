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
 * @li Windows: counts the logical CPUs in the process affinity mask.
 *   Single processor group: popcount of @c GetProcessAffinityMask().
 *   Multiple groups (>64 logical CPUs): a helper thread visits each
 *   process group via @c SetThreadGroupAffinity() and reads the exact
 *   per-group process mask the same way. Same "logical,
 *   affinity-aware" semantics as the Linux path.
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
 *
 * @note A null @p status is tolerated defensively and reported as
 *       @c 0. Prefer passing a valid status.
 */
uint32 get_num_logical_threads_impl(novaStatus_t *status) {

  if (status == nullptr) {
    return 0;
  }

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
#include <stdlib.h>

#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/threading/threads.h>
#include <windows.h>

#include "concurrency.h"

#if defined(_WIN32_WINNT) && (_WIN32_WINNT < 0x0601)
#error "NovaNN threading requires _WIN32_WINNT >= 0x0601"
#endif

/**
 * @brief Context for the multi-group affinity walk worker.
 *
 * @details
 * Carries the borrowed group list into
 * @c count_group_affinity_worker() and the accumulated result back
 * out. Lives on the spawner's stack; the worker is always joined
 * before it goes out of scope.
 */
typedef struct {
  const USHORT *groups; ///< Process groups to visit (borrowed).
  USHORT group_count;   ///< Number of entries in @c groups.
  uint32 total;         ///< Accumulated process-mask bits.
} WindowsGroupWalk;

/**
 * @brief Pin the calling thread into one processor group.
 *
 * @details
 * Tries the whole-group mask first (the common unrestricted case,
 * one syscall). When the process is affinity-restricted inside the
 * group that pin fails, so single CPUs are probed in turn; a single
 * success reveals the full process mask for the group because
 * @c GetProcessAffinityMask() reports the process mask, not the
 * thread mask.
 *
 * @param group  Processor group to enter.
 *
 * @return Nonzero when the thread now runs in @p group.
 */
static BOOL pin_thread_to_group(USHORT group) {
  GROUP_AFFINITY target = {
      .Mask = (~(KAFFINITY)0),
      .Group = group,
  };
  if (SetThreadGroupAffinity(GetCurrentThread(), &target, nullptr)) {
    return TRUE;
  }
  for (uint32 bit = 0; bit < 8u * (uint32)sizeof(KAFFINITY); ++bit) {
    GROUP_AFFINITY probe = {
        .Mask = (KAFFINITY)((KAFFINITY)1u << bit),
        .Group = group,
    };
    if (SetThreadGroupAffinity(GetCurrentThread(), &probe, nullptr)) {
      return TRUE;
    }
  }
  return FALSE;
}

/**
 * @brief Count process-mask bits across groups on a helper thread.
 *
 * @details
 * Runs on a transient thread with default affinity so it can enter
 * every process group even when the spawner's thread is pinned, and
 * so the spawner's own affinity is never disturbed. Reads the exact
 * per-group process mask: for the current process
 * @c GetProcessAffinityMask() resolves against the calling thread's
 * primary group.
 *
 * @param arg  @c WindowsGroupWalk with the groups to visit; receives
 *             the accumulated count.
 *
 * @return @c 0 on success, nonzero on the first failure (fail loud:
 *         a guessed pool size is worse than @ref novaInvalidNumThreads).
 */
static DWORD WINAPI count_group_affinity_worker(LPVOID arg) {
  WindowsGroupWalk *walk = arg;
  for (USHORT i = 0; i < walk->group_count; ++i) {
    if (!pin_thread_to_group(walk->groups[i])) {
      return 1;
    }
    DWORD_PTR process_mask = 0;
    DWORD_PTR system_mask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &process_mask,
                                &system_mask)) {
      return 1;
    }
    walk->total += (uint32)__builtin_popcountll(process_mask);
  }
  return 0;
}

/**
 * @brief Count logical CPUs from the Windows affinity state.
 *
 * @details
 * Single processor group: popcount of @c GetProcessAffinityMask(),
 * exact including restricted affinity. Multiple groups: list the
 * process groups with @c GetProcessGroupAffinity() and sum the exact
 * per-group process masks via @c count_group_affinity_worker() on a
 * helper thread. @c GetProcessAffinityMask() alone cannot serve the
 * multi-group case: it zeroes both masks when the process spans
 * groups, and otherwise reports only the primary group. Returns
 * @c 0 on any failure so the caller can report
 * @ref novaInvalidNumThreads.
 *
 * @return Number of affinity-allowed logical CPUs, or @c 0 on error.
 *
 * @see get_num_logical_threads_impl()  Public wrapper for this helper.
 */
static inline uint32 get_num_threads_windows_impl() {

  if (GetActiveProcessorGroupCount() <= 1) {
    DWORD_PTR process_mask;
    DWORD_PTR system_mask;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &process_mask,
                                &system_mask)) {
      return 0;
    }
    return (uint32)__builtin_popcountll(process_mask);
  }

  USHORT group_count = 0;

  if (GetProcessGroupAffinity(GetCurrentProcess(), &group_count, nullptr) ||
      GetLastError() != ERROR_INSUFFICIENT_BUFFER || group_count == 0) {
    return 0;
  }

  USHORT *groups = malloc(group_count * sizeof(*groups));
  if (groups == nullptr) {
    return 0;
  }

  if (!GetProcessGroupAffinity(GetCurrentProcess(), &group_count, groups)) {
    free(groups);
    return 0;
  }

  WindowsGroupWalk walk = {
      .groups = groups,
      .group_count = group_count,
      .total = 0,
  };

  uint32 total = 0;
  HANDLE worker =
      CreateThread(nullptr, 0, count_group_affinity_worker, &walk, 0, nullptr);
  if (worker != nullptr) {
    DWORD exit_code = 1;
    if (WaitForSingleObject(worker, INFINITE) == WAIT_OBJECT_0 &&
        GetExitCodeThread(worker, &exit_code) && exit_code == 0) {
      total = walk.total;
    }
    CloseHandle(worker);
  }

  free(groups);
  return total;
}

/**
 * @brief Windows implementation of the logical thread-count query.
 *
 * @details
 * Counts the logical CPUs in the process affinity state via
 * @c get_num_threads_windows_impl() (group-aware: exact per-group
 * process masks). A count of @c 0 is treated as a failure and
 * reported with @ref novaInvalidNumThreads.
 *
 * @param[out] status  Receives the result.  Set to @ref novaSuccess
 *                     on success, or to @ref novaInvalidNumThreads if
 *                     no CPUs are available to the process, or if the
 *                     affinity query itself failed.
  *
 * @return The number of logical threads available to the process, or
 *         @c 0 on error.
 *
 * @note A null @p status is tolerated defensively and reported as
 *       @c 0. Prefer passing a valid status.
 */
uint32 get_num_logical_threads_impl(novaStatus_t *status) {

  if (status == nullptr) {
    return 0;
  }

  auto num_threads = get_num_threads_windows_impl();

  if (num_threads == 0) {
    status->err = novaInvalidNumThreads;
    status->message = "Error obtaining the number of logical threads; "
                      "Windows affinity query failed";
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
 *
 * @note A null @p status is tolerated defensively and reported as
 *       @c 0. Prefer passing a valid status.
 */
uint32 get_num_logical_threads_impl(novaStatus_t *status) {
  if (status == nullptr) {
    return 0;
  }
  status->err = novaOsPlatformNotSupported;
  status->message = nova_get_error_msg(status->err, nullptr);
  return 0;
}
#endif
