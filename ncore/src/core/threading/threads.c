/**
 * @file threads.c
 * @brief Global and per-group thread-count management.
 */

#include <stdatomic.h>

#include <ncore/core/status.h>
#include <ncore/threading/threads.h>

#include "concurrency.h"
#include "groups/autograd.h"
#include "groups/compute.h"
#include "groups/dtloader.h"
#include "manager.h"

/**
 * @var atomic_global_thread_counter
 * @brief Global thread budget.
 *
 * @details Determines the total number of threads the library is
 * allowed to use.
 */
static _Atomic uint32 atomic_global_thread_counter = MIN_THREADS_PER_GROUP;

/**
 * @var atomic_global_thread_counter_initialized
 * @brief Whether the global budget was explicitly set.
 */
static _Atomic bool atomic_global_thread_counter_initialized = false;

/**
 * @brief Return the number of logical threads available to the
 *        process.
 *
 * @param[out] status  Receives the result. Set to @ref novaSuccess
 *                     on success, or an error code on failure.
 *
 * @return Number of logical threads, or @c 0 on error.
 */
uint32 get_num_logical_threads(novaStatus_t *status) {
  return get_num_logical_threads_impl(status);
}

/**
 * @brief Return the globally configured thread budget.
 *
 * @param[out] status  Receives @ref novaSuccess on success. May be
 *                     @c nullptr.
 *
 * @return Current global thread budget.
 */
uint32 get_configured_num_logical_threads(novaStatus_t *status) {
  auto value =
      atomic_load_explicit(&atomic_global_thread_counter, memory_order_relaxed);
  if (status != nullptr) {
    status->err = novaSuccess;
    status->message = nova_get_error_msg(novaSuccess, nullptr);
  }
  return value;
}

/**
 * @brief Set the global number of logical threads.
 *
 * @param[in] threads  Total threads to use. Must be at least
 *                     @ref MIN_THREADS_PER_GROUP.
 *
 * @return @ref novaStatus_t with @c err set to @ref novaSuccess on
 *         success, or @ref novaInvalidNumThreads when @p threads is
 *         less than @ref MIN_THREADS_PER_GROUP.
 *
 * @pre  @p threads must be at least @ref MIN_THREADS_PER_GROUP.
 */
novaStatus_t set_num_logical_threads(uint32 threads) {
  if (threads < MIN_THREADS_PER_GROUP) {
    return (novaStatus_t){
        .err = novaInvalidNumThreads,
        .message = nova_get_error_msg(novaInvalidNumThreads, nullptr),
    };
  }
  atomic_store_explicit(&atomic_global_thread_counter, threads,
                        memory_order_relaxed);
  atomic_store_explicit(&atomic_global_thread_counter_initialized, true,
                        memory_order_release);
  return (novaStatus_t){
      .err = novaSuccess,
      .message = nova_get_error_msg(novaSuccess, nullptr),
  };
}

typedef novaStatus_t (*set_threads_func_t)(uint32);
typedef uint32 (*get_threads_func_t)();

/**
 * @var set_threads_funcs
 * @brief Dispatch table for per-group setters.
 */
static const set_threads_func_t set_threads_funcs[NUM_PARALLEL_GROUPS] = {
    [ParallelComputeGroup] = set_compute_threads,
    [ParallelAutogradGroup] = set_autograd_threads,
    [ParallelDTLoaderGroup] = set_dtloader_threads,
};

/**
 * @var get_threads_funcs
 * @brief Dispatch table for per-group getters.
 */
static const get_threads_func_t get_threads_funcs[NUM_PARALLEL_GROUPS] = {
    [ParallelComputeGroup] = get_compute_threads,
    [ParallelAutogradGroup] = get_autograd_threads,
    [ParallelDTLoaderGroup] = get_dtloader_threads,
};

/**
 * @brief Return the number of threads assigned to a group.
 *
 * @param[in]  group   Group to query.
 * @param[out] status  Receives the result.
 *
 * @return Number of threads for @p group, or @c 0 on failure.
 *
 * @pre  @p status must not be @c nullptr.
 */
uint32 get_num_threads_from(ParallelGroups group, novaStatus_t *status) {
  if (group >= NUM_PARALLEL_GROUPS) {
    *status = (novaStatus_t){
        .err = novaInvalidParallelGroup,
        .message = nova_get_error_msg(novaInvalidParallelGroup, nullptr),
    };
    return 0;
  }
  return get_threads_funcs[group]();
}

/**
 * @brief Assign a thread count to a group.
 *
 * @param[in] group    Group to configure.
 * @param[in] threads  Threads to assign. Must be at least
 *                     @ref MIN_THREADS_PER_GROUP.
 *
 * @return @ref novaStatus_t with @c err set to @ref novaSuccess on
 *         success, or an error when @p group or @p threads is
 *         invalid.
 */
novaStatus_t set_num_threads_to(ParallelGroups group, uint32 threads) {
  if (group >= NUM_PARALLEL_GROUPS) {
    return (novaStatus_t){
        .err = novaInvalidParallelGroup,
        .message = nova_get_error_msg(novaInvalidParallelGroup, nullptr),
    };
  }
  atomic_store_explicit(&atomic_global_thread_counter_initialized, true,
                        memory_order_release);
  return set_threads_funcs[group](threads);
}

/**
 * @brief Query whether the global thread budget was set.
 *
 * @return @c true if configured, @c false otherwise.
 */
bool is_global_thread_count_initialized() {
  return atomic_load_explicit(&atomic_global_thread_counter_initialized,
                              memory_order_acquire);
}

/**
 * @brief Report whether a stratified budget holds no empty group.
 *
 * @details
 * Checks that every member of @p strat (@c compute, @c autograd,
 * @c dtloader) is nonzero. A zero member marks either the failure
 * sentinel @c {0,0,0} or a budget that would starve a pool, so both
 * are rejected. The check reads plain struct members and touches no
 * shared state.
 *
 * @param[in] strat  Budget to validate. Must not be @c nullptr.
 *
 * @return @c true when every member is nonzero, @c false otherwise.
 *
 * @pre  @p strat must not be @c nullptr.
 *
 * @see distribute_stratified_threads()  Applies a validated budget.
 * @see StratifiedThreads                Budget under validation.
 */
bool is_valid_stratification_result(const StratifiedThreads *strat) {
  return ((strat->compute != 0) && (strat->autograd != 0) &&
          (strat->dtloader != 0)) != 0;
}

/**
 * @brief Report whether the latest stratification result is usable.
 *
 * @details
 * Fetches the budget recorded by the latest @ref stratify_threads()
 * call via @ref get_last_stratification_result() and validates it
 * with @ref is_valid_stratification_result(). Before any successful
 * stratification the recorded result is @c {0,0,0}, so this returns
 * @c false.
 *
 * @return @c true when the recorded budget holds no empty group,
 *         @c false otherwise.
 *
 * @note Thread-safe. The recorded budget is read with acquire
 *       ordering, pairing with the release-ordered store in
 *       @ref stratify_threads().
 *
 * @see is_valid_stratification_result()  Validation applied.
 * @see get_last_stratification_result()  Recorded budget source.
 */
bool is_valid_latest_stratification_result() {
  auto result = get_last_stratification_result();
  return is_valid_stratification_result(&result);
}

/**
 * @brief Build a stratified thread budget for all groups.
 *
 * @details
 * Resolves the total thread budget: when the global budget was never
 * configured (@ref is_global_thread_count_initialized() returns
 * @c false) the total comes from @ref get_num_logical_threads()
 * (hardware affinity); otherwise it comes from
 * @ref get_configured_num_logical_threads() (user configuration).
 * The first call partitions the total via @ref stratify_threads(),
 * which records its result; later calls return the recorded budget
 * through @ref get_last_stratification_result() without recomputing,
 * since the global budget is immutable once set.
 *
 * @param[out] status  Receives the operation result. Set to
 *                     @ref novaSuccess on success, or to the error
 *                     produced by the underlying total query or by
 *                     the split. Must not be @c nullptr.
 *
 * @return Stratified budget, or @c {0,0,0} on failure.
 *
 * @pre  @p status must not be @c nullptr.
 *
 * @see stratify_threads()  Partitioning and recording.
 * @see is_stratification_complete()  Cache state query.
 * @see get_last_stratification_result()  Cached result source.
 */
StratifiedThreads get_stratified_threads(novaStatus_t *status) {

  uint32 total_threads;
  if (!is_global_thread_count_initialized()) {
    total_threads = get_num_logical_threads(status);
    if (status->err != novaSuccess) {
      return (StratifiedThreads){0, 0, 0};
    }
  } else {
    total_threads = get_configured_num_logical_threads(status);
    if (status->err != novaSuccess) {
      return (StratifiedThreads){0, 0, 0};
    }
  }

  if (!is_stratification_complete()) {
    return stratify_threads(total_threads, status);
  }
  // Since the global budget is immutable, the last stratification result is valid.
  return get_last_stratification_result();
}

/**
 * @brief Apply a stratified budget to the per-group counters.
 *
 * @details
 * Validates @p strat with @ref is_valid_stratification_result()
 * before touching any counter, then assigns its members to the
 * groups in @ref ParallelGroups order (compute, autograd, dtloader)
 * through @ref set_num_threads_to(). Assignment stops at the first
 * failing group and reports its error; groups assigned before the
 * failure keep their new values, since there is no rollback.
 *
 * @param[in] strat  Budget to apply. Must not be @c nullptr and must
 *                   hold no empty group.
 *
 * @return @ref novaStatus_t with @c err set to @ref novaSuccess on
 *         success, @ref novaInvalidPointer when @p strat is
 *         @c nullptr, @ref novaInvalidValue when the budget holds an
 *         empty group, or the error of the first failing per-group
 *         assignment.
 *
 * @pre  @p strat must not be @c nullptr.
 *
 * @see is_valid_stratification_result()  Validation applied.
 * @see set_num_threads_to()              Per-group assignment.
 * @see ParallelGroups                    Group order applied.
 */
novaStatus_t distribute_stratified_threads(const StratifiedThreads *strat) {
  if (strat == nullptr) {
    return (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Null stratification pointer.",
    };
  }

  if (!is_valid_stratification_result(strat)) {
    return (novaStatus_t){
        .err = novaInvalidValue,
        .message = "Invalid stratification result: zero threads in a group.",
    };
  }

  auto st = set_num_threads_to(ParallelComputeGroup, strat->compute);
  if (st.err != novaSuccess) {
    return st;
  }
  st = set_num_threads_to(ParallelAutogradGroup, strat->autograd);
  if (st.err != novaSuccess) {
    return st;
  }
  return set_num_threads_to(ParallelDTLoaderGroup, strat->dtloader);
}
