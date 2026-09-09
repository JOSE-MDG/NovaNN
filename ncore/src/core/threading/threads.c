/**
 * @file threads.c
 * @brief Global and per-group thread-count management.
 *
 * @details
 * Implements the public threading API declared in @ref threads.h:
 *
 * @li Global budget: @ref atomic_global_thread_counter with its
 *   initialized flag, written by @ref set_num_logical_threads() and
 *   read by @ref get_configured_num_logical_threads().
 * @li Per-group counters: assigned through the @ref set_threads_funcs
 *   dispatch table (@ref set_num_threads_to()) and read through
 *   @ref get_threads_funcs (@ref get_num_threads_from()).
 * @li Stratification: @ref get_stratified_threads() resolves the total
 *   and splits it once via @ref stratify_threads(), serving later
 *   calls from the recorded result;
 *   @ref distribute_stratified_threads() writes a validated budget
 *   into the counters.
 * @li Inspection: @ref print_thread_config() renders the budget, the
 *   recorded split and the live counters to stdout.
 *
 * @section thread-safety Thread Safety
 *
 * All shared state in this unit is atomic. The global budget and its
 * flag pair a relaxed counter with a release/acquire flag, and each
 * per-group counter is only ever loaded or stored atomically, so
 * every function here is safe to call concurrently. The
 * stratification record itself lives in @c manager.c under the same
 * release/acquire discipline.
 *
 * @see threads.h      Public API declarations.
 * @see manager.h      Split policy and last-result cache.
 * @see concurrency.h  Affinity-based thread count query.
 */

#include <inttypes.h>
#include <stdatomic.h>
#include <stdio.h>

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/tensor.h>
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
 * @var atomic_global_budget_set_once
 * @brief Single-shot latch for @ref set_num_logical_threads().
 *
 * @details The global budget may only be configured once per execution.
 * Kept separate from @ref atomic_global_thread_counter_initialized
 * (which per-group setters also raise) so a per-group assignment never
 * blocks the global call. Claimed with an atomic exchange, so
 * concurrent second callers are reliably rejected.
 */
static _Atomic bool atomic_global_budget_set_once = false;

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
 * @details
 * Configures the total thread budget once per execution. A second
 * successful call is impossible by design: the budget feeds the
 * stratification cache, and silently swapping it would strand every
 * recorded split. Repeats are rejected with @ref novaInvalidValue.
 *
 * @param[in] threads  Total threads to use. Must be at least
 *                     @ref MIN_THREADS_PER_GROUP.
 *
 * @return @ref novaStatus_t with @c err set to @ref novaSuccess on
 *         success, @ref novaInvalidNumThreads when @p threads is
 *         less than @ref MIN_THREADS_PER_GROUP, or
 *         @ref novaInvalidValue when the budget was already set.
 *
 * @pre  @p threads must be at least @ref MIN_THREADS_PER_GROUP.
 *
 * @note Thread-safe. The single-shot latch is claimed with an atomic
 *       exchange, so concurrent repeats are rejected deterministically.
 */
novaStatus_t set_num_logical_threads(uint32 threads) {
  if (threads < MIN_THREADS_PER_GROUP) {
    return (novaStatus_t){
        .err = novaInvalidNumThreads,
        .message = nova_get_error_msg(novaInvalidNumThreads, nullptr),
    };
  }
  if (atomic_exchange_explicit(&atomic_global_budget_set_once, true,
                               memory_order_acq_rel)) {
    return (novaStatus_t){
        .err = novaInvalidValue,
        .message =
            "Global thread budget is already configured; "
            "set_num_logical_threads() succeeds only once per execution.",
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

/**
 * @brief Provenance flags mirroring the per-group counters.
 */
static _Atomic GroupOrigin group_origin[NUM_PARALLEL_GROUPS] = {0};

/**
 * @brief Display label for a group origin flag.
 */
static const char *origin_label(GroupOrigin origin) {
  switch (origin) {
  case ThreadsOriginAuto:
    return "auto";
  case ThreadsOriginManual:
    return "manual";
  default:
    return "unset";
  }
}

/**
 * @brief Read the distribution mode from explicit origins.
 *
 * @details
 * Pure derivation behind @ref current_distribution_kind(): manual
 * when any group is manual, automatic otherwise when any group is
 * automatic, unset when nothing was ever assigned. Unset groups are
 * neutral in every combination.
 */
DistributionKind distribution_kind_of(GroupOrigin compute_origin,
                                      GroupOrigin autograd_origin,
                                      GroupOrigin dtloader_origin) {
  const bool any_manual = ((compute_origin == ThreadsOriginManual) ||
                           (autograd_origin == ThreadsOriginManual) ||
                           (dtloader_origin == ThreadsOriginManual)) != 0;
  if (any_manual) {
    return DistributionManual;
  }
  const bool any_auto = ((compute_origin == ThreadsOriginAuto) ||
                         (autograd_origin == ThreadsOriginAuto) ||
                         (dtloader_origin == ThreadsOriginAuto)) != 0;
  if (any_auto) {
    return DistributionAuto;
  }
  return DistributionUnset;
}

/**
 * @brief Read the live distribution mode of the group counters.
 */
DistributionKind current_distribution_kind() {
  return distribution_kind_of(
      atomic_load_explicit(&group_origin[ParallelComputeGroup],
                           memory_order_relaxed),
      atomic_load_explicit(&group_origin[ParallelAutogradGroup],
                           memory_order_relaxed),
      atomic_load_explicit(&group_origin[ParallelDTLoaderGroup],
                           memory_order_relaxed));
}

/**
 * @brief Print the oversubscription warning block.
 *
 * @details
 * Yellow, single block, fully computed numbers. Advisory only:
 * the assignment stands.
 */
static void print_oversubscription_warning(uint32 compute, uint32 autograd,
                                           uint32 dtloader, uint32 total,
                                           uint64_t counted) {
  printf("-- " NCORE_LOG_YELLOW
         "[Threads] WARNING: manual assignment oversubscribes the machine: "
         "counted %" PRIu64 " (compute %" PRIu32 " + autograd %" PRIu32
         " + dtloader %" PRIu32
         "; groups at 1 excluded) > machine total %" PRIu32
         ". Expect context-switching losses.\n" NCORE_LOG_RESET,
         counted, compute, autograd, dtloader, total);
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
 * @brief Assign a count to a group, recording its provenance.
 *
 * @details
 * Shared backend for the public setter (manual) and the
 * stratification distribute (auto). Range-checks the group;
 * per-group minimums are enforced by the dispatched setter.
 */
static novaStatus_t assign_group_threads(ParallelGroups group, uint32 threads,
                                         GroupOrigin origin) {
  if (group >= NUM_PARALLEL_GROUPS) {
    return (novaStatus_t){
        .err = novaInvalidParallelGroup,
        .message = nova_get_error_msg(novaInvalidParallelGroup, nullptr),
    };
  }
  novaStatus_t st = set_threads_funcs[group](threads);
  if (st.err == novaSuccess) {
    atomic_store_explicit(&group_origin[group], origin, memory_order_relaxed);
  }
  return st;
}

/**
 * @brief Return the number of threads assigned to a group.
 *
 * @param[in]  group   Group to query.
 * @param[out] status  Receives the result.
 *
 * @return Number of threads for @p group, or @c 0 on failure.
 *
 * @pre  @p status must not be @c nullptr.
 *
 * @note A null @p status is tolerated defensively and reported as
 *       @c 0, which is unambiguous since live counters never hold
 *       zero. Prefer passing a valid status.
 */
uint32 get_num_threads_from(ParallelGroups group, novaStatus_t *status) {
  if (status == nullptr) {
    return 0;
  }
  if (group >= NUM_PARALLEL_GROUPS) {
    *status = (novaStatus_t){
        .err = novaInvalidParallelGroup,
        .message = nova_get_error_msg(novaInvalidParallelGroup, nullptr),
    };
    return 0;
  }
  const uint32 count = get_threads_funcs[group]();
  *status = (novaStatus_t){
      .err = novaSuccess,
      .message = nova_get_error_msg(novaSuccess, nullptr),
  };
  return count;
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
  if (current_distribution_kind() == DistributionAuto) {
    return (novaStatus_t){
        .err = novaInvalidValue,
        .message = "Thread groups already stratified automatically; manual "
                   "assignment is rejected to keep a single distribution mode.",
    };
  }
  return assign_group_threads(group, threads, ThreadsOriginManual);
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
 *         A null budget is invalid and returns @c false.
 *
 * @pre  @p strat must not be @c nullptr.
 *
 * @see distribute_stratified_threads()  Applies a validated budget.
 * @see StratifiedThreads                Budget under validation.
 */
bool is_valid_stratification_result(const StratifiedThreads *strat) {
  if (strat == nullptr) {
    return false;
  }
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
 * @note A null @p status is tolerated defensively and reported as
 *       @c {0,0,0}, the documented failure sentinel. Prefer passing
 *       a valid status.
 *
 * @see stratify_threads()  Partitioning and recording.
 * @see is_stratification_complete()  Cache state query.
 * @see get_last_stratification_result()  Cached result source.
 */
StratifiedThreads get_stratified_threads(novaStatus_t *status) {

  if (status == nullptr) {
    return (StratifiedThreads){0, 0, 0};
  }

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

  if (current_distribution_kind() == DistributionManual) {
    return (novaStatus_t){
        .err = novaInvalidValue,
        .message = "Thread groups already assigned manually; "
                   "distribute_stratified_threads() is rejected to keep a "
                   "single distribution mode.",
    };
  }
  atomic_store_explicit(&atomic_global_thread_counter_initialized, true,
                        memory_order_release);
  auto st = assign_group_threads(ParallelComputeGroup, strat->compute,
                                 ThreadsOriginAuto);
  if (st.err != novaSuccess) {
    return st;
  }
  st = assign_group_threads(ParallelAutogradGroup, strat->autograd,
                            ThreadsOriginAuto);
  if (st.err != novaSuccess) {
    return st;
  }
  return assign_group_threads(ParallelDTLoaderGroup, strat->dtloader,
                              ThreadsOriginAuto);
}

/**
 * @brief Query whether the thread configuration is fully initialized.
 *
 * @details
 * Combines both halves of the setup: an explicitly set global budget
 * (@ref is_global_thread_count_initialized()) and at least one
 * successful stratification (@ref is_stratification_complete()).
 * @ref print_thread_config() branches on this: @c false means only a
 * partial view can be shown.
 *
 * @return @c true when the budget is set and a stratification result
 *         is recorded, @c false otherwise.
 *
 * @note Thread-safe. Both flags are read with acquire ordering.
 *
 * @see is_global_thread_count_initialized()  Budget half of the guard.
 * @see is_stratification_complete()  Stratification half of the guard.
 * @see print_thread_config()  Consumer of this guard.
 */
bool is_thread_config_initialized() {
  if (current_distribution_kind() == DistributionManual) {
    return true;
  }
  return (is_global_thread_count_initialized() &&
          is_stratification_complete()) != 0;
}

/**
 * @brief Report whether a tensor holds enough work to parallelize.
 *
 * @details
 * Replaces fixed thresholds like @c size > 100000 with a dynamic
 * check scaled by the actual thread count. The @p kind bitmask
 * selects which criteria must hold; every selected bit must pass.
 * Passing @c 0 is treated as @c ParallelizableByElements. Grains are
 * intentionally modest (a few thousand elements or a few tens of
 * kilobytes per thread) so the decision tracks the tensor, not a
 * magic number.
 *
 * @param[in] ten      Tensor to inspect. May be @c nullptr.
 * @param[in] threads  Thread count the parallel region would use.
 *                     Values @c 0 or @c 1 always yield @c false.
 * @param[in] kind     Bitmask of @ref ParallelizableBy criteria.
 *
 * @return @c true when the tensor satisfies every selected criterion
 *         for @p threads, @c false otherwise.
 *
 * @note Thread-safe. Reads only @p ten fields and @p threads; no
 *       shared state is touched.
 *
 * @see ParallelizableBy
 */
bool is_parallelizable(const struct Tensor *ten, uint32 threads,
                       ParallelizableBy kind) {
  if (ten == nullptr) {
    return false;
  }
  if (threads <= 1) {
    return false;
  }

  constexpr size_t min_elements_per_thread = 4096;
  constexpr size_t min_bytes_per_thread = 16 * 1024;

  if ((kind & ParallelizableByElements) != 0) {
    // logical_size counts unpacked elements (important for packed
    // dtypes like fp4 where size counts storage units).
    if (ten->logical_size < (size_t)threads * min_elements_per_thread) {
      return false;
    }
  }
  if ((kind & ParallelizableByBytes) != 0) {
    const size_t total_bytes = ten->storage->size_bytes;
    if (total_bytes < (size_t)threads * min_bytes_per_thread) {
      return false;
    }
  }
  if ((kind & ParallelizableByTensor) != 0) {
    if (ten->size == 0 || ten->logical_size == 0) {
      return false;
    }
    if (ten->ndims == 0) {
      return false;
    }
    if (!is_allocated(ten)) {
      return false;
    }
    if (ten->device != DEVICE_CPU) {
      return false;
    }
    if (ten->dtype >= NUM_DTYPES) {
      return false;
    }
    // Packed and quantized types pay unpack / dequant overhead, so
    // they need a larger grain before the OpenMP fork pays off.
    // is_quantizable_dtype covers fp4 and the q* families; Float4
    // is the only packed type with factor 2.
    if (is_quantizable_dtype(ten->dtype) ||
        dtype_packing_factor(ten->dtype) > 1) {
      constexpr size_t min_packed_elements_per_thread = 8192;
      if (ten->logical_size <
          (size_t)threads * min_packed_elements_per_thread) {
        return false;
      }
    }
    // Tiny items (fp8/int8/qint8, 1 byte) are especially sensitive to
    // false sharing and to per-element dispatch cost: require a byte
    // grain even when the caller only asked for ByTensor.
    if (ten->item_size == 1) {
      constexpr size_t min_tiny_bytes_per_thread = 8192;
      const size_t total_bytes = ten->storage->size_bytes;
      if (total_bytes < (size_t)threads * min_tiny_bytes_per_thread) {
        return false;
      }
    }
  }
  return true;
}

/**
 * @brief Print the current thread budget and its stratification.
 *
 * @details
 * Writes a human-readable summary to stdout with the @c NCORE_LOG_*
 * palette: green prefix, bold headings, cyan values, dim labels,
 * yellow for uninitialized or unavailable entries. Concise mode fits
 * the whole state on one line; verbose mode prints one aligned row
 * per entry, mirroring @ref printCudaDeviceInfo().
 *
 * Every row reads live data, so a @c false guard still shows the
 * hardware thread count and the per-group counters; only the
 * configured budget is marked @c NOT INITIALIZED. A failed hardware
 * query degrades its row to @c unavailable instead of hiding the rest.
 *
 * @param[in] verbose  If @c false, print the one-line summary. If
 *                     @c true, print the full block.
 *
 * @return @ref novaStatus_t with @c err set to @ref novaSuccess when
 *         the complete configuration was printed, or to
 *         @ref novaThreadNotInitialized when only a partial view was
 *         available.
 *
 * @note Thread-safe. Reads atomic counters and the recorded
 *       stratification with acquire ordering; performs no writes.
 *
 * @see is_thread_config_initialized()  Completeness guard.
 * @see get_last_stratification_result()  Recorded split shown.
 * @see print_device_info()  Device-side counterpart.
 */
novaStatus_t print_thread_config(bool verbose) {
  novaStatus_t query = {};
  const uint32 logical = get_num_logical_threads(&query);
  const bool logical_ok = (query.err == novaSuccess);
  const bool initialized = is_thread_config_initialized();
  const bool manual_mode = (current_distribution_kind() == DistributionManual);

  // Live per-group counters always hold a value, even before any
  // configuration, so they are shown in both modes unconditionally.
  const uint32 live_compute = get_compute_threads();
  const uint32 live_autograd = get_autograd_threads();
  const uint32 live_dtloader = get_dtloader_threads();
  const char *origin_compute = origin_label(atomic_load_explicit(
      &group_origin[ParallelComputeGroup], memory_order_relaxed));
  const char *origin_autograd = origin_label(atomic_load_explicit(
      &group_origin[ParallelAutogradGroup], memory_order_relaxed));
  const char *origin_dtloader = origin_label(atomic_load_explicit(
      &group_origin[ParallelDTLoaderGroup], memory_order_relaxed));

  if (!verbose) {
    if (manual_mode) {
      printf(NCORE_LOG_PREFIX
             " [Threads] manual (compute " NCORE_LOG_VALUE
             "%" PRIu32 NCORE_LOG_RESET " autograd " NCORE_LOG_VALUE
             "%" PRIu32 NCORE_LOG_RESET " dtloader " NCORE_LOG_VALUE
             "%" PRIu32 NCORE_LOG_RESET ")\n",
             live_compute, live_autograd, live_dtloader);
    } else if (initialized) {
      const uint32 total = get_configured_num_logical_threads(nullptr);
      const StratifiedThreads strat = get_last_stratification_result();
      printf(NCORE_LOG_PREFIX
             " [Threads] total " NCORE_LOG_VALUE "%" PRIu32 NCORE_LOG_RESET
             " (compute " NCORE_LOG_VALUE "%" PRIu32 NCORE_LOG_RESET
             " autograd " NCORE_LOG_VALUE "%" PRIu32 NCORE_LOG_RESET
             " dtloader " NCORE_LOG_VALUE "%" PRIu32 NCORE_LOG_RESET ")\n",
             total, strat.compute, strat.autograd, strat.dtloader);
    } else if (logical_ok) {
      printf(NCORE_LOG_PREFIX " [Threads] logical " NCORE_LOG_VALUE
                              "%" PRIu32 NCORE_LOG_RESET " " NCORE_LOG_YELLOW
                              "NOT INITIALIZED" NCORE_LOG_RESET "\n",
             logical);
    } else {
      printf(NCORE_LOG_PREFIX " [Threads] " NCORE_LOG_RED
                              "thread count unavailable" NCORE_LOG_RESET "\n");
    }
  } else {
    printf(NCORE_LOG_PREFIX NCORE_LOG_BOLD
           " === Thread Config ===\n" NCORE_LOG_RESET);
    if (logical_ok) {
      printf(NCORE_LOG_PREFIX
             "   " NCORE_LOG_DIM
             "Logical threads:   " NCORE_LOG_RESET NCORE_LOG_VALUE "%" PRIu32
             "\n" NCORE_LOG_RESET,
             logical);
    } else {
      printf(NCORE_LOG_PREFIX
             "   " NCORE_LOG_DIM
             "Logical threads:   " NCORE_LOG_RESET NCORE_LOG_YELLOW
             "unavailable\n" NCORE_LOG_RESET);
    }
    if (is_global_thread_count_initialized()) {
      const uint32 total = get_configured_num_logical_threads(nullptr);
      printf(NCORE_LOG_PREFIX
             "   " NCORE_LOG_DIM
             "Configured budget: " NCORE_LOG_RESET NCORE_LOG_VALUE "%" PRIu32
             "\n" NCORE_LOG_RESET,
             total);
      printf(NCORE_LOG_PREFIX
             "   " NCORE_LOG_DIM
             "Initialized:       " NCORE_LOG_RESET NCORE_LOG_VALUE
             "yes\n" NCORE_LOG_RESET);
    } else if (manual_mode) {
      printf(NCORE_LOG_PREFIX
             "   " NCORE_LOG_DIM
             "Configured budget: " NCORE_LOG_RESET NCORE_LOG_DIM
             "manual\n" NCORE_LOG_RESET);
      printf(NCORE_LOG_PREFIX
             "   " NCORE_LOG_DIM
             "Initialized:       " NCORE_LOG_RESET NCORE_LOG_VALUE
             "yes\n" NCORE_LOG_RESET);
    } else {
      printf(NCORE_LOG_PREFIX
             "   " NCORE_LOG_DIM
             "Configured budget: " NCORE_LOG_RESET NCORE_LOG_YELLOW
             "NOT INITIALIZED\n" NCORE_LOG_RESET);
      printf(NCORE_LOG_PREFIX
             "   " NCORE_LOG_DIM
             "Initialized:       " NCORE_LOG_RESET NCORE_LOG_YELLOW
             "no\n" NCORE_LOG_RESET);
    }
    printf(NCORE_LOG_PREFIX
           "   " NCORE_LOG_DIM
           "Compute:           " NCORE_LOG_RESET NCORE_LOG_VALUE
           "%" PRIu32 NCORE_LOG_RESET NCORE_LOG_DIM " (%s)\n" NCORE_LOG_RESET,
           live_compute, origin_compute);
    printf(NCORE_LOG_PREFIX
           "   " NCORE_LOG_DIM
           "Autograd:          " NCORE_LOG_RESET NCORE_LOG_VALUE
           "%" PRIu32 NCORE_LOG_RESET NCORE_LOG_DIM " (%s)\n" NCORE_LOG_RESET,
           live_autograd, origin_autograd);
    printf(NCORE_LOG_PREFIX
           "   " NCORE_LOG_DIM
           "DTLoader:          " NCORE_LOG_RESET NCORE_LOG_VALUE
           "%" PRIu32 NCORE_LOG_RESET NCORE_LOG_DIM " (%s)\n" NCORE_LOG_RESET,
           live_dtloader, origin_dtloader);
    if (logical_ok && manual_counts_oversubscribed(live_compute, live_autograd,
                                                   live_dtloader, logical)) {
      print_oversubscription_warning(
          live_compute, live_autograd, live_dtloader, logical,
          counted_thread_sum(live_compute, live_autograd, live_dtloader));
    }
  }

  if (initialized) {
    return (novaStatus_t){
        .err = novaSuccess,
        .message = nova_get_error_msg(novaSuccess, nullptr),
    };
  }
  return (novaStatus_t){
      .err = novaThreadNotInitialized,
      .message = nova_get_error_msg(novaThreadNotInitialized, nullptr),
  };
}
