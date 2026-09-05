/**
 * @file threads.h
 * @brief Public C API for thread-count management and thread
 *        stratification across NovaNN subsystems.
 *
 * @details
 * This header exposes the threading configuration API used to size
 * the worker pools of the CPU compute libraries and the internal
 * NovaNN subsystems. It provides:
 *
 * @li Logical thread queries — @ref get_num_logical_threads()
 *   reports how many logical hardware threads are available to the
 *   process (respecting CPU affinity / cgroup restrictions).
 * @li Global thread setting — @ref set_num_physical_threads()
 *   configures the total number of threads used by the library.
 * @li Managed groups — @ref set_num_threads_to() and
 *   @ref get_num_threads_from() assign and query thread counts for
 *   the groups enumerated in @ref ParallelGroups.
 * @li Stratification — @ref stratify_threads() partitions a total
 *   thread count across all managed groups into a @ref
 *   StratifiedThreads structure. @ref get_stratified_threads()
 *   resolves the total and caches the result, since the global budget
 *   is immutable once set.
 * @li Distribution — @ref distribute_stratified_threads() applies a
 *   @ref StratifiedThreads budget to the per-group counters.
 * @li Validation — @ref is_valid_stratification_result() and
 *   @ref is_valid_latest_stratification_result() report whether a
 *   budget holds no empty group.
 * @li Inspection — @ref print_thread_config() prints the current
 *   thread budget and its stratification to stdout, in concise or
 *   verbose form.
 * @li Parallelism — @ref is_parallelizable() decides whether a tensor
 *   holds enough work to amortize a parallel region for a given
 *   thread count and @ref ParallelizableBy criterion.
 *
 * @see concurrency.h  Low-level logical-thread query implementation.
 * @see status.h       novaStatus_t error reporting.
 */

#pragma once

#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
  * @enum ParallelGroups
  * @brief Identifies the thread pools managed by the NovaNN runtime.
  *
  * @details
  * Each value names a subsystem that consumes a dedicated set of
  * worker threads.  The groups are managed collectively through the
  * threading API: a total budget can be stratified across them via
  * @ref stratify_threads(), and individual groups can be configured
  * via @ref set_num_threads_to().
  */
typedef enum {
  ParallelComputeGroup, ///< Compute pool for high-performance CPU libraries (oneDNN, MKL, OpenBLAS, etc.).
  ParallelAutogradGroup, ///< Autograd engine worker pool.
  ParallelDTLoaderGroup, ///< Data-loading worker pool.
} ParallelGroups;

/**
 * @enum ParallelizableBy
 * @brief Criterion used by @ref is_parallelizable() to judge work size.
 *
 * @details
 * Each value is a bit flag so they combine with @c | . A tensor is
 * considered parallelizable only when every selected criterion is met:
 * @li @c ParallelizableByElements — enough logical elements per thread.
 * @li @c ParallelizableByBytes — enough storage bytes per thread.
 * @li @c ParallelizableByTensor — structural and type-driven sanity:
 *   not scalar or empty, allocated on @c DEVICE_CPU, valid @ref DType_,
 *   and with dtype-aware grain (packed and quantized types need more
 *   elements to amortize unpacking, tiny @c item_size needs more
 *   bytes to avoid false sharing).
 *
 * Typical use is a single flag or @c ByElements | @c ByTensor for
 * element-wise kernels and @c ByBytes | @c ByTensor for memory-bound
 * copies. Passing @c 0 is treated as @c ByElements.
 */
typedef enum ParallelizableBy : uint8_t {
  ParallelizableByElements = 1u << 0, ///< Grain on @c logical_size.
  ParallelizableByBytes = 1u << 1,    ///< Grain on @c storage->size_bytes.
  ParallelizableByTensor = 1u << 2,   ///< Structural checks on the tensor.
} ParallelizableBy;

/**
 * @struct StratifiedThreads
 * @brief Thread budget partitioned across the managed groups.
 *
 * @details
 * Produced by @ref stratify_threads() to describe how a total thread
 * count is distributed among the NovaNN worker pools.  The sum of
 * the three members equals the total number of threads that was
 * stratified, except for a total of 2, which yields @c {1,1,1} so
 * that no pool is left without threads.
 */
typedef struct {
  uint32 compute;  ///< Threads assigned to the compute group.
  uint32 autograd; ///< Threads assigned to the autograd group.
  uint32 dtloader; ///< Threads assigned to the data-loading group.
} StratifiedThreads;

/**
 * @def MIN_THREADS_PER_GROUP
 * @brief Minimum number of threads assigned to each parallel group.
 *
 * @details
 * Every parallel group (@ref ParallelGroups) is guaranteed at least
 * one thread. This value is the lower bound enforced when validating
 * per-group assignments via @ref set_num_threads_to() and
 * @ref set_compute_threads(), @ref set_autograd_threads(),
 * @ref set_dtloader_threads(), and when stratifying a total budget
 * via @ref stratify_threads(). Each group's atomic counter
 * determines the number of threads its pool is allowed to use.
 *
 * @see NUM_PARALLEL_GROUPS
 * @see ParallelGroups
 * @see StratifiedThreads
 */
#define MIN_THREADS_PER_GROUP 1

/**
 * @brief Set the global number of logical threads used by the library.
 *
 * @details
 * Configures the total logical thread budget that NovaNN and the
 * underlying high-performance libraries use for parallel execution.
 * The value is stored in @ref atomic_global_thread_counter and
 * determines the total number of threads the library is allowed to
  * use. The managed groups may then be configured individually via
  * @ref set_num_threads_to(). The value must be at least
  * @ref MIN_THREADS_PER_GROUP (1).
  *
  * The budget may only be set once per execution: a repeat call is
  * rejected with @ref novaInvalidValue, since silently swapping the
  * total would strand the recorded stratification.
  *
  * @param[in] threads  Total number of logical threads to use. Must be
  *                     at least @ref MIN_THREADS_PER_GROUP (1).
  *
  * @return @ref novaStatus_t with @c err set to @ref novaSuccess on
  *         success, @ref novaInvalidNumThreads when @p threads is
  *         less than @ref MIN_THREADS_PER_GROUP, or
  *         @ref novaInvalidValue when the budget was already set.
 *
 * @pre  @p threads must be at least @ref MIN_THREADS_PER_GROUP.
 *
 * @see set_num_logical_threads2group()  Configures individual groups.
 * @see stratify_threads()       Partitions a total across groups.
 * @see MIN_THREADS_PER_GROUP    Minimum per-group threads.
 */
novaStatus_t set_num_logical_threads(uint32 threads);

/**
  * @brief Return the number of logical threads available to the
  *        process.
  *
  * @details
  * Queries the hardware thread count through the platform-specific
  * implementation declared in @c concurrency.h.  On Linux the count is
  * derived from the CPU affinity mask (@c sched_getaffinity); on
  * Windows it is derived from the process affinity mask
  * (@c GetProcessAffinityMask).  The result reflects logical CPUs
  * (SMT/Hyperthreading siblings counted separately) and respects any
  * affinity restriction already in effect for the process (taskset,
  * cgroups, containers, etc.).
  *
  * @param[out] status  Receives the operation result.  Set to
  *                     @ref novaSuccess on success, or to
  *                     @ref novaInvalidNumThreads when the count could
  *                     not be determined.
  *
  * @return Number of logical threads available to the process, or
  *         @c 0 when the count could not be determined.
  *
  * @pre  @p status must not be @c nullptr.
  *
  * @see set_num_physical_threads()  Sets the global thread budget.
  * @see concurrency.h               Platform-specific implementation.
  */
uint32 get_num_logical_threads(novaStatus_t *status);

/**
 * @brief Assign a thread count to a managed group.
 *
 * @details
 * Configures the number of worker threads used by the given
 * @ref ParallelGroups group. The assigned value is stored in the
 * group's atomic counter and determines the number of threads that
 * pool is allowed to use. The value must be at least
 * @ref MIN_THREADS_PER_GROUP (1).
 *
 * @param[in] group    The managed group to configure.
 * @param[in] threads  Number of threads to assign to @p group. Must be
 *                     at least @ref MIN_THREADS_PER_GROUP (1).
 *
 * @return @ref novaStatus_t with @c err set to @ref novaSuccess on
 *         success, @ref novaInvalidParallelGroup when @p group is
 *         invalid, or @ref novaInvalidNumThreads when @p threads is
 *         less than @ref MIN_THREADS_PER_GROUP.
 *
 * @see get_num_threads_from()  Queries the assigned count.
 * @see ParallelGroups          Enum identifying managed groups.
 * @see MIN_THREADS_PER_GROUP   Minimum per-group threads.
 */
novaStatus_t set_num_threads_to(ParallelGroups group, uint32 threads);

/**
  * @brief Return the number of threads assigned to a managed group.
  *
  * @param[in]  group    The managed group to query.
  * @param[out] status  Receives the operation result.  Set to
  *                     @ref novaSuccess on success, or to an error
  *                     status describing the failure.
  *
  * @return Number of threads currently assigned to @p group, or
  *         @c 0 on failure.
  *
  * @pre  @p status must not be @c nullptr.
  *
  * @see set_num_threads_to()  Assigns the count.
  * @see ParallelGroups           Enum identifying managed groups.
  */
uint32 get_num_threads_from(ParallelGroups group, novaStatus_t *status);

/**
 * @brief Return the globally configured thread budget.
 *
 * @param[out] status  Receives @ref novaSuccess on success. May be
 *                     @c nullptr.
 *
 * @return Current global thread budget. Default is
 *         @ref MIN_THREADS_PER_GROUP before any successful call to
 *         @ref set_num_logical_threads().
 *
 * @see set_num_logical_threads()
 */
uint32 get_configured_num_logical_threads(novaStatus_t *status);

/**
 * @brief Query whether the global thread budget was set.
 *
 * @return @c true if configured, @c false otherwise.
 */
bool is_global_thread_count_initialized();

/**
 * @brief Build a stratified thread budget for all managed groups.
 *
 * @details
 * Resolves the total thread budget and partitions it via
 * @ref stratify_threads(). If the global budget was never configured
 * (@ref is_global_thread_count_initialized() returns @c false), the
 * total is obtained from @ref get_num_logical_threads() (hardware
 * affinity). Otherwise the total comes from
 * @ref get_configured_num_logical_threads() (user configuration).
 *
 * The first successful call records its result: later calls return the
 * cached budget via @ref get_last_stratification_result() without
 * recomputing, since the global budget is immutable once set
 * (@ref is_stratification_complete() reports whether the cached
 * result already exists).
 *
 * @param[out] status  Receives the operation result. Set to
 *                     @ref novaSuccess on success, or to the error
 *                     produced by the underlying query
 *                     (@ref novaInvalidNumThreads,
 *                     @ref novaOsPlatformNotSupported, etc.). Must
 *                     not be @c nullptr.
 *
 * @return A @ref StratifiedThreads structure whose members sum to the
 *         total budget (except for a total of 2, which overcommits to
 *         @c {1,1,1}), or @c {0,0,0} when the total could not be
 *         determined.
 *
 * @pre  @p status must not be @c nullptr.
 *
 * @see stratify_threads()  Partitions the total.
 * @see is_global_thread_count_initialized()  Selects the total source.
 * @see is_stratification_complete()  Reports whether a cached result exists.
 * @see get_last_stratification_result()  Returns the cached result.
 */
StratifiedThreads get_stratified_threads(novaStatus_t *status);

/**
  * @brief Report whether a stratified budget holds no empty group.
  *
  * @details
  * Checks that every member of @p strat (@c compute, @c autograd,
  * @c dtloader) is nonzero. A zero member marks either the failure
  * sentinel @c {0,0,0} or a budget that would starve a pool, so both
  * are rejected.
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
bool is_valid_stratification_result(const StratifiedThreads *strat);

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
  * @see is_valid_stratification_result()  Validation applied.
  * @see get_last_stratification_result()  Recorded budget source.
  */
bool is_valid_latest_stratification_result();

/**
  * @brief Apply a stratified budget to the per-group counters.
  *
  * @details
  * Assigns @p strat members to their groups in @ref ParallelGroups
  * order (compute, autograd, dtloader) through @ref set_num_threads_to().
  * The budget is validated with @ref is_valid_stratification_result()
  * before touching any counter; assignment stops at the first failing
  * group and reports its error.
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
novaStatus_t distribute_stratified_threads(const StratifiedThreads *strat);

/**
  * @brief Query whether the thread configuration is fully initialized.
  *
  * @details
  * Reports whether both halves of the setup are done: the global
  * thread budget was explicitly set (see
  * @ref is_global_thread_count_initialized()) and at least one
  * stratification succeeded (see @ref is_stratification_complete()).
  * @ref print_thread_config() uses this as its guard: a @c false
  * result means only a partial view can be shown.
  *
  * @return @c true when the budget is set and a stratification result
  *         is recorded, @c false otherwise.
  *
  * @see is_global_thread_count_initialized()  Budget half of the guard.
  * @see is_stratification_complete()  Stratification half of the guard.
  * @see print_thread_config()  Consumer of this guard.
  */
bool is_thread_config_initialized();

/**
  * @brief Print the current thread budget and its stratification.
  *
  * @details
  * Writes a human-readable summary to stdout using the @c NCORE_LOG_*
  * palette from @c macros.h (green prefix, bold headings, cyan values,
  * dim labels, yellow for uninitialized entries). With @p verbose set
  * to @c false a single summary line is printed; with @c true a full
  * block follows: logical thread count, configured budget, whether the
  * configuration is initialized, and the live per-group counters.
  *
 * The print never fails for lack of configuration: when
 * @ref is_thread_config_initialized() returns @c false, whatever is
 * known (hardware thread count, live counters) is shown with the
  * missing entries marked @c NOT INITIALIZED, and the status reports
  * @ref novaThreadNotInitialized. A hardware count query failure is
  * shown as @c unavailable without hiding the remaining rows.
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
  * @see get_stratified_threads()        Budget behind the summary.
  * @see get_last_stratification_result()  Recorded split shown.
  */
novaStatus_t print_thread_config(bool verbose);

/**
 * @brief Report whether a tensor holds enough work to parallelize.
 *
 * @details
 * Replaces fixed thresholds like @c size > 100000 with a dynamic
 * check that accounts for the actual thread count and the kind of
 * work. The @p kind bitmask selects which criteria must hold; every
 * selected bit must pass for the result to be @c true:
 * @li @c ParallelizableByElements — @c ten->logical_size is at least
 *   @c threads * grain elements.
 * @li @c ParallelizableByBytes — @c ten->size * @c ten->item_size is
 *   at least @c threads * grain bytes.
 * @li @c ParallelizableByTensor — structural and type-driven sanity:
 *   allocated on @c DEVICE_CPU with a valid @ref DType_, not scalar
 *   or empty, and with dtype-aware grain (packed and quantized types
 *   need more elements, 1-byte items need more bytes).
 * Combine flags with @c | ; passing @c 0 selects
 * @c ParallelizableByElements. A single thread or a null tensor is
 * never considered parallelizable.
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
                       ParallelizableBy kind);

#ifdef __cplusplus
}
#endif
