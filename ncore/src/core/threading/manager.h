/**
 * @file manager.h
 * @brief Internal declaration of the thread-budget stratification helper.
 *
 * @details
 * Declares @ref stratify_threads(), which partitions a total thread
 * count across the three managed groups. The public wrappers in
 * @ref threads.h (@ref get_stratified_threads() and per-group
 * setters) call this helper after resolving the total budget.
 *
 * Every successful stratification is recorded for
 * @ref get_last_stratification_result(), so @ref get_stratified_threads()
 * serves later calls from the cache while the global budget stays
 * immutable (@ref is_stratification_complete() reports whether the
 * recorded result already exists).
 *
 * This header also exposes the stratification tuning constants
 * (@ref DYNASTRAT_EQUAL_TH, @ref DYNASTRAT_LIGHT_TH,
 * @ref DYNASTRAT_STRICT_TH, @ref DYNASTRAT_FULL_TH and the
 * @ref DYNASTRAT_W_COMPUTE, @ref DYNASTRAT_W_AUTOGRAD,
 * @ref DYNASTRAT_W_DTLOADER target weights). The splitting policy
 * itself is documented in @c DYNASTRAT.md next to this header.
 *
 * @see manager.c     Implementation of the stratification policy.
 * @see DYNASTRAT.md  Design note describing the policy rationale.
 * @see threads.h     Public API that consumes this helper.
 * @see threads.c     Dispatch tables and global counter that feed
 *                    the total into @ref stratify_threads().
 */

#pragma once

#include <stdint.h>

#include <ncore/core/dtype.h>
#include <ncore/threading/threads.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum GroupOrigin
 * @brief Provenance of a per-group thread count.
 *
 * @details
 * Tracks how each group counter got its value: untouched since
 * process start, assigned by @ref distribute_stratified_threads(),
 * or assigned by a direct @ref set_num_threads_to() call.
 */
typedef enum GroupOrigin : uint8_t {
  ThreadsOriginUnset = 0,  ///< Never assigned; still the default.
  ThreadsOriginAuto = 1,   ///< Assigned by stratification distribute.
  ThreadsOriginManual = 2, ///< Assigned by a direct setter call.
} GroupOrigin;

/**
 * @enum DistributionKind
 * @brief Distribution mode of the three group counters.
 *
 * @details
 * Exactly one workflow owns the counters per process: automatic,
 * manual, or none yet. The mutating calls reject anything else, so
 * no mixed state exists.
 */
typedef enum DistributionKind : uint8_t {
  // typedef enum novaError_t : uint8_t {
  DistributionUnset = 0,  ///< No group was ever assigned.
  DistributionAuto = 1,   ///< Assigned groups came from distribute.
  DistributionManual = 2, ///< Assigned groups were set directly.
} DistributionKind;

/**
 * @def DYNASTRAT_EQUAL_TH
 * @brief Total thread count up to which the budget is split evenly.
 *
 * @details
 * Totals at or below this value use fair weights (@c 1/3 per pool),
 * so small budgets stay balanced instead of opening gaps that would
 * starve a pool. Must be less than @ref DYNASTRAT_FULL_TH.
 *
 * @see DYNASTRAT_FULL_TH   Boundary of full stratification.
 */
#define DYNASTRAT_EQUAL_TH 6

/**
 * @def DYNASTRAT_LIGHT_TH
 * @brief Total thread count from which a light ordering touch applies.
 *
 * @details
 * Totals in [@ref DYNASTRAT_LIGHT_TH, @ref DYNASTRAT_STRICT_TH) get a
 * single ordering pass that opens the minimum gap without enforcing
 * the strict three-level order, which does not fit yet. Below this
 * value the Hamilton result is kept as is.
 *
 * @see DYNASTRAT_STRICT_TH   Boundary of strict ordering.
 */
#define DYNASTRAT_LIGHT_TH 7

/**
 * @def DYNASTRAT_STRICT_TH
 * @brief Total thread count from which strict ordering is enforced.
 *
 * @details
 * Totals at or above this value satisfy @c compute > @c dtloader >
 * @c autograd strictly. Must be greater than @ref DYNASTRAT_LIGHT_TH.
 *
 * @see DYNASTRAT_LIGHT_TH   Boundary of the light touch regime.
 */
#define DYNASTRAT_STRICT_TH 12

/**
 * @def DYNASTRAT_FULL_TH
 * @brief Total thread count from which the target weights apply fully.
 *
 * @details
 * Totals at or above this value split by the target weights declared
 * in this header. Totals between @ref DYNASTRAT_EQUAL_TH and this
 * value interpolate linearly between fair weights and target
 * weights, keeping the transition continuous. Must be greater than
 * @ref DYNASTRAT_EQUAL_TH.
 *
 * @see DYNASTRAT_EQUAL_TH   Boundary of the fair regime.
 */
#define DYNASTRAT_FULL_TH 16

/**
 * @def DYNASTRAT_W_COMPUTE
 * @brief Target weight of the compute pool under full stratification.
 *
 * @details
 * Largest of the three weights: the compute pool parallelises math
 * and absorbs most of the growth. The three target weights must sum
 * to @c 1.0 and satisfy @c W_COMPUTE > @c W_DTLOADER > @c W_AUTOGRAD.
 *
 * @see DYNASTRAT_W_DTLOADER    Intermediate target weight.
 * @see DYNASTRAT_W_AUTOGRAD    Smallest target weight.
 */
#define DYNASTRAT_W_COMPUTE 0.60

/**
 * @def DYNASTRAT_W_AUTOGRAD
 * @brief Target weight of the autograd pool under full stratification.
 *
 * @details
 * Smallest of the three weights: the autograd pool parallelises DAG
 * nodes only when they are independent, so extra threads beyond a few
 * add contention rather than speed. See @ref DYNASTRAT_W_COMPUTE for
 * the joint invariant of the three weights.
 *
 * @see DYNASTRAT_W_COMPUTE     Largest target weight.
 * @see DYNASTRAT_W_DTLOADER    Intermediate target weight.
 */
#define DYNASTRAT_W_AUTOGRAD 0.15

/**
 * @def DYNASTRAT_W_DTLOADER
 * @brief Target weight of the data-loading pool under full stratification.
 *
 * @details
 * Intermediate weight: the data-loading pool needs more than the
 * minimum without hoarding. See @ref DYNASTRAT_W_COMPUTE for the
 * joint invariant of the three weights.
 *
 * @see DYNASTRAT_W_COMPUTE     Largest target weight.
 * @see DYNASTRAT_W_AUTOGRAD    Smallest target weight.
 */
#define DYNASTRAT_W_DTLOADER 0.25

/**
 * @brief Stratify a total thread count across all managed groups.
 *
 * @details
 * Partitions @p threads into per-group budgets and returns them in a
 * @ref StratifiedThreads structure, mapping the largest share to
 * @c compute, the smallest to @c autograd and the intermediate share
 * to @c dtloader. The policy has three regimes selected by
 * @p threads: fair split at or below @ref DYNASTRAT_EQUAL_TH, target
 * weights at or above @ref DYNASTRAT_FULL_TH, and a linear blend of
 * both weight sets between those boundaries.
 *
 * @par Overcommit
 * Totals of 2 or 3 yield @c {1,1,1}: every pool stays alive even
 * though the sum (3) exceeds the input when @p threads is 2. For
 * totals above 3 the members add up to @p threads exactly; no thread
 * is lost to rounding.
 *
 * @par Ordering
 * Each member is at least @ref MIN_THREADS_PER_GROUP (1) and reflects
 * the number of threads the corresponding pool is allowed to use.
 * Totals at or above @ref DYNASTRAT_STRICT_TH satisfy @c compute >
 * @c dtloader > @c autograd strictly. Smaller totals may tie: at or
 * below @ref DYNASTRAT_EQUAL_TH fairness wins over gaps, and totals
 * from @ref DYNASTRAT_LIGHT_TH up get only the minimum gap.
 *
 * @param[in] threads  Total number of threads to distribute. Must be
 *                     at least 2.
 * @param[out] status  Receives the operation result. Set to
 *                     @ref novaSuccess on success, or to
 *                     @ref novaInvalidNumThreads when @p threads is
 *                     less than 2. Must not be @c nullptr.
 *
 * @return A @ref StratifiedThreads structure describing the per-group
 *          thread budget, or @c {0,0,0} on failure.
 *
 * @pre  @p status must not be @c nullptr.
 *
 * @post On success the returned budget is recorded for
 *       @ref get_last_stratification_result().
 *
 * @see StratifiedThreads        Structure returned by this function.
 * @see set_num_threads_to()     Applies a per-group budget.
 * @see MIN_THREADS_PER_GROUP    Minimum per-group threads.
 * @see DYNASTRAT_EQUAL_TH       Fair regime boundary.
 * @see DYNASTRAT_FULL_TH        Full stratification boundary.
 */
StratifiedThreads stratify_threads(uint32 threads, novaStatus_t *status);

/**
 * @brief Report whether manual per-group counts oversubscribe the machine.
 *
 * @details
 * Pure predicate over explicit counts: reads no shared state, so unit
 * tests need no hardware. Groups parked at exactly 1 are excluded
 * from the sum (a lone thread marks an idle pool, not provisioned
 * capacity); an unknown machine total (0) never reports
 * oversubscription. Used to warn on manual stratifications that
 * invite context-switching losses; dynastrat output can never
 * trigger it (it conserves the total, and all-ones counts are
 * excluded).
 *
 * @param[in] compute        Compute-group count under test.
 * @param[in] autograd       Autograd-group count under test.
 * @param[in] dtloader       Data-loader-group count under test.
 * @param[in] machine_total  Machine thread total to fit, or 0 when
 *                           unknown.
 *
 * @return @c true when the counted sum exceeds @p machine_total.
 */
bool manual_counts_oversubscribed(uint32 compute, uint32 autograd,
                                  uint32 dtloader, uint32 machine_total);

/**
 * @brief Read the distribution mode from explicit origins.
 *
 * @details
 * Pure derivation: manual when any origin is manual, automatic
 * otherwise when any origin is automatic, unset when nothing was
 * ever assigned. Unset groups are neutral in every combination.
 */
DistributionKind distribution_kind_of(GroupOrigin compute_origin,
                                      GroupOrigin autograd_origin,
                                      GroupOrigin dtloader_origin);

/**
 * @brief Read the live distribution mode of the group counters.
 */
DistributionKind current_distribution_kind();

/**
 * @brief Counted threads for oversubscription checks.
 *
 * @details
 * Groups parked at exactly 1 thread are excluded: a lone thread
 * marks an idle pool, not provisioned capacity.
 */
static inline uint64_t counted_thread_sum(uint32 compute, uint32 autograd,
                                          uint32 dtloader) {
  uint64_t sum = 0;
  sum += (compute != 1) ? (uint64_t)compute : 0U;
  sum += (autograd != 1) ? (uint64_t)autograd : 0U;
  sum += (dtloader != 1) ? (uint64_t)dtloader : 0U;
  return sum;
}

/**
 * @brief Query whether any stratification already succeeded.
 *
 * @details
 * Reports whether @ref stratify_threads() completed at least once, in
 * which case @ref get_last_stratification_result() holds a recorded
 * budget. @ref get_stratified_threads() uses this to serve the cached
 * result instead of recomputing while the global budget stays
 * immutable.
 *
 * @return @c true after the first successful stratification,
 *         @c false before.
 *
 * @see stratify_threads()              Call recording the result.
 * @see get_last_stratification_result()  Recorded budget accessor.
 */
bool is_stratification_complete();

/**
 * @brief Return the budget recorded by the latest stratification.
 *
 * @details
 * Returns the @ref StratifiedThreads value stored by the latest
 * successful @ref stratify_threads() call. Before any success the
 * recorded value is @c {0,0,0}; callers must check it with
 * @ref is_valid_stratification_result() before use.
 *
 * @return The recorded budget, or @c {0,0,0} when nothing was
 *         recorded yet.
 *
 * @see stratify_threads()              Call recording the result.
 * @see is_stratification_complete()    Cache state query.
 * @see is_valid_stratification_result()  Recorded budget validation.
 */
StratifiedThreads get_last_stratification_result();

#ifdef __cplusplus
}
#endif
