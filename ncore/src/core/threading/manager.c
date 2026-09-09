/**
 * @file manager.c
 * @brief Thread-budget stratification across the managed groups.
 *
 * @details
 * Implements @ref stratify_threads() under the policy documented in
 * @c DYNASTRAT.md: fair split for small totals, target weights for
 * large totals, linear blend of both weight sets between those
 * regimes, Hamilton largest-remainder apportionment of the remainder
 * above the per-group minimum, and a final ordering pass that keeps
 * @c compute above @c dtloader and @c dtloader above @c autograd once
 * the total is large enough to hold both gaps.
 *
 * Every successful stratification is copied into @ref last_storage
 * and published through @ref last_ptr, so
 * @ref get_stratified_threads() can serve later calls from the cache
 * while the global budget stays immutable. Only the 8-byte pointer
 * travels atomically (a whole @ref StratifiedThreads fits no
 * lock-free atomic on every toolchain); the struct write is sequenced
 * before the publishing store. The split computation itself uses no
 * other shared state, allocates nothing and runs in constant time and
 * memory for the three managed groups.
 *
 * @see manager.h     Tuning constants and public contract.
 * @see DYNASTRAT.md  Policy rationale, worked traces and tuning guide.
 * @see threads.h     @ref StratifiedThreads definition.
 * @see threads.c     Caller resolving the total budget.
 */

#include <stdatomic.h>

#include "manager.h"
#include "ncore/threading/threads.h"

// Upper bound for the strict ordering loop: each pass closes at least
// one violated gap while the total stays constant, so a handful of
// passes always suffices; the bound keeps the runtime constant.
#define DYNASTRAT_MAX_ORDER_PASSES 10

/**
 * @var last_storage
 * @brief Backing slot for the recorded stratification budget.
 *
 * @details Plain (non-atomic) storage holding the latest successful
 * @ref StratifiedThreads value. Written only by @ref record_success()
 * before publishing @ref last_ptr, so readers that observe a
 * non-null pointer always see a fully written budget. Static storage
 * zero-initialises it to @c {0,0,0}.
 */
static StratifiedThreads last_storage;

/**
 * @var last_ptr
 * @brief Publication pointer for the recorded budget, or null.
 *
 * @details Null until the first @ref stratify_threads() success;
 * afterwards it permanently points at @ref last_storage. The pointer
 * alone travels atomically (8 bytes, lock-free on every toolchain),
 * which is what lets this cache link without @c libatomic. Read with
 * acquire ordering, written with release ordering.
 */
static _Atomic(StratifiedThreads *) last_ptr;

/**
 * @brief Record a successful stratification for the cache.
 *
 * @details
 * Copies @p result into @ref last_storage and then publishes
 * @ref last_ptr, with release ordering on the publish. The order is
 * load-bearing: readers dereference the pointer only after observing
 * it non-null (see @ref get_last_stratification_result()), and
 * release/acquire publishes exactly the writes sequenced before the
 * pointer store. Every success path in @ref stratify_threads()
 * funnels through here so none can diverge.
 *
 * @param[in] result  Budget to record. Always a successful split.
 *                    Must not be @c nullptr.
 */
static inline void record_success(const StratifiedThreads *result) {
  last_storage = *result;
  atomic_store_explicit(&last_ptr, &last_storage, memory_order_release);
}

/**
 * @brief Stratify a total thread count across all managed groups.
 *
 * @details
 * Partitions @p threads into per-group budgets under the DYNASTRAT
 * policy (see @c DYNASTRAT.md ): totals of 2 or 3 overcommit to
 * @c {1,1,1} so no pool starves; larger totals are split by fair
 * weights, target weights, or a linear blend of both depending on
 * the regime, apportioned with largest remainders and finished with
 * an ordering fixup. Every member of a successful result is at least
 * @ref MIN_THREADS_PER_GROUP, and totals above 3 add up exactly.
 *
 * Every success is copied into @ref last_storage and published
 * through @ref last_ptr with release ordering, so
 * @ref get_stratified_threads() can serve later calls from the
 * cache. Failures leave the recorded values untouched.
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
 * @note Thread-safe. The split itself uses only local state; the
 *       only shared writes are the release-ordered record stores.
 *
 * @see DYNASTRAT.md  Policy rationale and reference table.
 * @see manager.h     Tuning constants and public contract.
 */
StratifiedThreads stratify_threads(uint32 threads, novaStatus_t *status) {
  if (status == nullptr) {
    return (StratifiedThreads){
        .compute = 0,
        .autograd = 0,
        .dtloader = 0,
    };
  }

  if (threads < 2) {
    *status = (novaStatus_t){
        .err = novaInvalidNumThreads,
        .message = "Invalid thread count: must be at least 2.",
    };
    return (StratifiedThreads){
        .compute = 0,
        .autograd = 0,
        .dtloader = 0,
    };
  }

  if (threads <= 3) {
    // Overcommit by design: every pool stays alive even when the sum exceeds the input.
    *status = (novaStatus_t){
        .err = novaSuccess,
        .message = nova_get_error_msg(novaSuccess, nullptr),
    };
    const StratifiedThreads overcommit = {
        .compute = 1,
        .autograd = 1,
        .dtloader = 1,
    };
    record_success(&overcommit);
    return (StratifiedThreads){
        .compute = 1,
        .autograd = 1,
        .dtloader = 1,
    };
  }

  // Remainder above the per-group minimum: the only part that is stratified.
  const uint32 remainder =
      threads - (uint32)(NUM_PARALLEL_GROUPS * MIN_THREADS_PER_GROUP);
  if (remainder == 0) {
    // Unreachable with the unit minimum, kept so a larger minimum degrades to an even split.
    *status = (novaStatus_t){
        .err = novaSuccess,
        .message = nova_get_error_msg(novaSuccess, nullptr),
    };
    const StratifiedThreads even = {
        .compute = MIN_THREADS_PER_GROUP,
        .autograd = MIN_THREADS_PER_GROUP,
        .dtloader = MIN_THREADS_PER_GROUP,
    };
    record_success(&even);
    return even;
  }

  // Dynamic weights: fair split up to DYNASTRAT_EQUAL_TH, target
  // weights from DYNASTRAT_FULL_TH on, linear blend between both.
  double factor = 0.0;
  if (threads >= DYNASTRAT_FULL_TH) {
    factor = 1.0;
  } else if (threads > DYNASTRAT_EQUAL_TH) {
    factor = ((double)threads - (double)DYNASTRAT_EQUAL_TH) /
             ((double)DYNASTRAT_FULL_TH - (double)DYNASTRAT_EQUAL_TH);
  }
  const double fair = 1.0 / 3.0;
  // Index order matches the return struct: 0 compute, 1 autograd, 2 dtloader.
  const double weights[3] = {
      (fair * (1.0 - factor)) + (DYNASTRAT_W_COMPUTE * factor),
      (fair * (1.0 - factor)) + (DYNASTRAT_W_AUTOGRAD * factor),
      (fair * (1.0 - factor)) + (DYNASTRAT_W_DTLOADER * factor),
  };

  // Hamilton largest-remainder apportionment over the remainder.
  const double ideal[3] = {
      (double)remainder * weights[0],
      (double)remainder * weights[1],
      (double)remainder * weights[2],
  };
  // Truncation equals floor here: every ideal value is non-negative.
  uint32 base[3] = {
      (uint32)ideal[0],
      (uint32)ideal[1],
      (uint32)ideal[2],
  };
  const double rest[3] = {
      ideal[0] - (double)base[0],
      ideal[1] - (double)base[1],
      ideal[2] - (double)base[2],
  };
  // Three truncations lose fewer than 3 units, so the deficit is 0, 1 or 2.
  const uint32 deficit = remainder - (base[0] + base[1] + base[2]);

  // Hand the extra units to the largest remainders. Visit order encodes
  // tie priority (compute, dtloader, autograd): replacement happens only
  // on a strictly greater remainder, so exact ties keep the earlier pool.
  const uint32 priority[3] = {0, 2, 1};
  bool awarded[3] = {false, false, false};
  for (uint32 k = 0; k < deficit; ++k) {
    uint32 best = priority[0];
    double best_rest = -1.0;
    for (uint32 p = 0; p < 3; ++p) {
      const uint32 i = priority[p];
      if (!awarded[i] && rest[i] > best_rest) {
        best = i;
        best_rest = rest[i];
      }
    }
    awarded[best] = true;
    base[best] += 1;
  }

  auto compute = base[0] + MIN_THREADS_PER_GROUP;
  auto autograd = base[1] + MIN_THREADS_PER_GROUP;
  auto dtloader = base[2] + MIN_THREADS_PER_GROUP;

  if (threads >= DYNASTRAT_STRICT_TH) {
    // Both unit gaps fit from here: enforce the strict order.
    for (uint32 pass = 0; pass < DYNASTRAT_MAX_ORDER_PASSES; ++pass) {
      // Every move keeps the donor at or above the minimum.
      if (compute <= dtloader && dtloader > MIN_THREADS_PER_GROUP) {
        compute += 1;
        dtloader -= 1;
      }
      if (dtloader <= autograd && autograd > MIN_THREADS_PER_GROUP) {
        dtloader += 1;
        autograd -= 1;
      }
      if (compute > dtloader && dtloader > autograd) {
        break;
      }
    }
  } else if (threads >= DYNASTRAT_LIGHT_TH) {
    // Only the minimum gap fits: single light pass, no strict three-level order.
    if (compute <= dtloader && dtloader > MIN_THREADS_PER_GROUP) {
      compute += 1;
      dtloader -= 1;
    }
    if (dtloader < autograd && autograd > MIN_THREADS_PER_GROUP) {
      dtloader += 1;
      autograd -= 1;
    }
    if (compute == dtloader && dtloader > MIN_THREADS_PER_GROUP) {
      compute += 1;
      dtloader -= 1;
    }
    if (dtloader < autograd && autograd > MIN_THREADS_PER_GROUP) {
      dtloader += 1;
      autograd -= 1;
    }
  }
  // At or below DYNASTRAT_EQUAL_TH the Hamilton result stands: fairness wins over gaps.

  *status = (novaStatus_t){
      .err = novaSuccess,
      .message = nova_get_error_msg(novaSuccess, nullptr),
  };
  const StratifiedThreads recorded = {
      .compute = compute,
      .autograd = autograd,
      .dtloader = dtloader,
  };
  record_success(&recorded);
  return (StratifiedThreads){
      .compute = compute,
      .autograd = autograd,
      .dtloader = dtloader,
  };
}

/**
 * @brief Query whether any stratification already succeeded.
 *
 * @details
 * Reads @ref last_ptr with acquire ordering. Returns @c true once
 * @ref stratify_threads() has completed at least once, meaning
 * @ref get_last_stratification_result() serves a recorded budget
 * that @ref get_stratified_threads() can hand out from the cache.
 *
 * @return @c true after the first successful stratification,
 *         @c false before.
 *
 * @note Thread-safe. Pairs with the release-ordered publish in
 *       @ref record_success().
 *
 * @see stratify_threads()  Call publishing the pointer.
 * @see get_last_stratification_result()  Recorded budget accessor.
 */
bool is_stratification_complete() {
  return atomic_load_explicit(&last_ptr, memory_order_acquire) != nullptr;
}

/**
 * @brief Return the budget recorded by the latest stratification.
 *
 * @details
 * Loads @ref last_ptr with acquire ordering and dereferences it.
 * Before any successful @ref stratify_threads() call the pointer is
 * null and the result is @c {0,0,0}, so callers must validate the
 * result with @ref is_valid_stratification_result() before applying
 * it.
 *
 * @return The recorded budget, or @c {0,0,0} when nothing was
 *         recorded yet.
 *
 * @note Thread-safe. Pairs with the release-ordered publish in
 *       @ref record_success().
 *
 * @see stratify_threads()  Call storing the budget.
 * @see is_valid_stratification_result()  Recorded budget validation.
 */
StratifiedThreads get_last_stratification_result() {
  StratifiedThreads *recorded =
      atomic_load_explicit(&last_ptr, memory_order_acquire);
  if (recorded == nullptr) {
    return (StratifiedThreads){0, 0, 0};
  }
  return *recorded;
}

/**
 * @brief Report whether manual per-group counts oversubscribe the machine.
 *
 * @details
 * Pure predicate over explicit counts: no shared state is read, so
 * unit tests need no hardware. Groups parked at exactly 1 are
 * excluded from the sum; an unknown machine total (0) never
 * reports oversubscription.
 */
bool manual_counts_oversubscribed(uint32 compute, uint32 autograd,
                                  uint32 dtloader, uint32 machine_total) {
  if (machine_total == 0) {
    return false;
  }
  return counted_thread_sum(compute, autograd, dtloader) > machine_total;
}
