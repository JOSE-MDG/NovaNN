/**
 * @file decide.c
 * @brief Per-family CPU parallelization decisions.
 *
 * @details
 * Implements the predicates declared in decide.h. Every rule
 * follows the same floor: thread counts below 2, null work, or
 * failed structural checks refuse with no fork. Past the floor
 * each rule divides once to cap the effective count by useful
 * grains, so decisions cost O(1) with no allocation, no atomics,
 * and no shared state. The effective count never exceeds the
 * available threads.
 */

#include <stddef.h>
#include <stdint.h>

#include <ncore/threading/parallel/decide.h>
#include <ncore/threading/parallel/grains.h>
#include <ncore/threading/parallel/pattern.h>

/**
 * @brief Cap a grain quotient by the available threads.
 *
 * @details
 * Single quotient plus bound: threads beyond the useful grain
 * count stay idle by construction instead of forking.
 */
static inline ParallelDecision cap_count(size_t quot, uint32 threads) {
  ParallelDecision refusal = {
      .go = false,
      .num_threads = 1,
  };
  size_t eff = quot;
  if (eff > (size_t)threads) {
    eff = (size_t)threads;
  }
  if (eff < 2) {
    return refusal;
  }
  ParallelDecision verdict = {
      .go = true,
      .num_threads = (uint32)eff,
  };
  return verdict;
}

ParallelDecision decide_elementwise(const ElementwiseWork *work,
                                    uint32 threads) {
  ParallelDecision refusal = {
      .go = false,
      .num_threads = 1,
  };
  if (threads < 2 || work == nullptr || !work->valid) {
    return refusal;
  }
  const size_t grain = (int)work->heavy ? PARALLEL_GRAIN_ELEMENTWISE_PACKED
                                        : PARALLEL_GRAIN_ELEMENTWISE;
  if (work->item_size == 1 &&
      work->total_bytes < (size_t)threads * PARALLEL_GRAIN_TINY_BYTES) {
    return refusal;
  }
  return cap_count(work->logical_size / grain, threads);
}

ParallelDecision decide_layout(const LayoutWork *work, uint32 threads) {
  ParallelDecision refusal = {
      .go = false,
      .num_threads = 1,
  };
  if (threads < 2 || work == nullptr || !work->valid) {
    return refusal;
  }
  // Dense moves are plain copies handled by caller branches; the
  // grain rule below is identical either way.
  (void)work->is_dense;
  const size_t elem_grain = (int)work->heavy ? PARALLEL_GRAIN_ELEMENTWISE_PACKED
                                             : PARALLEL_GRAIN_ELEMENTWISE;
  if (work->item_size == 1 &&
      work->total_bytes < (size_t)threads * PARALLEL_GRAIN_TINY_BYTES) {
    return refusal;
  }
  size_t eff = work->num_elements / elem_grain;
  const size_t eff_bytes = work->total_bytes / PARALLEL_GRAIN_LAYOUT_BYTES;
  if (eff_bytes < eff) {
    eff = eff_bytes;
  }
  return cap_count(eff, threads);
}

ParallelDecision decide_reduction(const ReductionWork *work, uint32 threads) {
  ParallelDecision refusal = {
      .go = false,
      .num_threads = 1,
  };
  if (threads < 2 || work == nullptr || !work->valid) {
    return refusal;
  }
  // Thin extents only pay off over a wide batch; thin and narrow
  // together never amortize the partials plus combine.
  if (work->extent < PARALLEL_GRAIN_REDUCTION_EXTENT_FLOOR &&
      work->batch < (size_t)threads * PARALLEL_GRAIN_REDUCTION_BATCH_FACTOR) {
    return refusal;
  }
  const size_t grain = (int)work->heavy ? PARALLEL_GRAIN_ELEMENTWISE_PACKED
                                        : PARALLEL_GRAIN_REDUCTION;
  return cap_count(work->total / grain, threads);
}

ParallelDecision decide_gemm(const GemmWork *work, uint32 threads) {
  ParallelDecision refusal = {
      .go = false,
      .num_threads = 1,
  };
  if (threads < 2 || work == nullptr || !work->valid) {
    return refusal;
  }
  // Outer counts the parallel tiles as chunked by the caller, so a
  // single tile can never spread regardless of arithmetic volume.
  size_t eff = work->arith / PARALLEL_GRAIN_GEMM_OPS;
  if (work->outer < eff) {
    eff = work->outer;
  }
  return cap_count(eff, threads);
}

ParallelDecision decide_stencil(const StencilWork *work, uint32 threads) {
  ParallelDecision refusal = {
      .go = false,
      .num_threads = 1,
  };
  if (threads < 2 || work == nullptr || !work->valid) {
    return refusal;
  }
  return cap_count(work->points / PARALLEL_GRAIN_STENCIL, threads);
}

ParallelDecision decide_scan(const ScanWork *work, uint32 threads) {
  ParallelDecision refusal = {
      .go = false,
      .num_threads = 1,
  };
  if (threads < 2 || work == nullptr || !work->valid) {
    return refusal;
  }
  // Axis dependencies confine threads to whole scans.
  if (work->batch < 2) {
    return refusal;
  }
  size_t eff = work->total / PARALLEL_GRAIN_SCAN;
  if (work->batch < eff) {
    eff = work->batch;
  }
  return cap_count(eff, threads);
}

ParallelDecision decide_gather(const GatherWork *work, uint32 threads) {
  ParallelDecision refusal = {
      .go = false,
      .num_threads = 1,
  };
  if (threads < 2 || work == nullptr || !work->valid) {
    return refusal;
  }
  // Short segments pay random access per index with no reuse.
  if (work->segment < PARALLEL_GRAIN_GATHER_SEGMENT_FLOOR) {
    return refusal;
  }
  return cap_count(work->indices / PARALLEL_GRAIN_GATHER, threads);
}

ParallelDecision decide_sort(const SortWork *work, uint32 threads) {
  ParallelDecision refusal = {
      .go = false,
      .num_threads = 1,
  };
  if (threads < 2 || work == nullptr || !work->valid) {
    return refusal;
  }
  if (work->total < PARALLEL_GRAIN_SORT_FLOOR) {
    return refusal;
  }
  return cap_count(work->total / PARALLEL_GRAIN_SORT, threads);
}

ParallelDecision decide_serial(uint32 threads) {
  (void)threads;
  ParallelDecision refusal = {
      .go = false,
      .num_threads = 1,
  };
  return refusal;
}

ParallelDecision decide_fused(const ParallelDecision *stages, size_t count) {
  ParallelDecision refusal = {
      .go = false,
      .num_threads = 1,
  };
  if (stages == nullptr || count == 0) {
    return refusal;
  }
  uint32 eff = UINT32_MAX;
  for (size_t i = 0; i < count; ++i) {
    if (!stages[i].go) {
      return refusal;
    }
    if (stages[i].num_threads < eff) {
      eff = stages[i].num_threads;
    }
  }
  ParallelDecision verdict = {
      .go = true,
      .num_threads = eff,
  };
  return verdict;
}
