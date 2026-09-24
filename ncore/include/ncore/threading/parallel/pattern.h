/**
 * @file pattern.h
 * @brief Vocabulary for CPU parallelization decisions.
 *
 * @details
 * Declares @ref ParallelPattern (the operation family selecting the
 * rule), @ref ParallelDecision (the go verdict with its effective
 * thread count), and one plain work struct per family carrying only
 * the facts its rule reads. Work structs are built by call sites
 * (see helpers.h); decision functions (see decide.h) only read
 * them. All types are plain arithmetic so decisions stay O(1) with
 * no allocation and no shared state.
 *
 * @see decide.h   Per-family decision functions.
 * @see grains.h   Tunable per-thread grains consumed by the rules.
 * @see helpers.h  Call-site builders from @ref Tensor metadata.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <ncore/core/dtype.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum ParallelPattern
 * @brief Operation family selecting the parallelization rule.
 *
 * @details
 * Groups the CPU operation surface by execution shape rather than
 * by semantic directory: streaming maps share one rule, reductions
 * another, and so on. @c ParallelSerial covers work that never
 * parallelizes at operation level (random generation, FFT
 * internals, direct factorizations). @c ParallelFused carries no
 * rule of its own and routes through stage verdicts.
 */
typedef enum ParallelPattern : uint8_t {
  ParallelElementwise = 0, ///< Flat streaming maps.
  ParallelLayout = 1,      ///< Copies, permutes, casts, fills.
  ParallelReduction = 2,   ///< Axis reductions over a batch.
  ParallelGemm = 3,        ///< Dense matrix products over batches.
  ParallelStencil = 4,     ///< Convolutions, pooling, neighborhoods.
  ParallelScan = 5,        ///< Prefix scans along one axis.
  ParallelGather = 6,      ///< Indirect indexed reads and writes.
  ParallelSort = 7,        ///< Sorts and histograms.
  ParallelSerial = 8,      ///< Never parallel at operation level.
  ParallelFused = 9,       ///< Multi-stage composition, routed.
} ParallelPattern;

/**
 * @struct ParallelDecision
 * @brief Verdict of one parallelization decision.
 *
 * @details
 * Consumed directly by the OpenMP guard at the call site:
 * @c #pragma omp parallel for num_threads(num_threads)
 * schedule(static) if(go). When @c go is false the serial loop
 * body runs with no fork, so the refusal path costs one branch.
 */
typedef struct {
  bool go;            ///< True when a parallel region pays off.
  uint32 num_threads; ///< Effective threads, valid only when @c go.
} ParallelDecision;

/**
 * @struct ElementwiseWork
 * @brief Facts for one streaming-map decision.
 */
typedef struct {
  size_t logical_size; ///< Unpacked elements to process.
  size_t total_bytes;  ///< Backing bytes moved.
  size_t item_size;    ///< Bytes per storage unit.
  size_t packing;      ///< Unpacked elements per unit, 1 when N/A.
  bool heavy;          ///< True when unpack or dequant inflates cost.
  bool valid;          ///< Structural checks passed (see helpers.h).
} ElementwiseWork;

/**
 * @struct LayoutWork
 * @brief Facts for one layout or copy decision.
 */
typedef struct {
  size_t num_elements; ///< Storage units to move.
  size_t total_bytes;  ///< Backing bytes moved.
  size_t item_size;    ///< Bytes per storage unit.
  size_t packing;      ///< Unpacked elements per unit, 1 when N/A.
  bool heavy;          ///< True when unpack or dequant inflates cost.
  bool is_dense;       ///< True when the move is a plain copy.
  bool valid;          ///< Structural checks passed (see helpers.h).
} LayoutWork;

/**
 * @struct ReductionWork
 * @brief Facts for one reduction decision.
 */
typedef struct {
  size_t total;     ///< Elements covered (extent times batch).
  size_t extent;    ///< Elements reduced per output.
  size_t batch;     ///< Independent reductions.
  size_t item_size; ///< Bytes per storage unit.
  size_t packing;   ///< Unpacked elements per unit, 1 when N/A.
  bool heavy;       ///< True when unpack or dequant inflates cost.
  bool valid;       ///< Structural checks passed (see helpers.h).
} ReductionWork;

/**
 * @struct GemmWork
 * @brief Facts for one dense-product decision.
 */
typedef struct {
  size_t arith; ///< Multiply-adds (M times N times K times batch).
  size_t outer; ///< Independent row groups (M times batch).
  bool valid;   ///< Shape checks passed (see helpers.h).
} GemmWork;

/**
 * @struct StencilWork
 * @brief Facts for one neighborhood decision.
 */
typedef struct {
  size_t points; ///< Output points to compute.
  bool valid;    ///< Shape checks passed (see helpers.h).
} StencilWork;

/**
 * @struct ScanWork
 * @brief Facts for one prefix-scan decision.
 *
 * @details
 * Scans carry a dependency along the axis, so parallelism only
 * spans the batch: every thread owns whole scans.
 */
typedef struct {
  size_t total;  ///< Elements covered (extent times batch).
  size_t extent; ///< Elements per scan.
  size_t batch;  ///< Independent scans.
  bool valid;    ///< Shape checks passed (see helpers.h).
} ScanWork;

/**
 * @struct GatherWork
 * @brief Facts for one indirect-access decision.
 */
typedef struct {
  size_t indices; ///< Indexed elements to move.
  size_t segment; ///< Bytes per index (contiguous run length).
  bool valid;     ///< Shape checks passed (see helpers.h).
} GatherWork;

/**
 * @struct SortWork
 * @brief Facts for one sort or histogram decision.
 */
typedef struct {
  size_t total; ///< Elements to order or count.
  bool valid;   ///< Shape checks passed (see helpers.h).
} SortWork;

#ifdef __cplusplus
}
#endif
