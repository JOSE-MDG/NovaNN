/**
 * @file decide.h
 * @brief Per-family CPU parallelization decisions.
 *
 * @details
 * One pure predicate per operation family. Each function reads
 * only its work struct and the available thread count, runs O(1)
 * arithmetic (quotients capping the effective count by useful
 * grains, plus floor checks), and returns a @ref ParallelDecision for the
 * OpenMP guard. No allocation, no atomics, no shared state, no
 * status path: invalid inputs refuse with @c go set to false.
 * The effective count never exceeds the available threads, so a
 * region never forks more threads than useful grains.
 *
 * @see pattern.h  Work structs and the decision type.
 * @see grains.h   Per-thread grains consumed by the rules.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <ncore/threading/parallel/pattern.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Decide a streaming-map region.
 *
 * @param[in] work     Facts for the map. May be @c nullptr.
 * @param[in] threads  Available threads. Values below 2 refuse.
 *
 * @return Verdict with the effective count capped by useful grains.
 */
ParallelDecision decide_elementwise(const ElementwiseWork *work,
                                    uint32 threads);

/**
 * @brief Decide a layout or copy region.
 *
 * @param[in] work     Facts for the move. May be @c nullptr.
 * @param[in] threads  Available threads. Values below 2 refuse.
 *
 * @return Verdict with the effective count capped by useful grains.
 */
ParallelDecision decide_layout(const LayoutWork *work, uint32 threads);

/**
 * @brief Decide a reduction region.
 *
 * @param[in] work     Facts for the reduction. May be @c nullptr.
 * @param[in] threads  Available threads. Values below 2 refuse.
 *
 * @return Verdict with the effective count capped by useful grains.
 */
ParallelDecision decide_reduction(const ReductionWork *work, uint32 threads);

/**
 * @brief Decide a dense-product region.
 *
 * @param[in] work     Facts for the product. May be @c nullptr.
 * @param[in] threads  Available threads. Values below 2 refuse.
 *
 * @return Verdict with the effective count capped by useful grains.
 */
ParallelDecision decide_gemm(const GemmWork *work, uint32 threads);

/**
 * @brief Decide a neighborhood region.
 *
 * @param[in] work     Facts for the stencil. May be @c nullptr.
 * @param[in] threads  Available threads. Values below 2 refuse.
 *
 * @return Verdict with the effective count capped by useful grains.
 */
ParallelDecision decide_stencil(const StencilWork *work, uint32 threads);

/**
 * @brief Decide a prefix-scan region (batch span only).
 *
 * @param[in] work     Facts for the scan. May be @c nullptr.
 * @param[in] threads  Available threads. Values below 2 refuse.
 *
 * @return Verdict with the effective count capped by whole scans.
 */
ParallelDecision decide_scan(const ScanWork *work, uint32 threads);

/**
 * @brief Decide an indirect-access region.
 *
 * @param[in] work     Facts for the gather or scatter. May be
 *                     @c nullptr.
 * @param[in] threads  Available threads. Values below 2 refuse.
 *
 * @return Verdict with the effective count capped by useful grains.
 */
ParallelDecision decide_gather(const GatherWork *work, uint32 threads);

/**
 * @brief Decide a sort or histogram region.
 *
 * @param[in] work     Facts for the ordering. May be @c nullptr.
 * @param[in] threads  Available threads. Values below 2 refuse.
 *
 * @return Verdict with the effective count capped by useful grains.
 */
ParallelDecision decide_sort(const SortWork *work, uint32 threads);

/**
 * @brief Refuse unconditionally (serial family).
 *
 * @param[in] threads  Available threads, ignored.
 *
 * @return Verdict with @c go always false.
 */
ParallelDecision decide_serial(uint32 threads);

/**
 * @brief Combine stage verdicts for a fused region (all must clear).
 *
 * @param[in] stages  Stage verdicts. May be @c nullptr when empty.
 * @param[in] count   Number of stages.
 *
 * @return Verdict clearing only when every stage clears, with the
 *         smallest stage count.
 */
ParallelDecision decide_fused(const ParallelDecision *stages, size_t count);

#ifdef __cplusplus
}
#endif
