/**
 * @file grains.h
 * @brief Tunable per-thread grains for CPU parallelization decisions.
 *
 * @details
 * One grain per operation family: the minimum useful work per
 * thread before an OpenMP fork pays off on reference hardware.
 * Values are deliberately coarse (fork overhead dominates below
 * them) and stay tunable here without touching the rules in
 * decide.c. Packed and tiny-element cases reuse the plain grains
 * through multipliers instead of separate constants.
 *
 * @see decide.h  Rules consuming these grains.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @def PARALLEL_GRAIN_ELEMENTWISE
 * @brief Unpacked elements per thread for streaming maps.
 */
#define PARALLEL_GRAIN_ELEMENTWISE 8192

/**
 * @def PARALLEL_GRAIN_ELEMENTWISE_PACKED
 * @brief Unpacked elements per thread when unpacking is required.
 */
#define PARALLEL_GRAIN_ELEMENTWISE_PACKED 16384

/**
 * @def PARALLEL_GRAIN_LAYOUT_BYTES
 * @brief Moved bytes per thread for layout and copy work.
 */
#define PARALLEL_GRAIN_LAYOUT_BYTES 32768

/**
 * @def PARALLEL_GRAIN_REDUCTION
 * @brief Elements per thread for reductions (covers the combine).
 */
#define PARALLEL_GRAIN_REDUCTION 16384

/**
 * @def PARALLEL_GRAIN_REDUCTION_EXTENT_FLOOR
 * @brief Reduction extent below which batch must carry the work.
 */
#define PARALLEL_GRAIN_REDUCTION_EXTENT_FLOOR 32

/**
 * @def PARALLEL_GRAIN_REDUCTION_BATCH_FACTOR
 * @brief Batch multiple of threads compensating a thin extent.
 */
#define PARALLEL_GRAIN_REDUCTION_BATCH_FACTOR 4

/**
 * @def PARALLEL_GRAIN_GEMM_OPS
 * @brief Multiply-adds per thread for dense products.
 */
#define PARALLEL_GRAIN_GEMM_OPS 262144

/**
 * @def PARALLEL_GRAIN_STENCIL
 * @brief Output points per thread for neighborhoods.
 */
#define PARALLEL_GRAIN_STENCIL 4096

/**
 * @def PARALLEL_GRAIN_SCAN
 * @brief Total elements per thread for batched scans.
 */
#define PARALLEL_GRAIN_SCAN 16384

/**
 * @def PARALLEL_GRAIN_GATHER
 * @brief Indices per thread for indirect access.
 */
#define PARALLEL_GRAIN_GATHER 8192

/**
 * @def PARALLEL_GRAIN_GATHER_SEGMENT_FLOOR
 * @brief Segment bytes below which random access never pays off.
 */
#define PARALLEL_GRAIN_GATHER_SEGMENT_FLOOR 16

/**
 * @def PARALLEL_GRAIN_SORT_FLOOR
 * @brief Total elements below which ordering stays serial.
 */
#define PARALLEL_GRAIN_SORT_FLOOR 65536

/**
 * @def PARALLEL_GRAIN_SORT
 * @brief Elements per thread for sorts and histograms.
 */
#define PARALLEL_GRAIN_SORT 16384

/**
 * @def PARALLEL_GRAIN_TINY_BYTES
 * @brief Backing bytes per thread for one-byte element sizes.
 */
#define PARALLEL_GRAIN_TINY_BYTES 8192

#ifdef __cplusplus
}
#endif
