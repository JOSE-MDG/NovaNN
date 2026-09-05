/**
 * @file launch_config.h
 * @brief Shared GPU kernel launch geometry for the native backends.
 *
 * @details
 * Single place that turns a tensor size into a grid/block pair, so
 * CUDA and HIP launchers never drift apart with their own magic
 * numbers. The math is plain C on plain integers: no CUDA or HIP
 * headers are needed, which keeps this usable from @c .cu, @c .hip
 * and host @c .cpp alike, and testable on a machine without any GPU.
 * Each backend fetches its own caps (SM count, threads per block)
 * through its device-properties getter and builds @c dim3 itself.
 *
 * @see cudaDetectedDeviceProps_t  CUDA caps source.
 * @see hipDetectedDeviceProps_t   HIP caps source.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @def LAUNCH_BYTES_PER_BLOCK
 * @brief Target traffic per block, in bytes.
 *
 * @details
 * Grids are sized so every block moves roughly this much data. Small
 * transfers collapse to a handful of blocks instead of spraying idle
 * threads across the whole machine; large ones still saturate it
 * through the @ref LAUNCH_BLOCKS_PER_SM cap.
 */
#define LAUNCH_BYTES_PER_BLOCK (64u * 1024u)

/**
 * @def LAUNCH_BLOCKS_PER_SM
 * @brief Upper bound on blocks per streaming multiprocessor.
 *
 * @details
 * Caps the grid so a launch holds a few waves of blocks per SM:
 * enough to hide latency on memory-bound copies and casts, without
 * queueing thousands of blocks that only add scheduling overhead.
 * Mirrors the historical @c multiProcessorCount * 4 heuristic.
 */
#define LAUNCH_BLOCKS_PER_SM 4u

/**
 * @struct LaunchConfig
 * @brief Grid/block pair for a 1-D kernel launch.
 *
 * @details
 * Both members map directly onto @c dim3: @c blocks becomes
 * @c gridDim.x and @c threads becomes @c blockDim.x. Every kernel
 * launched through this struct must use a grid-stride loop, so any
 * grid size stays correct and only performance varies.
 */
typedef struct {
  unsigned int blocks;  ///< Number of blocks (@c gridDim.x). At least 1.
  unsigned int threads; ///< Threads per block (@c blockDim.x). Warp-aligned.
} LaunchConfig;

/**
 * @brief Resolve the grid/block pair for a tensor-sized 1-D launch.
 *
 * @details
 * Starts from 256 threads per block (a whole number of CUDA warps
 * and HIP wavefronts on every supported architecture) and scales the
 * grid with the traffic: one block per @ref LAUNCH_BYTES_PER_BLOCK
 * bytes, never more blocks than needed to cover the elements once,
 * never more than @ref LAUNCH_BLOCKS_PER_SM blocks per SM. An empty
 * tensor still yields a single block; grid-stride loops tolerate it.
 *
 * @param[in] num_elements          Element count to cover.
 * @param[in] item_size             Bytes per element. May be 0, in
 *                                  which case only the element count
 *                                  drives the grid.
 * @param[in] max_threads_per_block Device limit from the backend
 *                                  properties struct.
 * @param[in] sm_count              SM (CUDA) or CU (HIP) count from
 *                                  the backend properties struct.
 *
 * @return A @ref LaunchConfig with @c blocks >= 1 and @c threads
 *         clamped to @p max_threads_per_block (rounded down to a
 *         multiple of 64 when clamping, so warps and wavefronts stay
 *         whole).
 *
 * @note Pure arithmetic on integers: no runtime calls, no shared
 *       state, safe to call from anywhere including device-property
 *       queries on the host.
 *
 * @see LAUNCH_BYTES_PER_BLOCK
 * @see LAUNCH_BLOCKS_PER_SM
 */
static inline LaunchConfig
resolve_launch_config(size_t num_elements, size_t item_size,
                      unsigned int max_threads_per_block,
                      unsigned int sm_count) {
  LaunchConfig cfg = {1u, 256u};

  if (cfg.threads > max_threads_per_block) {
    // Degenerate caps guard: keep whole warps/wavefronts when the
    // device reports fewer than 256 threads per block.
    cfg.threads = (max_threads_per_block / 64u) * 64u;
    if (cfg.threads == 0u) {
      cfg.threads = max_threads_per_block;
    }
  }

  const unsigned long long total_bytes =
      (unsigned long long)num_elements * (unsigned long long)item_size;
  unsigned long long blocks =
      (total_bytes + LAUNCH_BYTES_PER_BLOCK - 1u) / LAUNCH_BYTES_PER_BLOCK;

  // Never launch more blocks than needed to cover every element once.
  const unsigned long long need =
      ((unsigned long long)num_elements + cfg.threads - 1u) / cfg.threads;
  if (blocks > need) {
    blocks = need;
  }
  if (sm_count > 0u) {
    const unsigned long long cap =
        (unsigned long long)sm_count * LAUNCH_BLOCKS_PER_SM;
    if (blocks > cap) {
      blocks = cap;
    }
  }
  if (blocks < 1u) {
    blocks = 1u;
  }
  if (blocks > (unsigned long long)~0u) {
    blocks = (unsigned long long)~0u;
  }
  cfg.blocks = (unsigned int)blocks;
  return cfg;
}

#ifdef __cplusplus
}
#endif
