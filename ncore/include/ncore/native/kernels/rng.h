/**
 * @file rng.h
 * @brief Device-agnostic entry points for RNG fill kernels.
 *
 * @details
 * Declares @ref launchRngFillU64() and @ref launchRngFillU32(), which
 * resolve the active GPU backend at run time and forward to
 * @ref launchCudaRngFillU64/U32 or @ref launchHipRngFillU64/U32.
 * Implemented in @c RngDispatchImpl.cpp through
 * @ref ncore::dispatch::launch().
 *
 * @see RngDispatchImpl.cpp  Dispatcher implementation.
 * @see RngFillKernel.cu    CUDA kernel implementation.
 * @see RngFillKernel.hip   HIP kernel implementation.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Fill device memory with 64-bit Philox draws.
 *
 * @details
 * Draws @p n values into @p dst, two per Philox block starting at
 * @p base_ctr. A request that is not a multiple of 2 consumes
 * its trailing block and drops the spare value, matching the host
 * path block granularity.
 *
 * @param[in]  key       Two key entries selecting the stream.
 * @param[in]  base_ctr  Four counter entries locating the first block.
 * @param[out] dst       Destination buffer in device memory. Must
 *                       hold at least @p n values.
 * @param[in]  n         Values to emit.
 *
 * @return @ref novaSuccess on success, or an error status describing
 *         the failure.
 *
 * @pre  A compute device (CUDA or HIP) must be available.
 * @post On success, @p dst holds the draws.
 *
 * Zero demand succeeds without touching @p dst.
 */
novaStatus_t launchRngFillU64(const uint32_t key[2], const uint32_t base_ctr[4],
                              uint64_t *dst, size_t n);

/**
 * @brief Fill device memory with 32-bit Philox draws.
 *
 * @details
 * Draws @p n values into @p dst, four per Philox block starting at
 * @p base_ctr. A request that is not a multiple of 4 consumes
 * its trailing block and drops the spare lanes.
 *
 * @param[in]  key       Two key entries selecting the stream.
 * @param[in]  base_ctr  Four counter entries locating the first block.
 * @param[out] dst       Destination buffer in device memory. Must
 *                       hold at least @p n values.
 * @param[in]  n         Values to emit.
 *
 * @return @ref novaSuccess on success, or an error status describing
 *         the failure.
 *
 * @pre  A compute device (CUDA or HIP) must be available.
 * @post On success, @p dst holds the draws.
 *
 * Zero demand succeeds without touching @p dst.
 */
novaStatus_t launchRngFillU32(const uint32_t key[2], const uint32_t base_ctr[4],
                              uint32_t *dst, size_t n);

#ifdef __cplusplus
}
#endif
