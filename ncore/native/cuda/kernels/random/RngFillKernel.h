/**
 * @file RngFillKernel.h
 * @brief Declarations for the CUDA RNG fill kernels.
 *
 * @details
 * Declares @ref launchCudaRngFillU64() and @ref launchCudaRngFillU32(),
 * the CUDA entry points for counter-based Philox fills. The
 * device-agnostic dispatcher in @c RngDispatchImpl.cpp calls them
 * after resolving the active backend.
 *
 * @see RngFillKernel.cu  Kernel implementation.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <ncore/core/status.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Fill CUDA device memory with 64-bit Philox draws.
 *
 * @details
 * Grid-stride kernel over Philox blocks sized with
 * @ref ncore::heuristics::kernels::resolveLaunchConfig() from the
 * detected CUDA properties. Two values per block. A trailing partial
 * request consumes its block and drops the spare value.
 *
 * @param[in]  key       Two key entries selecting the stream.
 * @param[in]  base_ctr  Four counter entries locating the first block.
 * @param[out] dst       Destination buffer in CUDA device memory.
 * @param[in]  n         Values to emit.
 * @param[in]  caller_stream Nullable caller stream; null launches on the
 *                            default stream.
 *
 * @return @ref novaStatus_t with the launch outcome
 *         (@ref novaKernelLaunchError on launch failure), or the
 *         property query error when the device caps cannot be read.
 */
novaStatus_t launchCudaRngFillU64(const uint32_t key[2],
                                  const uint32_t base_ctr[4], uint64_t *dst,
                                  size_t n, cudaStream_t caller_stream);

/**
 * @brief Fill CUDA device memory with 32-bit Philox draws.
 *
 * @details
 * Four values per block. A trailing partial request consumes its
 * block and drops the spare lanes.
 *
 * @param[in]  key       Two key entries selecting the stream.
 * @param[in]  base_ctr  Four counter entries locating the first block.
 * @param[out] dst       Destination buffer in CUDA device memory.
 * @param[in]  n         Values to emit.
 * @param[in]  caller_stream Nullable caller stream; null launches on the
 *                            default stream.
 *
 * @return @ref novaStatus_t with the launch outcome
 *         (@ref novaKernelLaunchError on launch failure), or the
 *         property query error when the device caps cannot be read.
 */
novaStatus_t launchCudaRngFillU32(const uint32_t key[2],
                                  const uint32_t base_ctr[4], uint32_t *dst,
                                  size_t n, cudaStream_t caller_stream);

#ifdef __cplusplus
}
#endif
