/**
 * @file RngFillKernel.h
 * @brief Declarations for the HIP RNG fill kernels. Same contract as
 * the CUDA header. See @c RngFillKernel.h in the CUDA tree.
 *
 * @see RngFillKernel.hip  Kernel implementation.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <ncore/core/status.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Fill HIP device memory with 64-bit Philox draws. Same
 * contract as @ref launchCudaRngFillU64.
 *
 * @param[in]  key       Two key entries selecting the stream.
 * @param[in]  base_ctr  Four counter entries locating the first block.
 * @param[out] dst       Destination buffer in HIP device memory.
 * @param[in]  n         Values to emit.
 * @param[in]  caller_stream Nullable caller stream; null launches on the
 *                            default stream.
 *
 * @return @ref novaStatus_t with the launch outcome.
 */
novaStatus_t launchHipRngFillU64(const uint32_t key[2],
                                 const uint32_t base_ctr[4], uint64_t *dst,
                                 size_t n, hipStream_t caller_stream);

/**
 * @brief Fill HIP device memory with 32-bit Philox draws. Same
 * contract as @ref launchCudaRngFillU32.
 *
 * @param[in]  key       Two key entries selecting the stream.
 * @param[in]  base_ctr  Four counter entries locating the first block.
 * @param[out] dst       Destination buffer in HIP device memory.
 * @param[in]  n         Values to emit.
 * @param[in]  caller_stream Nullable caller stream; null launches on the
 *                            default stream.
 *
 * @return @ref novaStatus_t with the launch outcome.
 */
novaStatus_t launchHipRngFillU32(const uint32_t key[2],
                                 const uint32_t base_ctr[4], uint32_t *dst,
                                 size_t n, hipStream_t caller_stream);

#ifdef __cplusplus
}
#endif
