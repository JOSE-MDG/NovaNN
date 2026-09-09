/**
 * @file ContiguousLayoutKernel.h
 * @brief Declarations for the contiguous-layout CUDA kernels.
 *
 * @details
 * Declares @ref launchCudaContiguousKernel(), the CUDA entry point
 * for materializing contiguous tensor layouts. The device-agnostic
 * dispatcher in @c ContiguousDispatchImpl.cpp calls it after
 * resolving the active backend.
 *
 * @see ContiguousLayoutKernel.cu  Kernel implementation.
 */

#pragma once

#include <ncore/core/status.h>
#include <ncore/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Materialize a contiguous copy of @p src into @p dst on CUDA.
 *
 * @details
 * Already-contiguous sources become views sharing storage; 1-D
 * tensors move through the device-to-device fast path; anything else
 * is collapsed and gathered by a grid-stride kernel whose geometry
 * comes from @ref ncore::heuristics::kernels::resolveLaunchConfig()
 * fed with the detected CUDA properties.
 *
 * @param[in]  src  Source tensor in CUDA device memory.
 * @param[out] dst  Destination tensor in CUDA device memory.
 *
 * @return @ref novaStatus_t with the launch outcome, or the property
 *         query error when the device caps cannot be read.
 */
novaStatus_t launchCudaContiguousKernel(const Tensor *restrict src,
                                        Tensor *restrict dst);

#ifdef __cplusplus
}
#endif
