/**
 * @file ContiguousLayoutKernel.h
 * @brief Declarations for the contiguous-layout HIP kernels.
 *
 * @details
 * Declares @ref launchHipContiguousKernel(), the HIP mirror of
 * @c launchCudaContiguousKernel(). The device-agnostic dispatcher in
 * @c ContiguousDispatchImpl.cpp calls it after resolving the active
 * backend.
 *
 * @see ContiguousLayoutKernel.hip  Kernel implementation.
 */

#pragma once

#include <ncore/core/status.h>
#include <ncore/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Materialize a contiguous copy of @p src into @p dst on HIP.
 *
 * @details
 * Already-contiguous sources become views sharing storage; 1-D
 * tensors move through the device-to-device fast path; anything else
 * is collapsed and gathered by a grid-stride kernel whose geometry
 * comes from @ref resolve_launch_config() fed with the detected HIP
 * properties.
 *
 * @param[in]  src  Source tensor in HIP device memory.
 * @param[out] dst  Destination tensor in HIP device memory.
 *
 * @return @ref novaStatus_t with the launch outcome, or the property
 *         query error when the device caps cannot be read.
 */
novaStatus_t launchHipContiguousKernel(const Tensor *src, Tensor *dst);

#ifdef __cplusplus
}
#endif
