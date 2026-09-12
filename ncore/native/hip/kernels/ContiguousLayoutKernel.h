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
#include <ncore/headeronly/macros.h>
#include <ncore/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Materialize a contiguous copy of @p src into @p dst on HIP.
 *
 * @details
 * Already-contiguous inputs are handled by the caller before dispatch
 * (see @c tensor.c ); this launcher assumes a non-contiguous source and
 * always materializes a copy: 1-D tensors move through the
 * device-to-device fast path; anything else is collapsed and gathered
 * by a grid-stride kernel whose geometry comes from
 * @ref ncore::heuristics::kernels::resolveLaunchConfig() fed with the
 * detected HIP properties.
 *
 * @param[in]  src  Source tensor in HIP device memory.
 * @param[out] dst  Destination tensor in HIP device memory.
 *
 * @return @ref novaStatus_t with the launch outcome
 *         (@ref novaKernelLaunchError on launch failure), or the property
 *         query error when the device caps cannot be read.
 */
novaStatus_t launchHipContiguousKernel(const Tensor *restrict src,
                                       Tensor *restrict dst);

#ifdef __cplusplus
}
#endif
