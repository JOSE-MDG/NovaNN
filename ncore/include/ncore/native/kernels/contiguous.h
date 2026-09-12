/**
 * @file contiguous.h
 * @brief Device-agnostic entry point for contiguous-layout kernels.
 *
 * @details
 * Declares @ref launchContiguousKernel(), which resolves the active
 * GPU backend at run time and forwards to
 * @ref launchCudaContiguousKernel() or
 * @ref launchHipContiguousKernel(). Implemented in
 * @c ContiguousDispatchImpl.cpp through
 * @ref ncore::dispatch::launch().
 *
 * @see ContiguousDispatchImpl.cpp  Dispatcher implementation.
 * @see ContiguousLayoutKernel.cu   CUDA kernel implementation.
 * @see ContiguousLayoutKernel.hip  HIP kernel implementation.
 */

#pragma once

#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <ncore/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Launch a contiguous-layout kernel on the detected device.
 *
 * @details
 * Callers handle already-contiguous inputs before dispatch (see @c tensor.c );
 * this entry point assumes a non-contiguous source and always materializes
 * a copy: dense one-dimensional tensors move through the device-to-device
 * fast path; column-contiguous two-dimensional views take a shared-memory
 * tile with fixed geometry; anything else is collapsed and gathered by a
 * grid-stride kernel with the widest provable vector width, sized for
 * the detected backend.
 *
 * @param[in]  src  Source tensor in device memory.
 * @param[in,out] dst  Destination tensor receiving the contiguous copy.
 *
 * @return @ref novaStatus_t with the launch outcome, or an error
 *         status when no backend can run.
 *
 * @pre  @p src and @p dst must be valid tensors.
 * @post On success, @p dst holds a contiguous copy of @p src.
 */
novaStatus_t launchContiguousKernel(const Tensor *restrict src,
                                    Tensor *restrict dst);

#ifdef __cplusplus
}
#endif
