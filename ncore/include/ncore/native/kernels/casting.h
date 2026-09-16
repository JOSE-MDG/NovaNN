/**
 * @file casting.h
 * @brief Device-agnostic entry point for dtype casting kernels.
 *
 * @details
 * Declares @ref launchDtypeCastingKernel(), which resolves the active
 * GPU backend at run time and forwards to
 * @ref launchCudaDtypeCastingKernel() or
 * @ref launchHipDtypeCastingKernel(). Implemented in
 * @c CastingDispatchImpl.cpp through
 * @ref ncore::dispatch::launch().
 *
 * @see CastingDispatchImpl.cpp  Dispatcher implementation.
 * @see DtypeCastingKernel.cu   CUDA kernel implementation.
 * @see DtypeCastingKernel.hip  HIP kernel implementation.
 */

#pragma once

#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <ncore/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Launch a dtype casting kernel from @p src to @p dst.
 *
 * @details
 * Performs an element-wise type cast from the source tensor's dtype to
 * the destination tensor's dtype.  Both tensors must have identical
 * shape and be located on the same device.  The dispatch logic selects
 * the appropriate backend kernel (CUDA or HIP) based on the detected
 * device.
 *
 * @param[in]      src  Source tensor.  Must have a supported dtype and be
 *                      allocated on the active compute device.
 * @param[in,out]  dst  Destination tensor.  Must have the target dtype,
 *                      matching shape, and be allocated on the same
 *                      device as @p src.
 *
 * @return @ref novaSuccess on success, or an error status describing
 *         the failure.
 *
 * @pre  Both @p src and @p dst must be valid, non-null tensors.
 * @pre  @p src and @p dst must have the same shape.
 * @pre  A compute device (CUDA or HIP) must be available and
 *       initialised.
 * @post On success, @p dst contains the casted elements.
 *
 * @warning Calling this function without a valid compute device
 *          results in @ref novaDeviceNotAvailable.
 */
novaStatus_t launchDtypeCastingKernel(const Tensor *restrict src,
                                      Tensor *restrict dst);

#ifdef __cplusplus
}
#endif
