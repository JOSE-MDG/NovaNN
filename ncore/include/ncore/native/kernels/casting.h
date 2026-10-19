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
 * Element-wise cast from @p src's dtype to @p dst's dtype; both
 * tensors must have identical shape and live on the same device.
 *
 * @param[in]          src  Source tensor with a supported dtype,
 *                          allocated on the active compute device.
 * @param[in,out]      dst  Destination tensor with the target dtype,
 *                          matching shape, and allocated on the same
 *                          device as @p src.
 *
 * @return @ref novaSuccess on success, or an error status describing
 *         the failure.
 *
 * @pre A compute device (CUDA or HIP) must be available and
 *      initialized.
 * @post On success, @p dst holds the cast values.
 *
 * @warning Without a valid compute device this fails with
 *          @ref novaDeviceNotAvailable.
 */
novaStatus_t launchDtypeCastingKernel(const Tensor *restrict src,
                                      Tensor *restrict dst);

#ifdef __cplusplus
}
#endif
