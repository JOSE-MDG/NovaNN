/**
 * @file DtypeCastingKernel.h
 * @brief HIP backend entry point for dtype casting dispatch.
 *
 * @details
 * Declares the C-linkage function that the device-agnostic dispatch
 * layer (@ref CastingDispatchImpl.cpp) calls to perform dtype casting
 * on HIP devices.
 *
 * @see DtypeCastingKernel.hip
 * @see launchDtypeCastingKernel()
 * @see CastingDispatchImpl.cpp
 */

#pragma once

#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <ncore/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Launch a dtype casting kernel on the HIP device.
 *
 * @details
 * Backend-specific entry point called by @ref launchDtypeCastingKernel
 * after the dispatch layer has resolved the active device to HIP.
 *
 * @param[in]  src  Source tensor.  Must reside in HIP device memory
 *                  and have a supported source dtype.
 * @param[in,out] dst  Destination tensor.  Must reside in HIP device
 *                     memory, have the target dtype, and match
 *                     @p src in shape.
 *
 * @return @ref novaSuccess on success, or an error status describing
 *         the failure.
 *
 * @pre  Both @p src and @p dst must be allocated on the HIP device.
 * @pre  @p src and @p dst must have identical shapes.
 * @post On success, @p dst contains the casted elements.
 *
 * @see launchDtypeCastingKernel()  Device-agnostic dispatch entry point.
 */
novaStatus_t launchHipDtypeCastingKernel(const Tensor *restrict src,
                                         Tensor *restrict dst);

#ifdef __cplusplus
}
#endif
