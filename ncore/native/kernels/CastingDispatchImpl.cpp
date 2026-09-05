/**
 * @file CastingDispatchImpl.cpp
 * @brief Runtime dispatch layer for dtype casting kernels.
 *
 * @details
 * This file implements the public @ref launchDtypeCastingKernel entry
 * point declared in @ref casting.h.  It resolves the active compute
 * device at run time and forwards the request to the appropriate
 * backend-specific kernel:
 *
 * @li CUDA — @ref launchCudaDtypeCastingKernel (compiled from
 *   @c DtypeCastingKernel.cu).
 * @li HIP — @ref launchHipDtypeCastingKernel (compiled from
 *   @c DtypeCastingKernel.hip).
 *
 * The dispatch uses a static @c std::map<DeviceKind, kernel_t> lookup
 * table populated at compile time based on which backends are enabled
 * (@c NOVA_HAS_CUDA, @c NOVA_HAS_HIP). The detection flow itself
 * (cached kind, probe, exactly-one-backend) lives in
 * @ref ncore::dispatch::launch(), shared with every launcher.
 *
 * @see casting.h               Public interface for this module.
 * @see KernelDispatcher.hpp    Shared backend-resolution flow.
 * @see DtypeCastingKernel.cu   CUDA kernel implementation.
 * @see DtypeCastingKernel.hip  HIP kernel implementation.
 */

#include <map>

#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/tensor.h>

#include "utils/KernelDispatcher.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef NOVA_HAS_CUDA
/**
 * @brief CUDA backend entry point for dtype casting dispatch.
 *
 * @details
 * Defined in @c DtypeCastingKernel.cu.  Dispatches to backend-specific
 * kernels based on the source and destination dtypes.
 *
 * @param[in]  src  Source tensor.  Must have a supported source dtype.
 * @param[in,out] dst  Destination tensor.  Must have the target dtype
 *                     and matching shape.
 *
 * @return @ref novaSuccess on success, or an error status.
 *
 * @see DtypeCastingKernel.cu
 * @see launchDtypeCastingKernel()
 */
extern novaStatus_t launchCudaDtypeCastingKernel(const Tensor *src,
                                                 Tensor *dst);
#endif

#ifdef NOVA_HAS_HIP
/**
 * @brief HIP backend entry point for dtype casting dispatch.
 *
 * @details
 * Defined in @c DtypeCastingKernel.hip.  Dispatches to backend-specific
 * kernels based on the source and destination dtypes.
 *
 * @param[in]  src  Source tensor.  Must have a supported source dtype.
 * @param[in,out] dst  Destination tensor.  Must have the target dtype
 *                     and matching shape.
 *
 * @return @ref novaSuccess on success, or an error status.
 *
 * @see DtypeCastingKernel.hip
 * @see launchDtypeCastingKernel()
 */
extern novaStatus_t launchHipDtypeCastingKernel(const Tensor *src, Tensor *dst);
#endif

#ifdef __cplusplus
}
#endif

namespace {

/**
 * @brief Function pointer type for backend-specific casting kernels.
 */
using kernel_t = novaStatus_t (*)(const Tensor *, Tensor *);

/**
 * @brief Static dispatch table mapping device kinds to casting kernels.
 *
 * @details
 * Populated at compile time. Entries for unavailable backends are
 * set to @c nullptr. Consumed by @ref ncore::dispatch::launch().
 */
const std::map<DeviceKind, kernel_t> KERNEL_DISPATCHER = {
#if defined(NOVA_HAS_CUDA) && !defined(NOVA_HAS_HIP)
    {CUDA_DEVICE, launchCudaDtypeCastingKernel},
    {HIP_DEVICE, nullptr},
    {NULL_DEVICE, nullptr}
#elif defined(NOVA_HAS_HIP) && !defined(NOVA_HAS_CUDA)
    {CUDA_DEVICE, nullptr},
    {HIP_DEVICE, launchHipDtypeCastingKernel},
    {NULL_DEVICE, nullptr},
#else
    {CUDA_DEVICE, nullptr}, {HIP_DEVICE, nullptr}, {NULL_DEVICE, nullptr}
#endif
};

} // namespace

/**
 * @brief Launch a dtype casting kernel on the detected compute device.
 *
 * @details
 * Entry point declared in @ref casting.h. Backend resolution and
 * error handling live in @ref ncore::dispatch::launch(); this body
 * only forwards the table and the tensors.
 *
 * @param[in]  src  Source tensor to cast from.
 * @param[in,out] dst  Destination tensor to cast into.
 *
 * @return @ref novaSuccess on success, or an error status.
 *
 * @pre  @p src and @p dst must be valid tensors with matching shape.
 * @pre  At least one compute device must be available.
 * @post On success, @p dst contains the casted elements.
 *
 * @warning If no compute device is available, returns
 *          @ref novaDeviceNotAvailable.
 */
extern "C" novaStatus_t launchDtypeCastingKernel(const Tensor *src,
                                                 Tensor *dst) {
  return ncore::dispatch::launch(KERNEL_DISPATCHER, src, dst);
}
