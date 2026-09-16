/**
 * @file ContiguousDispatchImpl.cpp
 * @brief Runtime dispatch layer for contiguous-layout kernels.
 *
 * @details
 * This file implements the public @ref launchContiguousKernel entry
 * point declared in @ref contiguous.h. It resolves the active
 * compute device at run time and forwards the request to the
 * appropriate backend-specific kernel:
 *
 * @li CUDA — @ref launchCudaContiguousKernel (compiled from
 *   @c ContiguousLayoutKernel.cu ).
 * @li HIP — @ref launchHipContiguousKernel (compiled from
 *   @c ContiguousLayoutKernel.hip ).
 *
 * The dispatch uses a static @c std::map<DeviceKind , kernel_t> lookup
 * table populated at compile time based on which backends are
 * enabled (@c NOVA_HAS_CUDA, @c NOVA_HAS_HIP ). The detection flow
 * itself lives in @ref ncore::dispatch::launch(), shared with every
 * launcher.
 *
 * @see contiguous.h            Public interface for this module.
 * @see KernelDispatcher.hpp    Shared backend-resolution flow.
 * @see ContiguousLayoutKernel.cu   CUDA kernel implementation.
 * @see ContiguousLayoutKernel.hip  HIP kernel implementation.
 */

#include <map>

#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <ncore/native/kernels/contiguous.h>
#include <ncore/tensor.h>

#include "utils/KernelDispatcher.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef NOVA_HAS_CUDA
/**
 * @brief CUDA backend entry point for contiguous-layout dispatch.
 *
 * @details
 * Defined in @c ContiguousLayoutKernel.cu. Materializes a contiguous
 * copy of the source tensor in CUDA device memory.
 *
 * @param[in]  src  Source tensor. Must reside in CUDA device memory.
 * @param[in,out] dst  Destination tensor in CUDA device memory.
 *
 * @return @ref novaSuccess on success, or an error status.
 *
 * @see ContiguousLayoutKernel.cu
 * @see launchContiguousKernel()
 */
extern novaStatus_t launchCudaContiguousKernel(const Tensor *restrict src,
                                               Tensor *restrict dst);
#endif

#ifdef NOVA_HAS_HIP
/**
 * @brief HIP backend entry point for contiguous-layout dispatch.
 *
 * @details
 * Defined in @c ContiguousLayoutKernel.hip. Materializes a contiguous
 * copy of the source tensor in HIP device memory.
 *
 * @param[in]  src  Source tensor. Must reside in HIP device memory.
 * @param[in,out] dst  Destination tensor in HIP device memory.
 *
 * @return @ref novaSuccess on success, or an error status.
 *
 * @see ContiguousLayoutKernel.hip
 * @see launchContiguousKernel()
 */
extern novaStatus_t launchHipContiguousKernel(const Tensor *restrict src,
                                              Tensor *restrict dst);
#endif

#ifdef __cplusplus
}
#endif

namespace {

/**
 * @brief Function pointer type for backend-specific contiguous kernels.
 */
using kernel_t = novaStatus_t (*)(const Tensor *restrict, Tensor *restrict);

/**
 * @brief Static dispatch table mapping device kinds to contiguous kernels.
 *
 * @details
 * Populated at compile time. Entries for unavailable backends are
 * set to @c nullptr. Consumed by @ref ncore::dispatch::launch().
 */
const std::map<DeviceKind, kernel_t> KERNEL_DISPATCHER = {
#if defined(NOVA_HAS_CUDA) && !defined(NOVA_HAS_HIP)
    {CUDA_DEVICE, launchCudaContiguousKernel},
    {HIP_DEVICE, nullptr},
    {NULL_DEVICE, nullptr}
#elif defined(NOVA_HAS_HIP) && !defined(NOVA_HAS_CUDA)
    {CUDA_DEVICE, nullptr},
    {HIP_DEVICE, launchHipContiguousKernel},
    {NULL_DEVICE, nullptr},
#else
    {CUDA_DEVICE, nullptr}, {HIP_DEVICE, nullptr}, {NULL_DEVICE, nullptr}
#endif
};

} // namespace

/**
 * @brief Launch a contiguous-layout kernel on the detected device.
 *
 * @details
 * Entry point declared in @ref contiguous.h. Backend resolution and
 * error handling live in @ref ncore::dispatch::launch(); this body
 * only forwards the table and the tensors.
 *
 * @param[in]  src  Source tensor to materialize.
 * @param[in,out] dst  Destination tensor receiving the copy.
 *
 * @return @ref novaSuccess on success, or an error status.
 *
 * @pre  @p src and @p dst must be valid tensors.
 * @pre  At least one compute device must be available.
 * @post On success, @p dst holds a contiguous copy of @p src.
 *
 * @warning If no compute device is available, returns
 *          @ref novaDeviceNotAvailable.
 */
extern "C" novaStatus_t launchContiguousKernel(const Tensor *restrict src,
                                               Tensor *restrict dst) {
  return ncore::dispatch::launch(KERNEL_DISPATCHER, src, dst);
}
