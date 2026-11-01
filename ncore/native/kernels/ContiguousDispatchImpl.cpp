/**
 * @file ContiguousDispatchImpl.cpp
 * @brief Runtime dispatch layer for contiguous-layout kernels.
 *
 * @details
 * Implements the public @ref launchContiguousKernel entry point
 * declared in @ref contiguous.h. It resolves the active compute
 * device at run time and forwards the request to the CUDA kernel
 * (@ref launchCudaContiguousKernel from @c ContiguousLayoutKernel.cu)
 * or the HIP kernel (@ref launchHipContiguousKernel from
 * @c ContiguousLayoutKernel.hip).  The dispatch uses a static
 * @c std::unordered_map<DeviceKind, kernel_t> lookup table populated
 * at compile time based on which backends are enabled
 * (@c NOVA_HAS_CUDA, @c NOVA_HAS_HIP). The detection flow itself
 * lives in @ref ncore::dispatch::launch(), shared with every
 * launcher.
 *
 * The backend kernel runs on the device engine stream, resolved here
 * before dispatch; the opaque stream no longer crosses the public API.
 *
 * @see contiguous.h            Public interface for this module.
 * @see KernelDispatcher.hpp    Shared backend-resolution flow.
 * @see ContiguousLayoutKernel.cu   CUDA kernel implementation.
 * @see ContiguousLayoutKernel.hip  HIP kernel implementation.
 */

#include <unordered_map>

#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <ncore/native/kernels/contiguous.h>
#include <ncore/tensor.h>

#ifdef NOVA_HAS_CUDA
#include "cuda/kernels/ContiguousLayoutKernel.h"
#include "cuda/memory/CudaEngineStream.hpp"
#endif
#ifdef NOVA_HAS_HIP
#include "hip/kernels/ContiguousLayoutKernel.h"
#include "hip/memory/HipEngineStream.hpp"
#endif

#include "utils/KernelDispatcher.hpp"

namespace {

/**
 * @brief Function pointer type for backend-specific contiguous kernels.
 */
using kernel_t = novaStatus_t (*)(const Tensor *restrict, Tensor *restrict);

#if defined(NOVA_HAS_CUDA) && !defined(NOVA_HAS_HIP)
novaStatus_t
launchCudaContiguousKernelOnEngineStream(const Tensor *restrict src,
                                         Tensor *restrict dst) {
  cudaStream_t stream = nullptr;
  const novaStatus_t status = CudaEngineStream::current(&stream);
  if (status.err != novaSuccess) {
    return status;
  }
  return launchCudaContiguousKernel(src, dst, stream);
}
#endif

#if defined(NOVA_HAS_HIP) && !defined(NOVA_HAS_CUDA)
novaStatus_t launchHipContiguousKernelOnEngineStream(const Tensor *restrict src,
                                                     Tensor *restrict dst) {
  hipStream_t stream = nullptr;
  const novaStatus_t status = HipEngineStream::current(&stream);
  if (status.err != novaSuccess) {
    return status;
  }
  return launchHipContiguousKernel(src, dst, stream);
}
#endif

/**
 * @brief Static dispatch table mapping device kinds to contiguous kernels.
 *
 * @details
 * Populated at compile time. Entries for unavailable backends are
 * set to @c nullptr. Consumed by @ref ncore::dispatch::launch().
 */
const std::unordered_map<DeviceKind, kernel_t> KERNEL_DISPATCHER = {
#if defined(NOVA_HAS_CUDA) && !defined(NOVA_HAS_HIP)
    {CUDA_DEVICE, launchCudaContiguousKernelOnEngineStream},
    {HIP_DEVICE, nullptr},
    {NULL_DEVICE, nullptr}
#elif defined(NOVA_HAS_HIP) && !defined(NOVA_HAS_CUDA)
    {CUDA_DEVICE, nullptr},
    {HIP_DEVICE, launchHipContiguousKernelOnEngineStream},
    {NULL_DEVICE, nullptr},
#else
    {CUDA_DEVICE, nullptr}, {HIP_DEVICE, nullptr}, {NULL_DEVICE, nullptr}
#endif
};

} // namespace

/**
 * @brief Backend-dispatching definition of @ref launchContiguousKernel().
 *
 * @details
 * Backend resolution and error handling live in
 * @ref ncore::dispatch::launch(); this body only forwards the table
 * and the tensors. The stream is resolved to the device engine stream
 * by the backend wrapper.
 */
extern "C" novaStatus_t launchContiguousKernel(const Tensor *restrict src,
                                               Tensor *restrict dst) {
  return ncore::dispatch::launch(KERNEL_DISPATCHER, src, dst);
}
