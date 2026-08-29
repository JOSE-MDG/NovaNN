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
 * (@c NOVA_HAS_CUDA, @c NOVA_HAS_HIP).
 *
 * @section device-detection-flow Device Detection Flow
 *
 * @li 1. If device detection has already been performed, use the cached
 *    result directly.
 * @li 2. Otherwise, probe CUDA and HIP availability and cache the result.
 * @li 3. If exactly one backend is available, use it; otherwise return
 *    @ref novaDeviceNotAvailable.
 *
 * @see casting.h               Public interface for this module.
 * @see DtypeCastingKernel.cu   CUDA kernel implementation.
 * @see DtypeCastingKernel.hip  HIP kernel implementation.
 */

#include <map>
#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/tensor.h>

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
 * Populated at compile time.  Entries for unavailable backends are
 * set to @c nullptr.  Accessed via @ref getDispatchedKernel.
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

/**
 * @brief Retrieve the casting kernel for the given device kind.
 *
 * @param[in]  kind   The device kind to look up.
 * @param[out] status Receives the operation result.
 *
 * @return Function pointer to the kernel, or @c nullptr if the device
 *         has no registered kernel.
 *
 * @pre  @p status must not be null.
 * @post On success, @p status->err is @ref novaSuccess.
 */
kernel_t getDispatchedKernel(DeviceKind kind, novaStatus_t *status) noexcept {

  if (kind != CUDA_DEVICE && kind != HIP_DEVICE && kind != NULL_DEVICE) {
    status->err = novaInvalidValue;
    status->message = "Invalid device kind specified for kernel dispatch\n";
    return nullptr;
  }

  const kernel_t kernel = KERNEL_DISPATCHER.at(kind);
  status->err = novaSuccess;
  status->message = nova_get_error_msg(status->err, nullptr);
  return kernel;
}

} // namespace

/**
 * @brief Launch a dtype casting kernel on the detected compute device.
 *
 * @details
 * Entry point declared in @ref casting.h.  Selects the appropriate
 * backend kernel using @ref getDispatchedKernel and invokes it.
 *
 * If device detection has not been performed yet, this function
 * probes CUDA and HIP availability and caches the result before
 * proceeding with the dispatch.
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

  novaStatus_t status;

  if (was_device_detection_done()) {

    auto kernel = getDispatchedKernel(get_detected_device_kind(), &status);

    if (status.err != novaSuccess) {
      return status;
    }

    if (kernel == nullptr || get_detected_device_kind() == NULL_DEVICE) {
      status.err = novaDeviceNotInitialized;
      status.message = "No kernel available for the detected device; device "
                       "may not be initialized\n";
      return status;
    }

    // Launch the kernel
    status = kernel(src, dst);
    return status;
  }

  if ((is_cuda_available() && !is_hip_available()) ||
      (!is_cuda_available() && is_hip_available())) {

    auto kernel = getDispatchedKernel(get_detected_device_kind(), &status);

    if (status.err != novaSuccess) {
      return status;
    }

    if (kernel == nullptr || get_detected_device_kind() == NULL_DEVICE) {
      status.err = novaDeviceNotInitialized;
      status.message = "No kernel available for the detected device; device "
                       "may not be initialized\n";
      return status;
    }
    return kernel(src, dst);
  }

  status.err = novaDeviceNotAvailable;
  status.message =
      "No compute device available; cannot launch casting kernel\n";
  return status;
}
