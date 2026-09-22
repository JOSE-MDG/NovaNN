/**
 * @file KernelDispatcher.hpp
 * @brief Shared backend-resolution flow for native kernel launchers.
 *
 * @details
 * Every device-agnostic launcher (@c launchDtypeCastingKernel(),
 * @c launchContiguousKernel() , and whatever comes next) used to
 * repeat the same detection flow: cached kind, probe CUDA/HIP, run
 * when exactly one backend answers. That logic lives here exactly
 * once. Each operation keeps only its genuinely per-op data: the
 * backend entry points and the table saying which backends provide
 * them.
 *
 * The launcher is a template so future operations can change
 * signatures freely: @p KernelFn and the forwarded arguments are
 * deduced at each call site, and this file never needs to change
 * for a new signature.
 *
 * @see CastingDispatchImpl.cpp     Casting launcher using this flow.
 * @see ContiguousDispatchImpl.cpp  Contiguous launcher using this flow.
 */

#pragma once

#include <unordered_map>
#include <utility>

#include <ncore/core/device.h>
#include <ncore/core/status.h>

namespace ncore::dispatch {
/**
 * @brief Resolve the active backend kernel and invoke it.
 *
 * @details
 * Mirrors the historical hand-written dispatchers step by step:
 * @li 1. If detection already ran, use the cached kind.
 * @li 2. Otherwise probe CUDA/HIP availability and continue only
 *    when exactly one backend answers.
 * @li 3. A null table entry (backend not compiled for this op) or
 *    @c NULL_DEVICE resolves to @ref novaDeviceNotInitialized.
 * Anything else is @ref novaDeviceNotAvailable.
 *
 * @tparam KernelFn  Backend function pointer type (deduced). Must be
 *                   callable with @p args and return @c novaStatus_t.
 * @tparam Args      Launcher argument types (deduced, forwarded).
 *
 * @param[in] table  Backend table indexed by @c DeviceKind. Entries
 *                   for uncompiled backends must be @c nullptr.
 * @param[in] args   Arguments forwarded untouched to the backend
 *                   kernel.
 *
 * @return Whatever the backend kernel returns, or the detection
 *         error when no backend can run.
 *
 * @note Thread-safe as long as the backend kernels are: this flow
 *       keeps no state and only reads the detection cache.
 */
template <typename KernelFn, typename... Args>
novaStatus_t launch(const std::unordered_map<DeviceKind, KernelFn> &table,
                    Args &&...args) {
  novaStatus_t status;

  auto run = [&](DeviceKind kind) -> novaStatus_t {
    if (kind != CUDA_DEVICE && kind != HIP_DEVICE && kind != NULL_DEVICE) {
      status.err = novaInvalidValue;
      status.message = "Invalid device kind specified for kernel dispatch\n";
      return status;
    }
    const KernelFn kernel = table.at(kind);
    if (kernel == nullptr || kind == NULL_DEVICE) {
      status.err = novaDeviceNotInitialized;
      status.message = "No kernel available for the detected device; device "
                       "may not be initialized\n";
      return status;
    }
    return kernel(std::forward<Args>(args)...);
  };

  if (was_device_detection_done()) {
    return run(get_detected_device_kind());
  }

  if ((is_cuda_available() && !is_hip_available()) ||
      (!is_cuda_available() && is_hip_available())) {
    return run(get_detected_device_kind());
  }

  status.err = novaDeviceNotAvailable;
  status.message = "No compute device available; cannot launch kernel\n";
  return status;
}

} // namespace ncore::dispatch
